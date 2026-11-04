from __future__ import annotations

import os
import unittest
from contextlib import ExitStack
from unittest import mock

import torch
import torch.nn.functional as F
from torch import Tensor

from cubic_serve.contracts.model import ScalarDType
from cubic_serve.executor.runner import ReferenceExecutionContext, configure_reference_math
from cubic_serve.guardrail.compare import (
    ComparisonContext,
    ComparisonPolicy,
    DifferenceClass,
    ReductionOrderProof,
    stable_digest,
)
from cubic_serve.guardrail.oracle import (
    ReductionWitnessMaterial,
    reduction_material_identity,
    verify_reduction_witness,
)
from cubic_serve.ops.cuda.linear_attention import (
    INDEX_OWNERSHIP_CONTRACT,
    CausalConv1DDecodePrimitive,
    GatedDeltaDecodePrimitive,
)
from cubic_serve.ops.eager.hybrid import gated_delta_net_recurrent
from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

RUN_REAL = os.environ.get("CUBIC_RUN_CUDA_GDN_TESTS") == "1"
# Source-review pins: llama.cpp c589f0ed/qwen35.cpp, vLLM dbf662c9/
# qwen_gdn_linear_attn.py, SGLang a7ee3999/linear/gdn_backend.py.
_C, _QH, _VH, _KD, _VD, _WIDTH, _SLOTS = 4, 16, 48, 128, 128, 4, 32
_PACKED = 2 * _QH * _KD + _VH * _VD
_SCALE = _KD**-0.5
_ROWS = {
    1: (-1, 0, -3, -2),
    2: (1, -2, -1, 0),
    4: (2, 0, 3, 1),
}


def _verify_reduction_only(expected: Tensor, actual: Tensor, operands: Tensor) -> str:
    """Bind a non-exact result to complete FP32 reduction operands."""

    indices = torch.arange(operands.shape[-1], dtype=torch.int64, device=operands.device).expand(
        operands.shape
    )
    material = ReductionWitnessMaterial(operands, indices, operands, indices)
    identity = reduction_material_identity(material)
    policy = ComparisonPolicy(
        DifferenceClass.REDUCTION_ORDER,
        reduction=ReductionOrderProof(
            operands_digest=identity.operands_digest,
            indices_digest=identity.indices_digest,
            reduction_length=identity.reduction_length,
            accumulator_dtype=ScalarDType.FP32,
            candidate_accumulator_dtype=ScalarDType.FP32,
        ),
    )
    witness = verify_reduction_witness(
        expected,
        actual,
        policy,
        ComparisonContext(
            op_signature_digest=stable_digest("gated-delta/recurrent"),
            candidate_launch_digest=stable_digest("triton/fp32-reassociation"),
        ),
        material,
    )
    return witness.witness_digest


def _oracle(
    logical_x: Tensor,
    logical_a: Tensor,
    logical_b: Tensor,
    weight: Tensor,
    conv_source: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    recurrent_source: Tensor,
) -> dict[str, Tensor]:
    """Independent PyTorch definition; it imports no production helper."""

    history = conv_source.float()
    products = history * weight[:, :-1].float().unsqueeze(0)
    accumulator = products.sum(-1) + logical_x.float() * weight[:, -1].float()
    conv_output = F.silu(accumulator).to(torch.bfloat16)
    conv_destination = torch.cat((conv_source[:, :, 1:], logical_x.unsqueeze(-1)), dim=-1)

    q = conv_output[:, : _QH * _KD].reshape(-1, _QH, _KD)
    k = conv_output[:, _QH * _KD : 2 * _QH * _KD].reshape(-1, _QH, _KD)
    value = conv_output[:, 2 * _QH * _KD :].reshape(-1, _VH, _VD)
    q_f32, k_f32 = q.float(), k.float()
    q_norm = q_f32 / torch.sqrt((q_f32 * q_f32).sum(-1, keepdim=True) + 1e-6)
    k_norm = k_f32 / torch.sqrt((k_f32 * k_f32).sum(-1, keepdim=True) + 1e-6)
    # Frozen semantic boundaries: FP32 reduction, BF16 rounding, then FP32 recurrence.
    q_norm = q_norm.to(torch.bfloat16).float()
    k_norm = k_norm.to(torch.bfloat16).float()
    beta = torch.sigmoid(logical_b.float()).to(torch.bfloat16).float()
    decay = torch.exp(-torch.exp(a_log.float()) * F.softplus(logical_a.float() + dt_bias.float()))
    repeated_q = q_norm.repeat_interleave(_VH // _QH, dim=1)
    repeated_k = k_norm.repeat_interleave(_VH // _QH, dim=1)
    decayed = recurrent_source * decay[:, :, None, None]
    correction = (value.float() - (decayed * repeated_k[:, :, None, :]).sum(-1)) * beta[:, :, None]
    recurrent_destination = decayed + correction[:, :, :, None] * repeated_k[:, :, None, :]
    core_output = (
        (recurrent_destination * (repeated_q[:, :, None, :] * _SCALE)).sum(-1).to(torch.bfloat16)
    )

    return {
        "conv_output": conv_output,
        "conv_destination": conv_destination,
        "core_output": core_output,
        "recurrent_destination": recurrent_destination,
    }


def _addresses(tensors: tuple[Tensor, ...]) -> tuple[int, ...]:
    return tuple(tensor.data_ptr() for tensor in tensors)


@unittest.skipUnless(RUN_REAL, "set CUBIC_RUN_CUDA_GDN_TESTS=1 for real CUDA qualification")
class GatedDeltaDecodePrimitiveTests(unittest.TestCase):
    def setUp(self) -> None:
        # Historical midpoint/reduction corpus qualifies the explicit FP32
        # activation mode. Native low-precision arithmetic has its own audit.
        self.enterContext(operator_strategy_scope(
            OperatorConfig(activation_dtype="fp32", attention_dtype="fp32"), device="", model=""
        ))

    def test_fp16_activation_keeps_explicit_recurrent_state_dtype(self) -> None:
        configure_reference_math()
        device = torch.device("cuda:0")
        for activation_dtype in (torch.float16, torch.bfloat16):
            for state_dtype in (torch.float32, torch.bfloat16):
                corpus = torch.ones(33, 3, device=device, dtype=activation_dtype)
                corpus[:, 2] = (torch.arange(33, device=device) % 5 + 1) / 4
                beta = ((torch.arange(33, device=device) % 3) / 8 + 0.333).to(activation_dtype)
                for capacity, logical in ((1, 1), (8, 8), (64, 33)):
                    packed = torch.empty(capacity, 3, device=device, dtype=activation_dtype)
                    normalized = torch.empty_like(packed)
                    a = torch.zeros(capacity, 1, device=device, dtype=activation_dtype)
                    b = torch.zeros_like(a)
                    a_log = torch.zeros(1, device=device, dtype=torch.float32)
                    dt_bias = torch.zeros(1, device=device, dtype=activation_dtype)
                    state = torch.zeros(2 * capacity, 1, 1, 1, device=device, dtype=state_dtype)
                    source = torch.arange(capacity, device=device, dtype=torch.int32)
                    destination = source + capacity
                    lengths = torch.zeros_like(source)
                    lengths[:logical] = 1
                    output = torch.empty(capacity, 1, 1, device=device, dtype=activation_dtype)
                    args = (
                        packed,
                        normalized,
                        a,
                        b,
                        a_log,
                        dt_bias,
                        state,
                        source,
                        destination,
                        lengths,
                        output,
                    )
                    primitive = GatedDeltaDecodePrimitive(*args, q_heads=1, scale=1.0)
                    packed.fill_(1)
                    primitive.launch(*args)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        primitive.launch(*args)
                    for reverse in (False, True):
                        state.zero_()
                        packed.fill_(float("nan"))
                        packed[:logical].copy_(
                            corpus[:logical].flip(0) if reverse else corpus[:logical]
                        )
                        b[:logical, 0].copy_(beta[:logical].flip(0) if reverse else beta[:logical])
                        oracle_state = state[:logical].clone()
                        for _ in range(3):
                            with ReferenceExecutionContext():
                                expected, oracle_state = gated_delta_net_recurrent(
                                    packed[:logical, 0].reshape(logical, 1, 1, 1),
                                    packed[:logical, 1].reshape(logical, 1, 1, 1),
                                    packed[:logical, 2].reshape(logical, 1, 1, 1),
                                    a[:logical].unsqueeze(1),
                                    b[:logical].unsqueeze(1),
                                    a_log,
                                    dt_bias,
                                    oracle_state,
                                    l2_epsilon=1e-6,
                                    output_dtype=activation_dtype,
                                )
                            graph.replay()
                            self.assertTrue(torch.equal(output[:logical], expected[:, 0]))
                            self.assertTrue(
                                torch.equal(state[capacity : capacity + logical], oracle_state)
                            )
                            self.assertEqual(state.dtype, state_dtype)
                            self.assertEqual(output.dtype, activation_dtype)
                            self.assertEqual(
                                torch.count_nonzero(state[capacity + logical :]).item(), 0
                            )
                            state[:logical].copy_(state[capacity : capacity + logical])

    def test_conv_silu_review_midpoints_with_concurrent_padding(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            if not torch.cuda.is_available():
                self.skipTest("CUDA is required")
            configure_reference_math()
            device = torch.device("cuda:0")
            # Real C33 convolution products where fast sigmoid crosses a BF16
            # midpoint. Factor them back into exact BF16 operands; no model files
            # or rounded synthetic pre-activation are needed for this regression.
            products = torch.tensor(
                [
                    [
                        -0.001030564308166504,
                        0.006103992462158203,
                        0.13540267944335938,
                        0.005610466003417969,
                    ],
                    [
                        -0.00009873881936073303,
                        0.0005766749382019043,
                        -0.001160740852355957,
                        0.006115436553955078,
                    ],
                    [
                        0.001371145248413086,
                        0.0039136409759521484,
                        0.0020363330841064453,
                        -0.03498649597167969,
                    ],
                    [-1.13360595703125, -0.98846435546875, -1.38134765625, -3.36676025390625],
                    [
                        0.00017523765563964844,
                        0.002576589584350586,
                        -0.028972625732421875,
                        -0.0014445781707763672,
                    ],
                ],
                dtype=torch.float32,
            )
            # A dyadic rescaling preserves the historical exact products while
            # keeping quotient factors in the normal range of both dtypes.
            factors = torch.arange(1, 256, dtype=torch.float32) / 256
            quotients = (products[..., None] / factors).to(dtype).float()
            valid = quotients * factors == products[..., None]
            self.assertTrue(bool(valid.any(-1).all()))
            selected = valid.int().argmax(-1)
            weight = factors[selected].to(dtype).to(device)
            window = quotients.gather(-1, selected[..., None]).squeeze(-1).to(dtype).to(device)
            capacity, channels = 64, products.shape[0]
            x = torch.empty((capacity, channels), device=device, dtype=dtype)
            state = torch.empty((capacity, channels, 3), device=device, dtype=dtype)
            indices = torch.arange(capacity, device=device, dtype=torch.int32)
            lengths = torch.zeros_like(indices)
            output = torch.empty_like(x)
            primitive = CausalConv1DDecodePrimitive(
                x,
                weight,
                state,
                indices,
                indices,
                lengths,
                output,
            )
            x.zero_()
            state.zero_()
            primitive.launch(x, weight, state, indices, indices, lengths, output)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                primitive.launch(x, weight, state, indices, indices, lengths, output)
            from cubic_serve.ops.eager.hybrid import depthwise_causal_conv1d

            for concurrency in (8, 32, 33):
                for poison in (37.0, -73.0):
                    x.fill_(poison)
                    state.fill_(poison)
                    x[:concurrency].copy_(window[:, -1])
                    state[:concurrency].copy_(window[:, :3])
                    lengths.zero_()
                    lengths[:concurrency].fill_(1)
                    output.fill_(float("nan"))
                    with ReferenceExecutionContext():
                        expected, expected_state = depthwise_causal_conv1d(
                            x[:concurrency, None],
                            weight,
                            state[:concurrency].clone(),
                            activation="silu",
                            output_dtype=dtype,
                        )
                    graph.replay()
                    torch.cuda.synchronize(device)
                    self.assertTrue(torch.equal(output[:concurrency], expected[:, 0]))
                    self.assertTrue(torch.equal(state[:concurrency], expected_state))
                    self.assertTrue(bool((state[concurrency:] == poison).all()))
                    self.assertTrue(bool(torch.isnan(output[concurrency:]).all()))

    def test_bfloat16_min_normal_update_is_not_flushed_to_zero(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        configure_reference_math()
        device = torch.device("cuda:0")
        tiny = torch.finfo(torch.bfloat16).tiny
        packed = torch.tensor([[1.0, 1.0, 2.0 * tiny]], device=device, dtype=torch.bfloat16)
        normalized_qk = torch.empty_like(packed)
        a = torch.zeros((1, 1), device=device, dtype=torch.bfloat16)
        b = torch.zeros_like(a)
        a_log = torch.zeros((1,), device=device, dtype=torch.float32)
        dt_bias = torch.zeros((1,), device=device, dtype=torch.bfloat16)
        state = torch.zeros((2, 1, 1, 1), device=device, dtype=torch.bfloat16)
        source = torch.zeros((1,), device=device, dtype=torch.int32)
        destination = torch.ones_like(source)
        lengths = torch.ones_like(source)
        output = torch.empty((1, 1, 1), device=device, dtype=torch.bfloat16)
        primitive = GatedDeltaDecodePrimitive(
            packed,
            normalized_qk,
            a,
            b,
            a_log,
            dt_bias,
            state,
            source,
            destination,
            lengths,
            output,
            q_heads=1,
            scale=1.0,
        )

        primitive.launch(
            packed,
            normalized_qk,
            a,
            b,
            a_log,
            dt_bias,
            state,
            source,
            destination,
            lengths,
            output,
        )
        torch.cuda.synchronize(device)

        self.assertEqual(float(state[1, 0, 0, 0]), tiny)

    def test_bfloat16_state_matches_independent_eager_for_512_steps(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        configure_reference_math()
        device = torch.device("cuda:0")
        steps, q_heads, value_heads, key_dim, value_dim = 512, 2, 4, 16, 8
        packed_dim = 2 * q_heads * key_dim + value_heads * value_dim
        torch.manual_seed(8181)
        packed = torch.randn((steps, packed_dim), device=device, dtype=torch.bfloat16)
        a = torch.randn((steps, value_heads), device=device, dtype=torch.bfloat16)
        b = torch.randn_like(a)
        a_log = torch.randn((value_heads,), device=device, dtype=torch.float32)
        dt_bias = torch.randn((value_heads,), device=device, dtype=torch.bfloat16)
        seed = torch.randn((value_heads, value_dim, key_dim), device=device, dtype=torch.bfloat16)

        bf16_state = torch.empty(
            (2, value_heads, value_dim, key_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        bf16_state[0].copy_(seed)
        bf16_state[1].zero_()
        source = torch.zeros((1,), device=device, dtype=torch.int32)
        destination = torch.ones_like(source)
        lengths = torch.ones_like(source)
        bf16_output = torch.empty((1, value_heads, value_dim), device=device, dtype=torch.bfloat16)
        normalized_qk = torch.empty_like(packed[:1])
        bf16 = GatedDeltaDecodePrimitive(
            packed[:1],
            normalized_qk,
            a[:1],
            b[:1],
            a_log,
            dt_bias,
            bf16_state,
            source,
            destination,
            lengths,
            bf16_output,
            q_heads=q_heads,
            scale=key_dim**-0.5,
        )
        oracle_state = seed.unsqueeze(0)
        state_witnesses: list[str] = []
        output_witnesses: list[str] = []
        for step in range(steps):
            src, dst = step & 1, 1 - (step & 1)
            source.fill_(src)
            destination.fill_(dst)
            bf16.launch(
                packed[step : step + 1],
                normalized_qk,
                a[step : step + 1],
                b[step : step + 1],
                a_log,
                dt_bias,
                bf16_state,
                source,
                destination,
                lengths,
                bf16_output,
            )
            q_width = q_heads * key_dim
            query = packed[step : step + 1, :q_width].reshape(1, 1, q_heads, key_dim)
            key = packed[step : step + 1, q_width : 2 * q_width].reshape(1, 1, q_heads, key_dim)
            value = packed[step : step + 1, 2 * q_width :].reshape(1, 1, value_heads, value_dim)
            oracle_source = oracle_state
            with ReferenceExecutionContext():
                oracle_output, oracle_state = gated_delta_net_recurrent(
                    query,
                    key,
                    value,
                    a[step : step + 1].unsqueeze(1),
                    b[step : step + 1].unsqueeze(1),
                    a_log,
                    dt_bias,
                    oracle_state,
                    l2_epsilon=1e-6,
                    output_dtype=torch.bfloat16,
                )
            state_delta = (bf16_state[dst].float() - oracle_state[0].float()).abs()
            query_f32, key_f32 = query[:, 0].float(), key[:, 0].float()
            query_f32 = (
                (query_f32 / torch.sqrt((query_f32 * query_f32).sum(-1, keepdim=True) + 1e-6))
                .to(torch.bfloat16)
                .float()
            )
            key_f32 = (
                (key_f32 / torch.sqrt((key_f32 * key_f32).sum(-1, keepdim=True) + 1e-6))
                .to(torch.bfloat16)
                .float()
            )
            repeats = value_heads // q_heads
            query_f32 = query_f32.repeat_interleave(repeats, dim=1)
            key_f32 = key_f32.repeat_interleave(repeats, dim=1)
            beta = torch.sigmoid(b[step : step + 1].float()).to(torch.bfloat16).float()
            decay = torch.exp(
                -torch.exp(a_log.float()) * F.softplus(a[step : step + 1].float() + dt_bias.float())
            )
            decayed = oracle_source.float() * decay[:, :, None, None]
            coefficient = beta[:, :, None, None] * key_f32[:, :, None, :]
            memory_terms = decayed * key_f32[:, :, None, :]
            state_terms = torch.cat(
                (
                    decayed.unsqueeze(-1),
                    (coefficient * value[:, 0].float().unsqueeze(-1)).unsqueeze(-1),
                    -coefficient.unsqueeze(-1) * memory_terms.unsqueeze(-2),
                ),
                dim=-1,
            )
            if bool(torch.count_nonzero(state_delta).item()):
                state_witnesses.append(
                    _verify_reduction_only(oracle_state, bf16_state[dst].unsqueeze(0), state_terms)
                )
            output_terms = oracle_state.float() * (query_f32 * (key_dim**-0.5))[:, :, None, :]
            if not torch.equal(bf16_output, oracle_output[:, -1]):
                output_witnesses.append(
                    _verify_reduction_only(oracle_output[:, -1], bf16_output, output_terms)
                )
            output_delta = (bf16_output.float() - oracle_output[:, -1].float()).abs()
            self.assertTrue(torch.isfinite(output_delta).all())
        self.assertTrue(all(state_witnesses))
        self.assertTrue(all(output_witnesses))
        torch.cuda.synchronize(device)

    def test_bfloat16_state_storage_replays_with_fp32_math(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        device = torch.device("cuda:0")
        q_heads, value_heads, key_dim, value_dim = 2, 4, 16, 8
        packed_dim = 2 * q_heads * key_dim + value_heads * value_dim
        torch.manual_seed(8180)
        packed = torch.randn((1, packed_dim), device=device, dtype=torch.bfloat16)
        a = torch.randn((1, value_heads), device=device, dtype=torch.bfloat16)
        b = torch.randn_like(a)
        a_log = torch.randn((value_heads,), device=device, dtype=torch.float32)
        dt_bias = torch.randn((value_heads,), device=device, dtype=torch.bfloat16)
        state = torch.randn(
            (2, value_heads, value_dim, key_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        source = torch.zeros((1,), device=device, dtype=torch.int32)
        destination = torch.ones_like(source)
        lengths = torch.ones_like(source)
        output = torch.empty((1, value_heads, value_dim), device=device, dtype=torch.bfloat16)
        normalized_qk = torch.empty_like(packed)
        primitive = GatedDeltaDecodePrimitive(
            packed,
            normalized_qk,
            a,
            b,
            a_log,
            dt_bias,
            state,
            source,
            destination,
            lengths,
            output,
            q_heads=q_heads,
            scale=key_dim**-0.5,
        )
        primitive.launch(
            packed,
            normalized_qk,
            a,
            b,
            a_log,
            dt_bias,
            state,
            source,
            destination,
            lengths,
            output,
        )
        torch.cuda.synchronize(device)
        eager = output.clone(), state[1].clone()
        state[1].zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            primitive.launch(
                packed,
                normalized_qk,
                a,
                b,
                a_log,
                dt_bias,
                state,
                source,
                destination,
                lengths,
                output,
            )
        graph.replay()
        torch.cuda.synchronize(device)
        self.assertTrue(torch.equal(output, eager[0]))
        self.assertTrue(torch.equal(state[1], eager[1]))

    def test_qwen38_m4_graph_isolated_m1_m2_m4_for_both_a_log_dtypes(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        device = torch.device("cuda:0")
        self.assertEqual(
            INDEX_OWNERSHIP_CONTRACT,
            "cubic.arena.generation-checked-cow-or-exclusive-indices.v2",
        )

        for a_log_dtype in (torch.bfloat16, torch.float32):
            with self.subTest(a_log_dtype=a_log_dtype):
                self._qualify_dtype(device, a_log_dtype)

    def test_checkpointed_prefill_keeps_original_history_under_aliasing(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        channels = 264  # Cover a partial channel tile as well as the first tile.
        device = torch.device("cuda:0")
        weight = (torch.arange(channels * 4).reshape(channels, 4) % 5 - 2).bfloat16()
        for length in (1, 2, 3, 16, 65, 256):
            minimum = 1 << (length - 1).bit_length()
            for capacity in sorted({minimum, min(256, minimum * 2)}):
                for offset in (0, 1) if length < capacity else (0,):
                    for alias in (False, True):
                        with self.subTest(
                            length=length, capacity=capacity, offset=offset, alias=alias
                        ):
                            poison = 41.0 if offset else -41.0
                            x = torch.full((capacity, channels), poison, dtype=torch.bfloat16)
                            x[offset : offset + length] = (
                                torch.arange(length * channels).reshape(length, channels) % 7 - 3
                            ).bfloat16()
                            seed = torch.full(
                                (capacity + 2, channels, 3), -19.0, dtype=torch.bfloat16
                            )
                            seed[0] = (
                                torch.arange(channels * 3).reshape(channels, 3) % 7 - 3
                            ).bfloat16()
                            destinations = [0] * length if alias else list(range(1, length + 1))
                            source = torch.full((capacity,), capacity + 1, dtype=torch.int32)
                            target = source.clone()
                            source[offset : offset + length] = 0
                            target[offset : offset + length] = torch.tensor(
                                destinations, dtype=torch.int32
                            )
                            segments = torch.zeros(capacity, dtype=torch.int32)
                            segments[offset] = length
                            output = torch.full_like(x, poison)
                            operands = tuple(
                                t.to(device)
                                for t in (x, weight, seed, source, target, segments, output)
                            )
                            primitive = CausalConv1DDecodePrimitive(
                                *operands, silu=False, checkpoint_destinations=True
                            )
                            primitive.launch(*operands)
                            torch.cuda.synchronize(device)
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                primitive.launch(*operands)
                            operands[2].copy_(seed)
                            operands[-1].fill_(poison)
                            graph.replay()
                            torch.cuda.synchronize(device)
                            # Independent integer-valued FP32 convolution; all
                            # arithmetic is exact and uses no production helper.
                            expected_state = seed.clone()
                            expected = output.clone()
                            history = seed[0].clone()
                            for step, destination in enumerate(destinations):
                                row = x[offset + step]
                                expected[offset + step] = (
                                    (history.float() * weight[:, :-1].float()).sum(-1)
                                    + row.float() * weight[:, -1].float()
                                ).bfloat16()
                                history = torch.cat((history[:, 1:], row[:, None]), dim=-1)
                                expected_state[destination] = history
                            self.assertTrue(torch.equal(operands[-1].cpu(), expected))
                            self.assertTrue(torch.equal(operands[2].cpu(), expected_state))

    def test_exclusive_in_place_decode_is_bit_exact_to_cow(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        device = torch.device("cuda:0")
        capacity, q_heads, value_heads = 1, 2, 4
        key_dim, value_dim, width, slots = 16, 8, 4, 2
        packed_dim = 2 * q_heads * key_dim + value_heads * value_dim
        torch.manual_seed(4039)
        x = torch.randn((capacity, packed_dim), device=device).to(torch.bfloat16)
        weight = torch.randn((packed_dim, width), device=device).to(torch.bfloat16)
        a = torch.randn((capacity, value_heads), device=device).to(torch.bfloat16)
        b = torch.randn_like(a)
        a_log = torch.randn((value_heads,), device=device, dtype=torch.float32)
        dt_bias = torch.randn((value_heads,), device=device).to(torch.bfloat16)
        conv_seed = torch.randn((slots, packed_dim, width - 1), device=device).to(torch.bfloat16)
        recurrent_seed = torch.randn((slots, value_heads, value_dim, key_dim), device=device)
        length = torch.ones((1,), device=device, dtype=torch.int32)

        results = []
        for destination in (1, 0):
            conv_state = conv_seed.clone()
            recurrent_state = recurrent_seed.clone()
            source = torch.zeros((1,), device=device, dtype=torch.int32)
            target = torch.full_like(source, destination)
            conv_output = torch.empty_like(x)
            core_output = torch.empty(
                (capacity, value_heads, value_dim),
                device=device,
                dtype=torch.bfloat16,
            )
            normalized_qk = torch.empty_like(conv_output)
            conv = CausalConv1DDecodePrimitive(
                x,
                weight,
                conv_state,
                source,
                target,
                length,
                conv_output,
                checkpoint_destinations=True,
            )
            gdn = GatedDeltaDecodePrimitive(
                conv_output,
                normalized_qk,
                a,
                b,
                a_log,
                dt_bias,
                recurrent_state,
                source,
                target,
                length,
                core_output,
                q_heads=q_heads,
                scale=key_dim**-0.5,
                checkpoint_destinations=True,
            )
            conv.launch(x, weight, conv_state, source, target, length, conv_output)
            gdn.launch(
                conv_output,
                normalized_qk,
                a,
                b,
                a_log,
                dt_bias,
                recurrent_state,
                source,
                target,
                length,
                core_output,
            )
            torch.cuda.synchronize(device)
            results.append(
                (
                    conv_output.clone(),
                    core_output.clone(),
                    conv_state[destination].clone(),
                    recurrent_state[destination].clone(),
                )
            )
        self.assertTrue(all(torch.equal(left, right) for left, right in zip(*results, strict=True)))

    def test_mixed_segments_equal_sequential_length_one_and_ignore_zero_rows(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        device = torch.device("cuda:0")
        capacity, q_heads, value_heads = 8, 2, 4
        key_dim, value_dim, width, slots = 16, 8, 4, 24
        packed_dim = 2 * q_heads * key_dim + value_heads * value_dim
        torch.manual_seed(4040)
        x = torch.randn((capacity, packed_dim), device=device).to(torch.bfloat16)
        weight = torch.randn((packed_dim, width), device=device).to(torch.bfloat16)
        a = torch.randn((capacity, value_heads), device=device).to(torch.bfloat16)
        b = torch.randn_like(a)
        a_log = torch.randn((value_heads,), device=device, dtype=torch.float32)
        dt_bias = torch.randn((value_heads,), device=device).to(torch.bfloat16)
        conv_seed = torch.randn((slots, packed_dim, width - 1), device=device).to(torch.bfloat16)
        recurrent_seed = torch.randn(
            (slots, value_heads, value_dim, key_dim),
            device=device,
            dtype=torch.float32,
        )
        src = torch.tensor([0, 17, 18, 1, 2, 19, 20, 21], device=device, dtype=torch.int32)
        dst = torch.tensor([4, 17, 18, 5, 6, 19, 20, 21], device=device, dtype=torch.int32)
        segment_lengths = torch.tensor([3, 0, 0, 1, 2, 0, 0, 0], device=device, dtype=torch.int32)
        conv_state = conv_seed.clone()
        recurrent_state = recurrent_seed.clone()
        conv_output = torch.full_like(x, -77.0)
        core_output = torch.full(
            (capacity, value_heads, value_dim),
            -77.0,
            device=device,
            dtype=torch.bfloat16,
        )
        normalized_qk = torch.empty_like(conv_output)
        conv = CausalConv1DDecodePrimitive(
            x, weight, conv_state, src, dst, segment_lengths, conv_output
        )
        gdn = GatedDeltaDecodePrimitive(
            conv_output,
            normalized_qk,
            a,
            b,
            a_log,
            dt_bias,
            recurrent_state,
            src,
            dst,
            segment_lengths,
            core_output,
            q_heads=q_heads,
            scale=key_dim**-0.5,
        )

        # Capture the fixed capacity once; segment lengths remain replay inputs.
        conv.launch(x, weight, conv_state, src, dst, segment_lengths, conv_output)
        gdn.launch(
            conv_output,
            normalized_qk,
            a,
            b,
            a_log,
            dt_bias,
            recurrent_state,
            src,
            dst,
            segment_lengths,
            core_output,
        )
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            conv.launch(x, weight, conv_state, src, dst, segment_lengths, conv_output)
            gdn.launch(
                conv_output,
                normalized_qk,
                a,
                b,
                a_log,
                dt_bias,
                recurrent_state,
                src,
                dst,
                segment_lengths,
                core_output,
            )
        conv_state.copy_(conv_seed)
        recurrent_state.copy_(recurrent_seed)
        conv_output.fill_(-77.0)
        core_output.fill_(-77.0)
        graph.replay()
        torch.cuda.synchronize(device)

        ref_conv_state = conv_seed.clone()
        ref_recurrent_state = recurrent_seed.clone()
        ref_conv_output = torch.empty_like(x)
        ref_core_output = torch.empty_like(core_output)
        ref_normalized_qk = torch.empty_like(ref_conv_output)
        one_src = torch.zeros((1,), device=device, dtype=torch.int32)
        one_dst = torch.zeros_like(one_src)
        one_length = torch.ones_like(one_src)
        ref_conv = CausalConv1DDecodePrimitive(
            x[:1],
            weight,
            ref_conv_state,
            one_src,
            one_dst,
            one_length,
            ref_conv_output[:1],
        )
        ref_gdn = GatedDeltaDecodePrimitive(
            ref_conv_output[:1],
            ref_normalized_qk[:1],
            a[:1],
            b[:1],
            a_log,
            dt_bias,
            ref_recurrent_state,
            one_src,
            one_dst,
            one_length,
            ref_core_output[:1],
            q_heads=q_heads,
            scale=key_dim**-0.5,
        )
        for head, length, source_slot, destination_slot in (
            (0, 3, 0, 4),
            (3, 1, 1, 5),
            (4, 2, 2, 6),
        ):
            current = source_slot
            for step in range(length):
                final = step == length - 1
                next_slot = destination_slot if final else 8 + head + step
                one_src.fill_(current)
                one_dst.fill_(next_slot)
                row = slice(head + step, head + step + 1)
                ref_conv.launch(
                    x[row],
                    weight,
                    ref_conv_state,
                    one_src,
                    one_dst,
                    one_length,
                    ref_conv_output[row],
                )
                ref_gdn.launch(
                    ref_conv_output[row],
                    ref_normalized_qk[row],
                    a[row],
                    b[row],
                    a_log,
                    dt_bias,
                    ref_recurrent_state,
                    one_src,
                    one_dst,
                    one_length,
                    ref_core_output[row],
                )
                current = next_slot
        torch.cuda.synchronize(device)

        self.assertTrue(torch.equal(conv_output[:6], ref_conv_output[:6]))
        self.assertTrue(torch.equal(core_output[:6], ref_core_output[:6]))
        self.assertTrue(torch.equal(conv_state[4:7], ref_conv_state[4:7]))
        self.assertTrue(torch.equal(recurrent_state[4:7], ref_recurrent_state[4:7]))
        self.assertTrue(torch.equal(conv_state[17:22], conv_seed[17:22]))
        self.assertTrue(torch.equal(recurrent_state[17:22], recurrent_seed[17:22]))
        self.assertTrue(torch.all(conv_output[6:] == -77.0))
        self.assertTrue(torch.all(core_output[6:] == -77.0))

    def test_mixed_segment_checkpoints_keep_identical_length_one_waves_equal(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        device = torch.device("cuda:0")
        capacity, q_heads, value_heads = 8, _QH, _VH
        key_dim, value_dim, width, slots = _KD, _VD, _WIDTH, 256
        packed_dim = 2 * q_heads * key_dim + value_heads * value_dim
        torch.manual_seed(4050)
        x = torch.randn((capacity, packed_dim), device=device).to(torch.bfloat16)
        x[1].copy_(x[0])
        weight = torch.randn((packed_dim, width), device=device).to(torch.bfloat16)
        a = torch.randn((capacity, value_heads), device=device).to(torch.bfloat16)
        b = torch.randn_like(a)
        a[1].copy_(a[0])
        b[1].copy_(b[0])
        a_log = torch.randn((value_heads,), device=device, dtype=torch.float32)
        dt_bias = torch.randn((value_heads,), device=device).to(torch.bfloat16)
        conv_state = torch.randn((slots, packed_dim, width - 1), device=device).to(torch.bfloat16)
        recurrent_state = torch.randn(
            (slots, value_heads, value_dim, key_dim),
            device=device,
            dtype=torch.float32,
        )
        conv_state[144].copy_(conv_state[48])
        recurrent_state[144].copy_(recurrent_state[48])
        src = torch.tensor((48, 144, 4, 5, 6, 7, 8, 9), device=device, dtype=torch.int32)
        dst = torch.tensor((0, 96, 10, 11, 12, 13, 14, 15), device=device, dtype=torch.int32)
        lengths = torch.tensor((1, 1, 2, 0, 4, 0, 0, 0), device=device, dtype=torch.int32)
        conv_output = torch.empty_like(x)
        core_output = torch.empty(
            (capacity, value_heads, value_dim), device=device, dtype=torch.bfloat16
        )
        normalized_qk = torch.empty_like(conv_output)
        conv = CausalConv1DDecodePrimitive(
            x,
            weight,
            conv_state,
            src,
            dst,
            lengths,
            conv_output,
            checkpoint_destinations=True,
        )
        gdn = GatedDeltaDecodePrimitive(
            conv_output,
            normalized_qk,
            a,
            b,
            a_log,
            dt_bias,
            recurrent_state,
            src,
            dst,
            lengths,
            core_output,
            q_heads=q_heads,
            scale=key_dim**-0.5,
            checkpoint_destinations=True,
        )
        conv.launch(x, weight, conv_state, src, dst, lengths, conv_output)
        gdn.launch(
            conv_output,
            normalized_qk,
            a,
            b,
            a_log,
            dt_bias,
            recurrent_state,
            src,
            dst,
            lengths,
            core_output,
        )
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            conv.launch(x, weight, conv_state, src, dst, lengths, conv_output)
            gdn.launch(
                conv_output,
                normalized_qk,
                a,
                b,
                a_log,
                dt_bias,
                recurrent_state,
                src,
                dst,
                lengths,
                core_output,
            )

        for wave in range(3):
            graph.replay()
            torch.cuda.synchronize(device)
            self.assertTrue(torch.equal(conv_output[0], conv_output[1]), wave)
            self.assertTrue(torch.equal(core_output[0], core_output[1]), wave)
            first_dst, second_dst = int(dst[0]), int(dst[1])
            self.assertTrue(torch.equal(conv_state[first_dst], conv_state[second_dst]), wave)
            self.assertTrue(
                torch.equal(recurrent_state[first_dst], recurrent_state[second_dst]), wave
            )
            src[0], src[1] = dst[0], dst[1]
            dst[0], dst[1] = (48, 144) if wave % 2 == 0 else (0, 96)
            x[0].normal_()
            x[1].copy_(x[0])
            a[0].normal_()
            a[1].copy_(a[0])
            b[0].normal_()
            b[1].copy_(b[0])

    def _qualify_dtype(self, device: torch.device, a_log_dtype: torch.dtype) -> None:
        torch.manual_seed(8181 if a_log_dtype is torch.bfloat16 else 8282)
        logical_x = (torch.randn((_C, _PACKED), device=device) * 0.25).to(torch.bfloat16)
        logical_a = (torch.randn((_C, _VH), device=device) * 0.25).to(torch.bfloat16)
        logical_b = (torch.randn((_C, _VH), device=device) * 0.25).to(torch.bfloat16)
        weight = (torch.randn((_PACKED, _WIDTH), device=device) * 0.2).to(torch.bfloat16)
        a_log = (torch.randn((_VH,), device=device) * 0.2).to(a_log_dtype)
        dt_bias = (torch.randn((_VH,), device=device) * 0.2).to(torch.bfloat16)
        conv_seed = (torch.randn((_SLOTS, _PACKED, _WIDTH - 1), device=device) * 0.2).to(
            torch.bfloat16
        )
        recurrent_seed = (
            torch.randn((_SLOTS, _VH, _VD, _KD), device=device, dtype=torch.float32) * 0.2
        )

        x = torch.empty((_C, _PACKED), device=device, dtype=torch.bfloat16)
        a = torch.empty((_C, _VH), device=device, dtype=torch.bfloat16)
        b = torch.empty_like(a)
        conv_state = conv_seed.clone()
        recurrent_state = recurrent_seed.clone()
        src = torch.arange(_C, device=device, dtype=torch.int32)
        dst = src + 8
        segment_lengths = torch.ones((_C,), device=device, dtype=torch.int32)
        conv_output = torch.empty_like(x)
        core_output = torch.empty((_C, _VH, _VD), device=device, dtype=torch.bfloat16)
        normalized_qk = torch.empty_like(conv_output)
        conv = CausalConv1DDecodePrimitive(
            x, weight, conv_state, src, dst, segment_lengths, conv_output
        )
        gdn = GatedDeltaDecodePrimitive(
            conv_output,
            normalized_qk,
            a,
            b,
            a_log,
            dt_bias,
            recurrent_state,
            src,
            dst,
            segment_lengths,
            core_output,
            q_heads=_QH,
            scale=_SCALE,
        )
        self.assertIn(str(a_log_dtype), gdn.launch_signature)
        tensors = (
            x,
            a,
            b,
            weight,
            a_log,
            dt_bias,
            conv_state,
            recurrent_state,
            src,
            dst,
            segment_lengths,
            conv_output,
            normalized_qk,
            core_output,
        )
        addresses = _addresses(tensors)
        self.assertFalse(any(isinstance(value, Tensor) for value in vars(conv).values()))
        self.assertFalse(any(isinstance(value, Tensor) for value in vars(gdn).values()))

        stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.device(device), torch.cuda.stream(stream):
            for _ in range(2):
                conv.launch(x, weight, conv_state, src, dst, segment_lengths, conv_output)
                gdn.launch(
                    conv_output,
                    normalized_qk,
                    a,
                    b,
                    a_log,
                    dt_bias,
                    recurrent_state,
                    src,
                    dst,
                    segment_lengths,
                    core_output,
                )
        stream.synchronize()
        self._tripwire_launch(
            device,
            stream,
            conv,
            gdn,
            (x, weight, conv_state, src, dst, segment_lengths, conv_output),
            (
                conv_output,
                normalized_qk,
                a,
                b,
                a_log,
                dt_bias,
                recurrent_state,
                src,
                dst,
                segment_lengths,
                core_output,
            ),
        )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            conv.launch(x, weight, conv_state, src, dst, segment_lengths, conv_output)
            gdn.launch(
                conv_output,
                normalized_qk,
                a,
                b,
                a_log,
                dt_bias,
                recurrent_state,
                src,
                dst,
                segment_lengths,
                core_output,
            )
        stream.synchronize()

        by_m: dict[int, dict[str, Tensor]] = {}
        for logical_m in (1, 2, 4):
            poison_results: list[dict[str, Tensor]] = []
            for poison in (37.0, -73.0):
                source, destination = self._fill_launch(
                    logical_m,
                    poison,
                    logical_x,
                    logical_a,
                    logical_b,
                    conv_seed,
                    recurrent_seed,
                    x,
                    a,
                    b,
                    conv_state,
                    recurrent_state,
                    src,
                    dst,
                    segment_lengths,
                )
                graph.replay()
                torch.cuda.synchronize(device)
                physical = torch.tensor(
                    [_ROWS[logical_m].index(index) for index in range(logical_m)],
                    device=device,
                )
                destination_tensor = torch.tensor(destination, device=device)
                actual = {
                    "conv_output": conv_output[physical].clone(),
                    "conv_destination": conv_state[destination_tensor].clone(),
                    "core_output": core_output[physical].clone(),
                    "recurrent_destination": recurrent_state[destination_tensor].clone(),
                }
                expected = _oracle(
                    logical_x[:logical_m],
                    logical_a[:logical_m],
                    logical_b[:logical_m],
                    weight,
                    conv_seed[torch.tensor(source, device=device)],
                    a_log,
                    dt_bias,
                    recurrent_seed[torch.tensor(source, device=device)],
                )
                self.assertTrue(torch.equal(actual["conv_output"], expected["conv_output"]))
                self.assertTrue(
                    torch.equal(actual["conv_destination"], expected["conv_destination"])
                )
                # Triton's fixed 128-wide tree and eager torch.sum may round the
                # final FP32 dot in a different order; recurrent state remains exact.
                torch.testing.assert_close(
                    actual["core_output"],
                    expected["core_output"],
                    rtol=2**-7,
                    atol=2**-14,
                    msg=f"core output mismatch for logical M={logical_m}, poison={poison}",
                )
                self.assertTrue(
                    torch.equal(actual["recurrent_destination"], expected["recurrent_destination"])
                )
                self.assertTrue(torch.equal(conv_state[:logical_m], conv_seed[:logical_m]))
                self.assertTrue(
                    torch.equal(recurrent_state[:logical_m], recurrent_seed[:logical_m])
                )
                poison_results.append(actual)
            for name in poison_results[0]:
                self.assertTrue(
                    torch.equal(poison_results[0][name], poison_results[1][name]),
                    f"padding poison reached active {name} for logical M={logical_m}",
                )
            by_m[logical_m] = poison_results[0]

        for logical_m in (1, 2):
            for name, value in by_m[logical_m].items():
                self.assertTrue(torch.equal(value, by_m[4][name][:logical_m]))
        self.assertEqual(addresses, _addresses(tensors))
        self._assert_kd16_fma_regression_exact(device, a_log_dtype)
        self._assert_kd128_exact(device, a_log_dtype)
        with self.assertRaises(ValueError):
            GatedDeltaDecodePrimitive(
                conv_output,
                normalized_qk,
                a,
                b,
                a_log.to(torch.float16),
                dt_bias,
                recurrent_state,
                src,
                dst,
                segment_lengths,
                core_output,
                q_heads=_QH,
                scale=_SCALE,
            )

    def _assert_kd16_fma_regression_exact(
        self, device: torch.device, a_log_dtype: torch.dtype
    ) -> None:
        """A fixed KD=16 fixture closes only after mul/add stop contracting."""

        torch.manual_seed(1212)
        capacity = 1
        key_dim = 16
        packed = torch.randn((capacity, 2 * key_dim + 1), device=device, dtype=torch.bfloat16)
        a = torch.randn((capacity, 1), device=device, dtype=torch.bfloat16)
        b = torch.randn_like(a)
        a_log = torch.randn((1,), device=device).to(a_log_dtype)
        dt_bias = torch.randn((1,), device=device, dtype=torch.bfloat16)
        state = torch.randn((2, 1, 1, key_dim), device=device, dtype=torch.float32)
        initial = state.clone()
        src = torch.tensor([0], device=device, dtype=torch.int32)
        dst = torch.tensor([1], device=device, dtype=torch.int32)
        segment_lengths = torch.ones((1,), device=device, dtype=torch.int32)
        output = torch.empty((capacity, 1, 1), device=device, dtype=torch.bfloat16)
        normalized_qk = torch.empty_like(packed)
        primitive = GatedDeltaDecodePrimitive(
            packed,
            normalized_qk,
            a,
            b,
            a_log,
            dt_bias,
            state,
            src,
            dst,
            segment_lengths,
            output,
            q_heads=1,
            scale=key_dim**-0.5,
        )
        primitive.launch(
            packed,
            normalized_qk,
            a,
            b,
            a_log,
            dt_bias,
            state,
            src,
            dst,
            segment_lengths,
            output,
        )
        torch.cuda.synchronize(device)

        q = packed[:, :key_dim].float()
        k = packed[:, key_dim : 2 * key_dim].float()
        value = packed[:, 2 * key_dim].float()
        q = (q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16).float()
        k = (k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16).float()
        beta = torch.sigmoid(b[:, 0].float()).to(torch.bfloat16).float()
        decay = torch.exp(-torch.exp(a_log.float()) * F.softplus(a[:, 0].float() + dt_bias.float()))
        decayed = initial[:capacity, 0, 0] * decay[:, None]
        memory = (decayed * k).sum(-1)
        residual = (value - memory) * beta
        expected_state = decayed + residual[:, None] * k
        expected_output = (expected_state * (q * (key_dim**-0.5))).sum(-1).to(torch.bfloat16)
        self.assertTrue(torch.equal(state[dst.long(), 0, 0], expected_state))
        self.assertTrue(torch.equal(output[:, 0, 0], expected_output))

    def _assert_kd128_exact(self, device: torch.device, a_log_dtype: torch.dtype) -> None:
        """The Qwen-like shape closes after reduction leaves forbid FMA."""

        torch.manual_seed(2312)
        key_dim = value_dim = 128
        packed = torch.randn((1, 2 * key_dim + value_dim), device=device, dtype=torch.bfloat16)
        a = torch.randn((1, 1), device=device, dtype=torch.bfloat16)
        b = torch.randn_like(a)
        a_log = torch.randn((1,), device=device).to(a_log_dtype)
        dt_bias = torch.randn((1,), device=device, dtype=torch.bfloat16)
        state = torch.randn((2, 1, value_dim, key_dim), device=device, dtype=torch.float32) * 0.2
        source = state[0:1].clone()
        src = torch.tensor([0], device=device, dtype=torch.int32)
        dst = torch.tensor([1], device=device, dtype=torch.int32)
        segment_lengths = torch.ones((1,), device=device, dtype=torch.int32)
        output = torch.empty((1, 1, value_dim), device=device, dtype=torch.bfloat16)
        normalized_qk = torch.empty_like(packed)
        primitive = GatedDeltaDecodePrimitive(
            packed,
            normalized_qk,
            a,
            b,
            a_log,
            dt_bias,
            state,
            src,
            dst,
            segment_lengths,
            output,
            q_heads=1,
            scale=key_dim**-0.5,
        )

        primitive.launch(
            packed,
            normalized_qk,
            a,
            b,
            a_log,
            dt_bias,
            state,
            src,
            dst,
            segment_lengths,
            output,
        )
        torch.cuda.synchronize(device)
        expected_state, expected_output = self._direct_oracle(
            packed, a, b, a_log, dt_bias, source, key_dim
        )
        self.assertTrue(torch.equal(output, expected_output))
        self.assertTrue(torch.equal(state[1:2], expected_state))

    @staticmethod
    def _direct_oracle(
        packed: Tensor,
        a: Tensor,
        b: Tensor,
        a_log: Tensor,
        dt_bias: Tensor,
        source: Tensor,
        key_dim: int,
    ) -> tuple[Tensor, Tensor]:
        """Independent direct-GDN oracle; candidate exposes no intermediates."""

        q = packed[:, :key_dim].float().unsqueeze(1)
        k = packed[:, key_dim : 2 * key_dim].float().unsqueeze(1)
        value = packed[:, 2 * key_dim :].float().unsqueeze(1)
        q = (q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16).float()
        k = (k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16).float()
        beta = torch.sigmoid(b.float()).to(torch.bfloat16).float()
        decay = torch.exp(-torch.exp(a_log.float()) * F.softplus(a.float() + dt_bias.float()))
        decayed = source * decay[:, :, None, None]
        memory = (decayed * k[:, :, None, :]).sum(-1)
        residual = (value - memory) * beta[:, :, None]
        destination = decayed + residual[:, :, :, None] * k[:, :, None, :]
        core = (destination * (q[:, :, None, :] * (key_dim**-0.5))).sum(-1).to(torch.bfloat16)
        return destination, core

    def _fill_launch(
        self,
        logical_m: int,
        poison: float,
        logical_x: Tensor,
        logical_a: Tensor,
        logical_b: Tensor,
        conv_seed: Tensor,
        recurrent_seed: Tensor,
        x: Tensor,
        a: Tensor,
        b: Tensor,
        conv_state: Tensor,
        recurrent_state: Tensor,
        src: Tensor,
        dst: Tensor,
        segment_lengths: Tensor,
    ) -> tuple[list[int], list[int]]:
        rows = _ROWS[logical_m]
        active_source = list(range(logical_m))
        active_destination = [8 + index for index in range(logical_m)]
        padding_slots = {24 + (-row - 1) for row in rows if row < 0}
        self.assertEqual(len(padding_slots), _C - logical_m)
        self.assertTrue(set(active_source).isdisjoint(active_destination))
        self.assertTrue(set(active_source).isdisjoint(padding_slots))
        self.assertTrue(set(active_destination).isdisjoint(padding_slots))

        conv_state.copy_(conv_seed)
        recurrent_state.copy_(recurrent_seed)
        source_indices: list[int] = []
        destination_indices: list[int] = []
        for physical, row in enumerate(rows):
            if row >= 0:
                x[physical].copy_(logical_x[row])
                a[physical].copy_(logical_a[row])
                b[physical].copy_(logical_b[row])
                source_indices.append(row)
                destination_indices.append(8 + row)
            else:
                scratch = 24 + (-row - 1)
                x[physical].fill_(poison)
                a[physical].fill_(poison / 8.0)
                b[physical].fill_(-poison / 8.0)
                conv_state[scratch].fill_(poison)
                recurrent_state[scratch].fill_(poison / 16.0)
                source_indices.append(scratch)
                destination_indices.append(scratch)
        src.copy_(torch.tensor(source_indices, device=src.device, dtype=torch.int32))
        dst.copy_(torch.tensor(destination_indices, device=dst.device, dtype=torch.int32))
        segment_lengths.copy_(
            torch.tensor(
                [int(row >= 0) for row in rows],
                device=segment_lengths.device,
                dtype=torch.int32,
            )
        )
        return active_source, active_destination

    def _tripwire_launch(
        self,
        device: torch.device,
        stream: torch.cuda.Stream,
        conv: CausalConv1DDecodePrimitive,
        gdn: GatedDeltaDecodePrimitive,
        conv_operands: tuple[Tensor, ...],
        gdn_operands: tuple[Tensor, ...],
    ) -> None:
        def forbidden(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("primitive launch performed allocation, clone, item, or sync")

        with ExitStack() as stack:
            for owner, names in (
                (
                    torch,
                    (
                        "empty",
                        "empty_like",
                        "empty_strided",
                        "zeros",
                        "zeros_like",
                        "ones",
                        "ones_like",
                        "full",
                        "full_like",
                        "tensor",
                    ),
                ),
                (
                    Tensor,
                    (
                        "clone",
                        "item",
                        "new_empty",
                        "new_empty_strided",
                        "new_zeros",
                        "new_ones",
                        "new_full",
                    ),
                ),
                (torch.cuda, ("synchronize",)),
                (torch.cuda.Stream, ("synchronize",)),
            ):
                for name in names:
                    stack.enter_context(mock.patch.object(owner, name, forbidden))
            with torch.cuda.device(device), torch.cuda.stream(stream):
                conv.launch(*conv_operands)
                gdn.launch(*gdn_operands)
        stream.synchronize()


@unittest.skipUnless(RUN_REAL, "CUDA opt-in")
class NativeGDNTests(unittest.TestCase):
    def test_value_tiles_preserve_real_head_shape_with_permuted_padding(self) -> None:
        for dtype in (torch.bfloat16, torch.float16):
            for logical, capacity, dim, value_dim, qh, vh in (
                (1, 1, 64, 96, 2, 6),
                (8, 8, 128, 128, 4, 12),
                (33, 64, 128, 128, 4, 12),
                (3, 8, 96, 192, 1, 4),
            ):
                torch.manual_seed(914)
                packed = torch.randn(capacity, 2 * qh * dim + vh * value_dim,
                                     device="cuda", dtype=dtype)
                normalized = torch.empty_like(packed)
                a = torch.randn(capacity, vh, device="cuda", dtype=dtype)
                b = torch.randn_like(a)
                al = torch.zeros(vh, device="cuda")
                dt = torch.zeros(vh, device="cuda", dtype=dtype)
                source = torch.arange(capacity, device="cuda", dtype=torch.int32)
                destination = source + capacity
                rows = torch.randperm(capacity, device="cuda")[:logical]
                lengths = torch.zeros_like(source)
                lengths[rows] = 1
                packed[lengths == 0] = torch.nan
                seed = torch.randn(capacity, vh, value_dim, dim,
                                   device="cuda", dtype=torch.bfloat16) * 0.01
                expected = None
                for tile in (16, 32, 64, 128):
                    config = OperatorConfig.from_dict({"version": 1, "overrides": {
                        "gdn": {"native_value_tile": tile}}})
                    state = torch.cat((seed, torch.full_like(seed, 19)))
                    out = torch.full((capacity, vh, value_dim), 19, device="cuda", dtype=dtype)
                    args = (packed, normalized, a, b, al, dt, state,
                            source, destination, lengths, out)
                    with operator_strategy_scope(config, device="", model=""):
                        op = GatedDeltaDecodePrimitive(*args, q_heads=qh, scale=dim**-0.5)
                    op.launch(*args)
                    torch.cuda.synchronize()
                    from triton.compiler.compiler import CompiledKernel
                    from cubic_serve.runtime.quantized_qualification import _bound_composition_plan
                    self.assertIsInstance(op._compiled_normalize, CompiledKernel)
                    self.assertIsInstance(op._compiled_recurrence, CompiledKernel)
                    # Inspecting the warmed primitive must bind its real images
                    # without another forward, reference operator or allocation.
                    self.assertTrue(_bound_composition_plan(
                        op, kernels_only=True, workload=True))
                    actual = (out.clone(), state.clone())
                    if expected is None:
                        expected = actual
                    else:
                        self.assertTrue(all(torch.equal(x, y)
                                            for x, y in zip(actual, expected, strict=True)))
                    self.assertTrue(torch.equal(state[:capacity], seed))
                    self.assertTrue((out[lengths == 0] == 19).all().item())
                    self.assertTrue((state[capacity:][lengths == 0] == 19).all().item())

    def test_native_segment_checkpoints_match_separate_decode_steps(self) -> None:
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype), operator_strategy_scope(
                OperatorConfig(), device="", model="",
            ):
                torch.manual_seed(497)
                cap, dim, tokens = 8, 16, 7
                packed = torch.randn(cap, 3 * dim, device="cuda", dtype=dtype)
                packed[-1] = torch.nan
                normalized = torch.empty_like(packed)
                a = torch.randn(cap, 1, device="cuda", dtype=dtype)
                b = torch.randn_like(a)
                al = torch.zeros(1, device="cuda")
                dt = torch.zeros(1, device="cuda", dtype=dtype)
                state = torch.randn(cap, 1, dim, dim, device="cuda", dtype=torch.bfloat16)
                seed = state[0].clone()
                source = torch.zeros(cap, device="cuda", dtype=torch.int32)
                destination = torch.arange(1, cap + 1, device="cuda", dtype=torch.int32)
                destination[-1] = -999  # no segment starts here; it must never be dereferenced
                lengths = torch.zeros_like(source)
                lengths[0] = tokens
                output = torch.full((cap, 1, dim), -99.0, device="cuda", dtype=dtype)
                args = (packed, normalized, a, b, al, dt, state,
                        source, destination, lengths, output)
                primitive = GatedDeltaDecodePrimitive(
                    *args, q_heads=1, scale=0.25, checkpoint_destinations=True
                )
                primitive.launch(*args)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    primitive.launch(*args)
                graph.replay()
                torch.cuda.synchronize()
                checkpoints = state[1:].clone()
                segmented_output = output[:tokens].clone()
                self.assertTrue(torch.equal(state[0], seed))
                self.assertTrue(torch.equal(output[-1], torch.full_like(output[-1], -99.0)))
                one = packed[:1].clone()
                one_norm = torch.empty_like(one)
                one_a, one_b = a[:1].clone(), b[:1].clone()
                one_state = seed[None].clone()
                zero = torch.zeros(1, device="cuda", dtype=torch.int32)
                one_dst = zero.clone()
                one_length = torch.ones_like(zero)
                one_out = torch.empty(1, 1, dim, device="cuda", dtype=dtype)
                single_args = (one, one_norm, one_a, one_b, al, dt, one_state,
                               zero, one_dst, one_length, one_out)
                single = GatedDeltaDecodePrimitive(*single_args, q_heads=1, scale=0.25)
                for step in range(tokens):
                    one.copy_(packed[step:step+1])
                    one_a.copy_(a[step:step+1])
                    one_b.copy_(b[step:step+1])
                    single.launch(*single_args)
                    torch.cuda.synchronize()
                    self.assertTrue(torch.equal(one_state[0], checkpoints[step]))
                    self.assertTrue(torch.equal(one_out[0], segmented_output[step]))

    def test_fp16_mma_accumulator_midpoint_uses_explicit_rn(self) -> None:
        # The exact product is -9.298673830926418e-9. Adding it to the
        # BF16 midpoint -0.22314453125 must round downward in FP32 before
        # BF16 storage; an MMA nonzero accumulator can instead keep the tie.
        with operator_strategy_scope(OperatorConfig(), device="", model=""):
            packed = torch.zeros(1, 48, device="cuda", dtype=torch.float16)
            packed[0, 0] = 1
            packed[0, 16] = 0.0006093978881835938
            packed[0, 17] = 1
            decayed = torch.tensor(-0.22314453125, dtype=torch.float16)
            k = torch.tensor(0.0006093978881835938, dtype=torch.float16)
            memory = (decayed.float() * k.float()).half()
            v = (memory - torch.tensor(2**-15, dtype=torch.float16)).half()
            packed[0, 32:] = float(v)
            normalized = torch.empty_like(packed)
            a = torch.full((1, 1), -6.0, device="cuda", dtype=torch.float16)
            b = torch.zeros_like(a)
            al = torch.zeros(1, device="cuda")
            dt = torch.zeros(1, device="cuda", dtype=torch.float16)
            state = torch.zeros(2, 1, 16, 16, device="cuda", dtype=torch.bfloat16)
            state[0, 0, :, 0] = -0.2236328125
            source = torch.zeros(1, device="cuda", dtype=torch.int32)
            destination = torch.ones_like(source)
            lengths = torch.ones_like(source)
            output = torch.empty(1, 1, 16, device="cuda", dtype=torch.float16)
            args = (packed, normalized, a, b, al, dt, state,
                    source, destination, lengths, output)
            primitive = GatedDeltaDecodePrimitive(*args, q_heads=1, scale=0.25)
            primitive.launch(*args)
            torch.cuda.synchronize()
            expected = (decayed.double() + ((v - memory) * 0.5).double()
                        * k.double()).float().bfloat16()
            self.assertEqual(float(expected), -0.2236328125)
            self.assertTrue(torch.equal(state[1, 0, :, 0], expected.cuda().expand(16)))

    def test_dtype_mma_state_rounding_and_permuted_graph_rows(self) -> None:
        configure_reference_math()
        from cubic_serve.ops.cuda.linear_attention.gated_delta_decode import (
            _gated_delta_decode_kernel,
        )
        for dtype in (torch.float16, torch.bfloat16):
            for state_dtype in (torch.bfloat16, torch.float32):
                with self.subTest(dtype=dtype, state=state_dtype), operator_strategy_scope(
                    OperatorConfig(), device="", model="",
                ):
                    common = None
                    for capacity, logical in ((1, 1), (8, 8), (64, 33)):
                        torch.manual_seed(233)
                        dim, heads = 16, 2
                        packed = torch.zeros(capacity, 2 * dim + heads * dim,
                                             device="cuda", dtype=dtype)
                        packed[:, 0] = 1
                        packed[:, dim] = 1
                        packed[:, 2 * dim:] = 0.25
                        normalized = torch.empty_like(packed)
                        a = torch.zeros(capacity, heads, device="cuda", dtype=dtype)
                        b = torch.zeros_like(a)
                        al = torch.zeros(heads, device="cuda")
                        dt = torch.zeros(heads, device="cuda", dtype=dtype)
                        state = torch.full((capacity, heads, dim, dim), 0.125,
                                           device="cuda", dtype=state_dtype)
                        source = torch.arange(capacity, device="cuda", dtype=torch.int32)
                        destination = source.clone()
                        lengths = torch.ones_like(source)
                        rows = torch.randperm(capacity, device="cuda")[:logical]
                        padding = torch.ones(capacity, device="cuda", dtype=torch.bool)
                        padding[rows] = False
                        packed[padding] = torch.nan
                        state[padding] = torch.nan
                        out = torch.empty(capacity, heads, dim, device="cuda", dtype=dtype)
                        args = (packed, normalized, a, b, al, dt, state,
                                source, destination, lengths, out)
                        op = GatedDeltaDecodePrimitive(*args, q_heads=1, scale=0.25)
                        op.launch(*args)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            op.launch(*args)
                        state[rows] = 0.125
                        for _ in range(128):
                            graph.replay()
                        torch.cuda.synchronize()
                        actual, actual_state = out[rows].clone(), state[rows].clone()
                        with ReferenceExecutionContext():
                            q = torch.zeros(logical, 128, 1, dim, device="cuda", dtype=dtype)
                            q[..., 0] = 1
                            expected, final_state = gated_delta_net_recurrent(
                                q, q.clone(),
                                torch.full((logical, 128, heads, dim), 0.25,
                                           device="cuda", dtype=dtype),
                                decay_input=torch.zeros(logical, 128, heads,
                                                        device="cuda", dtype=dtype),
                                beta_input=torch.zeros(logical, 128, heads,
                                                       device="cuda", dtype=dtype),
                                a_log=al, dt_bias=dt, l2_epsilon=1e-6,
                                state=torch.full_like(actual_state, 0.125), output_dtype=dtype,
                            )
                        self.assertTrue(torch.equal(actual, expected[:, -1]))
                        self.assertTrue(torch.equal(actual_state, final_state))
                        if common is not None:
                            self.assertTrue(torch.equal(common[0], actual[0]))
                            self.assertTrue(torch.equal(common[1], actual_state[0]))
                        common = actual[0].clone(), actual_state[0].clone()
                        compiled = _gated_delta_decode_kernel.warmup(
                            *args, grid=(1, capacity * heads), CAPACITY=capacity,
                            SCALE=0.25, Q_HEADS=1, V_HEADS=heads, K_DIM=dim, V_DIM=dim,
                            PACKED_DIM=2 * dim + heads * dim, BLOCK_K=16, BLOCK_V=16,
                            CHECKPOINTS=False, STATE_BF16=state_dtype is torch.bfloat16,
                            STATE_STRIDE=state.stride(0),
                            NATIVE=True, ACT_TYPE=op._activation_type, num_warps=4, num_stages=3,
                        )
                        suffix = "bf16" if dtype is torch.bfloat16 else "f16"
                        self.assertIn(f"fma.rn.{suffix}x2", compiled.asm["ptx"])
                        self.assertIn(f".f32.{suffix}.{suffix}.f32", compiled.asm["ptx"])


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(RUN_REAL, "set CUBIC_RUN_CUDA_GDN_TESTS=1")
class GDNPrecomputedCoefficientTests(unittest.TestCase):
    def test_segmented_padded_state_identity(self):
        from cubic_serve.ops.cuda.linear_attention.gated_delta_decode import (
            GatedDeltaDecodePrimitive,
        )
        from cubic_serve.ops.cuda.pointwise.qwen_decode import PackedGDNProjectionSplitPrimitive

        for dtype in (torch.bfloat16, torch.float16):
            for H in (12, 16):
                for logical, C in ((1, 1), (33, 64), (128, 128)):
                    torch.manual_seed(31)
                    D = 128
                    QH = 4
                    p = torch.randn(C, H * (D + 2), device="cuda", dtype=dtype)
                    p[logical:] = float("nan")
                    x = torch.randn(C, 2 * QH * D + H * D, device="cuda", dtype=dtype)
                    x[logical:] = float("nan")
                    a = torch.empty(C, H, device="cuda", dtype=dtype)
                    b = torch.empty_like(a)
                    z = torch.empty(C, H * D, device="cuda", dtype=dtype)
                    co = torch.empty(C, H, 2, device="cuda")
                    al = torch.randn(H, device="cuda")
                    dt = torch.randn(H, device="cuda", dtype=dtype)
                    split = PackedGDNProjectionSplitPrimitive(
                        p, a, b, z, coefficients=co, a_log=al, dt_bias=dt
                    )
                    seed = torch.randn(2 * C, H, D, D, device="cuda", dtype=torch.bfloat16) * 0.01
                    states = [seed.clone(), seed.clone()]
                    outs = [torch.zeros(C, H, D, device="cuda", dtype=dtype) for _ in range(2)]
                    norm = [torch.empty_like(x), torch.empty_like(x)]
                    src = torch.arange(C, device="cuda", dtype=torch.int32)
                    dst = src + C
                    seg = torch.zeros_like(src)
                    seg[:logical] = 1
                    # Explicit segment continuation, with a per-token checkpoint destination.
                    if logical > 3:
                        seg[0] = 3
                        seg[1:3] = 0
                    ops = [
                        GatedDeltaDecodePrimitive(
                            x,
                            norm[i],
                            a,
                            b,
                            al,
                            dt,
                            states[i],
                            src,
                            dst,
                            seg,
                            outs[i],
                            q_heads=QH,
                            scale=D**-0.5,
                            checkpoint_destinations=True,
                        )
                        for i in range(2)
                    ]
                    for step in range(8):
                        split.launch(p, a, b, z, coefficients=co, a_log=al, dt_bias=dt)
                        for i in range(2):
                            ops[i].launch(
                                x,
                                norm[i],
                                a,
                                b,
                                al,
                                dt,
                                states[i],
                                src,
                                dst,
                                seg,
                                outs[i],
                                coefficients=co if i else None,
                            )
                        torch.cuda.synchronize()
                        assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16)), (
                            dtype,
                            H,
                            logical,
                            "output",
                            step,
                        )
                        assert torch.equal(
                            states[0].view(torch.int16), states[1].view(torch.int16)
                        ), (dtype, H, logical, "state", step)
                        for s in states:
                            s[:C].copy_(s[C:])


def test_native_recurrent_slot_stride_preserves_graph_bytes_and_neighbor_fields():
    import pytest
    if not RUN_REAL:
        pytest.skip("explicit CUDA GDN audit")
    from cubic_serve.ops.cuda.linear_attention.gated_delta_native import bind
    from cubic_serve.runtime.cache.shared_pool import SharedCacheField, SharedCacheLayout

    configure_reference_math()
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(381)
    rows, slots, qh, vh, kd, vd = 4, 12, 1, 2, 32, 32
    width = 2 * qh * kd + vh * vd
    packed = torch.randn(rows, width, device=device, dtype=torch.bfloat16).mul_(.1)
    a = torch.randn(rows, vh, device=device, dtype=packed.dtype)
    b = torch.randn_like(a)
    alog = torch.zeros(vh, device=device)
    bias = torch.zeros(vh, device=device, dtype=packed.dtype)
    sources = torch.tensor([0, 0, -991, 3], dtype=torch.int32, device=device)
    destinations = torch.tensor([5, 6, -993, 8], dtype=torch.int32, device=device)
    segments = torch.tensor([2, 0, 0, 1], dtype=torch.int32, device=device)
    activity = torch.tensor([1, 1, 0, 1], dtype=torch.int32, device=device)
    for dtype in (torch.bfloat16, torch.float32):
        layout = SharedCacheLayout((SharedCacheField("matrix", dtype, (vh, vd, kd)),
                                    SharedCacheField("neighbor", torch.int64, (17,))))
        raw = torch.full((slots * layout.entry_bytes,), 0x5A, dtype=torch.uint8, device=device)
        fields = layout.views(raw, slots)
        state = fields["matrix"]
        seed = torch.randn(state.shape, device=device, dtype=dtype).mul_(.1)
        neighbor = fields["neighbor"].clone()
        for compute in ("simt", "mma"):
            state.copy_(seed)
            contiguous = seed.clone()
            outputs, calls = [], []
            for storage in (contiguous, state):
                output = torch.full((rows, vh, vd), 19., dtype=packed.dtype, device=device)
                # MMA consumes the supplied producer output; both copies carry
                # exactly the same bytes. This test proves addressing/capture
                # identity, not the independent numerical contract of that producer.
                normalized = packed.clone()
                calls.append(bind((packed, normalized, a, b, alog, bias, storage,
                                   sources, destinations, segments, output), activity,
                                  q_heads=qh, scale=kd**-.5, state_compute=compute,
                                  normalized_inputs=compute == "mma"))
                outputs.append(output)
            for call in calls:
                call.launch()
            torch.cuda.synchronize()
            assert torch.equal(state, contiguous)
            assert torch.equal(outputs[0], outputs[1])
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                calls[1].launch()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(state, contiguous)
            assert torch.equal(outputs[0], outputs[1])
            assert torch.equal(fields["neighbor"], neighbor)
            assert torch.equal(state[:5], seed[:5])
            graph.reset()


def test_serving_recurrent_slot_stride_preserves_checkpoints_and_neighbors():
    import pytest
    if not RUN_REAL:
        pytest.skip("explicit CUDA GDN audit")
    from cubic_serve.runtime.cache.shared_pool import SharedCacheField, SharedCacheLayout

    configure_reference_math()
    torch.manual_seed(382)
    device = torch.device("cuda", torch.cuda.current_device())
    rows, slots, qh, vh, kd, vd = 4, 12, 1, 2, 32, 32
    width = 2 * qh * kd + vh * vd
    packed = torch.randn(rows, width, device=device, dtype=torch.bfloat16).mul_(.1)
    a = torch.randn(rows, vh, device=device, dtype=packed.dtype)
    b = torch.randn_like(a)
    alog = torch.zeros(vh, device=device)
    bias = torch.zeros(vh, device=device, dtype=packed.dtype)
    sources = torch.tensor([0, 0, -991, 3], dtype=torch.int32, device=device)
    destinations = torch.tensor([5, 6, -993, 8], dtype=torch.int32, device=device)
    segments = torch.tensor([2, 0, 0, 1], dtype=torch.int32, device=device)
    config = OperatorConfig.from_dict({"version": 1, "overrides": {
        "gdn": {"native_value_tile": 16}}})
    for dtype in (torch.bfloat16, torch.float32):
        layout = SharedCacheLayout((SharedCacheField("matrix", dtype, (vh, vd, kd)),
                                    SharedCacheField("neighbor", torch.int64, (17,))))
        raw = torch.full((slots * layout.entry_bytes,), 0x5A, dtype=torch.uint8, device=device)
        fields = layout.views(raw, slots)
        state = fields["matrix"]
        seed = torch.randn(state.shape, device=device, dtype=dtype).mul_(.1)
        state.copy_(seed)
        neighbor = fields["neighbor"].clone()
        outputs, calls = [], []
        for storage in (seed.clone(), state):
            output = torch.full((rows, vh, vd), 19., dtype=packed.dtype, device=device)
            normalized = torch.empty_like(packed)
            args = (packed, normalized, a, b, alog, bias, storage,
                    sources, destinations, segments, output)
            with operator_strategy_scope(config, device="", model=""):
                primitive = GatedDeltaDecodePrimitive(*args, q_heads=qh, scale=kd**-.5,
                                                      checkpoint_destinations=True)
            calls.append((primitive, args))
            outputs.append(output)
        for primitive, args in calls:
            primitive.launch(*args)
        torch.cuda.synchronize()
        assert torch.equal(state, calls[0][1][6])
        assert torch.equal(outputs[0], outputs[1])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            calls[1][0].launch(*calls[1][1])
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(state, calls[0][1][6])
        assert torch.equal(outputs[0], outputs[1])
        assert torch.equal(fields["neighbor"], neighbor)
        assert torch.equal(state[:5], seed[:5])
        graph.reset()


def test_native_geometry_receipt_is_written_and_reused_with_real_language_dtype(tmp_path, monkeypatch):
    import pytest
    if not RUN_REAL:
        pytest.skip('explicit CUDA GDN audit')
    from cubic_serve.ops.cuda import launch_cost
    from cubic_serve.ops.cuda.startup import startup_weight_identity
    monkeypatch.setenv('CUBIC_SERVE_CACHE_DIR', str(tmp_path))
    launch_cost.startup_operator_choices.cache_clear()
    c, qh, vh, kd, vd = 4, 8, 24, 128, 128
    device, dtype = torch.device('cuda', 0), torch.bfloat16
    qkv = torch.zeros((c, 2 * qh * kd + vh * vd), device=device, dtype=dtype)
    normalized = torch.empty_like(qkv)
    a = torch.zeros((c, vh), device=device, dtype=dtype)
    b = torch.zeros_like(a)
    al = torch.zeros(vh, device=device, dtype=dtype)
    dt = torch.zeros_like(al)
    state = torch.zeros((2 * c, vh, vd, kd), device=device, dtype=dtype)
    src = torch.arange(c, device=device, dtype=torch.int32)
    dst, lengths = src + c, torch.ones_like(src)
    out = torch.empty((c, vh, vd), device=device, dtype=dtype)
    args = qkv, normalized, a, b, al, dt, state, src, dst, lengths, out
    with (operator_strategy_scope(OperatorConfig(), device='', model=''),
          startup_weight_identity('isolated-gdn-geometry-fixture', allow_calibration=True)):
        first = GatedDeltaDecodePrimitive(*args, q_heads=qh, scale=kd**-.5)
        files = list(tmp_path.glob('operators/*/launch/*.json'))
        assert files, 'real tl.dtype metadata must not disable the persistent path'
        stamps = {p.name: p.stat().st_mtime_ns for p in files}
        launch_cost.startup_operator_choices.cache_clear()
        def forbidden(*a, **kw):
            raise AssertionError('valid persisted geometry was measured again')
        monkeypatch.setattr(launch_cost, '_time_graph', forbidden)
        second = GatedDeltaDecodePrimitive(*args, q_heads=qh, scale=kd**-.5)
        assert first.launch_signature == second.launch_signature
        assert stamps == {p.name: p.stat().st_mtime_ns for p in files}


def test_packed_layer_state_addressing_matches_flattened_contiguous_storage():
    import pytest
    if not RUN_REAL:
        pytest.skip('explicit CUDA packed-layer state proof')
    from cubic_serve.runtime.cache.shared_pool import SharedCacheField, SharedCacheLayout
    configure_reference_math()
    torch.manual_seed(825)
    c,n,l,qh,vh,kd,vd=4,12,3,1,2,32,32
    d=2*qh*kd+vh*vd
    layout=SharedCacheLayout((SharedCacheField('conv',torch.bfloat16,(l,d,3)),
        SharedCacheField('matrix',torch.bfloat16,(l,vh,vd,kd)),
        SharedCacheField('guard',torch.int64,(17,))))
    raw=torch.full((n*layout.entry_bytes,),0x39,dtype=torch.uint8,device='cuda')
    fields=layout.views(raw,n)
    for name in ('conv','matrix'):fields[name].copy_(torch.randn(fields[name].shape,device='cuda').mul_(.1))
    protected=fields['guard'].clone()
    source=torch.tensor([2,2,-991,10],dtype=torch.int32,device='cuda')
    dest=torch.tensor([17,20,-991,28],dtype=torch.int32,device='cuda')
    lengths=torch.tensor([2,0,0,1],dtype=torch.int32,device='cuda')
    x=torch.randn(c,d,device='cuda',dtype=torch.bfloat16).mul_(.1)
    weight=torch.randn(d,4,device='cuda',dtype=torch.bfloat16).mul_(.1)
    a=torch.zeros(c,vh,device='cuda',dtype=x.dtype);b=a.clone()
    alog=torch.zeros(vh,device='cuda');bias=torch.zeros(vh,device='cuda',dtype=x.dtype)
    records=[]
    for packed in (False,True):
        conv=fields['conv'] if packed else fields['conv'].clone().reshape(n*l,d,3)
        state=fields['matrix'] if packed else fields['matrix'].clone().reshape(n*l,vh,vd,kd)
        convolved=torch.full_like(x,19);norm=torch.empty_like(x)
        out=torch.full((c,vh,vd),19,device='cuda',dtype=x.dtype)
        cvargs=(x,weight,conv,source,dest,lengths,convolved)
        gdargs=(convolved,norm,a,b,alog,bias,state,source,dest,lengths,out)
        cv=CausalConv1DDecodePrimitive(*cvargs,checkpoint_destinations=True)
        gd=GatedDeltaDecodePrimitive(*gdargs,q_heads=qh,scale=kd**-.5,checkpoint_destinations=True)
        cv.launch(*cvargs);gd.launch(*gdargs)
        if packed:
            graph=torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):cv.launch(*cvargs);gd.launch(*gdargs)
                graph.replay();torch.cuda.synchronize()
            finally:graph.reset()
        records.append((conv.reshape(n*l,d,3),state.reshape(n*l,vh,vd,kd),convolved,out))
    for left,right in zip(*records,strict=True):assert torch.equal(left,right)
    assert torch.equal(fields['guard'],protected)
