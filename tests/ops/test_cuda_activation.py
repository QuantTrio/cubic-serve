from __future__ import annotations

import os
import unittest

import torch
import triton
import triton.language as tl

from cubic_serve.ops.cuda.activation import exp_nonpositive, sigmoid, silu
from cubic_serve.ops.cuda.attention.paged_decode import PagedGQADecodePrimitive
from cubic_serve.ops.cuda.pointwise.qwen_decode import SwiGLUPrimitive
from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope


@triton.jit
def _probe(X, E, S, Y, N: tl.constexpr, DTYPE: tl.constexpr):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    x = tl.load(X + offsets, offsets < N, 0.0)
    tl.store(E + offsets, exp_nonpositive(-tl.abs(x), DTYPE), offsets < N)
    tl.store(S + offsets, sigmoid(x, DTYPE), offsets < N)
    tl.store(Y + offsets, silu(x, DTYPE), offsets < N)


@unittest.skipUnless(os.environ.get("CUBIC_RUN_CUDA_ACTIVATION_TESTS") == "1", "CUDA opt-in")
class ActivationTests(unittest.TestCase):
    def test_all_finite_low_precision_inputs_and_graph_identity(self) -> None:
        for dtype, kernel_dtype, error_bits in (
            (torch.float16, tl.float16, 9.9),
            (torch.bfloat16, tl.bfloat16, 7.0),
        ):
            with self.subTest(dtype=dtype):
                self._audit(dtype, kernel_dtype, error_bits)

    def test_fp16_fallback_preserves_finite_bf16_tails(self) -> None:
        source = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        source = source[torch.isfinite(source)].contiguous().cuda()
        exponential = torch.empty(source.shape, dtype=torch.float16, device="cuda")
        gate = torch.empty_like(exponential)
        output = torch.empty_like(source)
        _probe[(triton.cdiv(source.numel(), 256),)](
            source, exponential, gate, output, source.numel(), tl.float16
        )
        torch.cuda.synchronize()
        x = source.cpu()
        e = exponential.cpu()
        numerator = torch.where(x >= 0, torch.ones_like(e), e)
        denominator = torch.ones_like(e) + e
        expected_gate = (numerator.double() / denominator.double()).half()
        self.assertTrue(torch.equal(gate.cpu(), expected_gate))
        saturated = x.double().abs() > 65504.0
        safe = torch.where(saturated, 0.0, x).half()
        product = (safe.double() * expected_gate.double()).half().bfloat16()
        expected = torch.where(saturated, torch.where(x > 0, x, 0.0), product)
        self.assertTrue(torch.equal(output.cpu(), expected))
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertTrue(
            torch.equal(output[source.float() > 65504.0], source[source.float() > 65504.0])
        )

        from unittest.mock import patch

        from cubic_serve.executor.runner import ReferenceExecutionContext
        from cubic_serve.ops.eager.activation import silu as reference_silu

        with (
            ReferenceExecutionContext(),
            operator_strategy_scope(OperatorConfig(softmax_dtype="auto"), device="", model=""),
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)),
        ):
            reference = reference_silu(source)
        self.assertTrue(bool(torch.isfinite(reference).all()))
        large = source.float() > 65504.0
        self.assertTrue(torch.equal(reference[large], source[large]))

    def test_native_softmax_long_uniform_context_is_finite(self) -> None:
        with operator_strategy_scope(OperatorConfig(softmax_dtype="auto"), device="", model=""):
            for dtype in (torch.float16, torch.bfloat16):
                # Equal logits have the largest possible unscaled denominator.
                query = torch.zeros((1, 1, 16), dtype=dtype, device="cuda")
                key = torch.zeros((8192, 1, 16, 1, 16), dtype=dtype, device="cuda")
                value = torch.ones_like(key)
                table = torch.arange(8192, dtype=torch.int32, device="cuda")[None]
                lengths = torch.tensor([65536], dtype=torch.int32, device="cuda")
                output = torch.empty_like(query)
                partial = torch.empty((1, 1, 32, 16), device="cuda")
                statistics = torch.empty((1, 1, 32, 2), device="cuda")
                arrivals = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
                primitive = PagedGQADecodePrimitive(
                    query,
                    key,
                    value,
                    table,
                    lengths,
                    output,
                    layer=0,
                    partial=partial,
                    statistics=statistics,
                    arrivals=arrivals,
                )
                primitive.launch(query, key, value, table, lengths, output)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    primitive.launch(query, key, value, table, lengths, output)
                for length in (65536, 131072):
                    lengths.fill_(length)
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertTrue(torch.equal(output, torch.ones_like(output)))
                    self.assertEqual(int(arrivals.count_nonzero()), 0)

    def test_native_swiglu_cross_capacity_and_permuted_padding(self) -> None:
        with operator_strategy_scope(OperatorConfig(softmax_dtype="auto"), device="", model=""):
            for dtype in (torch.float16, torch.bfloat16):
                gen = torch.Generator().manual_seed(447)
                gate = torch.randn((33, 32), generator=gen).to(dtype).cuda()
                up = torch.randn((33, 32), generator=gen).to(dtype).cuda()
                common = None
                for capacity, logical in ((1, 1), (8, 8), (64, 33)):
                    a = torch.full((capacity, 32), float("nan"), dtype=dtype, device="cuda")
                    b = torch.empty_like(a)
                    out = torch.empty_like(a)
                    primitive = SwiGLUPrimitive(a, b, out)
                    primitive.launch(a, b, out)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        primitive.launch(a, b, out)
                    rows = torch.arange(logical - 1, -1, -1, device="cuda")
                    a.fill_(float("nan"))
                    b.fill_(float("nan"))
                    a[rows] = gate[:logical]
                    b[rows] = up[:logical]
                    graph.replay()
                    torch.cuda.synchronize()
                    ordered = out[rows].clone()
                    self.assertTrue(bool(torch.isfinite(ordered).all()))
                    if common is not None:
                        self.assertTrue(torch.equal(ordered[: common.shape[0]], common))
                    common = ordered

    def test_native_route_weights_have_independent_exponential_enclosure(self) -> None:
        from cubic_serve.ops.cuda.moe.fused import _sort_and_normalize_topk

        for dtype, kernel_dtype, bits in (
            (torch.float16, tl.float16, 9.9),
            (torch.bfloat16, tl.bfloat16, 7.0),
        ):
            base = torch.tensor(
                [-3.484375, -3.640625, -3.703125, -3.984375, -4.0, -4.34375, -4.375, -4.4375],
                dtype=torch.float32,
            )
            ids = torch.tensor([15, 2, 9, 0, 12, 1, 7, 4], dtype=torch.int64)
            order = ids.argsort()
            argument = (base - base.max()).to(dtype) * torch.tensor(1.4426950408889634, dtype=dtype)
            exact = torch.exp2(argument.double())
            floor = torch.finfo(dtype).tiny if dtype is torch.bfloat16 else 2.0**-24
            error = exact * 2.0**-bits + floor
            lower, upper = (exact - error).clamp_min(0), exact + error
            gamma = 16 * 2.0**-24 / (1 - 16 * 2.0**-24)
            lo = lower / (upper.sum() * (1 + gamma))
            hi = upper / (lower.sum() * (1 - gamma))
            baseline = None
            for capacity, logical in ((8, 8), (64, 33), (128, 33)):
                scores = base.cuda().repeat(capacity, 1)
                expert_ids = torch.full((capacity, 8), -1, dtype=torch.int64, device="cuda")
                row_map = torch.arange(logical - 1, -1, -1, device="cuda")
                expert_ids[row_map] = ids.cuda()
                route_ids = torch.empty_like(expert_ids)
                weights = torch.empty((capacity, 8), dtype=dtype, device="cuda")

                def launch(
                    capacity=capacity,
                    scores=scores,
                    expert_ids=expert_ids,
                    route_ids=route_ids,
                    weights=weights,
                    kernel_dtype=kernel_dtype,
                ):
                    _sort_and_normalize_topk[(capacity,)](
                        scores,
                        expert_ids,
                        route_ids,
                        weights,
                        TOP_K=8,
                        BLOCK=8,
                        ACT_TYPE=kernel_dtype,
                        num_warps=1,
                    )

                launch()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    launch()
                graph.replay()
                torch.cuda.synchronize()
                actual = weights[row_map].cpu()
                ulp = (
                    torch.nextafter(actual, torch.full_like(actual, float("inf"))).double()
                    - actual.double()
                ).abs()
                self.assertTrue(bool((actual.double() >= lo[order] - ulp / 2).all()))
                self.assertTrue(bool((actual.double() <= hi[order] + ulp / 2).all()))
                self.assertTrue(
                    torch.equal(route_ids[row_map].cpu(), ids[order].expand(logical, -1))
                )
                self.assertFalse(bool(weights[logical:].view(torch.int16).count_nonzero()))
                self.assertTrue(bool((route_ids[logical:] == -1).all()))
                if baseline is not None:
                    self.assertTrue(torch.equal(actual[: baseline.shape[0]], baseline))
                baseline = actual

    def _audit(self, dtype, kernel_dtype, error_bits):
        bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
        x = bits.view(dtype)
        x = x[torch.isfinite(x)].contiguous().cuda()
        exponential, gate, output = (torch.empty_like(x) for _ in range(3))
        compiled = _probe[(triton.cdiv(x.numel(), 256),)](
            x, exponential, gate, output, x.numel(), kernel_dtype
        )
        torch.cuda.synchronize()
        self.assertIn("ex2.approx", compiled.asm["ptx"])
        expected_outputs = [t.clone() for t in (exponential, gate, output)]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _probe[(triton.cdiv(x.numel(), 256),)](
                x, exponential, gate, output, x.numel(), kernel_dtype
            )
        graph.replay()
        torch.cuda.synchronize()
        for actual, expected in zip((exponential, gate, output), expected_outputs, strict=True):
            self.assertTrue(torch.equal(actual, expected))

        # Independent producing-boundary witness: PTX's documented exp2 error,
        # applied to the actual low-precision argument, including subnormal output.
        source = x.cpu()
        argument = -source.abs() * torch.tensor(1.4426950408889634, dtype=dtype)
        exact_exp = torch.exp2(argument.double())
        actual_exp = exponential.cpu()
        floor = torch.finfo(dtype).tiny if dtype is torch.bfloat16 else 2.0**-24
        bound = exact_exp * 2.0**-error_bits + floor
        self.assertTrue(bool(((actual_exp.double() - exact_exp).abs() <= bound).all()))
        # Candidate-input causal rule: denominator and division consume the
        # independently bounded candidate exponential, never a mixed producer.
        denominator = torch.ones_like(actual_exp) + actual_exp
        numerator = torch.where(source >= 0, torch.ones_like(source), actual_exp)
        expected_gate = (numerator.double() / denominator.double()).to(dtype)
        self.assertTrue(torch.equal(gate.cpu(), expected_gate))
        self.assertTrue(
            torch.equal(output.cpu(), (source.double() * expected_gate.double()).to(dtype))
        )
        self.assertTrue(bool(torch.isfinite(output).all()))


class ActivationPolicyTests(unittest.TestCase):
    def test_softmax_override_preserves_low_precision_activations(self) -> None:
        from unittest.mock import patch

        from cubic_serve.executor.runner import ReferenceExecutionContext
        from cubic_serve.ops.cuda.activation import activation_type, softmax_type
        from cubic_serve.ops.eager.activation import sigmoid, silu, softmax

        auto = OperatorConfig(softmax_dtype="auto")
        mixed = OperatorConfig.from_dict({"version": 1, "softmax_dtype": "fp32"})
        self.assertNotEqual(auto.digest, mixed.digest)
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.tensor([[-3.125, -0.25, 0.5, 2.125]], dtype=dtype)
            with ReferenceExecutionContext():
                with operator_strategy_scope(auto, device="", model=""):
                    expected_silu, expected_sigmoid = silu(x), sigmoid(x)
                with operator_strategy_scope(mixed, device="", model=""):
                    self.assertTrue(torch.equal(silu(x), expected_silu))
                    self.assertTrue(torch.equal(sigmoid(x), expected_sigmoid))
                    actual_softmax = softmax(x, dim=-1, storage_dtype=dtype)
                    with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
                        self.assertEqual(
                            activation_type(dtype, torch.device("cuda:0")),
                            tl.float16 if dtype is torch.float16 else tl.bfloat16,
                        )
                        self.assertEqual(softmax_type(dtype, torch.device("cuda:0")), tl.float32)
                with operator_strategy_scope(
                    OperatorConfig(activation_dtype="fp32"), device="", model=""
                ):
                    self.assertTrue(torch.equal(
                        actual_softmax, softmax(x, dim=-1, storage_dtype=dtype)
                    ))
        with self.assertRaises(ValueError):
            OperatorConfig(softmax_dtype="fp16")

    def test_policy_is_immutable_serializable_and_capability_resolved(self) -> None:
        import pickle
        from unittest.mock import patch

        from cubic_serve.ops.cuda.activation import activation_type

        auto = OperatorConfig.from_dict({"version": 1})
        self.assertEqual((auto.attention_dtype, auto.activation_dtype, auto.softmax_dtype),
                         ("auto", "auto", "fp32"))
        wide = OperatorConfig.from_dict({"version": 1, "activation_dtype": "fp32"})
        self.assertEqual(pickle.loads(pickle.dumps(wide)), wide)
        self.assertNotEqual(auto.digest, wide.digest)
        device = torch.device("cuda:0")
        with operator_strategy_scope(auto, device="", model=""):
            with patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)):
                self.assertEqual(activation_type(torch.bfloat16, device), tl.float16)
            with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
                self.assertEqual(activation_type(torch.bfloat16, device), tl.bfloat16)
            self.assertEqual(activation_type(torch.float16, device), tl.float16)
        with operator_strategy_scope(wide, device="", model=""):
            self.assertEqual(activation_type(torch.bfloat16, device), tl.float32)
        with self.assertRaises(ValueError):
            OperatorConfig.from_dict({"version": 1, "activation_dtype": "fp64"})

    def test_serve_exposes_explicit_fp32_override(self) -> None:
        from cubic_serve.cli import build_parser

        args = build_parser().parse_args(["serve", "/checkpoint", "--activation-dtype", "fp32"])
        self.assertEqual(args.activation_dtype, "fp32")
        mixed = build_parser().parse_args([
            "serve", "/checkpoint", "--activation-dtype", "auto", "--softmax-dtype", "fp32"
        ])
        self.assertEqual((mixed.activation_dtype, mixed.softmax_dtype), ("auto", "fp32"))


if __name__ == "__main__":
    unittest.main()
