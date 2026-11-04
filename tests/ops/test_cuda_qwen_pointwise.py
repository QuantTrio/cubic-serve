from __future__ import annotations

import os
import unittest
from contextlib import ExitStack
from unittest import mock

import torch
import torch.nn.functional as F
from torch import Tensor

from cubic_serve.ops.cuda.pointwise import (
    BF16PackedGDNProjectionSplitPrimitive,
    BF16PackedSwiGLUPrimitive,
    BF16ResidualAddPrimitive,
    BF16SigmoidOutputGatePrimitive,
    BF16SwiGLUPrimitive,
    GatedRMSNormPrimitive,
    QwenAttentionQGateSplitPrimitive,
)
from cubic_serve.ops.cuda.pointwise import qwen_decode as implementation
from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

RUN_REAL = os.environ.get("CUBIC_RUN_CUDA_QWEN_POINTWISE_TESTS") == "1"
if RUN_REAL:
    from cubic_serve.executor.runner import configure_reference_math

    configure_reference_math()
_C, _HIDDEN, _INTERMEDIATE, _HEADS, _HEAD_DIM = 4, 128, 192, 3, 64
_ROWS = {
    1: (-1, 0, -3, -2),
    2: (1, -2, -1, 0),
    4: (2, 0, 3, 1),
}


def _oracle(
    residual: Tensor,
    update: Tensor,
    mlp_gate: Tensor,
    mlp_up: Tensor,
    attention: Tensor,
    attention_gate: Tensor,
    norm_inputs: Tensor,
    norm_gate: Tensor,
    norm_weight: Tensor,
) -> dict[str, Tensor]:
    """Independent PyTorch definition; no production helper is reused."""

    residual_output = (residual + update).to(residual.dtype)
    mlp_silu = F.silu(mlp_gate).to(residual.dtype)
    mlp_output = (mlp_silu * mlp_up).to(residual.dtype)
    sigmoid = torch.sigmoid(attention_gate).to(residual.dtype)
    attention_output = (attention * sigmoid).to(residual.dtype)

    norm_fp32 = norm_inputs.float()
    variance = (norm_fp32 * norm_fp32).sum(-1, keepdim=True) / norm_inputs.shape[-1]
    normalized_bf16 = (norm_fp32 * torch.rsqrt(variance + 1e-6)).to(residual.dtype)
    weighted = normalized_bf16 * norm_weight
    if norm_weight.dtype is residual.dtype:
        weighted = weighted.to(residual.dtype)
    activated_gate = F.silu(norm_gate.float())
    norm_output = (weighted.float() * activated_gate).to(residual.dtype)
    return {
        "residual": residual_output,
        "swiglu": mlp_output,
        "sigmoid_gate": attention_output,
        "gated_rms_norm": norm_output,
    }


def _addresses(tensors: tuple[Tensor, ...]) -> tuple[int, ...]:
    return tuple(tensor.data_ptr() for tensor in tensors)


@unittest.skipUnless(
    RUN_REAL, "set CUBIC_RUN_CUDA_QWEN_POINTWISE_TESTS=1 for real CUDA qualification"
)
class QwenPointwisePrimitiveTests(unittest.TestCase):
    def setUp(self) -> None:
        # Historical midpoint/reduction corpus qualifies the explicit FP32
        # activation mode. Native low-precision arithmetic has its own audit.
        self.enterContext(operator_strategy_scope(
            OperatorConfig(activation_dtype="fp32"), device="", model=""
        ))

    def test_fp32_weight_gate_division_keeps_activation_midpoints(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            self._qualify_gated_midpoints(dtype)

    def _qualify_gated_midpoints(self, dtype: torch.dtype) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        device = torch.device("cuda:0")
        generator = torch.Generator(device=device).manual_seed(33)
        shape = (64, 64, 128)
        inputs = (torch.randn(shape, generator=generator, device=device) * 0.02).to(dtype)
        gate = torch.randn(shape, generator=generator, device=device).to(dtype)
        weight = torch.randn(128, generator=generator, device=device) * 4
        output = torch.empty_like(inputs)
        variance = inputs.float().square().mean(-1, keepdim=True)
        normalized = (inputs.float() * torch.rsqrt(variance + 1e-6)).to(dtype)
        expected = (normalized * weight * F.silu(gate.float())).to(dtype)
        primitive = GatedRMSNormPrimitive(inputs, gate, weight, output, epsilon=1e-6)
        primitive.launch(inputs, gate, weight, output)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            primitive.launch(inputs, gate, weight, output)
        graph.replay()
        torch.cuda.synchronize(device)
        self.assertTrue(torch.equal(output, expected))
        original_inputs, original_gate = inputs.clone(), gate.clone()
        for logical in (8, 32, 33):
            for poison in (37.0, -73.0):
                inputs.copy_(original_inputs)
                gate.copy_(original_gate)
                inputs[logical:].fill_(poison)
                gate[logical:].fill_(-poison)
                graph.replay()
                torch.cuda.synchronize(device)
                self.assertTrue(torch.equal(output[:logical], expected[:logical]))

    def test_packed_gdn_projection_split_is_exact_and_graph_composable(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            if not torch.cuda.is_available():
                self.skipTest("CUDA is required")
            device = torch.device("cuda:0")
            if torch.cuda.get_device_capability(device) != (12, 0):
                self.skipTest("this qualification is pinned to SM120")

            widths = 3, 5, 17
            torch.manual_seed(9449)
            packed = torch.randn((_C, sum(widths)), device=device).to(dtype)
            a = torch.empty((_C, widths[0]), device=device, dtype=dtype)
            b = torch.empty((_C, widths[1]), device=device, dtype=dtype)
            z = torch.empty((_C, widths[2]), device=device, dtype=dtype)
            primitive = BF16PackedGDNProjectionSplitPrimitive(packed, a, b, z)
            self.assertIn("a=3,b=5,z=17,packed-gdn", primitive.launch_signature)
            self.assertFalse(any(isinstance(value, Tensor) for value in vars(primitive).values()))
            stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.device(device), torch.cuda.stream(stream):
                primitive.launch(packed, a, b, z)
                primitive.launch(packed, a, b, z)
            stream.synchronize()
            self._tripwire_launch(device, stream, (primitive,), ((packed, a, b, z),))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                primitive.launch(packed, a, b, z)
            graph.replay()
            stream.synchronize()
            self.assertTrue(torch.equal(a, packed[:, : widths[0]]))
            self.assertTrue(torch.equal(b, packed[:, widths[0] : sum(widths[:2])]))
            self.assertTrue(torch.equal(z, packed[:, sum(widths[:2]) :]))

    def test_packed_swiglu_is_exact_and_graph_composable(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            if not torch.cuda.is_available():
                self.skipTest("CUDA is required")
            device = torch.device("cuda:0")
            if torch.cuda.get_device_capability(device) != (12, 0):
                self.skipTest("this qualification is pinned to SM120")

            torch.manual_seed(9017)
            packed = (torch.randn((_C, 2 * _INTERMEDIATE), device=device) * 1.1).to(dtype)
            output = torch.empty((_C, _INTERMEDIATE), device=device, dtype=dtype)
            primitive = BF16PackedSwiGLUPrimitive(packed, output)
            self.assertIn("packed-gate-up", primitive.launch_signature)
            self.assertFalse(any(isinstance(value, Tensor) for value in vars(primitive).values()))
            stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.device(device), torch.cuda.stream(stream):
                primitive.launch(packed, output)
                primitive.launch(packed, output)
            stream.synchronize()
            self._tripwire_launch(device, stream, (primitive,), ((packed, output),))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                primitive.launch(packed, output)
            graph.replay()
            stream.synchronize()
            gate, up = packed.chunk(2, dim=-1)
            expected = (F.silu(gate).to(dtype) * up).to(dtype)
            self.assertTrue(torch.equal(output, expected))

    def test_q_gate_split_is_exact_in_one_m4_graph(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            if not torch.cuda.is_available():
                self.skipTest("CUDA is required")
            device = torch.device("cuda:0")
            if torch.cuda.get_device_capability(device) != (12, 0):
                self.skipTest("this qualification is pinned to SM120")

            torch.manual_seed(9351)
            logical = torch.randn((_C, 24, 2, 256), device=device).to(dtype)
            projected = torch.empty_like(logical)
            query = torch.empty((_C, 24, 256), device=device, dtype=dtype)
            gate = torch.empty_like(query)
            primitive = QwenAttentionQGateSplitPrimitive(projected, query, gate)
            self.assertIn("qh=24,d=256", primitive.launch_signature)
            self.assertFalse(any(isinstance(value, Tensor) for value in vars(primitive).values()))
            addresses = _addresses((projected, query, gate))

            def fill(
                logical_m: int,
                poison: float,
                projected: Tensor = projected,
                logical: Tensor = logical,
            ) -> None:
                for physical_row, logical_row in enumerate(_ROWS[logical_m]):
                    if logical_row >= 0:
                        projected[physical_row].copy_(logical[logical_row])
                    else:
                        projected[physical_row].fill_(poison + physical_row / 8.0)

            fill(4, 37.0)
            stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.device(device), torch.cuda.stream(stream):
                primitive.launch(projected, query, gate)
                primitive.launch(projected, query, gate)
            stream.synchronize()
            self._tripwire_launch(device, stream, (primitive,), ((projected, query, gate),))

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                primitive.launch(projected, query, gate)
            stream.synchronize()

            by_m: dict[int, tuple[Tensor, Tensor]] = {}
            for logical_m in (1, 2, 4):
                poison_results: list[tuple[Tensor, Tensor]] = []
                for poison in (37.0, -73.0):
                    fill(logical_m, poison)
                    graph.replay()
                    torch.cuda.synchronize(device)
                    self.assertTrue(torch.equal(query, projected[:, :, 0]))
                    self.assertTrue(torch.equal(gate, projected[:, :, 1]))
                    rows = [_ROWS[logical_m].index(row) for row in range(logical_m)]
                    poison_results.append((query[rows].clone(), gate[rows].clone()))
                self.assertTrue(torch.equal(poison_results[0][0], poison_results[1][0]))
                self.assertTrue(torch.equal(poison_results[0][1], poison_results[1][1]))
                by_m[logical_m] = poison_results[0]
            for logical_m in (1, 2):
                self.assertTrue(torch.equal(by_m[logical_m][0], by_m[4][0][:logical_m]))
                self.assertTrue(torch.equal(by_m[logical_m][1], by_m[4][1][:logical_m]))
            self.assertEqual(addresses, _addresses((projected, query, gate)))

    def test_one_m4_graph_isolates_m1_m2_m4_for_both_norm_weight_dtypes(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        device = torch.device("cuda:0")
        if torch.cuda.get_device_capability(device) != (12, 0):
            self.skipTest("this qualification is pinned to SM120")

        for weight_dtype in (torch.bfloat16, torch.float32):
            with self.subTest(weight_dtype=weight_dtype):
                self._qualify_dtype(device, weight_dtype)

    def test_fp16_graph_and_independent_pointwise_boundaries(self) -> None:
        for weight_dtype in (torch.float16, torch.float32):
            self._qualify_dtype(torch.device("cuda:0"), weight_dtype, torch.float16)

    def _qualify_dtype(
        self,
        device: torch.device,
        weight_dtype: torch.dtype,
        activation_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        torch.manual_seed(9101 if weight_dtype is torch.bfloat16 else 9202)
        logical = {
            "residual": (torch.randn((_C, _HIDDEN), device=device) * 0.7).to(activation_dtype),
            "update": (torch.randn((_C, _HIDDEN), device=device) * 0.7).to(activation_dtype),
            "mlp_gate": (torch.randn((_C, _INTERMEDIATE), device=device) * 1.3).to(
                activation_dtype
            ),
            "mlp_up": (torch.randn((_C, _INTERMEDIATE), device=device) * 0.8).to(activation_dtype),
            "attention": (torch.randn((_C, _HIDDEN), device=device) * 0.7).to(activation_dtype),
            "attention_gate": (torch.randn((_C, _HIDDEN), device=device) * 1.3).to(
                activation_dtype
            ),
            "norm_inputs": (torch.randn((_C, _HEADS, _HEAD_DIM), device=device) * 0.7).to(
                activation_dtype
            ),
            "norm_gate": (torch.randn((_C, _HEADS, _HEAD_DIM), device=device) * 1.3).to(
                activation_dtype
            ),
        }
        weight = (torch.randn((_HEAD_DIM,), device=device) * 0.2 + 1.0).to(weight_dtype)
        physical = {name: torch.empty_like(tensor) for name, tensor in logical.items()}
        outputs = {
            "residual": torch.empty((_C, _HIDDEN), device=device, dtype=activation_dtype),
            "swiglu": torch.empty((_C, _INTERMEDIATE), device=device, dtype=activation_dtype),
            "sigmoid_gate": torch.empty((_C, _HIDDEN), device=device, dtype=activation_dtype),
            "gated_rms_norm": torch.empty(
                (_C, _HEADS, _HEAD_DIM), device=device, dtype=activation_dtype
            ),
        }
        primitives = (
            BF16ResidualAddPrimitive(physical["residual"], physical["update"], outputs["residual"]),
            BF16SwiGLUPrimitive(physical["mlp_gate"], physical["mlp_up"], outputs["swiglu"]),
            BF16SigmoidOutputGatePrimitive(
                physical["attention"],
                physical["attention_gate"],
                outputs["sigmoid_gate"],
            ),
            GatedRMSNormPrimitive(
                physical["norm_inputs"],
                physical["norm_gate"],
                weight,
                outputs["gated_rms_norm"],
                epsilon=1e-6,
            ),
        )
        expected_weight = (
            "weight="
            + {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[weight_dtype]
        )
        self.assertIn(expected_weight, primitives[-1].launch_signature)
        for primitive in primitives:
            self.assertFalse(any(isinstance(value, Tensor) for value in vars(primitive).values()))

        tensors = (*physical.values(), weight, *outputs.values())
        addresses = _addresses(tensors)
        operands = (
            (physical["residual"], physical["update"], outputs["residual"]),
            (physical["mlp_gate"], physical["mlp_up"], outputs["swiglu"]),
            (physical["attention"], physical["attention_gate"], outputs["sigmoid_gate"]),
            (
                physical["norm_inputs"],
                physical["norm_gate"],
                weight,
                outputs["gated_rms_norm"],
            ),
        )
        stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.device(device), torch.cuda.stream(stream):
            for _ in range(2):
                for primitive, arguments in zip(primitives, operands, strict=True):
                    primitive.launch(*arguments)
        stream.synchronize()
        self._tripwire_launch(device, stream, primitives, operands)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for primitive, arguments in zip(primitives, operands, strict=True):
                primitive.launch(*arguments)
        stream.synchronize()

        by_m: dict[int, dict[str, Tensor]] = {}
        for logical_m in (1, 2, 4):
            poison_results: list[dict[str, Tensor]] = []
            for poison in (37.0, -73.0):
                self._fill(logical_m, poison, logical, physical)
                graph.replay()
                torch.cuda.synchronize(device)
                gather = torch.tensor(
                    [_ROWS[logical_m].index(row) for row in range(logical_m)], device=device
                )
                actual = {name: tensor[gather].clone() for name, tensor in outputs.items()}
                expected = _oracle(
                    logical["residual"][:logical_m],
                    logical["update"][:logical_m],
                    logical["mlp_gate"][:logical_m],
                    logical["mlp_up"][:logical_m],
                    logical["attention"][:logical_m],
                    logical["attention_gate"][:logical_m],
                    logical["norm_inputs"][:logical_m],
                    logical["norm_gate"][:logical_m],
                    weight,
                )
                for name in actual:
                    self.assertTrue(
                        torch.equal(actual[name], expected[name]),
                        f"non-exact {name} for logical M={logical_m}, {weight_dtype}",
                    )
                poison_results.append(actual)
            for name in outputs:
                self.assertTrue(
                    torch.equal(poison_results[0][name], poison_results[1][name]),
                    f"padding poison reached active {name} for logical M={logical_m}",
                )
            by_m[logical_m] = poison_results[0]

        for logical_m in (1, 2):
            for name in outputs:
                self.assertTrue(torch.equal(by_m[logical_m][name], by_m[4][name][:logical_m]))
        self.assertEqual(addresses, _addresses(tensors))
        self._assert_scalar_nonlinear_boundaries(device, weight_dtype, activation_dtype)

    def _fill(
        self,
        logical_m: int,
        poison: float,
        logical: dict[str, Tensor],
        physical: dict[str, Tensor],
    ) -> None:
        for physical_row, logical_row in enumerate(_ROWS[logical_m]):
            for name, target in physical.items():
                if logical_row >= 0:
                    target[physical_row].copy_(logical[name][logical_row])
                else:
                    target[physical_row].fill_(poison + physical_row / 8.0)

    def _assert_scalar_nonlinear_boundaries(
        self, device: torch.device, weight_dtype: torch.dtype, activation_dtype: torch.dtype
    ) -> None:
        """Reduction length one leaves no REDUCTION_ORDER waiver for nonlinear math."""

        gate = torch.linspace(-8.0, 8.0, _C, device=device).reshape(_C, 1).to(activation_dtype)
        up = torch.linspace(0.25, 1.0, _C, device=device).reshape(_C, 1).to(activation_dtype)
        output = torch.empty_like(gate)
        BF16SwiGLUPrimitive(gate, up, output).launch(gate, up, output)
        torch.cuda.synchronize(device)
        self.assertTrue(
            torch.equal(output, (F.silu(gate).to(activation_dtype) * up).to(activation_dtype))
        )

        BF16SigmoidOutputGatePrimitive(up, gate, output).launch(up, gate, output)
        torch.cuda.synchronize(device)
        self.assertTrue(
            torch.equal(
                output, (up * torch.sigmoid(gate).to(activation_dtype)).to(activation_dtype)
            )
        )

        inputs = up.reshape(_C, 1, 1)
        norm_gate = gate.reshape(_C, 1, 1)
        weight = torch.tensor([0.875], device=device, dtype=weight_dtype)
        norm_output = torch.empty_like(inputs)
        primitive = GatedRMSNormPrimitive(inputs, norm_gate, weight, norm_output, epsilon=1e-6)
        primitive.launch(inputs, norm_gate, weight, norm_output)
        torch.cuda.synchronize(device)
        expected = _oracle(
            up,
            up,
            gate,
            up,
            up,
            gate,
            inputs,
            norm_gate,
            weight,
        )["gated_rms_norm"]
        self.assertTrue(torch.equal(norm_output, expected))

    def _tripwire_launch(
        self,
        device: torch.device,
        stream: torch.cuda.Stream,
        primitives: tuple[object, ...],
        operands: tuple[tuple[Tensor, ...], ...],
    ) -> None:
        def forbidden(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise AssertionError("primitive launch allocated, cloned, synchronized, or called item")

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
                        "compile",
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
                (type(implementation._q_gate_split_kernel), ("_do_compile",)),
            ):
                for name in names:
                    stack.enter_context(mock.patch.object(owner, name, forbidden))
            with torch.cuda.device(device), torch.cuda.stream(stream):
                for primitive, arguments in zip(primitives, operands, strict=True):
                    primitive.launch(*arguments)  # type: ignore[attr-defined]
        stream.synchronize()


@unittest.skipUnless(RUN_REAL and torch.cuda.is_available(), "requires CUDA opt-in")
class ResidentPageUpdateTests(unittest.TestCase):
    def test_explicit_activity_preserves_inactive_records_and_typed_payloads(self) -> None:
        for capacity in (1, 4, 17):
            pages = capacity + 3
            key = torch.full((pages, 2, 7), 3, dtype=torch.bfloat16, device="cuda")
            value = key.clone()
            written = torch.ones((pages, 2), dtype=torch.bool, device="cuda")
            metadata = [torch.full((capacity,), 19, dtype=torch.int32, device="cuda")
                        for _ in range(3)]
            table = torch.full((capacity, 3), -77, dtype=torch.int32, device="cuda")
            fields = tuple(torch.full((pages, 2, 5), initial, dtype=dtype, device="cuda")
                           for dtype, initial in ((torch.int64, 2**54 + 1),
                                                  (torch.float32, 9), (torch.bool, True)))
            tensors = (key, value, written, *metadata, table)
            original = tuple(t.cpu() for t in (*tensors, *fields))
            for active_count in sorted({0, 1, capacity}):
                updates_cpu = torch.tensor([[i, i % 3, i + 1] for i in range(capacity)],
                                           dtype=torch.int32)
                activity_cpu = torch.arange(capacity) < active_count
                # Inactive indices may be poisoned or valid; neither defines activity.
                updates_cpu[~activity_cpu] = -1234567
                if active_count == 0:
                    updates_cpu[0] = torch.tensor([0, 0, 0])
                updates = updates_cpu.cuda()
                activity = activity_cpu.cuda()
                implementation.update_resident_pages(
                    updates, *tensors, page_fields=fields,
                    update_activity=activity, compile_only=True,
                )
                for actual, before in zip((*tensors, *fields), original, strict=True):
                    self.assertTrue(torch.equal(actual.cpu(), before))
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    implementation.update_resident_pages(
                        updates, *tensors, page_fields=fields, update_activity=activity,
                    )
                for permutation in (torch.arange(capacity), torch.arange(capacity - 1, -1, -1)):
                    for actual, before in zip((*tensors, *fields), original, strict=True):
                        actual.copy_(before)
                    updates.copy_(updates_cpu[permutation])
                    activity.copy_(activity_cpu[permutation])
                    graph.replay()
                    expected = [t.clone() for t in original]
                    k, v, w, o, s, d, mapping, *aux = expected
                    for row, column, page in updates_cpu[activity_cpu].tolist():
                        k[page] = 0; v[page] = 0; w[page] = False
                        o[row] = 0; s[row] = page; d[row] = page; mapping[row, column] = page
                        for payload in aux:
                            payload[page] = 0
                    for actual, before in zip((*tensors, *fields), expected, strict=True):
                        self.assertTrue(torch.equal(actual.cpu(), before))
                for actual, before in zip((*tensors, *fields), original, strict=True):
                    actual.copy_(before)

    def test_independent_stores_padding_permutation_and_compile_only(self) -> None:
        torch.manual_seed(811)
        for dtype in (torch.float16, torch.bfloat16):
            for m in (1, 4, 8, 17, 128, 256):
                for shape in ((3, 7, 13), (2, 16, 4, 128)):
                    pages = 2 * m + 5
                    key = torch.randn((pages, *shape), dtype=dtype, device="cuda")
                    value = torch.randn_like(key)
                    written = torch.ones((pages, 3, 7), dtype=torch.bool, device="cuda")
                    metadata = [torch.full((m,), 37, dtype=torch.int32, device="cuda")
                                for _ in range(3)]
                    table = torch.full((m, 9), -77, dtype=torch.int32, device="cuda")
                    actual = (key, value, written, *metadata, table)
                    expected = [x.cpu() for x in actual]
                    for n in sorted({0, 1, m // 2, m}):
                        physical = torch.randperm(m)[:n]
                        page = torch.randperm(pages)[:n]
                        column = torch.randint(0, 9, (n,))
                        updates_cpu = torch.stack((physical, column, page), dim=1).int()
                        updates = updates_cpu.to("cuda")
                        implementation.update_resident_pages(updates, *actual, compile_only=True)
                        for a, e in zip(actual, expected, strict=True):
                            self.assertTrue(torch.equal(a.cpu(), e), "compile mutated storage")
                        # CPU scalar stores are independent of the CUDA indexing implementation.
                        k, v, w, o, s, d, t = expected
                        for row, col, slot in updates_cpu.tolist():
                            k[slot] = 0
                            v[slot] = 0
                            w[slot] = False
                            o[row] = 0
                            s[row] = slot
                            d[row] = slot
                            t[row, col] = slot
                        implementation.update_resident_pages(updates, *actual)
                        for a, e in zip(actual, expected, strict=True):
                            self.assertTrue(torch.equal(a.cpu(), e), (dtype, m, shape, n))
                        # Disjoint owned writes commute, including across repeated generations.
                        permutation = torch.randperm(n, device="cuda")
                        implementation.update_resident_pages(updates[permutation], *actual)
                        for a, e in zip(actual, expected, strict=True):
                            self.assertTrue(torch.equal(a.cpu(), e), "permutation changed state")

    def test_typed_auxiliary_fields_share_page_reset_and_capture(self) -> None:
        for capacity in (1, 4):
            pages = 2 * capacity + 3
            key = torch.full((pages, 2, 4, 1, 8), 7, dtype=torch.bfloat16, device="cuda")
            value = key.clone()
            written = torch.ones((pages, 2, 4), dtype=torch.bool, device="cuda")
            indices = [
                torch.full((capacity,), 19, dtype=torch.int32, device="cuda") for _ in range(3)
            ]
            table = torch.full((capacity, 3), -1, dtype=torch.int32, device="cuda")
            fields = tuple(
                torch.full((pages, 2, 13), initial, dtype=dtype, device="cuda")
                for dtype, initial in ((torch.bfloat16, 5), (torch.float32, 9),
                                       (torch.int64, 2**54 + 1), (torch.bool, True))
            )
            updates_cpu = torch.tensor(
                [[i, i % 3, i + capacity] for i in range(capacity)], dtype=torch.int32
            )
            updates = updates_cpu.cuda()
            tensors = (key, value, written, *indices, table)
            snapshots = tuple(t.cpu() for t in (*tensors, *fields))
            implementation.update_resident_pages(
                updates, *tensors, page_fields=fields, compile_only=True
            )
            for actual, expected in zip((*tensors, *fields), snapshots, strict=True):
                self.assertTrue(torch.equal(actual.cpu(), expected))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                implementation.update_resident_pages(updates, *tensors, page_fields=fields)
            for actual, expected in zip((*tensors, *fields), snapshots, strict=True):
                actual.copy_(expected)
            graph.replay()
            for field, before in zip(fields, snapshots[-len(fields):], strict=True):
                expected = before.clone()
                expected[updates_cpu[:, 2].long()] = 0
                self.assertTrue(torch.equal(field.cpu(), expected))
            for row, column, page in updates_cpu.tolist():
                self.assertEqual(int(table[row, column]), page)
                self.assertTrue((key[page] == 0).all())
            updates.copy_(updates.flip(0))
            graph.replay()
            for field, before in zip(fields, snapshots[-len(fields):], strict=True):
                expected = before.clone()
                expected[updates_cpu[:, 2].long()] = 0
                self.assertTrue(torch.equal(field.cpu(), expected))


@unittest.skipUnless(RUN_REAL and torch.cuda.is_available(), "requires CUDA opt-in")
class ResidentInputUpdateTests(unittest.TestCase):
    def test_captured_update_and_startup_restore(self) -> None:
        from types import SimpleNamespace as NS

        from cubic_serve.runtime.performance import _CapturedGraphRank

        torch.manual_seed(318)
        for m in (1, 2, 4, 8, 17, 128, 256):
            for dtype in (torch.bfloat16, torch.float16):
                for first in (False, True):
                    with self.subTest(m=m, dtype=dtype, first=first):
                        positions = torch.randint(0, 30000, (3, m), device="cuda")
                        offsets = torch.randint(0, 100, (m,), device="cuda", dtype=torch.int32)
                        lengths = offsets.clone() + 100
                        tokens = torch.randint(-1000, 5000, (m,), device="cuda")
                        ids = torch.full_like(tokens, -9)
                        mask = torch.full((m, 1), -7, device="cuda", dtype=dtype)
                        ordinals = torch.full_like(tokens, 731)
                        activity = torch.randint(0, 2, (m,), device="cuda", dtype=torch.int32)
                        buffers = NS(position_ids=positions, token_offsets=offsets,
                                     sequence_lengths=lengths, token_ids=ids,
                                     row_activity=activity)
                        owner = NS(body=NS(buffers=buffers, stage=0 if first else 1,
                                          embedding_span=range(1000, 2000),
                                          embedding_mask=mask, hidden=mask),
                                   _sampling=NS(ordinals=ordinals), _token_ids=tokens,
                                   slot=NS(stream=torch.cuda.Stream()))
                        actual = [positions, offsets, lengths, ids, mask, ordinals]
                        expected = [x.clone() for x in actual]
                        saved_tokens, saved_activity = tokens.clone(), activity.clone()
                        graph = _CapturedGraphRank._capture_input_update(owner)
                        for a, b in zip(actual, expected, strict=True):
                            self.assertTrue(torch.equal(a, b), "startup changed inputs")
                        self.assertTrue(torch.equal(tokens, saved_tokens))
                        self.assertTrue(torch.equal(activity, saved_activity))
                        for n in sorted({0, 1, max(1, m // 2), m}):
                            for step in range(3):
                                tokens.add_(107)
                                selected = torch.randperm(m, device="cuda")[:n]
                                activity.zero_()
                                activity[selected] = 1
                                active = activity.bool()
                                graph.replay()
                                if step == 1 and n:
                                    offsets[selected[0]] = 0
                                p, o, length_ref, i, mask_ref, ordinal_ref = expected
                                p[:, active] += 1
                                o[active] += 1
                                length_ref[active] += 1
                                ordinal_ref[active] += 1
                                if step == 1 and n:
                                    o[selected[0]] = 0
                                if first:
                                    owned = (tokens[active] >= 1000) & (tokens[active] < 2000)
                                    i[active] = torch.where(owned, tokens[active] - 1000, 0)
                                    mask_ref[active] = owned[:, None].to(dtype)
                                for a, b in zip(actual, expected, strict=True):
                                    self.assertTrue(torch.equal(a, b), (m, first, n, step))


if __name__ == "__main__":
    unittest.main()


def test_sigmoid_gated_norm_reference_graph_padding_and_cross_m():
    import pytest

    from cubic_serve.executor.runner import ReferenceExecutionContext
    from cubic_serve.ops.eager.hybrid import gated_rms_norm

    if not RUN_REAL:
        pytest.skip("explicit CUDA pointwise audit required")
    for policy in ("fp32", "auto"):
        with operator_strategy_scope(OperatorConfig(activation_dtype=policy), device="", model=""):
            for dtype in (torch.float16, torch.bfloat16):
                width = 128
                indices = torch.arange(33 * width, device="cuda").reshape(33, 1, width)
                inputs = torch.where(indices % 3 == 0, -1.0, 1.0).to(dtype)
                gates = ((indices % 5) - 2).to(dtype)
                for weight_dtype in (dtype, torch.float32):
                    weight = ((torch.arange(width, device="cuda") % 4) / 4).to(weight_dtype)
                    with ReferenceExecutionContext():
                        expected = gated_rms_norm(
                            inputs,
                            gates,
                            weight,
                            epsilon=3.0,
                            output_dtype=dtype,
                            activation="sigmoid",
                        )
                    for capacity, logical in ((1, 1), (8, 8), (64, 33)):
                        x = torch.empty(capacity, 1, width, dtype=dtype, device="cuda")
                        gate, out = torch.empty_like(x), torch.empty_like(x)
                        primitive = GatedRMSNormPrimitive(
                            x, gate, weight, out, epsilon=3.0, activation="sigmoid"
                        )
                        primitive.launch(x, gate, weight, out)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            primitive.launch(x, gate, weight, out)
                        for reverse in (False, True):
                            x.fill_(float("nan"))
                            gate.fill_(float("nan"))
                            x[:logical].copy_(
                                inputs[:logical].flip(0) if reverse else inputs[:logical]
                            )
                            gate[:logical].copy_(
                                gates[:logical].flip(0) if reverse else gates[:logical]
                            )
                            graph.replay()
                            actual = out[:logical].flip(0) if reverse else out[:logical]
                            assert torch.equal(actual, expected[:logical])
