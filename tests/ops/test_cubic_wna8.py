"""Independent numerical, instruction and cross-M checks for Cubic INT8 MMA."""

import os
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from cubic_serve.executor.runner import configure_reference_math
from cubic_serve.guardrail.quantization import check_linear_enclosure
from cubic_serve.models.quantization import QuantizationScheme, WeightFormat
from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
from cubic_serve.ops.strategies import OperatorConfig


def matrix(bits, group=32, metadata="float32-scale-float16-ab", k=256, n=64, group_out=1):
    shape = n // group_out, (k + group - 1) // group
    p = {"weight_packed": torch.randint(0, 256, (n, (k * bits + 7) // 8), dtype=torch.uint8)}
    if metadata == "float32-scale-float16-ab":
        p.update(
            weight_scale=torch.rand(shape) * 0.2,
            weight_a=torch.full(shape, 0.75, dtype=torch.float16),
            weight_b=torch.full(shape, 0.125, dtype=torch.float16),
        )
    elif metadata == "int8-scale-int4-ab-global-fp32":
        p.update(
            weight_scale=torch.randint(0, 127, shape, dtype=torch.int8),
            weight_ab=torch.randint(0, 256, shape, dtype=torch.uint8),
            weight_scale_global=torch.tensor([0.002]),
            weight_a_global=torch.tensor([0.0625]),
            weight_b_global=torch.tensor([0.0625]),
        )
    else:
        s = torch.full(shape, 0.125, dtype=torch.float16).view(torch.int16).int() >> 1
        p.update(
            weight_metadata=(s | (torch.randint(0, 4, shape) << 14)).to(torch.uint16),
            weight_curve_a=torch.tensor([[0.75, 0.875, 1.0, 1.125]], dtype=torch.float16),
            weight_curve_b=torch.tensor([[0.125, 0.25, 0.0, -0.125]], dtype=torch.float16),
        )
    return QuantizedMatrix(
        p, QuantizationScheme(WeightFormat.CUBIC, bits, group, group_out, metadata), (n, k)
    )


class NativeAvailabilityTests(unittest.TestCase):
    def test_unavailable_native_mode_fails_before_operand_preparation(self):
        from cubic_serve.quantization.extensions.cubic import linear

        operators = SimpleNamespace(config=SimpleNamespace(compute_mode="wna8"))
        with self.assertRaisesRegex(ValueError, "WNA8 native linear kernel is unavailable"):
            linear.bind_native(operators, None, None, None, None)
        with self.assertRaisesRegex(ValueError, "WNA8 native grouped linear kernel is unavailable"):
            linear.bind_grouped(operators, None, None, None, None)

    def test_wna16_keeps_existing_native_bindings(self):
        from cubic_serve.quantization.extensions.cubic import linear
        from cubic_serve.quantization.extensions.cubic.policy import CubicLinearParameters

        operators = SimpleNamespace(config=SimpleNamespace(compute_mode="wna16"))
        params, result = CubicLinearParameters(), object()
        with patch(
            "cubic_serve.quantization.extensions.cubic.marlin.bind_planned_linear",
            return_value=result,
        ) as single:
            self.assertIs(linear.bind_native(operators, None, None, None, params), result)
            single.assert_called_once_with(operators, None, None, None, params)
        with patch(
            "cubic_serve.quantization.extensions.cubic.marlin.bind_planned_grouped",
            return_value=result,
        ) as grouped:
            self.assertIs(linear.bind_grouped(operators, None, None, None, params), result)
            grouped.assert_called_once_with(operators, None, None, None, params)


@unittest.skipUnless(os.environ.get("CUBIC_TEST_QUANT_CUDA") == "1", "explicit CUDA audit required")
class WNA8Tests(unittest.TestCase):
    """Legacy compatibility arithmetic only; never performance qualification."""

    def setUp(self):
        from cubic_serve.quantization.extensions.cubic import wna8

        # Keep every existing numerical/ownership fixture on the explicit legacy
        # compatibility implementation. Production native hooks must reject it.
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch(
            "cubic_serve.quantization.extensions.cubic.linear.bind_native",
            side_effect=lambda op, x, w, y, p, fp8_token_scales=None:
                wna8.bind_linear(op, x, w, y, p),
        ))
        stack.enter_context(patch(
            "cubic_serve.quantization.extensions.cubic.linear.bind_grouped",
            side_effect=wna8.bind_pair,
        ))

    def test_pair_unequal_k_partitions_and_single_row_identity(self):
        from cubic_serve.ops.cuda.linear.quantized import PackedProjectionDispatch

        configure_reference_math()
        op = PackedLinearOperators(OperatorConfig(compute_mode="wna8"))
        weights = (matrix(4, n=64, k=1024), matrix(4, n=4096, k=1024))
        # Larger than the paired members even at the maximum eight K partitions.
        large = matrix(4, n=8 * (64 + 4096) + 64, k=1024)
        row = torch.randn(1, 1024, device="cuda:0", dtype=torch.bfloat16)
        common = None
        for capacity in (1, 4, 8, 64, 129, 256):
            x = row.expand(capacity, -1).contiguous()
            outputs = tuple(
                torch.empty(capacity, w.shape[0], device=x.device, dtype=x.dtype) for w in weights
            )
            dispatch = PackedProjectionDispatch(op)
            dispatch.prepare_group(x, weights, outputs)
            # Grow the same owner's reusable pool after binding the pair.
            extra = torch.empty(capacity, large.shape[0], device=x.device, dtype=x.dtype)
            dispatch.prepare(x, large, extra)
            dispatch.seal()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                dispatch.launch(x, large, extra)
                for w, y in zip(weights, outputs, strict=True):
                    dispatch.launch(x, w, y)
                dispatch.launch(x, large, extra)
            graph.replay()
            torch.cuda.synchronize()
            for w, y in zip(weights, outputs, strict=True):
                check_linear_enclosure(x, w, y, compute_mode="wna8")
                self.assertTrue(torch.equal(y, y[:1].expand_as(y)))
            if common is not None:
                self.assertTrue(
                    all(torch.equal(y[:1], c) for y, c in zip(outputs, common, strict=True))
                )
            common = tuple(y[:1].clone() for y in outputs)

    def test_paired_projection_format_padding_and_private_pool(self):
        import triton

        from cubic_serve.ops.cuda.linear.quantized import PackedProjectionDispatch
        from cubic_serve.quantization.common.int8 import Int8Activation
        from cubic_serve.quantization.extensions.cubic.wna8 import _quantize

        configure_reference_math()
        for dtype in (torch.float16, torch.bfloat16):
            for bits in range(2, 9):
                corpus = torch.randn(17, 259, device="cuda:0", dtype=dtype)
                weights = (matrix(bits, k=259, n=137), matrix(bits, k=259, n=71))
                common = None
                owners = []
                for capacity in (17, 32, 128, 256):
                    op = PackedLinearOperators(OperatorConfig(compute_mode="wna8"))
                    x = torch.full((capacity, 259), float("nan"), device="cuda:0", dtype=dtype)
                    permutation = torch.randperm(capacity, device=x.device)[:17]
                    x[permutation] = corpus
                    # Alternate floating and registered resident INT8 boundaries.
                    if capacity in (32, 256):
                        codes = torch.empty_like(x, dtype=torch.int8)
                        scales = torch.empty(capacity, device=x.device)
                        _quantize[(capacity,)](
                            x,
                            codes,
                            scales,
                            259,
                            triton.next_power_of_2(259),
                            num_warps=4,
                            enable_fp_fusion=False,
                        )
                        op.register_int8_input(Int8Activation(codes, scales, dtype))
                        operand = codes
                    else:
                        operand = x
                    outputs = tuple(
                        torch.empty(capacity, w.shape[0], device=x.device, dtype=dtype)
                        for w in weights
                    )
                    expected = tuple(torch.empty_like(y) for y in outputs)
                    separate = tuple(
                        op.bind_linear(operand, w, y)
                        for w, y in zip(weights, expected, strict=True)
                    )
                    dispatch = PackedProjectionDispatch(op)
                    dispatch.prepare_group(operand, weights, outputs)
                    dispatch.seal()
                    bindings = [b for _, b in dispatch._entries.values()]
                    self.assertEqual(
                        len(bindings[0]._launches), 2 if operand.dtype == torch.int8 else 3
                    )
                    self.assertEqual(len(bindings[1]._launches), 0)
                    owners.append(dispatch)
                    pointers = [
                        d._scratch_pools[next(iter(d._scratch_pools))].data_ptr() for d in owners
                    ]
                    self.assertEqual(len(pointers), len(set(pointers)))
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for w, y in zip(weights, outputs, strict=True):
                            dispatch.launch(operand, w, y)
                    graph.replay()
                    for binding in separate:
                        binding.launch()
                    torch.cuda.synchronize()
                    actual = tuple(y[permutation].clone() for y in outputs)
                    for w, y, ref in zip(weights, actual, expected, strict=True):
                        self.assertTrue(torch.equal(y, ref[permutation]))
                        check_linear_enclosure(corpus, w, y, compute_mode="wna8")
                    if common is not None:
                        self.assertTrue(
                            all(torch.equal(a, b) for a, b in zip(actual, common, strict=True))
                        )
                    common = actual

    def test_dispatch_partial_reuse_preserves_outputs_and_lane_ownership(self):
        from cubic_serve.ops.cuda.linear.quantized import PackedProjectionDispatch

        configure_reference_math()
        for dtype in (torch.float16, torch.bfloat16):
            op = PackedLinearOperators(OperatorConfig(compute_mode="wna8"))
            owners = []
            for _ in range(2):
                dispatch = PackedProjectionDispatch(op)
                entries = []
                for bits, n in ((3, 64), (4, 256), (6, 96)):
                    x = torch.randn(17, 256, device="cuda:0", dtype=dtype)
                    weight = matrix(bits, n=n)
                    output = torch.empty(17, n, device=x.device, dtype=dtype)
                    dispatch.prepare(x, weight, output)
                    entries.append((x, weight, output))
                dispatch.seal()
                bindings = [binding for _, binding in dispatch._entries.values()]
                pointers = {b.scratch[0].untyped_storage().data_ptr() for b in bindings}
                self.assertEqual(len(pointers), 1)
                self.assertTrue(all(b.reusable_scratch for b in bindings))
                self.assertTrue(all(b.scratch[0].dtype == torch.int32 for b in bindings))
                owners.append((dispatch, entries, pointers))
            self.assertTrue(owners[0][2].isdisjoint(owners[1][2]))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                # Alternate independent owners and differently sized partials.
                for index in (2, 0, 1):
                    for dispatch, entries, _ in owners:
                        dispatch.launch(*entries[index])
            for _ in range(2):
                for _, entries, _ in owners:
                    for x, _, _ in entries:
                        x.normal_()
                graph.replay()
                torch.cuda.synchronize()
                for _, entries, _ in owners:
                    for x, weight, output in entries:
                        check_linear_enclosure(x, weight, output, compute_mode="wna8")

    def test_subnormal_activation_does_not_overflow_reciprocal(self):
        from cubic_serve.quantization.extensions.cubic.reference import wna8_activation
        from cubic_serve.quantization.extensions.cubic.wna8 import _quantize

        x = torch.tensor(
            [[0, 1e-39, -1e-39, 2e-39], [0, 1e-30, -1e-30, 2e-30]],
            device="cuda:0",
            dtype=torch.bfloat16,
        )
        q = torch.empty_like(x, dtype=torch.int8)
        scales = torch.empty(2, device=x.device)
        _quantize[(2,)](x, q, scales, 4, 4, num_warps=4, enable_fp_fusion=False)
        actual = q.float() * scales[:, None]
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertTrue(torch.equal(actual, wna8_activation(x, matrix(4).encoding)))

    def test_output_groups_tile_tails_and_zero_activation(self):
        configure_reference_math()
        for bits in range(2, 9):
            w = matrix(bits, group=64, k=173, n=66, group_out=2)
            op = PackedLinearOperators(OperatorConfig(compute_mode="wna8"))
            corpus = torch.randn(17, 173, device="cuda:0", dtype=torch.float16)
            corpus[0].zero_()
            expected = None
            for capacity in (17, 32, 64):
                x = torch.full(
                    (capacity, 173), float("nan"), device=corpus.device, dtype=corpus.dtype
                )
                x[:17].copy_(corpus)
                y = op.linear(x, w)[:17]
                check_linear_enclosure(corpus, w, y, compute_mode="wna8")
                self.assertEqual(torch.count_nonzero(y[0]).item(), 0)
                if expected is None:
                    expected = y.clone()
                else:
                    self.assertTrue(torch.equal(y, expected))

    def test_all_bits_dtypes_metadata_graph_and_row_identity(self):
        configure_reference_math()
        torch.set_num_threads(2)
        torch.manual_seed(417)
        for bits in range(2, 9):
            for dtype in (torch.float16, torch.bfloat16):
                for metadata in (
                    "float32-scale-float16-ab",
                    "int8-scale-int4-ab-global-fp32",
                    "e5m9-scale-curve2-tensor-fp16",
                ):
                    with self.subTest(bits=bits, dtype=dtype, metadata=metadata):
                        w = matrix(bits, metadata=metadata)
                        op = PackedLinearOperators(OperatorConfig(compute_mode="wna8"))
                        corpus = torch.randn(33, 256, device="cuda:0", dtype=dtype) * 0.3
                        expected = None
                        for cap, logical in ((64, 33), (1, 1), (16, 9), (128, 33)):
                            x = torch.empty(cap, 256, device="cuda:0", dtype=dtype)
                            x.fill_(float("nan"))
                            x[:logical].copy_(corpus[:logical])
                            y = torch.empty(cap, 64, device=x.device, dtype=dtype)
                            binding = op.bind_linear(x, w, y)
                            binding.launch()
                            torch.cuda.synchronize()
                            ptx = binding._launches[1].kernel.asm["ptx"]
                            self.assertIn("mma.sync.aligned.m16n8k32", ptx)
                            self.assertIn(".s32.s8.s8.s32", ptx)
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                binding.launch()
                            graph.replay()
                            torch.cuda.synchronize()
                            check_linear_enclosure(x[:logical], w, y[:logical], compute_mode="wna8")
                            if expected is None:
                                expected = y[:logical].clone()
                            else:
                                self.assertTrue(torch.equal(y[:logical], expected[:logical]))
                            x[:logical].copy_(corpus[:logical].flip(0))
                            graph.replay()
                            self.assertTrue(torch.equal(y[:logical].flip(0), expected[:logical]))

    def test_group_geometry_and_cache_invalidation(self):
        configure_reference_math()
        op = PackedLinearOperators(OperatorConfig(compute_mode="wna8"))
        for group in (32, 64, 128, 512):
            w = matrix(3, group=group, k=512)
            x = torch.randn(17, 512, device="cuda:0", dtype=torch.bfloat16)
            y = op.linear(x, w)
            check_linear_enclosure(x, w, y, compute_mode="wna8")
            before = len(op.prepared_weights)
            w.payload["weight_scale"].mul_(2)
            z = op.linear(x, w)
            self.assertEqual(len(op.prepared_weights), before + 2)
            check_linear_enclosure(x, w, z, compute_mode="wna8")
            self.assertFalse(torch.equal(y, z))
