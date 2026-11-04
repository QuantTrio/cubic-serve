import os
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
import triton
import triton.language as tl

from cubic_serve.executor.runner import ReferenceExecutionContext, configure_reference_math
from cubic_serve.guardrail.quantization import check_linear_enclosure
from cubic_serve.models.quantization import ActivationFormat, QuantizationScheme, WeightFormat
from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.ops.cuda.linear.quantized import (
    PackedLinearOperators,
    PackedProjectionDispatch,
    _fp8_quant,
    _nvfp4_quant,
)
from cubic_serve.ops.eager.quantization import decode_matrix, embedding_matrix, quantize_activation
from cubic_serve.ops.strategies import OperatorConfig
from cubic_serve.quantization.extensions.cubic.kernels import load_level_words


@triton.jit
def _decode_cubic_level_words(
    W,
    S,
    Z,
    A,
    B,
    GS,
    GA,
    GB,
    PART,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    GI: tl.constexpr,
    GO: tl.constexpr,
):
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    col = tl.program_id(1) * 64 + tl.arange(0, 64)
    value = load_level_words(
        W, S, Z, A, B, GS, GA, GB, PART, row[:, None], col[None, :], N, K, 1, 4, GI, GO, K // 2
    )
    tl.store(Y + row[:, None] * K + col[None, :], value, (row[:, None] < N) & (col[None, :] < K))


@unittest.skipUnless(os.environ.get("CUBIC_TEST_QUANT_CUDA") == "1", "explicit CUDA audit required")
class NativeQuantizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_reference_math()
        torch.set_num_threads(4)

    def test_sealed_eager_plain_linear_does_not_capture_new_rows(self):
        from cubic_serve.ops.cuda.launch_cost import LinearLaunchCost

        x = torch.full((2808, 1152), 0.125, device="cuda:0", dtype=torch.bfloat16)
        weight = torch.ones((256, 1152), device=x.device, dtype=x.dtype)
        operators = PackedLinearOperators(OperatorConfig(compute_mode="wna8"))
        operators.seal_dynamic_calibration()
        with patch.object(LinearLaunchCost, "select",
                          side_effect=AssertionError("request-time CUDA Graph capture")):
            output = operators.linear(x, weight)
        self.assertTrue(torch.equal(output, torch.full_like(output, 144)))

    def test_plain_selected_tiles_cross_m_and_poison(self):
        generator = torch.Generator(device="cuda:0").manual_seed(173)
        for dtype in (torch.bfloat16, torch.float16):
            for n, k in ((10240, 320), (1031, 321)):
                weight = torch.randn(n, k, device="cuda:0", dtype=dtype,
                                     generator=generator) * 0.01
                corpus = torch.randn(3, k, device="cuda:0", dtype=dtype,
                                     generator=generator)
                first = torch.empty(1, n, device="cuda:0", dtype=dtype)
                PackedLinearOperators().bind_linear(corpus[:1], weight, first).launch()
                common = None
                for m in (4, 32, 256):
                    x = torch.full((m, k), float("nan"), device="cuda:0", dtype=dtype)
                    indices = torch.tensor((m - 1, 0, m // 2), device=x.device)
                    x.index_copy_(0, indices, corpus)
                    output = torch.empty(m, n, device=x.device, dtype=dtype)
                    rng = torch.cuda.get_rng_state().clone()
                    binding = PackedLinearOperators().bind_linear(x, weight, output)
                    self.assertTrue(torch.equal(rng, torch.cuda.get_rng_state()))
                    self.assertIn("._plain_mm.", binding._launches[0].operator_id)
                    binding.launch()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        binding.launch()
                    graph.replay()
                    actual = output.index_select(0, indices)
                    check_linear_enclosure(corpus, weight, actual)
                    self.assertTrue(torch.equal(actual[:1], first))
                    if common is not None:
                        self.assertTrue(torch.equal(actual, common))
                    common = actual.clone()
                    graph.reset()

    def test_underfilled_plain_fixed_partitions_cross_m_and_poison(self):
        torch.manual_seed(910)
        for dtype in (torch.bfloat16, torch.float16):
            for n, k in ((4, 10240), (12, 2560), (24, 2560), (4, 2112),
                         (160, 2560), (320, 10240), (640, 10240), (2560, 2560)):
                weight = torch.randn(n, k, device="cuda:0", dtype=dtype) * 0.01
                corpus = torch.randn(3, k, device="cuda:0", dtype=dtype)
                common = None
                partitions = None
                for m in (4, 16, 64, 256, 5120):
                    x = torch.full((m, k), float("nan"), device="cuda:0", dtype=dtype)
                    indices = torch.tensor((m - 1, m // 2, 0), device=x.device)
                    x.index_copy_(0, indices, corpus)
                    output = torch.empty(m, n, device=x.device, dtype=dtype)
                    binding = PackedLinearOperators().bind_linear(x, weight, output)
                    self.assertIn("_plain_partition_mm", binding._launches[0].operator_id)
                    count = binding.scratch[0].shape[0]
                    if partitions is not None:
                        self.assertEqual(count, partitions)
                    partitions = count
                    binding.launch()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        binding.launch()
                    graph.replay()
                    actual = output.index_select(0, indices)
                    check_linear_enclosure(corpus, weight, actual)
                    if common is not None:
                        self.assertTrue(torch.equal(actual, common))
                    common = actual.clone()
                first = torch.empty(1, n, device=corpus.device, dtype=dtype)
                PackedLinearOperators().bind_linear(corpus[:1], weight, first).launch()
                self.assertTrue(torch.equal(first, common[:1]))

    def test_partitioned_groups_private_scratch_activity_and_cross_m(self):
        from cubic_serve.ops.cuda.linear.grouped import bind_partitioned_linear_group

        torch.manual_seed(291)
        for dtype in (torch.bfloat16, torch.float16):
            k = 2048
            weights = tuple(torch.randn(n, k, device="cuda:0", dtype=dtype) * 0.01
                            for n in (128, 8, 64))
            corpus = torch.randn(3, k, device="cuda:0", dtype=dtype)
            common = None
            for capacity in (1, 4, 32, 256):
                count = min(3, capacity)
                x = torch.full((capacity, k), float("nan"), device="cuda:0", dtype=dtype)
                x[:count].copy_(corpus[:count])
                outputs = tuple(torch.empty(capacity, w.shape[0], device=x.device, dtype=dtype)
                                for w in weights)
                active = torch.tensor([count], device=x.device, dtype=torch.int32)
                binding = bind_partitioned_linear_group(x, weights, outputs, active)
                self.assertIsNotNone(binding)
                scratch = torch.full_like(binding.scratch[0], float("nan"))
                binding.rebind_private_scratch(scratch)
                self.assertIs(binding.scratch[0], scratch)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    binding.launch()
                graph.replay()
                for weight, output in zip(weights, outputs, strict=True):
                    check_linear_enclosure(corpus[:count], weight, output[:count])
                    self.assertTrue((output[count:] == 0).all().item())
                current = tuple(y[:1].clone() for y in outputs)
                if common is not None:
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(common, current)))
                common = current
                for invalid in (0, capacity + 1):
                    active.fill_(invalid)
                    for output in outputs:
                        output.fill_(13)
                    graph.replay()
                    self.assertTrue(all((y == 13).all().item() for y in outputs))
                graph.reset()

    def test_host_materialization_identity_is_shared_across_buckets(self):
        weight = QuantizedMatrix(
            {"weight": torch.zeros(64, 128, dtype=torch.float8_e4m3fn),
             "weight_scale": torch.ones(64, 1)},
            QuantizationScheme(WeightFormat.FP8, 8, 1), (64, 128),
        )
        cache = {}
        first = PackedLinearOperators(materialized_weights=cache)
        sibling = PackedLinearOperators(materialized_weights=cache)
        device = torch.device("cuda:0")
        matrix = first.materialize_weight(weight, device)
        self.assertIs(matrix, first.materialize_weight(weight, device))
        self.assertIs(matrix, sibling.materialize_weight(weight, device))
        self.assertEqual(len(cache), 1)
        self.assertTrue(all(t.device.type == "cpu" for t in weight.payload.values()))
        # Mutation invalidates the representation, rather than serving stale data.
        weight.payload["weight_scale"].add_(1)
        self.assertIsNot(matrix, sibling.materialize_weight(weight, device))

    def test_declared_plain_groups_preserve_outputs_padding_and_cross_m(self):
        torch.manual_seed(174)
        for dtype in (torch.bfloat16, torch.float16):
            weights = tuple(torch.randn(n, 256, device="cuda:0", dtype=dtype) * 0.05
                            for n in (384, 16, 16, 256))
            corpus = torch.randn(33, 256, device="cuda:0", dtype=dtype) * 0.1
            common = None
            for cap, m in ((1, 1), (8, 8), (64, 33)):
                x = torch.zeros(cap, 256, device="cuda:0", dtype=dtype)
                results = []
                for grouped in (False, True):
                    config = OperatorConfig.from_dict({"version": 1, "overrides": {
                        "linear": {"grouped_projections": grouped}
                    }})
                    outputs = tuple(torch.empty(cap, w.shape[0], device=x.device, dtype=dtype)
                                    for w in weights)
                    dispatch = PackedProjectionDispatch(PackedLinearOperators(config))
                    dispatch.prepare_group(x, weights, outputs)
                    dispatch.seal()
                    self.assertEqual(len(dispatch.project_artifacts()), 1 if grouped else 4)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for w, y in zip(weights, outputs, strict=True):
                            dispatch.launch(x, w, y)
                    x.fill_(float("nan"))
                    x[:m] = corpus[:m]
                    graph.replay()
                    torch.cuda.synchronize()
                    for w, y in zip(weights, outputs, strict=True):
                        check_linear_enclosure(x[:m], w, y[:m])
                    results.append(tuple(y[:m].clone() for y in outputs))
                    graph.reset()
                self.assertTrue(all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                                    for a, b in zip(*results, strict=True)))
                first = tuple(y[:1] for y in results[1])
                if common is not None:
                    self.assertTrue(all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                                        for a, b in zip(common, first, strict=True)))
                common = first

    def test_cubic_levels_preserve_compact_weights_and_share_only_immutable_tables(self):
        disabled = OperatorConfig.from_dict(
            {"version": 1, "overrides": {"cubic_linear": {"level_table_max_bits": 0}}}
        )
        # Exercise the decoder independently of its resident-memory admission.
        # Production admission is covered separately by the policy boundaries.
        tables = OperatorConfig.from_dict(
            {"version": 1, "overrides": {"cubic_linear": {
                "level_table_max_bits": 8, "level_table_max_packed_ratio": 0.0,
            }}}
        )
        for bits in range(2, 9):
            for group in (32, 128):
                n, k = 256, 128
                matrix = QuantizedMatrix(
                    {
                        "weight_packed": torch.randint(
                            0, 256, (n, k * bits // 8), dtype=torch.uint8, device="cuda:0"
                        ),
                        "weight_scale": torch.randint(
                            0, 128, (n, k // group), dtype=torch.int8, device="cuda:0"
                        ),
                        "weight_ab": torch.arange(n, device="cuda:0")
                        .to(torch.uint8)[:, None]
                        .expand(n, k // group)
                        .contiguous(),
                        "weight_scale_global": torch.tensor([0.000125, 0.00025], device="cuda:0"),
                        "weight_a_global": torch.tensor([0.03125, 0.0625], device="cuda:0"),
                        "weight_b_global": torch.tensor([0.0625, 0.03125], device="cuda:0"),
                    },
                    QuantizationScheme(
                        WeightFormat.CUBIC,
                        bits,
                        group,
                        metadata="int8-scale-int4-ab-global-fp32",
                    ),
                    (n, k),
                    (128, 128),
                )
                shared = {}
                for dtype in (torch.bfloat16, torch.float16):
                    operators = PackedLinearOperators(tables, prepared_weights=shared)
                    corpus = torch.randn(33, k, device="cuda:0", dtype=dtype)
                    expected = PackedLinearOperators(disabled).linear(corpus, matrix)
                    actual = operators.linear(corpus, matrix)
                    if bits == 4:
                        from cubic_serve.quantization.extensions.cubic.linear import (
                            _levels,
                            arguments,
                        )

                        args, _ = arguments(operators, matrix)
                        levels = _levels(operators, matrix, args, dtype)
                        decoded = torch.empty((n, k), device=corpus.device, dtype=dtype)
                        _decode_cubic_level_words[(triton.cdiv(n, 16), triton.cdiv(k, 64))](
                            args[0],
                            levels,
                            *args[2:],
                            decoded,
                            n,
                            k,
                            matrix.encoding.group_in,
                            matrix.encoding.group_out,
                        )
                        reference = decode_matrix(matrix).to(dtype)
                        self.assertTrue(
                            torch.equal(decoded.view(torch.uint8), reference.view(torch.uint8))
                        )
                    self.assertTrue(
                        torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
                    )
                    with ReferenceExecutionContext():
                        check_linear_enclosure(corpus, matrix, actual)
                    if bits == 4:
                        host_matrix = matrix.to("cpu")
                        prepared, materialized = {}, {}
                        host_ops = PackedLinearOperators(
                            prepared_weights=prepared, materialized_weights=materialized
                        )
                        host_actual = host_ops.linear(corpus, host_matrix)
                        self.assertTrue(torch.equal(
                            actual.view(torch.uint8), host_actual.view(torch.uint8)
                        ))
                        table_count = len(prepared)
                        sibling = PackedLinearOperators(
                            prepared_weights=prepared, materialized_weights=materialized
                        )
                        one = sibling.linear(corpus[:1], host_matrix)
                        self.assertTrue(torch.equal(one.view(torch.uint8), actual[:1].view(torch.uint8)))
                        self.assertEqual(len(prepared), table_count)
                        self.assertEqual(len(materialized), 1)
                    before = tuple(id(value[1]) for value in shared.values())
                    second_bucket = PackedLinearOperators(tables, prepared_weights=shared)
                    single = second_bucket.linear(corpus[:1], matrix)
                    self.assertEqual(before, tuple(id(value[1]) for value in shared.values()))
                    self.assertTrue(
                        torch.equal(single.view(torch.uint8), actual[:1].view(torch.uint8))
                    )
                    self.assertTrue(
                        all(
                            v[1].numel() == n * (k // group) * (1 << (bits - 1))
                            for v in shared.values()
                        )
                    )
                self.assertEqual(len(shared), 2)  # dtype boundaries cannot share a table

    def test_cubic_magnitude_codebook_preserves_all_bit_widths(self):
        # Exhaust every encoded value, including the reserved negative-zero
        # code, against the independent checkpoint decoder. No GEMM tolerance
        # can hide a changed codebook or final weight rounding here.
        for bits in range(1, 9):
            k = 256
            codes = torch.arange(k, dtype=torch.int64) % (1 << bits)
            packed = torch.zeros(k * bits // 8, dtype=torch.uint8)
            for column, code in enumerate(codes.tolist()):
                for bit in range(bits):
                    offset = column * bits + bit
                    packed[offset // 8] |= ((code >> bit) & 1) << (offset % 8)
            weight = QuantizedMatrix(
                {
                    "weight_packed": packed.expand(4, -1).contiguous().cuda(),
                    "weight_metadata": torch.tensor(
                        [[0x1000 | (curve << 14)] * (k // 32) for curve in range(4)],
                        dtype=torch.uint16,
                        device="cuda:0",
                    ),
                    "weight_curve_a": torch.tensor(
                        [[0.5, 0.75, 1.0, 1.25]], dtype=torch.float16, device="cuda:0"
                    ),
                    "weight_curve_b": torch.tensor(
                        [[-0.25, 0.125, 0.25, -0.125]], dtype=torch.float16, device="cuda:0"
                    ),
                },
                QuantizationScheme(
                    WeightFormat.CUBIC, bits, 32, metadata="e5m9-scale-curve2-tensor-fp16"
                ),
                (4, k),
            )
            ids = torch.arange(4, device="cuda:0")
            reference = decode_matrix(weight)
            for dtype in (torch.float16, torch.bfloat16):
                actual = PackedLinearOperators().embedding(ids, weight, dtype=dtype)
                self.assertTrue(torch.equal(actual, reference.to(dtype)), (bits, dtype))

    def test_cubic_split_k_graph_enclosure_and_cross_m_identity(self):
        # Same reduction plan across all physical M, including poisoned padding.
        for dtype in (torch.bfloat16, torch.float16):
            for original in self.matrices(dtype):
                if (
                    not isinstance(original, QuantizedMatrix)
                    or original.encoding.format is not WeightFormat.CUBIC
                ):
                    continue
                for bits in (4, 6, 8):
                    # Enough output CTAs to exercise the reuse regime as well
                    # as the small-M parallelism regime on the audited device.
                    copies = 64
                    n, k = original.shape[0] * copies, 4096
                    payload = {
                        key: (
                            value.repeat(copies, 32)
                            if key in ("weight_metadata", "weight_scale", "weight_ab")
                            else value
                        )
                        for key, value in original.payload.items()
                    }
                    payload["weight_packed"] = torch.randint(
                        0,
                        256,
                        (n, k * bits // 8),
                        dtype=torch.uint8,
                        device="cuda:0",
                    )
                    weight = QuantizedMatrix(
                        payload,
                        replace(original.encoding, bits=bits),
                        (n, k),
                        tuple(count * copies for count in original.partitions),
                    )
                    corpus = torch.randn(33, k, dtype=dtype, device="cuda:0")
                    common = None
                    for capacity, logical in ((1, 1), (32, 31), (64, 33), (256, 33), (512, 33)):
                        inputs = torch.full(
                            (capacity, k), float("nan"), dtype=dtype, device="cuda:0"
                        )
                        inputs[:logical] = corpus[:logical]
                        output = torch.empty(capacity, n, dtype=dtype, device="cuda:0")
                        binding = PackedLinearOperators().bind_linear(inputs, weight, output)
                        binding.launch()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            binding.launch()
                        graph.replay()
                        torch.cuda.synchronize()
                        actual = output[:logical].clone()
                        with ReferenceExecutionContext():
                            check_linear_enclosure(corpus[:logical], weight, actual)
                        if common is not None:
                            self.assertTrue(torch.equal(common, actual[:1]))
                        common = actual[:1].clone()
                        inputs[:logical] = corpus[:logical].flip(0)
                        graph.replay()
                        torch.cuda.synchronize()
                        self.assertTrue(torch.equal(output[:logical].flip(0), actual))

    def test_serial_projection_scratch_is_shared_and_dispatch_private(self):
        weight = next(
            w
            for w in self.matrices(torch.bfloat16)
            if isinstance(w, QuantizedMatrix) and w.encoding.format is WeightFormat.CUBIC
        )
        weights = (weight.rows(0, 16), weight, weight.rows(0, 24))
        x = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda:0")
        expected = tuple(PackedLinearOperators().linear(x, w).clone() for w in weights)
        ops = PackedLinearOperators()
        dispatches = []
        for _ in range(2):
            dispatch = PackedProjectionDispatch(ops)
            outputs = tuple(torch.empty_like(y) for y in expected)
            for w, y in zip(weights, outputs, strict=True):
                dispatch.prepare(x, w, y)
            dispatch.seal()
            self.assertEqual(len(dispatch.scratch), 1)
            max_elements = max(
                binding.scratch[0].numel() for _, binding in dispatch._entries.values()
            )
            self.assertEqual(dispatch.scratch[0].numel(), max_elements)
            self.assertEqual(
                len(
                    {
                        binding.scratch[0].untyped_storage().data_ptr()
                        for _, binding in dispatch._entries.values()
                    }
                ),
                1,
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for w, y in zip(weights, outputs, strict=True):
                    dispatch.launch(x, w, y)
            for _ in range(2):
                dispatch.scratch[0].fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(
                    all(torch.equal(a, b) for a, b in zip(outputs, expected, strict=True))
                )
            dispatches.append((dispatch, graph))
        self.assertNotEqual(
            dispatches[0][0].scratch[0].data_ptr(), dispatches[1][0].scratch[0].data_ptr()
        )
        for dispatch, _ in dispatches:
            for _, binding in dispatch._entries.values():
                with self.assertRaisesRegex(ValueError, "unlaunched"):
                    binding.rebind_private_scratch(binding.scratch[0])

    def test_sealed_projection_occurrences_reject_new_inputs_and_accept_equal_views(self):
        ops = PackedLinearOperators()
        dispatch = PackedProjectionDispatch(ops)
        x = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda:0")
        weight = torch.randn(16, 128, dtype=torch.bfloat16, device="cuda:0")
        output = torch.empty(8, 16, dtype=torch.bfloat16, device="cuda:0")
        dispatch.prepare(x, weight, output)
        with self.assertRaisesRegex(ValueError, "sealed"):
            dispatch.launch(x, weight, output)
        dispatch.seal()
        with self.assertRaisesRegex(ValueError, "sealed"):
            dispatch.prepare(x, weight, output)
        with self.assertRaisesRegex(ValueError, "startup"):
            dispatch.launch(x.clone(), weight, output)
        dispatch.launch(x.view_as(x), weight, output.view_as(output))
        expected = output.clone()
        graph = torch.cuda.CUDAGraph()
        with patch.object(torch, "empty", side_effect=AssertionError("allocation")):
            with torch.cuda.graph(graph):
                dispatch.launch(x.view_as(x), weight, output.view_as(output))
            graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(expected, output))

    def test_fp8_input_shards_preserve_full_token_scale_in_static_graph(self):
        # The largest operand is deliberately on the last shard. Local amax
        # would change the other shards' codes and hence the mathematical input.
        for dtype in (torch.float16, torch.bfloat16):
            torch.manual_seed(702)
            corpus = torch.randn(33, 512, dtype=dtype, device="cuda:0")
            corpus[:, -1] = 448
            scheme = QuantizationScheme(
                WeightFormat.FP8, 8, 1, activation=ActivationFormat.FP8_TOKEN
            )
            reference = quantize_activation(corpus, scheme)
            codes = torch.randint(-4, 5, (16, 512), device="cuda:0").to(torch.float8_e4m3fn)
            weight = QuantizedMatrix(
                {"weight": codes, "weight_scale": torch.full((16, 1), 0.125, device="cuda:0")},
                scheme,
                (16, 512),
            )
            for tp in (1, 2, 4):
                width = 512 // tp
                for rank in range(tp):
                    start = rank * width
                    shard = weight.columns(start, start + width)
                    expected = (
                        (
                            reference.codes[:, start : start + width].double()
                            @ codes[:, start : start + width].double().t()
                        )
                        * reference.scales.double()
                        * 0.125
                    )
                    expected = expected.to(dtype)
                    for capacity, logical in ((1, 1), (8, 8), (64, 33)):
                        x = torch.empty(capacity, width, dtype=dtype, device="cuda:0")
                        scales = torch.ones(capacity, dtype=torch.float32, device="cuda:0")
                        output = torch.empty(capacity, 16, dtype=dtype, device="cuda:0")
                        binding = PackedLinearOperators().bind_linear(
                            x, shard, output, fp8_token_scales=scales
                        )
                        x.zero_()
                        binding.launch()
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            binding.launch()
                        for order in (torch.arange(logical), torch.arange(logical - 1, -1, -1)):
                            x.fill_(float("nan"))
                            x[:logical].copy_(corpus[order, start : start + width])
                            scales[:logical].copy_(reference.scales[order, 0])
                            before = scales.clone()
                            graph.replay()
                            torch.cuda.synchronize()
                            self.assertTrue(torch.equal(scales, before))
                            self.assertTrue(torch.equal(output[:logical], expected[order]))
                            # Witness the quantization boundary independently of GEMM.
                            actual_codes = binding._launches[0].tensors[1][:logical]
                            self.assertTrue(
                                torch.equal(
                                    actual_codes.view(torch.uint8),
                                    reference.codes[order, start : start + width].view(torch.uint8),
                                )
                            )

    def matrices(self, dtype):
        device = "cuda:0"
        n, k = 40, 128
        yield torch.randn(n, k, device=device, dtype=dtype) * 0.05
        yield QuantizedMatrix(
            {
                "qweight": torch.randint(
                    -(2**31), 2**31 - 1, (k, n // 8), dtype=torch.int32, device=device
                ),
                "qzeros": torch.full((1, n // 8), 0x77777777, dtype=torch.int32, device=device),
                "scales": torch.full((1, n), 0.03125, dtype=torch.float16, device=device),
            },
            QuantizationScheme(WeightFormat.AWQ, 4, 128),
            (n, k),
        )
        for mode in ("int8-scale-int4-ab-global-fp32", "e5m9-scale-curve2-tensor-fp16"):
            e = QuantizationScheme(WeightFormat.CUBIC, 6, 32, metadata=mode)
            p = {
                "weight_packed": torch.randint(
                    0, 256, (n, k * 6 // 8), device=device, dtype=torch.uint8
                )
            }
            if mode.startswith("int8"):
                p.update(
                    weight_scale=torch.ones(n, 4, device=device, dtype=torch.int8),
                    weight_ab=torch.full((n, 4), 0x23, device=device, dtype=torch.uint8),
                    weight_scale_global=torch.tensor([0.125, 0.25], device=device),
                    weight_a_global=torch.tensor([0.03125, 0.0625], device=device),
                    weight_b_global=torch.tensor([0.0625, 0.03125], device=device),
                )
            else:
                p.update(
                    weight_metadata=torch.full((n, 4), 0x1000, device=device, dtype=torch.uint16),
                    weight_curve_a=torch.full((2, 4), 0.75, device=device, dtype=torch.float16),
                    weight_curve_b=torch.full((2, 4), 0.125, device=device, dtype=torch.float16),
                )
            yield QuantizedMatrix(p, e, (n, k), (16, 24))
        yield QuantizedMatrix(
            {
                "weight": torch.randn(n, k, device=device).to(torch.float8_e4m3fn),
                "weight_scale": torch.full((n, 1), 0.125, device=device),
            },
            QuantizationScheme(WeightFormat.FP8, 8, 1, activation=ActivationFormat.FP8_TOKEN),
            (n, k),
        )
        yield QuantizedMatrix(
            {
                "weight_packed": torch.randint(
                    0, 256, (n, k // 2), device=device, dtype=torch.uint8
                ),
                "weight_scale": torch.ones(n, k // 16, device=device).to(torch.float8_e4m3fn),
                "weight_global_scale": torch.tensor([64.0], device=device),
                "input_global_scale": torch.ones(1, device=device),
            },
            QuantizationScheme(WeightFormat.NVFP4, 4, 16, activation=ActivationFormat.NVFP4_LOCAL),
            (n, k),
        )

    def test_dtypes_enclosures_row_permutations_and_masked_tiles(self):
        with ReferenceExecutionContext():
            for dtype in (torch.float16, torch.bfloat16):
                for weight in self.matrices(dtype):
                    operator = PackedLinearOperators()
                    corpus = torch.randn(33, 128, device="cuda:0", dtype=dtype)
                    # Prevent any hidden reference decoder from satisfying native execution.
                    with patch(
                        "cubic_serve.ops.eager.quantization.decode_weight",
                        side_effect=AssertionError("reference fallback"),
                    ):
                        isolated = operator.linear(corpus[:1], weight)
                        eight = operator.linear(corpus[:8], weight)
                        all_rows = operator.linear(corpus, weight)
                        reverse = operator.linear(corpus.flip(0), weight).flip(0)
                    self.assertTrue(torch.equal(isolated, all_rows[:1]))
                    self.assertTrue(torch.equal(eight, all_rows[:8]))
                    self.assertTrue(torch.equal(reverse, all_rows))
                    check_linear_enclosure(corpus, weight, all_rows)
                    if isinstance(weight, QuantizedMatrix):
                        ids = torch.tensor([[0, 17, 39]], device="cuda:0")
                        expected = embedding_matrix(ids, weight, dtype=dtype)
                        actual = operator.embedding(ids, weight, dtype=dtype)
                        self.assertTrue(torch.equal(actual, expected))

    def test_bound_graph_replay_static_scratch_and_poisoned_padding(self):
        # Primitive diagnostic: this does not issue a model/serving graph seal.
        with ReferenceExecutionContext():
            for dtype in (torch.float16, torch.bfloat16):
                for weight in self.matrices(dtype):
                    operator = PackedLinearOperators()
                    corpus = torch.randn(33, 128, dtype=dtype, device="cuda:0")
                    expected = operator.linear(corpus, weight)
                    for capacity, logical in ((1, 1), (8, 8), (64, 33)):
                        x = torch.zeros(capacity, 128, dtype=dtype, device="cuda:0")
                        output = torch.empty(capacity, 40, dtype=dtype, device="cuda:0")
                        binding = operator.bind_linear(x, weight, output)
                        artifacts = binding.project_artifacts()
                        self.assertEqual(len(artifacts), len(binding.launch()))
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        # Both warm capture and replay need no new tensor, JIT
                        # specialization, or reference operator.
                        with (
                            patch.object(torch, "empty", side_effect=AssertionError("allocation")),
                            patch.object(
                                torch, "empty_like", side_effect=AssertionError("allocation")
                            ),
                            patch(
                                "cubic_serve.ops.eager.quantization.decode_weight",
                                side_effect=AssertionError("reference fallback"),
                            ),
                        ):
                            with torch.cuda.graph(graph):
                                binding.launch()
                            for reverse in (False, True):
                                x.fill_(float("nan"))
                                selected = corpus[:logical].flip(0) if reverse else corpus[:logical]
                                x[:logical].copy_(selected)
                                graph.replay()
                                actual = output[:logical].flip(0) if reverse else output[:logical]
                                self.assertTrue(torch.equal(actual, expected[:logical]))
                        check_linear_enclosure(x[:logical], weight, output[:logical])
                        self.assertEqual(binding.project_artifacts(), artifacts)
                        ids = torch.arange(capacity, device=x.device) % 40
                        embeddings = torch.empty(capacity, 128, dtype=dtype, device=x.device)
                        lookup = operator.bind_embedding(ids, weight, embeddings)
                        lookup.launch()
                        torch.cuda.synchronize()
                        lookup_graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(lookup_graph):
                            lookup.launch()
                        ids.copy_(ids.flip(0))
                        lookup_graph.replay()
                        reference = (
                            torch.nn.functional.embedding(ids, weight)
                            if isinstance(weight, torch.Tensor)
                            else embedding_matrix(ids, weight, dtype=dtype)
                        )
                        self.assertTrue(torch.equal(embeddings, reference))
                        # Mutating a bound input ABI must fail before any launch.
                        x.transpose_(0, 1)
                        with self.assertRaisesRegex(ValueError, "changed after binding"):
                            binding.launch()

    def test_activation_codes_scales_midpoints_zero_and_underflow(self):
        for dtype in (torch.float16, torch.bfloat16):
            values = torch.tensor(
                [
                    [
                        0,
                        0.25,
                        0.75,
                        1.25,
                        1.75,
                        2.5,
                        3.5,
                        5,
                        6,
                        -0.25,
                        -0.75,
                        -1.25,
                        -2.5,
                        -3.5,
                        -5,
                        -6,
                    ]
                ],
                dtype=dtype,
                device="cuda:0",
            )
            values = torch.cat((values, torch.zeros_like(values), values * torch.finfo(dtype).tiny))
            for format, activation in (
                (WeightFormat.FP8, ActivationFormat.FP8_TOKEN),
                (WeightFormat.NVFP4, ActivationFormat.NVFP4_LOCAL),
            ):
                scheme = QuantizationScheme(
                    format,
                    8 if format is WeightFormat.FP8 else 4,
                    1 if format is WeightFormat.FP8 else 16,
                    activation=activation,
                )
                inverse = torch.ones(1, device="cuda:0")
                expected = quantize_activation(values.cpu(), scheme, global_inverse=inverse.cpu())
                if format is WeightFormat.FP8:
                    codes = torch.empty_like(values, dtype=torch.float8_e4m3fn)
                    scales = torch.empty(values.shape[0], device="cuda:0")
                    _fp8_quant[(values.shape[0],)](
                        values, codes.view(torch.uint8), scales, 16, 16, enable_fp_fusion=False
                    )
                else:
                    codes = torch.empty_like(values, dtype=torch.uint8)
                    scales = torch.empty(
                        (values.shape[0], 1), device="cuda:0", dtype=torch.float8_e4m3fn
                    )
                    _nvfp4_quant[(values.shape[0], 1)](
                        values, inverse, codes, scales.view(torch.uint8), 16, enable_fp_fusion=False
                    )
                self.assertTrue(
                    torch.equal(
                        codes.cpu().view(torch.uint8).flatten(),
                        expected.codes.view(torch.uint8).flatten(),
                    )
                )
                self.assertTrue(
                    torch.equal(scales.cpu().float().flatten(), expected.scales.float().flatten())
                )

    def test_cubic_bit_widths_group_origins_and_partial_input_group(self):
        operator = PackedLinearOperators()
        with ReferenceExecutionContext():
            for bits in range(1, 9):
                n, k, group_out, group_in = 32, 130, 4, 32
                scheme = QuantizationScheme(
                    WeightFormat.CUBIC, bits, group_in, group_out, "float32-scale-float16-ab"
                )
                matrix = QuantizedMatrix(
                    {
                        "weight_packed": torch.randint(
                            0, 256, (n, (k * bits + 7) // 8), device="cuda:0", dtype=torch.uint8
                        ),
                        "weight_scale": torch.full((n // group_out, 5), 0.25, device="cuda:0"),
                        "weight_a": torch.full(
                            (n // group_out, 5), 0.75, device="cuda:0", dtype=torch.float16
                        ),
                        "weight_b": torch.full(
                            (n // group_out, 5), 0.125, device="cuda:0", dtype=torch.float16
                        ),
                    },
                    scheme,
                    (n, k),
                    (16, 16),
                )
                x = torch.randn(33, k, device="cuda:0", dtype=torch.float16)
                active = (
                    operator
                    if bits != 3
                    else PackedLinearOperators(
                        OperatorConfig.from_dict(
                            {
                                "version": 1,
                                "overrides": {
                                    "cubic_linear": {
                                        "fixed_block_n": 64,
                                        "fixed_block_k": 64,
                                        "embedding_block_k": 64,
                                    }
                                },
                            }
                        )
                    )
                )
                y = active.linear(x, matrix)
                self.assertTrue(torch.equal(y[:1], active.linear(x[:1], matrix)))
                check_linear_enclosure(x, matrix, y)
                ids = torch.tensor([[0, 15, 16, 31]], device="cuda:0")
                self.assertTrue(
                    torch.equal(
                        active.embedding(ids, matrix, dtype=x.dtype),
                        embedding_matrix(ids, matrix, dtype=x.dtype),
                    )
                )
