"""Original Cubic codes, group lifetimes, and logical-row ownership in Marlin."""

import os

import pytest
import torch

from cubic_serve.executor.runner import configure_reference_math
from cubic_serve.guardrail.quantization import check_linear_enclosure
from cubic_serve.models.quantization import QuantizationScheme, WeightFormat
from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
from cubic_serve.quantization.api import RoutedRows
from cubic_serve.quantization.common.marlin.parameters import MarlinParameters, pipeline_compatible
from cubic_serve.quantization.extensions.cubic.marlin import _bind
from cubic_serve.quantization.extensions.cubic.policy import CubicMoEParameters


def test_pipeline_closes_single_or_paired_fragments_and_group_lifetime():
    from dataclasses import replace

    p = MarlinParameters()
    assert pipeline_compatible(p, group=512)
    assert not pipeline_compatible(p, group=96)
    assert not pipeline_compatible(replace(p, marlin_block_k=64), group=32)
    assert pipeline_compatible(replace(p, marlin_block_k=64), group=32, single_subtile=True)
    assert not pipeline_compatible(replace(p, marlin_block_k=32), group=32)
    assert not pipeline_compatible(replace(p, marlin_stages=3), group=512)
    assert pipeline_compatible(replace(p, marlin_block_n=128, marlin_warps=8), group=512)
    assert pipeline_compatible(replace(p, marlin_stages=2), group=256)


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required"
)
@pytest.mark.parametrize("bits,group", [(2, 32), (4, 512), (8, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_grouped_dense_preserves_original_reduction_and_output_ownership(bits, group, dtype):
    from cubic_serve.quantization.extensions.cubic.marlin import _bind_dense, bind_linear

    configure_reference_math()
    torch.set_num_threads(1)
    torch.manual_seed(418)
    device, n, k = "cuda:0", 128, 2048
    weights = tuple(
        QuantizedMatrix(
            {
                "weight_packed": torch.randint(
                    0, 256, (n, k * bits // 8), device=device, dtype=torch.uint8
                ),
                "weight_scale": torch.randint(
                    0, 120, (n, k // group), device=device, dtype=torch.int8
                ),
                "weight_ab": torch.randint(
                    0, 256, (n, k // group), device=device, dtype=torch.uint8
                ),
                "weight_scale_global": torch.tensor([0.001, 0.002], device=device),
                "weight_a_global": torch.tensor([0.07, 0.04], device=device),
                "weight_b_global": torch.tensor([0.031, 0.061], device=device),
            },
            QuantizationScheme(
                WeightFormat.CUBIC, bits, group, metadata="int8-scale-int4-ab-global-fp32"
            ),
            (n, k), (n // 2, n // 2),
        )
        for _ in range(3)
    )
    operators = PackedLinearOperators()
    corpus = torch.randn(65, k, device=device, dtype=dtype) * 0.02
    first = None
    for m, bm in ((1, 16), (3, 16), (8, 16), (17, 32), (33, 32), (65, 64)):
        x = corpus[:m].clone()
        storage = tuple(torch.full((66, n), float("nan"), device=device, dtype=dtype)
                        for _ in weights)
        outputs = tuple(t[:m] for t in storage)
        params = MarlinParameters(
            marlin_auto_geometry=False, marlin_block_m=bm, marlin_stages=2 if bm == 64 else 4
        )
        expected = []
        for weight, output in zip(weights, outputs, strict=True):
            binding = bind_linear(operators, x, weight, output, params, fixed_partitions=4)
            assert binding is not None
            binding.launch()
            expected.append(output.clone())
        grouped = _bind_dense(operators, x, weights, outputs, params, fixed_partitions=4)
        assert grouped is not None
        assert len(grouped._launches) == 2
        grouped.launch()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            grouped.launch()
        graph.replay()
        for weight, output, original, protected in zip(weights, outputs, expected, storage,
                                                      strict=True):
            assert torch.equal(output.view(torch.uint8), original.view(torch.uint8))
            check_linear_enclosure(x, weight, output)
            assert torch.isnan(protected[m:]).all()
        if first is not None:
            assert all(torch.equal(o[:1].view(torch.uint8), f.view(torch.uint8))
                       for o, f in zip(outputs, first, strict=True))
        first = tuple(o[:1].clone() for o in outputs)
        permutation = torch.arange(m - 1, -1, -1, device=device)
        x.copy_(corpus[:m][permutation])
        graph.replay()
        for output, original, protected in zip(outputs, expected, storage, strict=True):
            assert torch.equal(output.view(torch.uint8), original[permutation].view(torch.uint8))
            assert torch.isnan(protected[m:]).all()
        graph.reset()


def _assert_compact_storage(operators, groups):
    from cubic_serve.quantization.extensions.cubic.marlin import _prepare

    base = groups[0][0]
    bits = base.encoding.bits
    codes, metadata = _prepare(
        operators,
        groups,
        base.payload["weight_packed"].device,
        fragment_bits=4 if bits <= 4 else 8,
    )
    original_bytes = sum(w.payload["weight_packed"].numel() for group in groups for w in group)
    assert codes.numel() * codes.element_size() == original_bytes
    header = metadata[:24].contiguous().view(torch.int32).cpu().tolist()
    variant, n, group_count, experts, data_offset, _ = header
    expected_metadata_bytes = n * group_count * experts * (2 if variant else 8)
    assert metadata.numel() - data_offset == expected_metadata_bytes
    # Cache lookup must reuse compact storage, not make another widened copy.
    again = _prepare(operators, groups, codes.device, fragment_bits=4 if bits <= 4 else 8)
    assert again[0].data_ptr() == codes.data_ptr()
    assert again[1].data_ptr() == metadata.data_ptr()


@pytest.mark.parametrize("bits", range(2, 9))
def test_unqualified_single_subtile_rejected_before_binding(bits):
    n, k = 128, 512
    weight = QuantizedMatrix(
        {
            "weight_packed": torch.zeros((n, k * bits // 8), dtype=torch.uint8),
            "weight_scale": torch.zeros((n, k // 32), dtype=torch.int8),
            "weight_ab": torch.zeros((n, k // 32), dtype=torch.uint8),
            "weight_scale_global": torch.ones(1),
            "weight_a_global": torch.ones(1),
            "weight_b_global": torch.zeros(1),
        },
        QuantizationScheme(
            WeightFormat.CUBIC, bits, 32, metadata="int8-scale-int4-ab-global-fp32",
        ),
        (n, k), (n,),
    )
    # Admission must reject before metadata preparation or any CUDA operation.
    indices = torch.zeros(1, dtype=torch.int32)
    rows = RoutedRows(indices, indices, indices, indices, indices, 1, 1)
    result = _bind(
        PackedLinearOperators(), torch.empty((1, k), dtype=torch.bfloat16),
        ((weight,),), torch.empty((1, n), dtype=torch.bfloat16), rows, 1,
        CubicMoEParameters(marlin_block_k=64, marlin_auto_geometry=False),
    )
    assert result is None


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required"
)
@pytest.mark.parametrize(
    "bits,group,metadata",
    [
        *(
            (bits, group, "int8-scale-int4-ab-global-fp32")
            for bits, group in [(5, 512), *((bits, 32) for bits in range(2, 9))]
        ),
        (5, 32, "e5m9-scale-curve2-tensor-fp16"),
        (8, 512, "e5m9-scale-curve2-tensor-fp16"),
        (2, 32, "float32-scale-float16-ab"),
        (6, 128, "float32-scale-float16-ab"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_projection_original_oracle_padding_and_permutations(bits, group, metadata, dtype):
    configure_reference_math()
    torch.set_num_threads(1)
    torch.manual_seed(792)
    device, e, n, k = "cuda:0", 3, 128, 512

    def matrix():
        payload = {
            "weight_packed": torch.randint(
                0, 256, (n, k * bits // 8), device=device, dtype=torch.uint8
            ),
        }
        if metadata == "int8-scale-int4-ab-global-fp32":
            payload.update(
                {
                    "weight_scale": torch.randint(
                        0, 120, (n, k // group), device=device, dtype=torch.int8
                    ),
                    "weight_ab": torch.randint(
                        0, 256, (n, k // group), device=device, dtype=torch.uint8
                    ),
                    "weight_scale_global": torch.tensor([0.001, 0.002], device=device),
                    "weight_a_global": torch.tensor([0.07, 0.04], device=device),
                    "weight_b_global": torch.tensor([0.031, 0.061], device=device),
                }
            )
        elif metadata == "e5m9-scale-curve2-tensor-fp16":
            scale = (torch.rand(n, k // group, device=device) * 0.1).half()
            curve = torch.randint(0, 4, scale.shape, device=device)
            payload.update(
                {
                    "weight_metadata": ((scale.view(torch.int16).int() >> 1) | (curve << 14)).to(
                        torch.uint16
                    ),
                    "weight_curve_a": (0.8 + torch.rand(2, 4, device=device) * 0.4).half(),
                    "weight_curve_b": (torch.rand(2, 4, device=device) * 0.2 - 0.1).half(),
                }
            )
        else:
            payload.update(
                {
                    "weight_scale": torch.rand(n, k // group, device=device) * 0.1,
                    "weight_a": (0.8 + torch.rand(n, k // group, device=device) * 0.4).half(),
                    "weight_b": (torch.rand(n, k // group, device=device) * 0.2 - 0.1).half(),
                }
            )
        return QuantizedMatrix(
            payload,
            QuantizationScheme(WeightFormat.CUBIC, bits, group, metadata=metadata),
            (n, k),
            (n // 2, n // 2),
        )

    weights = tuple(matrix() for _ in range(e))
    corpus = torch.randn(33, k, device=device, dtype=dtype) * 0.02
    operators = PackedLinearOperators()
    _assert_compact_storage(operators, (weights,))
    first = None
    for cap, m, bm, bn, warps in ((1, 1, 16, 64, 4), (8, 8, 16, 128, 8), (64, 33, 32, 64, 4)):
        x = torch.full((cap, k), float("nan"), device=device, dtype=dtype)
        x[:m].copy_(corpus[:m])
        output = torch.full((cap * e, n), float("nan"), device=device, dtype=dtype)
        counts = torch.full((e,), m, device=device, dtype=torch.int32)
        starts = torch.arange(e, device=device, dtype=torch.int32) * m
        blocks = torch.arange(e, device=device, dtype=torch.int32) * ((m + bm - 1) // bm)
        total = torch.tensor([e * ((m + bm - 1) // bm)], device=device, dtype=torch.int32)
        row_map = torch.cat(
            [torch.arange(m, device=device, dtype=torch.int32) * e + expert for expert in range(e)]
        )
        candidate = _bind(
            operators,
            x,
            (weights,),
            output,
            RoutedRows(counts, starts, blocks, total, row_map, m, e),
            e,
            CubicMoEParameters(
                marlin_auto_geometry=False, marlin_block_m=bm, marlin_block_n=bn, marlin_warps=warps
            ),
        )
        assert candidate is not None
        binding = candidate[1]
        binding.launch()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            binding.launch()
        graph.replay()
        actual = output.view(cap, e, n)[:m].clone()
        for expert, weight in enumerate(weights):
            check_linear_enclosure(corpus[:m], weight, actual[:, expert])
        assert torch.isnan(output.view(cap, e, n)[m:]).all()
        if first is not None:
            assert torch.equal(actual[:1].view(torch.uint8), first)
        first = actual[:1].view(torch.uint8).clone()
        # Move requests to noncontiguous physical rows; padding has no identity.
        physical = (torch.arange(m, device=device, dtype=torch.int32) * 17) % cap
        x.fill_(float("nan"))
        x[physical.long()] = corpus[:m]
        row_map.copy_(torch.cat([physical * e + expert for expert in range(e)]))
        output.fill_(float("nan"))
        graph.replay()
        assert torch.equal(
            output.view(cap, e, n)[physical.long()].view(torch.uint8), actual.view(torch.uint8)
        )
        graph.reset()

    # One-hot contractions expose every original weight directly. An enclosure
    # alone could hide a small metadata decoding error inside its reduction bound.
    from cubic_serve.quantization.extensions.cubic.reference import decode_cubic

    identity = torch.eye(k, dtype=dtype, device=device)
    output = torch.empty(k * e, n, dtype=dtype, device=device)
    counts = torch.full((e,), k, dtype=torch.int32, device=device)
    starts = torch.arange(e, dtype=torch.int32, device=device) * k
    blocks = torch.arange(e, dtype=torch.int32, device=device) * (k // 16)
    total = torch.tensor([e * k // 16], dtype=torch.int32, device=device)
    rows = torch.cat(
        [torch.arange(k, dtype=torch.int32, device=device) * e + expert for expert in range(e)]
    )
    candidate = _bind(
        operators,
        identity,
        (weights,),
        output,
        RoutedRows(counts, starts, blocks, total, rows, k, e),
        e,
        CubicMoEParameters(marlin_auto_geometry=False),
    )
    assert candidate is not None
    candidate[1].launch()
    for expert, weight in enumerate(weights):
        expected = (
            decode_cubic(weight.payload, weight.encoding, weight.shape, weight.partitions)
            .to(dtype)
            .T.contiguous()
        )
        # Addition to the zero accumulator canonicalizes zero weight signs.
        expected[expected == 0] = 0
        actual = output.view(k, e, n)[:, expert].contiguous()
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_emitted_quotient_constants_match_division_on_every_original_code():
    import ctypes
    import ctypes.util
    import re
    import struct

    from cubic_serve.quantization.extensions.cubic.marlin import _kernel_source

    library = ctypes.CDLL(ctypes.util.find_library("m"))
    fma = library.fmaf
    fma.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
    fma.restype = ctypes.c_float
    for bits in range(2, 9):
        source = _kernel_source(MarlinParameters(), "kBFloat16", bits, 32)
        match = re.search(
            r"__fmaf_rn\(fabsf\(q\), ([^,]+)f, "
            r"__fmul_rn\(fabsf\(q\), ([^)]+)f\)\)",
            source,
        )
        assert match is not None
        hi, lo = (float.fromhex(value) for value in match.groups())
        denominator = (1 << (bits - 1)) - 1
        for magnitude in range(denominator + 1):
            actual = fma(magnitude, hi, ctypes.c_float(magnitude * lo))
            expected = struct.pack("f", magnitude / denominator)
            assert struct.pack("f", actual) == expected, (bits, magnitude)


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required"
)
@pytest.mark.parametrize(
    "bits,group", [(2, 32), (3, 128), (4, 512), (5, 32), (6, 128), (7, 512), (8, 32)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("fixed_partitions", [0, 4, 8])
def test_dense_carrier_original_weights_and_graph_tail(bits, group, dtype, fixed_partitions):
    from cubic_serve.quantization.extensions.cubic.marlin import bind_linear
    from cubic_serve.quantization.extensions.cubic.reference import decode_cubic

    configure_reference_math()
    torch.set_num_threads(1)
    torch.manual_seed(792)
    device, n = "cuda:0", 128
    k = {0: 512, 2: 4352, 5: 5120, 8: 4096}.get(fixed_partitions, 2048)
    payload = {
        "weight_packed": torch.randint(
            0, 256, (n, k * bits // 8), device=device, dtype=torch.uint8
        ),
        "weight_scale": torch.randint(0, 120, (n, k // group), device=device, dtype=torch.int8),
        "weight_ab": torch.randint(0, 256, (n, k // group), device=device, dtype=torch.uint8),
        "weight_scale_global": torch.tensor([0.001, 0.002], device=device),
        "weight_a_global": torch.tensor([0.07, 0.04], device=device),
        "weight_b_global": torch.tensor([0.031, 0.061], device=device),
    }
    weight = QuantizedMatrix(
        payload,
        QuantizationScheme(
            WeightFormat.CUBIC, bits, group, metadata="int8-scale-int4-ab-global-fp32"
        ),
        (n, k),
        (n // 2, n // 2),
    )
    operators = PackedLinearOperators()
    _assert_compact_storage(operators, ((weight,),))
    corpus = torch.randn(128, k, device=device, dtype=dtype) * 0.02
    first = None
    for m, bm in ((1, 16), (8, 16), (37, 32), (64, 32), (65, 64), (128, 64)):
        x = corpus[:m].clone()
        storage = torch.full((128, n), float("nan"), device=device, dtype=dtype)
        output = storage[:m]
        bound = bind_linear(
            operators,
            x,
            weight,
            output,
            MarlinParameters(
                marlin_auto_geometry=False, marlin_block_m=bm, marlin_stages=2 if bm == 64 else 4
            ),
            fixed_partitions=fixed_partitions,
        )
        assert bound is not None
        bound.launch()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            bound.launch()
        graph.replay()
        actual = output.clone()
        check_linear_enclosure(x, weight, actual)
        assert torch.isnan(storage[m:]).all()
        if first is not None:
            assert torch.equal(first.view(torch.uint8), actual[:1].view(torch.uint8))
        first = actual[:1].clone()
        permutation = torch.arange(m - 1, -1, -1, device=device)
        x.copy_(corpus[:m][permutation])
        graph.replay()
        assert torch.equal(output.view(torch.uint8), actual[permutation].view(torch.uint8))
        assert torch.isnan(storage[m:]).all()
        graph.reset()

    identity = torch.eye(k, device=device, dtype=dtype)
    output = torch.empty(k, n, device=device, dtype=dtype)
    bound = bind_linear(
        operators,
        identity,
        weight,
        output,
        MarlinParameters(marlin_auto_geometry=False),
        fixed_partitions=fixed_partitions,
    )
    assert bound is not None
    bound.launch()
    expected = (
        decode_cubic(payload, weight.encoding, weight.shape, weight.partitions)
        .to(dtype)
        .T.contiguous()
    )
    expected[expected == 0] = 0
    assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required"
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dense_non_power_of_two_k_partitions(dtype):
    test_dense_carrier_original_weights_and_graph_tail(4, 512, dtype, 5)


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required"
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dense_odd_k_partition_shared_reduction_handoff(dtype):
    # 4352 / (2 * 128) = 17 K tiles: the final operand stage aliases the
    # reduction scratch, unlike an even-length partition used in earlier cases.
    test_dense_carrier_original_weights_and_graph_tail(6, 32, dtype, 2)


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required"
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_device_enclosure_matches_cpu_and_rejects_corruption(dtype):
    from cubic_serve.ops.eager.quantization import decode_matrix

    configure_reference_math()
    torch.set_num_threads(2)
    torch.manual_seed(194)
    n, k, group = 7, 384, 128
    for bits in range(2, 9):
        for metadata in (
            "float32-scale-float16-ab",
            "int8-scale-int4-ab-global-fp32",
            "e5m9-scale-curve2-tensor-fp16",
        ):
            payload = {
                "weight_packed": torch.randint(0, 256, (n, k * bits // 8), dtype=torch.uint8)
            }
            if metadata == "float32-scale-float16-ab":
                payload.update(
                    weight_scale=torch.rand(n, 3) * 0.1,
                    weight_a=(torch.rand(n, 3) + 0.5).half(),
                    weight_b=(torch.rand(n, 3) * 0.2 - 0.1).half(),
                )
            elif metadata == "int8-scale-int4-ab-global-fp32":
                payload.update(
                    weight_scale=torch.randint(0, 120, (n, 3), dtype=torch.int8),
                    weight_ab=torch.randint(0, 256, (n, 3), dtype=torch.uint8),
                    weight_scale_global=torch.tensor([0.001, 0.002]),
                    weight_a_global=torch.tensor([0.07, 0.04]),
                    weight_b_global=torch.tensor([0.031, 0.061]),
                )
            else:
                scales = (torch.rand(n, 3) * 0.1).half()
                curves = torch.randint(0, 4, (n, 3))
                payload.update(
                    weight_metadata=((scales.view(torch.int16).int() >> 1) | (curves << 14)).to(
                        torch.uint16
                    ),
                    weight_curve_a=(torch.rand(2, 4) + 0.5).half(),
                    weight_curve_b=(torch.rand(2, 4) * 0.2 - 0.1).half(),
                )
            cpu = QuantizedMatrix(
                payload,
                QuantizationScheme(WeightFormat.CUBIC, bits, group, metadata=metadata),
                (n, k),
                (3, 4),
            )
            gpu = QuantizedMatrix(
                {key: value.cuda() for key, value in payload.items()},
                cpu.encoding,
                cpu.shape,
                cpu.partitions,
            )
            decoded = decode_matrix(cpu)
            assert torch.equal(
                decoded.view(torch.uint8), decode_matrix(gpu).cpu().view(torch.uint8)
            )
            x = (torch.randn(5, k) * 0.02).to(dtype).cuda()
            y = (x.cpu().double() @ decoded.to(dtype).double().T).to(dtype).cuda()
            host = check_linear_enclosure(x, gpu, y, output_rows=3, reference_device="cpu")
            device = check_linear_enclosure(x, gpu, y, output_rows=3)
            assert device["max_abs_error"] == pytest.approx(host["max_abs_error"], abs=1e-14)
            assert device["max_bound"] == pytest.approx(host["max_bound"], abs=1e-14)
            corrupted = y.clone()
            corrupted[2, 5] += 0.25
            with pytest.raises(AssertionError, match="exceeds"):
                check_linear_enclosure(x, gpu, corrupted, output_rows=3)


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required"
)
@pytest.mark.parametrize("bits", [3, 6])
def test_compact_repack_large_bit_address(bits):
    from cubic_serve.quantization.common.marlin.repack import repack_signed_codes

    n, k = 131072, 5120
    source = torch.zeros((n, k * bits // 8), dtype=torch.uint8, device="cuda")
    result = repack_signed_codes(source, bits=bits, fragment_bits=4 if bits <= 4 else 8)
    assert result.numel() * result.element_size() == source.numel()
    for start in (0, result.numel() // 2 // bits * bits, result.numel() - bits):
        words = result.reshape(-1)[start : start + bits].cpu().tolist()
        for i, actual in enumerate(words):
            expected = sum(int((32 * (start + i) + j) % bits == bits - 1) << j for j in range(32))
            assert actual & 0xFFFFFFFF == expected
