"""Original-format numerical contracts for shared row-local CUDA contractions."""

import os

import pytest
import torch

from cubic_serve.guardrail.quantization import check_linear_enclosure
from cubic_serve.models.quantization import QuantizationScheme
from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
from cubic_serve.ops.strategies import OperatorConfig
from cubic_serve.quantization.common.small_rows import bind
from cubic_serve.quantization.extensions.cubic.small_rows import prepare as cubic_prepare
from cubic_serve.quantization.extensions.int.small_rows import prepare as int_prepare


def _pack(codes, bits):
    """Scalar bitstream writer independent of the performance decoders."""
    n, k = codes.shape
    stride = ((k * bits + 31) // 32) * 4
    payload = bytearray(n * stride)
    for row, values in enumerate(codes.tolist()):
        for column, code in enumerate(values):
            for bit in range(bits):
                offset = column * bits + bit
                if code & (1 << bit):
                    payload[row * stride + offset // 8] |= 1 << (offset % 8)
    return torch.frombuffer(payload, dtype=torch.int32).clone().reshape(n, -1)


def _matrix(format_name, bits, group, k, generator):
    n, groups = 65, k // group
    codes = torch.randint(1 << bits, (n, k), generator=generator)
    scale = torch.rand(n, groups, generator=generator) * .03 + .005
    payload = {"weight_packed": _pack(codes, bits)}
    if format_name == "cubic":
        payload["weight_packed"] = payload["weight_packed"].view(torch.uint8)
        payload.update(
            weight_scale=scale,
            weight_a=(torch.rand(n, groups, generator=generator) + .25).half(),
            weight_b=(torch.rand(n, groups, generator=generator) - .5).half(),
        )
        encoding = QuantizationScheme(
            "cubic-bitstream-v1", bits, group, metadata="float32-scale-float16-ab"
        )
    else:
        zeros = torch.randint(1 << bits, (groups, n), generator=generator)
        payload.update(
            weight_scale=scale.half(),
            weight_zero_point=_pack(zeros, bits).t().contiguous(),
            weight_shape=torch.tensor([n, k], dtype=torch.int64),
        )
        encoding = QuantizationScheme("int-pack-v1", bits, group, metadata="dense:asym:F16")
    return QuantizedMatrix(payload, encoding, (n, k), (n,))


@pytest.mark.skipif(
    os.environ.get("CUBIC_RUN_CUDA_PACKED_ROW_TESTS") != "1",
    reason="explicit real CUDA selected-plan qualification",
)
@pytest.mark.parametrize("format_name", ["cubic", "int"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("bits", [4, 8])
def test_selected_row_recipe_has_an_independent_oracle_and_exact_row_transport(format_name, dtype, bits):
    """Audit the real candidate recipe, including expanded and permuted rows."""
    from cubic_serve.quantization.common import small_rows_cost as selection
    from cubic_serve.quantization.registry import get_extension

    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator().manual_seed(4389)
    operators = PackedLinearOperators(config=OperatorConfig(compute_mode="wna16"))
    matrix = _matrix(format_name, bits, 128, 4096, generator)
    prepare = cubic_prepare if format_name == "cubic" else int_prepare
    x = torch.randn(4, 4096, generator=generator).to(device=device, dtype=dtype)
    output = torch.empty(4, matrix.shape[0], device=device, dtype=dtype)
    candidate = bind(x, (prepare(operators, matrix, device),), (output,), warps=4, partitions=2)
    candidate.launch()
    torch.cuda.synchronize(device)
    check_linear_enclosure(x, matrix, output, compute_mode="wna16")
    params = operators._parameters(matrix)
    key = selection._plan_key(operators, x, (matrix,), params)
    operators._packed_row_plans = {(key, 4): ("simt", 4, 2)}
    plan = selection.row_arithmetic_plan(operators, 2)
    linear = get_extension(matrix.encoding.format).module("linear")
    for rows in ((0, 1), (2, 3), (3, 2)):
        row_indices = torch.tensor(rows, device=device)
        actual_input = x.index_select(0, row_indices)
        actual_output = torch.empty_like(output[:2])
        with selection.row_arithmetic_scope(plan):
            witness = selection.choose(operators, actual_input, (matrix,),
                                       (actual_output,), params, linear)
        witness.launch()
        torch.cuda.synchronize(device)
        check_linear_enclosure(actual_input, matrix, actual_output, compute_mode="wna16")
        assert torch.equal(actual_output, output.index_select(0, row_indices))


@pytest.mark.skipif(
    os.environ.get("CUBIC_RUN_CUDA_PACKED_ROW_TESTS") != "1",
    reason="explicit real CUDA packed-row qualification",
)
@pytest.mark.parametrize("format_name", ["cubic", "int"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_strided_projection_publication_preserves_other_columns(format_name, dtype):
    if not torch.cuda.is_available():
        pytest.skip("real CUDA device required")
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator().manual_seed(9231)
    operators = PackedLinearOperators(config=OperatorConfig(compute_mode="wna16"))
    prepare = cubic_prepare if format_name == "cubic" else int_prepare
    matrices = tuple(_matrix(format_name, 4, 128, 512, generator) for _ in range(2))
    prepared = tuple(prepare(operators, matrix, device) for matrix in matrices)
    n, k = matrices[0].shape
    for m in (1, 3, 4):
        x = torch.randn(m, k, generator=generator).to(device=device, dtype=dtype)
        backing = tuple(torch.full((m, n + 19), 13., device=device, dtype=dtype)
                        for _ in prepared)
        outputs = tuple(y[:, 7:7 + n] for y in backing)
        contiguous = tuple(torch.empty_like(y, memory_format=torch.contiguous_format)
                           for y in outputs)
        oracle = bind(x, prepared, contiguous, warps=4, partitions=1)
        candidate = bind(x, prepared, outputs, warps=4, partitions=1)
        oracle.launch()
        candidate.launch()
        torch.cuda.synchronize(device)
        for matrix, actual, expected, storage in zip(
                matrices, outputs, contiguous, backing, strict=True):
            assert torch.equal(actual, expected)
            assert torch.all(storage[:, :7] == 13.)
            assert torch.all(storage[:, 7 + n:] == 13.)
            check_linear_enclosure(x, matrix, actual.contiguous(), compute_mode="wna16")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            candidate.launch()
        graph.replay()
        torch.cuda.synchronize(device)
        for actual, expected in zip(outputs, contiguous, strict=True):
            assert torch.equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("CUBIC_RUN_CUDA_PACKED_ROW_TESTS") != "1",
    reason="explicit real CUDA packed-row qualification",
)
@pytest.mark.parametrize("format_name", ["cubic", "int"])
@pytest.mark.parametrize("bits", range(2, 9))
@pytest.mark.parametrize("members", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_original_weight_enclosure_cross_m_permutation_and_capture(
    format_name, bits, members, dtype
):
    if not torch.cuda.is_available():
        pytest.skip("real CUDA device required")
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator().manual_seed(1173 + bits)
    operators = PackedLinearOperators(config=OperatorConfig(compute_mode="wna16"))
    prepare = cubic_prepare if format_name == "cubic" else int_prepare
    for k, group, partition_counts in (
        (1280, 40, (1, 2)),
        (4096, 128, (1, 2, 4)),
        (4096, 512, (1, 2, 4)),
    ):
        matrices = tuple(_matrix(format_name, bits, group, k, generator) for _ in range(members))
        prepared = tuple(prepare(operators, matrix, device) for matrix in matrices)
        corpus = (torch.randn(4, k, generator=generator) * .1).to(device=device, dtype=dtype)
        for partitions in partition_counts:
            baseline = None
            for m in (1, 2, 3, 4):
                inputs = corpus[:m].clone()
                outputs = tuple(torch.empty(m, matrix.shape[0], device=device, dtype=dtype)
                                for matrix in matrices)
                binding = bind(inputs, prepared, outputs, partitions=partitions)
                binding.launch()
                torch.cuda.synchronize(device)
                for matrix, output in zip(matrices, outputs, strict=True):
                    check_linear_enclosure(inputs, matrix, output, compute_mode="wna16")
                if baseline is not None:
                    assert all(torch.equal(output[:old.shape[0]], old)
                               for output, old in zip(outputs, baseline, strict=True))
                baseline = tuple(output.clone() for output in outputs)
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph):
                        binding.launch()
                    inputs.copy_(corpus[:m].flip(0))
                    graph.replay()
                    torch.cuda.synchronize(device)
                    assert all(torch.equal(output, old.flip(0))
                               for output, old in zip(outputs, baseline, strict=True))
                    if m > 1:
                        inputs[1:].fill_(float("nan"))
                        graph.replay()
                        torch.cuda.synchronize(device)
                        assert all(torch.equal(output[:1], old[-1:])
                                   for output, old in zip(outputs, baseline, strict=True))
                        assert all(torch.isnan(output[1:]).all() for output in outputs)
                finally:
                    graph.reset()


@pytest.mark.skipif(
    os.environ.get("CUBIC_RUN_CUDA_PACKED_ROW_TESTS") != "1",
    reason="explicit real CUDA packed-row qualification",
)
@pytest.mark.parametrize("format_name", ["int", "cubic-int8", "cubic-e5m9"])
@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_transposed_mma_original_oracle_row_permutation_and_poison(format_name, bits, dtype):
    """Independent full-K math plus physical-row isolation across MMA row tiles."""
    from cubic_serve.quantization.common.small_mma import bind as mma_bind
    from cubic_serve.quantization.extensions.cubic.small_rows import prepare_mma as cp
    from cubic_serve.quantization.extensions.int.small_rows import prepare_mma as ip

    if not torch.cuda.is_available():
        pytest.skip("real CUDA device required")
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator().manual_seed(1931 + bits)
    operators = PackedLinearOperators(config=OperatorConfig(compute_mode="wna16"))
    for k, group in ((1024, 128), (1024, 512), (1152, 128)):
        n = 64
        matrices = []
        for _ in range(2):
            if format_name == "int":
                matrix = _matrix("int", bits, group, k, generator).rows(0, n)
            else:
                codes = torch.randint(1 << bits, (n, k), generator=generator)
                payload = {"weight_packed": _pack(codes, bits).view(torch.uint8)}
                if format_name == "cubic-int8":
                    metadata = "int8-scale-int4-ab-global-fp32"
                    payload.update(
                        weight_scale=torch.randint(0, 32, (n, k//group),
                                                   dtype=torch.int8, generator=generator),
                        weight_ab=torch.randint(0, 256, (n, k//group),
                                                dtype=torch.uint8, generator=generator),
                        weight_a_global=torch.tensor([.013]),
                        weight_b_global=torch.tensor([.005]),
                        weight_scale_global=torch.tensor([.001]),
                    )
                else:
                    metadata = "e5m9-scale-curve2-tensor-fp16"
                    scale = (torch.rand(n, k//group, generator=generator)*.02+.001).half()
                    curves = torch.randint(0, 4, (n, k//group), generator=generator)
                    payload.update(
                        weight_metadata=((scale.view(torch.int16).int() >> 1)
                                         | (curves << 14)).to(torch.uint16),
                        weight_curve_a=torch.tensor([[.9, 1.0, 1.1, 1.2]]).half(),
                        weight_curve_b=torch.tensor([[-.1, 0, .1, .2]]).half(),
                    )
                matrix = QuantizedMatrix(payload, QuantizationScheme(
                    "cubic-bitstream-v1", bits, group, metadata=metadata), (n, k), (n,))
            matrices.append(matrix)
        prepared = tuple((ip if format_name == "int" else cp)(operators, w, device)
                         for w in matrices)
        assert all(w is not None for w in prepared)
        corpus = torch.randn(4, k, generator=generator).to(device=device, dtype=dtype)
        first_row = None
        for physical, logical in ((1, 1), (2, 2), (4, 3), (8, 4), (16, 4), (32, 4)):
            inputs = torch.full((physical, k), float("nan"), device=device, dtype=dtype)
            inputs[:logical].copy_(corpus[:logical])
            outputs = tuple(torch.empty(physical, n, device=device, dtype=dtype) for _ in matrices)
            binding = mma_bind(inputs, prepared, outputs)
            binding.launch()
            torch.cuda.synchronize(device)
            for matrix, output in zip(matrices, outputs, strict=True):
                check_linear_enclosure(inputs[:logical], matrix, output[:logical],
                                       compute_mode="wna16")
            if first_row is not None:
                assert all(torch.equal(y[:1].view(torch.uint8), old.view(torch.uint8))
                           for y, old in zip(outputs, first_row, strict=True))
            else:
                first_row = tuple(y[:1].clone() for y in outputs)
            assert all(torch.isnan(y[logical:]).all() for y in outputs)
            expected = tuple(y[:logical].clone() for y in outputs)
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    binding.launch()
                inputs[:logical].copy_(corpus[:logical].flip(0))
                graph.replay()
                torch.cuda.synchronize(device)
                assert all(torch.equal(y[:logical].view(torch.uint8), old.flip(0).view(torch.uint8))
                           for y, old in zip(outputs, expected, strict=True))
            finally:
                graph.reset()
