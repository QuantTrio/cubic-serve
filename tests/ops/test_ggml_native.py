"""Portable packed compute checks; no serving-throughput claim or startup corpus."""

import numpy as np
import pytest
import torch
from gguf import GGML_QUANT_SIZES
from gguf import GGMLQuantizationType as Kind

from cubic_serve.quantization.api import RoutedRows
from cubic_serve.quantization.extensions.ggml.native import bind_matmul
from cubic_serve.quantization.extensions.ggml.reference import decode_rows

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _weight(kind, rows, groups=2):
    block, size = GGML_QUANT_SIZES[kind]
    rng = np.random.default_rng(503)
    raw = rng.integers(0, 256, (rows * groups, size), dtype=np.uint8)
    offset = 208 if kind == Kind.Q6_K else 0
    raw[:, offset:offset + 2] = rng.uniform(-0.02, 0.02, raw.shape[0]).astype(
        "<f2"
    ).view(np.uint8).reshape(-1, 2)
    return raw.reshape(rows, groups * size), groups * block


def _enclosed(actual, x, weight):
    # The original weight rounds once to the MMA operand dtype. Its products
    # are exact FP32 for these finite normal fixtures. A conservative K+32
    # FP32 addition bound encloses the WMMA reduction before final rounding.
    left, right = x.double().cpu(), weight.to(x.dtype).double().cpu()
    exact = left @ right.T
    length = x.shape[1] + 32
    gamma = (length * 2**-24) / (1 - length * 2**-24)
    radius = gamma * (left.abs() @ right.abs().T)
    lower = (exact - radius).to(x.dtype).double()
    upper = (exact + radius).to(x.dtype).double()
    observed = actual.double().cpu()
    assert bool(((observed >= lower) & (observed <= upper)).all())


@pytest.mark.parametrize("kind", [Kind.Q8_0, Kind.Q6_K, Kind.IQ3_S, Kind.IQ4_NL, Kind.IQ4_XS])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("column_warps", [1, 4])
def test_cooperative_k_keeps_original_operands_and_private_padding(kind, dtype, column_warps):
    block, _ = GGML_QUANT_SIZES[kind]
    raw, physical_k = _weight(kind, 29, groups=(1027 + block - 1) // block)
    k = physical_k - 7
    weight = torch.from_numpy(decode_rows(raw, int(kind), physical_k))[:, :k]
    x = torch.randn((3, k), device="cuda", dtype=dtype)
    packed = torch.from_numpy(raw).cuda()
    output = torch.full((3, 29), 41, device="cuda", dtype=dtype)
    active = torch.tensor([1, 0, 1], device="cuda", dtype=torch.int32)
    binding = bind_matmul(x, packed, int(kind), output, activity=active,
                          vector_rows=1, column_warps=column_warps)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        binding.launch()
    graph.replay()
    torch.cuda.synchronize()
    _enclosed(output[[0, 2]], x[[0, 2]], weight)
    assert bool((output[1] == 41).all())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [1, 3, 17])
def test_q8_shared_integer_binding_and_group_keep_original_math(tmp_path, monkeypatch, dtype, m):
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
    from cubic_serve.ops.cuda.startup import startup_weight_identity
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.quantization.extensions.ggml.checkpoint import scheme
    from cubic_serve.ops.strategies import OperatorConfig

    monkeypatch.setenv("CUBIC_SERVE_CACHE_DIR", str(tmp_path))
    matrices, expected = [], []
    for n in (64, 128):
        raw, k = _weight(Kind.Q8_0, n, groups=8)
        scales = raw.reshape(n, 8, 34)[:, :, :2]
        scales[:] = np.frombuffer(np.float16(0.015625).tobytes(), dtype=np.uint8)
        matrices.append(QuantizedMatrix({"weight_packed": torch.from_numpy(raw)},
                                        scheme(int(Kind.Q8_0)), (n, k)))
        expected.append(torch.from_numpy(decode_rows(raw, int(Kind.Q8_0), k)))
    x = torch.randn((m, k), device="cuda", dtype=dtype)
    outputs = tuple(torch.empty((m, matrix.shape[0]), device="cuda", dtype=dtype)
                    for matrix in matrices)
    operators = PackedLinearOperators(OperatorConfig())
    with startup_weight_identity(("ggml-q8-cuda-test", m), allow_calibration=True):
        bindings = operators.bind_linear_groups(x, tuple(matrices), outputs)
    for binding in bindings:
        binding.launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for binding in bindings:
            binding.launch()
    graph.replay()
    torch.cuda.synchronize()
    for output, weight in zip(outputs, expected, strict=True):
        _enclosed(output, x, weight)


@pytest.mark.parametrize("kind,offset,k", [(Kind.IQ4_NL, 17, 80), (Kind.IQ3_S, 129, 383)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("vector_rows", [0, 1, 4])
def test_partial_original_blocks_and_unaligned_logical_tail(kind, offset, k, dtype, vector_rows):
    block, _ = GGML_QUANT_SIZES[kind]
    groups = (offset + k + block - 1) // block
    raw, physical_k = _weight(kind, 19, groups=groups)
    decoded = torch.from_numpy(decode_rows(raw, int(kind), physical_k))[:, offset:offset + k]
    x = torch.randn((3, k), device="cuda", dtype=dtype)
    output = torch.empty((3, 19), device="cuda", dtype=dtype)
    binding = bind_matmul(x, torch.from_numpy(raw).cuda(), int(kind), output,
                          vector_rows=vector_rows, column_offset=offset)
    binding.launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        binding.launch()
    graph.replay()
    torch.cuda.synchronize()
    _enclosed(output, x, decoded)


@pytest.mark.parametrize("kind", [Kind.Q8_0, Kind.Q6_K, Kind.IQ3_S, Kind.IQ4_NL, Kind.IQ4_XS])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [1, 3, 17])
@pytest.mark.parametrize("vector_rows", [0, 1, 4])
def test_original_blocks_dense_and_private_padding(kind, dtype, m, vector_rows):
    raw, k = _weight(kind, 37)
    weight = torch.from_numpy(decode_rows(raw, int(kind), k))
    x = torch.randn((m, k), device="cuda", dtype=dtype)
    packed = torch.from_numpy(raw).cuda()
    output = torch.full((m, 37), 23, device="cuda", dtype=dtype)
    active = torch.ones(m, device="cuda", dtype=torch.bool)
    binding = bind_matmul(x, packed, int(kind), output, activity=active, vector_rows=vector_rows)
    binding.launch()
    torch.cuda.synchronize()
    _enclosed(output, x, weight)
    # Replay uses the same bound addresses and explicit activity, never segment
    # starts. Poisoned inactive rows must retain their private sentinel.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        binding.launch()
    active[-1] = False
    output.fill_(23)
    graph.replay()
    torch.cuda.synchronize()
    assert bool((output[-1] == 23).all())
    if m > 1:
        _enclosed(output[:-1], x[:-1], weight)


@pytest.mark.parametrize("kind", [Kind.IQ3_S, Kind.IQ4_NL])
@pytest.mark.parametrize("vector_rows", [0, 1, 4])
def test_routed_expert_order_and_token_row_ownership(kind, vector_rows):
    raw, k = _weight(kind, 3 * 37)
    decoded = torch.from_numpy(decode_rows(raw, int(kind), k)).reshape(3, 37, k)
    x = torch.randn((3, k), device="cuda", dtype=torch.bfloat16)
    packed = torch.from_numpy(raw.reshape(3, 37, -1)).cuda()
    output = torch.full((6, 37), 29, device="cuda", dtype=x.dtype)
    def device(values):
        return torch.tensor(values, device="cuda", dtype=torch.int32)
    count = 2 if vector_rows == 1 else 1
    rows = RoutedRows(device([2, 2, 2]), device([0, 2, 4]), device([0, count, 2 * count]),
                      device([3 * count]), device([0, 3, 2, 5, 1, 4]))
    activity = torch.tensor([True, False, True], device="cuda")
    binding = bind_matmul(x, packed, int(kind), output, rows=rows, input_top_k=2,
                          activity=activity, vector_rows=vector_rows)
    binding.launch()
    torch.cuda.synchronize()
    for route, expert in enumerate([0, 2, 1, 0, 2, 1]):
        if route // 2 == 1:
            assert bool((output[route] == 29).all())
        else:
            _enclosed(output[route:route + 1], x[route // 2:route // 2 + 1], decoded[expert])
