"""Original-row lookup and immutable host lifetime, not serving benchmarks."""

import numpy as np
import pytest
import torch
from gguf import GGML_QUANT_SIZES
from gguf import GGMLQuantizationType as Kind

from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.quantization.extensions.ggml.checkpoint import scheme
from cubic_serve.quantization.extensions.ggml.embedding import bind_lookup
from cubic_serve.quantization.extensions.ggml.reference import decode_rows
from cubic_serve.quantization.registry import extensions, register

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _matrix(kind, rows, *, host):
    from cubic_serve.quantization.extensions.ggml.extension import EXTENSION

    if not any(e.format_id == EXTENSION.format_id for e in extensions()):
        register(EXTENSION)
    block, size = GGML_QUANT_SIZES[kind]
    groups = 512 // block
    rng = np.random.default_rng(9501)
    raw = rng.integers(0, 256, (rows * groups, size), dtype=np.uint8)
    offset = 208 if kind == Kind.Q6_K else 0
    raw[:, offset:offset + 2] = rng.uniform(-0.04, 0.04, raw.shape[0]).astype(
        "<f2").view(np.uint8).reshape(-1, 2)
    raw = raw.reshape(rows, groups * size)
    tensor = torch.from_numpy(raw).pin_memory() if host else torch.from_numpy(raw).cuda()
    return QuantizedMatrix({"weight_packed": tensor}, scheme(int(kind)), (rows, 512)), raw


@pytest.mark.parametrize("kind", [Kind.Q8_0, Kind.Q6_K, Kind.IQ3_S, Kind.IQ4_NL, Kind.IQ4_XS])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("host", [False, True])
def test_shard_owner_gap_and_inactive_rows(kind, dtype, host):
    first, raw_first = _matrix(kind, 31, host=host)
    second, raw_second = _matrix(kind, 11, host=host)
    ids = torch.tensor([[3, 44], [30, 35], [51, 6]], device="cuda", dtype=torch.int64)
    activity = torch.tensor([1, 0, 1], device="cuda", dtype=torch.int32)
    scale = torch.tensor([0.75], device="cuda", dtype=torch.float32)
    out = torch.full((3, 1024), 23, device="cuda", dtype=dtype)
    binding = bind_lookup(ids, ((0, first), (40, second)), out, scale=scale, activity=activity)
    binding.launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        binding.launch()
    out.fill_(23)
    graph.replay()
    torch.cuda.synchronize()
    a = torch.from_numpy(decode_rows(raw_first, int(kind), 512))
    b = torch.from_numpy(decode_rows(raw_second, int(kind), 512))
    expected = torch.full((3, 2, 512), 23, dtype=dtype)
    expected[0, 0] = (a[3] * 0.75).to(dtype)
    expected[0, 1] = (b[4] * 0.75).to(dtype)
    expected[2, 0] = 0
    expected[2, 1] = (a[6] * 0.75).to(dtype)
    assert torch.equal(out.cpu(), expected.reshape(3, 1024))


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_plain_embedding_uses_existing_integer_id_abi(id_dtype):
    matrix, raw = _matrix(Kind.Q8_0, 8, host=False)
    ids = torch.tensor([0, 7, 2], dtype=id_dtype, device="cuda")
    out = torch.empty((3, 512), dtype=torch.bfloat16, device="cuda")
    binding = bind_lookup(ids, ((0, matrix),), out)
    binding.launch()
    torch.cuda.synchronize()
    reference = torch.from_numpy(decode_rows(raw, int(Kind.Q8_0), 512))[[0, 7, 2]].bfloat16()
    assert torch.equal(out.cpu(), reference)
