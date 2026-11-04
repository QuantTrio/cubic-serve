"""Dense A8 fragment transport and strict FP32 fold, independent of a model."""
import os
from dataclasses import replace

import pytest
import torch

from cubic_serve.quantization.common.marlin import cost
from cubic_serve.quantization.common.marlin.a8_partial_reduce import bind, source_identity


def test_native_reducer_identity_does_not_change_existing_plan_identity():
    spec = cost.ProbeFormat(8, torch.bfloat16, torch.bfloat16, torch.float16, 0, 1., 1.)
    assert "native_dense_reduce" not in cost._calibration_identity(spec)
    assert cost._reduction_identity(spec) == cost._calibration_implementation_identity()
    native = replace(spec, activation_dtype=torch.int8, dense_linear=True, native_dense_reduce=True)
    assert cost._calibration_identity(native)["native_dense_reduce"] is True
    assert cost._reduction_identity(native) == source_identity()
    assert cost._reduction_identity(native) != cost._reduction_identity(spec)


def _encode_fragments(terms, rows, capacity, block_m, block_n, device):
    """Invert the native lane/register layout, independently of the CUDA address expression."""
    partitions, _, n = terms.shape
    local = torch.arange(block_m * block_n)
    lane = local.div(4, rounding_mode="floor") % block_n
    fragment = local.div(4 * block_n, rounding_mode="floor") * 4 + local % 4
    r = (fragment // 16) * 16 + ((fragment % 4) // 2) * 8 + (lane % 32) // 4
    c = (lane // 32) * 32 + ((fragment % 16) // 8) * 16
    c += ((fragment % 8) // 4) * 8 + (lane % 4) * 2 + fragment % 2
    tiles_n = n // block_n
    data = torch.full((((capacity + block_m - 1) // block_m) * tiles_n
                       * partitions * block_m * block_n,), float("nan"), device=device)
    for logical, physical in enumerate(rows):
        tile_m, inner_row = divmod(physical, block_m)
        selected = r == inner_row
        for tile_n in range(tiles_n):
            indices = ((tile_m * tiles_n + tile_n) * partitions
                       + torch.arange(partitions)[:, None]) * block_m * block_n + local[selected]
            values = terms[:, logical, tile_n * block_n + c[selected]]
            data[indices.flatten().to(device)] = values.flatten().to(device)
    return data


@pytest.mark.skipif(os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA check")
@pytest.mark.parametrize("capacity,n,bm,bn,partitions", [
    *((m, 320, 16 if m < 32 else 32 if m == 32 else 64, 64, 40)
      for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)),
    (16, 384, 16, 128, 20), (4, 2560, 16, 64, 3),
    (4, 320, 16, 64, 1), (4, 320, 16, 64, 64),
])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_partial_fold_independent_fp32_oracle_and_row_permutations(capacity, n, bm, bn, partitions, dtype):
    device = torch.device("cuda", 0)
    count = min(3, capacity)
    terms = torch.randn(partitions, 3, n, generator=torch.Generator().manual_seed(761))
    expected = torch.zeros(count, n)
    for term in terms[:, :count]:
        expected.add_(term)
    expected = expected.to(dtype)
    mappings = [list(range(count))]
    if capacity > 1:
        mappings.append([capacity - 1, 0, capacity // 2][:count])
    for rows in mappings:
        data = _encode_fragments(terms[:, :count], rows, capacity, bm, bn, device)
        before = data.view(torch.int32).clone()
        storage = torch.full((capacity * n + 32,), 17., device=device, dtype=dtype)
        output = storage[:capacity * n].view(capacity, n)
        operation = bind(data, output, m=capacity, n=n, block_m=bm, block_n=bn, partitions=partitions)
        operation.launch()
        torch.cuda.synchronize(device)
        bits = torch.int32 if dtype == torch.float32 else torch.int16
        assert torch.equal(output[rows].cpu().view(bits), expected.view(bits))
        assert torch.equal(data.view(torch.int32), before)
        assert bool((storage[capacity * n:] == 17).all())


@pytest.mark.skipif(os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA check")
@pytest.mark.parametrize("bits,expected", [(1, 3), (-2147483648, 0)])
def test_partial_fold_preserves_subnormals_and_signed_zero(bits, expected):
    device = torch.device("cuda", 0)
    terms = torch.full((3, 1, 64), bits, dtype=torch.int32).view(torch.float32)
    data = _encode_fragments(terms, [0], 1, 16, 64, device)
    output = torch.empty(1, 64, device=device)
    bind(data, output, m=1, n=64, block_m=16, block_n=64, partitions=3).launch()
    torch.cuda.synchronize(device)
    assert bool((output.view(torch.int32).cpu() == expected).all())
