"""Encoded host transport keeps byte order, ownership and completion exact."""

import os

import pytest
import torch

from cubic_serve.quantization.common.binding import concatenate_to_device


@pytest.mark.parametrize("dim", [0, 1, -2])
def test_host_concatenation_retains_shape_dtype_and_noncontiguous_order(dim):
    parts = tuple(torch.arange(24, dtype=torch.int32).reshape(4, 6).T + i * 100
                  for i in range(3))
    actual = concatenate_to_device(parts, torch.device("cpu"), dim)
    assert torch.equal(actual, torch.cat(parts, dim=dim))
    assert actual.is_contiguous()
    promoted = concatenate_to_device((parts[0], parts[1].float()), torch.device("cpu"), dim)
    assert torch.equal(promoted, torch.cat((parts[0], parts[1].float()), dim=dim))


@pytest.mark.skipif(os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1",
                    reason="explicit CUDA transport audit required")
@pytest.mark.parametrize("count", [1, 3, 19])
@pytest.mark.parametrize("dtype", [torch.int32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("pinned", [False, True])
def test_cuda_host_sources_can_be_mutated_after_completed_transport(count, dtype, pinned):
    # Floating fixtures include arbitrary storage words, including NaNs. This
    # is a byte transport witness, so compare integer views without arithmetic.
    source = torch.arange(count * 4 * 9, dtype=torch.int32).reshape(count, 4, 9)
    if dtype != torch.int32:
        source = source.to(torch.int16).mul_(251).view(dtype)
    source = source.transpose(1, 2).contiguous().transpose(1, 2)
    if pinned:
        source = source.pin_memory()
    parts = tuple(source.unbind())
    expected = torch.cat(parts).contiguous().view(torch.uint8).clone()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = concatenate_to_device(parts, torch.device("cuda"))
    source.zero_()
    assert actual.is_contiguous()
    assert torch.equal(actual.cpu().view(torch.uint8), expected)


def test_scalar_concatenation_retains_torch_rejection():
    with pytest.raises(RuntimeError):
        concatenate_to_device((torch.tensor(1),), torch.device("cpu"))
