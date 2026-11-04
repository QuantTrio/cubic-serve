"""Host audit classification preserves every storage encoding and byte identity."""

import pytest
import torch

from cubic_serve.runtime.performance import _exact_snapshot, _snapshot_is_finite


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_every_16_bit_encoding_against_torch_finite_oracle(dtype):
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype)
    finite = torch.isfinite(values)
    assert _snapshot_is_finite(values[finite])
    assert not _snapshot_is_finite(values)
    for invalid in values[~finite]:
        assert not _snapshot_is_finite(invalid)
    assert _snapshot_is_finite(values[:0])
    assert _snapshot_is_finite(values[finite][:128].reshape(16, 8).T)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_snapshot_requires_finite_values_and_exact_storage(dtype):
    value = torch.tensor([[0.0, -0.0], [1.0, -1.0]], dtype=dtype).T
    assert _exact_snapshot({"state": value}, {"state": value.clone()})
    other = value.clone()
    other[0, 0] = -0.0
    assert not _exact_snapshot({"state": value}, {"state": other})
    for invalid in (float("nan"), float("inf"), -float("inf")):
        other.fill_(invalid)
        assert not _exact_snapshot({"state": other}, {"state": other.clone()})
    assert not _exact_snapshot({"state": value}, {"state": value.flatten()})
    assert not _exact_snapshot({"state": value}, {"other": value})
    assert _snapshot_is_finite(value.clone().requires_grad_())


def test_snapshot_keeps_integer_ownership_exact():
    value = torch.tensor([0, 2**62 + 1], dtype=torch.int64)
    assert _exact_snapshot({"owners": value}, {"owners": value.clone()})
    other = value.clone()
    other[1] += 1
    assert not _exact_snapshot({"owners": value}, {"owners": other})
    assert not _exact_snapshot({"owners": value}, {"owners": value.to(torch.float64)})
