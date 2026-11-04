"""Independent format reference; never calls candidate CUDA kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from cubic_serve.models.quantization import QuantizationScheme

from collections.abc import Mapping

import torch
from torch import Tensor

from cubic_serve.models.quantization import (
    QuantizationError,
)
from cubic_serve.quantization.common.reference import (
    _positive_finite,
    _tensor,
)


def decode_awq(payload: Mapping[str, Tensor], shape: tuple[int, int], group_in: int) -> Tensor:
    """Return FP32 [N,K]; packed GEMM words store eight reordered N nibbles."""
    n, k = shape
    if n % 8 or k % group_in:
        raise QuantizationError("AWQ physical matrix must close packing and input groups")
    weight = _tensor(payload, "qweight", torch.int32, (k, n // 8)).to(torch.int64)
    zeros = _tensor(payload, "qzeros", torch.int32, (k // group_in, n // 8)).to(torch.int64)
    scales = _tensor(payload, "scales", torch.float16, (k // group_in, n)).float()
    _positive_finite(scales, "AWQ scales", zero=True)
    order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=weight.device) * 4
    codes = ((weight[..., None] >> order) & 15).reshape(k, n).float()
    zero = ((zeros[..., None] >> order) & 15).reshape(k // group_in, n).float()
    groups = torch.arange(k, device=weight.device) // group_in
    return ((codes - zero[groups]) * scales[groups]).t().contiguous()


def decode(
    payload: Mapping[str, Tensor],
    encoding: QuantizationScheme,
    shape: tuple[int, int],
    partitions: tuple[int, ...] | None,
) -> Tensor:
    return decode_awq(payload, shape, encoding.group_in)
