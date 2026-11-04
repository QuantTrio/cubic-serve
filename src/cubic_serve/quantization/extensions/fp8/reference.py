"""Independent format reference; never calls candidate CUDA kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from cubic_serve.models.quantization import QuantizationScheme
    from cubic_serve.quantization.common.reference import ActivationQuantization

from collections.abc import Mapping

import torch
from torch import Tensor

from cubic_serve.models.quantization import (
    QuantizationError,
)
from cubic_serve.quantization.common.reference import (
    ActivationQuantization,
    _positive_finite,
    _tensor,
)


def decode_fp8(payload: Mapping[str, Tensor], shape: tuple[int, int]) -> Tensor:
    weight = _tensor(payload, "weight", torch.float8_e4m3fn, shape).float()
    scale = _tensor(payload, "weight_scale", torch.float32, (shape[0], 1))
    _positive_finite(scale, "FP8 weight scale", zero=True)
    if not bool(torch.isfinite(weight).all()):
        raise QuantizationError("FP8 checkpoint contains nonfinite weights")
    return weight * scale


def decode(
    payload: Mapping[str, Tensor],
    encoding: QuantizationScheme,
    shape: tuple[int, int],
    partitions: tuple[int, ...] | None,
) -> Tensor:
    return decode_fp8(payload, shape)


def quantize(
    inputs: Tensor, encoding: QuantizationScheme, global_inverse: Tensor | None
) -> ActivationQuantization:
    values = inputs.float()
    # CT dynamic-token E4M3: minimum scale matches its bounded reciprocal.
    scales = (
        (values.abs().amax(dim=-1, keepdim=True).double() / 448.0)
        .float()
        .clamp_min(1.0 / (448.0 * 512.0))
    )
    codes = (values.double() / scales.double()).float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return ActivationQuantization(codes, scales, codes.float() * scales)
