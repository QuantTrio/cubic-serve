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


def unpack_fp4(packed: Tensor, shape: tuple[int, int]) -> Tensor:
    n, k = shape
    if k % 2 or packed.dtype is not torch.uint8 or tuple(packed.shape) != (n, k // 2):
        raise QuantizationError("NVFP4 packed rows require contiguous nibble pairs")
    words = packed.to(torch.int32)
    codes = torch.stack((words & 15, words >> 4), dim=-1).reshape(n, k)
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=packed.device)
    return levels[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)


def decode_nvfp4(payload: Mapping[str, Tensor], shape: tuple[int, int]) -> Tensor:
    n, k = shape
    if k % 16:
        raise QuantizationError("NVFP4 input groups must contain 16 values")
    packed = _tensor(payload, "weight_packed", torch.uint8, (n, k // 2))
    scale = _tensor(payload, "weight_scale", torch.float8_e4m3fn, (n, k // 16)).float()
    global_inverse = _tensor(payload, "weight_global_scale", torch.float32, (1,))
    _positive_finite(global_inverse, "NVFP4 global inverse scale")
    _positive_finite(scale, "NVFP4 block scales", zero=True)
    values = unpack_fp4(packed, shape).reshape(n, k // 16, 16)
    return (values * scale[..., None] / global_inverse).reshape(n, k)


def decode(
    payload: Mapping[str, Tensor],
    encoding: QuantizationScheme,
    shape: tuple[int, int],
    partitions: tuple[int, ...] | None,
) -> Tensor:
    if encoding.metadata == "modelopt":
        n, k = shape
        packed = _tensor(payload, "weight", torch.uint8, (n, k // 2))
        scale = _tensor(payload, "weight_scale", torch.float8_e4m3fn, (n, k // 16)).float()
        global_scale = _tensor(payload, "weight_scale_2", torch.float32, ())
        _positive_finite(global_scale, "ModelOpt weight global scale")
        _positive_finite(scale, "ModelOpt block scales", zero=True)
        values = unpack_fp4(packed, shape).reshape(n, k // 16, 16)
        return (values * scale[..., None] * global_scale).reshape(n, k)
    return decode_nvfp4(payload, shape)


def quantize(
    inputs: Tensor, encoding: QuantizationScheme, global_inverse: Tensor | None
) -> ActivationQuantization:
    values = inputs.float()
    if inputs.shape[-1] % 16:
        raise QuantizationError("NVFP4 activation requires complete groups of 16")
    if (
        global_inverse is None
        or global_inverse.numel() != 1
        or global_inverse.dtype is not torch.float32
        or global_inverse.device != inputs.device
    ):
        raise QuantizationError("NVFP4 activation requires its scalar global inverse scale")
    _positive_finite(global_inverse, "NVFP4 input global inverse scale")
    grouped = values.reshape(*values.shape[:-1], -1, 16)
    scales = (
        ((grouped.abs().amax(dim=-1, keepdim=True) * global_inverse).double() / 6.0)
        .float()
        .clamp(max=448.0)
        .to(torch.float8_e4m3fn)
    )
    decoded_scale = scales.float()
    safe_scale = torch.where(decoded_scale > 0, decoded_scale, 1.0)
    scaled = (
        ((grouped * global_inverse).double() / safe_scale.double()).float().abs().clamp(max=6.0)
    )
    levels = values.new_tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    distances = (scaled[..., None] - levels).abs()
    nearest = distances.argmin(dim=-1)
    upper = (nearest + 1).clamp(max=7)
    tie = distances.gather(-1, upper[..., None]).squeeze(-1) == distances.gather(
        -1, nearest[..., None]
    ).squeeze(-1)
    nearest = torch.where(tie & ((nearest & 1) == 1), upper, nearest)
    codes = (nearest | (torch.signbit(grouped).to(torch.int64) << 3)).to(torch.uint8)
    codes = torch.where(decoded_scale > 0, codes, 0)
    rounded = levels[codes.long() & 7] * torch.where(codes >= 8, -1.0, 1.0)
    return ActivationQuantization(
        codes.reshape(values.shape),
        scales.squeeze(-1),
        (rounded * decoded_scale / global_inverse).reshape(values.shape),
    )


def activation_from_payload(inputs, payload, encoding):
    if not bool(torch.isfinite(inputs).all()):
        raise QuantizationError("activation quantization requires finite operands")
    if encoding.activation == "identity":
        return ActivationQuantization(
            inputs, inputs.new_ones((), dtype=torch.float32), inputs.float()
        )
    if encoding.metadata != "modelopt":
        return quantize(inputs, encoding, payload.get("input_global_scale"))
    scale = _tensor(payload, "input_scale", torch.float32, ())
    _positive_finite(scale, "ModelOpt input global scale")
    quantized = quantize(inputs, encoding, scale.reciprocal())
    codes = quantized.codes.reshape(*inputs.shape[:-1], -1, 16)
    levels = inputs.new_tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
    values = levels[codes.long() & 7] * torch.where(codes >= 8, -1.0, 1.0)
    decoded = values * quantized.scales.float()[..., None] * scale
    return ActivationQuantization(quantized.codes, quantized.scales, decoded.reshape(inputs.shape))
