"""Independent format reference; never calls candidate CUDA kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from cubic_serve.models.quantization import QuantizationScheme

import math
from collections.abc import Mapping

import torch
from torch import Tensor

from cubic_serve.models.quantization import (
    QuantizationError,
    QuantizationScheme,
)
from cubic_serve.quantization.common.reference import (
    _positive_finite,
    _tensor,
)
from cubic_serve.quantization.common.reference import wna8_activation as wna8_operand


def unpack_cubic(packed: Tensor, *, bits: int, input_size: int) -> Tensor:
    if packed.dtype is not torch.uint8 or packed.ndim != 2 or not 1 <= bits <= 8 or input_size <= 0:
        raise QuantizationError("Cubic bitstream requires uint8 rows and a valid bit width")
    required = math.ceil(input_size * bits / 8)
    if packed.shape[1] != required:
        raise QuantizationError("Cubic packed row length does not match logical K")
    words = packed.to(torch.int32)
    positions = torch.arange(input_size, device=packed.device) * bits
    byte, shift = positions // 8, positions % 8
    low = words[:, byte] >> shift
    high = words[:, (byte + 1).clamp(max=required - 1)] << (8 - shift)
    raw = torch.where(shift + bits > 8, low | high, low) & ((1 << bits) - 1)
    if bits == 1:
        return (raw * 2 - 1).to(torch.int8)
    sign = 1 << (bits - 1)
    signed = torch.where(raw >= sign, raw - (1 << bits), raw)
    return torch.where(signed == -sign, 0, signed).to(torch.int8)


def cubic_metadata(
    payload: Mapping[str, Tensor],
    encoding: QuantizationScheme,
    shape: tuple[int, int],
    partitions: tuple[int, ...],
) -> tuple[Tensor, Tensor, Tensor]:
    n, k = shape
    go, gi = encoding.group_out, encoding.group_in
    if not partitions or sum(partitions) != n or any(p <= 0 or p % go for p in partitions):
        raise QuantizationError("Cubic output partitions must preserve output-group boundaries")
    groups = (n // go, math.ceil(k / gi))
    if encoding.metadata == "float32-scale-float16-ab":
        return (
            _tensor(payload, "weight_scale", torch.float32, groups),
            _tensor(payload, "weight_a", torch.float16, groups),
            _tensor(payload, "weight_b", torch.float16, groups),
        )
    if encoding.metadata == "int8-scale-int4-ab-global-fp32":
        scale = _tensor(payload, "weight_scale", torch.int8, groups)
        ab = _tensor(payload, "weight_ab", torch.uint8, groups).to(torch.int32)
        counts = torch.tensor([p // go for p in partitions], device=scale.device)

        def expand(name: str) -> Tensor:
            value = _tensor(payload, name, torch.float32, (len(partitions),))
            if not bool(torch.isfinite(value).all()):
                raise QuantizationError("Cubic global metadata must be finite")
            return torch.repeat_interleave(value, counts)[:, None]

        low, high = ab & 15, ab >> 4
        a = torch.where(low >= 8, low - 16, low).float()
        b = torch.where(high >= 8, high - 16, high).float()
        return (
            scale.float() * expand("weight_scale_global"),
            (1.0 + a * expand("weight_a_global")).half(),
            (b * expand("weight_b_global")).half(),
        )
    metadata = _tensor(payload, "weight_metadata", torch.uint16, groups).to(torch.int32)
    a = _tensor(payload, "weight_curve_a", torch.float16, (len(partitions), 4))
    b = _tensor(payload, "weight_curve_b", torch.float16, (len(partitions), 4))
    scale = ((metadata & 0x3FFF) << 1).to(torch.int16).view(torch.float16).float()
    part = torch.repeat_interleave(
        torch.arange(len(partitions), device=a.device),
        torch.tensor([p // go for p in partitions], device=a.device),
    )[:, None]
    curve = (metadata >> 14).long()
    return scale, a[part, curve], b[part, curve]


def decode_cubic(
    payload: Mapping[str, Tensor],
    encoding: QuantizationScheme,
    shape: tuple[int, int],
    partitions: tuple[int, ...],
) -> Tensor:
    codes = unpack_cubic(
        _tensor(payload, "weight_packed", torch.uint8), bits=encoding.bits, input_size=shape[1]
    )
    if codes.shape[0] != shape[0]:
        raise QuantizationError("Cubic packed output rows differ from logical N")
    scale, a, b = cubic_metadata(payload, encoding, shape, partitions)
    _positive_finite(scale, "Cubic decoded scale", zero=True)
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        raise QuantizationError("Cubic curve coefficients must be finite")
    out_group = torch.arange(shape[0], device=codes.device) // encoding.group_out
    in_group = torch.arange(shape[1], device=codes.device) // encoding.group_in
    index = (out_group[:, None], in_group[None, :])
    s, aa, bb = scale[index], a[index].float(), b[index].float()
    values = codes.float()
    if encoding.bits == 1:
        return values * s
    t = (values.abs().double() / ((1 << (encoding.bits - 1)) - 1)).float()
    return values.sign() * (s * (t * (aa + t * (bb + t * (1.0 - aa - bb)))))


def decode(
    payload: Mapping[str, Tensor],
    encoding: QuantizationScheme,
    shape: tuple[int, int],
    partitions: tuple[int, ...] | None,
) -> Tensor:
    return decode_cubic(payload, encoding, shape, partitions or (shape[0],))


def wna8_activation(inputs: Tensor, encoding: QuantizationScheme) -> Tensor:
    """Independent symmetric per-token INT8 quantization, returned in FP32."""
    from cubic_serve.quantization.common.reference import wna8_activation as common_activation

    return common_activation(inputs)


def wna8_weight(matrix) -> Tensor:
    """Decode original Cubic groups, then independently requantize each output row."""
    encoding, shape, payload = matrix.encoding, matrix.shape, matrix.payload
    if not 2 <= encoding.bits <= 8:
        raise QuantizationError("WNA8 requires Cubic W2-W8")
    code = unpack_cubic(payload['weight_packed'], bits=encoding.bits, input_size=shape[1])
    scale, a, b = cubic_metadata(payload, encoding, shape, matrix.partitions or (shape[0],))
    rows = torch.arange(shape[0], device=code.device) // encoding.group_out
    columns = torch.arange(shape[1], device=code.device) // encoding.group_in
    index = rows[:, None], columns[None, :]
    aa, bb = a[index].float(), b[index].float()
    t = (code.abs().double() / ((1 << (encoding.bits - 1)) - 1)).float()
    f = t * (aa + t * (bb + t * ((1. - aa) - bb)))
    value = (scale[index] * f) * code.sign()
    row_scale = (value.abs().amax(-1, keepdim=True).double() / 127.).float()
    row_scale = torch.where(row_scale > 0, row_scale, 1.)
    carrier = (value.double() / row_scale.double()).float().round().clamp(-127, 127)
    return carrier * row_scale
