"""Independent CPU evaluation of original GGML weight blocks.

This is reference decoding, never a performance fallback. Coordinate-based
indexing follows ggml-quants.c at llama.cpp 74bca3950054eed1795f525cb9cf06bfe796492d.
Only the immutable IQ3_S codebook is shared with the pinned upstream package;
none of its dequantization routines evaluate this reference.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType

from cubic_serve.models.quantization import QuantizationError

from .encoding import iq3_grid

if TYPE_CHECKING:
    from torch import Tensor

    from cubic_serve.models.quantization import QuantizationScheme
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix

_IQ4_VALUES = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=np.float32,
)


def _half(blocks: np.ndarray, offset: int) -> np.ndarray:
    return blocks[:, offset:offset + 2].copy().view("<f2").astype(np.float32)


def decode_rows(payload: np.ndarray, type_id: int, columns: int) -> np.ndarray:
    """Decode complete original rows to FP32 without a weight-rounding boundary."""
    kind = GGMLQuantizationType(type_id)
    block, size = GGML_QUANT_SIZES[kind]
    if type(columns) is not int or columns <= 0 or columns % block:
        raise QuantizationError("GGML reference requires complete positive row blocks")
    if (payload.dtype != np.uint8 or payload.ndim != 2 or payload.shape[0] == 0
            or payload.shape[1] != columns // block * size):
        raise QuantizationError("GGML payload bytes disagree with logical row geometry")
    raw = np.ascontiguousarray(payload)
    rows = raw.shape[0]
    if kind in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
        return raw.view("<f4" if kind == GGMLQuantizationType.F32 else "<f2").astype(
            np.float32, copy=True
        ).reshape(rows, columns)
    if kind == GGMLQuantizationType.BF16:
        bits = raw.view("<u2").astype(np.uint32) << 16
        return bits.view(np.float32).reshape(rows, columns)
    blocks = raw.reshape(-1, size)
    # Supported block coordinates fit in 16 bits. Avoid promoting every byte
    # lookup/shift to an eight-byte temporary; FP32 arithmetic is unchanged.
    position = np.arange(block, dtype=np.uint16)
    if kind == GGMLQuantizationType.Q8_0:
        result = _half(blocks, 0) * blocks[:, 2:].view(np.int8).astype(np.float32)
    elif kind == GGMLQuantizationType.Q6_K:
        # Decode one 32-value quarter of each 128-value half at a time.
        # Broadcasting the two original 16-value scales avoids expanding a
        # full FP32 scale matrix. Keep both original FP32 products separate.
        count = blocks.shape[0]
        ql = blocks[:, :128].reshape(count, 2, 64)
        qh = blocks[:, 128:192].reshape(count, 2, 32)
        scales = blocks[:, 192:208].view(np.int8).reshape(count, 2, 8)
        d = _half(blocks, 208).reshape(count, 1, 1, 1)
        result = np.empty((count, 2, 4, 32), dtype=np.float32)
        for quarter in range(4):
            start = (quarter % 2) * 32
            low = (ql[:, :, start:start + 32] >> (4 * (quarter // 2))) & 15
            high = (qh >> (2 * quarter)) & 3
            codes = low.astype(np.int16) + (high.astype(np.int16) << 4) - 32
            group_scale = d * scales[:, :, 2 * quarter:2 * quarter + 2, None].astype(np.float32)
            result[:, :, quarter, :] = (
                group_scale * codes.astype(np.float32).reshape(count, 2, 2, 16)
            ).reshape(count, 2, 32)
    elif kind == GGMLQuantizationType.IQ4_NL:
        codes = (blocks[:, 2 + position % 16] >> (4 * (position // 16))) & 15
        result = _half(blocks, 0) * _IQ4_VALUES[codes]
    elif kind == GGMLQuantizationType.IQ4_XS:
        group = position // 32
        low = (blocks[:, 4 + group // 2] >> (4 * (group % 2))) & 15
        high_word = blocks[:, 2:4].copy().view("<u2")
        high = (high_word >> (2 * group)) & 3
        scales = (low.astype(np.int16) | (high.astype(np.int16) << 4)) - 32
        codes = (blocks[:, 8 + group * 16 + position % 16]
                 >> (4 * (position % 32 // 16))) & 15
        result = (_half(blocks, 0) * scales.astype(np.float32)) * _IQ4_VALUES[codes]
    elif kind == GGMLQuantizationType.IQ3_S:
        # Gather each four-value grid once, rather than repeating its byte
        # decoding four times. Keep the original d*scale, grid, sign FP32
        # multiplication boundaries for every logical coordinate.
        vector = np.arange(64, dtype=np.uint16)
        low = blocks[:, 2:66].astype(np.uint16)
        high = (blocks[:, 66 + vector // 8] >> (vector % 8)) & 1
        grid = iq3_grid()[low | (high.astype(np.uint16) << 8)].reshape(-1, 8, 8, 4)
        group = np.arange(8, dtype=np.uint16)
        scale = (blocks[:, 106 + group // 2] >> (4 * (group % 2))) & 15
        group_scale = _half(blocks, 0) * (1 + 2 * scale).astype(np.float32)
        values = (group_scale[:, :, None, None] * grid).reshape(-1, 256)
        sign_bit = np.unpackbits(blocks[:, 74:106], axis=1, bitorder="little")
        result = values * (1 - 2 * sign_bit.astype(np.int8))
    else:
        raise QuantizationError(f"GGML independent reference does not yet implement {kind.name}")
    return result.reshape(rows, columns)


def decode(payload: Mapping[str, Tensor], encoding: QuantizationScheme,
           shape: tuple[int, int], partitions: tuple[int, ...] | None = None) -> Tensor:
    """Existing independent weight-reference contract, preserving its device."""
    import torch

    from .checkpoint import column_offset, kind

    original = payload["weight_packed"]
    block, size = GGML_QUANT_SIZES[kind(encoding)]
    decoded = decode_rows(original.detach().cpu().numpy(), int(kind(encoding)),
                          original.shape[1] // size * block)
    offset = column_offset(encoding)
    # decode_rows always creates owned FP32 storage. Preserve compact ownership
    # for sliced columns, but do not copy a second full-span decoded matrix.
    if offset != 0 or shape[1] != decoded.shape[1] or not decoded.flags.c_contiguous:
        decoded = decoded[:, offset:offset + shape[1]].copy()
    return torch.from_numpy(decoded).to(original.device)


def wna8_operand(inputs: Tensor) -> Tensor:
    """Independent original Q8_1 groups: FP32 codes, then FP16 scale storage."""
    import torch

    if inputs.shape[-1] % 32:
        raise QuantizationError("GGML Q8_1 requires complete 32-value scale groups")
    values = inputs.float().reshape(*inputs.shape[:-1], -1, 32)
    # The native upstream contract uses div.rn.f32 twice. Torch's scalar
    # division can instead multiply by a rounded reciprocal, changing codes
    # at half-integer boundaries. Round independent FP64 quotients to FP32.
    scale = (values.abs().amax(-1, keepdim=True).double() / 127.0).float()
    denominator = torch.where(scale > 0, scale, 1.0)
    ratio = (values.double() / denominator.double()).float()
    # llama.cpp roundf is ties-away-from-zero, not Torch round's ties-to-even.
    magnitude = ratio.abs()
    integral = torch.floor(magnitude)
    codes = torch.copysign(integral + (magnitude - integral >= 0.5), ratio)
    return (codes * scale.to(torch.float16).float()).reshape(inputs.shape)


def wna8_weight(matrix: QuantizedMatrix) -> Tensor:
    return decode(matrix.payload, matrix.encoding, matrix.shape, matrix.partitions)


def wna8_linear(inputs: Tensor, matrix: QuantizedMatrix) -> Tensor:
    from cubic_serve.ops.eager.linear import linear

    return linear(wna8_operand(inputs), wna8_weight(matrix), output_dtype=inputs.dtype)


def prepare_wna8_linear(inputs: Tensor, matrix: QuantizedMatrix, *,
                        output_rows: int, byte_budget: int):
    """Reuse one activation producer while streaming independently decoded rows.

    Decoding batches do not change canonical output-tile GEMM shapes. The
    conservative workspace model includes encoded/decoded coordinate temporaries,
    the retained FP32 activation and output storage; nothing survives this call.
    """
    import torch

    from cubic_serve.ops.eager.linear import linear

    retained = inputs.numel() * 4
    input_rows = inputs.numel() // matrix.shape[1]
    per_row = 64 * matrix.shape[1] + 32 * input_rows
    rows = (byte_budget - retained) // per_row
    if rows < output_rows:
        return None
    rows = rows // output_rows * output_rows
    activation = wna8_operand(inputs)

    def apply_part(part: QuantizedMatrix) -> Tensor:
        weight = wna8_weight(part)
        return torch.cat([
            linear(activation, weight[start:start + output_rows], output_dtype=inputs.dtype)
            for start in range(0, weight.shape[0], output_rows)
        ], dim=-1)

    return rows, apply_part
