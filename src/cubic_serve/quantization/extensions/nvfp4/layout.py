"""Physical payload headers and original ownership-preserving slice coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cubic_serve.models.quantization import QuantizationScheme
    from cubic_serve.models.quantized_checkpoint import LogicalWeight

from cubic_serve.models.quantization import (
    QuantizationError,
)

ROW_ALIGNMENT = 1
CONCAT_AXIS = 0
SCALAR_PAYLOADS = ("weight_global_scale", "input_global_scale", "weight_scale_2", "input_scale")


def payload_names(encoding: QuantizationScheme) -> tuple[str, str, str, str]:
    """Packed values, block scale, input global, weight global in ABI order."""
    if encoding.metadata == "modelopt":
        return "weight", "weight_scale", "input_scale", "weight_scale_2"
    return "weight_packed", "weight_scale", "input_global_scale", "weight_global_scale"


def payload_headers(
    weight: LogicalWeight, encoding: QuantizationScheme
) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    n, k = weight.shape
    if k % 16:
        raise QuantizationError("NVFP4 logical K does not close groups of 16")
    if encoding.metadata == "modelopt":
        return (
            ("weight", "U8", (n, k // 2)),
            ("weight_scale", "F8_E4M3", (n, k // 16)),
            ("weight_scale_2", "F32", ()),
            ("input_scale", "F32", ()),
        )
    result = [
        ("weight_packed", "U8", (n, k // 2)),
        ("weight_scale", "F8_E4M3", (n, k // 16)),
        ("weight_global_scale", "F32", (1,)),
    ]
    # CT stores this tensor even when an embedding has no activation input.
    result.append(("input_global_scale", "F32", (1,)))
    return tuple(result)


def slice_index(
    shape: tuple[int, ...],
    partitions: tuple[int, ...],
    encoding: QuantizationScheme,
    key: str,
    axis: int,
    start: int,
    end: int,
) -> tuple[tuple[slice, ...], tuple[int, ...]]:
    scalar = encoding.metadata == "modelopt" and key in ("weight_scale_2", "input_scale")
    if encoding.metadata == "modelopt" and key == "weight":
        key = "weight_packed"
    full = slice(None)
    if axis == 0:
        group = encoding.group_out
        if start % group or end % group:
            raise QuantizationError("output slice must preserve original scale groups")
        parts, indices, offset = [], [], 0
        for index, count in enumerate(partitions or (shape[0],)):
            overlap = min(end, offset + count) - max(start, offset)
            if overlap > 0:
                parts.append(overlap)
                indices.append(index)
            offset += count
        selected = tuple(parts)
        if scalar:
            return (), selected
        if key in ("weight_global_scale", "input_global_scale"):
            return (full,), selected
        if key.endswith("_global") or key in ("weight_curve_a", "weight_curve_b"):
            return (slice(indices[0], indices[-1] + 1),), selected
        if key in ("weight", "weight_packed"):
            return (slice(start, end), full), selected
        return (slice(start // group, end // group), full), selected
    group = encoding.group_in
    if start % group or (end % group and end != shape[1]):
        raise QuantizationError("input slice must preserve original scale groups")
    if scalar:
        return (), partitions
    if (
        key in ("weight_global_scale", "input_global_scale")
        or key.endswith("_global")
        or key in ("weight_curve_a", "weight_curve_b")
    ):
        return (full,), partitions
    if key == "weight_packed":
        if start * encoding.bits % 8:
            raise QuantizationError("input slice must preserve original packed bytes")
        return (full, slice(start * encoding.bits // 8, (end * encoding.bits + 7) // 8)), partitions
    return (full, slice(start // group, (end + group - 1) // group)), partitions
