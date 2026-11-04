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
SCALAR_PAYLOADS = ()


def payload_headers(
    weight: LogicalWeight, encoding: QuantizationScheme
) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    n, k = weight.shape
    return (("weight", "F8_E4M3", (n, k)), ("weight_scale", "F32", (n, 1)))


def slice_index(
    shape: tuple[int, ...],
    partitions: tuple[int, ...],
    encoding: QuantizationScheme,
    key: str,
    axis: int,
    start: int,
    end: int,
) -> tuple[tuple[slice, ...], tuple[int, ...]]:
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
        if key in ("weight_global_scale", "input_global_scale"):
            return (full,), selected
        if key.endswith("_global") or key in ("weight_curve_a", "weight_curve_b"):
            return (slice(indices[0], indices[-1] + 1),), selected
        if key in ("weight", "weight_packed"):
            return (slice(start, end), full), selected
        return (slice(start // group, end // group), full), selected
    return ((full, slice(start, end)) if key == "weight" else (full,)), partitions
