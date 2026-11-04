"""Physical payload headers and original ownership-preserving slice coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cubic_serve.models.quantization import QuantizationScheme
    from cubic_serve.models.quantized_checkpoint import LogicalWeight

import math

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
    if n % encoding.group_out:
        raise QuantizationError("Cubic logical rows do not close output groups")
    groups = (n // encoding.group_out, math.ceil(k / encoding.group_in))
    parts = weight.output_partitions or (n,)
    if any(p % encoding.group_out for p in parts):
        raise QuantizationError("Cubic partition cuts an output group")
    packed = ("weight_packed", "U8", (n, math.ceil(k * encoding.bits / 8)))
    if encoding.metadata == "int8-scale-int4-ab-global-fp32":
        return (
            packed,
            ("weight_scale", "I8", groups),
            ("weight_ab", "U8", groups),
            *(
                (name, "F32", (len(parts),))
                for name in ("weight_scale_global", "weight_a_global", "weight_b_global")
            ),
        )
    if encoding.metadata == "e5m9-scale-curve2-tensor-fp16":
        return (
            packed,
            ("weight_metadata", "U16", groups),
            ("weight_curve_a", "F16", (len(parts), 4)),
            ("weight_curve_b", "F16", (len(parts), 4)),
        )
    return (
        packed,
        ("weight_scale", "F32", groups),
        ("weight_a", "F16", groups),
        ("weight_b", "F16", groups),
    )


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
    group = encoding.group_in
    # A shard wholly inside one original group retains that group's metadata
    # unchanged: every local column still indexes group zero. Multi-group
    # slices must align, since their local group boundaries otherwise shift.
    one_group = start // group == (end - 1) // group
    if not one_group and (start % group or (end % group and end != shape[1])):
        raise QuantizationError("input slice must preserve original scale groups")
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


def stacked_payload_name(source: str, key: str) -> str:
    """Exporter names for original [expert, projection, ...] payloads."""
    return source + "_" + key.removeprefix("weight_")
