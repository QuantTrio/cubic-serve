"""Physical payload headers and original ownership-preserving slice coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cubic_serve.models.quantization import QuantizationScheme
    from cubic_serve.models.quantized_checkpoint import LogicalWeight

from cubic_serve.models.quantization import (
    QuantizationError,
)

ROW_ALIGNMENT = 8
CONCAT_AXIS = 1
SCALAR_PAYLOADS = ()


def payload_headers(
    weight: LogicalWeight, encoding: QuantizationScheme
) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    n, k = weight.shape
    if n % 8 or k % encoding.group_in:
        raise QuantizationError("AWQ logical dimensions do not close packing/groups")
    return (
        ("qweight", "I32", (k, n // 8)),
        ("qzeros", "I32", (k // encoding.group_in, n // 8)),
        ("scales", "F16", (k // encoding.group_in, n)),
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
        if start % 8 or end % 8:
            raise QuantizationError("AWQ output slices must preserve packed words")
        low, high = (start, end) if key == "scales" else (start // 8, end // 8)
        return (full, slice(low, high)), ()
    group = encoding.group_in
    if start % group or (end % group and end != shape[1]):
        raise QuantizationError("input slice must preserve original scale groups")
    low, high = (start, end) if key == "qweight" else (start // group, end // group)
    return (slice(low, high), full), partitions
