"""Shared startup binding for packed, row-local SIMT projections.

This is an operator candidate, not a graph seal or a performance selector.
Format adapters supply original packed codes and compact metadata;
the common contraction never receives model names, layer names or device IDs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from cubic_serve.ops.cuda.native_compiler import compile_native
from cubic_serve.ops.cuda.native import CudaScalar
from cubic_serve.quantization.common.binding import PackedOperatorBinding


@dataclass(frozen=True)
class SmallRowsWeight:
    codes: Tensor
    metadata: Tensor
    shape: tuple[int, int]
    bits: int
    group: int
    decoder: bytes
    constants: tuple[tuple[str, str], ...] = ()
    elements: int = 8


def bind(
    x: Tensor,
    weights: tuple[SmallRowsWeight, ...],
    outputs: tuple[Tensor, ...],
    *,
    warps: int | None = None,
    partitions: int = 1,
) -> PackedOperatorBinding:
    """Bind immutable operands before capture; preserve each member's output."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("packed small-row binding must precede graph capture")
    if len(weights) not in (1, 2) or len(weights) != len(outputs):
        raise ValueError("packed small-row contraction requires one or two members")
    first = weights[0]
    n, k = first.shape
    if (x.ndim != 2 or min(n, k, x.shape[0]) <= 0 or x.shape[1] != k
            or not x.is_cuda or not x.is_contiguous()
            or x.dtype not in (torch.bfloat16, torch.float16)
            or x.data_ptr() % 16
            or k % 128 or not 2 <= first.bits <= 8 or first.group < 32):
        raise ValueError("packed small-row geometry or storage is unsupported")
    quantum = math.lcm(128, first.group)
    upper = min(1024, 32 * first.group)
    tile_k = max((tile for tile in range(quantum, upper + 1, quantum) if k % tile == 0),
                 default=quantum)
    elements = math.gcd(first.elements, tile_k // 32, first.group)
    raw_bytes = int(dict(first.constants)["FORMAT_RAW_BYTES"])
    per_warp = len(weights) * (2 * (tile_k * first.bits // 8 +
                    16 * ((tile_k // first.group * raw_bytes + 15) // 16)) +
                    tile_k // first.group * 8 + x.shape[0] * 4)
    if warps is None:
        warps = next((w for w in (16, 8, 4) if w >= partitions and w * per_warp <= 48 * 1024), 0)
    if (first.group > tile_k or tile_k % first.group or k % tile_k
            or warps not in (4, 8, 16) or partitions not in (1, 2, 4)
            or warps % partitions or k // tile_k < partitions):
        raise ValueError("packed small-row geometry or storage is unsupported")
    if warps * per_warp > 48 * 1024:
        raise ValueError("packed small-row staging exceeds the static shared-memory budget")
    for w, y in zip(weights, outputs, strict=True):
        if (w.shape != first.shape or w.bits != first.bits or w.group != first.group
                or w.decoder != first.decoder or w.constants != first.constants
                or w.elements != first.elements or w.elements not in (2, 4, 8)
                or elements not in (2, 4, 8)
                or w.codes.dtype != torch.uint8 or w.codes.shape != (n, k * w.bits // 8)
                or not w.codes.is_contiguous() or w.codes.device != x.device
                or w.codes.data_ptr() % 16
                or w.metadata.dtype != torch.uint8 or w.metadata.ndim != 1
                or not w.metadata.is_contiguous() or w.metadata.device != x.device
                or w.metadata.data_ptr() % 16
                or y.shape != (x.shape[0], n) or y.dtype != x.dtype or y.device != x.device
                or y.stride(1) != 1 or y.stride(0) < n):
            raise ValueError("packed small-row operands have incompatible contracts")
    readonly = (x, *(w.codes for w in weights), *(w.metadata for w in weights))
    for index, y in enumerate(outputs):
        for other in (*readonly, *outputs[:index]):
            other_extent = 1 + sum((size - 1) * stride for size, stride in
                                   zip(other.shape, other.stride(), strict=True))
            output_extent = (y.shape[0] - 1) * y.stride(0) + n
            if (y.data_ptr() < other.data_ptr() + other_extent * other.element_size()
                    and other.data_ptr() < y.data_ptr() + output_extent * y.element_size()):
                raise ValueError("packed small-row outputs must not alias another operand")
    definitions = dict(
        ROWS=str(x.shape[0]), BITS=str(first.bits),
        GI=str(first.group), MEMBERS=str(len(weights)), WARPS=str(warps),
        ELEMENTS=str(elements), TILE_K=str(tile_k),
        IS_BF16=str(int(x.dtype == torch.bfloat16)),
        CUBIC_ROW_RUNTIME_GEOMETRY="1",
        X_STORAGE=str(0 if k * x.shape[0] * 2 + warps * per_warp <= 48 * 1024
                      else 1 if partitions == 1 and
                      tile_k * x.shape[0] * 4 + warps * per_warp <= 48 * 1024 else 2),
    )
    if set(definitions).intersection(dict(first.constants)):
        raise ValueError("format constants cannot redefine the shared contraction")
    constants = dict(first.constants)
    zero_offset = int(constants.pop("ZERO_OFFSET", "0"))
    raw_offset = int(constants.pop("RAW_OFFSET", "0"))
    descriptors = int(constants.pop("DESCRIPTOR_COUNT", "0"))
    alignment = math.gcd(tile_k // first.group * raw_bytes, k // first.group * raw_bytes)
    definitions["METADATA_VECTOR_BYTES"] = str(next(v for v in (16, 8, 4, 2) if alignment % v == 0))
    definitions.update(constants)
    prefix = "".join(f"#define {name} {value}\n" for name, value in definitions.items()).encode()
    sources = (("entry.cu", prefix + Path(__file__).with_suffix(".cu").read_bytes()),
               ("decoder.cuh", first.decoder))
    module, entry = compile_native(sources, x.device, "packed_small_rows")
    second = weights[-1]
    columns = warps // partitions
    launch = module.bind(
        entry, (x, first.codes, second.codes, first.metadata, second.metadata,
                outputs[0], outputs[-1], CudaScalar("i32", n), CudaScalar("i32", k),
                CudaScalar("i64", outputs[0].stride(0)),
                CudaScalar("i64", outputs[-1].stride(0)), CudaScalar("i32", partitions),
                CudaScalar("i64", zero_offset), CudaScalar("i64", raw_offset),
                CudaScalar("i32", descriptors)),
        operator_id="cubic.cuda.packed.small_rows.v1",
        grid=((n + columns - 1) // columns, 1, 1), block=(warps * 32, 1, 1),
        shared_bytes=k * x.shape[0] * 2 if definitions["X_STORAGE"] == "0" else 0,
    )
    return PackedOperatorBinding(outputs[0], (launch,), row_tile_count=1)
