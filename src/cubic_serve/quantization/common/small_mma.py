"""Format-owned active-group decode in a shared transposed MMA contraction.

The original compact codes and metadata remain resident. Only a CTA's current
groups expand to BF16/F16. K ownership and ascending pair reduction are fixed
across row counts; startup graph qualification remains the caller's obligation.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from cubic_serve.ops.cuda.native_compiler import compile_native
from cubic_serve.quantization.common.binding import PackedOperatorBinding
from cubic_serve.quantization.common.small_rows import SmallRowsWeight


def tile_k(k: int, group: int) -> int:
    """Largest bounded staging divisor retaining original checkpoint groups."""
    quantum = math.lcm(128, group)
    return max((tile for tile in range(quantum, max(512, quantum) + 1, quantum)
                if k % tile == 0), default=quantum)


def shared_bytes(*, rows: int, bits: int, group: int, raw_bytes: int, tile_k: int) -> int:
    tile_m = math.ceil(rows / 8) * 8
    codes = 16 * (tile_k * bits // 8 + 16)
    raw = 16 * math.ceil(tile_k // group * raw_bytes / 16) * 16
    metadata = 16 * (tile_k // group) * 8
    levels = 16 * ((tile_k // group) * (1 << (bits - 1)) + 4) * 4
    # XOR swizzling is local to a64-value stripe. A fractional K warp owns
    # padded private input storage; it still loads only its real K values.
    input_stride = math.ceil((tile_k // 8) / 64) * 64
    return max(codes + raw + metadata + levels + tile_m * input_stride * 8 * 2,
               8 * (tile_m // 8) * 32 * 4 * 4)


def bind(x: torch.Tensor, weights: tuple[SmallRowsWeight, ...],
         outputs: tuple[torch.Tensor, ...]) -> PackedOperatorBinding:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("packed MMA binding must precede capture")
    if len(weights) not in (1, 2) or len(outputs) != len(weights):
        raise ValueError("packed MMA requires one or two projection members")
    first = weights[0]
    n, k = first.shape
    # The eight K warps each own complete 16-value MMA fragments. Checkpoint
    # groups may span warps but never straddle the staging-tile boundary.
    staging_k = tile_k(k, first.group)
    if (not x.is_cuda or not x.is_contiguous() or x.dtype not in
            (torch.bfloat16, torch.float16) or x.ndim != 2 or x.shape[1] != k
            or not 1 <= x.shape[0] <= 32 or n % 16 or k % staging_k
            or staging_k % first.group or not 2 <= first.bits <= 8 or x.data_ptr() % 16):
        raise ValueError("unsupported packed MMA geometry")
    raw_bytes = int(dict(first.constants)["FORMAT_RAW_BYTES"])
    size = shared_bytes(rows=x.shape[0], bits=first.bits, group=first.group,
                        raw_bytes=raw_bytes, tile_k=staging_k)
    if size > 48 * 1024:
        raise ValueError("packed MMA exceeds the portable static shared budget")
    readonly = (x, *(w.codes for w in weights), *(w.metadata for w in weights))
    for index, (w, y) in enumerate(zip(weights, outputs, strict=True)):
        if (w.shape != first.shape or w.bits != first.bits or w.group != first.group
                or w.decoder != first.decoder or w.constants != first.constants
                or w.codes.dtype != torch.uint8 or w.codes.numel() != n*k*w.bits//8
                or w.metadata.dtype != torch.uint8 or w.metadata.ndim != 1
                or not all(t.is_contiguous() and t.device == x.device
                           for t in (w.codes, w.metadata, y))
                or w.codes.data_ptr() % 16 or w.metadata.data_ptr() % 16
                or y.shape != (x.shape[0], n) or y.dtype != x.dtype):
            raise ValueError("packed MMA operands differ from the sealed ABI")
        for other in (*readonly, *outputs[:index]):
            if (y.data_ptr() < other.data_ptr() + other.numel()*other.element_size()
                    and other.data_ptr() < y.data_ptr() + y.numel()*y.element_size()):
                raise ValueError("packed MMA output aliases another operand")
    definitions = dict(OUTPUT_N=str(n), INPUT_K=str(k), ROWS=str(x.shape[0]),
                       BITS=str(first.bits), GI=str(first.group), WARPS="8",
                       TILE_K=str(staging_k), ELEMENTS="2",
                       IS_BF16=str(int(x.dtype == torch.bfloat16)))
    if set(definitions).intersection(dict(first.constants)):
        raise ValueError("format constants cannot redefine the MMA contraction")
    definitions.update(first.constants)
    prefix = "".join(f"#define {name} {value}\n" for name, value in definitions.items()).encode()
    sources = (("entry.cu", prefix + Path(__file__).with_suffix(".cu").read_bytes()),
               ("decoder.cuh", first.decoder))
    module, entry = compile_native(sources, x.device, "packed_small_mma")
    second = weights[-1]
    launch = module.bind(entry, (x, first.codes, second.codes, first.metadata,
                                second.metadata, outputs[0], outputs[-1]),
                         operator_id="cubic.cuda.packed.small_mma.v1",
                         grid=(n//16*len(weights), 1, 1), block=(256, 1, 1))
    return PackedOperatorBinding(outputs[0], (launch,), row_tile_count=1)
