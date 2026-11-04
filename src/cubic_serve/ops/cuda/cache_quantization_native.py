"""Startup-bound native KV codecs with runtime geometry and shared receipts."""

from pathlib import Path

import torch

from cubic_serve.ops.cuda.native import CudaScalar
from cubic_serve.ops.cuda.native_compiler import compile_native
from cubic_serve.runtime.cache.quantization import CacheDType, CacheGroupAxis, GroupedCacheLayout


def _divisor64(divisor: int) -> tuple[int, int]:
    # The same invariant-division rule as the shared signed-32 index helper,
    # widened for nonnegative INT64 absolute positions (no context truncation).
    shift = (divisor - 1).bit_length()
    multiplier = ((1 << 64) * ((1 << shift) - divisor)) // divisor + 1
    return multiplier if multiplier < (1 << 63) else multiplier - (1 << 64), shift


def bind_cache_launch(
    incoming, payload, scales, layout: GroupedCacheLayout, *,
    tail=None, sources=None, destinations=None, segments=None, activity=None,
    positions=None, pages=None, written=None, layers=1, layer=0,
    store_groups=True, store_tail=True,
):
    """Called after the common cache binding has validated format and ownership."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("native cache codecs must be bound before capture")
    if layout.spec.dtype not in (
        CacheDType.FP8, CacheDType.INT8, CacheDType.INT6, CacheDType.INT4,
        CacheDType.BF16, CacheDType.FP16,
    ):
        raise ValueError("native cache writer has not implemented this storage encoding")
    column = layout.spec.axis is CacheGroupAxis.TOKENS and layout.spec.dtype.quantized
    append = written is not None
    rows, heads, dim = incoming.shape
    group = layout.spec.group_size
    packed_pitch = (((group + 3) // 4) | 1) * 4
    shared = 32 * (packed_pitch + (group * 2 if append else 0)) if column else 0
    # Resource-based admission, independent of model, SKU or benchmark shape.
    properties = torch.cuda.get_device_properties(incoming.device)
    if shared + 512 > properties.shared_memory_per_block_optin:
        raise ValueError("native cache group exceeds the device shared-memory limit")
    action = int(append) * 2 + int(column)
    # Only semantic operators and native scalar types define binary families.
    # M, heads, dimensions, group width, pages and TP remain runtime operands.
    definitions = dict(
        CACHE_ACTION=action, CACHE_BF16=int(incoming.dtype == torch.bfloat16),
        CACHE_QUANTIZED=int(layout.spec.dtype.quantized),
        CACHE_INTEGER=int(layout.spec.dtype in (CacheDType.INT8, CacheDType.INT6, CacheDType.INT4)),
        CACHE_BITS=layout.spec.dtype.bits if layout.spec.dtype.quantized else 8,
        CACHE_STORE_GROUPS=int(store_groups) if append else 1,
        CACHE_STORE_TAIL=int(store_tail) if column and append else 0,
    )
    prefix = "".join(f"#define {k} {v}\n" for k, v in definitions.items()).encode()
    module, entry = compile_native(
        (("entry.cu", prefix + Path(__file__).with_suffix(".cu").read_bytes()),),
        incoming.device, "cache_quantization_native",
    )
    group_count = (dim + group - 1) // group
    grid = ((rows if append else (rows + group - 1) // group),
            (heads * dim + 31) // 32, 1) if column else (
                (rows * heads * group_count + 3) // 4, 1, 1)
    scalars = (action, int(incoming.dtype == torch.bfloat16),
               rows, layers, layer, layout.tokens, heads, dim, group,
               int(layout.spec.dtype.quantized), int(store_groups), int(store_tail),
               int(destinations is not None and destinations.dtype == torch.int64))
    divisors = (_divisor64(group), _divisor64(layout.tokens),
                _divisor64(layout.packed_group_bytes))
    return module.bind(
        entry,
        (incoming, payload, scales, tail, sources, destinations, segments, activity,
         positions, pages, written, *(CudaScalar("i32", v) for v in scalars),
         *(item for multiplier, shift in divisors
           for item in (CudaScalar("i64", multiplier), CudaScalar("i32", shift))),
         *(CudaScalar("i64", stride) for stride in (
             payload.stride(0) * payload.element_size(),
             0 if scales is None else scales.stride(0),
             0 if tail is None else tail.stride(0),
             0 if written is None else written.stride(0),
         ))),
        operator_id=f"cubic.cuda.cache.{layout.spec.dtype.value}.rne.v1",
        grid=grid, block=(128, 1, 1),
        shared_bytes=shared,
    )
