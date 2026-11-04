"""Native Q/K/V rotation with caller-owned, explicitly disjoint output scratch."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor

from cubic_serve.ops.cuda.native import CudaScalar, NativeLaunch
from cubic_serve.ops.cuda.native_compiler import compile_native


def bind_cache_rotation(
    query: Tensor | None, key: Tensor, value: Tensor, *,
    query_output: Tensor | None, key_output: Tensor | None, value_output: Tensor | None,
    rotate_key: bool, rotate_value: bool,
) -> NativeLaunch:
    """Pair Q with encoded K, and independently rotate encoded V, in one launch.

    The caller retains raw producers and owns all scratch under its graph lease.
    There is no quantization of Q, allocation during replay, or historical cache
    expansion. Disabled groups neither consume nor produce a rotation buffer.
    """
    if type(rotate_key) is not bool or type(rotate_value) is not bool:
        raise TypeError("cache rotation sides require booleans")
    if not (rotate_key or rotate_value):
        raise ValueError("cache rotation requires an encoded side")
    if (key.ndim != 3 or key.shape != value.shape or min(key.shape) < 1
            or key.dtype not in (torch.bfloat16, torch.float16) or key.device.type != "cuda"):
        raise ValueError("cache rotation requires positive native CUDA token/head geometry")
    if (query is not None and (query.ndim != 3 or query.shape[0] != key.shape[0]
                              or query.shape[1] < 1 or query.shape[-1] != key.shape[-1])):
        raise ValueError("paired query rotation must share token and head width geometry")
    inputs = (query, key, value)
    outputs = (query_output, key_output, value_output)
    enabled = (rotate_key, rotate_key, rotate_value)
    counts = []
    for source, destination, active in zip(inputs, outputs, enabled):
        if source is not None and (source.device != key.device or source.dtype != key.dtype
                                   or not source.is_contiguous()):
            raise ValueError("cache rotation requires contiguous matching native operands")
        if not active:
            if destination is not None:
                raise ValueError("disabled cache rotation must not own output scratch")
            counts.append(0)
            continue
        if (source is None or destination is None or destination.shape != source.shape
                or destination.dtype != source.dtype or destination.device != source.device
                or not destination.is_contiguous()):
            raise ValueError("enabled cache rotation requires matching native output scratch")
        counts.append(source.numel() // source.shape[-1])
    for i, destination in enumerate(outputs):
        if destination is None:
            continue
        first = destination.data_ptr()
        last = first + destination.numel() * destination.element_size()
        for other in (*inputs, *outputs[i + 1:]):
            if other is not None:
                start = other.data_ptr()
                end = start + other.numel() * other.element_size()
                if first < end and start < last:
                    raise ValueError("cache rotation outputs must not alias any input or output")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("cache rotation must be bound before capture")
    dim = key.shape[-1]
    group = dim & -dim
    shared = 4 * dim
    if shared > torch.cuda.get_device_properties(key.device).shared_memory_per_block_optin:
        raise ValueError("cache rotation exceeds device shared-memory resources")
    prefix = f"#define NATIVE_BF16 {int(key.dtype is torch.bfloat16)}\n".encode()
    module, entry = compile_native(
        (("entry.cu", prefix + Path(__file__).with_suffix(".cu").read_bytes()),),
        key.device, "cache_rotation_native",
    )
    return module.bind(
        entry, (*inputs, *outputs, *(CudaScalar("i32", n) for n in counts),
                CudaScalar("i32", dim), CudaScalar("i32", group),
                CudaScalar("f32", 1.0 / math.sqrt(group))),
        operator_id="cubic.cuda.cache.orthogonal-native-rne.v1",
        grid=(sum(counts), 1, 1), block=(128, 1, 1), shared_bytes=shared,
    )
