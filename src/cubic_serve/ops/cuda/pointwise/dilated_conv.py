"""State-explicit causal depthwise convolution for packed sequence segments."""

from __future__ import annotations

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from torch import Tensor

from cubic_serve.ops.cuda.activation import activation_type, silu
from cubic_serve.quantization.common.binding import PackedOperatorBinding, _PackedLaunch
from cubic_serve.runtime.cache.quantization import slot_storage_contiguous, storage_regions_overlap


@triton.jit  # type: ignore[untyped-decorator]
def _dilated_conv(  # type: ignore[no-untyped-def]
    X,
    STATES,
    SOURCES,
    DESTINATIONS,
    SEGMENTS,
    ACTIVE,
    WEIGHT,
    OUTPUT,
    ROWS: tl.constexpr,
    CHANNELS: tl.constexpr,
    KERNEL: tl.constexpr,
    DILATION: tl.constexpr,
    HISTORY: tl.constexpr,
    HB: tl.constexpr,
    CB: tl.constexpr,
    ACT: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
):
    start = tl.program_id(0)
    channels = tl.program_id(1) * CB + tl.arange(0, CB)
    length = tl.load(SEGMENTS + start)
    if (tl.load(ACTIVE + start) != 0) & (length > 0):
        source = tl.load(SOURCES + start).to(tl.int64)
        h = tl.arange(0, HB)
        history = tl.load(
            STATES + source * STATE_STRIDE + h[:, None] * CHANNELS + channels[None, :],
            (source >= 0) & (h[:, None] < HISTORY) & (channels[None, :] < CHANNELS),
            0,
        )
        for step in range(tl.minimum(length, ROWS - start)):
            row = start + step
            if tl.load(ACTIVE + row) != 0:
                current = tl.load(X + row * CHANNELS + channels, channels < CHANNELS, 0)
                acc = tl.full((CB,), 0, tl.float32)
                for tap in tl.static_range(KERNEL):
                    if tap == KERNEL - 1:
                        value = current.to(tl.float32)
                    else:
                        index = tl.full((1, CB), tap * DILATION, tl.int32)
                        value = tl.sum(tl.gather(history, index, 0).to(tl.float32), 0)
                    weight = tl.load(WEIGHT + channels * KERNEL + tap, channels < CHANNELS, 0)
                    acc = acc + value * weight.to(tl.float32)
                # The checkpoint rule stores convolution before evaluating SiLU.
                stored = acc.to(OUTPUT.dtype.element_ty).to(tl.float32)
                result = silu(stored, ACT)
                tl.store(OUTPUT + row * CHANNELS + channels, result, channels < CHANNELS)
                shifted = tl.broadcast_to(tl.minimum(h[:, None] + 1, HB - 1), (HB, CB))
                history = tl.gather(history, shifted, 0)
                history = tl.where(h[:, None] == HISTORY - 1, current[None, :], history)
                destination = tl.load(DESTINATIONS + row).to(tl.int64)
                tl.store(
                    STATES + destination * STATE_STRIDE + h[:, None] * CHANNELS + channels[None, :],
                    history,
                    (destination >= 0) & (h[:, None] < HISTORY) & (channels[None, :] < CHANNELS),
                )


def bind_dilated_depthwise_conv(
    x: Tensor,
    states: Tensor,
    sources: Tensor,
    destinations: Tensor,
    segment_lengths: Tensor,
    row_activity: Tensor,
    weight: Tensor,
    output: Tensor,
    *,
    dilation: int,
) -> PackedOperatorBinding:
    """Bind stored-convolution/SiLU with per-token history checkpoints.

    History is oldest-first [slot,(kernel-1)*dilation,channel]. Negative source
    means zero history; negative destination omits a checkpoint. The caller owns
    bounds, slot generations, disjoint destinations, and cross-segment source
    immutability. Continuation entries are real rows, never padding indicators.
    """
    operands = (x, states, sources, destinations, segment_lengths, row_activity, weight, output)
    if x.device.type != "cuda" or any(
        t.device != x.device or (t is not states and not t.is_contiguous()) for t in operands
    ):
        raise ValueError("dilated convolution requires contiguous same-device CUDA operands")
    if not slot_storage_contiguous(states):
        raise ValueError("convolution history requires dense, nonoverlapping slot payloads")
    if x.ndim != 2 or min(x.shape) <= 0 or x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("convolution input must be a positive FP16/BF16 row matrix")
    rows, channels = x.shape
    if (
        type(dilation) is not int
        or dilation <= 0
        or weight.ndim != 3
        or weight.shape[:2] != (channels, 1)
        or weight.shape[2] < 2
        or output.shape != x.shape
        or any(t.dtype != x.dtype for t in (states, weight, output))
    ):
        raise ValueError("convolution weight, output, dtype or dilation differs")
    kernel = weight.shape[2]
    history = (kernel - 1) * dilation
    if states.ndim != 3 or states.shape[0] <= 0 or states.shape[1:] != (history, channels):
        raise ValueError("convolution history must store every intervening input token")
    if any(t.dtype != torch.int32 or t.shape != (rows,) for t in operands[2:6]):
        raise ValueError("convolution ownership/segments require one INT32 value per row")
    for writable in (states, output):
        if any(
            writable is not t
            and storage_regions_overlap(writable, t)
            for t in operands
        ):
            raise ValueError("convolution writable operands may not alias other operands")
    call = _PackedLaunch.compile(
        _dilated_conv,
        operands,
        (rows, channels, kernel, dilation, history, triton.next_power_of_2(history), 32,
         activation_type(x.dtype, x.device), states.stride(0)),
        (rows, triton.cdiv(channels, 32)),
        {"num_warps": 4, "enable_fp_fusion": False},
    )
    return PackedOperatorBinding(output, (call,))
