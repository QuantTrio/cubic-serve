"""Low-precision activation math with explicit, bounded numerical exceptions.

PTX ex2 supports FP16 from SM75 and BF16 from SM90. Unsupported BF16
transcendentals use FP16. Division uses FP32 RN internally because PTX has no
half-precision division instruction; its result is immediately rounded back.
Softmax reductions retain FP32 to avoid long-context denominator overflow.
These are mathematical policies, not workload-dependent tuning parameters.
"""

from __future__ import annotations

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from triton.language.extra import libdevice  # type: ignore[import-untyped]

from cubic_serve.ops.strategies import activation_precision, softmax_precision


def activation_type(dtype: torch.dtype, device: torch.device) -> tl.dtype:
    """Resolve once at binding, before capture. No capability queries in replay."""
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("activation storage must be FP16 or BF16")
    if activation_precision() == "fp32":
        return tl.float32
    if dtype is torch.float16:
        return tl.float16
    return tl.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 9 else tl.float16


def softmax_type(dtype: torch.dtype, device: torch.device) -> tl.dtype:
    """Bind softmax independently from SiLU and sigmoid before graph capture."""
    return tl.float32 if softmax_precision() == "fp32" else activation_type(dtype, device)


@triton.jit  # type: ignore[untyped-decorator]
def exp_nonpositive(x, DTYPE: tl.constexpr):  # type: ignore[no-untyped-def]
    """Exponential on x <= 0; no positive-exponential overflow branch."""
    if tl.float32 == DTYPE:
        result = libdevice.exp(x.to(tl.float32))
    else:
        argument = (x.to(DTYPE) * tl.full((), 1.4426950408889634, DTYPE)).to(DTYPE)
        if tl.bfloat16 == DTYPE:
            result = tl.inline_asm_elementwise(
                "ex2.approx.ftz.bf16 $0, $1;",
                "=h,h",
                [argument],
                dtype=tl.bfloat16,
                is_pure=True,
                pack=1,
            )
        else:
            result = tl.inline_asm_elementwise(
                "ex2.approx.f16 $0, $1;",
                "=h,h",
                [argument],
                dtype=tl.float16,
                is_pure=True,
                pack=1,
            )
    return result


@triton.jit  # type: ignore[untyped-decorator]
def sigmoid(x, DTYPE: tl.constexpr):  # type: ignore[no-untyped-def]
    if tl.float32 == DTYPE:
        result = tl.div_rn(1.0, 1.0 + libdevice.exp(-x.to(tl.float32)))
    else:
        magnitude = tl.abs(x).to(DTYPE)
        exponential = exp_nonpositive(-magnitude, DTYPE)
        denominator = (tl.full((), 1.0, DTYPE) + exponential).to(DTYPE)
        numerator = tl.where(x >= 0, tl.full((), 1.0, DTYPE), exponential)
        result = tl.div_rn(numerator.to(tl.float32), denominator.to(tl.float32)).to(DTYPE)
    return result


@triton.jit  # type: ignore[untyped-decorator]
def silu(x, DTYPE: tl.constexpr):  # type: ignore[no-untyped-def]
    if tl.float32 == DTYPE:
        result = tl.div_rn(x.to(tl.float32), 1.0 + libdevice.exp(-x.to(tl.float32)))
    else:
        gate = sigmoid(x, DTYPE)
        if tl.float16 == DTYPE:
            # A finite BF16 positive tail must survive FP16 fallback unchanged.
            saturated = tl.abs(x) > 65504.0
            safe_x = tl.where(saturated, 0.0, x).to(DTYPE)
            value = (safe_x * gate).to(DTYPE)
            result = tl.where(saturated, tl.where(x > 0, x, 0.0), value.to(x.dtype))
        else:
            result = (x.to(DTYPE) * gate).to(DTYPE)
    return result
