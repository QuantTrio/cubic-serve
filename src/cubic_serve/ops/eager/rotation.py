"""Independent rounded orthogonal transform for quantized cache operands.

This reference rule never selects or launches a performance implementation.
Each head uses complete, normalized Walsh-Hadamard blocks. The largest
power-of-two divisor of the head width gives repeated equal blocks without
padding, cropping, or mixing heads. The same symmetric transform is its inverse.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from cubic_serve.executor.runner import require_reference_execution


def rotate_cache_reference(values: Tensor, *, output_dtype: torch.dtype | None = None) -> Tensor:
    """Round each FP32 butterfly stage, normalize, then round the output once.

    Forward Q/K/V retain their native BF16/FP16 dtype. An inverse receives the
    attention FP32 accumulator: it must precede the final native output rounding.
    Input storage is immutable, including noncontiguous reference views.
    """
    require_reference_execution()
    supported = (torch.float16, torch.bfloat16, torch.float32)
    dtype = values.dtype if output_dtype is None else output_dtype
    if (values.ndim < 1 or values.numel() == 0 or values.shape[-1] <= 0
            or values.dtype not in supported or dtype not in supported):
        raise ValueError("cache rotation requires nonempty native or FP32 floating operands")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("cache rotation requires finite reference operands")
    width = values.shape[-1]
    block = width & -width
    work = values.float().reshape(*values.shape[:-1], -1, block).clone()
    step = 1
    while step < block:
        pairs = work.reshape(*work.shape[:-1], -1, 2, step)
        left, right = pairs[..., 0, :].clone(), pairs[..., 1, :].clone()
        pairs[..., 0, :] = left + right
        pairs[..., 1, :] = left - right
        step *= 2
    # An explicit FP32 constant fixes this mathematical rounding boundary.
    normalization = torch.tensor(1.0 / math.sqrt(block), dtype=torch.float32,
                                 device=values.device)
    result = (work * normalization).reshape(values.shape).to(dtype)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("cache rotation is outside its declared floating range")
    return result
