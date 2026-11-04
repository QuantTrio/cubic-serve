"""Independent staged activation rules, separate from CUDA approximation code.

Low-precision exponential reference values use PyTorch's exp2. Native CUDA
ex2.approx requires its documented per-input enclosure as a separate witness;
reference execution never calls the candidate activation kernel.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import torch
from torch import Tensor

from cubic_serve.executor.runner import require_reference_execution
from cubic_serve.ops.strategies import activation_precision, softmax_precision


_reference_memo: ContextVar[Any | None] = ContextVar("reference_activation_memo", default=None)


@contextmanager
def reference_activation_memo(memo):
    """Share bounded pure arithmetic only inside an independent audit."""
    token = _reference_memo.set(memo)
    try:
        yield
    finally:
        _reference_memo.reset(token)


def compute_dtype(storage: torch.dtype, device: torch.device) -> torch.dtype:
    if activation_precision() == "fp32" or storage in (torch.float32, torch.float64):
        return torch.float32
    if storage not in (torch.float16, torch.bfloat16):
        raise ValueError("activation storage must be FP16 or BF16")
    if (
        storage is torch.bfloat16
        and device.type == "cuda"
        and torch.cuda.get_device_capability(device)[0] < 9
    ):
        return torch.float16
    return storage


def exp_nonpositive(x: Tensor, dtype: torch.dtype) -> Tensor:
    require_reference_execution()
    if dtype is torch.float32:
        return torch.exp(x.float())
    argument = x.to(dtype) * torch.tensor(1.4426950408889634, dtype=dtype, device=x.device)
    result = torch.exp2(argument)
    if dtype is torch.bfloat16:
        result = torch.where(result < torch.finfo(dtype).tiny, 0.0, result)
    return result


def sigmoid(x: Tensor, *, storage_dtype: torch.dtype | None = None) -> Tensor:
    require_reference_execution()
    storage = storage_dtype or x.dtype
    dtype = compute_dtype(storage, x.device)
    memo = _reference_memo.get()
    if memo is not None:
        return memo.activation(_sigmoid, x, storage_dtype=storage, compute_dtype=dtype)
    return _sigmoid(x, dtype)


def _sigmoid(x: Tensor, dtype: torch.dtype) -> Tensor:
    if dtype is torch.float32:
        return torch.sigmoid(x.float())
    exponential = exp_nonpositive(-x.abs(), dtype)
    numerator = torch.where(x >= 0, torch.ones_like(exponential), exponential)
    return numerator / (torch.ones_like(exponential) + exponential)


def silu(x: Tensor, *, storage_dtype: torch.dtype | None = None) -> Tensor:
    require_reference_execution()
    storage = storage_dtype or x.dtype
    dtype = compute_dtype(storage, x.device)
    if dtype is torch.float32:
        return torch.nn.functional.silu(x.float())
    gate = sigmoid(x, storage_dtype=storage)
    if dtype is torch.float16:
        saturated = x.float().abs() > 65504.0
        safe = torch.where(saturated, 0.0, x).to(dtype)
        return torch.where(saturated, torch.where(x > 0, x, 0.0), (safe * gate).to(x.dtype))
    return x.to(dtype) * gate


def softmax(x: Tensor, *, dim: int, storage_dtype: torch.dtype) -> Tensor:
    require_reference_execution()
    dtype = (
        torch.float32
        if softmax_precision() == "fp32"
        else compute_dtype(storage_dtype, x.device)
    )
    # Subtraction and reduction stay wide: finite FP16 extremes can overflow
    # their difference, and an equal-logit 64K row overflows an FP16 denominator.
    shifted = x.float() - x.float().amax(dim=dim, keepdim=True)
    numerator = exp_nonpositive(shifted, dtype)
    denominator = numerator.float().sum(dim=dim, keepdim=True)
    # A fused attention consumer has no stored low-precision P boundary.
    # Keep the normalization ratio wide, as the online kernel does; callers
    # with a real route-weight tensor round at that tensor's storage boundary.
    return numerator.float() / denominator


def silu_stored(x: Tensor) -> Tensor:
    return silu(x).to(x.dtype)


def sigmoid_stored(x: Tensor) -> Tensor:
    return sigmoid(x).to(x.dtype)
