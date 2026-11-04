"""Caller-owned INT8 activations and quantization boundaries for WNA8."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def quantize_rows(X, Q, S, K: tl.constexpr, BLOCK: tl.constexpr):
    """Shared floating boundary -> caller-owned signed codes and token scales."""
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    values = tl.load(X + row * K + k, k < K, 0).to(tl.float32)
    codes, scale = encode(values)
    tl.store(Q + row * K + k, codes, k < K)
    tl.store(S + row, scale)


@triton.jit
def quantize_rows_padded_groups(
    X, Q, S, K: tl.constexpr, PADDED_K: tl.constexpr,
    GROUP: tl.constexpr, PADDED_GROUP: tl.constexpr, BLOCK: tl.constexpr,
):
    """Quantize one original row while making K32-aligned groups for native MMA."""
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    source = column // PADDED_GROUP * GROUP + column % PADDED_GROUP
    valid = (column < PADDED_K) & (column % PADDED_GROUP < GROUP) & (source < K)
    values = tl.load(X + row * K + source, valid, 0).to(tl.float32)
    codes, scale = encode(values)
    tl.store(Q + row * PADDED_K + column, codes, column < PADDED_K)
    tl.store(S + row, scale)


@triton.jit
def pad_int8_groups(
    X, Q, K: tl.constexpr, PADDED_K: tl.constexpr,
    GROUP: tl.constexpr, PADDED_GROUP: tl.constexpr, BLOCK: tl.constexpr,
):
    """Copy producer-owned codes to Marlin's group-aligned operand layout."""
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    source = column // PADDED_GROUP * GROUP + column % PADDED_GROUP
    valid = (column < PADDED_K) & (column % PADDED_GROUP < GROUP) & (source < K)
    codes = tl.load(X + row * K + source, valid, 0)
    tl.store(Q + row * PADDED_K + column, codes, column < PADDED_K)


@triton.jit
def dot_int8(a, b):
    """Common exact integer contraction; formats own code extraction and scaling."""
    return tl.dot(a, b, out_dtype=tl.int32)


@dataclass(frozen=True, slots=True)
class Int8Activation:
    codes: torch.Tensor
    scales: torch.Tensor
    boundary_dtype: torch.dtype
    floating: torch.Tensor | None = None

    def __post_init__(self):
        q, s = self.codes, self.scales
        if (
            not q.is_cuda
            or q.dtype != torch.int8
            or q.ndim != 2
            or min(q.shape) < 1
            or not q.is_contiguous()
            or s.device != q.device
            or s.dtype != torch.float32
            or s.shape != (q.shape[0],)
            or not s.is_contiguous()
            or self.boundary_dtype not in (torch.float16, torch.bfloat16)
            or q.untyped_storage().data_ptr() == s.untyped_storage().data_ptr()
        ):
            raise ValueError("INT8 activation requires private CUDA int8 [M,K] and FP32 [M]")
        if self.floating is not None and (
            self.floating.shape != q.shape
            or self.floating.dtype != self.boundary_dtype
            or self.floating.device != q.device
            or not self.floating.is_contiguous()
            or self.floating.untyped_storage().data_ptr()
            in (
                q.untyped_storage().data_ptr(),
                s.untyped_storage().data_ptr(),
            )
        ):
            raise ValueError("floating producer exit must be private and match its boundary")


@triton.jit
def encode(values):
    scale = tl.div_rn(tl.max(tl.abs(values), 0), 127.0)
    scale = tl.where(scale > 0, scale, 1.0)
    if scale >= 1.1754943508222875e-38:
        normalized = values * tl.div_rn(1.0, scale)
    else:
        normalized = tl.div_rn(values, scale)
    codes = libdevice.nearbyint(normalized)
    return tl.minimum(127.0, tl.maximum(-127.0, codes)).to(tl.int8), scale
