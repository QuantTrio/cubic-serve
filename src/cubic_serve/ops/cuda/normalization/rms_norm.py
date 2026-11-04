"""Caller-owned, row-invariant Qwen zero-centered RMSNorm.

llama.cpp c589f0ed reduces each row independently; vLLM 6cddad41 and
SGLang 00fbb6e8 expose batch-invariant RMSNorm paths.  This primitive fixes
the row width and reduction tree in its launch ABI and owns no tensor or graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from torch import Tensor
from triton.language.extra import libdevice  # type: ignore[import-untyped]


@triton.jit  # type: ignore[untyped-decorator]
def _zero_centered_rms_norm(  # type: ignore[no-untyped-def]
    x_ptr,
    weight_ptr,
    output_ptr,
    stride_xm,
    stride_ym,
    width: tl.constexpr,
    epsilon: tl.constexpr,
    block: tl.constexpr,
    groups: tl.constexpr = 1,
):
    row = tl.program_id(0) // groups
    group_offset = (tl.program_id(0) % groups) * width
    columns = tl.arange(0, block)
    mask = columns < width
    x = tl.load(x_ptr + row * stride_xm + group_offset + columns, mask=mask, other=0.0).to(
        tl.float32
    )
    weight = tl.load(weight_ptr + group_offset + columns, mask=mask, other=0.0).to(tl.float32)
    variance = libdevice.div_rn(tl.sum(x * x, axis=0), float(width))
    inverse_rms = libdevice.rsqrt_rn(variance + epsilon)
    output = x * inverse_rms * (weight + 1.0)
    tl.store(output_ptr + row * stride_ym + group_offset + columns, output, mask=mask)


@triton.jit  # type: ignore[untyped-decorator]
def _bf16_residual_zero_centered_rms_norm(  # type: ignore[no-untyped-def]
    residual_ptr,
    update_ptr,
    summed_ptr,
    weight_ptr,
    output_ptr,
    stride_m,
    width: tl.constexpr,
    epsilon: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block)
    mask = columns < width
    offsets = row * stride_m + columns
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    update = tl.load(update_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    # Preserve the standalone residual kernel's activation store boundary before
    # RMSNorm consumes the sum.
    summed = (residual + update).to(summed_ptr.dtype.element_ty)
    tl.store(summed_ptr + offsets, summed, mask=mask)
    values = summed.to(tl.float32)
    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    variance = libdevice.div_rn(tl.sum(values * values, axis=0), float(width))
    inverse_rms = libdevice.rsqrt_rn(variance + epsilon)
    tl.store(
        output_ptr + offsets,
        values * inverse_rms * (weight + 1.0),
        mask=mask,
    )


@dataclass(frozen=True, slots=True)
class ZeroCenteredRMSNormMeta:
    """Complete fixed launch identity; row count includes graph padding."""

    device: torch.device
    rows: int
    width: int
    x_stride: tuple[int, int]
    output_stride: tuple[int, int]
    dtype: torch.dtype
    epsilon: float
    block: int
    groups: int = 1


class ZeroCenteredRMSNormPrimitive:
    """Launch into caller-owned FP16/BF16 storage without allocation or capture."""

    def __init__(
        self,
        x: Tensor,
        weight: Tensor,
        output: Tensor,
        *,
        epsilon: float,
        group_size: int | None = None,
    ) -> None:
        self._validate(x, weight, output, epsilon)
        width = x.shape[1]
        group_size = width if group_size is None else group_size
        if type(group_size) is not int or group_size <= 0 or width % group_size:
            raise ValueError("RMSNorm group size must divide the affine width")
        self.meta = ZeroCenteredRMSNormMeta(
            device=x.device,
            rows=x.shape[0],
            width=width,
            x_stride=(x.stride(0), x.stride(1)),
            output_stride=(output.stride(0), output.stride(1)),
            dtype=x.dtype,
            epsilon=float(epsilon),
            block=triton.next_power_of_2(group_size),
            groups=width // group_size,
        )

    @property
    def launch_signature(self) -> str:
        meta = self.meta
        signature = (
            f"rows={meta.rows},width={meta.width},block={meta.block},"
            f"eps={meta.epsilon.hex()},acc=fp32,out={_dtype_name(meta.dtype)},weight=one-plus"
        )
        return signature if meta.groups == 1 else signature + f",groups={meta.groups}"

    def launch(self, x: Tensor, weight: Tensor, output: Tensor) -> None:
        """Enqueue one fixed-shape kernel on the current CUDA stream."""

        self._validate(x, weight, output, self.meta.epsilon)
        meta = self.meta
        if (
            x.device != meta.device
            or x.dtype != meta.dtype
            or x.shape != (meta.rows, meta.width)
            or x.stride() != meta.x_stride
            or output.stride() != meta.output_stride
        ):
            raise ValueError("RMSNorm tensors do not match the fixed launch meta")
        _zero_centered_rms_norm[(meta.rows * meta.groups,)](
            x,
            weight,
            output,
            meta.x_stride[0],
            meta.output_stride[0],
            width=meta.width // meta.groups,
            epsilon=meta.epsilon,
            block=meta.block,
            groups=meta.groups,
            num_warps=8 if meta.block >= 4096 else 4,
        )

    @staticmethod
    def _validate(x: Tensor, weight: Tensor, output: Tensor, epsilon: float) -> None:
        if any(not isinstance(tensor, Tensor) for tensor in (x, weight, output)):
            raise TypeError("RMSNorm operands must be tensors")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("RMSNorm epsilon must be finite and positive")
        if (
            x.device.type != "cuda"
            or weight.device != x.device
            or output.device != x.device
            or x.dtype not in (torch.float16, torch.bfloat16)
            or weight.dtype is not x.dtype
            or output.dtype is not x.dtype
            or x.ndim != 2
            or weight.ndim != 1
            or output.shape != x.shape
            or weight.shape[0] != x.shape[1]
            or not x.is_contiguous()
            or not weight.is_contiguous()
            or not output.is_contiguous()
            or min(x.shape) <= 0
        ):
            raise ValueError(
                "expected contiguous non-empty same-device CUDA FP16/BF16 [rows,width]"
            )


class ResidualRMSNormPrimitive:
    """Fuse the declared activation residual boundary with zero-centered RMSNorm."""

    def __init__(
        self,
        residual: Tensor,
        update: Tensor,
        summed: Tensor,
        weight: Tensor,
        output: Tensor,
        *,
        epsilon: float,
    ) -> None:
        self._validate(residual, update, summed, weight, output, epsilon)
        width = residual.shape[1]
        self.meta = ZeroCenteredRMSNormMeta(
            device=residual.device,
            rows=residual.shape[0],
            width=width,
            x_stride=(residual.stride(0), residual.stride(1)),
            output_stride=(output.stride(0), output.stride(1)),
            dtype=residual.dtype,
            epsilon=float(epsilon),
            block=triton.next_power_of_2(width),
        )

    @property
    def launch_signature(self) -> str:
        meta = self.meta
        return (
            f"rows={meta.rows},width={meta.width},block={meta.block},"
            f"eps={meta.epsilon.hex()},residual={_dtype_name(meta.dtype)},weight=one-plus"
        )

    def launch(
        self,
        residual: Tensor,
        update: Tensor,
        summed: Tensor,
        weight: Tensor,
        output: Tensor,
    ) -> None:
        self._validate(residual, update, summed, weight, output, self.meta.epsilon)
        meta = self.meta
        if (
            residual.device != meta.device
            or residual.dtype != meta.dtype
            or residual.shape != (meta.rows, meta.width)
            or residual.stride() != meta.x_stride
            or update.stride() != meta.x_stride
            or summed.stride() != meta.x_stride
            or output.stride() != meta.output_stride
        ):
            raise ValueError("residual RMSNorm tensors do not match the fixed launch meta")
        _bf16_residual_zero_centered_rms_norm[(meta.rows,)](
            residual,
            update,
            summed,
            weight,
            output,
            meta.x_stride[0],
            width=meta.width,
            epsilon=meta.epsilon,
            block=meta.block,
            num_warps=8 if meta.block >= 4096 else 4,
        )

    @staticmethod
    def _validate(
        residual: Tensor,
        update: Tensor,
        summed: Tensor,
        weight: Tensor,
        output: Tensor,
        epsilon: float,
    ) -> None:
        operands = residual, update, summed, weight, output
        if any(not isinstance(tensor, Tensor) for tensor in operands):
            raise TypeError("residual RMSNorm operands must be tensors")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("residual RMSNorm epsilon must be finite and positive")
        if (
            residual.device.type != "cuda"
            or any(tensor.device != residual.device for tensor in operands)
            or residual.dtype not in (torch.float16, torch.bfloat16)
            or any(tensor.dtype != residual.dtype for tensor in operands)
            or residual.ndim != 2
            or update.shape != residual.shape
            or summed.shape != residual.shape
            or output.shape != residual.shape
            or weight.shape != (residual.shape[1],)
            or any(not tensor.is_contiguous() for tensor in operands)
            or min(residual.shape) <= 0
            or summed.untyped_storage().data_ptr()
            in {
                residual.untyped_storage().data_ptr(),
                update.untyped_storage().data_ptr(),
            }
        ):
            raise ValueError(
                "expected non-aliasing contiguous CUDA FP16/BF16 residual RMSNorm operands"
            )


def _dtype_name(dtype: torch.dtype) -> str:
    return "bf16" if dtype is torch.bfloat16 else "fp16"


# Preserve imports and BF16 launch signatures of existing qualified callers.
BF16ResidualRMSNormPrimitive = ResidualRMSNormPrimitive


__all__ = [
    "BF16ResidualRMSNormPrimitive",
    "ResidualRMSNormPrimitive",
    "ZeroCenteredRMSNormMeta",
    "ZeroCenteredRMSNormPrimitive",
]
