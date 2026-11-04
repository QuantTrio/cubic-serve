"""Caller-owned FP16/BF16 partial M-RoPE with FP32 rotation and final activation rounding.

llama.cpp c841aeeb and vLLM 73fac1e1 establish leading NeoX half-split
rotation and untouched tails; SGLang ba433bb4 supplies the interleaved layout. Cubic computes
rotation in FP32 and rounds only when storing the activation.
Cosine/sine generation is deliberately out of scope.

GraphProgram qualification must warm both fixed Triton specializations before
seal/capture. This primitive owns no tensor, graph, stream, or compile policy.
"""

from __future__ import annotations

from typing_extensions import TypeAliasType

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from torch import Tensor

_TensorABI = TypeAliasType("_TensorABI", tuple[tuple[int, ...], tuple[int, ...], torch.dtype, torch.device])


@triton.jit  # type: ignore[untyped-decorator]
def _apply(  # type: ignore[no-untyped-def]
    source,
    cosine,
    sine,
    output,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    columns = tl.arange(0, BLOCK)
    valid, rotary = columns < D, columns < R
    base = (row * HEADS + head) * D
    values = tl.load(source + base + columns, mask=valid).to(tl.float32)
    pairs = tl.where(columns < HALF, columns + HALF, columns - HALF)
    paired = tl.load(source + base + pairs, mask=rotary, other=0.0).to(tl.float32)
    rotated = tl.where(columns < HALF, -paired, paired)
    cos = tl.load(cosine + row * R + columns, mask=rotary, other=0.0).to(tl.float32)
    sin = tl.load(sine + row * R + columns, mask=rotary, other=0.0).to(tl.float32)
    left = values * cos
    right = rotated * sin
    rotated_value = left + right
    tl.store(output + base + columns, tl.where(rotary, rotated_value, values), mask=valid)


def _abi(tensor: Tensor) -> _TensorABI:
    if not isinstance(tensor, Tensor):
        raise TypeError("M-RoPE operands must be tensors")
    return tuple(tensor.shape), tensor.stride(), tensor.dtype, tensor.device


class QwenMRoPEApplyPrimitive:
    """Apply supplied ``[C,R]`` activation-dtype cos/sin to Q and K caller outputs."""

    def __init__(
        self,
        query: Tensor,
        key: Tensor,
        cosine: Tensor,
        sine: Tensor,
        out_query: Tensor,
        out_key: Tensor,
    ) -> None:
        tensors = query, key, cosine, sine, out_query, out_key
        abis = tuple(_abi(tensor) for tensor in tensors)
        if (
            query.dtype not in (torch.float16, torch.bfloat16)
            or any(
                tensor.device.type != "cuda"
                or tensor.dtype != query.dtype
                or not tensor.is_contiguous()
                for tensor in tensors
            )
            or len({tensor.device for tensor in tensors}) != 1
        ):
            raise ValueError("M-RoPE operands must be contiguous same-dtype CUDA FP16/BF16 tensors")
        capacity = query.shape[0] if query.ndim == 3 else -1
        rotary = cosine.shape[1] if cosine.ndim == 2 else -1
        if not (
            capacity > 0
            and min(query.shape) > 0
            and key.ndim == 3
            and key.shape[0] == capacity
            and key.shape[2] == query.shape[2]
            and min(key.shape) > 0
            and cosine.shape == (capacity, rotary)
            and _abi(sine) == _abi(cosine)
            and rotary > 0
            and rotary % 2 == 0
            and rotary <= query.shape[2]
            and _abi(out_query) == _abi(query)
            and _abi(out_key) == _abi(key)
        ):
            raise ValueError("operands do not match fixed [C,H,D]/[C,R] partial M-RoPE ABI")
        self._abis = abis
        self._meta = (
            capacity,
            query.shape[1],
            key.shape[1],
            query.shape[2],
            rotary,
            triton.next_power_of_2(query.shape[2]),
        )

    @property
    def launch_signature(self) -> str:
        capacity, query_heads, key_heads, dim, rotary, _ = self._meta
        dtype = "bf16" if self._abis[0][2] is torch.bfloat16 else "fp16"
        return (
            f"c={capacity},qh={query_heads},kvh={key_heads},d={dim},r={rotary},"
            f"fp32-rotation,{dtype}-output"
        )

    def launch(
        self,
        query: Tensor,
        key: Tensor,
        cosine: Tensor,
        sine: Tensor,
        out_query: Tensor,
        out_key: Tensor,
    ) -> None:
        tensors = query, key, cosine, sine, out_query, out_key
        if tuple(_abi(tensor) for tensor in tensors) != self._abis:
            raise ValueError("M-RoPE operands changed after fixed-ABI binding")
        capacity, query_heads, key_heads, dim, rotary, block = self._meta
        _apply[(capacity, query_heads)](
            query,
            cosine,
            sine,
            out_query,
            HEADS=query_heads,
            D=dim,
            R=rotary,
            HALF=rotary // 2,
            BLOCK=block,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _apply[(capacity, key_heads)](
            key,
            cosine,
            sine,
            out_key,
            HEADS=key_heads,
            D=dim,
            R=rotary,
            HALF=rotary // 2,
            BLOCK=block,
            num_warps=4,
            enable_fp_fusion=False,
        )


__all__ = ["QwenMRoPEApplyPrimitive"]
