"""Fixed-ABI FP16/BF16 token embedding gather.

llama.cpp c589f0ed gathers token rows directly; vLLM 680e2177 and SGLang
a1fe4e30 keep vocabulary sharding/masking outside the unquantized gather.  The
primitive below therefore owns only the rank-local row lookup.  GraphProgram
must validate global or shard-local token IDs before fill and warm this fixed
Triton specialization before capture; launch deliberately performs neither.
"""

from __future__ import annotations

from typing_extensions import TypeAliasType

from typing import cast

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from torch import Tensor

from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.ops.cuda.embedding.immutable import embedding_pointer
from cubic_serve.ops.cuda.quantized_formats import fp8_e4m3fn_decode
from cubic_serve.quantization.common.binding import PackedOperatorBinding, _PackedLaunch
from cubic_serve.quantization.registry import get_extension

_BLOCK_H = 256
_TensorABI = TypeAliasType("_TensorABI", tuple[tuple[int, ...], tuple[int, ...], torch.dtype, torch.device])


@triton.jit  # type: ignore[untyped-decorator]
def _gather(  # type: ignore[no-untyped-def]
    token_ids,
    weight,
    output,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    token = tl.load(token_ids + row).to(tl.int64)
    values = tl.load(weight + token * H + columns, mask=columns < H)
    tl.store(output + row * H + columns, values, mask=columns < H)


def _abi(tensor: Tensor) -> _TensorABI:
    if not isinstance(tensor, Tensor):
        raise TypeError("token embedding operands must be tensors")
    return tuple(tensor.shape), tensor.stride(), tensor.dtype, tensor.device


class BF16TokenEmbeddingPrimitive:
    """Gather ``[C]`` token IDs from one caller-owned FP16/BF16 ``[V,H]`` table."""

    def __init__(self, token_ids: Tensor, weight: Tensor, output: Tensor) -> None:
        tensors = token_ids, weight, output
        abis = tuple(_abi(tensor) for tensor in tensors)
        if (
            token_ids.device.type != "cuda"
            or token_ids.dtype is not torch.int64
            or token_ids.ndim != 1
            or weight.device != token_ids.device
            or output.device != token_ids.device
            or weight.dtype not in (torch.bfloat16, torch.float16)
            or output.dtype != weight.dtype
            or weight.ndim != 2
            or output.shape != (token_ids.shape[0], weight.shape[1])
            or min(token_ids.shape) <= 0
            or min(weight.shape) <= 0
            or any(not tensor.is_contiguous() for tensor in tensors)
        ):
            raise ValueError(
                "expected contiguous same-device CUDA int64 [C], matching FP16/BF16 [V,H] and [C,H]"
            )
        self._abis = abis
        self._meta = token_ids.shape[0], weight.shape[0], weight.shape[1]

    @property
    def launch_signature(self) -> str:
        capacity, vocabulary, hidden = self._meta
        return f"c={capacity},v={vocabulary},h={hidden},block={_BLOCK_H},out=bf16"

    def launch(self, token_ids: Tensor, weight: Tensor, output: Tensor) -> None:
        """Enqueue the fixed gather; caller already proved every ID is in range."""

        if tuple(_abi(tensor) for tensor in (token_ids, weight, output)) != self._abis:
            raise ValueError("token embedding operands changed after fixed-ABI binding")
        capacity, _, hidden = self._meta
        _gather[(capacity, triton.cdiv(hidden, _BLOCK_H))](
            token_ids,
            weight,
            output,
            H=hidden,
            BLOCK_H=_BLOCK_H,
            num_warps=4,
        )


__all__ = ["BF16TokenEmbeddingPrimitive"]


@triton.jit  # type: ignore[untyped-decorator]
def _scaled_shard_gather(  # type: ignore[no-untyped-def]
    IDS,
    ACTIVE,
    STARTS,
    ENDS,
    POINTERS,
    SCALE,
    OUTPUT,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    SHARDS: tl.constexpr,
    FP8: tl.constexpr,
    WEIGHT_TYPE: tl.constexpr,
    SB: tl.constexpr,
    WB: tl.constexpr,
):
    lookup = tl.program_id(0)
    row = lookup // HEADS
    if tl.load(ACTIVE + row) != 0:
        token = tl.load(IDS + lookup)
        shard = tl.arange(0, SB)
        start = tl.load(STARTS + shard, shard < SHARDS, 0)
        end = tl.load(ENDS + shard, shard < SHARDS, 0)
        owns = (shard < SHARDS) & (token >= start) & (token < end)
        owner = tl.max(tl.where(owns, shard, -1), 0)
        begin = tl.sum(tl.where(owns, start, 0), 0)
        pointer = tl.load(POINTERS + owner, owner >= 0, 0).to(tl.pointer_type(WEIGHT_TYPE))
        column = tl.program_id(1) * WB + tl.arange(0, WB)
        code = tl.load(
            pointer + (token - begin) * WIDTH + column, (owner >= 0) & (column < WIDTH), 0
        )
        value = fp8_e4m3fn_decode(code).to(tl.float32) if FP8 else code.to(tl.float32)
        value = value * tl.load(SCALE).to(tl.float32)
        tl.store(OUTPUT + lookup * WIDTH + column, value, column < WIDTH)


def bind_scaled_sharded_embedding(
    ids: Tensor,
    shards: tuple[tuple[int, Tensor | QuantizedMatrix], ...],
    scale: Tensor,
    row_activity: Tensor,
    output: Tensor,
) -> PackedOperatorBinding:
    """Gather owned original rows without concatenating or expanding the table.

    ``shards`` contains sorted (global row start, local tensor) pairs. Shards may
    leave gaps owned by other ranks; an active nonowner emits zero for reduction
    by the caller. Inactive output rows remain untouched. FP8 decoding uses the
    format-owned portable byte-carrier implementation, then FP32 direct scaling
    and one output storage rounding. All original shard tensors stay alive for
    the binding lifetime. This primitive does not choose or perform collectives.
    """
    if (
        ids.device.type != "cuda"
        or ids.dtype != torch.int64
        or ids.ndim != 2
        or min(ids.shape) <= 0
    ):
        raise ValueError("scaled embedding IDs must be positive contiguous CUDA INT64 [M,H]")
    if type(shards) is not tuple or not shards:
        raise ValueError("scaled embedding requires an immutable nonempty shard inventory")
    first = shards[0][1]
    if isinstance(first, QuantizedMatrix):
        binder = getattr(
            get_extension(first.encoding.format).module("linear"),
            "bind_scaled_sharded_embedding",
            None,
        )
        if binder is None:
            raise ValueError("packed format has no sharded embedding implementation")
        return cast(PackedOperatorBinding, binder(ids, shards, scale, row_activity, output))
    if any(not isinstance(weight, Tensor) for _, weight in shards):
        raise ValueError("embedding shards must use one storage format")
    scalar_shards = cast(tuple[tuple[int, Tensor], ...], shards)
    if (
        first.ndim != 2
        or min(first.shape) <= 0
        or first.dtype
        not in (
            torch.bfloat16,
            torch.float16,
            torch.float8_e4m3fn,
        )
    ):
        raise ValueError("scaled embedding weights require FP16/BF16/E4M3 matrices")
    width = first.shape[1]
    operands = (ids, scale, row_activity, output, *(weight for _, weight in scalar_shards))
    if any(t.device != ids.device or not t.is_contiguous() for t in operands[:4]) or any(
        not weight.is_contiguous() for _, weight in scalar_shards
    ):
        raise ValueError("scaled embedding operands must be contiguous on the execution device")
    if (
        scale.numel() != 1
        or scale.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or row_activity.dtype != torch.int32
        or row_activity.shape != (ids.shape[0],)
        or output.dtype not in (torch.float16, torch.bfloat16)
        or output.shape != (ids.shape[0], ids.shape[1] * width)
    ):
        raise ValueError("scaled embedding scale/activity/output geometry differs")
    scalar = float(scale.item())
    if not 0 < scalar < float("inf"):
        raise ValueError("scaled embedding direct scale must be finite and positive")
    previous_end = 0
    for start, weight in scalar_shards:
        if (
            type(start) is not int
            or start < previous_end
            or weight.ndim != 2
            or weight.shape[0] <= 0
            or weight.shape[1] != width
            or weight.dtype != first.dtype
            or start + weight.shape[0] >= 2**63
        ):
            raise ValueError(
                "embedding shards require sorted disjoint INT64 ranges and uniform rows"
            )
        previous_end = start + weight.shape[0]
    if any(
        t is not output and t.untyped_storage().data_ptr() == output.untyped_storage().data_ptr()
        for t in operands
    ):
        raise ValueError("embedding output may not alias lookup inputs or weight storage")
    starts, ends, pointers = (
        torch.tensor(values, dtype=torch.int64, device=ids.device)
        for values in (
            [start for start, _ in scalar_shards],
            [start + weight.shape[0] for start, weight in scalar_shards],
            [embedding_pointer(weight, ids.device) for _, weight in scalar_shards],
        )
    )
    fp8 = first.dtype == torch.float8_e4m3fn
    weight_type = (
        tl.uint8 if fp8 else (tl.bfloat16 if first.dtype == torch.bfloat16 else tl.float16)
    )
    call = _PackedLaunch.compile(
        _scaled_shard_gather,
        (ids, row_activity, starts, ends, pointers, scale, output),
        (
            ids.shape[1],
            width,
            len(shards),
            fp8,
            weight_type,
            triton.next_power_of_2(len(shards)),
            256,
        ),
        (ids.numel(), triton.cdiv(width, 256)),
        {"num_warps": 4, "enable_fp_fusion": False},
    )
    return PackedOperatorBinding(output, (call,), scratch=(*operands[4:], starts, ends, pointers))
