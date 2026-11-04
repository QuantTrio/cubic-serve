"""Integer-only Marlin repacking over N-interleaved or K-packed source words."""

from __future__ import annotations

import math
from typing import Literal

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from torch import Tensor

from cubic_serve.ops.cuda.native import CudaScalar

from .compiler import compile_module


def repack_words(
    source: Tensor, *, bits: int, axis: Literal["n", "k"] = "n", output: Tensor | None = None
) -> Tensor:
    """Preserve leading batch/expert order; never reconstruct floating weights."""
    if (
        bits not in (4, 8)
        or axis not in ("n", "k")
        or source.dtype != torch.int32
        or source.ndim < 2
        or not source.is_cuda
        or not source.is_contiguous()
    ):
        raise ValueError("Marlin repack requires contiguous CUDA packed integer words")
    rows, words = source.shape[-2:]
    k = rows * (32 // bits) if axis == "k" else rows
    n = words if axis == "k" else words * (32 // bits)
    if not k or not n or k % 16 or n % 64:
        raise ValueError("Marlin repack geometry must close its 16 by 64 code tiles")
    shape = (*source.shape[:-2], k // 16, n * bits // 2)
    if output is None:
        output = torch.empty(shape, dtype=torch.int32, device=source.device)
    elif (
        output.shape != shape
        or output.dtype != torch.int32
        or output.device != source.device
        or not output.is_contiguous()
    ):
        raise ValueError("Marlin repack destination differs from its packed layout")
    if (
        source.data_ptr() < output.data_ptr() + output.numel() * output.element_size()
        and output.data_ptr() < source.data_ptr() + source.numel() * source.element_size()
    ):
        raise ValueError("Marlin repack source and destination must not overlap")
    code = f"""#include "repack.cuh"
    template __global__ void marlin::awq_marlin_repack_kernel<256,{bits},false>(
        const uint32_t*,uint32_t*,int,int);"""
    if axis == "k":
        code = f"""#include "gptq_repack.cuh"
        template __global__ void marlin::gptq_marlin_repack_kernel<256,{bits},false,false>(
            const uint32_t*,const uint32_t*,uint32_t*,int,int);"""
    module, entry = compile_module(code, source.device)
    inputs = source.reshape(-1, rows, words)
    outputs = output.reshape(-1, k // 16, n * bits // 2)
    for index in range(inputs.shape[0]):
        module.bind(
            entry,
            (inputs[index],)
            + ((None,) if axis == "k" else ())
            + (outputs[index], CudaScalar("i32", k), CudaScalar("i32", n)),
            operator_id="cubic.cuda.marlin.repack.v1",
            grid=(math.ceil(k / 16), 1, 1),
            block=(256, 1, 1),
            shared_bytes=8 * 16 * 64 * bits // 8,
        ).launch()
    return output


def repack_signed_codes(
    source: Tensor, *, bits: int, fragment_bits: int, offset_binary: bool = False,
    legacy_words: bool = False, logical_k: int | None = None,
    activation_8bit: bool = False, source_group_in: int = 0,
    padded_group_in: int = 0, output: Tensor | None = None,
    zero_points: Tensor | None = None, source_bits: int | None = None,
) -> Tensor:
    """Reorder row-K packed signed codes into MMA order without widening storage."""
    input_bits = bits if source_bits is None else source_bits
    centered = zero_points is not None
    if (
        source.ndim != 2
        or source.dtype != torch.uint8
        or not source.is_cuda
        or not source.is_contiguous()
        or not 2 <= bits <= 8
        or not 2 <= input_bits <= 8
        or fragment_bits not in (4, 8)
        or bits > fragment_bits
        or (centered and (not activation_8bit or bits != input_bits + 1
                          or not source_group_in or not padded_group_in))
    ):
        raise ValueError("signed repack requires compact bytes and a legal fragment width")
    n, row_bytes = source.shape
    if logical_k is None and row_bytes * 8 % input_bits:
        raise ValueError("packed row does not close its code bit width")
    source_k = row_bytes * 8 // input_bits if logical_k is None else logical_k
    expected_bytes = (
        math.ceil(source_k / (32 // input_bits)) * 4
        if legacy_words else math.ceil(source_k * input_bits / 8)
    )
    if row_bytes != expected_bytes or (legacy_words and not offset_binary and not activation_8bit):
        raise ValueError("packed source row stride disagrees with logical K")
    groups = math.ceil(source_k / source_group_in) if source_group_in else 0
    if centered and (
        zero_points is None or zero_points.dtype != torch.int32
        or not zero_points.is_cuda or not zero_points.is_contiguous()
        or zero_points.device != source.device
        or zero_points.shape != (
            math.ceil(n / (32 // input_bits)) if legacy_words
            else math.ceil(n * input_bits / 32), groups
        )
    ):
        raise ValueError("centered A8 repack requires original packed group zero points")
    if bool(source_group_in) != bool(padded_group_in) or (
        source_group_in and (
            not activation_8bit or source_group_in < 1
            or padded_group_in < source_group_in or padded_group_in % 32
        )
    ):
        raise ValueError("A8 group padding requires original groups and complete K32 tiles")
    k = (
        math.ceil(source_k / source_group_in) * padded_group_in
        if source_group_in else source_k
    )
    tile_k, tile_n = (32, 32) if activation_8bit else (16, 64)
    if not n or not k or n % tile_n or k % tile_k:
        raise ValueError("packed MMA layout requires complete kernel tiles")
    shape = (k // 16, n * bits // 2)
    if output is None:
        output = torch.empty(shape, device=source.device, dtype=torch.int32)
    elif (output.shape != shape or output.dtype != torch.int32
          or output.device != source.device or not output.is_contiguous()):
        raise ValueError("signed repack destination differs from its MMA layout")
    _pack_signed_codes[(triton.cdiv(output.numel(), 256),)](
        source, source if zero_points is None else zero_points, output,
        n, k, source_k, input_bits, bits,
        fragment_bits, offset_binary, legacy_words, activation_8bit,
        source_group_in, padded_group_in, row_bytes, centered, groups,
    )
    return output


@triton.jit
def _source_codes(S, Z, code, N: tl.constexpr, K: tl.constexpr,
                  SOURCE_K: tl.constexpr, B: tl.constexpr,
                  OUT_B: tl.constexpr, C: tl.constexpr,
                  OFFSET_BINARY: tl.constexpr = False, LEGACY_WORDS: tl.constexpr = False,
                  ACTIVATION_8BIT: tl.constexpr = False,
                  SOURCE_GROUP: tl.constexpr = 0, PADDED_GROUP: tl.constexpr = 0,
                  ROW_BYTES: tl.constexpr = 0, CENTER_ZERO: tl.constexpr = False,
                  GROUPS: tl.constexpr = 0):
    tile = code // 1024
    pos = code % 1024
    if ACTIVATION_8BIT:
        word = pos // (32 // C)
        lane = pos % (32 // C)
        if C == 4:
            thread = word // 4
            warp = word % 4
            row = thread % 4 * 4 + lane // 2 + lane % 2 * 16
        else:
            thread = word // 8
            warp = word % 8 // 2
            row = thread % 4 * 4 + word % 2 * 16 + lane
        col = warp // 2 * 16 + thread // 4 + warp % 2 * 8
        n = (tile % (N // 32)) * 32 + col
        k = (tile // (N // 32)) * 32 + row
    else:
        word = pos // (32 // C)
        lane = pos % (32 // C)
        if C == 4:
            thread = word // 4
            warp = word % 4
            index = (lane % 4) * 2 + lane // 4
            col = warp * 16 + thread // 4 + (index // 4) * 8
            i = index % 4
        else:
            thread = word // 8
            warp = (word % 8) // 2
            col = warp * 16 + thread // 4 + (word % 2) * 8
            i = (lane % 2) * 2 + lane // 2
        row = thread % 4 * 2 + (i // 2) * 8 + i % 2
        n = (tile % (N // 64)) * 64 + col
        k = (tile // (N // 64)) * 16 + row
    if SOURCE_GROUP:
        source_k = k // PADDED_GROUP * SOURCE_GROUP + k % PADDED_GROUP
        valid_group = k % PADDED_GROUP < SOURCE_GROUP
    else:
        source_k = k
        valid_group = True
    bit = (
        source_k // (32 // B) * 32 + source_k % (32 // B) * B
        if LEGACY_WORDS else source_k * B
    )
    stride: tl.constexpr = ROW_BYTES if ROW_BYTES else K * B // 8
    valid = (code < N * K) & valid_group & (source_k < SOURCE_K)
    a = tl.load(S + n * stride + bit // 8, valid, 0).to(tl.uint32)
    b = tl.load(
        S + n * stride + tl.minimum(bit // 8 + 1, stride - 1), valid, 0
    ).to(tl.uint32)
    value = ((a | (b << 8)) >> ((bit % 8).to(tl.int32))) & ((1 << B) - 1)
    value = tl.where(valid, value, 1 << (B - 1))
    if CENTER_ZERO:
        zero_bit = (
            n // (32 // B) * 32 + n % (32 // B) * B
            if LEGACY_WORDS else n * B
        )
        zero_word = zero_bit // 32
        zero_group = source_k // SOURCE_GROUP
        low = tl.load(Z + zero_word * GROUPS + zero_group, valid, 0).to(tl.uint32)
        high = tl.load(Z + (zero_word + 1) * GROUPS + zero_group,
                       valid & (zero_bit % 32 + B > 32), 0).to(tl.uint32)
        zero = ((low.to(tl.uint64) | (high.to(tl.uint64) << 32))
                >> (zero_bit % 32).to(tl.int32)) & ((1 << B) - 1)
        centered = value.to(tl.int32) - zero.to(tl.int32) + (1 << (OUT_B - 1))
        centered = tl.where(valid, centered, 1 << (OUT_B - 1)).to(tl.uint32)
        return centered if OFFSET_BINARY else centered ^ (1 << (OUT_B - 1))
    return value if OFFSET_BINARY else value ^ (1 << (B - 1))


@triton.jit
def _pack_signed_codes(S, Z, D, N: tl.constexpr, K: tl.constexpr,
                       SOURCE_K: tl.constexpr, B: tl.constexpr,
                       OUT_B: tl.constexpr, C: tl.constexpr,
                       OFFSET_BINARY: tl.constexpr = False, LEGACY_WORDS: tl.constexpr = False,
                       ACTIVATION_8BIT: tl.constexpr = False,
                       SOURCE_GROUP: tl.constexpr = 0, PADDED_GROUP: tl.constexpr = 0,
                       ROW_BYTES: tl.constexpr = 0, CENTER_ZERO: tl.constexpr = False,
                       GROUPS: tl.constexpr = 0):
    # Packed bit offsets can exceed INT32 even when the byte tensor does not.
    w = tl.program_id(0).to(tl.int64) * 256 + tl.arange(0, 256)
    first = w * 32 // OUT_B
    acc = tl.full((256,), 0, tl.uint32)
    for i in range(triton.cdiv(32, OUT_B) + 1):
        code = first + i
        off = (code * OUT_B - w * 32).to(tl.int32)
        v = _source_codes(
            S, Z, code, N, K, SOURCE_K, B, OUT_B, C, OFFSET_BINARY,
            LEGACY_WORDS, ACTIVATION_8BIT, SOURCE_GROUP, PADDED_GROUP,
            ROW_BYTES, CENTER_ZERO, GROUPS,
        )
        v = tl.where(off < 0, v >> (-off), v << off)
        acc |= tl.where(off < 32, v, 0)
    tl.store(D + w, acc, w < N * K * OUT_B // 32)


@triton.jit  # type: ignore[untyped-decorator]
def _pad_routes(  # type: ignore[no-untyped-def]
    COUNTS,
    STARTS,
    BLOCK_STARTS,
    TOTAL,
    ROWS,
    PADDED,
    EXPERTS,
    TOKENS,
    ROUTES: tl.constexpr,
    BM: tl.constexpr,
):
    expert = tl.program_id(0)
    count = tl.load(COUNTS + expert)
    start = tl.load(STARTS + expert)
    block = tl.load(BLOCK_STARTS + expert)
    lane = tl.arange(0, BM)
    for b in range(tl.cdiv(count, BM)):
        row = b * BM + lane
        route = tl.load(ROWS + start + row, row < count, ROUTES)
        tl.store(PADDED + (block + b) * BM + lane, route)
        tl.store(EXPERTS + block + b, expert)
    if expert == 0:
        tl.store(TOKENS, tl.load(TOTAL) * BM)


@triton.jit
def _merge_dense_partials(
    partials,
    output,
    m: tl.constexpr,
    n: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    partitions: tl.constexpr,
    block: tl.constexpr,
    output_ns: tl.constexpr = (),
    physical_ns: tl.constexpr = (),
    routed: tl.constexpr = False,
    int8_activation: tl.constexpr = False,
):
    """Unpack native accumulator fragments; fixed ascending FP32 K fold.

    The A8 MMA epilogue has a different fragment layout from the FP16/BF16
    epilogue. Only real output rows are consumed or written.
    """
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    row, column = offsets // n, offsets % n
    destination = row
    data = partials
    if routed:
        # Only the current route prefix owns initialized partials. Capacity
        # beyond that prefix and padding entries have no output identity.
        data = partials[0]
        count = tl.load(partials[2])
        destination = tl.load(partials[1] + row, row < count, m)
    active = (destination >= 0) & (destination < m)
    tile = (row // block_m) * (n // block_n) + column // block_n
    r, c = row % block_m, column % block_n
    if int8_activation:
        # In the A8 epilogue each thread owns pairs of adjacent columns.
        # Its four float registers map to columns c, c+1, c+8, c+9.
        lane = 32 * (c // 32) + 4 * (r % 8) + (c % 8) // 2
        fragment = 16 * (r // 16) + 8 * ((c % 32) // 16) + 4 * ((c % 16) // 8)
        fragment += 2 * ((r % 16) // 8) + c % 2
        local = (block_n * (fragment // 4) + lane) * 4 + fragment % 4
    elif block_m == 8:
        lane = (r // 2) + (c % 8) * 4 + (c // 64) * 32
        fragment = c % 64 // 16
        local = fragment * (block_n // 64 * 32) * 4 + lane * 4 + (c % 16 // 8) * 2 + r % 2
    else:
        lane = (r % 8) * 4 + (c % 8) // 2 + (c // 64) * 32
        fragment = (r // 16) * 8 + (c % 64 // 16) * 2 + (c % 16 // 8)
        local = fragment * (block_n // 64 * 32) * 4 + lane * 4 + (r % 16 // 8) * 2 + c % 2
    accumulator = tl.full((block,), 0, tl.float32)
    for partition in range(partitions):
        value = tl.load(
            data + (tile * partitions + partition) * block_m * block_n + local, active, 0
        )
        accumulator = accumulator + value
    if len(output_ns):
        first = 0
        for group in tl.static_range(len(output_ns)):
            width = output_ns[group]
            valid = active & (column >= first) & (column < first + width)
            tl.store(output[group] + destination * width + column - first, accumulator, valid)
            first += physical_ns[group]
    else:
        tl.store(output + destination * n + column, accumulator, active)
