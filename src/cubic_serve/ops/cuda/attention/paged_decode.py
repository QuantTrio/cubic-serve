"""Fixed-address FP16/BF16 paged-GQA decode and KV page mutation primitives."""

from __future__ import annotations

from typing_extensions import TypeAliasType

import hashlib
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, NamedTuple, cast

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from torch import Tensor
from triton.compiler.compiler import CompiledKernel  # type: ignore[import-untyped]
from triton.language.extra import libdevice  # type: ignore[import-untyped]

from cubic_serve.ops.cuda.activation import exp_nonpositive, softmax_type
from cubic_serve.ops.cuda.native import NativeLaunch
from cubic_serve.ops.cuda.compiled_artifact import (
    CompiledArtifactIdentity,
    LaunchReceipt,
    TensorBinding,
    bind_tensor,
    project_triton_artifact,
    validate_launch_receipt,
)
from cubic_serve.ops.strategies import attention_precision
from cubic_serve.ops.cuda.launch_cost import prediction_interval
from cubic_serve.ops.cuda.cache_quantization import (
    PagedKVEncoding, PagedKVReadWindow, cache_group_indices, load_cache_group,
)
from cubic_serve.runtime.cache.quantization import (
    CacheDType, CacheGroupAxis, slot_storage_contiguous, storage_regions_overlap,
)

_ATTN_TILE, _COPY_TILE = 16, 256
_KV_SPLITS = 32
BF16_PAGED_GQA_OPERATOR_ID = "cubic.cuda.bf16-paged-gqa.triton.v2"
FP16_PAGED_GQA_OPERATOR_ID = "cubic.cuda.fp16-paged-gqa.triton.v2"
_TensorABI = TypeAliasType("_TensorABI", tuple[tuple[int, ...], tuple[int, ...], torch.dtype, torch.device])
PAGED_GQA_TREE_ABI = {
    "id": "cubic.paged-gqa.online-activation.v3",
    "block_n": _ATTN_TILE,
    "iteration_bound": "runtime-sequence-length",
    "qk": "fp32-reassociate",
    "softmax": "activation-exponent-fp32-online-max-sum",
    "kv_partition": "fixed-count-contiguous-splits",
    "kv_splits": _KV_SPLITS,
    "split_merge": "fp32-online-last-arrival-fixed-order",
    "pv": "fp32-reassociate",
    "scalar": "sub_rn,expf,div_rn,mul_rn,add_rn,bf16-rne",
}
@triton.jit
def _rn_exp(value):  # type: ignore[no-untyped-def]
    return libdevice.exp(value)
@triton.jit
def _rn_div(left, right):  # type: ignore[no-untyped-def]
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;", "=f,f,f", [left, right],
        dtype=tl.float32, is_pure=True, pack=1,
    )
@triton.jit
def _rn_mul(left, right):  # type: ignore[no-untyped-def]
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=f,f,f", [left, right],
        dtype=tl.float32, is_pure=True, pack=1,
    )
@triton.jit
def _rn_add(left, right):  # type: ignore[no-untyped-def]
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;", "=f,f,f", [left, right],
        dtype=tl.float32, is_pure=True, pack=1,
    )
@triton.jit
def _rn_sub(left, right):  # type: ignore[no-untyped-def]
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;", "=f,f,f", [left, right],
        dtype=tl.float32, is_pure=True, pack=1,
    )
@triton.jit  # type: ignore[untyped-decorator]
def _scalar_abi_probe(  # type: ignore[no-untyped-def]
    left,
    right,
    addend,
    output,
    ITEMS: tl.constexpr,
    TILE: tl.constexpr,
):
    offsets = tl.arange(0, TILE)
    mask = offsets < ITEMS
    lhs = tl.load(left + offsets, mask=mask)
    rhs = tl.load(right + offsets, mask=mask)
    extra = tl.load(addend + offsets, mask=mask)
    product = _rn_mul(lhs, rhs)
    base = offsets * 5
    tl.store(output + base, _rn_exp(lhs), mask=mask)
    tl.store(output + base + 1, _rn_div(lhs, rhs), mask=mask)
    tl.store(output + base + 2, product, mask=mask)
    tl.store(output + base + 3, _rn_add(product, extra), mask=mask)
    tl.store(output + base + 4, _rn_sub(lhs, extra), mask=mask)
@triton.jit  # type: ignore[untyped-decorator]
def _copy_page(  # type: ignore[no-untyped-def]
    key,
    value,
    written,
    source_pages,
    destination_pages,
    PAGE_ITEMS: tl.constexpr,
    MASK_ITEMS: tl.constexpr,
    TILE: tl.constexpr,
    VALUE_ITEMS: tl.constexpr = 0,
    KEY_STRIDE: tl.constexpr = 0,
    VALUE_STRIDE: tl.constexpr = 0,
    MASK_STRIDE: tl.constexpr = 0,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.program_id(1) * TILE + tl.arange(0, TILE)
    source = tl.load(source_pages + row).to(tl.int64)
    destination = tl.load(destination_pages + row).to(tl.int64)
    copy = source != destination
    source_kv = source * (KEY_STRIDE if KEY_STRIDE else PAGE_ITEMS) + offsets
    destination_kv = destination * (KEY_STRIDE if KEY_STRIDE else PAGE_ITEMS) + offsets
    kv_mask = (offsets < PAGE_ITEMS) & copy
    tl.store(key + destination_kv, tl.load(key + source_kv, mask=kv_mask), mask=kv_mask)
    value_items: tl.constexpr = VALUE_ITEMS if VALUE_ITEMS else PAGE_ITEMS
    value_mask = (offsets < value_items) & copy
    tl.store(value + destination * (VALUE_STRIDE if VALUE_STRIDE else value_items) + offsets,
             tl.load(value + source * (VALUE_STRIDE if VALUE_STRIDE else value_items) + offsets,
                     mask=value_mask), mask=value_mask)
    source_mask = source * (MASK_STRIDE if MASK_STRIDE else MASK_ITEMS) + offsets
    destination_mask = destination * (MASK_STRIDE if MASK_STRIDE else MASK_ITEMS) + offsets
    mask = (offsets < MASK_ITEMS) & copy
    tl.store(
        written + destination_mask,
        tl.load(written + source_mask, mask=mask),
        mask=mask,
    )
@triton.jit  # type: ignore[untyped-decorator]
def _copy_page_field(  # type: ignore[no-untyped-def]
    data, source_pages, destination_pages, ITEMS: tl.constexpr, TILE: tl.constexpr,
    STRIDE: tl.constexpr = 0,
):
    row = tl.program_id(0)
    offset = tl.program_id(1) * TILE + tl.arange(0, TILE)
    source = tl.load(source_pages + row).to(tl.int64)
    destination = tl.load(destination_pages + row).to(tl.int64)
    mask = (source != destination) & (offset < ITEMS)
    value = tl.load(data + source * (STRIDE if STRIDE else ITEMS) + offset, mask, 0)
    tl.store(data + destination * (STRIDE if STRIDE else ITEMS) + offset, value, mask)


@triton.jit  # type: ignore[untyped-decorator]
def _write_token(  # type: ignore[no-untyped-def]
    key,
    value,
    written,
    destination_pages,
    token_offsets,
    new_key,
    new_value,
    LAYERS: tl.constexpr,
    LAYER: tl.constexpr,
    PAGE_BLOCK: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TILE: tl.constexpr,
    KEY_STRIDE: tl.constexpr = 0, VALUE_STRIDE: tl.constexpr = 0,
    WRITTEN_STRIDE: tl.constexpr = 0,
):
    row = tl.program_id(0).to(tl.int64)
    flat = tl.program_id(1) * TILE + tl.arange(0, TILE)
    destination = tl.load(destination_pages + row).to(tl.int64)
    token = tl.load(token_offsets + row).to(tl.int64)
    head, dim = flat // HEAD_DIM, flat % HEAD_DIM
    mask = flat < KV_HEADS * HEAD_DIM
    cache = (
        (LAYER * PAGE_BLOCK + token) * KV_HEADS + head
    ) * HEAD_DIM + dim
    incoming = (row * KV_HEADS + head) * HEAD_DIM + dim
    dense_stride: tl.constexpr = LAYERS * PAGE_BLOCK * KV_HEADS * HEAD_DIM
    tl.store(key + destination * (KEY_STRIDE if KEY_STRIDE else dense_stride) + cache,
             tl.load(new_key + incoming, mask=mask), mask=mask)
    tl.store(value + destination * (VALUE_STRIDE if VALUE_STRIDE else dense_stride) + cache,
             tl.load(new_value + incoming, mask=mask), mask=mask)
    bit = destination * (WRITTEN_STRIDE if WRITTEN_STRIDE else LAYERS * PAGE_BLOCK) + LAYER * PAGE_BLOCK + token
    tl.store(written + bit + flat, 1, mask=flat == 0)
@triton.jit  # type: ignore[untyped-decorator]
def _load_attention_cache(data, scales, dense_offsets, page_layer, token, head, channel,
                          incoming, tail, start_positions, start_rows, sources, row, positions,
                          mask, PAGE_BLOCK: tl.constexpr, KV_HEADS: tl.constexpr,
                          HEAD_DIM: tl.constexpr, GROUP: tl.constexpr,
                          COLUMN: tl.constexpr, OUTPUT: tl.constexpr,
                          LAYERS: tl.constexpr, LAYER: tl.constexpr,
                          WINDOW: tl.constexpr, STRIDES: tl.constexpr = (), SIDE: tl.constexpr = 0):
    native_mask = tl.full((), False, tl.int1)
    tail_mask = tl.full((), False, tl.int1)
    if WINDOW:
        start = tl.load(start_positions + row)
        first_row = tl.load(start_rows + row)
        native_mask = positions >= start
        if GROUP:
            query_position = start + row - first_row
            sealed_before_query = (query_position // GROUP * GROUP if COLUMN else query_position)
            native_mask &= positions >= sealed_before_query
            if COLUMN:
                tail_mask = (positions >= sealed_before_query) & (positions < start)
    sealed_mask = mask & ~native_mask & ~tail_mask
    if GROUP:
        group, member = cache_group_indices(
            token, head, channel, PAGE_BLOCK, KV_HEADS, HEAD_DIM, GROUP, COLUMN)
        if COLUMN:
            groups_per_page: tl.constexpr = KV_HEADS * HEAD_DIM * triton.cdiv(PAGE_BLOCK, GROUP)
        else:
            groups_per_page: tl.constexpr = PAGE_BLOCK * KV_HEADS * triton.cdiv(HEAD_DIM, GROUP)
        if len(STRIDES):
            physical_page = page_layer // LAYERS
            group += (page_layer % LAYERS) * groups_per_page
            result = load_cache_group(data + physical_page * STRIDES[SIDE],
                                     scales + physical_page * STRIDES[2 + SIDE],
                                     group, member, sealed_mask, GROUP, 8, True, OUTPUT)
        else:
            group += page_layer * groups_per_page
            result = load_cache_group(data, scales, group, member, sealed_mask,
                                     GROUP, 8, True, OUTPUT)
    else:
        if len(STRIDES):
            physical_page = page_layer // LAYERS
            dense_offsets += physical_page * (
                STRIDES[SIDE] - LAYERS * PAGE_BLOCK * KV_HEADS * HEAD_DIM)
        result = tl.load(data + dense_offsets, sealed_mask, other=0)
    if WINDOW:
        incoming_row = first_row + positions - start
        fresh = tl.load(incoming + (incoming_row * KV_HEADS + head) * HEAD_DIM + channel,
                        mask & native_mask, other=0)
        result = tl.where(native_mask, fresh, result)
        if GROUP and COLUMN:
            source = tl.load(sources + row).to(tl.int64)
            tail_stride: tl.constexpr = STRIDES[4 + SIDE] if len(STRIDES) else LAYERS * GROUP * KV_HEADS * HEAD_DIM
            tail_offset = (source * tail_stride +
                           ((LAYER * GROUP + positions % GROUP) * KV_HEADS + head) * HEAD_DIM + channel)
            old = tl.load(tail + tail_offset, mask & tail_mask, other=0)
            result = tl.where(tail_mask, old, result)
    return result


@triton.jit  # type: ignore[untyped-decorator]
def _decode_body(  # type: ignore[no-untyped-def]
    query,
    key,
    value,
    page_table,
    seq_lens,
    output,
    partial,
    statistics,
    arrivals,
    selected_indices,
    key_scales,
    value_scales,
    incoming_key, incoming_value, key_tail, value_tail, start_positions, start_rows, sources,
    LAYERS: tl.constexpr,
    LAYER: tl.constexpr,
    PAGE_BLOCK: tl.constexpr,
    TABLE_PAGES: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    ACT_TYPE: tl.constexpr,
    SELECTED_WIDTH: tl.constexpr = 0,
    K_GROUP: tl.constexpr = 0,
    V_GROUP: tl.constexpr = 0,
    K_COLUMN: tl.constexpr = True,
    V_COLUMN: tl.constexpr = False,
    WINDOW: tl.constexpr = False,
    CACHE_STRIDES: tl.constexpr = (),
):
    row = tl.program_id(0).to(tl.int64)
    query_head = tl.program_id(1).to(tl.int64)
    kv_head = query_head // (QUERY_HEADS // KV_HEADS)
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < HEAD_DIM
    query_offsets = (row * QUERY_HEADS + query_head) * HEAD_DIM + dims
    q = tl.load(query + query_offsets, mask=dim_mask).to(tl.float32)
    length = tl.load(seq_lens + row)
    split = tl.program_id(2).to(tl.int64)
    span = tl.cdiv(length, KV_SPLITS)
    split_start = split * span
    split_end = tl.minimum(split_start + span, length)
    maximum, denominator = -float("inf"), 0.0
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    # The table width is an address-capacity ABI, not executable work.  A
    # captured long-context graph must visit only the tiles visible to this
    # row; otherwise C1 decode cost scales with the configured maximum context.
    for start in tl.range(split_start, split_end, BLOCK_N):
        positions = start + tl.arange(0, BLOCK_N)
        valid = positions < split_end
        if SELECTED_WIDTH:
            positions = tl.load(
                selected_indices + row * SELECTED_WIDTH + positions, valid, 0
            ).to(tl.int64)
        pages = tl.load(
            page_table + row * TABLE_PAGES + positions // PAGE_BLOCK,
            mask=valid,
            other=0,
        ).to(tl.int64)
        tokens = positions % PAGE_BLOCK
        offsets = (
            ((pages[:, None] * LAYERS + LAYER) * PAGE_BLOCK + tokens[:, None]) * KV_HEADS + kv_head
        ) * HEAD_DIM + dims[None, :]
        load_mask = valid[:, None] & dim_mask[None, :]
        keys = _load_attention_cache(
            key, key_scales, offsets, pages[:, None] * LAYERS + LAYER,
            tokens[:, None], kv_head, dims[None, :],
            incoming_key, key_tail, start_positions, start_rows, sources,
            row, positions[:, None], load_mask,
            PAGE_BLOCK, KV_HEADS, HEAD_DIM, K_GROUP, K_COLUMN,
            query.dtype.element_ty, LAYERS, LAYER, WINDOW,
            STRIDES=CACHE_STRIDES, SIDE=0).to(tl.float32)
        logits = tl.where(
            valid,
            _rn_mul(tl.sum(_rn_mul(keys, q[None, :]), 1), SCALE),
            -float("inf"),
        )
        next_maximum = tl.maximum(maximum, tl.max(logits, 0))
        old_scale = _rn_exp(_rn_sub(maximum, next_maximum))
        probabilities = exp_nonpositive(_rn_sub(logits, next_maximum), ACT_TYPE).to(tl.float32)
        values = _load_attention_cache(
            value, value_scales, offsets, pages[:, None] * LAYERS + LAYER,
            tokens[:, None], kv_head, dims[None, :],
            incoming_value, value_tail, start_positions, start_rows, sources,
            row, positions[:, None], load_mask,
            PAGE_BLOCK, KV_HEADS, HEAD_DIM, V_GROUP, V_COLUMN,
            query.dtype.element_ty, LAYERS, LAYER, WINDOW,
            STRIDES=CACHE_STRIDES, SIDE=1).to(tl.float32)
        accumulator = _rn_add(
            _rn_mul(accumulator, old_scale),
            tl.sum(_rn_mul(probabilities[:, None], values), 0),
        )
        denominator = _rn_add(_rn_mul(denominator, old_scale), tl.sum(probabilities, 0))
        maximum = next_maximum
    if KV_SPLITS == 1:
        tl.store(output + query_offsets, _rn_div(accumulator, denominator), mask=dim_mask)
    else:
        base = ((row * QUERY_HEADS + query_head) * KV_SPLITS + split) * HEAD_DIM
        stat = ((row * QUERY_HEADS + query_head) * KV_SPLITS + split) * 2
        tl.store(partial + base + dims, accumulator, mask=dim_mask)
        tl.store(statistics + stat, maximum)
        tl.store(statistics + stat + 1, denominator)
        ticket = tl.atomic_add(
            arrivals + row * QUERY_HEADS + query_head, 1, sem="acq_rel", scope="gpu"
        )
        if ticket == KV_SPLITS - 1:
            merged_maximum, merged_denominator = -float("inf"), 0.0
            merged = tl.zeros((BLOCK_D,), dtype=tl.float32)
            for part in tl.range(0, KV_SPLITS, loop_unroll_factor=1):
                part_base = ((row * QUERY_HEADS + query_head) * KV_SPLITS + part) * HEAD_DIM
                part_stat = ((row * QUERY_HEADS + query_head) * KV_SPLITS + part) * 2
                part_maximum = tl.load(statistics + part_stat)
                part_denominator = tl.load(statistics + part_stat + 1)
                next_maximum = tl.maximum(merged_maximum, part_maximum)
                old_scale = _rn_exp(_rn_sub(merged_maximum, next_maximum))
                part_scale = _rn_exp(_rn_sub(part_maximum, next_maximum))
                merged = _rn_add(
                    _rn_mul(merged, old_scale),
                    _rn_mul(tl.load(partial + part_base + dims, mask=dim_mask), part_scale),
                )
                merged_denominator = _rn_add(
                    _rn_mul(merged_denominator, old_scale),
                    _rn_mul(part_denominator, part_scale),
                )
                merged_maximum = next_maximum
            tl.store(
                output + query_offsets,
                _rn_div(merged, merged_denominator),
                mask=dim_mask,
            )
            tl.atomic_xchg(
                arrivals + row * QUERY_HEADS + query_head, 0, sem="release", scope="gpu"
            )


@triton.jit  # type: ignore[untyped-decorator]
def _decode_qkv_body(  # type: ignore[no-untyped-def]
    query, key, value, page_table, seq_lens, output, partial, statistics, arrivals,
    selected_indices, key_scales, value_scales,
    incoming_key, incoming_value, key_tail, value_tail, start_positions, start_rows, sources,
    LAYERS: tl.constexpr, LAYER: tl.constexpr, PAGE_BLOCK: tl.constexpr,
    TABLE_PAGES: tl.constexpr, QUERY_HEADS: tl.constexpr, KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr, KV_SPLITS: tl.constexpr, ACT_TYPE: tl.constexpr,
    SELECTED_WIDTH: tl.constexpr = 0,
    K_GROUP: tl.constexpr = 0, V_GROUP: tl.constexpr = 0,
    K_COLUMN: tl.constexpr = True, V_COLUMN: tl.constexpr = False,
    WINDOW: tl.constexpr = False,
    CACHE_STRIDES: tl.constexpr = (),
):
    # Group only heads sharing K/V. The grouping and reduction never depend on M.
    row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
    GROUP_SIZE: tl.constexpr = QUERY_HEADS // KV_HEADS
    HEAD_TILES: tl.constexpr = triton.cdiv(GROUP_SIZE, 16)
    kv_head = group // HEAD_TILES
    local_heads = (group % HEAD_TILES) * 16 + tl.arange(0, 16)
    heads = kv_head * GROUP_SIZE + local_heads
    head_mask = local_heads < GROUP_SIZE
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < HEAD_DIM
    q_offsets = (row * QUERY_HEADS + heads[:, None]) * HEAD_DIM + dims[None, :]
    q = tl.load(query + q_offsets, head_mask[:, None] & dim_mask[None, :], other=0)
    length = tl.load(seq_lens + row)
    split = tl.program_id(2).to(tl.int64)
    span = tl.cdiv(length, KV_SPLITS)
    split_start = split * span
    split_end = tl.minimum(split_start + span, length)
    maximum = tl.full((16,), -float("inf"), tl.float32)
    denominator = tl.zeros((16,), tl.float32)
    accumulator = tl.zeros((16, BLOCK_D), tl.float32)
    for start in tl.range(split_start, split_end, BLOCK_N):
        positions = start + tl.arange(0, BLOCK_N)
        valid = positions < split_end
        if SELECTED_WIDTH:
            positions = tl.load(
                selected_indices + row * SELECTED_WIDTH + positions, valid, 0
            ).to(tl.int64)
        pages = tl.load(page_table + row * TABLE_PAGES + positions // PAGE_BLOCK,
                        mask=valid, other=0).to(tl.int64)
        offsets = (((pages * LAYERS + LAYER) * PAGE_BLOCK + positions % PAGE_BLOCK)
                   * KV_HEADS + kv_head) * HEAD_DIM
        k = _load_attention_cache(
            key, key_scales, offsets[None, :] + dims[:, None],
            pages[None, :] * LAYERS + LAYER, (positions % PAGE_BLOCK)[None, :],
            kv_head, dims[:, None],
            incoming_key, key_tail, start_positions, start_rows, sources,
            row, positions[None, :], dim_mask[:, None] & valid[None, :],
            PAGE_BLOCK, KV_HEADS, HEAD_DIM, K_GROUP, K_COLUMN, query.dtype.element_ty,
            LAYERS, LAYER, WINDOW,
            STRIDES=CACHE_STRIDES, SIDE=0)
        # Both MMA inputs remain FP16/BF16; only the dot accumulator is FP32.
        scores = tl.dot(q, k, out_dtype=tl.float32)
        logits = tl.where(valid[None, :], _rn_mul(scores, SCALE), -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(logits, 1))
        old_scale = _rn_exp(_rn_sub(maximum, next_maximum))
        probabilities = exp_nonpositive(
            _rn_sub(logits, next_maximum[:, None]), ACT_TYPE
        ).to(tl.float32)
        v = _load_attention_cache(
            value, value_scales, offsets[:, None] + dims[None, :],
            pages[:, None] * LAYERS + LAYER, (positions % PAGE_BLOCK)[:, None],
            kv_head, dims[None, :],
            incoming_value, value_tail, start_positions, start_rows, sources,
            row, positions[:, None], valid[:, None] & dim_mask[None, :],
            PAGE_BLOCK, KV_HEADS, HEAD_DIM, V_GROUP, V_COLUMN, query.dtype.element_ty,
            LAYERS, LAYER, WINDOW,
            STRIDES=CACHE_STRIDES, SIDE=1)
        weighted = tl.dot(probabilities.to(v.dtype), v, out_dtype=tl.float32)
        # Online softmax normalization and split merge retain their wide safe rule.
        accumulator = _rn_add(_rn_mul(accumulator, old_scale[:, None]), weighted)
        denominator = _rn_add(_rn_mul(denominator, old_scale), tl.sum(probabilities, 1))
        maximum = next_maximum
    if KV_SPLITS == 1:
        tl.store(output + q_offsets, _rn_div(accumulator, denominator[:, None]),
                 head_mask[:, None] & dim_mask[None, :])
    else:
        base = (row * QUERY_HEADS + heads) * KV_SPLITS + split
        tl.store(partial + base[:, None] * HEAD_DIM + dims[None, :], accumulator,
                 head_mask[:, None] & dim_mask[None, :])
        tl.store(statistics + base * 2, maximum, head_mask)
        tl.store(statistics + base * 2 + 1, denominator, head_mask)
        tickets = tl.atomic_add(arrivals + row * QUERY_HEADS + heads, 1,
                                mask=head_mask, sem="acq_rel", scope="gpu")
        # Each head can elect a different last-arriving CTA. Never read a head's
        # scratch until that head has acquired all producer publications.
        last = head_mask & (tickets == KV_SPLITS - 1)
        if tl.sum(last.to(tl.int32), 0) > 0:
            merged_maximum = tl.full((16,), -float("inf"), tl.float32)
            merged_denominator = tl.zeros((16,), tl.float32)
            merged = tl.zeros((16, BLOCK_D), tl.float32)
            for part in tl.range(0, KV_SPLITS, loop_unroll_factor=1):
                part_base = (row * QUERY_HEADS + heads) * KV_SPLITS + part
                part_maximum = tl.load(statistics + part_base * 2, last, other=-float("inf"))
                part_denominator = tl.load(statistics + part_base * 2 + 1, last, other=0)
                next_maximum = tl.maximum(merged_maximum, part_maximum)
                old_scale = _rn_exp(_rn_sub(merged_maximum, next_maximum))
                part_scale = _rn_exp(_rn_sub(part_maximum, next_maximum))
                part_output = tl.load(partial + part_base[:, None] * HEAD_DIM + dims[None, :],
                                      last[:, None] & dim_mask[None, :], other=0)
                merged = _rn_add(_rn_mul(merged, old_scale[:, None]),
                                  _rn_mul(part_output, part_scale[:, None]))
                merged_denominator = _rn_add(_rn_mul(merged_denominator, old_scale),
                                              _rn_mul(part_denominator, part_scale))
                merged_maximum = next_maximum
            tl.store(output + q_offsets, _rn_div(merged, merged_denominator[:, None]),
                     last[:, None] & dim_mask[None, :])
            tl.atomic_xchg(arrivals + row * QUERY_HEADS + heads, 0,
                           mask=last, sem="release", scope="gpu")


@triton.jit
def _decode(query, key, value, page_table, seq_lens, output, partial, statistics,
            arrivals, selected_indices, LAYERS: tl.constexpr, LAYER: tl.constexpr,
            PAGE_BLOCK: tl.constexpr, TABLE_PAGES: tl.constexpr,
            QUERY_HEADS: tl.constexpr, KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
            BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr, SCALE: tl.constexpr,
            KV_SPLITS: tl.constexpr, ACT_TYPE: tl.constexpr, SELECTED_WIDTH: tl.constexpr = 0,
            CACHE_STRIDES: tl.constexpr = ()):
    # The ordinary launch ABI remains unchanged; unused scale operands inline
    # away before code generation. Both formats share the same attention math.
    _decode_body(query, key, value, page_table, seq_lens, output, partial, statistics,
                 arrivals, selected_indices, key, value,
                 key, value, key, value, seq_lens, seq_lens, seq_lens,
                 LAYERS, LAYER, PAGE_BLOCK,
                 TABLE_PAGES, QUERY_HEADS, KV_HEADS, HEAD_DIM, BLOCK_D, BLOCK_N,
                 SCALE, KV_SPLITS, ACT_TYPE, SELECTED_WIDTH, CACHE_STRIDES=CACHE_STRIDES)


@triton.jit
def _decode_qkv(query, key, value, page_table, seq_lens, output, partial, statistics,
                arrivals, selected_indices, LAYERS: tl.constexpr, LAYER: tl.constexpr,
                PAGE_BLOCK: tl.constexpr, TABLE_PAGES: tl.constexpr,
                QUERY_HEADS: tl.constexpr, KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
                BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr, SCALE: tl.constexpr,
                KV_SPLITS: tl.constexpr, ACT_TYPE: tl.constexpr, SELECTED_WIDTH: tl.constexpr = 0,
                CACHE_STRIDES: tl.constexpr = ()):
    _decode_qkv_body(query, key, value, page_table, seq_lens, output, partial, statistics,
                     arrivals, selected_indices, key, value,
                     key, value, key, value, seq_lens, seq_lens, seq_lens,
                     LAYERS, LAYER, PAGE_BLOCK,
                     TABLE_PAGES, QUERY_HEADS, KV_HEADS, HEAD_DIM, BLOCK_D, BLOCK_N,
                     SCALE, KV_SPLITS, ACT_TYPE, SELECTED_WIDTH, CACHE_STRIDES=CACHE_STRIDES)


@triton.jit
def _decode_encoded(query, key, value, page_table, seq_lens, output, partial, statistics,
                    arrivals, selected_indices, key_scales, value_scales,
                    incoming_key, incoming_value, key_tail, value_tail,
                    start_positions, start_rows, sources, row_activity,
                    LAYERS: tl.constexpr, LAYER: tl.constexpr, PAGE_BLOCK: tl.constexpr,
                    TABLE_PAGES: tl.constexpr, QUERY_HEADS: tl.constexpr,
                    KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr,
                    BLOCK_N: tl.constexpr, SCALE: tl.constexpr, KV_SPLITS: tl.constexpr,
                    ACT_TYPE: tl.constexpr, SELECTED_WIDTH: tl.constexpr,
                    K_GROUP: tl.constexpr, V_GROUP: tl.constexpr,
                    K_COLUMN: tl.constexpr, V_COLUMN: tl.constexpr, NATIVE: tl.constexpr,
                    WINDOW: tl.constexpr, HAS_ACTIVITY: tl.constexpr,
                    CACHE_STRIDES: tl.constexpr = ()):
    if HAS_ACTIVITY:
        row = tl.program_id(0).to(tl.int64)
        if tl.load(row_activity + row) == 0:
            if tl.program_id(2) == 0:
                dims = tl.arange(0, BLOCK_D)
                if NATIVE:
                    group: tl.constexpr = QUERY_HEADS // KV_HEADS
                    tiles: tl.constexpr = triton.cdiv(group, 16)
                    tile = tl.program_id(1)
                    head = tile // tiles * group + tile % tiles * 16 + tl.arange(0, 16)
                    valid = head < (tile // tiles + 1) * group
                    offsets = (row * QUERY_HEADS + head[:, None]) * HEAD_DIM + dims[None, :]
                    tl.store(output + offsets, 0, valid[:, None] & (dims[None, :] < HEAD_DIM))
                else:
                    offsets = (row * QUERY_HEADS + tl.program_id(1)) * HEAD_DIM + dims
                    tl.store(output + offsets, 0, dims < HEAD_DIM)
            return
    if NATIVE:
        _decode_qkv_body(query, key, value, page_table, seq_lens, output, partial, statistics,
                         arrivals, selected_indices, key_scales, value_scales,
                         incoming_key, incoming_value, key_tail, value_tail,
                         start_positions, start_rows, sources, LAYERS, LAYER,
                         PAGE_BLOCK, TABLE_PAGES, QUERY_HEADS, KV_HEADS, HEAD_DIM, BLOCK_D,
                         BLOCK_N, SCALE, KV_SPLITS, ACT_TYPE, SELECTED_WIDTH,
                         K_GROUP, V_GROUP, K_COLUMN, V_COLUMN, WINDOW, CACHE_STRIDES)
    else:
        _decode_body(query, key, value, page_table, seq_lens, output, partial, statistics,
                     arrivals, selected_indices, key_scales, value_scales,
                     incoming_key, incoming_value, key_tail, value_tail,
                     start_positions, start_rows, sources, LAYERS, LAYER,
                     PAGE_BLOCK, TABLE_PAGES, QUERY_HEADS, KV_HEADS, HEAD_DIM, BLOCK_D,
                     BLOCK_N, SCALE, KV_SPLITS, ACT_TYPE, SELECTED_WIDTH,
                     K_GROUP, V_GROUP, K_COLUMN, V_COLUMN, WINDOW, CACHE_STRIDES)


@dataclass(frozen=True)
class PagedDecodeCost:
    """Measured decode and last-arrival merge costs within a witnessed domain.

    Coefficients belong to the bound dtype, head geometry and cache layout.
    Rows are physical graph rows, never public client concurrency. The model
    prices a uniform token population; the selector considers short, full and
    partial token tiles rather than assuming every request is at its limit.
    """

    coefficients: tuple[float, ...]
    error: float
    sms: int
    head_programs: int
    max_rows: int
    max_tokens: int

    def __post_init__(self) -> None:
        if (len(self.coefficients) != 7
                or any(not math.isfinite(v) or v < 0 for v in self.coefficients)
                or not math.isfinite(self.error) or not 0 <= self.error < 1
                or any(type(v) is not int or v <= 0 for v in
                       (self.sms, self.head_programs, self.max_rows, self.max_tokens))):
            raise ValueError("paged decode cost requires a finite fit and measured domain")

    @staticmethod
    def features(rows: int, heads: int, tokens: int, splits: int, sms: int) -> tuple[float, ...]:
        if (any(type(v) is not int or v <= 0 for v in (rows, heads, tokens, splits, sms))
                or splits > _KV_SPLITS or splits & (splits - 1)):
            raise ValueError("paged decode cost requires positive geometry and legal splits")
        span = math.ceil(tokens / splits)
        iterations = math.ceil(span / _ATTN_TILE)
        active_partitions = math.ceil(tokens / span)
        extra_waves = max(0, math.ceil(rows * heads * active_partitions / sms) - 1)
        # Critical-path CTAs pay a complete extra issue wave; average occupancy
        # underprices the partially occupied last wave. Partial token tiles
        # still issue a tile, but have less valid load/exponent work.
        return (
            1.0, float(extra_waves), float(iterations - 1),
            float((iterations - 1) * extra_waves), float(splits - 1),
            span / _ATTN_TILE - (iterations - 1), float(splits > 1),
        )

    def predict(self, rows: int, tokens: int, splits: int) -> float:
        if not 0 < rows <= self.max_rows or not 0 < tokens <= self.max_tokens:
            raise ValueError("paged decode cost prediction is outside its measured domain")
        features = self.features(rows, self.head_programs, tokens, splits, self.sms)
        return sum(a * b for a, b in zip(self.coefficients, features, strict=True))

    def select(
        self, rows: int, tokens: int, *, fallback: int,
        min_gain_percent: float, max_error_percent: float,
    ) -> int:
        if not 0 <= min_gain_percent < 100 or not 0 <= max_error_percent < 100:
            raise ValueError("paged decode cost margins must be percentages below 100")
        self.features(rows, self.head_programs, tokens, fallback, self.sms)
        if (rows > self.max_rows or tokens > self.max_tokens
                or self.error * 100 > max_error_percent):
            return fallback
        limit = min(_KV_SPLITS, max(1, tokens // _ATTN_TILE))
        candidates = tuple(1 << i for i in range(limit.bit_length()))
        if fallback not in candidates:
            return fallback
        populations = {1, tokens}
        boundary = _ATTN_TILE
        while boundary <= tokens:
            populations.update(value for value in (boundary - 1, boundary, boundary + 1)
                               if 0 < value <= tokens)
            boundary *= 2
        times = {p: tuple(self.predict(rows, length, p) for length in sorted(populations))
                 for p in candidates}
        lower = tuple(min(prediction_interval(values[i], self.error)[0]
                          for values in times.values())
                      for i in range(len(populations)))
        # Absolute worst regret prevents tiny short-row latencies from
        # dominating a much larger long-row loss. Uncertainty is charged to
        # both sides, and a near tie keeps the already bound baseline.
        scores = {p: max(prediction_interval(t, self.error)[1] - lo
                         for t, lo in zip(values, lower, strict=True))
                  for p, values in times.items()}
        best = min(candidates, key=lambda p: (scores[p], p))
        return (best if scores[best] < scores[fallback] * (1 - min_gain_percent / 100)
                else fallback)


def paged_gqa_kv_splits(
    capacity: int, *, max_tokens: int | None = None, min_tiles_per_split: int = 16,
    head_programs: int | None = None, multiprocessors: int | None = None,
    full_context: bool = False,
) -> int:
    """Resolve a startup-bound reduction plan for this graph's tensor geometry.

    A bounded selected-token domain need not pay the full long-context split
    count. Start from min_tiles_per_split nominal tiles per partition, rounded
    down to a power of two under the shared maximum. When device resources and
    per-row head programs are known, also cover one SM wave across the graph's
    rows. TP head sharding must not leave the device mostly idle, while larger
    M already supplies parallel work. Bound coverage by available token tiles
    and the shared workspace limit. The numerical witness records the actual
    per-bucket split count; live lengths do not change the bound plan.

    Full-context readers instead bound useful parallelism by head programs,
    SM count and logarithmic context waves. A fixed plan covers the declared
    context envelope without allocating the maximum partial grid to every row.
    """

    if type(capacity) is not int or capacity <= 0:
        raise ValueError("paged-GQA capacity must be a positive integer")
    if type(min_tiles_per_split) is not int or min_tiles_per_split <= 0:
        raise ValueError("paged-GQA minimum partition work must be positive")
    if (head_programs is None) != (multiprocessors is None) or (
        head_programs is not None and (
            type(head_programs) is not int or head_programs <= 0
            or type(multiprocessors) is not int or multiprocessors <= 0
        )
    ):
        raise ValueError("paged-GQA head programs and SM count must both be positive")
    if type(full_context) is not bool:
        raise ValueError("paged-GQA full-context policy must be boolean")
    if full_context and (max_tokens is None or head_programs is None):
        raise ValueError("full-context partitioning requires context and device geometry")
    if max_tokens is None:
        return _KV_SPLITS
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("paged-GQA token domain must be a positive integer")
    budget = max(1, max_tokens // (_ATTN_TILE * min_tiles_per_split))
    work_splits = 1 << (budget.bit_length() - 1)
    if head_programs is not None:
        assert multiprocessors is not None
        # Long-context readers need several waves to cover memory latency.
        # Bound the envelope by tensor work rather than giving every row the
        # maximum split count. Keep selected-token policy unchanged.
        waves = max(1, math.log2(max_tokens / _ATTN_TILE)) if full_context else 1
        coverage = math.ceil(multiprocessors * waves / (capacity * head_programs))
        coverage = 1 << (coverage - 1).bit_length()
        tiles = max(1, max_tokens // _ATTN_TILE)
        coverage = min(coverage, 1 << (tiles.bit_length() - 1))
        work_splits = coverage if full_context else max(work_splits, coverage)
    return min(_KV_SPLITS, work_splits)
def paged_gqa_fingerprint_payload(
    query: Tensor, key: Tensor, page_table: Tensor, *, layer: int, scale: float,
    cache_encoding: PagedKVEncoding | None = None,
    cache_window: PagedKVReadWindow | None = None,
    implementation: Literal["auto", "cuda", "triton"] = "auto",
) -> dict[str, object]:
    """Resolve source, runtime, launch, and numerical-tree identity internally."""
    if implementation not in ("auto", "cuda", "triton"):
        raise ValueError("unknown paged-attention implementation")
    if implementation == "cuda" and (cache_encoding is None or attention_precision() != "auto"):
        raise ValueError("native encoded attention requires native QKV operands")
    source = Path(__file__).read_bytes()
    result = {
        "source": "b2s256:" + hashlib.blake2s(source).hexdigest(),
        "torch": str(torch.__version__),
        "triton": triton.__version__,
        "compiled_cuda": torch._C._cuda_getCompiledVersion(),
        "runtime_cuda": torch.version.cuda,
        "sm": torch.cuda.get_device_capability(query.device),
        "tree": PAGED_GQA_TREE_ABI,
        "dot_operands": attention_precision(),
        "probability_boundary": "v-dtype-before-dot" if attention_precision() == "auto" else "fp32",
        "activation": str(softmax_type(query.dtype, query.device)),
        "activation_source": hashlib.blake2s(
            (Path(__file__).parents[1] / "activation.py").read_bytes()
        ).hexdigest(),
        "launch": {
            "query": (tuple(query.shape), tuple(query.stride()), str(query.dtype)),
            "cache": (tuple(key.shape), tuple(key.stride()), str(key.dtype)),
            "table": (tuple(page_table.shape), tuple(page_table.stride())),
            "layer": layer,
            "scale": scale.hex(),
            "num_warps": 4,
            "num_stages": 2,
        },
    }
    if cache_encoding is not None:
        result["cache_encoding"] = {
            "key": cache_encoding.key.spec.identity,
            "value": cache_encoding.value.spec.identity,
            "rotation": cache_encoding.rotation_contract,
            "native_window": cache_window is not None,
            "source": hashlib.blake2s(
                (Path(__file__).parents[1] / "cache_quantization.py").read_bytes()
            ).hexdigest(),
            "temporal_rule": "strictly-prior-committed-prefix-current-native-v1",
        }
        use_cuda = attention_precision() == "auto" and implementation != "triton"
        launch = cast(dict[str, object], result["launch"])
        launch["implementation"] = "native-mma16816" if use_cuda else "triton"
        launch["num_stages"] = 1 if use_cuda else 2
        if use_cuda:
            root = Path(__file__).parent
            files = (root / "paged_decode_native.py", root / "paged_decode_native.cu",
                     root.parent / "mma.cuh", root.parent / "activation_native.cuh")
            result["native_sources"] = {
                path.name: hashlib.blake2s(path.read_bytes()).hexdigest() for path in files
            }
            result["tree"] = {
                **PAGED_GQA_TREE_ABI,
                "qk_partial_merge": "four-strided-K-warp-partitions; RN-add warp 0,1,2,3",
            }
    return result
def paged_gqa_scalar_probe(
    device: torch.device,
) -> tuple[tuple[Tensor, Tensor, Tensor], Tensor]:
    """Run the exact scalar helpers used by decode on adversarial FP32 inputs."""
    def values(bits: tuple[int, ...]) -> Tensor:
        signed = tuple(item if item < 2**31 else item - 2**32 for item in bits)
        return torch.tensor(signed, dtype=torch.int32, device=device).view(torch.float32)
    left = values(
        (0x3F6F3174, 0x405A531A, 0xC1BFE489, 0x3F000001, 0xBF7FFFFF, 1, 0x41200000, 0x00800000)
    )
    right = values(
        (
            0x3EEC751A,
            0x40490FDB,
            0x40490FDB,
            0x3F7FFFFF,
            0x40000001,
            0x3F800000,
            0x3F000000,
            0x40000000,
        )
    )
    addend = values(
        (0xBF089E8B, 0xBE800001, 0x3EAAAAAB, 0xBF000000, 0x3F000001, 0, 0xBF800000, 0x00800000)
    )
    output = torch.empty((left.numel(), 5), dtype=torch.float32, device=device)
    _scalar_abi_probe[(1,)](left, right, addend, output, ITEMS=left.numel(), TILE=16, num_warps=1)
    return (left, right, addend), output
class FixedLaunchMeta(NamedTuple):
    """Tensor identities and scalar compile-time choices; never owns a tensor."""
    tensors: tuple[_TensorABI, ...]
    layer: int | None = None
    scale: float | None = None


BF16PagedGQATensorBinding = TensorBinding
BF16PagedGQAArtifactIdentity = CompiledArtifactIdentity
BF16PagedGQALaunchReceipt = LaunchReceipt


def _abi(tensor: Tensor) -> _TensorABI:
    return tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device
def _bind(
    *tensors: Tensor, layer: int | None = None, scale: float | None = None
) -> FixedLaunchMeta:
    return FixedLaunchMeta(tuple(_abi(tensor) for tensor in tensors), layer, scale)
def _matches(meta: FixedLaunchMeta, *tensors: Tensor) -> bool:
    return tuple(_abi(tensor) for tensor in tensors) == meta.tensors
def _cache(key: Tensor, value: Tensor, written: Tensor | None = None) -> tuple[int, ...]:
    bad_written = written is not None and (
        written.shape != key.shape[:3]
        or written.dtype is not torch.bool
        or written.device != key.device
        or not slot_storage_contiguous(written)
    )
    if (
        key.device.type != "cuda"
        or key.dtype not in (torch.float16, torch.bfloat16)
        or key.ndim != 5
        or not slot_storage_contiguous(key)
        or min(key.shape) <= 0
        or _abi(value) != _abi(key)
        or bad_written
    ):
        raise ValueError("cache ABI requires contiguous CUDA FP16/BF16 K/V and bool written")
    return tuple(key.shape)
def _index(tensor: Tensor, capacity: int, device: torch.device) -> bool:
    return (
        tensor.shape == (capacity,)
        and tensor.dtype is torch.int32
        and tensor.device == device
        and tensor.is_contiguous()
    )
class PageCOWCopyPrimitive:
    """Copy K/V/written once; a padding row self-copies only private scratch."""
    def __init__(
        self, key: Tensor, value: Tensor, written: Tensor, source: Tensor, destination: Tensor,
        *, page_fields: tuple[Tensor, ...] = (),
        cache_encoding: PagedKVEncoding | None = None,
    ) -> None:
        if cache_encoding is None:
            _cache(key, value, written)
        else:
            pages, layers, block, _, _ = cache_encoding.validate(key, value)
            if (key.device.type != "cuda" or written.shape != (pages, layers, block)
                    or written.device != key.device or written.dtype is not torch.bool
                    or not slot_storage_contiguous(written)):
                raise ValueError("encoded page copy requires its logical written mask")
            if any(scale is not None and not any(scale is field for field in page_fields)
                   for scale in (cache_encoding.key_scales, cache_encoding.value_scales)):
                raise ValueError("encoded page copy must include every scale field")
        capacity = source.shape[0] if source.ndim == 1 else -1
        if capacity <= 0 or not all(
            _index(tensor, capacity, key.device) for tensor in (source, destination)
        ):
            raise ValueError("page-copy indices require fixed contiguous CUDA int32 [C]")
        if any(
            field.ndim < 2 or field.shape[0] != key.shape[0]
            or field.device != key.device or not slot_storage_contiguous(field)
            or field.numel() == 0
            or field.dtype not in (
                torch.bfloat16, torch.float16, torch.float32,
                torch.int32, torch.int64, torch.bool,
            )
            for field in page_fields
        ):
            raise ValueError("page fields require contiguous typed storage and matching page count")
        all_tensors = (key, value, written, source, destination, *page_fields)
        if any(storage_regions_overlap(field, other) for field in page_fields
               for other in all_tensors if other is not field):
            raise ValueError("page field storage may not alias another bound operand")
        self._page_fields = page_fields
        self.meta = _bind(*all_tensors)
        items = max(key[0].numel(), value[0].numel(), written[0].numel())
        self._grid = capacity, triton.cdiv(items, _COPY_TILE)
    def launch(
        self, key: Tensor, value: Tensor, written: Tensor, source: Tensor, destination: Tensor
    ) -> None:
        if not _matches(self.meta, key, value, written, source, destination, *self._page_fields):
            raise ValueError("page-copy tensors changed after binding")
        _copy_page[self._grid](
            key,
            value,
            written,
            source,
            destination,
            PAGE_ITEMS=key[0].numel(),
            MASK_ITEMS=written[0].numel(),
            TILE=_COPY_TILE,
            VALUE_ITEMS=value[0].numel(),
            KEY_STRIDE=key.stride(0), VALUE_STRIDE=value.stride(0), MASK_STRIDE=written.stride(0),
            num_warps=4,
        )
        for field in self._page_fields:
            items = field[0].numel()
            _copy_page_field[(source.numel(), triton.cdiv(items, _COPY_TILE))](
                field, source, destination, ITEMS=items, STRIDE=field.stride(0),
                TILE=_COPY_TILE, num_warps=4,
            )

class PageTokenWritePrimitive:
    """Write one fixed layer and its written bit without recopying the page."""
    def __init__(
        self,
        key: Tensor,
        value: Tensor,
        written: Tensor,
        destination: Tensor,
        token: Tensor,
        new_key: Tensor,
        new_value: Tensor,
        *,
        layer: int,
    ) -> None:
        _, layers, _, heads, dim = _cache(key, value, written)
        capacity = new_key.shape[0] if new_key.ndim == 3 else -1
        if not (
            new_key.shape == (capacity, heads, dim)
            and _abi(new_value) == _abi(new_key)
            and new_key.device == key.device
            and new_key.dtype == key.dtype
            and new_key.is_contiguous()
            and all(_index(item, capacity, key.device) for item in (destination, token))
            and type(layer) is int
            and 0 <= layer < layers
        ):
            raise ValueError("page-token tensors do not match the fixed cache dtype ABI")
        self.meta = _bind(key, value, written, destination, token, new_key, new_value, layer=layer)
        self._grid = capacity, triton.cdiv(heads * dim, _COPY_TILE)
    def launch(
        self,
        key: Tensor,
        value: Tensor,
        written: Tensor,
        destination: Tensor,
        token: Tensor,
        new_key: Tensor,
        new_value: Tensor,
    ) -> None:
        tensors = key, value, written, destination, token, new_key, new_value
        if not _matches(self.meta, *tensors):
            raise ValueError("page-token tensors changed after binding")
        _, layers, block, heads, dim = self.meta.tensors[0][0]
        assert self.meta.layer is not None
        _write_token[self._grid](
            *tensors,
            LAYERS=layers,
            LAYER=self.meta.layer,
            PAGE_BLOCK=block,
            KV_HEADS=heads,
            HEAD_DIM=dim,
            TILE=_COPY_TILE,
            KEY_STRIDE=key.stride(0), VALUE_STRIDE=value.stride(0), WRITTEN_STRIDE=written.stride(0),
            num_warps=4,
        )
class PagedGQADecodePrimitive:
    """One retained decode artifact with an immutable fixed-address launch ABI.

    Optional selected_indices maps ordered attention positions to original
    logical KV token positions. In that mode seq_lens counts selected entries;
    the producer must validate unique, increasing, visible indices and counts
    within the bound selection width. Page ownership and leases are unchanged.
    """

    __slots__ = (
        "_activation_type",
        "_artifact",
        "_attention_precision",
        "_encoding",
        "_window",
        "_scale_bindings",
        "_bindings",
        "_block_d",
        "_grid",
        "_last_bindings",
        "_last_stream",
        "_ordinal",
        "_receipt_token",
        "_selection",
        "_signature",
        "_workspace",
        "meta",
    )

    def __init__(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        page_table: Tensor,
        seq_lens: Tensor,
        output: Tensor,
        *,
        layer: int,
        scale: float | None = None,
        partial: Tensor | None = None,
        statistics: Tensor | None = None,
        arrivals: Tensor | None = None,
        selected_indices: Tensor | None = None,
        cache_encoding: PagedKVEncoding | None = None,
        cache_window: PagedKVReadWindow | None = None,
        implementation: Literal["auto", "cuda", "triton"] = "auto",
    ) -> None:
        if implementation not in ("auto", "cuda", "triton"):
            raise ValueError("unknown paged-attention implementation")
        capacity, layers, block, query_heads, kv_heads, dim, table_pages, resolved_scale = (
            _validate_decode_layout(
                query,
                key,
                value,
                page_table,
                seq_lens,
                output,
                layer=layer,
                scale=scale,
                cache_encoding=cache_encoding,
            )
        )
        if cache_window is not None:
            if cache_encoding is None:
                raise ValueError("a native cache window requires an encoded historical prefix")
            cache_window.validate(query, cache_encoding, layers=layers)
        selected_width = 0
        if selected_indices is not None:
            if (
                selected_indices.device != query.device
                or selected_indices.dtype != torch.int32
                or selected_indices.ndim != 2
                or selected_indices.shape[0] != capacity
                or selected_indices.shape[1] <= 0
                or not selected_indices.is_contiguous()
            ):
                raise ValueError("selected logical KV indices require contiguous INT32 [M,K]")
            selected_width = selected_indices.shape[1]
        selection = () if selected_indices is None else (selected_indices,)
        self._selection = tuple(bind_tensor(tensor) for tensor in selection)
        provided = partial, statistics, arrivals
        if all(tensor is None for tensor in provided):
            workspace: tuple[Tensor, ...] = ()
        elif all(isinstance(tensor, Tensor) for tensor in provided):
            workspace = cast(tuple[Tensor, ...], provided)
        else:
            raise ValueError("paged-GQA split workspace must be complete")
        splits = 1
        if workspace:
            partial, statistics, arrivals = workspace
            # Workspace geometry binds the immutable partition plan. Its count
            # participates in the launch signature and numerical receipt.
            splits = partial.shape[2] if partial.ndim == 4 else 0
            if not (
                0 < splits <= _KV_SPLITS
                and splits & (splits - 1) == 0
                and partial.shape == (capacity, query_heads, splits, dim)
                and partial.dtype is torch.float32
                and statistics.shape == (capacity, query_heads, splits, 2)
                and statistics.dtype is torch.float32
                and arrivals.shape == (capacity, query_heads)
                and arrivals.dtype is torch.int32
                and all(
                    tensor.device == query.device and tensor.is_contiguous()
                    for tensor in workspace
                )
            ):
                raise ValueError("paged-GQA split workspace differs from its fixed ABI")
        self.meta = _bind(
            query, key, value, page_table, seq_lens, output, layer=layer, scale=resolved_scale
        )
        self._encoding = cache_encoding
        self._window = cache_window
        scale_tensors = (() if cache_encoding is None else tuple(
            t for t in (cache_encoding.key_scales, cache_encoding.value_scales) if t is not None))
        self._scale_bindings = tuple(bind_tensor(t) for t in scale_tensors)
        self._bindings = tuple(
            bind_tensor(tensor)
            for tensor in (query, key, value, page_table, seq_lens, output,
                           *workspace, *selection, *scale_tensors,
                           *(() if cache_window is None else cache_window.tensors))
        )
        bound_tensors = (query, key, value, page_table, seq_lens, output,
                         *workspace, *selection, *scale_tensors,
                         *(() if cache_window is None else cache_window.tensors))
        if (cache_encoding is not None and any(storage_regions_overlap(left, right)
                and not (cache_encoding.shared_storage is not None
                         and cache_encoding.shared_storage.allows_domain_overlap(left,right))
                for i, left in enumerate(bound_tensors) for right in bound_tensors[i + 1:])):
            raise ValueError("paged attention operands must own non-aliasing storage")
        self._workspace = tuple(bind_tensor(tensor) for tensor in workspace)
        self._activation_type = softmax_type(query.dtype, query.device)
        self._attention_precision = attention_precision()
        native = self._attention_precision == "auto"
        use_cuda = cache_encoding is not None and native and implementation != "triton"
        if cache_encoding is not None and cache_encoding.enable_rotation and not use_cuda:
            raise ValueError("rotated cache attention requires the native CUDA implementation")
        if cache_encoding is not None and not use_cuda and any(
            layout.spec.dtype.quantized and layout.spec.dtype is not CacheDType.FP8
            for layout in (cache_encoding.key, cache_encoding.value)
        ):
            raise ValueError("integer cache attention requires the native CUDA implementation")
        if implementation == "cuda" and not use_cuda:
            raise ValueError("native encoded attention requires native QKV operands")
        self._grid = (capacity, kv_heads * triton.cdiv(query_heads // kv_heads, 16)
                      if native else query_heads, splits)
        self._block_d = (triton.cdiv(dim, 16) * 16 if use_cuda
                         else max(16 if native else 1, 1 << (dim - 1).bit_length()))
        output_dtype = "bf16" if query.dtype is torch.bfloat16 else "fp16"
        self._signature = (
            f"capacity={capacity},layers={layers},layer={layer},page_block={block},"
            f"table_pages={table_pages},query_heads={query_heads},kv_heads={kv_heads},"
            f"head_dim={dim},block_d={self._block_d},block_n={_ATTN_TILE},"
            f"activation={self._activation_type},dot_operands={self._attention_precision},"
            f"kv_splits={splits},scale={resolved_scale.hex()},mask=seq_len,acc=fp32,out={output_dtype}"
        )
        if selected_width:
            self._signature += f",selected_width={selected_width},seq_len=selected_count"
        encoding_arguments = ()
        encoding_options = {}
        if cache_encoding is not None:
            k, v = cache_encoding.key.spec, cache_encoding.value.spec
            self._signature += f",encoded_key={k.identity!r},encoded_value={v.identity!r}"
            self._signature += f",cache_rotation={cache_encoding.rotation_contract!r}"
            encoding_arguments = (
                output if cache_encoding.key_scales is None else cache_encoding.key_scales,
                output if cache_encoding.value_scales is None else cache_encoding.value_scales,
                *( (output, output, output, output, seq_lens, seq_lens, seq_lens, seq_lens)
                   if cache_window is None else (
                       cache_window.key, cache_window.value,
                       output if cache_window.key_tail is None else cache_window.key_tail,
                       output if cache_window.value_tail is None else cache_window.value_tail,
                       cache_window.start_positions, cache_window.start_rows, cache_window.sources,
                       seq_lens if cache_window.activity is None else cache_window.activity,
                   )),
            )
            encoding_options = dict(
                K_GROUP=k.group_size if k.dtype.quantized else 0,
                V_GROUP=v.group_size if v.dtype.quantized else 0,
                K_COLUMN=k.axis is CacheGroupAxis.TOKENS,
                V_COLUMN=v.axis is CacheGroupAxis.TOKENS, NATIVE=native,
                WINDOW=cache_window is not None,
                HAS_ACTIVITY=cache_window is not None and cache_window.activity is not None,
            )
            self._signature += f",native_window={cache_window is not None}"
        encoding_options["CACHE_STRIDES"] = (
            key.stride(0), value.stride(0),
            0 if cache_encoding is None or cache_encoding.key_scales is None else cache_encoding.key_scales.stride(0),
            0 if cache_encoding is None or cache_encoding.value_scales is None else cache_encoding.value_scales.stride(0),
            0 if cache_window is None or cache_window.key_tail is None else cache_window.key_tail.stride(0),
            0 if cache_window is None or cache_window.value_tail is None else cache_window.value_tail.stride(0),
        )
        launch_workspace = workspace if workspace else (output, output, seq_lens)
        if use_cuda:
            from cubic_serve.ops.cuda.attention.paged_decode_native import bind_native_decode

            self._signature += ",backend=native-mma16816,qk-partitions=four-strided-fixed-order"
            scratch_options = ({} if not workspace else dict(
                partial=workspace[0], statistics=workspace[1], arrivals=workspace[2],
            ))
            self._artifact = bind_native_decode(
                self, query, key, value, page_table, seq_lens, output,
                selected_indices=selected_indices, **scratch_options,
            )
            self._receipt_token, self._ordinal = object(), 0
            self._last_stream = None
            self._last_bindings = self._bindings
            return
        with torch.cuda.device(query.device):
            kernel = (_decode_encoded if cache_encoding is not None
                      else _decode_qkv if native else _decode)
            artifact = kernel.warmup(
                query,
                key,
                value,
                page_table,
                seq_lens,
                output,
                *launch_workspace,
                output if selected_indices is None else selected_indices,
                *encoding_arguments,
                grid=self._grid,
                LAYERS=layers,
                LAYER=layer,
                PAGE_BLOCK=block,
                TABLE_PAGES=table_pages,
                QUERY_HEADS=query_heads,
                KV_HEADS=kv_heads,
                HEAD_DIM=dim,
                BLOCK_D=self._block_d,
                BLOCK_N=_ATTN_TILE,
                SCALE=resolved_scale,
                KV_SPLITS=splits,
                ACT_TYPE=self._activation_type,
                SELECTED_WIDTH=selected_width,
                **encoding_options,
                num_warps=4,
                num_stages=2,
            )
            if not isinstance(artifact, CompiledKernel):
                raise RuntimeError("Triton did not return a compiled paged-GQA artifact")
            artifact._init_handles()
        self._artifact, self._receipt_token, self._ordinal = artifact, object(), 0
        self._last_stream: int | None = None
        self._last_bindings = self._bindings

    @property
    def reduction_abi(self) -> dict[str, object]:
        dtype = "bf16" if self.meta.tensors[0][2] is torch.bfloat16 else "fp16"
        encoding = {} if self._encoding is None else {
            "cache_storage": {
                "key": self._encoding.key.spec.identity,
                "value": self._encoding.value.spec.identity,
                "rotation": self._encoding.rotation_contract,
                "scale": "fp32-absmax-div-rn-tiny-floor-zero-one",
                "decode": f"fp32-mul-rn-then-{dtype}-rne",
                "temporal_rule": "strictly-prior-committed-prefix-current-native-v1",
                "native_window": self._window is not None,
            },
        }
        return {
            **PAGED_GQA_TREE_ABI, "activation": str(self._activation_type),
            **encoding,
            "kv_splits": self._grid[2],
            "qk": "qkv-mma-fp32-acc" if self._attention_precision == "auto" else "fp32-reassociate",
            "pv": ("v-dtype-probability-mma-fp32-acc"
                   if self._attention_precision == "auto" else "fp32-reassociate"),
            "scalar": (
                "sub_rn,expf-online-scale,activation-exp,div_rn,mul_rn,add_rn,"
                f"{dtype}-rne"
            ),
            **({"qk_partial_merge": "four-strided-K-warp-partitions; RN-add warp 0,1,2,3"}
               if isinstance(self._artifact, NativeLaunch) else {}),
        }

    @property
    def layer(self) -> int:
        assert self.meta.layer is not None
        return self.meta.layer

    @property
    def scale(self) -> float:
        assert self.meta.scale is not None
        return self.meta.scale

    @property
    def grid(self) -> tuple[int, int, int]:
        return self._grid

    @property
    def launch_signature(self) -> str:
        return self._signature

    def project_artifact(self) -> BF16PagedGQAArtifactIdentity:
        return project_bf16_paged_gqa_artifact(self)

    def validate_operands(
        self, query: Tensor, key: Tensor, value: Tensor, page_table: Tensor,
        seq_lens: Tensor, output: Tensor,
    ) -> None:
        validate_bf16_paged_gqa_operands(self, query, key, value, page_table, seq_lens, output)

    def validate_bound_operands(
        self, query: Tensor, key: Tensor, value: Tensor, page_table: Tensor,
        seq_lens: Tensor, output: Tensor,
    ) -> None:
        tensors = query, key, value, page_table, seq_lens, output
        validate_bf16_paged_gqa_bound_operands(self, *tensors)

    def launch(
        self, query: Tensor, key: Tensor, value: Tensor, page_table: Tensor,
        seq_lens: Tensor, output: Tensor,
    ) -> BF16PagedGQALaunchReceipt:
        return launch_bf16_paged_gqa(self, query, key, value, page_table, seq_lens, output)

    def validate_receipt(self, receipt: BF16PagedGQALaunchReceipt) -> None:
        validate_bf16_paged_gqa_receipt(self, receipt)


def _require_paged_gqa_primitive(primitive: object) -> BF16PagedGQADecodePrimitive:
    if type(primitive) is not BF16PagedGQADecodePrimitive:
        raise TypeError("exact paged-GQA primitive is required")
    return primitive


def project_bf16_paged_gqa_artifact(
    primitive: BF16PagedGQADecodePrimitive,
) -> BF16PagedGQAArtifactIdentity:
    primitive = _require_paged_gqa_primitive(primitive)
    if isinstance(primitive._artifact, NativeLaunch):
        return replace(primitive._artifact.project_artifacts()[0],
                       launch_signature=primitive._signature)
    return project_triton_artifact(
        primitive._artifact,
        operator_id=("cubic.cuda.encoded-paged-gqa.v1" if primitive._encoding is not None
                     else BF16_PAGED_GQA_OPERATOR_ID
                     if primitive.meta.tensors[0][2] is torch.bfloat16
                     else FP16_PAGED_GQA_OPERATOR_ID),
        device=primitive.meta.tensors[0][3],
        launch_signature=primitive._signature,
        grid=primitive._grid,
    )


def validate_bf16_paged_gqa_operands(
    primitive: BF16PagedGQADecodePrimitive, query: Tensor, key: Tensor, value: Tensor,
    page_table: Tensor, seq_lens: Tensor, output: Tensor,
) -> None:
    primitive = _require_paged_gqa_primitive(primitive)
    if primitive.meta.layer is None or primitive.meta.scale is None:
        raise ValueError("paged decode primitive omits its scalar specialization")
    _validate_decode_layout(
        query,
        key,
        value,
        page_table,
        seq_lens,
        output,
        layer=primitive.meta.layer,
        scale=primitive.meta.scale,
        cache_encoding=primitive._encoding,
    )
    if not _matches(primitive.meta, query, key, value, page_table, seq_lens, output):
        raise ValueError("paged decode operands differ from the compiled specialization")


def validate_bf16_paged_gqa_bound_operands(
    primitive: BF16PagedGQADecodePrimitive, query: Tensor, key: Tensor, value: Tensor,
    page_table: Tensor, seq_lens: Tensor, output: Tensor,
) -> None:
    primitive = _require_paged_gqa_primitive(primitive)
    tensors = query, key, value, page_table, seq_lens, output
    validate_bf16_paged_gqa_operands(primitive, *tensors)
    if tuple(bind_tensor(tensor) for tensor in tensors) != primitive._bindings[: len(tensors)]:
        raise ValueError("paged decode qualification requires its exact bound tensors")
    if primitive._encoding is not None:
        scale_tensors = tuple(t for t in (
            primitive._encoding.key_scales, primitive._encoding.value_scales) if t is not None)
        window_tensors = () if primitive._window is None else primitive._window.tensors
        extra = tuple(bind_tensor(t) for t in (*scale_tensors, *window_tensors))
        if extra != primitive._bindings[len(tensors) + len(primitive._workspace)
                                        + len(primitive._selection):]:
            raise ValueError("encoded cache bindings changed after startup qualification")


def launch_bf16_paged_gqa(
    primitive: BF16PagedGQADecodePrimitive, query: Tensor, key: Tensor, value: Tensor,
    page_table: Tensor, seq_lens: Tensor, output: Tensor,
) -> BF16PagedGQALaunchReceipt:
    primitive = _require_paged_gqa_primitive(primitive)
    tensors = query, key, value, page_table, seq_lens, output
    validate_bf16_paged_gqa_bound_operands(primitive, *tensors)
    device = primitive.meta.tensors[0][3]
    with torch.cuda.device(device):
        stream = int(torch.cuda.current_stream(device).cuda_stream)
        workspace = tuple(binding.pointer for binding in primitive._workspace)
        launch_workspace = workspace if workspace else (output, output, seq_lens)
        selection = tuple(binding.pointer for binding in primitive._selection) or (output,)
        if isinstance(primitive._artifact, NativeLaunch):
            primitive._artifact.launch()
        elif primitive._encoding is None:
            primitive._artifact[primitive._grid](
                *tensors, *launch_workspace, *selection, stream=stream)
        else:
            encoding = primitive._encoding
            scales = (output if encoding.key_scales is None else encoding.key_scales,
                      output if encoding.value_scales is None else encoding.value_scales)
            window = primitive._window
            window_args = ((output, output, output, output, seq_lens, seq_lens, seq_lens, seq_lens)
                           if window is None else (
                               window.key, window.value,
                               output if window.key_tail is None else window.key_tail,
                               output if window.value_tail is None else window.value_tail,
                               window.start_positions, window.start_rows, window.sources,
                               seq_lens if window.activity is None else window.activity,
                           ))
            primitive._artifact[primitive._grid](
                *tensors, *launch_workspace, *selection, *scales, *window_args, stream=stream)
    primitive._ordinal += 1
    primitive._last_bindings = (
        *tuple(bind_tensor(tensor) for tensor in tensors),
        *primitive._workspace,
        *primitive._selection,
        *primitive._scale_bindings,
        *(() if primitive._window is None else
          tuple(bind_tensor(t) for t in primitive._window.tensors)),
    )
    primitive._last_stream = stream
    return LaunchReceipt(
        id(primitive._artifact), primitive._last_bindings, primitive._signature,
        primitive._grid, stream, primitive._ordinal, primitive._receipt_token,
    )


def validate_bf16_paged_gqa_receipt(
    primitive: BF16PagedGQADecodePrimitive,
    receipt: object,
) -> None:
    primitive = _require_paged_gqa_primitive(primitive)
    if primitive._last_stream is None:
        raise ValueError("paged-GQA primitive has no completed launch receipt")
    try:
        validate_launch_receipt(
            receipt,
            artifact_object_id=id(primitive._artifact),
            bindings=primitive._last_bindings,
            launch_signature=primitive._signature,
            grid=primitive._grid,
            stream=primitive._last_stream,
            ordinal=primitive._ordinal,
            primitive_token=primitive._receipt_token,
        )
    except ValueError as error:
        raise ValueError(
            f"paged-GQA launch receipt does not close this primitive launch: {error}"
        ) from error


def _validate_decode_layout(
    query: Tensor, key: Tensor, value: Tensor, page_table: Tensor,
    seq_lens: Tensor, output: Tensor,
    *,
    layer: int,
    scale: float | None,
    cache_encoding: PagedKVEncoding | None = None,
) -> tuple[int, int, int, int, int, int, int, float]:
    tensors = query, key, value, page_table, seq_lens, output
    if any(not isinstance(tensor, Tensor) for tensor in tensors):
        raise TypeError("paged decode operands must be tensors")
    if cache_encoding is None:
        _, layers, block, kv_heads, dim = _cache(key, value)
    else:
        _, layers, block, kv_heads, dim = cache_encoding.validate(key, value)
        if any(data.dtype != query.dtype for data, layout in
               ((key, cache_encoding.key), (value, cache_encoding.value))
               if not layout.spec.dtype.quantized):
            raise ValueError("plain cache sides must retain the query compute dtype")
    capacity = query.shape[0] if query.ndim == 3 else -1
    table_pages = page_table.shape[1] if page_table.ndim == 2 else -1
    query_heads = query.shape[1] if query.ndim == 3 else -1
    if not (
        capacity > 0
        and query.shape[2] == dim
        and query_heads % kv_heads == 0
        and query.device == key.device
        and query.dtype in (torch.bfloat16, torch.float16)
        and (cache_encoding is not None or query.dtype == key.dtype)
        and query.is_contiguous()
        and _abi(output) == _abi(query)
        and page_table.shape == (capacity, table_pages)
        and table_pages > 0
        and page_table.dtype is torch.int32
        and page_table.device == key.device
        and page_table.is_contiguous()
        and _index(seq_lens, capacity, key.device)
        and type(layer) is int
        and 0 <= layer < layers
        and not any(storage_regions_overlap(left, right) for i, left in enumerate(tensors)
                    for right in tensors[i + 1:])
    ):
        raise ValueError("paged decode requires non-aliasing fixed FP16/BF16 GQA tensors")
    resolved_scale = dim**-0.5 if scale is None else scale
    if (
        type(resolved_scale) is not float
        or not math.isfinite(resolved_scale)
        or resolved_scale <= 0
    ):
        raise ValueError("attention scale must be one finite positive float")
    return capacity, layers, block, query_heads, kv_heads, dim, table_pages, resolved_scale


# The historical class name remains an import-compatible alias.
BF16PagedGQADecodePrimitive = PagedGQADecodePrimitive
