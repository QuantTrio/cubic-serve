"""Portable BF16/FP16 routed-expert kernels for fixed-capacity decode graphs.

The execution shape follows llama.cpp 0f3a71be (``qwen35moe.cpp``), vLLM
2a4e3cc3 (``fused_moe.py``), and SGLang 99b91095 (``qwen2_moe.py``): a
global stable top-k router, expert-major padded tensor-core tiles, then a
deterministic route-order combine.  Expert ownership is a contiguous EP slice;
non-owned routes write positive zero and are combined by the caller's canonical
rank reduction.
"""

from __future__ import annotations

from cubic_serve.ops.cuda.startup import weight_bound_calibration_key

import math
from collections.abc import Callable
from dataclasses import replace

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from torch import Tensor

from cubic_serve.ops.cuda.activation import (
    activation_type,
    exp_nonpositive,
    sigmoid,
    silu,
    softmax_type,
)
from cubic_serve.ops.cuda.launch_cost import (
    MatmulGeometry,
    current_linear_cost,
    device_resources,
    matmul_geometries,
    select_launch,
)
from cubic_serve.ops.strategies import launch_parameters, moe_parameters, parameter_overrides

_TensorABI = tuple[tuple[int, ...], tuple[int, ...], torch.dtype, torch.device]


@triton.jit  # type: ignore[untyped-decorator]
def _stable_topk_logits(  # type: ignore[no-untyped-def]
    logits,
    row_activity,
    active_rows,
    top_logits,
    top_ids,
    EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    active = (row < tl.load(active_rows)) & (tl.load(row_activity + row) > 0)
    expert_ids = tl.arange(0, BLOCK)
    valid = expert_ids < EXPERTS
    scores = tl.load(logits + row * EXPERTS + expert_ids, mask=valid, other=-float("inf")).to(
        tl.float32
    )
    for slot in tl.static_range(TOP_K):
        best_score = tl.max(scores, axis=0)
        candidates = tl.where(scores == best_score, expert_ids, EXPERTS)
        best_expert = tl.min(candidates, axis=0)
        tl.store(top_logits + row * TOP_K + slot, tl.where(active, best_score, 0.0))
        tl.store(top_ids + row * TOP_K + slot, tl.where(active, best_expert, -1))
        scores = tl.where(expert_ids == best_expert, -float("inf"), scores)


@triton.jit  # type: ignore[untyped-decorator]
def _sort_and_normalize_topk(  # type: ignore[no-untyped-def]
    top_logits,
    top_ids,
    route_ids,
    route_weights,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
    ACT_TYPE: tl.constexpr = tl.float32,
):
    row = tl.program_id(0)
    slots = tl.arange(0, BLOCK)
    valid = slots < TOP_K
    scores = tl.load(top_logits + row * TOP_K + slots, mask=valid, other=-float("inf"))
    ids = tl.load(top_ids + row * TOP_K + slots, mask=valid, other=0x7FFFFFFF)
    active = tl.sum((valid & (ids >= 0)).to(tl.int32), axis=0) > 0
    maximum = tl.max(scores, axis=0)
    # The exponent follows the bound activation policy; normalize in FP32.
    probabilities = exp_nonpositive(scores - maximum, ACT_TYPE).to(tl.float32)
    denominator = tl.sum(probabilities, axis=0)
    for output_slot in tl.static_range(TOP_K):
        expert = tl.where(active, tl.min(ids, axis=0), -1)
        source = active & (ids == expert)
        probability = tl.sum(tl.where(source, probabilities, 0.0), axis=0)
        tl.store(route_ids + row * TOP_K + output_slot, expert)
        tl.store(
            route_weights + row * TOP_K + output_slot,
            tl.where(active, probability / denominator, 0.0),
        )
        ids = tl.where(source, 0x7FFFFFFF, ids)


@triton.jit  # type: ignore[untyped-decorator]
def _stable_topk_routes(  # type: ignore[no-untyped-def]
    logits,
    activity,
    active_rows,
    top_logits,
    top_ids,
    route_ids,
    route_weights,
    counts,
    cursors,
    E: tl.constexpr,
    K: tl.constexpr,
    LOCAL: tl.constexpr,
    B: tl.constexpr,
    S: tl.constexpr,
    C: tl.constexpr,
    ACT: tl.constexpr,
    INITIALIZE_COUNTS: tl.constexpr = True,
):
    # Preserve selected-score normalization and ascending logical expert order.
    # Only this CTA initializes counts; the next launch owns counting atomics.
    row = tl.program_id(0)
    active = (row < tl.load(active_rows)) & (tl.load(activity + row) > 0)
    ex = tl.arange(0, B)
    values = tl.load(logits + row * E + ex, ex < E, -float("inf")).to(tl.float32)
    slot = tl.arange(0, S)
    scores = tl.full((S,), -float("inf"), tl.float32)
    ids = tl.full((S,), 0x7FFFFFFF, tl.int64)
    for i in tl.static_range(K):
        best = tl.max(values, 0)
        expert = tl.min(tl.where(values == best, ex, E), 0)
        stored_score = tl.where(active, best, 0.0)
        stored_id = tl.where(active, expert, -1).to(tl.int64)
        tl.store(top_logits + row * K + i, stored_score)
        tl.store(top_ids + row * K + i, stored_id)
        scores = tl.where(slot == i, stored_score, scores)
        ids = tl.where(slot == i, stored_id, ids)
        values = tl.where(ex == expert, -float("inf"), values)
    active = tl.sum(((slot < K) & (ids >= 0)).to(tl.int32), 0) > 0
    maximum = tl.max(scores, 0)
    probs = exp_nonpositive(scores - maximum, ACT).to(tl.float32)
    denominator = tl.sum(probs, 0)
    for i in tl.static_range(K):
        expert = tl.where(active, tl.min(ids, 0), -1)
        source = active & (ids == expert)
        probability = tl.sum(tl.where(source, probs, 0.0), 0)
        tl.store(route_ids + row * K + i, expert)
        tl.store(route_weights + row * K + i, tl.where(active, probability / denominator, 0.0))
        ids = tl.where(source, 0x7FFFFFFF, ids)
    if INITIALIZE_COUNTS and row == 0:
        ci = tl.arange(0, C)
        tl.store(counts + ci, 0, ci < LOCAL)
        tl.store(cursors + ci, 0, ci < LOCAL)


@triton.jit  # type: ignore[untyped-decorator]
def _count_local_routes(  # type: ignore[no-untyped-def]
    route_ids,
    counts,
    FIRST_EXPERT: tl.constexpr,
    EXPERTS: tl.constexpr,
    ROUTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    route = tl.load(route_ids + offsets, mask=offsets < ROUTES, other=-1)
    local = route - FIRST_EXPERT
    owned = (offsets < ROUTES) & (local >= 0) & (local < EXPERTS)
    tl.atomic_add(counts + local, 1, mask=owned)


@triton.jit  # type: ignore[untyped-decorator]
def _compact_local_routes(  # type: ignore[no-untyped-def]
    route_ids, counts, starts, block_starts, total_blocks, compact_routes,
    FIRST_EXPERT: tl.constexpr, EXPERTS: tl.constexpr, ROUTES: tl.constexpr,
    BLOCK_M: tl.constexpr, EXPERT_BLOCK: tl.constexpr, ROUTE_BLOCK: tl.constexpr,
):
    """One CTA owns histogram, prefix and stable route scatter together."""
    offset = tl.arange(0, ROUTE_BLOCK)
    expert = tl.load(route_ids + offset, offset < ROUTES, -1).to(tl.int32) - FIRST_EXPERT
    owned = (offset < ROUTES) & (expert >= 0) & (expert < EXPERTS)
    count = tl.histogram(expert, EXPERT_BLOCK, mask=owned)
    ids = tl.arange(0, EXPERT_BLOCK)
    inclusive = tl.cumsum(count, axis=0)
    blocks = tl.cdiv(count, BLOCK_M)
    inclusive_blocks = tl.cumsum(blocks, axis=0)
    tl.store(counts + ids, count, ids < EXPERTS)
    tl.store(starts + ids, inclusive - count, ids < EXPERTS)
    tl.store(block_starts + ids, inclusive_blocks - blocks, ids < EXPERTS)
    tl.store(total_blocks, tl.sum(blocks, axis=0))
    # Original route ordinal breaks ties and remains the only row identity.
    # Unowned routes sort behind every local expert and never write row_map.
    key = tl.where(owned, expert, EXPERTS).to(tl.int64) * ROUTE_BLOCK + offset
    ordered = tl.sort(key, descending=False)
    tl.store(compact_routes + offset, ordered % ROUTE_BLOCK,
             ordered < EXPERTS * ROUTE_BLOCK)


@triton.jit  # type: ignore[untyped-decorator]
def _prefix_local_routes(  # type: ignore[no-untyped-def]
    counts,
    starts,
    block_starts,
    total_blocks,
    EXPERTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK: tl.constexpr,
):
    experts = tl.arange(0, BLOCK)
    count = tl.load(counts + experts, mask=experts < EXPERTS, other=0)
    inclusive = tl.cumsum(count, axis=0)
    blocks = tl.cdiv(count, BLOCK_M)
    inclusive_blocks = tl.cumsum(blocks, axis=0)
    tl.store(starts + experts, inclusive - count, mask=experts < EXPERTS)
    tl.store(block_starts + experts, inclusive_blocks - blocks, mask=experts < EXPERTS)
    tl.store(total_blocks, tl.sum(blocks, axis=0))


@triton.jit  # type: ignore[untyped-decorator]
def _scatter_local_routes(  # type: ignore[no-untyped-def]
    route_ids,
    starts,
    cursors,
    compact_routes,
    FIRST_EXPERT: tl.constexpr,
    EXPERTS: tl.constexpr,
    ROUTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    route = tl.load(route_ids + offsets, mask=offsets < ROUTES, other=-1)
    local = route - FIRST_EXPERT
    owned = (offsets < ROUTES) & (local >= 0) & (local < EXPERTS)
    position = tl.atomic_add(cursors + local, 1, mask=owned)
    start = tl.load(starts + local, mask=owned, other=0)
    tl.store(compact_routes + start + position, offsets, mask=owned)


@triton.jit  # type: ignore[untyped-decorator]
def _packed_swiglu_inplace(  # type: ignore[no-untyped-def]
    packed,
    route_ids,
    FIRST_EXPERT: tl.constexpr,
    EXPERTS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    ACT_TYPE: tl.constexpr = tl.float32,
):
    route = tl.program_id(0)
    expert = tl.load(route_ids + route)
    if expert < FIRST_EXPERT or expert >= FIRST_EXPERT + EXPERTS:
        return
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = columns < WIDTH
    gate_offsets = route * (2 * WIDTH) + columns
    gate = tl.load(packed + gate_offsets, mask=valid, other=0.0).to(tl.float32)
    up = tl.load(packed + gate_offsets + WIDTH, mask=valid, other=0.0).to(tl.float32)
    activated = silu(gate, ACT_TYPE).to(packed.dtype.element_ty)
    product = (activated * up.to(packed.dtype.element_ty)).to(packed.dtype.element_ty)
    tl.store(packed + gate_offsets, product, mask=valid)


@triton.jit  # type: ignore[untyped-decorator]
def _expert_gemm(  # type: ignore[no-untyped-def]
    inputs,
    weights,
    route_ids,
    output,
    FIRST_EXPERT: tl.constexpr,
    EXPERTS: tl.constexpr,
    M: tl.constexpr,
    TOP_K: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INPUT_STRIDE: tl.constexpr,
    ROUTED_INPUT: tl.constexpr,
    FUSED_SWIGLU: tl.constexpr,
    ACT_TYPE: tl.constexpr = tl.float32,
    ROUTE_INDEXED: tl.constexpr = False,
):
    if ROUTE_INDEXED:
        route_index = tl.program_id(0)
        global_expert = tl.load(route_ids + route_index)
        expert = global_expert - FIRST_EXPERT
        if expert < 0 or expert >= EXPERTS:
            return
        tile_m = route_index // (BLOCK_M * TOP_K)
        tile_n = tl.program_id(1)
        # One representative per (row tile, expert), preserving the original
        # multi-row weight reuse. Repeated routes do not duplicate GEMMs.
        candidates = tile_m * BLOCK_M * TOP_K + tl.arange(
            0, triton.next_power_of_2(BLOCK_M * TOP_K)
        )
        valid = (candidates < M * TOP_K) & (
            candidates < (tile_m + 1) * BLOCK_M * TOP_K
        )
        candidate_experts = tl.load(route_ids + candidates, mask=valid, other=-1)
        earlier = valid & (candidates < route_index) & (candidate_experts == global_expert)
        if tl.sum(earlier.to(tl.int32), 0) > 0:
            return
    else:
        expert = tl.program_id(0)
        tile = tl.program_id(1)
        tiles_n = tl.cdiv(N, BLOCK_N)
        tile_m = tile // tiles_n
        tile_n = tile % tiles_n
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_rows = rows < M
    global_expert = FIRST_EXPERT + expert
    slot = tl.full((BLOCK_M,), -1, tl.int32)
    for route in tl.static_range(TOP_K):
        selected = (
            tl.load(route_ids + rows * TOP_K + route, mask=valid_rows, other=-1) == global_expert
        )
        slot = tl.where(selected, route, slot)
    selected_rows = valid_rows & (slot >= 0)
    if tl.sum(selected_rows.to(tl.int32), axis=0) == 0:
        return

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start in range(0, K, BLOCK_K):
        inner = start + tl.arange(0, BLOCK_K)
        input_row = rows * TOP_K + slot if ROUTED_INPUT else rows
        if FUSED_SWIGLU:
            gate = tl.load(
                inputs + input_row[:, None] * (2 * K) + inner[None, :],
                mask=selected_rows[:, None] & (inner[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            up = tl.load(
                inputs + input_row[:, None] * (2 * K) + K + inner[None, :],
                mask=selected_rows[:, None] & (inner[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            activated = silu(gate, ACT_TYPE).to(inputs.dtype.element_ty)
            x = (activated * up.to(inputs.dtype.element_ty)).to(inputs.dtype.element_ty)
        else:
            x = tl.load(
                inputs + input_row[:, None] * INPUT_STRIDE + inner[None, :],
                mask=selected_rows[:, None] & (inner[None, :] < K),
                other=0.0,
            )
        weight = tl.load(
            weights + expert * N * K + columns[None, :] * K + inner[:, None],
            mask=(inner[:, None] < K) & (columns[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(x, weight)
    destination = (rows[:, None] * TOP_K + slot[:, None]) * N + columns[None, :]
    tl.store(
        output + destination,
        accumulator,
        mask=selected_rows[:, None] & (columns[None, :] < N),
    )


@triton.jit  # type: ignore[untyped-decorator]
def _compact_expert_gemm(  # type: ignore[no-untyped-def]
    inputs,
    weights,
    counts,
    starts,
    block_starts,
    total_blocks,
    compact_routes,
    output,
    EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INPUT_STRIDE: tl.constexpr,
    ROUTED_INPUT: tl.constexpr,
    FUSED_SWIGLU: tl.constexpr,
    ACT_TYPE: tl.constexpr = tl.float32,
):
    block = tl.program_id(0)
    if block >= tl.load(total_blocks):
        return
    expert_ids = tl.arange(0, triton.next_power_of_2(EXPERTS))
    valid_experts = expert_ids < EXPERTS
    expert_counts = tl.load(counts + expert_ids, mask=valid_experts, other=0)
    expert_blocks = tl.cdiv(expert_counts, BLOCK_M)
    first_blocks = tl.load(block_starts + expert_ids, mask=valid_experts, other=0)
    expert = tl.sum(
        (valid_experts & (block >= first_blocks + expert_blocks)).to(tl.int32), axis=0
    )
    if expert >= EXPERTS:
        return
    count = tl.load(counts + expert)
    start = tl.load(starts + expert)
    first_block = tl.load(block_starts + expert)
    lanes = (block - first_block) * BLOCK_M + tl.arange(0, BLOCK_M)
    valid_rows = lanes < count
    flat_routes = tl.load(compact_routes + start + lanes, mask=valid_rows, other=0)
    rows = flat_routes // TOP_K
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for inner_start in range(0, K, BLOCK_K):
        inner = inner_start + tl.arange(0, BLOCK_K)
        input_row = flat_routes if ROUTED_INPUT else rows
        if FUSED_SWIGLU:
            gate = tl.load(
                inputs + input_row[:, None] * (2 * K) + inner[None, :],
                mask=valid_rows[:, None] & (inner[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            up = tl.load(
                inputs + input_row[:, None] * (2 * K) + K + inner[None, :],
                mask=valid_rows[:, None] & (inner[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            activated = silu(gate, ACT_TYPE).to(inputs.dtype.element_ty)
            x = (activated * up.to(inputs.dtype.element_ty)).to(inputs.dtype.element_ty)
        else:
            x = tl.load(
                inputs + input_row[:, None] * INPUT_STRIDE + inner[None, :],
                mask=valid_rows[:, None] & (inner[None, :] < K),
                other=0.0,
            )
        weight = tl.load(
            weights + expert * N * K + columns[None, :] * K + inner[:, None],
            mask=(inner[:, None] < K) & (columns[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(x, weight)
    destination = flat_routes[:, None] * N + columns[None, :]
    tl.store(
        output + destination,
        accumulator,
        mask=valid_rows[:, None] & (columns[None, :] < N),
    )


@triton.jit  # type: ignore[untyped-decorator]
def _route_combine(  # type: ignore[no-untyped-def]
    routes,
    route_ids,
    route_weights,
    output,
    FIRST_EXPERT: tl.constexpr,
    EXPERTS: tl.constexpr,
    ELEMENTS: tl.constexpr,
    WIDTH: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < ELEMENTS
    row = offsets // WIDTH
    column = offsets % WIDTH
    accumulator = tl.zeros((BLOCK,), output.dtype.element_ty)
    for route in tl.static_range(TOP_K):
        expert = tl.load(route_ids + row * TOP_K + route, mask=mask, other=-1)
        owned = (expert >= FIRST_EXPERT) & (expert < FIRST_EXPERT + EXPERTS)
        value = tl.load(
            routes + (row * TOP_K + route) * WIDTH + column,
            mask=mask & owned,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(route_weights + row * TOP_K + route, mask=mask, other=0.0).to(tl.float32)
        weighted = (value * weight).to(output.dtype.element_ty)
        accumulator = (accumulator + weighted).to(output.dtype.element_ty)
    tl.store(output + offsets, accumulator, mask=mask)


@triton.jit  # type: ignore[untyped-decorator]
def _shared_routed_combine(  # type: ignore[no-untyped-def]
    shared,
    shared_gate,
    routed,
    output,
    ELEMENTS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    ACT_TYPE: tl.constexpr = tl.float32,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < ELEMENTS
    row = offsets // WIDTH
    gate = tl.load(shared_gate + row, mask=mask, other=0.0).to(tl.float32)
    activated = sigmoid(gate, ACT_TYPE).to(shared.dtype.element_ty)
    shared_value = tl.load(shared + offsets, mask=mask, other=0.0).to(tl.float32)
    routed_value = tl.load(routed + offsets, mask=mask, other=0.0).to(tl.float32)
    gated = (shared_value.to(shared.dtype.element_ty) * activated).to(shared.dtype.element_ty)
    gated = gated.to(tl.float32)
    tl.store(output + offsets, routed_value + gated, mask=mask)


def _abi(tensor: Tensor) -> _TensorABI:
    return tuple(tensor.shape), tensor.stride(), tensor.dtype, tensor.device


class MoeBranchStreams:
    """Graph-capturable fork/join for independent shared and routed experts."""

    __slots__ = ("_cursor", "_device", "_done", "_ready", "shared")

    def __init__(self, device: torch.device, branch_count: int) -> None:
        if branch_count < 1:
            raise ValueError("MoE branch schedule must contain at least one fork")
        self._device = device
        self.shared = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
        self._ready = tuple(torch.cuda.Event() for _ in range(branch_count))  # type: ignore[no-untyped-call]
        self._done = tuple(torch.cuda.Event() for _ in range(branch_count))  # type: ignore[no-untyped-call]
        self._cursor = 0

    def begin(self) -> None:
        self._cursor = 0

    def fork_shared(self):  # type: ignore[no-untyped-def]
        if self._cursor >= len(self._ready):
            raise RuntimeError("MoE branch schedule exceeded its fixed graph ABI")
        main = torch.cuda.current_stream(self._device)
        ready = self._ready[self._cursor]
        ready.record(main)
        self.shared.wait_event(ready)
        return torch.cuda.stream(self.shared)

    def join_shared(self) -> None:
        done = self._done[self._cursor]
        done.record(self.shared)
        torch.cuda.current_stream(self._device).wait_event(done)
        self._cursor += 1


class RoutedExpertsPrimitive:
    """One fixed-M, fixed-top-k local expert slice with caller-owned buffers."""

    def __init__(
        self,
        hidden: Tensor,
        router_logits: Tensor,
        row_activity: Tensor,
        active_rows: Tensor,
        gate_up: Tensor,
        down: Tensor,
        output: Tensor,
        *,
        first_expert: int,
        top_k: int,
    ) -> None:
        if (
            hidden.device.type != "cuda"
            or not all(
                item.device == hidden.device and item.is_contiguous()
                for item in (
                    router_logits,
                    row_activity,
                    active_rows,
                    gate_up,
                    down,
                    output,
                )
            )
            or hidden.dtype not in (torch.bfloat16, torch.float16)
            or any(
                item.dtype != hidden.dtype
                for item in (
                    hidden,
                    router_logits,
                    gate_up,
                    down,
                    output,
                )
            )
            or hidden.ndim != 2
            or router_logits.ndim != 2
            or router_logits.shape[0] != hidden.shape[0]
            or row_activity.shape != (hidden.shape[0],)
            or row_activity.dtype is not torch.int32
            or active_rows.shape != (1,)
            or active_rows.dtype is not torch.int32
            or gate_up.ndim != 3
            or down.ndim != 3
            or gate_up.shape[0] != down.shape[0]
            or gate_up.shape[1] != 2 * down.shape[2]
            or gate_up.shape[2] != hidden.shape[1]
            or down.shape[1] != hidden.shape[1]
            or output.shape != hidden.shape
            or type(first_expert) is not int
            or first_expert < 0
            or first_expert + gate_up.shape[0] > router_logits.shape[1]
            or type(top_k) is not int
            or not 0 < top_k <= router_logits.shape[1]
        ):
            raise ValueError("routed experts differ from the CUDA BF16/FP16 MoE ABI")
        m, hidden_size = hidden.shape
        experts, packed_width, _ = gate_up.shape
        intermediate = packed_width // 2
        self._abis = tuple(
            _abi(item)
            for item in (
                hidden,
                router_logits,
                row_activity,
                active_rows,
                gate_up,
                down,
                output,
            )
        )
        self._meta = (
            m,
            hidden_size,
            intermediate,
            experts,
            router_logits.shape[1],
            first_expert,
            top_k,
        )
        self._activation_type = activation_type(hidden.dtype, hidden.device)
        self._softmax_type = softmax_type(hidden.dtype, hidden.device)
        self.top_logits = torch.empty((m, top_k), dtype=torch.float32, device=hidden.device)
        self.top_ids = torch.empty((m, top_k), dtype=torch.int64, device=hidden.device)
        self.route_ids = torch.empty_like(self.top_ids)
        self.route_weights = torch.empty((m, top_k), dtype=hidden.dtype, device=hidden.device)
        self.packed = torch.zeros(
            (m, top_k, 2 * intermediate), dtype=hidden.dtype, device=hidden.device
        )
        self.routes = torch.zeros(
            (m, top_k, hidden_size), dtype=hidden.dtype, device=hidden.device
        )
        self._parameters = moe_parameters(
            hidden=hidden_size,
            intermediate=intermediate,
            experts=experts,
            global_experts=router_logits.shape[1],
            top_k=top_k,
        )
        self._compact = (
            m * top_k >= router_logits.shape[1] * self._parameters.compact_min_routes_per_expert
        )
        self.route_counts = torch.zeros((experts,), dtype=torch.int32, device=hidden.device)
        self.route_starts = torch.empty_like(self.route_counts)
        self.route_block_starts = torch.empty_like(self.route_counts)
        self.route_cursors = torch.zeros_like(self.route_counts)
        self.total_route_blocks = torch.empty((1,), dtype=torch.int32, device=hidden.device)
        self.compact_routes = torch.empty((m * top_k,), dtype=torch.int32, device=hidden.device)
        p = self._parameters
        self._up_geometry = MatmulGeometry(p.block_m, p.block_n, p.block_k,
                                           p.gemm_warps, p.gemm_stages)
        self._down_geometry = replace(
            self._up_geometry, block_n=p.compact_down_block_n if self._compact else p.block_n,
        )
        self._select_geometry(hidden, router_logits, gate_up, down)

    def _select_geometry(
        self, hidden: Tensor, router_logits: Tensor, gate_up: Tensor, down: Tensor,
    ) -> None:
        """Rank-local physical placement; never alter logical expert ownership.

        The compact occupancy estimate counts expected nonempty local experts,
        whereas the sparse kernel scans M rows for each local expert. A private
        uniform routing fixture calibrates complete up/activation/down/combine
        cost. Production routing and arithmetic boundaries are unchanged.
        """
        m, h, intermediate, experts, global_experts, first, top_k = self._meta
        limits = launch_parameters()
        if limits.max_candidates == 1:
            return
        context = current_linear_cost()
        resources = context.device if context else device_resources(hidden.device)
        explicit = parameter_overrides("moe", hidden=h, intermediate=intermediate,
                                       experts=experts, global_experts=global_experts, top_k=top_k)
        locked = frozenset(name for name, key in (
            ("block_m", "block_m"), ("block_n", "block_n"),
            ("warps", "gemm_warps"), ("stages", "gemm_stages"),
        ) if key in explicit)
        if self._compact:
            nonempty = max(1.0, experts * (1.0 - (1.0 - top_k / global_experts) ** m))
            groups = max(1, math.ceil(nonempty))
            rows = max(1, math.ceil(m * top_k * experts / global_experts / nonempty))
        else:
            groups, rows = experts, m
        kwargs = dict(rows=rows, groups=groups, element_bytes=hidden.element_size())
        ups = matmul_geometries(resources, **kwargs, columns=(2 * intermediate,), reduction=h,
                                baseline=self._up_geometry, locked=locked)
        candidates = [(self._up_geometry, self._down_geometry)]
        for up in ups[1:]:
            down_base = replace(self._down_geometry, block_m=up.block_m)
            down_locked = locked | {"block_m"}
            if "compact_down_block_n" in explicit:
                down_locked |= {"block_n"}
            downs = matmul_geometries(resources, **kwargs, columns=(h,), reduction=intermediate,
                                      baseline=down_base, locked=down_locked)
            candidates.append((up, downs[-1]))
        if len(candidates) == 1:
            return
        key = ("routed", hidden.device, hidden.dtype, self._meta[:5], top_k,
               self._activation_type, self._softmax_type, self._compact, tuple(candidates))
        cache = context.operator_choices if context else {}
        if key in cache:
            self._up_geometry, self._down_geometry = candidates[cache[key]]
            return
        generator = torch.Generator(device=hidden.device).manual_seed(0)
        x = torch.randn(hidden.shape, dtype=hidden.dtype, device=hidden.device,
                        generator=generator).mul_(0.01)
        logits = torch.full_like(router_logits, -80)
        ids = (torch.arange(m, device=hidden.device)[:, None]
               + torch.arange(top_k, device=hidden.device)[None, :] * (global_experts // top_k)
               + first) % global_experts
        logits.scatter_(1, ids, torch.arange(top_k, 0, -1, device=hidden.device,
                                            dtype=hidden.dtype)[None, :].expand(m, -1))
        activity = torch.ones(m, dtype=torch.int32, device=hidden.device)
        active = torch.tensor([m], dtype=torch.int32, device=hidden.device)
        output = torch.empty_like(hidden)

        def bind(index: int) -> None:
            self._up_geometry, self._down_geometry = candidates[index]

        def prepare() -> None:
            self.packed.zero_()
            self.routes.zero_()

        chosen = select_launch(
            range(len(candidates)), device=hidden.device, bind=bind, prepare=prepare,
            launch=lambda: self.launch(x, logits, activity, active, gate_up, down, output),
            snapshot=lambda: (output, self.route_ids, self.route_weights, self.packed, self.routes),
            cache_key=weight_bound_calibration_key(("plain-moe", key)),
        )
        cache[key] = chosen


    @property
    def launch_signature(self) -> str:
        m, hidden, intermediate, experts, global_experts, first, top_k = self._meta
        return (
            f"act={self._activation_type},softmax={self._softmax_type},"
            f"m={m},h={hidden},i={intermediate},e={experts}/{global_experts},first={first},"
            f"topk={top_k},dtype={self._abis[0][2]},fp32-router,parameters={self._parameters},compact={self._compact},"
            f"up={self._up_geometry},down={self._down_geometry}"
        )

    def launch(
        self,
        hidden: Tensor,
        router_logits: Tensor,
        row_activity: Tensor,
        active_rows: Tensor,
        gate_up: Tensor,
        down: Tensor,
        output: Tensor,
    ) -> None:
        self._launch(hidden, router_logits, row_activity, active_rows, gate_up, down, output, None)

    def launch_qualification(
        self,
        hidden: Tensor,
        router_logits: Tensor,
        row_activity: Tensor,
        active_rows: Tensor,
        gate_up: Tensor,
        down: Tensor,
        output: Tensor,
        observer: Callable[[str, Tensor], None],
    ) -> None:
        """Expose coherent eager-only stage boundaries without changing the hot launch."""

        if torch.cuda.is_current_stream_capturing():
            raise ValueError("routed-expert qualification is eager-only")
        self._launch(
            hidden,
            router_logits,
            row_activity,
            active_rows,
            gate_up,
            down,
            output,
            observer,
        )

    def _launch(
        self,
        hidden: Tensor,
        router_logits: Tensor,
        row_activity: Tensor,
        active_rows: Tensor,
        gate_up: Tensor,
        down: Tensor,
        output: Tensor,
        observer: Callable[[str, Tensor], None] | None,
    ) -> None:
        operands = hidden, router_logits, row_activity, active_rows, gate_up, down, output
        if tuple(_abi(item) for item in operands) != self._abis:
            raise ValueError("routed-expert operands do not match their fixed launch ABI")
        m, hidden_size, intermediate, experts, _global_experts, first, top_k = self._meta
        _stable_topk_logits[(m,)](
            router_logits,
            row_activity,
            active_rows,
            self.top_logits,
            self.top_ids,
            EXPERTS=_global_experts,
            TOP_K=top_k,
            BLOCK=triton.next_power_of_2(_global_experts),
            num_warps=8,
        )
        _sort_and_normalize_topk[(m,)](
            self.top_logits,
            self.top_ids,
            self.route_ids,
            self.route_weights,
            TOP_K=top_k,
            BLOCK=triton.next_power_of_2(top_k),
            ACT_TYPE=self._softmax_type,
            num_warps=1,
        )
        if observer is not None:
            observer("route_ids", self.route_ids)
            observer("route_weights", self.route_weights)
        up, down_geometry = self._up_geometry, self._down_geometry
        block_m = up.block_m
        block_n, block_k = up.block_n, up.block_k
        route_indexed = self._parameters.route_indexed(m, experts, top_k, block_m)
        if self._compact:
            route_count = m * top_k
            self.route_counts.zero_()
            self.route_cursors.zero_()
            _count_local_routes[(triton.cdiv(route_count, 256),)](
                self.route_ids,
                self.route_counts,
                FIRST_EXPERT=first,
                EXPERTS=experts,
                ROUTES=route_count,
                BLOCK=256,
                num_warps=4,
            )
            _prefix_local_routes[(1,)](
                self.route_counts,
                self.route_starts,
                self.route_block_starts,
                self.total_route_blocks,
                EXPERTS=experts,
                BLOCK_M=block_m,
                BLOCK=triton.next_power_of_2(experts),
                num_warps=1,
            )
            _scatter_local_routes[(triton.cdiv(route_count, 256),)](
                self.route_ids,
                self.route_starts,
                self.route_cursors,
                self.compact_routes,
                FIRST_EXPERT=first,
                EXPERTS=experts,
                ROUTES=route_count,
                BLOCK=256,
                num_warps=4,
            )
            maximum_blocks = triton.cdiv(route_count, block_m) + experts
            gemm_grid = (maximum_blocks, triton.cdiv(2 * intermediate, block_n))
            _compact_expert_gemm[gemm_grid](
                hidden,
                gate_up,
                self.route_counts,
                self.route_starts,
                self.route_block_starts,
                self.total_route_blocks,
                self.compact_routes,
                self.packed,
                EXPERTS=experts,
                TOP_K=top_k,
                N=2 * intermediate,
                K=hidden_size,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                INPUT_STRIDE=hidden_size,
                ROUTED_INPUT=False,
                FUSED_SWIGLU=False,
                ACT_TYPE=self._activation_type,
                num_warps=up.warps,
                num_stages=up.stages,
            )
        else:
            gemm_grid = (
                m * top_k if route_indexed else experts,
                (1 if route_indexed else triton.cdiv(m, block_m))
                * triton.cdiv(2 * intermediate, block_n),
            )
            _expert_gemm[gemm_grid](
                hidden,
                gate_up,
                self.route_ids,
                self.packed,
                FIRST_EXPERT=first,
                EXPERTS=experts,
                M=m,
                TOP_K=top_k,
                N=2 * intermediate,
                K=hidden_size,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                INPUT_STRIDE=hidden_size,
                ROUTED_INPUT=False,
                FUSED_SWIGLU=False,
                ACT_TYPE=self._activation_type,
                ROUTE_INDEXED=route_indexed,
                num_warps=up.warps,
                num_stages=up.stages,
            )
        if observer is not None:
            observer("gate_up", self.packed)
        route_count = m * top_k
        _packed_swiglu_inplace[(route_count, triton.cdiv(intermediate, 256))](
            self.packed,
            self.route_ids,
            FIRST_EXPERT=first,
            EXPERTS=experts,
            WIDTH=intermediate,
            BLOCK=256,
            ACT_TYPE=self._activation_type,
            num_warps=4,
        )
        if observer is not None:
            observer("active", self.packed[..., :intermediate])
        block_n = down_geometry.block_n
        if self._compact:
            down_block_n = block_n
            gemm_grid = (
                maximum_blocks,
                triton.cdiv(hidden_size, down_block_n),
            )
            _compact_expert_gemm[gemm_grid](
                self.packed,
                down,
                self.route_counts,
                self.route_starts,
                self.route_block_starts,
                self.total_route_blocks,
                self.compact_routes,
                self.routes,
                EXPERTS=experts,
                TOP_K=top_k,
                N=hidden_size,
                K=intermediate,
                BLOCK_M=block_m,
                BLOCK_N=down_block_n,
                BLOCK_K=block_k,
                INPUT_STRIDE=2 * intermediate,
                ROUTED_INPUT=True,
                FUSED_SWIGLU=False,
                ACT_TYPE=self._activation_type,
                num_warps=down_geometry.warps,
                num_stages=down_geometry.stages,
            )
        else:
            gemm_grid = (
                m * top_k if route_indexed else experts,
                (1 if route_indexed else triton.cdiv(m, block_m))
                * triton.cdiv(hidden_size, block_n),
            )
            _expert_gemm[gemm_grid](
                self.packed,
                down,
                self.route_ids,
                self.routes,
                FIRST_EXPERT=first,
                EXPERTS=experts,
                M=m,
                TOP_K=top_k,
                N=hidden_size,
                K=intermediate,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                INPUT_STRIDE=2 * intermediate,
                ROUTED_INPUT=True,
                FUSED_SWIGLU=False,
                ACT_TYPE=self._activation_type,
                ROUTE_INDEXED=route_indexed,
                num_warps=down_geometry.warps,
                num_stages=down_geometry.stages,
            )
        if observer is not None:
            observer("routes", self.routes)
        elements = m * hidden_size
        _route_combine[(triton.cdiv(elements, 256),)](
            self.routes,
            self.route_ids,
            self.route_weights,
            output,
            FIRST_EXPERT=first,
            EXPERTS=experts,
            ELEMENTS=elements,
            WIDTH=hidden_size,
            TOP_K=top_k,
            BLOCK=256,
            num_warps=4,
            # Weighted routes and each partial sum have distinct activation-dtype
            # boundaries; a low-precision FMA would silently skip one rounding.
            enable_fp_fusion=False,
        )
        if observer is not None:
            observer("output", output)


class MoeSharedCombinePrimitive:
    """Gate the TP-sharded shared expert and add the rank-local routed partial."""

    def __init__(self, shared: Tensor, shared_gate: Tensor, routed: Tensor, output: Tensor) -> None:
        if (
            shared.device.type != "cuda"
            or shared.ndim != 2
            or shared.shape != routed.shape
            or output.shape != shared.shape
            or shared_gate.shape != (shared.shape[0], 1)
            or shared.dtype not in (torch.bfloat16, torch.float16)
            or any(
                item.dtype != shared.dtype
                for item in (
                    shared,
                    shared_gate,
                    routed,
                    output,
                )
            )
            or not all(
                item.device == shared.device and item.is_contiguous()
                for item in (
                    shared_gate,
                    routed,
                    output,
                )
            )
            or not shared.is_contiguous()
        ):
            raise ValueError("shared/routed combine differs from the CUDA BF16/FP16 MoE ABI")
        self._abis = tuple(_abi(item) for item in (shared, shared_gate, routed, output))
        self._meta = shared.numel(), shared.shape[1]
        self._activation_type = activation_type(shared.dtype, shared.device)

    def launch(self, shared: Tensor, shared_gate: Tensor, routed: Tensor, output: Tensor) -> None:
        operands = shared, shared_gate, routed, output
        if tuple(_abi(item) for item in operands) != self._abis:
            raise ValueError("shared/routed operands do not match their fixed launch ABI")
        elements, width = self._meta
        _shared_routed_combine[(triton.cdiv(elements, 256),)](
            *operands,
            ELEMENTS=elements,
            WIDTH=width,
            BLOCK=256,
            ACT_TYPE=self._activation_type,
            num_warps=4,
            # Preserve the rounded gated shared output before routed addition.
            enable_fp_fusion=False,
        )


# Compatibility names for already-bound native BF16 model executors.
BF16RoutedExpertsPrimitive = RoutedExpertsPrimitive
BF16MoeSharedCombinePrimitive = MoeSharedCombinePrimitive


__all__ = (
    "BF16MoeSharedCombinePrimitive",
    "BF16RoutedExpertsPrimitive",
    "MoeBranchStreams",
    "MoeSharedCombinePrimitive",
    "RoutedExpertsPrimitive",
)
