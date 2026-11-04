"""Causal paged block selection with deterministic score/index ordering.

The caller owns compression, normalization, RoPE and cache lifecycle. This
consumer only scores already-compressed keys and returns original token IDs.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from torch import Tensor
from triton.language.extra import libdevice

from cubic_serve.quantization.common.binding import PackedOperatorBinding, _PackedLaunch
from cubic_serve.runtime.cache.quantization import slot_storage_contiguous, storage_regions_overlap


@triton.jit  # type: ignore[untyped-decorator]
def _scores(  # type: ignore[no-untyped-def]
    QUERY,
    KEYS,
    TABLE,
    LENGTHS,
    ACTIVE,
    ORDER,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    GROUPS: tl.constexpr,
    TABLE_PAGES: tl.constexpr,
    PAGE_GROUPS: tl.constexpr,
    COMPRESSION: tl.constexpr,
    KEEP: tl.constexpr,
    SCALE: tl.constexpr,
    DB: tl.constexpr,
    GB: tl.constexpr,
    PARTS: tl.constexpr = 1,
    RANK: tl.constexpr = 0,
    CACHE_STRIDE: tl.constexpr = 0,
):
    row = tl.program_id(0)
    active = tl.load(ACTIVE + row) != 0
    complete = tl.load(LENGTHS + row) // COMPRESSION
    # A bounded startup grid covers a resource-sized wave. Device lengths
    # retain the select-all bypass and delimit the exact live rank-owned keys.
    if not active or complete <= KEEP:
        return
    local_complete = tl.minimum(GROUPS, tl.maximum(0, tl.cdiv(complete - RANK, PARTS)))
    for tile in range(tl.program_id(1), tl.cdiv(local_complete, GB), tl.num_programs(1)):
        group = tile * GB + tl.arange(0, GB)
        logical_group = group * PARTS + RANK
        valid = (logical_group < complete) & (group < GROUPS)
        score = tl.full((GB,), 0, tl.float32)
        if active & (complete > KEEP):
            page = tl.load(TABLE + row * TABLE_PAGES + group // PAGE_GROUPS, valid, 0).to(tl.int64)
            d = tl.arange(0, DB)
            key = tl.load(
                KEYS
                + page[:, None].to(tl.int64) * CACHE_STRIDE
                + (group[:, None] % PAGE_GROUPS) * WIDTH
                + d[None, :],
                valid[:, None] & (d[None, :] < WIDTH),
                0,
            ).to(tl.float32)
            for head in tl.static_range(HEADS):
                query = tl.load(QUERY + (row * HEADS + head) * WIDTH + d, d < WIDTH, 0).to(tl.float32)
                dot = tl.sum(key * query[None, :], 1)
                score += tl.maximum(dot, 0.0)
            score = tl.div_rn(score, SCALE)
        # Nonnegative finite IEEE FP32 is ordered by its unsigned bits. The low
        # word reverses the original block index, making every tie unique.
        bits = score.to(tl.uint32, bitcast=True).to(tl.int64)
        order = (bits << 32) | (0xFFFFFFFF - logical_group.to(tl.int64))
        tl.store(ORDER + row * GROUPS + group, tl.where(valid, order, -1), group < GROUPS)


@triton.jit  # type: ignore[untyped-decorator]
def _select_tree(  # type: ignore[no-untyped-def]
    INPUT,
    OUTPUT,
    LENGTHS,
    ACTIVE,
    INPUT_STRIDE: tl.constexpr,
    OUTPUT_STRIDE: tl.constexpr,
    KEEP: tl.constexpr,
    CAPACITY: tl.constexpr,
    COMPRESSION: tl.constexpr,
    COVERAGE: tl.constexpr,
    FIRST: tl.constexpr,
    LOG_CAPACITY: tl.constexpr,
    PARTS: tl.constexpr = 1,
    RANK: tl.constexpr = 0,
    CANDIDATES: tl.constexpr = 0,
    EMIT_IDS: tl.constexpr = False,
    GROUPS: tl.constexpr = 0,
):
    row, chunk = tl.program_id(0), tl.program_id(1)
    if tl.load(ACTIVE + row) == 0:
        return
    complete = tl.load(LENGTHS + row) // COMPRESSION
    extent = CANDIDATES if CANDIDATES else tl.maximum(0, (complete + PARTS - 1 - RANK) // PARTS)
    if complete <= KEEP:
        return
    if chunk * COVERAGE >= extent:
        # Root-local candidate output must be neutral even for an empty owner.
        if chunk == 0:
            tl.store(OUTPUT + row * OUTPUT_STRIDE + tl.arange(0, CAPACITY), -1)
        return
    valid = extent if FIRST else tl.cdiv(extent, COVERAGE // 2) * CAPACITY
    if FIRST:
        i = chunk * (2 * CAPACITY) + tl.arange(0, 2 * CAPACITY)
        keys = tl.load(INPUT + row * INPUT_STRIDE + i, i < valid, -1)
        if CAPACITY == 1:
            # Keep the rank-one vector ABI; tl.topk(..., 1) returns a scalar.
            best = tl.full((1,), 0, tl.int64) + tl.max(keys, 0)
        else:
            best = tl.topk(keys, CAPACITY)
    else:
        j = tl.arange(0, CAPACITY)
        i = chunk * (2 * CAPACITY) + j
        best = tl.load(INPUT + row * INPUT_STRIDE + i, i < valid, -1)
        if chunk * (2 * CAPACITY) + CAPACITY < valid:
            other_i = chunk * (2 * CAPACITY) + (2 * CAPACITY - 1 - j)
            other = tl.load(INPUT + row * INPUT_STRIDE + other_i, other_i < valid, -1)
            # Descending children become bitonic after reversing the second.
            # Pairwise maxima retain the top half; log2(CAPACITY) compare/swap
            # rounds sort it without repeating a complete sort network.
            best = tl.maximum(best, other)
            for shift in tl.static_range(LOG_CAPACITY - 1, -1, -1):
                peer = tl.gather(best, j ^ (1 << shift), axis=0)
                best = tl.where(
                    (j & (1 << shift)) == 0, tl.maximum(best, peer), tl.minimum(best, peer)
                )
    j = tl.arange(0, CAPACITY)
    if EMIT_IDS:
        # The final merge already owns the complete score-ordered top-k.
        # Materialize its canonical original-block order without another launch.
        ids = tl.where(j < KEEP, (0xFFFFFFFF - (best & 0xFFFFFFFF)).to(tl.int32), GROUPS)
        if CAPACITY > 1:
            ids = tl.sort(ids, descending=False)
        tl.store(OUTPUT + row * OUTPUT_STRIDE + j, ids, j < KEEP)
    else:
        tl.store(OUTPUT + row * OUTPUT_STRIDE + chunk * CAPACITY + j, best)


@triton.jit  # type: ignore[untyped-decorator]
def _ordered_ids(  # type: ignore[no-untyped-def]
    INPUT,
    OUTPUT,
    LENGTHS,
    ACTIVE,
    INPUT_STRIDE: tl.constexpr,
    KEEP: tl.constexpr,
    CAPACITY: tl.constexpr,
    COMPRESSION: tl.constexpr,
    GROUPS: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(ACTIVE + row) == 0:
        return
    complete = tl.load(LENGTHS + row) // COMPRESSION
    i = tl.arange(0, CAPACITY)
    if complete <= KEEP:
        ids = tl.where(i < complete, i, GROUPS)
    else:
        keys = tl.load(INPUT + row * INPUT_STRIDE + i)
        ids = tl.where(i < KEEP, (0xFFFFFFFF - (keys & 0xFFFFFFFF)).to(tl.int32), GROUPS)
        if CAPACITY > 1:
            ids = tl.sort(ids, descending=False)
    tl.store(OUTPUT + row * KEEP + i, ids, i < KEEP)


@triton.jit  # type: ignore[untyped-decorator]
def _expand(  # type: ignore[no-untyped-def]
    IDS,
    LENGTHS,
    ACTIVE,
    OUTPUT,
    COUNTS,
    KEEP: tl.constexpr,
    COMPRESSION: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    ID_STRIDE: tl.constexpr = 0,
):
    row = tl.program_id(0)
    if tl.load(ACTIVE + row) != 0:
        length = tl.load(LENGTHS + row)
        complete = length // COMPRESSION
        selected = tl.minimum(complete, KEEP) * COMPRESSION
        count = selected + length % COMPRESSION
        i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        stride: tl.constexpr = ID_STRIDE if ID_STRIDE else KEEP
        block = tl.load(IDS + row * stride + i // COMPRESSION,
                        (complete > KEEP) & (i < selected) & (i < OUT_WIDTH), 0)
        block = tl.where(complete <= KEEP, i // COMPRESSION, block)
        token = tl.where(
            i < selected,
            block * COMPRESSION + i % COMPRESSION,
            complete * COMPRESSION + i - selected,
        )
        tl.store(OUTPUT + row * OUT_WIDTH + i, tl.where(i < count, token, -1), i < OUT_WIDTH)
        if tl.program_id(1) == 0:
            tl.store(COUNTS + row, count)




class PagedBlockSelection:
    """Fixed-address selection using a geometry-sized score workspace.

    Score scratch costs ``8*M*page_table_width*groups_per_page`` bytes. Selection
    keeps ``min(budget/compression, max_groups)`` entries per row. All operands
    and merge scratch are allocated before capture; replay reads no host lengths.
    With P=next_power_of_two(keep), the two merge buffers cost
    8*M*P*(ceil(groups/(2*P)) + ceil(ceil(groups/(2*P))/2)) bytes.
    Complete groups that fit the budget bypass ranking on the device.
    The executor proves live lengths, page indices, finite operands and exclusive
    output ownership. Inactive rows neither read cache pages nor write outputs.
    """

    def __init__(
        self,
        query: Tensor,
        keys: Tensor,
        page_table: Tensor,
        visible_lengths: Tensor,
        row_activity: Tensor,
        output: Tensor,
        counts: Tensor,
        *,
        compression: int,
        token_budget: int,
        parts: int = 1,
        rank: int = 0,
        group=None,
        workspace: Callable[[str, tuple[int, ...], torch.dtype], Tensor] | None = None,
    ) -> None:
        tensors = (query, keys, page_table, visible_lengths, row_activity, output, counts)
        if query.device.type != "cuda" or any(
            t.device != query.device or (t is not keys and not t.is_contiguous()) for t in tensors
        ):
            raise ValueError("block selection requires contiguous same-device CUDA tensors")
        if not slot_storage_contiguous(keys):
            raise ValueError("selection cache requires dense nonoverlapping pages")
        if (
            query.ndim != 3
            or min(query.shape) <= 0
            or query.dtype not in (torch.bfloat16, torch.float16)
            or keys.ndim != 3
            or min(keys.shape) <= 0
            or keys.shape[-1] != query.shape[-1]
            or keys.dtype != query.dtype
            or page_table.ndim != 2
            or page_table.shape[0] != query.shape[0]
            or page_table.shape[1] <= 0
            or type(compression) is not int
            or compression <= 0
            or type(token_budget) is not int
            or token_budget < compression
            or token_budget % compression
        ):
            raise ValueError("block selection query/cache/compression geometry differs")
        rows, heads, width = query.shape
        if (
            any(t.dtype != torch.int32 for t in tensors[2:])
            or any(t.shape != (rows,) for t in (visible_lengths, row_activity, counts))
            or output.shape != (rows, token_budget + compression - 1)
        ):
            raise ValueError("block selection metadata/output geometry differs")
        for writable in (output, counts):
            if any(
                t is not writable
                and t.untyped_storage().data_ptr() == writable.untyped_storage().data_ptr()
                for t in tensors
            ):
                raise ValueError("selection output/counts may not alias operands")
        if (type(parts) is not int or parts < 1 or type(rank) is not int or not 0 <= rank < parts):
            raise ValueError("indexer ownership requires a valid partition count and lane")
        if parts > 1:
            import torch.distributed as dist
            if group is None or dist.get_world_size(group) != parts or dist.get_rank(group) != rank:
                raise ValueError("indexer process group disagrees with its block ownership")
        local_groups = page_table.shape[1] * keys.shape[1]
        groups = local_groups * parts
        if groups * compression >= 2**31:
            raise ValueError("token selection exceeds INT32 logical indexing")
        keep = min(token_budget // compression, groups)
        scratch_storage: set[int] = set()
        operand_storage = {t.untyped_storage().data_ptr() for t in tensors}

        def empty(role: str, *shape: int, dtype: torch.dtype = torch.int64) -> Tensor:
            # The caller may reuse this complete selection's scratch only after
            # launch() and its exchanges have consumed it. Distinct roles within
            # one selection remain disjoint; outputs, KV and epochs stay private.
            value = (torch.empty(shape, device=query.device, dtype=dtype) if workspace is None
                     else workspace(role, shape, dtype))
            if (value.shape != shape or value.dtype != dtype or value.device != query.device
                    or not value.is_contiguous()):
                raise ValueError("selection workspace differs from its declared ABI")
            pointer = value.untyped_storage().data_ptr()
            if pointer in operand_storage or pointer in scratch_storage:
                raise ValueError("selection workspace aliases a live operand or scratch role")
            scratch_storage.add(pointer)
            return value

        order = empty("scores", rows, local_groups)
        # Logical token bounds above also prove every block ID fits INT32.
        # Scores retain their lossless INT64 encoding; decisions carry IDs only.
        sorted_ids = empty("sorted_ids", rows, keep, dtype=torch.int32)
        # Bound launch metadata by device thread capacity, not the model's
        # maximum context. Longer live contexts advance through identical GB
        # tiles inside each program; their arithmetic and ownership stay fixed.
        properties = torch.cuda.get_device_properties(query.device)
        program_threads = 4 * 32
        hardware_threads = properties.multi_processor_count * properties.max_threads_per_multi_processor
        score_tiles = min(triton.cdiv(local_groups, 16),
                          max(1, triton.cdiv(hardware_threads, rows * program_threads)))
        self._scores = PackedOperatorBinding(
            order,
            (
                _PackedLaunch.compile(
                    _scores,
                    (query, keys, page_table, visible_lengths, row_activity, order),
                    (
                        heads,
                        width,
                        local_groups,
                        page_table.shape[1],
                        keys.shape[1],
                        compression,
                        keep,
                        math.sqrt(width),
                        triton.next_power_of_2(width),
                        16, parts, rank, keys.stride(0),
                    ),
                    (rows, score_tiles),
                    {"num_warps": 4, "enable_fp_fusion": False},
                ),
            ),
        )
        # Each leaf retains min(keep, all local keys), rounded up to P.
        # A global top-keep key belongs to that local candidate set; sorted
        # pairwise merges preserve the invariant without re-sorting the runs.
        # P, tree depth and scratch follow budget/context geometry, never M/SKU.
        p = triton.next_power_of_2(min(keep, local_groups))
        chunks = triton.cdiv(local_groups, 2 * p)
        tree_workspace = (
            empty("tree_even", rows, chunks * p),
            empty("tree_odd", rows, max(1, triton.cdiv(chunks, 2)) * p),
        )
        packed_candidates = (empty("local_candidates", rows, p)
                             if parts > 1 else None)
        source, coverage, level = order, 2 * p, 0
        self._tree: list[PackedOperatorBinding] = []
        while True:
            target = (packed_candidates if parts > 1 and chunks == 1
                      else tree_workspace[level % 2])
            call = _PackedLaunch.compile(
                _select_tree,
                (source, target, visible_lengths, row_activity),
                (
                    source.shape[1],
                    target.shape[1],
                    keep,
                    p,
                    compression,
                    coverage,
                    level == 0,
                    p.bit_length() - 1, parts, rank, 0,
                ),
                (rows, chunks),
                {"num_warps": 4},
            )
            self._tree.append(PackedOperatorBinding(target, (call,)))
            source = target
            if chunks == 1:
                break
            chunks = triton.cdiv(chunks, 2)
            coverage *= 2
            level += 1
        self._candidate_exchange = None
        self._decision_exchange = None
        extra_scratch = []
        self._global_tree = []
        merge_active = row_activity
        if parts > 1:
            from cubic_serve.ops.cuda.collective.column_gather import ColumnGather

            # Only rank zero merges the global decision. Other ranks do not
            # repeat global top-k, but receive its lossless selected block IDs.
            candidates = empty("global_candidates", rows, p * parts)
            self._candidate_exchange = ColumnGather(
                packed_candidates.view(torch.bfloat16), candidates.view(torch.bfloat16), group,
                ranking_guard=(visible_lengths, row_activity, (keep + 1) * compression),
                destination_rank=0,
            )
            merge_active = row_activity if rank == 0 else torch.zeros_like(row_activity)
            source = candidates
            total = parts * p
            candidate_run = p
            p = triton.next_power_of_2(keep)
            chunks = triton.cdiv(total, 2 * p)
            coverage, level = 2 * p, 0
            while True:
                target = (sorted_ids if chunks == 1 else
                          empty(f"global_tree_{level}", rows, chunks * p))
                call = _PackedLaunch.compile(
                    _select_tree, (source, target, visible_lengths, merge_active),
                    (source.shape[1], target.shape[1], keep, p, compression, coverage,
                     level == 0 and candidate_run != p, p.bit_length() - 1,
                     1, 0, total, chunks == 1, groups),
                    (rows, chunks), {"num_warps": 4},
                )
                self._global_tree.append(PackedOperatorBinding(target, (call,)))
                if target is not sorted_ids:
                    extra_scratch.append(target)
                source = target
                if chunks == 1:
                    break
                chunks = triton.cdiv(chunks, 2)
                coverage *= 2
                level += 1
            extra_scratch.extend((packed_candidates, candidates,
                                  self._candidate_exchange.arena.epochs))
            if rank != 0:
                extra_scratch.append(merge_active)
        self._ids = None
        if parts == 1:
            call = _PackedLaunch.compile(
                _ordered_ids,
                (source, sorted_ids, visible_lengths, merge_active),
                (source.shape[1], keep, p, compression, groups),
                (rows,),
                {"num_warps": 4},
            )
            self._ids = PackedOperatorBinding(sorted_ids, (call,))
        expansion_ids = sorted_ids
        if parts > 1:
            decisions = empty("decisions", rows, keep * parts, dtype=torch.int32)
            self._decision_exchange = ColumnGather(
                sorted_ids.view(torch.bfloat16), decisions.view(torch.bfloat16), group,
                ranking_guard=(visible_lengths, row_activity, (keep + 1) * compression),
                source_rank=0,
            )
            expansion_ids = decisions
            extra_scratch.extend((decisions, self._decision_exchange.arena.epochs))
        self._expand = PackedOperatorBinding(
            output,
            (
                _PackedLaunch.compile(
                    _expand,
                    (expansion_ids, visible_lengths, row_activity, output, counts),
                    (keep, compression, output.shape[1], 256, expansion_ids.shape[1]),
                    (rows, triton.cdiv(output.shape[1], 256)),
                    {"num_warps": 4},
                ),
            ),
        )
        self.output, self.counts = output, counts
        self.scratch = order, sorted_ids, *tree_workspace, *extra_scratch
        self._keep = keep

    def launch(self) -> None:
        self._scores.launch()
        for stage in self._tree:
            stage.launch()
        if self._candidate_exchange is not None:
            self._candidate_exchange.launch()
            for stage in self._global_tree:
                stage.launch()
        if self._ids is not None:
            self._ids.launch()
        if self._decision_exchange is not None:
            self._decision_exchange.launch()
        self._expand.launch()


@triton.jit  # type: ignore[untyped-decorator]
def _compress_keys(  # type: ignore[no-untyped-def]
    KEYS,
    POSITIONS,
    LOGICAL,
    HISTORY,
    POSITION_HISTORY,
    SOURCES,
    DESTINATIONS,
    SEGMENTS,
    ACTIVE,
    POOLED,
    GROUP_POSITIONS,
    COMPLETE,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    RATIO: tl.constexpr,
    RB: tl.constexpr,
    DB: tl.constexpr,
    HISTORY_STRIDE: tl.constexpr,
    POSITION_STRIDE: tl.constexpr,
):
    start = tl.program_id(0)
    length = tl.load(SEGMENTS + start)
    if (tl.load(ACTIVE + start) != 0) & (length > 0):
        source = tl.load(SOURCES + start).to(tl.int64)
        r, d, axis = tl.arange(0, RB), tl.arange(0, DB), tl.arange(0, 4)
        history = tl.load(
            HISTORY + source * HISTORY_STRIDE + r[:, None] * WIDTH + d[None, :],
            (source >= 0) & (r[:, None] < RATIO) & (d[None, :] < WIDTH),
            0,
        )
        positions = tl.load(
            POSITION_HISTORY + source * POSITION_STRIDE + r[:, None] * 3 + axis[None, :],
            (source >= 0) & (r[:, None] < RATIO) & (axis[None, :] < 3),
            0,
        )
        shift_keys = tl.broadcast_to(tl.minimum(r[:, None] + 1, RB - 1), (RB, DB))
        shift_positions = tl.broadcast_to(tl.minimum(r[:, None] + 1, RB - 1), (RB, 4))
        for step in range(tl.minimum(length, ROWS - start)):
            row = start + step
            if tl.load(ACTIVE + row) != 0:
                key = tl.load(KEYS + row * WIDTH + d, d < WIDTH, 0)
                position = tl.load(POSITIONS + row * 3 + axis, axis < 3, 0)
                history = tl.gather(history, shift_keys, 0)
                history = tl.where(r[:, None] == RATIO - 1, key[None, :], history)
                positions = tl.gather(positions, shift_positions, 0)
                positions = tl.where(r[:, None] == RATIO - 1, position[None, :], positions)
                complete = (tl.load(LOGICAL + row) + 1) % RATIO == 0
                pooled = tl.div_rn(
                    tl.sum(tl.where(r[:, None] < RATIO, history.to(tl.float32), 0), 0), float(RATIO)
                )
                first = tl.sum(tl.where(r[:, None] == 0, positions, 0), 0)
                tl.store(POOLED + row * WIDTH + d, tl.where(complete, pooled, 0), d < WIDTH)
                tl.store(GROUP_POSITIONS + row * 3 + axis, tl.where(complete, first, 0), axis < 3)
                tl.store(COMPLETE + row, complete)
                destination = tl.load(DESTINATIONS + row).to(tl.int64)
                tl.store(
                    HISTORY + destination * HISTORY_STRIDE + r[:, None] * WIDTH + d[None, :],
                    history,
                    (destination >= 0) & (r[:, None] < RATIO) & (d[None, :] < WIDTH),
                )
                tl.store(
                    POSITION_HISTORY + destination * POSITION_STRIDE + r[:, None] * 3 + axis[None, :],
                    positions,
                    (destination >= 0) & (r[:, None] < RATIO) & (axis[None, :] < 3),
                )


def bind_causal_key_compression(
    keys: Tensor,
    positions: Tensor,
    logical_positions: Tensor,
    history: Tensor,
    position_history: Tensor,
    sources: Tensor,
    destinations: Tensor,
    segment_lengths: Tensor,
    row_activity: Tensor,
    pooled: Tensor,
    group_positions: Tensor,
    complete: Tensor,
) -> PackedOperatorBinding:
    """Mean complete groups, preserving exact group-start MRoPE positions.

    Ring storage is represented oldest-first to give chunked and M1 executions
    the identical reduction tree. The compression ratio is history.shape[1].
    Every logical token may checkpoint its stored raw keys and INT64 positions;
    incomplete groups publish a false completion flag and zero scratch outputs.
    Caller proves contiguous logical positions within each active segment and
    initial history provenance, including fresh sequences starting at zero.
    """
    tensors = (
        keys,
        positions,
        logical_positions,
        history,
        position_history,
        sources,
        destinations,
        segment_lengths,
        row_activity,
        pooled,
        group_positions,
        complete,
    )
    if keys.device.type != "cuda" or any(
        t.device != keys.device or (t is not history and t is not position_history
                                   and not t.is_contiguous()) for t in tensors
    ):
        raise ValueError("compression requires contiguous same-device CUDA operands")
    if not all(slot_storage_contiguous(t) for t in (history, position_history)):
        raise ValueError("compression history requires dense nonoverlapping slots")
    if (
        keys.ndim != 2
        or min(keys.shape) <= 0
        or keys.dtype not in (torch.bfloat16, torch.float16)
        or history.ndim != 3
        or min(history.shape) <= 0
        or history.shape[-1] != keys.shape[-1]
    ):
        raise ValueError("compression keys/history require positive FP16/BF16 geometry")
    rows, width = keys.shape
    slots, ratio, _ = history.shape
    if (
        pooled.shape != keys.shape
        or history.dtype != keys.dtype
        or pooled.dtype != keys.dtype
        or positions.shape != (rows, 3)
        or group_positions.shape != positions.shape
        or position_history.shape != (slots, ratio, 3)
        or logical_positions.shape != (rows,)
        or any(
            t.dtype != torch.int64
            for t in (positions, position_history, logical_positions, group_positions)
        )
        or any(
            t.dtype != torch.int32 or t.shape != (rows,)
            for t in (sources, destinations, segment_lengths, row_activity, complete)
        )
    ):
        raise ValueError("compression state, position or row metadata differs")
    for writable in (history, position_history, pooled, group_positions, complete):
        if any(
            t is not writable
            and storage_regions_overlap(t, writable)
            for t in tensors
        ):
            raise ValueError("compression writable operands may not alias another operand")
    call = _PackedLaunch.compile(
        _compress_keys,
        tensors,
        (rows, width, ratio, triton.next_power_of_2(ratio), triton.next_power_of_2(width),
         history.stride(0), position_history.stride(0)),
        (rows,),
        {"num_warps": 4, "enable_fp_fusion": False},
    )
    return PackedOperatorBinding(pooled, (call,))


@triton.jit  # type: ignore[untyped-decorator]
def _store_compressed(  # type: ignore[no-untyped-def]
    VALUES,
    CACHE,
    PAGES,
    OFFSETS,
    COMPLETE,
    ACTIVE,
    WIDTH: tl.constexpr,
    PAGE_GROUPS: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
    PARTS: tl.constexpr = 1,
    RANK: tl.constexpr = 0,
    CACHE_STRIDE: tl.constexpr = 0,
):
    row = tl.program_id(0)
    owned = (tl.load(OFFSETS + row) // RATIO) % PARTS == RANK
    if (tl.load(ACTIVE + row) != 0) & (tl.load(COMPLETE + row) != 0) & owned:
        d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        page = tl.load(PAGES + row).to(tl.int64)
        group = tl.load(OFFSETS + row) // (RATIO * PARTS)
        values = tl.load(VALUES + row * WIDTH + d, d < WIDTH, 0)
        tl.store(CACHE + page * CACHE_STRIDE + group * WIDTH + d, values, d < WIDTH)


def bind_compressed_page_write(
    values: Tensor,
    cache: Tensor,
    page_destinations: Tensor,
    token_offsets: Tensor,
    complete: Tensor,
    row_activity: Tensor,
    *,
    compression: int,
    parts: int = 1,
    rank: int = 0,
) -> PackedOperatorBinding:
    """Publish normalized/rotated complete keys into an owned auxiliary page.

    COW must precede this write. The caller proves page/offset bounds, a main KV
    page size of cache.shape[1]*compression, and disjoint complete-group writes.
    """
    tensors = values, cache, page_destinations, token_offsets, complete, row_activity
    if values.device.type != "cuda" or any(
        t.device != values.device or (t is not cache and not t.is_contiguous()) for t in tensors
    ):
        raise ValueError("compressed page write requires contiguous same-device CUDA tensors")
    if not slot_storage_contiguous(cache):
        raise ValueError("compressed cache requires dense nonoverlapping pages")
    if (
        values.ndim != 2
        or min(values.shape) <= 0
        or cache.ndim != 3
        or min(cache.shape) <= 0
        or cache.shape[-1] != values.shape[-1]
        or cache.dtype != values.dtype
        or values.dtype not in (torch.bfloat16, torch.float16)
        or type(compression) is not int
        or compression <= 0
        or type(parts) is not int or parts < 1
        or type(rank) is not int or not 0 <= rank < parts
        or any(t.dtype != torch.int32 or t.shape != (values.shape[0],) for t in tensors[2:])
    ):
        raise ValueError("compressed page write geometry differs")
    if any(
        storage_regions_overlap(t, cache)
        for t in (values, *tensors[2:])
    ):
        raise ValueError("compressed cache may not alias write inputs")
    call = _PackedLaunch.compile(
        _store_compressed,
        tensors,
        (values.shape[1], cache.shape[1], compression, 128, parts, rank, cache.stride(0)),
        (values.shape[0], triton.cdiv(values.shape[1], 128)),
        {"num_warps": 4},
    )
    return PackedOperatorBinding(cache, (call,))


@triton.jit
def _prepare_indexer(
    PROJECTED, QWEIGHT, QNORMALIZED, COSINE, SINE, POSITIONS, LOGICAL,
    HISTORY, POSITION_HISTORY, SOURCES, DESTINATIONS, SEGMENTS, ACTIVE,
    PACKED, GROUP_POSITIONS, COMPLETE,
    ROWS: tl.constexpr, HEADS: tl.constexpr, WIDTH: tl.constexpr,
    KEY_WIDTH: tl.constexpr, PACK_WIDTH: tl.constexpr, RATIO: tl.constexpr,
    ROTARY: tl.constexpr, EPSILON: tl.constexpr,
    HB: tl.constexpr, DB: tl.constexpr, KB: tl.constexpr, RB: tl.constexpr,
    HISTORY_STRIDE: tl.constexpr, POSITION_STRIDE: tl.constexpr,
):
    start = tl.program_id(0)
    length = tl.load(SEGMENTS + start)
    if tl.load(ACTIVE + start) != 0 and length > 0:
        source = tl.load(SOURCES + start).to(tl.int64)
        r, d, axis = tl.arange(0, RB), tl.arange(0, KB), tl.arange(0, 4)
        history = tl.load(HISTORY + source * HISTORY_STRIDE + r[:, None] * KEY_WIDTH + d[None, :],
                          (source >= 0) & (r[:, None] < RATIO) & (d[None, :] < KEY_WIDTH), 0)
        positions = tl.load(POSITION_HISTORY + source * POSITION_STRIDE + r[:, None] * 3 + axis[None, :],
                            (source >= 0) & (r[:, None] < RATIO) & (axis[None, :] < 3), 0)
        shifts = tl.broadcast_to(tl.minimum(r[:, None] + 1, RB - 1), (RB, KB))
        pshifts = tl.broadcast_to(tl.minimum(r[:, None] + 1, RB - 1), (RB, 4))
        for step in range(tl.minimum(length, ROWS - start)):
            row = start + step
            if tl.load(ACTIVE + row) != 0:
                if HEADS > 0:
                    h, col = tl.arange(0, HB), tl.arange(0, DB)
                    q = tl.load(PROJECTED + row * (HEADS * WIDTH + KEY_WIDTH)
                                + h[:, None] * WIDTH + col[None, :],
                                (h[:, None] < HEADS) & (col[None, :] < WIDTH), 0).to(tl.float32)
                    weight = tl.load(QWEIGHT + col, col < WIDTH, 0).to(tl.float32)
                    variance = libdevice.div_rn(tl.sum(q * q, 1), float(WIDTH))
                    inverse = libdevice.rsqrt_rn(variance + EPSILON)
                    normalized = (q * inverse[:, None] * (weight[None, :] + 1.0)).to(QNORMALIZED.dtype.element_ty)
                    tl.store(QNORMALIZED + (row * HEADS + h[:, None]) * WIDTH + col[None, :],
                             normalized, (h[:, None] < HEADS) & (col[None, :] < WIDTH))
                    values = normalized.to(tl.float32)
                    pair = tl.where(col < ROTARY // 2, col + ROTARY // 2, col - ROTARY // 2)
                    paired = tl.gather(values, tl.broadcast_to(pair[None, :], (HB, DB)), 1)
                    paired = tl.where(col[None, :] < ROTARY // 2, -paired, paired)
                    cos = tl.load(COSINE + row * ROTARY + col, col < ROTARY, 0).to(tl.float32)
                    sin = tl.load(SINE + row * ROTARY + col, col < ROTARY, 0).to(tl.float32)
                    rotated = tl.where(col[None, :] < ROTARY, values * cos[None, :] + paired * sin[None, :], values)
                    tl.store(PACKED + row * PACK_WIDTH + h[:, None] * WIDTH + col[None, :],
                             rotated, (h[:, None] < HEADS) & (col[None, :] < WIDTH))
                key = tl.load(PROJECTED + row * (HEADS * WIDTH + KEY_WIDTH) + HEADS * WIDTH + d,
                              d < KEY_WIDTH, 0)
                position = tl.load(POSITIONS + axis * ROWS + row, axis < 3, 0)
                history = tl.gather(history, shifts, 0)
                history = tl.where(r[:, None] == RATIO - 1, key[None, :], history)
                positions = tl.gather(positions, pshifts, 0)
                positions = tl.where(r[:, None] == RATIO - 1, position[None, :], positions)
                complete = (tl.load(LOGICAL + row) + 1) % RATIO == 0
                pooled = libdevice.div_rn(tl.sum(tl.where(r[:, None] < RATIO, history.to(tl.float32), 0), 0), float(RATIO))
                first = tl.sum(tl.where(r[:, None] == 0, positions, 0), 0)
                tl.store(PACKED + row * PACK_WIDTH + PACK_WIDTH - KEY_WIDTH + d,
                         tl.where(complete, pooled, 0), d < KEY_WIDTH)
                tl.store(GROUP_POSITIONS + axis * ROWS + row, tl.where(complete, first, 0), axis < 3)
                tl.store(COMPLETE + row, complete)
                destination = tl.load(DESTINATIONS + row).to(tl.int64)
                tl.store(HISTORY + destination * HISTORY_STRIDE + r[:, None] * KEY_WIDTH + d[None, :],
                         history, (destination >= 0) & (r[:, None] < RATIO) & (d[None, :] < KEY_WIDTH))
                tl.store(POSITION_HISTORY + destination * POSITION_STRIDE + r[:, None] * 3 + axis[None, :],
                         positions, (destination >= 0) & (r[:, None] < RATIO) & (axis[None, :] < 3))
            else:
                tl.store(GROUP_POSITIONS + axis * ROWS + row, 0, axis < 3)
                tl.store(COMPLETE + row, 0)
    else:
        # Continuation rows are written by their segment owner. Padding alone
        # owns no request state but needs safe indices for rotary-table gather.
        if tl.load(ACTIVE + start) == 0:
            axis = tl.arange(0, 4)
            tl.store(GROUP_POSITIONS + axis * ROWS + start, 0, axis < 3)
            tl.store(COMPLETE + start, 0)


@triton.jit
def _finish_indexer(
    GATHERED, QUERY, POOLED, KNORMALIZED, KWEIGHT, COSINE, SINE,
    COMPLETE, ACTIVE, OFFSETS, PAGES, CACHE,
    ROWS: tl.constexpr, HEADS: tl.constexpr, WIDTH: tl.constexpr,
    PARTS: tl.constexpr, RANK: tl.constexpr, PACK_WIDTH: tl.constexpr,
    PAGE_GROUPS: tl.constexpr, RATIO: tl.constexpr, ROTARY: tl.constexpr,
    EPSILON: tl.constexpr, QB: tl.constexpr, DB: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(ACTIVE + row) != 0:
        qcol = tl.arange(0, QB)
        head, d = qcol // WIDTH, qcol % WIDTH
        owner = ((head + 1) * PARTS - 1) // HEADS
        local_head = head - HEADS * owner // PARTS
        q = tl.load(GATHERED + (row * PARTS + owner) * PACK_WIDTH + local_head * WIDTH + d,
                    qcol < HEADS * WIDTH, 0)
        tl.store(QUERY + row * HEADS * WIDTH + qcol, q, qcol < HEADS * WIDTH)
        col = tl.arange(0, DB)
        kw: tl.constexpr = WIDTH // PARTS
        pooled = tl.load(GATHERED + (row * PARTS + col // kw) * PACK_WIDTH
                         + PACK_WIDTH - kw + col % kw, col < WIDTH, 0)
        tl.store(POOLED + row * WIDTH + col, pooled, col < WIDTH)
        group = tl.load(OFFSETS + row) // RATIO
        if tl.load(COMPLETE + row) != 0 and group % PARTS == RANK:
            x = pooled.to(tl.float32)
            weight = tl.load(KWEIGHT + col, col < WIDTH, 0).to(tl.float32)
            variance = libdevice.div_rn(tl.sum(x * x, 0), float(WIDTH))
            inverse = libdevice.rsqrt_rn(variance + EPSILON)
            norm = (x * inverse * (weight + 1.0)).to(KNORMALIZED.dtype.element_ty)
            tl.store(KNORMALIZED + row * WIDTH + col, norm, col < WIDTH)
            values = norm.to(tl.float32)
            pair = tl.where(col < ROTARY // 2, col + ROTARY // 2, col - ROTARY // 2)
            paired = tl.gather(values, pair, 0)
            paired = tl.where(col < ROTARY // 2, -paired, paired)
            cos = tl.load(COSINE + row * ROTARY + col, col < ROTARY, 0).to(tl.float32)
            sin = tl.load(SINE + row * ROTARY + col, col < ROTARY, 0).to(tl.float32)
            rotated = tl.where(col < ROTARY, values * cos + paired * sin, values)
            page = tl.load(PAGES + row).to(tl.int64)
            tl.store(CACHE + page * CACHE_STRIDE + (group // PARTS) * WIDTH + col,
                     rotated, col < WIDTH)


class IndexerPreparation:
    """One current-query/pooled-key exchange between two fused local stages."""

    def __init__(self, projected, qweight, kweight, cosine, sine, positions, logical,
                 history, position_history, sources, destinations, segments, activity,
                 offsets, pages, cache, query, group_positions, group_cosine, group_sine,
                 *, plan, group, epsilon):
        from cubic_serve.ops.cuda.collective.column_gather import ColumnGather
        import torch.distributed as dist
        import math

        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("indexer normalization requires a finite positive epsilon")
        rows = projected.shape[0]
        dtype, device = projected.dtype, projected.device
        if (plan.size < 2 or group is None or dist.get_world_size(group) != plan.size
                or dist.get_rank(group) != plan.rank):
            raise ValueError("indexer preparation requires its declared TP group")
        tensors = (projected, qweight, kweight, cosine, sine, positions, logical, history,
                   position_history, sources, destinations, segments, activity, offsets,
                   pages, cache, query, group_positions, group_cosine, group_sine)
        if any(t.device != device or (all(t is not field for field in
                   (history, position_history, cache)) and not t.is_contiguous()) for t in tensors):
            raise ValueError("indexer preparation requires contiguous same-device storage")
        if not all(slot_storage_contiguous(t) for t in (history, position_history, cache)):
            raise ValueError("indexer state/cache requires dense nonoverlapping slots")
        if (device.type != 'cuda' or dtype not in (torch.bfloat16, torch.float16)
                or projected.shape != (rows, plan.local_query_heads * plan.head_dim + plan.key_width)
                or qweight.shape != (plan.head_dim,) or kweight.shape != qweight.shape
                or query.shape != (rows, plan.query_heads, plan.head_dim)
                or history.shape[1:] != (plan.compression, plan.key_width)
                or position_history.shape != (*history.shape[:2], 3)
                or positions.shape != (3, rows) or group_positions.shape != positions.shape
                or cosine.ndim != 2 or cosine.shape[0] != rows
                or not 0 < cosine.shape[1] <= plan.head_dim or cosine.shape[1] % 2
                or any(t.shape != cosine.shape for t in (sine, group_cosine, group_sine))
                or cache.ndim != 3 or cache.shape[-1] != plan.head_dim
                or any(t.dtype != dtype for t in (qweight, kweight, query, cache, history,
                                                  cosine, sine, group_cosine, group_sine))
                or any(t.dtype != torch.int32 or t.shape != (rows,) for t in
                       (sources, destinations, segments, activity, offsets, pages))
                or logical.dtype != torch.int64 or logical.shape != (rows,)
                or any(t.dtype != torch.int64 for t in (positions, position_history, group_positions))):
            raise ValueError("indexer preparation geometry/dtypes disagree with ownership")
        self.plan, self.projected = plan, projected
        self.qweight, self.kweight, self.epsilon = qweight, kweight, epsilon
        self.activity, self.offsets = activity, offsets
        # Empty query owners still bind a valid private scratch pointer. HEADS=0
        # removes all query math and stores; this storage is not a logical head.
        self.normalized_query = torch.empty((rows, max(1, plan.local_query_heads), plan.head_dim), device=device, dtype=dtype)
        self.pooled = torch.empty((rows, plan.head_dim), device=device, dtype=dtype)
        self.normalized_key = torch.empty_like(self.pooled)
        self.complete = torch.empty((rows,), device=device, dtype=torch.int32)
        packed = torch.zeros((rows, plan.exchange_width), device=device, dtype=dtype)
        gathered = torch.empty((rows, plan.exchange_width * plan.size), device=device, dtype=dtype)
        self.exchange = ColumnGather(packed, gathered, group)
        self.scratch = (self.normalized_query, self.pooled, self.normalized_key, self.complete,
                        packed, gathered, self.exchange.arena.epochs)
        call = _PackedLaunch.compile(
            _prepare_indexer,
            (projected, qweight, self.normalized_query, cosine, sine, positions, logical,
             history, position_history, sources, destinations, segments, activity,
             packed, group_positions, self.complete),
            (rows, plan.local_query_heads, plan.head_dim, plan.key_width, plan.exchange_width,
             plan.compression, cosine.shape[1], epsilon,
             triton.next_power_of_2(max(1, plan.local_query_heads)), triton.next_power_of_2(plan.head_dim),
             triton.next_power_of_2(plan.key_width), triton.next_power_of_2(plan.compression),
             history.stride(0), position_history.stride(0)),
            (rows,), {"num_warps": 4, "enable_fp_fusion": False},
        )
        self._prepare = PackedOperatorBinding(packed, (call,))
        call = _PackedLaunch.compile(
            _finish_indexer,
            (gathered, query, self.pooled, self.normalized_key, kweight, group_cosine, group_sine,
             self.complete, activity, offsets, pages, cache),
            (rows, plan.query_heads, plan.head_dim, plan.size, plan.rank, plan.exchange_width,
             cache.shape[1], plan.compression, cosine.shape[1], epsilon,
             triton.next_power_of_2(plan.query_heads * plan.head_dim), triton.next_power_of_2(plan.head_dim),
             cache.stride(0)),
            (rows,), {"num_warps": 4, "enable_fp_fusion": False},
        )
        self._finish = PackedOperatorBinding(query, (call,))

    def prepare(self):
        self._prepare.launch()

    def finish(self):
        self.exchange.launch()
        self._finish.launch()
