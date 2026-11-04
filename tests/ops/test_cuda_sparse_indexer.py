import os

import pytest
import torch

from cubic_serve.executor.runner import ReferenceExecutionContext, configure_reference_math
from cubic_serve.ops.cuda.attention.sparse_indexer import PagedBlockSelection
from cubic_serve.ops.eager.sparse_indexer import select_blocks

pytestmark = pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required"
)


def test_capacity_scratch_is_unconsumed_outside_live_ranking_extent():
    configure_reference_math()
    torch.manual_seed(882)
    pages, page_groups, width, heads = 16384, 4, 128, 4
    dtype = torch.bfloat16
    queries = torch.randint(-2, 3, (1, heads, width)).to(dtype)
    keys = torch.randint(-2, 3, (4096, width)).to(dtype)
    permutation = torch.randperm(pages)
    cache = torch.full((pages, page_groups, width), float("nan"), device="cuda", dtype=dtype)
    cache[permutation[:1024].cuda()] = keys.reshape(1024, page_groups, width).cuda()
    lengths = (16384, 7, 2051, 2052, 4096, 2047)
    reference = {}
    with ReferenceExecutionContext():
        for length in lengths:
            chosen = select_blocks(
                queries[:, None], keys[None], torch.tensor([[length]]),
                compression=4, token_budget=2048,
            )[0, 0]
            reference[length] = chosen[chosen >= 0].sort().values
    common = {}
    # The large bucket forces multiple live tiles through each resource-bounded
    # score program; the same sealed graph also replays select-all lengths.
    for capacity in (1, 8, 512):
        query = torch.zeros(capacity, heads, width, device="cuda", dtype=dtype)
        table = permutation.int().cuda()[None].expand(capacity, -1).contiguous()
        visible, active, counts = [
            torch.zeros(capacity, device="cuda", dtype=torch.int32) for _ in range(3)
        ]
        output = torch.empty(capacity, 2051, device="cuda", dtype=torch.int32)
        op = PagedBlockSelection(
            query, cache, table, visible, active, output, counts,
            compression=4, token_budget=2048,
        )
        op.launch()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op.launch()
        for length in lengths:
            row = length % capacity
            query.fill_(float("nan"))
            query[row].copy_(queries[0])
            active.zero_()
            active[row] = 1
            visible.fill_(2**30)
            visible[row] = length
            output.fill_(-77)
            counts.fill_(-33)
            for scratch in op.scratch:
                scratch.fill_(torch.iinfo(scratch.dtype).max)
            graph.replay()
            actual, n = output.cpu(), counts.cpu()
            expected = reference[length]
            assert n[row] == expected.numel()
            assert torch.equal(actual[row, :n[row]], expected)
            assert (actual[row, n[row]:] == -1).all()
            inactive = torch.arange(capacity) != row
            assert (actual[inactive] == -77).all()
            assert (n[inactive] == -33).all()
            if length in common:
                assert torch.equal(actual[row], common[length])
            common[length] = actual[row].clone()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("capacity", [1, 8, 16])
def test_serial_selection_workspace_preserves_private_results(dtype, capacity):
    configure_reference_math()
    torch.manual_seed(926)
    pool = {}

    def workspace(role, shape, storage):
        key = role, shape, storage
        if key not in pool:
            pool[key] = torch.empty(shape, dtype=storage, device="cuda")
        return pool[key]

    pages, groups, width, heads = 65, 4, 128, 4
    cache = torch.randint(-2, 3, (pages, groups, width), device="cuda").to(dtype)
    queries = [torch.randint(-2, 3, (capacity, heads, width), device="cuda").to(dtype)
               for _ in range(2)]
    permutation = torch.randperm(pages, device="cuda")
    table = permutation.int()[None].expand(capacity, -1).contiguous()
    visible = torch.full((capacity,), 1037, device="cuda", dtype=torch.int32)
    active = torch.ones_like(visible)
    active[-1] = int(capacity == 1)
    blocks = []
    for index in range(4):
        output = torch.empty(capacity, 15, device="cuda", dtype=torch.int32)
        counts = torch.empty_like(visible)
        blocks.append(PagedBlockSelection(
            queries[index % 2], cache, table, visible, active, output, counts,
            compression=4, token_budget=12, workspace=workspace if index >= 2 else None,
        ))
    assert blocks[2].scratch[0].data_ptr() == blocks[3].scratch[0].data_ptr()
    for block in blocks:
        block.launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for block in blocks:
            block.launch()
    for length in (1, 19, 1037):
        visible.fill_(length)
        for value in pool.values():
            value.fill_(torch.iinfo(value.dtype).max)
        for block in blocks:
            block.output.fill_(-77)
            block.counts.fill_(-33)
        graph.replay()
        for index in range(2):
            assert torch.equal(blocks[index].output, blocks[index + 2].output)
            assert torch.equal(blocks[index].counts, blocks[index + 2].counts)
            with ReferenceExecutionContext():
                expected = select_blocks(
                    queries[index][:1, None].cpu(), cache[permutation].flatten(0, 1)[None].cpu(),
                    torch.tensor([[length]]), compression=4, token_budget=12,
                )[0, 0]
            chosen = expected[expected >= 0].sort().values
            assert torch.equal(blocks[index + 2].output[0, :chosen.numel()].cpu(), chosen)
    graph.reset()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("pages,budget", [(2, 4), (16, 4), (16, 12), (257, 12), (257, 2048)])
def test_selection_reference_cross_m_ties_causal_tail_and_padding(dtype, pages, budget):
    configure_reference_math()
    torch.manual_seed(521)
    heads, width, page_groups, compression = 4, 128, 4, 4
    queries = torch.randint(-2, 3, (6, heads, width)).to(dtype)
    keys = torch.randint(-2, 3, (pages * page_groups, width)).to(dtype)
    keys[5:9] = keys[3]  # Stable index ordering at equal scores.
    queries[1].zero_()  # All scores tied.
    lengths = [1, 19, 7, 12, 17, pages * page_groups * compression - 3]
    reference = []
    with ReferenceExecutionContext():
        for row, length in enumerate(lengths):
            selected = select_blocks(
                queries[row : row + 1, None],
                keys[None],
                torch.tensor([[length]]),
                compression=compression,
                token_budget=budget,
            )[0, 0]
            reference.append(selected[selected >= 0].sort().values)
    # Physical pages are permuted; logical coordinates must drive ranking/ties.
    permutation = torch.randperm(pages)
    cache = torch.full((pages + 2, page_groups, width), float("nan"), dtype=dtype, device="cuda")
    cache[permutation.cuda()] = keys.reshape(pages, page_groups, width).cuda()
    common = {}
    for capacity in (1, 8, 16):
        query = torch.empty((capacity, heads, width), dtype=dtype, device="cuda")
        table = torch.empty((capacity, pages), dtype=torch.int32, device="cuda")
        visible, active, counts = [
            torch.zeros(capacity, dtype=torch.int32, device="cuda") for _ in range(3)
        ]
        output = torch.empty((capacity, budget + compression - 1), dtype=torch.int32, device="cuda")
        op = PagedBlockSelection(
            query,
            cache,
            table,
            visible,
            active,
            output,
            counts,
            compression=compression,
            token_budget=budget,
        )
        op.launch()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op.launch()
        mappings = (
            [(row,) for row in range(6)]
            if capacity == 1
            else [(0, 1, 2, 3, 4, 5), (5, 1, 3, 0, 4, 2)]
        )
        for mapping in mappings:
            query.fill_(float("nan"))
            table.fill_(2**30)
            visible.fill_(2**30)
            active.zero_()
            output.fill_(-77)
            counts.fill_(-33)
            for physical, logical in enumerate(mapping):
                query[physical].copy_(queries[logical])
                table[physical].copy_(permutation)
                visible[physical] = lengths[logical]
                active[physical] = 1
            graph.replay()
            actual, n = output.cpu(), counts.cpu()
            for physical, logical in enumerate(mapping):
                expected = reference[logical]
                assert n[physical] == expected.numel()
                assert torch.equal(actual[physical, : n[physical]], expected)
                assert (actual[physical, n[physical] :] == -1).all()
                if logical in common:
                    assert torch.equal(actual[physical], common[logical])
                common[logical] = actual[physical].clone()
            assert (actual[len(mapping) :] == -77).all()
            assert (n[len(mapping) :] == -33).all()


def test_selection_large_block_count_preserves_stable_ties():
    configure_reference_math()
    groups, compression, budget = 65536, 4, 2048
    query = torch.ones((1, 4, 128), dtype=torch.bfloat16, device="cuda")
    cache = torch.ones((groups // 8, 8, 128), dtype=torch.bfloat16, device="cuda")
    table = torch.arange(groups // 8, dtype=torch.int32, device="cuda")[None]
    visible = torch.tensor([groups * compression - 1], dtype=torch.int32, device="cuda")
    active = torch.ones_like(visible)
    output = torch.empty((1, budget + compression - 1), dtype=torch.int32, device="cuda")
    counts = torch.empty_like(visible)
    op = PagedBlockSelection(
        query,
        cache,
        table,
        visible,
        active,
        output,
        counts,
        compression=compression,
        token_budget=budget,
    )
    op.launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.launch()
    graph.replay()
    expected = torch.cat(
        (torch.arange(budget), torch.arange((groups - 1) * compression, groups * compression - 1))
    ).int()
    assert counts.item() == expected.numel()
    assert torch.equal(output[0].cpu(), expected)


@pytest.mark.parametrize(
    "dtype,ratio,width", [(torch.bfloat16, 4, 128), (torch.float16, 3, 37), (torch.bfloat16, 1, 32)]
)
@pytest.mark.parametrize("strided", [False, True])
def test_compression_checkpoint_chunking_positions_padding_and_page_write(dtype, ratio, width, strided):
    from cubic_serve.ops.cuda.attention.sparse_indexer import (
        bind_causal_key_compression,
        bind_compressed_page_write,
    )

    configure_reference_math()
    torch.manual_seed(893)
    starts, lengths = (2, 0), (7, 3)
    sequences = [torch.randint(-9, 10, (length, width)).to(dtype) / 8 for length in lengths]
    initial = torch.randint(-9, 10, (ratio, width)).to(dtype) / 8
    initial_positions = torch.arange(ratio * 3).reshape(ratio, 3).long() + 2**54
    position_rows = [
        torch.arange(length * 3).reshape(length, 3).long() + 2**54 + 99 for length in lengths
    ]
    expected = {}
    for owner, sequence in enumerate(sequences):
        history = initial.clone() if owner == 0 else torch.zeros_like(initial)
        positions = initial_positions.clone() if owner == 0 else torch.zeros_like(initial_positions)
        for step, key in enumerate(sequence):
            history = torch.cat((history[1:], key[None]))
            positions = torch.cat((positions[1:], position_rows[owner][step : step + 1]))
            complete = (starts[owner] + step + 1) % ratio == 0
            pooled = (
                history.float().mean(0).to(dtype) if complete else torch.zeros(width, dtype=dtype)
            )
            first = positions[0].clone() if complete else torch.zeros(3, dtype=torch.int64)
            expected[owner, step] = history.clone(), positions.clone(), pooled, first, complete
    common = {}
    for capacity in (16, 32):
        keys = torch.zeros((capacity, width), dtype=dtype, device="cuda")
        pos = torch.zeros((capacity, 3), dtype=torch.int64, device="cuda")
        logical = torch.zeros(capacity, dtype=torch.int64, device="cuda")
        if strided:
            from cubic_serve.runtime.cache.shared_pool import SharedCacheField, SharedCacheLayout
            layout = SharedCacheLayout((SharedCacheField("history", dtype, (ratio, width)),
                                        SharedCacheField("position", torch.int64, (ratio, 3)),
                                        SharedCacheField("guard", torch.uint8, (7,))))
            raw = torch.full((24 * layout.entry_bytes,), 0x5A, device="cuda", dtype=torch.uint8)
            fields = layout.views(raw, 24)
            history, ph = fields["history"], fields["position"]
        else:
            history = torch.zeros((24, ratio, width), dtype=dtype, device="cuda")
            ph = torch.zeros((24, ratio, 3), dtype=torch.int64, device="cuda")
        source, dest, segments, active, complete, pages, offsets = [
            torch.zeros(capacity, dtype=torch.int32, device="cuda") for _ in range(7)
        ]
        pooled = torch.empty_like(keys)
        gp = torch.empty_like(pos)
        cache_backing = torch.full((24, 4 if strided else 2, width), 51, dtype=dtype, device="cuda")
        cache = cache_backing[:, :2]
        op = bind_causal_key_compression(
            keys, pos, logical, history, ph, source, dest, segments, active, pooled, gp, complete
        )
        write = bind_compressed_page_write(
            pooled, cache, pages, offsets, complete, active, compression=ratio
        )
        op.launch()
        write.launch()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op.launch()
            write.launch()
        for order in ((0, 1), (1, 0)):
            keys.fill_(float("nan"))
            pos.fill_(-111)
            logical.fill_(2**60)
            source.fill_(999)
            dest.fill_(999)
            segments.fill_(999)
            active.zero_()
            pages.fill_(999)
            offsets.fill_(999)
            complete.fill_(99)
            pooled.fill_(77)
            gp.fill_(-77)
            history.fill_(33)
            ph.fill_(-33)
            cache.fill_(88)
            history[0].copy_(initial)
            ph[0].copy_(initial_positions)
            cache_expected = cache.cpu().clone()
            mapping = []
            row = 0
            for owner in order:
                for step in range(lengths[owner]):
                    position = starts[owner] + step
                    keys[row].copy_(sequences[owner][step])
                    pos[row].copy_(position_rows[owner][step])
                    logical[row] = position
                    source[row] = 0 if owner == 0 else -1
                    dest[row] = 1 + owner * 10 + step
                    segments[row] = lengths[owner] if step == 0 else 0
                    active[row] = 1
                    page = owner * 10 + position // (2 * ratio)
                    pages[row] = page
                    offsets[row] = position % (2 * ratio)
                    if expected[owner, step][-1]:
                        cache_expected[page, (position // ratio) % 2] = expected[owner, step][2]
                    mapping.append((row, owner, step))
                    row += 1
            graph.replay()
            if strided:
                assert (fields["guard"] == 0x5A).all()
                assert (cache_backing[:, 2:] == 51).all()
            out, outpos, flags, states, positions = (
                pooled.cpu(),
                gp.cpu(),
                complete.cpu(),
                history.cpu(),
                ph.cpu(),
            )
            for physical, owner, step in mapping:
                expected_h, expected_p, expected_mean, expected_pos, flag = expected[owner, step]
                assert torch.equal(states[1 + owner * 10 + step], expected_h)
                assert torch.equal(positions[1 + owner * 10 + step], expected_p)
                assert torch.equal(out[physical], expected_mean)
                assert torch.equal(outpos[physical], expected_pos)
                assert flags[physical] == flag
                if (owner, step) in common:
                    assert torch.equal(out[physical], common[owner, step])
                common[owner, step] = out[physical].clone()
            assert torch.equal(cache.cpu(), cache_expected)
            assert torch.equal(states[0], initial)
            assert torch.equal(positions[0], initial_positions)
            assert (out[row:] == 77).all() and (outpos[row:] == -77).all()
            assert (flags[row:] == 99).all() and (states[21:] == 33).all()
    # Identical row and state results for single-token in-place execution.
    keys = torch.empty((1, width), dtype=dtype, device="cuda")
    pos = torch.empty((1, 3), dtype=torch.int64, device="cuda")
    logical = torch.zeros(1, dtype=torch.int64, device="cuda")
    history, ph = initial[None].cuda(), initial_positions[None].cuda()
    source = torch.zeros(1, dtype=torch.int32, device="cuda")
    dest = source.clone()
    segments = torch.ones_like(source)
    active = torch.ones_like(source)
    pooled, gp, complete = torch.empty_like(keys), torch.empty_like(pos), torch.empty_like(source)
    op = bind_causal_key_compression(
        keys, pos, logical, history, ph, source, dest, segments, active, pooled, gp, complete
    )
    keys.zero_()
    pos.zero_()
    op.launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.launch()
    history[0].copy_(initial)
    ph[0].copy_(initial_positions)
    for step in range(lengths[0]):
        keys[0].copy_(sequences[0][step])
        pos[0].copy_(position_rows[0][step])
        logical.fill_(starts[0] + step)
        graph.replay()
        assert torch.equal(pooled[0].cpu(), common[0, step])
        assert torch.equal(history[0].cpu(), expected[0, step][0])
        assert torch.equal(ph[0].cpu(), expected[0, step][1])


def test_compression_selection_attention_share_one_causal_graph():
    from cubic_serve.ops.cuda.attention.paged_decode import PagedGQADecodePrimitive
    from cubic_serve.ops.cuda.attention.sparse_indexer import (
        bind_causal_key_compression,
        bind_compressed_page_write,
    )
    from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

    configure_reference_math()
    torch.manual_seed(471)
    ratio, page_size, budget, width = 4, 8, 8, 128
    raw = torch.randint(-4, 5, (17, width)).to(torch.bfloat16)
    index_queries = torch.randint(-2, 3, (3, 4, width)).to(torch.bfloat16)
    values = torch.randint(-8, 9, (17, 2, 64)).to(torch.bfloat16)
    page_order = torch.tensor([2, 0, 3], dtype=torch.int32)
    expected_indices, expected_output = [], []
    with ReferenceExecutionContext():
        for step in range(3):
            visible = 15 + step
            pooled = (
                raw[: visible // ratio * ratio].reshape(-1, ratio, width).float().mean(1).bfloat16()
            )
            selected = select_blocks(
                index_queries[step : step + 1, None],
                pooled[None],
                torch.tensor([[visible]]),
                compression=ratio,
                token_budget=budget,
            )[0, 0]
            selected = selected[selected >= 0].sort().values
            expected_indices.append(selected)
            expected_output.append(
                values[selected.long()].float().mean(0).repeat_interleave(2, 0).bfloat16()
            )
    common = {}
    with operator_strategy_scope(
        OperatorConfig(attention_dtype="fp32", softmax_dtype="fp32"), device="", model=""
    ):
        for capacity in (1, 8):
            keys = torch.zeros((capacity, width), dtype=torch.bfloat16, device="cuda")
            positions = torch.zeros((capacity, 3), dtype=torch.int64, device="cuda")
            logical = torch.zeros(capacity, dtype=torch.int64, device="cuda")
            history = torch.zeros((4, ratio, width), dtype=torch.bfloat16, device="cuda")
            position_history = torch.zeros((4, ratio, 3), dtype=torch.int64, device="cuda")
            source, destination, segments, active, complete, pages, offsets, lengths = [
                torch.zeros(capacity, dtype=torch.int32, device="cuda") for _ in range(8)
            ]
            pooled, group_positions = torch.empty_like(keys), torch.empty_like(positions)
            compressed = torch.zeros(
                (5, page_size // ratio, width), dtype=torch.bfloat16, device="cuda"
            )
            iq = torch.zeros((capacity, 4, width), dtype=torch.bfloat16, device="cuda")
            table = torch.full((capacity, 3), 4, dtype=torch.int32, device="cuda")
            selection = torch.zeros(
                (capacity, budget + ratio - 1), dtype=torch.int32, device="cuda"
            )
            counts = torch.ones(capacity, dtype=torch.int32, device="cuda")
            query = torch.zeros((capacity, 4, 64), dtype=torch.bfloat16, device="cuda")
            main_key = torch.zeros((5, 1, page_size, 2, 64), dtype=torch.bfloat16, device="cuda")
            main_value = torch.zeros_like(main_key)
            for token in range(17):
                main_value[int(page_order[token // page_size]), 0, token % page_size].copy_(
                    values[token]
                )
            result = torch.empty_like(query)
            compress = bind_causal_key_compression(
                keys,
                positions,
                logical,
                history,
                position_history,
                source,
                destination,
                segments,
                active,
                pooled,
                group_positions,
                complete,
            )
            write = bind_compressed_page_write(
                pooled, compressed, pages, offsets, complete, active, compression=ratio
            )
            choose = PagedBlockSelection(
                iq,
                compressed,
                table,
                lengths,
                active,
                selection,
                counts,
                compression=ratio,
                token_budget=budget,
            )
            attend = PagedGQADecodePrimitive(
                query,
                main_key,
                main_value,
                table,
                counts,
                result,
                layer=0,
                selected_indices=selection,
            )

            def launch(
                ops=(compress, write, choose),
                attention=attend,
                operands=(query, main_key, main_value, table, counts, result),
            ):
                for op in ops:
                    op.launch()
                attention.launch(*operands)

            launch()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch()
            history[0].copy_(raw[10:14])
            position_history[0].copy_(torch.arange(10, 14)[:, None].expand(-1, 3))
            for group in range(3):
                token = group * ratio
                compressed[int(page_order[token // page_size]), (token % page_size) // ratio].copy_(
                    raw[token : token + ratio].float().mean(0).bfloat16()
                )
            mappings = [(0,), (1,), (2,)] if capacity == 1 else [(0, 1, 2)]
            for mapping in mappings:
                active.zero_()
                segments.zero_()
                for physical, step in enumerate(mapping):
                    position = 14 + step
                    keys[physical].copy_(raw[position])
                    positions[physical].fill_(position)
                    logical[physical] = position
                    source[physical] = 0
                    destination[physical] = 0 if capacity == 1 else physical + 1
                    segments[physical] = len(mapping) if physical == 0 else 0
                    active[physical] = 1
                    pages[physical] = page_order[position // page_size]
                    offsets[physical] = position % page_size
                    lengths[physical] = position + 1
                    iq[physical].copy_(index_queries[step])
                    table[physical].copy_(page_order)
                graph.replay()
                output, ids, n = result.cpu(), selection.cpu(), counts.cpu()
                for physical, step in enumerate(mapping):
                    assert torch.equal(ids[physical, : n[physical]], expected_indices[step])
                    assert torch.equal(output[physical], expected_output[step])
                    if step in common:
                        assert torch.equal(output[physical], common[step])
                    common[step] = output[physical].clone()
            assert torch.equal(compressed[0, 1].cpu(), raw[12:16].float().mean(0).bfloat16())
            final_slot = 0 if capacity == 1 else 3
            assert torch.equal(history[final_slot].cpu(), raw[13:17])
            if capacity != 1:
                assert torch.equal(history[0].cpu(), raw[10:14])


def _distributed_selector_worker(rank, parts, rendezvous):
    import torch.distributed as dist
    from datetime import timedelta

    configure_reference_math()
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=parts,
                            timeout=timedelta(seconds=90), device_id=torch.device("cuda", rank))
    for rows, ratio, budget, groups_per_page, pages in (
        (1, 4, 4, 8, 1),  # Ranking with empty block owners.
        (4, 2, 10, 8, 1),  # Local sorted runs smaller than the global keep width.
        (1, 4, 12, 8, 8), (4, 2, 8, 8, 16), (128, 4, 16, 8, 4),
        # At a large row capacity, rank-owned keys span repeated score tiles.
        (512, 4, 16, 8, 64),
    ):
        heads, dim = 3, 32
        torch.manual_seed(73)
        full = torch.randint(-2, 3, (rows * pages, groups_per_page, dim), device="cuda").to(torch.bfloat16)
        cache = full[:, rank::parts].contiguous()
        q = torch.randint(-2, 3, (rows, heads, dim), device="cuda").to(torch.bfloat16)
        table = torch.arange(rows * pages, device="cuda", dtype=torch.int32).reshape(rows, pages)
        lengths = torch.ones(rows, device="cuda", dtype=torch.int32)
        active = torch.ones_like(lengths)
        output = torch.empty(rows, budget + ratio - 1, device="cuda", dtype=torch.int32)
        counts = torch.empty_like(lengths)
        control_out = torch.empty_like(output)
        control_counts = torch.empty_like(counts)
        pool = {}

        def workspace(role, shape, dtype, pool=pool):
            key = role, shape, dtype
            if key not in pool:
                pool[key] = torch.empty(shape, device="cuda", dtype=dtype)
            return pool[key]

        op = PagedBlockSelection(q, cache, table, lengths, active, output, counts,
                                compression=ratio, token_budget=budget, parts=parts, rank=rank,
                                group=dist.group.WORLD, workspace=workspace)
        peer = PagedBlockSelection(
            q, cache, table, lengths, active, torch.empty_like(output), torch.empty_like(counts),
            compression=ratio, token_budget=budget, parts=parts, rank=rank,
            group=dist.group.WORLD, workspace=workspace,
        )
        assert op.scratch[0].data_ptr() == peer.scratch[0].data_ptr()
        control = PagedBlockSelection(q, full, table, lengths, active, control_out, control_counts,
                                     compression=ratio, token_budget=budget)
        op.launch()
        peer.launch()
        control.launch()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op.launch()
            peer.launch()
        for generation, length in enumerate((1, budget, budget + ratio, ratio * groups_per_page * pages - 1,
                                             2, budget + 1, budget + ratio * 3, budget - 1)):
            lengths.fill_(length)
            active.fill_(1)
            if rows > 1:
                active[-1] = 0
                lengths[-1] = 2**30
            # Ties and changing rank progress must preserve original global IDs.
            if generation % 2:
                q.zero_()
            else:
                q.fill_(1)
            output.fill_(-77)
            control_out.fill_(-77)
            counts.fill_(-33)
            control_counts.fill_(-33)
            peer.output.fill_(-77)
            peer.counts.fill_(-33)
            if rank == generation % parts:
                torch.cuda._sleep(100000)
            graph.replay()
            control.launch()
            assert torch.equal(counts, control_counts)
            assert torch.equal(output, control_out)
            assert torch.equal(peer.counts, control_counts)
            assert torch.equal(peer.output, control_out)
            # Independent eager selector uses full original logical key order.
            with ReferenceExecutionContext():
                expected = select_blocks(q[:1, None].cpu(), full[:pages].flatten(0, 1)[None].cpu(),
                                         lengths[:1, None].long().cpu(), compression=ratio, token_budget=budget)
            n = int(counts[0])
            assert torch.equal(output[0, :n].cpu(), expected[0, 0, :n].sort().values)
        graph.reset()
        dist.barrier()
        op._candidate_exchange.arena.close()
        op._decision_exchange.arena.close()
        peer._candidate_exchange.arena.close()
        peer._decision_exchange.arena.close()
    dist.barrier()
    os._exit(0)


@pytest.mark.parametrize("parts", (2, 4))
def test_distributed_blocks_global_selection_and_device_guard(parts):
    import tempfile
    import torch.multiprocessing as mp
    if torch.cuda.device_count() < parts:
        pytest.skip("insufficient admitted devices")
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_distributed_selector_worker, args=(parts, f"file://{directory}/init"), nprocs=parts, join=True)


def _distributed_preparation_worker(rank, parts, rendezvous, wide=False):
    import torch.distributed as dist
    from datetime import timedelta
    from cubic_serve.models.parallel_weights import IndexerTP
    from cubic_serve.ops.cuda.attention.sparse_indexer import IndexerPreparation
    from cubic_serve.ops.cuda.normalization.rms_norm import ZeroCenteredRMSNormPrimitive
    from cubic_serve.ops.cuda.position.qwen_mrope import QwenMRoPEApplyPrimitive

    configure_reference_math()
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=parts,
                            timeout=timedelta(seconds=90), device_id=torch.device("cuda", rank))
    for dtype in ((torch.bfloat16,) if wide else (torch.float16,torch.bfloat16)):
        for rows, heads, width, ratio in (((1,1,80,3),(4,3,128,4)) if wide else ((1, 1, 80, 3), (4, 3, 128, 4), (4, 7, 128, 4), (128, 4, 128, 4))):
            plan = IndexerTP(heads, width, ratio, parts, rank)
            torch.manual_seed(63)
            page_tokens = ratio * parts * 2
            page_count = 2
            qw = (torch.randint(-3, 4, (width,), device="cuda") / 4).to(dtype)
            kw = qw.clone()
            projected = torch.empty(rows, plan.local_query_heads * width + plan.key_width, device="cuda", dtype=dtype)
            normalized_q = torch.empty(rows, heads, width, device="cuda", dtype=dtype)
            full_q = torch.empty_like(normalized_q)
            expected_q = torch.empty_like(normalized_q)
            pooled = torch.empty(rows, width, device="cuda", dtype=dtype)
            norm_pool = torch.empty_like(pooled)
            rotated_pool = torch.empty(rows, 1, width, device="cuda", dtype=dtype)
            angles = torch.randn(rows, 64, device="cuda")
            cos, sin = angles.cos().to(dtype), angles.sin().to(dtype)
            positions = torch.zeros(3, rows, device="cuda", dtype=torch.int64)
            logical = torch.zeros(rows, device="cuda", dtype=torch.int64)
            base=1025 if wide else 0
            sources = torch.arange(rows, device="cuda", dtype=torch.int32)+base
            destinations = sources.clone()
            segments = torch.ones_like(sources)
            active = torch.ones_like(sources)
            offsets, pages = torch.zeros_like(sources), torch.zeros_like(sources)
            history = torch.zeros(rows, ratio, plan.key_width, device="cuda", dtype=dtype)
            p_history = torch.zeros(rows, ratio, 3, device="cuda", dtype=torch.int64)
            reference_history = torch.zeros(rows, ratio, width, device="cuda", dtype=dtype)
            cache = torch.zeros(rows * page_count, 2, width, device="cuda", dtype=dtype)
            root_history,root_positions,root_cache=history,p_history,cache
            if wide:
                from cubic_serve.runtime.cache.shared_pool import SharedCacheField,SharedCacheLayout
                from cubic_serve.runtime.cache.virtual_memory import VirtualAllocation
                entries=(SharedCacheField('history',dtype,tuple(history.shape[1:])),
                         SharedCacheField('positions',torch.int64,tuple(p_history.shape[1:])),
                         SharedCacheField('cache',dtype,tuple(cache.shape[1:])))
                stride=32<<20
                plain=SharedCacheLayout(entries)
                packed_layout=SharedCacheLayout((*entries,SharedCacheField('gap',torch.uint8,(stride-plain.entry_bytes,))))
                count=rows*page_count
                virtual=VirtualAllocation((base+count)*stride,torch.device('cuda',rank))
                for index in (*range(count+1),*range(base,base+count)):
                    virtual.ensure_range(index*stride,plain.entry_bytes)
                raw=virtual.tensor(((base+count)*stride,),torch.uint8)
                fields=packed_layout.views(raw,base+count)
                virtual.seal()
                assert virtual.backed_bytes<=(2*count+1)*virtual.granularity
                root_history,root_positions,root_cache=(fields[name] for name in ('history','positions','cache'))
                history=root_history[base:base+rows]
                p_history=root_positions[base:base+rows]
                cache=root_cache[base:base+count]
            query = torch.empty_like(normalized_q)
            group_pos = torch.empty_like(positions)
            op = IndexerPreparation(projected, qw, kw, cos, sin, positions, logical, root_history,
                root_positions, sources, destinations, segments, active, offsets, pages, root_cache,
                query, group_pos, cos, sin, plan=plan, group=dist.group.WORLD, epsilon=1e-6)
            qnorm = ZeroCenteredRMSNormPrimitive(full_q.flatten(0, 1), qw, normalized_q.flatten(0, 1), epsilon=1e-6)
            knorm = ZeroCenteredRMSNormPrimitive(pooled, kw, norm_pool, epsilon=1e-6)
            rope = QwenMRoPEApplyPrimitive(normalized_q, norm_pool[:, None], cos, sin, expected_q, rotated_pool)
            projected.zero_()
            op.prepare(); op.finish()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                op.prepare(); op.finish()
            history.zero_(); p_history.zero_(); cache.zero_()
            for step in range(ratio * parts + 1):
                full = (torch.arange(rows * (heads + 1) * width, device="cuda").reshape(rows, -1)
                        + step * 3).remainder(17).sub(8).to(dtype)
                projected.copy_(torch.cat([full[:, lo:hi] for lo, hi in plan.projection_ranges], dim=1))
                full_q.copy_(full[:, :heads * width].reshape_as(full_q))
                logical.fill_(step)
                positions.fill_(step)
                offsets.fill_(step % page_tokens)
                pages.copy_(torch.arange(rows, device="cuda", dtype=torch.int32)*page_count+step//page_tokens+base)
                active.fill_(1)
                logical_rows = rows if rows == 1 else rows - 1
                if rows > 1:
                    active[-1] = 0
                    projected[-1].fill_(float("nan"))
                sources.copy_(torch.arange(rows, device="cuda", dtype=torch.int32)+base)
                if step == 0:
                    sources.fill_(-1)
                reference_history[:logical_rows] = torch.cat((reference_history[:logical_rows, 1:],
                                                              full[:logical_rows, None, heads * width:]), 1)
                query.fill_(37)
                graph.replay()
                qnorm.launch(full_q.flatten(0, 1), qw, normalized_q.flatten(0, 1))
                pooled.copy_(reference_history.float().mean(1).to(dtype))
                knorm.launch(pooled, kw, norm_pool)
                rope.launch(normalized_q, norm_pool[:, None], cos, sin, expected_q, rotated_pool)
                assert torch.equal(query[:logical_rows], expected_q[:logical_rows])
                lo = rank * plan.key_width
                assert torch.equal(history, reference_history[..., lo:lo + plan.key_width])
                if (step + 1) % ratio == 0:
                    assert torch.equal(group_pos[:, :logical_rows], torch.full_like(group_pos[:, :logical_rows], step + 1 - ratio))
                    if (step // ratio) % parts == rank:
                        ids = pages[:logical_rows].long()
                        local = (step % page_tokens) // (ratio * parts)
                        assert torch.equal(cache[ids-base, local], rotated_pool[:logical_rows, 0])
                if rows > 1:
                    assert (query[-1] == 37).all()
                    assert not cache[(rows - 1) * page_count:].any()
            # One prefill segment: zero segment lengths are continuation rows,
            # not padding. Each token's durable checkpoint slot holds its prefix.
            if rows > 1:
                history.zero_(); p_history.zero_(); cache.zero_()
                sources.fill_(-1)
                segments.zero_(); segments[0] = logical_rows
                logical.copy_(torch.arange(rows, device="cuda", dtype=torch.int64))
                positions.copy_(logical[None].expand(3, -1))
                offsets.copy_(logical.to(torch.int32).remainder(page_tokens))
                pages.copy_(logical.to(torch.int32)//page_tokens+base)
                # The larger prefill needs its own pages, independent of decode slots.
                graph.replay()
                ring = torch.zeros_like(reference_history[0])
                for row in range(logical_rows):
                    ring = torch.cat((ring[1:], full[row:row + 1, heads * width:]), dim=0)
                    assert torch.equal(history[row], ring[:, lo:lo + plan.key_width])
                    pooled[row].copy_(ring.float().mean(0).to(dtype))
                    if (row + 1) % ratio == 0:
                        assert (group_pos[:, row] == row + 1 - ratio).all()
                assert torch.equal(query[:logical_rows], expected_q[:logical_rows])
                assert not history[-1].any()
                knorm.launch(pooled, kw, norm_pool)
                rope.launch(normalized_q, norm_pool[:, None], cos, sin, expected_q, rotated_pool)
                for row in range(ratio - 1, logical_rows, ratio):
                    if (row // ratio) % parts == rank:
                        assert torch.equal(cache[row // page_tokens, (row % page_tokens) // (ratio * parts)],
                                           rotated_pool[row, 0])
            if wide:
                assert not root_history[:rows*page_count+1].any()
                assert not root_positions[:rows*page_count+1].any()
                assert not root_cache[:rows*page_count+1].any()
            graph.reset()
            dist.barrier()
            op.exchange.arena.close()
    dist.barrier()
    os._exit(0)


@pytest.mark.parametrize("parts", (2, 4))
def test_distributed_fused_preparation_rounding_and_unique_cache(parts):
    import tempfile
    import torch.multiprocessing as mp
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_distributed_preparation_worker, args=(parts, f"file://{directory}/init"), nprocs=parts, join=True)


@pytest.mark.skipif(os.environ.get('CUBIC_TEST_QUANT_CUDA')!='1' or torch.cuda.device_count()<2,
                    reason='explicit TP2 wide shared-address proof')
def test_distributed_fused_preparation_with_wide_shared_state_and_page_strides():
    import tempfile
    import torch.multiprocessing as mp
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_distributed_preparation_worker,args=(2,f'file://{directory}/init',True),nprocs=2,join=True)
