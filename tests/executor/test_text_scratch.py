"""Scratch ownership and cross-bucket replay; not serving throughput evidence."""

import os

import pytest
import torch

from cubic_serve.executor.text_graph import (
    GraphScratchPool, TextGraphBindings, full_attention_partitions,
)


@pytest.mark.parametrize("rows", [1, 1024])
@pytest.mark.parametrize("encoded", [False, True])
def test_typed_capture_padding_respects_physical_pages_and_protects_the_owner_pool(rows, encoded):
    from types import SimpleNamespace
    from cubic_serve.executor.text_serving import FlashNextDecodeStage
    from cubic_serve.runtime.cache import CacheArena, CacheArenaSpec, KVLayout, StateSlotSpec
    from cubic_serve.runtime.cache.quantization import (
        CacheDType, CacheGroupAxis, CacheStorageSpec, KVCacheStorageSpec,
    )
    from cubic_serve.runtime.cache.shared_pool import SharedCacheStorage

    storage = (KVCacheStorageSpec(
        CacheStorageSpec(CacheDType.INT4, CacheGroupAxis.TOKENS, 32),
        CacheStorageSpec(CacheDType.INT4, CacheGroupAxis.CHANNELS, 32),
    ) if encoded else KVCacheStorageSpec())
    layout = KVLayout((0,), 32, 1, 32, storage=storage)
    schema = CacheArenaSpec("typed", 1, 1,
        tuple(sorted((StateSlotSpec("state", torch.bfloat16, (4,)),
                      *layout.quantization_state_specs), key=lambda spec: spec.state_id)),
        scratch_capacity=rows, scratch_arena_ids=("a", "b"))
    shared = SharedCacheStorage(((schema, layout),), budget_bytes=1 << 18, device="cpu")
    arena = CacheArena(shared.arena_specs[0], layout, "cpu", rank_id=0, shared_storage=shared)
    shared.backing.fill_(79)
    protected = shared.backing[:shared.budget_bytes].clone()
    for ordinal, lane in enumerate(("a", "b")):
        stage = FlashNextDecodeStage.__new__(FlashNextDecodeStage)
        stage.arena, stage.captured_m = arena, rows
        stage.weights = SimpleNamespace(device=torch.device("cpu"))
        stage.resource = object()
        stage.embedding_mask = stage.feature_rows = None
        stage.buffers = SimpleNamespace(
            segment_lengths=torch.ones(rows, dtype=torch.int32),
            row_activity=torch.ones(rows, dtype=torch.bool),
            position_ids=torch.ones(3, rows, dtype=torch.int64),
            token_offsets=torch.ones(rows, dtype=torch.int32),
            sequence_lengths=torch.ones(rows, dtype=torch.int32),
            page_source=torch.empty(rows, dtype=torch.int32),
            page_destination=torch.empty(rows, dtype=torch.int32),
            page_table=torch.empty(rows, 2, dtype=torch.int32),
            state_slot_maps=(torch.empty(2, 1, rows, dtype=torch.int32),),
        )
        stage._prepare_padding(lane)
        stage.prepare_scratch()
        start = shared.page_scratch_start + ordinal * (1 if encoded else rows)
        expected = ([start] * rows if encoded else list(range(start, start + rows)))
        assert stage.buffers.page_source.tolist() == expected
        assert stage.buffers.page_destination.tolist() == expected
        assert stage.buffers.page_table.tolist() == [[page, page] for page in expected]
        assert stage._scratch_page_reset.tolist() == sorted(set(expected))
        assert (stage.buffers.state_slot_maps[0] == arena.spec.state_slot_capacity + ordinal).all()
        assert max(expected) < arena.key.shape[0]
        assert not stage.buffers.row_activity.any()
        assert not arena.key[stage._scratch_page_reset].any()
        assert torch.equal(shared.backing[:shared.budget_bytes], protected)


@pytest.fixture
def bindings(monkeypatch):
    empty = torch.empty

    def allocate(*args, **kwargs):
        return empty(*args, **dict(kwargs, device="cpu"))

    monkeypatch.setattr(torch, "empty", allocate)
    return lambda pool: TextGraphBindings(
        None, torch.device("cuda:4"), torch.bfloat16, shared_scratch=pool,
    )


def test_smaller_buckets_share_backing_and_keep_their_contiguous_abi(bindings):
    pool = {}
    large, small = bindings(pool), bindings(pool)
    a = large.buffer("activation", 64, 128)
    b = small.buffer("activation", 7, 128)
    assert a.data_ptr() == b.data_ptr()
    assert b.shape == (7, 128) and b.is_contiguous() and b.stride() == (128, 1)
    a.fill_(11)
    b.fill_(3)
    assert torch.all(a[:7] == 3) and torch.all(a[7:] == 11)


def test_distinct_live_shapes_within_a_bucket_remain_disjoint(bindings):
    owner = bindings({})
    a = owner.buffer("activation", 64, 128)
    b = owner.buffer("activation", 7, 128)
    assert a.data_ptr() != b.data_ptr()
    assert owner.buffer("activation", 64, 128) is a
    a.fill_(11)
    b.fill_(3)
    assert torch.all(a == 11)


def test_growth_preserves_permanent_m1_and_role_geometry_and_dtype_ownership(bindings):
    pool = {}
    one, large, small = bindings(pool), bindings(pool), bindings(pool)
    c1 = one.buffer("activation", 1, 128)
    c1.fill_(19)
    a = large.buffer("activation", 64, 128)
    a.fill_(3)
    assert c1.data_ptr() != a.data_ptr() and torch.all(c1 == 19)
    assert small.buffer("activation", 7, 128).data_ptr() == a.data_ptr()
    for value in (small.buffer("other", 7, 128),
                  small.buffer("activation", 7, 64),
                  small.buffer("activation", 7, 128, dtype=torch.float32)):
        assert value.data_ptr() != a.data_ptr()


def test_target_and_provider_pools_remain_disjoint(bindings):
    a = bindings({}).buffer("activation", 64, 128)
    b = bindings({}).buffer("activation", 64, 128)
    assert a.data_ptr() != b.data_ptr()


def test_serial_attention_storage_preserves_consumed_outputs_and_private_owners(bindings):
    target_pool = GraphScratchPool()
    provider_pool = GraphScratchPool(target_pool.serial)
    target, provider = bindings(target_pool), bindings(provider_pool)
    partial = target.buffer("attention.partial", 64, 6, 32, 8,
                            dtype=torch.float32, serial=True)
    proposal = provider.buffer("attention.partial", 4, 6, 32, 8,
                               dtype=torch.float32, serial=True)
    assert partial.data_ptr() == proposal.data_ptr()
    target_output = target.buffer("feedback", 64, 48)
    provider_output = provider.buffer("feedback", 64, 48)
    assert target_output.data_ptr() != provider_output.data_ptr()
    partial.fill_(3)
    target_output.copy_(partial.sum(dim=2).flatten(1))
    expected = target_output.clone()
    proposal.fill_(7)
    provider_output.fill_(11)
    assert torch.equal(target_output, expected)
    assert torch.all(partial[4:] == 3)
    with pytest.raises(ValueError, match="serial graph scratch"):
        provider.buffer("attention.partial", 4, 6, 32, 8, private=True, serial=True)


def test_serial_scratch_keeps_dtype_geometry_and_simultaneous_shapes_disjoint(bindings):
    pool = GraphScratchPool()
    target, provider = bindings(pool), bindings(GraphScratchPool(pool.serial))
    a = target.buffer("attention.partial", 64, 6, 32, 8,
                      dtype=torch.float32, serial=True)
    for b in (target.buffer("attention.partial", 4, 6, 32, 8,
                            dtype=torch.float32, serial=True),
              provider.buffer("attention.partial", 4, 6, 32, 8, serial=True)):
        assert a.data_ptr() != b.data_ptr()
    different_geometry = provider.buffer("attention.partial", 4, 6, 16, 8,
                                         dtype=torch.float32, serial=True)
    assert different_geometry.data_ptr() == a.data_ptr()
    assert different_geometry.shape == (4, 6, 16, 8) and different_geometry.is_contiguous()


@pytest.mark.parametrize("sms,heads,owners", [(16, 4, 1), (80, 2, 2), (156, 1, 2),
                                           (156, 4, 1), (224, 4, 2)])
def test_serial_partition_portfolio_fits_original_private_storage(
        bindings, monkeypatch, sms, heads, owners):
    from types import SimpleNamespace
    from cubic_serve.ops.cuda.attention.paged_decode import paged_gqa_kv_splits
    from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda _: SimpleNamespace(multi_processor_count=sms))
    capacities = tuple(2**i for i in range(11))
    pools = [GraphScratchPool(capacities=capacities)]
    pools += [pools[0].sibling() for _ in range(owners - 1)]
    buffers = []
    with operator_strategy_scope(OperatorConfig(attention_dtype="auto"), device="", model=""):
        old = {}
        for m in capacities:
            s = paged_gqa_kv_splits(m, max_tokens=262144, head_programs=heads,
                                   multiprocessors=sms, full_context=True)
            old[s] = max(old.get(s, 0), m * s)
        for m in (1, *reversed(capacities[1:])):
            for pool in pools:
                s = full_attention_partitions(m, max_tokens=262144, query_heads=heads * 6,
                                              kv_heads=heads, head_dim=8,
                                              device=torch.device("cuda:4"), shared_scratch=pool)
                assert 1 <= s <= 32 and s & (s - 1) == 0
                assert m * s <= owners * sum(old.values())
                helper = bindings(pool)
                for role, width in (("attention.partial", 8), ("attention.statistics", 2)):
                    buffers.append(helper.buffer(role, m, heads * 6, s, width,
                                                  dtype=torch.float32, serial=True))
        stores = {b.untyped_storage().data_ptr(): b.untyped_storage().nbytes() for b in buffers}
        # The permanent M1 backing predates the largest-first shared allocation,
        # just as it did in the original per-family portfolio.
        original_bytes = owners * (sum(old.values()) + 32) * heads * 6 * (8 + 2) * 4
        assert sum(stores.values()) <= original_bytes


@pytest.mark.skipif(os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1",
                    reason="explicit terminal-output lease proof")
def test_terminal_logits_are_consumed_before_the_next_bucket_and_keep_feedback_private():
    device = torch.device('cuda:0')
    provider_pool, target_pool = {}, {}

    def owner(pool):
        return TextGraphBindings(None, device, torch.bfloat16, shared_scratch=pool)

    permanent = owner(provider_pool).buffer('mtp.sample_logits', 1, 8)
    largest = owner(provider_pool).buffer('mtp.sample_logits', 16, 8)
    target = owner(target_pool).buffer('decode.sample_logits', 16, 8)
    assert permanent.data_ptr() != largest.data_ptr() != target.data_ptr()
    stages = []
    for rows in (4, 8):
        head, alignment = owner(provider_pool), owner(provider_pool)
        logits = head.buffer('mtp.sample_logits', rows, 8)
        unused_logits = alignment.buffer('mtp.sample_logits', 2 * rows, 8)
        assert logits.data_ptr() == unused_logits.data_ptr() == largest.data_ptr()
        feedback = head.buffer('feedback', rows, 8, private=True)
        aligned = alignment.buffer('feedback', 2 * rows, 8, private=True)
        assert feedback.data_ptr() != aligned.data_ptr()
        inputs, durable = torch.zeros_like(feedback), torch.zeros_like(feedback)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            feedback.copy_(inputs * 2 + 1)
            logits.copy_(feedback)
            target[:rows].copy_(inputs * 3 + 2)
            # The alignment phase exports feedback and deliberately has no
            # projection. Its output must not clobber carried head feedback.
            aligned.fill_(-173.)
            durable.copy_(logits + target[:rows])
        stages.append((rows, inputs, durable, feedback, graph))
    for iteration, index in enumerate((0, 1, 1, 0, 1, 0)):
        rows, inputs, durable, feedback, graph = stages[index]
        values = (torch.arange(rows * 8, device=device).reshape(rows, 8) % 11).bfloat16()
        values.add_(iteration)
        inputs.copy_(values)
        largest.fill_(91.)
        target.fill_(47.)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(durable, values * 5 + 3)
        assert torch.equal(feedback, values * 2 + 1)
        assert (largest[rows:] == 91.).all()
    for _, _, _, _, graph in stages:
        graph.reset()


def test_feedback_boundaries_remain_private_across_composed_provider_buckets(bindings):
    pool = {}
    alignment, proposal = bindings(pool), bindings(pool)
    a = alignment.buffer("boundary", 64, 128, private=True)
    b = proposal.buffer("boundary", 16, 128, private=True)
    assert a.data_ptr() != b.data_ptr()
    a.fill_(11)
    b.fill_(3)
    assert torch.all(a == 11)


@pytest.mark.skipif(os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1",
                    reason="explicit CUDA ownership audit required")
def test_shared_scratch_graphs_match_independent_cpu_math_across_m_and_poison():
    device = torch.device("cuda:0")
    pool, cases = {}, []
    stream = torch.cuda.Stream(device=device)
    for physical in (1, 128, 64, 32, 16, 8, 4, 2):
        helper = TextGraphBindings(None, device, torch.bfloat16, shared_scratch=pool)
        scratch = helper.buffer("activation", physical, 64)
        x = torch.zeros_like(scratch)
        output = torch.full((physical + 2, 64), -99, dtype=x.dtype, device=device)

        def launch():
            torch.mul(x, 2, out=scratch)
            torch.add(scratch, 3, out=output[1:-1])

        stream.wait_stream(torch.cuda.current_stream(device))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            launch()
        stream.synchronize()
        cases.append((physical, scratch, x, output, graph))
    assert cases[1][1].data_ptr() == cases[-1][1].data_ptr()
    assert cases[0][1].data_ptr() != cases[1][1].data_ptr()
    try:
        for physical, scratch, x, output, graph in cases + list(reversed(cases)):
            for logical in sorted({1, max(1, physical - 1), physical}):
                host = torch.full((physical, 64), float("nan"), dtype=torch.bfloat16)
                host[:logical] = torch.arange(logical * 64).reshape(logical, 64) % 17
                host[:logical] = host[:logical].flip(0)
                scratch.fill_(float("nan"))
                x.copy_(host)
                output.fill_(-99)
                stream.wait_stream(torch.cuda.current_stream(device))
                graph.replay()
                stream.synchronize()
                actual = output.cpu()
                expected = (host[:logical] * 2 + 3).bfloat16()
                assert torch.equal(actual[1:logical + 1], expected)
                assert torch.all(actual[0] == -99) and torch.all(actual[-1] == -99)
                assert torch.isnan(actual[logical + 1:-1]).all()
    finally:
        for *_, graph in cases:
            graph.reset()


@pytest.mark.skipif(os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1",
                    reason="explicit CUDA ownership audit required")
def test_provider_alignment_selection_keeps_exported_sources_disjoint():
    from cubic_serve.executor.draft_graph import select_head

    device = torch.device("cuda:0")
    pool, owners, depth, width = {}, 32, 3, 64
    physical = owners * (depth + 1)
    alignment = TextGraphBindings(None, device, torch.bfloat16, shared_scratch=pool)
    proposal = TextGraphBindings(None, device, torch.bfloat16, shared_scratch=pool)
    scratch = alignment.buffer("intermediate", physical, width)
    head_scratch = proposal.buffer("intermediate", owners, width)
    norm = alignment.buffer("exported", physical, width, private=True)
    head_norm = proposal.buffer("exported", owners, width, private=True)
    assert scratch.data_ptr() == head_scratch.data_ptr()
    assert norm.data_ptr() != head_norm.data_ptr()
    raw, head_raw = torch.empty_like(norm), torch.empty_like(head_norm)
    ctx = torch.tensor([owners], dtype=torch.int32, device=device)
    decisions = torch.zeros((owners, 2), dtype=torch.int64, device=device)
    ordinal = torch.arange(owners, dtype=torch.int64, device=device)
    next_ordinal = torch.empty_like(ordinal)
    valid = torch.empty(owners, dtype=torch.bool, device=device)
    host = (torch.arange(physical * width).reshape(physical, width) % 29).bfloat16()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        select_head[(owners, 1)](ctx, decisions, raw, norm, head_raw, head_norm,
                                ordinal, next_ordinal, valid, width, width, 1, 256, depth)
    stream.synchronize()
    try:
        for iteration in range(8):
            accepted = (torch.arange(owners) + iteration) % (depth + 1)
            decisions[:, 0].copy_(accepted)
            norm.copy_(host)
            raw.copy_(host + 1)
            scratch.fill_(float("nan"))
            head_scratch.fill_(float("nan"))
            stream.wait_stream(torch.cuda.current_stream(device))
            graph.replay()
            stream.synchronize()
            rows = torch.arange(owners) * (depth + 1) + accepted
            assert torch.equal(head_norm.cpu(), host[rows])
            assert torch.equal(head_raw.cpu(), (host + 1)[rows])
            assert torch.equal(next_ordinal.cpu(), torch.arange(owners) + accepted + 1)
            assert valid.all().item()
            assert torch.equal(norm.cpu(), host)
    finally:
        graph.reset()


@pytest.mark.skipif(os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1",
                    reason="explicit nested capture-pool lifetime proof")
@pytest.mark.parametrize("physical", [1, 8, 32])
def test_mtp_variant_pools_preserve_nested_live_temporaries_and_exported_outputs(
    monkeypatch, physical,
):
    from types import SimpleNamespace
    from cubic_serve.executor.draft_graph import CapturedDraftRound
    from cubic_serve.executor.resident_window import append_captured_graph

    monkeypatch.setattr("cubic_serve.executor.draft_graph.barrier", lambda: None)
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    x = torch.zeros((physical, 64), device=device)
    model_output, review_output = torch.zeros_like(x), torch.zeros_like(x)
    protected = torch.full((physical + 2, 64), -999., device=device)
    round_ = CapturedDraftRound.__new__(CapturedDraftRound)
    round_._graph_pool = torch.cuda.graph_pool_handle()
    round_._review_graph_pool = torch.cuda.graph_pool_handle()
    round_._review_graphs, round_._review_capture_open = {}, True
    round_._retain_audit_graph = True
    round_._capture_review_only = round_._capture_target_constraints = False
    round_._capture_candidates = round_._capture_budget = False
    round_.ctx = x
    round_.target = SimpleNamespace(slot=SimpleNamespace(stream=stream),
                                    _sampling=SimpleNamespace(constraint_bound=False))
    round_.head = SimpleNamespace(_sampling=SimpleNamespace(constraint_bound=False))
    stream.wait_stream(torch.cuda.current_stream(device))
    model = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(model, stream=stream):
        model_output.copy_(x * 7 + 2)
    model.instantiate()

    def review():
        # This temporary must survive the nested model's private allocations.
        temporary = x * 5
        append_captured_graph(model)
        review_output.copy_(temporary + model_output)

    def parent():
        # This one must also survive review and model allocations.
        temporary = x * 3
        if torch.cuda.is_current_stream_capturing():
            round_.append_review_graph()
        else:
            round_.captured_review_graph().replay()
        protected[1:-1].copy_(temporary + review_output)

    round_.finish_review = review
    round_.constraint_continuation_forward_body = parent
    parents = []
    try:
        for budget, candidates in ((False, False), (False, True), (True, True), (True, False)):
            round_._capture_budget, round_._capture_candidates = budget, candidates
            round_.capture_review_graph(stream)
            round_.capture_constraint_continuation()
            parents.append(round_.constraint_continuation_forward_graph)
        assert all(graph.pool() == round_._graph_pool for graph in parents)
        assert all(graph.pool() == round_._review_graph_pool
                   for graph in round_._review_graphs.values())
        assert parents[0].pool() != next(iter(round_._review_graphs.values())).pool()
        assert model.pool() not in (parents[0].pool(),
                                   next(iter(round_._review_graphs.values())).pool())
        for iteration, index in enumerate((3, 0, 2, 1, 0, 3, 1, 2)):
            logical = max(1, physical - iteration % physical)
            values = (torch.arange(physical * 64).reshape(physical, 64) % 19).float()
            values = values.flip(0) + iteration
            values[logical:] = -31 - iteration
            x.copy_(values)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                parents[index].replay()
            stream.synchronize()
            assert torch.equal(protected[1:-1].cpu(), values * 15 + 2)
            assert torch.all(protected[[0, -1]] == -999)
            assert torch.equal(x.cpu(), values)
    finally:
        for graph in parents:
            graph.reset()
        for graph in round_._review_graphs.values():
            graph.reset()
        model.reset()


@pytest.mark.skipif(os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1",
                    reason="explicit CUDA activation lifetime proof")
@pytest.mark.parametrize("bits,symmetric,rows,n,k,group", [
    (4, False, 1, 320, 2560, 128),
    (8, False, 7, 67, 259, 40),
    (4, False, 33, 320, 256, 128),
])
@pytest.mark.parametrize("owner", ["text", "dispatch"])
def test_serial_a8_private_activations_reuse_storage_without_exported_aliases(
    bits, symmetric, rows, n, k, group, owner,
):
    from tests.ops.test_int_quantization import fixture
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators, PackedProjectionDispatch
    from cubic_serve.ops.strategies import OperatorConfig
    from cubic_serve.quantization.common.reference import wna8_activation
    from cubic_serve.quantization.extensions.int.reference import wna8_weight

    device = torch.device('cuda:0')
    weights = [fixture(bits, n=width, k=k, group=group, sym=symmetric)[0]
               for width in (n, 2 * n)]
    operators = PackedLinearOperators(OperatorConfig(compute_mode='wna8'))
    helper = TextGraphBindings(operators, device, torch.bfloat16)
    x = torch.zeros((rows, k), dtype=torch.bfloat16, device=device)
    outputs = [torch.empty((rows, weight.shape[0]), dtype=x.dtype, device=device)
               for weight in weights]
    if owner == "dispatch":
        dispatch = PackedProjectionDispatch(operators)
        for weight, out in zip(weights, outputs):
            dispatch.prepare(x, weight, out)
        dispatch.seal()
        bindings = [binding for _, binding in dispatch._entries.values()]
    else:
        bindings = [operators.bind_linear(x, weight, out) for weight, out in zip(weights, outputs)]
    references = [torch.empty_like(out) for out in outputs]
    originals = [operators.bind_linear(x, weight, out) for weight, out in zip(weights, references)]
    for binding in bindings:
        assert binding.reusable_scratch_indices
        if owner == "text":
            helper._reuse_linear_scratch(binding, 'test.serial', False)
        # A producer and its correction/merge can need several simultaneous
        # temporaries of one dtype. Sharing must retain these disjoint slots.
        pointers = [tensor.untyped_storage().data_ptr() for tensor in binding.scratch]
        assert len(pointers) == len(set(pointers))
    for index in set(bindings[0].reusable_scratch_indices) & set(bindings[1].reusable_scratch_indices):
        left, right = bindings[0].scratch[index], bindings[1].scratch[index]
        if left.dtype == right.dtype and (owner == "dispatch" or left.shape == right.shape):
            assert left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    graphs = [torch.cuda.CUDAGraph() for _ in (*weights, *weights)]
    try:
        for graph, binding in zip(graphs, (*bindings, *originals)):
            with torch.cuda.graph(graph, stream=stream):binding.launch()
        for iteration in range(3):
            host = (torch.arange(rows * k).reshape(rows, k).remainder(23) - iteration - 11).bfloat16() / 32
            x.copy_(host.to(device))
            # Poison every reused buffer before each producer. Consumer output
            # and weights remain independent from the reused live range.
            for index in (1, 0, 1):
                binding = bindings[index]
                for tensor in binding.scratch:
                    tensor.fill_(17)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):graphs[index].replay()
                stream.synchronize()
                with torch.cuda.stream(stream):graphs[index + len(weights)].replay()
                stream.synchronize()
                assert torch.equal(outputs[index].view(torch.uint8), references[index].view(torch.uint8))
                if iteration == 0:
                    from cubic_serve.guardrail.quantization import check_linear_enclosure
                    check_linear_enclosure(x, weights[index], outputs[index], compute_mode='wna8')
                snapshot = outputs[index].clone()
                for tensor in binding.scratch:tensor.zero_()
                assert torch.equal(outputs[index], snapshot)
        # An exported result cannot be substituted for a private temporary.
        fresh = operators.bind_linear(x, weights[0], outputs[0])
        index = fresh.reusable_scratch_indices[0]
        with pytest.raises(ValueError):fresh.rebind_scratch(index, outputs[0])
        with pytest.raises(ValueError):fresh.rebind_scratch(len(fresh.scratch), fresh.scratch[0])
    finally:
        for graph in graphs:graph.reset()


@pytest.mark.skipif(os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1",
                    reason="explicit CUDA grouped activation lifetime proof")
@pytest.mark.parametrize("owner", ["text", "dispatch"])
def test_grouped_a8_correction_and_split_keep_all_private_temporary_contracts(owner):
    from tests.ops.test_int_quantization import fixture
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators, PackedProjectionDispatch
    from cubic_serve.ops.strategies import OperatorConfig
    from cubic_serve.guardrail.quantization import check_linear_enclosure

    device = torch.device('cuda:0')
    x = (torch.arange(7 * 259, device=device).reshape(7, 259).remainder(17) - 8).bfloat16() / 32
    weights = tuple(fixture(8, n=67, k=259, group=40, sym=False)[0] for _ in range(2))
    operators = PackedLinearOperators(OperatorConfig(compute_mode='wna8'))
    outputs = tuple(torch.empty((7, 67), device=device, dtype=x.dtype) for _ in weights)
    expected = tuple(torch.empty_like(out) for out in outputs)
    if owner == "dispatch":
        dispatch = PackedProjectionDispatch(operators)
        dispatch.prepare_group(x, weights, outputs)
        dispatch.seal()
        binding = next(b for _, b in dispatch._entries.values() if b._launches)
    else:
        binding = operators.bind_linear_group(x, weights, outputs)
    reference = operators.bind_linear_group(x, weights, expected)
    assert set(binding.reusable_scratch_indices) == set(range(len(binding.scratch)))
    helper = TextGraphBindings(operators, device, x.dtype)
    if owner == "text":
        helper._reuse_linear_scratch(binding, 'test.group', False)
    graph, original = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    try:
        with torch.cuda.graph(graph, stream=stream):binding.launch()
        with torch.cuda.graph(original, stream=stream):reference.launch()
        for _ in range(2):
            for tensor in binding.scratch:tensor.fill_(19)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                graph.replay();original.replay()
            stream.synchronize()
            for actual, old, weight in zip(outputs, expected, weights):
                assert torch.equal(actual.view(torch.uint8), old.view(torch.uint8))
                check_linear_enclosure(x, weight, actual, compute_mode='wna8')
    finally:
        graph.reset();original.reset()
