"""Draft graph phases preserve their declared producer and ownership boundaries."""

import math
import os
from types import SimpleNamespace

import pytest
import torch

from cubic_serve.executor.draft_graph import CapturedDraftRound
from cubic_serve.executor.mtp_runner import Qwen3_5TPMTPBody
from cubic_serve.executor.text_serving import FlashNextDecodeStage


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA page write-set proof")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32,
                                  torch.int64, torch.bool])
@pytest.mark.parametrize("strided", [False, True])
def test_new_page_clear_preserves_every_unowned_byte_and_metadata(dtype, strided):
    from cubic_serve.executor.draft_graph import clear_new_pages

    rows, width, guard = 6, 259, 16
    itemsize = torch.empty((), dtype=dtype).element_size()
    # Interleaved page fields have a dense payload but a larger page pitch.
    # Random bytes in every gap witness protection of neighboring fields.
    stride = ((width * itemsize + 128 + 15) // 16 * 16 // itemsize
              if strided else width)
    generator = torch.Generator().manual_seed(845)
    original = torch.randint(0, 256, ((rows * stride + 2 * guard) * itemsize,),
                             dtype=torch.uint8, generator=generator)
    storage = original.cuda().view(dtype)
    field = torch.as_strided(storage, (rows, width), (stride, 1), guard)
    # Explicit logical activity is independent of continuation/segment starts.
    # Inactive rows include an invalid page index: masked writes must not touch it.
    activity = torch.tensor([1, 0, 1, 1, 0, 1], dtype=torch.int32)
    offsets = torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.int32)
    pages = torch.tensor([4, 2, 1, 0, 999, 3], dtype=torch.int64)
    expected = original.clone()
    for page in (4, 0, 3):
        start = (guard + page * stride) * itemsize
        expected[start:start + width * itemsize].zero_()
    for order in (torch.arange(rows), torch.arange(rows - 1, -1, -1)):
        inputs = tuple(value[order].contiguous().cuda() for value in (activity, offsets, pages))

        def launch():
            clear_new_pages[(rows, 2)](*inputs, field, width, 256, STRIDE=field.stride(0))

        storage.view(torch.uint8).copy_(original)
        launch()
        torch.cuda.synchronize()
        assert torch.equal(storage.view(torch.uint8).cpu(), expected)
        storage.view(torch.uint8).copy_(original)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                launch()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(storage.view(torch.uint8).cpu(), expected)
            for actual, source in zip(inputs, (activity, offsets, pages), strict=True):
                assert torch.equal(actual.cpu(), source[order])
        finally:
            graph.reset()


@pytest.mark.parametrize("capturing", [False, True])
@pytest.mark.parametrize("produce_logits", [False, True])
def test_composition_reuses_the_exact_owned_body_phase(monkeypatch, capturing, produce_logits):
    from cubic_serve.executor.draft_graph import _launch_body
    from cubic_serve.executor import resident_window

    calls = []
    graph = SimpleNamespace(replay=lambda: calls.append("replay"))
    stage = SimpleNamespace(captured_body_graph=lambda **kw: (
        calls.append(kw["produce_logits"]) or graph),
        launch=lambda **kw: pytest.fail("sealed body was expanded again"))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(resident_window, "append_captured_graph", lambda value: calls.append(("child", value)))
    _launch_body(stage, produce_logits=produce_logits)
    assert calls == [produce_logits, ("child", graph) if capturing else "replay"]


def test_composition_preserves_legacy_body_calling_conventions():
    from cubic_serve.executor.draft_graph import _launch_body

    seen = []
    _launch_body(SimpleNamespace(launch=lambda: seen.append("full")))
    _launch_body(SimpleNamespace(launch=lambda **kw: seen.append(kw)), produce_logits=False)
    assert seen == ["full", {"produce_logits": False}]


@pytest.mark.parametrize('budget', [False, True])
def test_ordinary_replay_uses_its_sealed_forward_before_completion(budget):
    calls = []
    round_ = CapturedDraftRound.__new__(CapturedDraftRound)
    round_._forward_graphs = {mode: SimpleNamespace(
        replay=lambda mode=mode: calls.append(('forward', mode))) for mode in (False, True)}
    completion = SimpleNamespace(replay=lambda: calls.append(('completion', budget)))
    round_.replay_ordinary(completion, budget=budget)
    assert calls == [('forward', budget), ('completion', budget)]
    del round_._forward_graphs[budget]
    calls.clear()
    with pytest.raises(RuntimeError, match='no sealed forward/completion'):
        round_.replay_ordinary(completion, budget=budget)
    assert calls == []


def test_ordinary_forward_capture_is_closed_before_serving(monkeypatch):
    round_ = CapturedDraftRound.__new__(CapturedDraftRound)
    round_._review_capture_open = False
    with pytest.raises(RuntimeError, match='closed after construction'):
        round_.capture_forward_graph(None)
    round_._review_capture_open = True
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: True)
    with pytest.raises(RuntimeError, match='sealed before parent capture'):
        round_.capture_forward_graph(None)


def test_body_composition_cannot_select_an_uncaptured_feedback_phase():
    stage = FlashNextDecodeStage.__new__(FlashNextDecodeStage)
    stage._closed, stage._captured_once = False, True
    stage._graph, stage._state_graph = object(), None
    assert stage.captured_body_graph() is stage._graph
    with pytest.raises(ValueError, match="not captured"):
        stage.captured_body_graph(produce_logits=False)


def test_packed_composition_reuses_serial_body_and_rejects_uncaptured_phases():
    from cubic_serve.executor.quantized_model_runner import QuantizedDecodeStage

    stage = QuantizedDecodeStage.__new__(QuantizedDecodeStage)
    stage._captured_once, stage._graph = True, object()
    stage._use_overlap = True
    stage._overlap_resources = (object(),)
    assert stage.captured_body_graph() is stage._graph
    assert stage.captured_body_graph() is not stage.owned_decode_graph
    with pytest.raises(ValueError, match="not captured"):
        stage.captured_body_graph(produce_logits=False)
    stage._graph = None
    with pytest.raises(ValueError, match="not available"):
        stage.captured_body_graph()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sealed packed body composition")
@pytest.mark.parametrize("capacity", [1, 2, 8, 32])
def test_packed_child_preserves_input_order_and_durable_state(capacity):
    from cubic_serve.executor.draft_graph import _launch_body
    from cubic_serve.executor.quantized_model_runner import QuantizedDecodeStage

    stream = torch.cuda.Stream()
    inputs = torch.zeros(capacity, device="cuda", dtype=torch.int64)
    state = torch.zeros_like(inputs)
    temporary = torch.empty_like(inputs)
    output = torch.empty_like(inputs)
    child = torch.cuda.CUDAGraph(keep_graph=True)
    parent = torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream())
    stage = QuantizedDecodeStage.__new__(QuantizedDecodeStage)
    stage._graph, stage._captured_once = child, True
    try:
        with torch.cuda.graph(child, stream=stream):
            torch.add(state, inputs, out=temporary)
            state.copy_(temporary)
            torch.mul(temporary, 3, out=output)
        child.instantiate()
        with torch.cuda.graph(parent, stream=stream):
            inputs.add_(1)
            _launch_body(stage)
            output.add_(7)
        stream.synchronize()
        # Independent integer recurrence checks incoming and outgoing edges,
        # repeated state consumption and nontrivial row order without tolerance.
        for order in (torch.arange(capacity), torch.arange(capacity - 1, -1, -1)):
            original = order + 11
            state.copy_(original)
            inputs.copy_(order)
            stream.wait_stream(torch.cuda.current_stream())
            for step in range(1, 4):
                with torch.cuda.stream(stream):
                    parent.replay()
                stream.synchronize()
                expected = original + step * order + step * (step + 1) // 2
                assert torch.equal(state.cpu(), expected)
                assert torch.equal(output.cpu(), 3 * expected + 7)
    finally:
        stream.synchronize()
        parent.reset()
        child.reset()


@pytest.mark.parametrize("capacity,logical", [(1, 1), (4, 3), (16, 9)])
def test_draft_transaction_refreshes_padding_after_inline_sparse_metadata(capacity, logical):
    from cubic_serve.executor.draft import TransactionalDraftBinding
    from cubic_serve.runtime import performance as serving

    # Include the permanent M1 bucket and qualified non-anchor logical rows.
    rank = serving._CapturedGraphRank.__new__(serving._CapturedGraphRank)
    rank.captured_m, rank._bound_active_rows = capacity, logical
    rank.max_sequence_tokens, rank.vocab_size, rank._table_pages = 64, 128, 1
    rank._block_size = 32
    rank._local_state_layers = torch.arange(1, dtype=torch.int32)
    rank._local_slot_offsets = rank._local_state_layers.numpy()
    rank._position_axes_slice = None
    rank._state_destination_column, rank._metadata_columns = 8, 12
    rank._source_slots_slice, rank._destination_slots_slice = slice(9, 10), slice(10, 11)
    rank._page_table_slice = slice(11, 12)
    rank._metadata_hosts = (torch.full((capacity, 12), -777, dtype=torch.int32),)
    rank._metadata_host_arrays = (rank._metadata_hosts[0].numpy(),)
    rank._index_hosts = (torch.empty(2 * capacity, dtype=torch.int64),)
    rank._index_host_arrays = (rank._index_hosts[0].numpy(),)
    rank._index_page_offset = capacity
    rank._scratch_pages = tuple(range(32, 32 + capacity))
    rank._scratch_states = tuple(range(64, 64 + capacity))
    buffers = SimpleNamespace(
        position_ids=torch.zeros(3, capacity, dtype=torch.int32),
        page_source=torch.zeros(capacity, dtype=torch.int32),
        page_destination=torch.arange(16, 16 + capacity, dtype=torch.int32),
        token_offsets=torch.ones(capacity, dtype=torch.int32),
        sequence_lengths=torch.ones(capacity, dtype=torch.int32),
        segment_lengths=torch.ones(capacity, dtype=torch.int32),
        page_table=torch.zeros(capacity, 1, dtype=torch.int32),
        state_slot_maps=(torch.zeros(2, 1, capacity, dtype=torch.int32),),
        # Inline alignment has sparse live rows beyond the cached host prefix.
        row_activity=torch.tensor(
            [int(i == 0 or i >= capacity - 2) for i in range(capacity)], dtype=torch.int32),
    )
    rank.body = SimpleNamespace(stage=1, buffers=buffers, arena=SimpleNamespace(
        spec=SimpleNamespace(page_capacity=32, state_slot_capacity=64,
                             scratch_capacity=capacity, scratch_arena_ids=("test",),
                             state_scratch_capacity=None,state_scratch_capacity_per_arena=capacity)))
    provider = serving._CapturedMTPRank.__new__(serving._CapturedMTPRank)
    provider._draft_binding = TransactionalDraftBinding(rank)
    provider.reserve_state_slots(tuple(range(logical)))
    rows = tuple(serving._BatchRow(
        f"r{i}", 17, 1, i, i, i, i, (i,), is_decode=True,
    ) for i in range(logical))
    count, _, _ = rank._stage_batch_rows(rows, host_slot=0)
    rank._bind_staged_metadata(rank._metadata_hosts[0][:count], count, 1)
    buffers.row_activity[:logical].fill_(1)
    buffers.row_activity[logical:count].zero_()
    written = torch.zeros(32 + capacity, dtype=torch.bool)
    written[buffers.page_destination[buffers.row_activity.bool()].long()] = True
    assert torch.equal(torch.nonzero(written).flatten(), torch.arange(logical))
    assert set(buffers.page_destination[logical:].tolist()) <= set(rank._scratch_pages)


def test_draft_transaction_invalidates_every_legacy_host_slot():
    from cubic_serve.runtime.performance import _CapturedMTPRank

    provider = _CapturedMTPRank.__new__(_CapturedMTPRank)
    provider._draft_binding, provider.captured_m = None, 16
    provider._bind_host_counts = [9, 10]
    provider.reserve_state_slots((0, 1))
    assert provider._bind_host_counts == [16, 16]


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("inflight", [False, True])
def test_constraint_phase_preparation_orders_upload_and_drains_failed_forward(monkeypatch, failed, inflight):
    from contextlib import nullcontext

    calls = []
    stream = SimpleNamespace(wait_stream=lambda other: calls.append("wait-mask"))
    current = SimpleNamespace(wait_stream=lambda other: calls.append("wait-forward"))
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    round_ = CapturedDraftRound.__new__(CapturedDraftRound)
    round_._constraint_forward_inflight = inflight
    round_.target = SimpleNamespace(slot=SimpleNamespace(stream=stream), _batch_phase_failed=False)
    round_.target_constraint_forward_graph = SimpleNamespace(replay=lambda: calls.append("forward"))
    round_.completed = SimpleNamespace(record=lambda *args: calls.append("completion"),
                                       synchronize=lambda: calls.append("drain"))
    tail = SimpleNamespace(replay=lambda: calls.append("verify"))

    def prepare():
        calls.append("mask")
        if failed:
            raise ValueError("private mask preparation failure")

    if failed:
        with pytest.raises(ValueError, match="private mask"):
            round_._replay_target_constraints(tail, prepare, current)
        assert calls == ([] if inflight else ["forward"]) + ["mask", "completion", "drain", "wait-forward"]
        assert not round_._constraint_forward_inflight
        assert round_.target._batch_phase_failed and round_._last_transfer
    else:
        round_._replay_target_constraints(tail, prepare, current)
        assert calls == ([] if inflight else ["forward"]) + ["mask", "wait-mask", "verify", "completion"]
        assert not round_.target._batch_phase_failed


@pytest.mark.parametrize("target,callback", [(False, lambda: None), (True, 7)])
def test_constraint_preparation_requires_its_explicit_sampling_phase(target, callback):
    round_ = CapturedDraftRound.__new__(CapturedDraftRound)
    with pytest.raises(RuntimeError, match="preparation requires"):
        round_.run((), (), target_constraints=target, prepare_constraints=callback)


@pytest.mark.parametrize('owners,work_ms,forward_ms,budget,expected', [
    (1, 30, 10, 4, 1), (4, 1, 10, 4, 1), (4, 4, 10, 4, 2),
    (4, 10, 10, 4, 4), (4, 10, 10, 2, 2), (16, .2, 10, 4, 1),
])
def test_constraint_mask_parallelism_prices_work_against_the_forward_window(
    monkeypatch, owners, work_ms, forward_ms, budget, expected,
):
    from cubic_serve.runtime import performance as perf

    monkeypatch.setattr(perf.time, 'monotonic_ns', lambda: 0)
    rows = tuple(SimpleNamespace(request_id=str(owner), token_id=7)
                 for owner in range(owners) for _ in range(2))
    segments = tuple(SimpleNamespace(start=2*i, stop=2*i+2) for i in range(owners))
    selected = []
    sampling = SimpleNamespace(constraint_bound=False, bind_constraints=lambda *args: None)
    def run(*args, prepare_constraints, **kwargs):
        prepare_constraints()
        return ((0, 7),) * owners
    round_ = SimpleNamespace(depth=1, run=run, _constraint_forward_queued_ns=0,
        _constraint_forward_ms=forward_ms, _constraint_mask_work_per_owner_ns=work_ms*1e6,
        target=SimpleNamespace(_sampling=sampling), head=SimpleNamespace(_sampling=sampling))
    def masks(proposals, *, workers):
        selected.append(workers)
        return {key: ((None, None), 1) for key in proposals}
    grammar = SimpleNamespace(bindings={str(i): object() for i in range(owners)},
        pool=SimpleNamespace(batch_mask_workers=budget), target_prefix_masks_many=masks,
        commit=lambda outputs: None)
    perf._execute_target_constrained_carried_round(round_, rows, segments, grammar)
    assert selected == [expected]


@pytest.mark.parametrize('inflight', [False, True])
def test_constraint_continuation_exit_drains_and_poisons_only_unfinished_work(inflight):
    calls = []
    round_ = CapturedDraftRound.__new__(CapturedDraftRound)
    round_._constraint_forward_inflight = inflight
    round_._constraint_continuation_binding = object()
    round_.target = SimpleNamespace(
        _batch_phase_failed=False,
        slot=SimpleNamespace(stream=SimpleNamespace(synchronize=lambda: calls.append('drain'))))
    round_.close_constraint_continuation()
    assert calls == (['drain'] if inflight else [])
    assert round_.target._batch_phase_failed is inflight
    assert round_._constraint_continuation_binding is None
    assert not round_._constraint_forward_inflight


@pytest.mark.parametrize('damage', [None, 'owner', 'source', 'target_table',
                                   'provider_table', 'target_extent', 'provider_extent'])
@pytest.mark.parametrize('depth', [1, 3, 7])
def test_constraint_continuation_bind_checks_permuted_ownership_before_any_upload(damage, depth):
    import numpy as np
    from cubic_serve.runtime.continuous_batch import DraftWindowBinding

    copies = []
    def host(shape, dtype=np.int64):
        array = np.zeros(shape, dtype=dtype)
        return SimpleNamespace(numpy=lambda: array)
    def device(name):
        return SimpleNamespace(copy_=lambda value, **kw: copies.append((name, value.numpy().copy())))
    round_ = CapturedDraftRound.__new__(CapturedDraftRound)
    round_._constraint_forward_inflight = False
    round_._constraint_continuation_binding = None
    round_._last_target_constraints = True
    round_.constraint_continuation_forward_graph = object()
    stride = depth + 1
    round_.depth, round_.stride, round_.C, round_.CS = depth, stride, 2, 4 + depth
    owners, targets, providers = (1, 0), ((10, 11), (12, 13)), ((20, 21), (22, 23))
    round_.host_array = np.zeros(1 + 2 * round_.CS, dtype=np.int32)
    round_.host_array[0] = 2
    ctx = round_.host_array[1:].reshape(2, round_.CS)
    ctx[:, 1] = owners
    ctx[:, 2 + depth:] = providers
    round_.metadata_host = host((2 * stride, 13), np.int32)
    metadata = round_.metadata_host.numpy()
    for physical, owner in enumerate(owners):
        metadata[physical * stride:physical * stride + stride, 9:11] = np.arange(2) + 2 * owner
        metadata[physical * stride + depth, 8] = owner
        metadata[physical * stride:physical * stride + stride, 11:13] = targets[physical]
    round_.pool = SimpleNamespace(metadata=[('a', 2, 7, None, 2, None, (8,) * depth)] * 2)
    round_.target = SimpleNamespace(_source_slots_slice=slice(9, 11),
        _state_destination_column=8, _page_table_slice=slice(11, 13),
        _block_size=16, _scratch_pages=tuple(range(100, 100 + 2 * stride)))
    round_.index_host, round_.index_device = host(2 * stride), device('reset-index')
    round_._resident_window = SimpleNamespace(
        staging={name: host(2) for name in ('target_limits', 'provider_limits')},
        target_limits=device('target'), provider_limits=device('provider'))
    target_positions, target_limits, provider_limits = (2, 2), (32, 32), (32, 32)
    if damage == 'owner': ctx[0, 1] = 0
    elif damage == 'source': metadata[0, 9] = 99
    elif damage == 'target_table': metadata[0, 11] = 99
    elif damage == 'provider_table': ctx[0, 2 + depth] = 99
    elif damage == 'target_extent': target_limits = (33, 32)
    elif damage == 'provider_extent': provider_limits = (4, 32)
    binding = DraftWindowBinding(depth, owners, target_positions, target_limits, provider_limits,
                                 (20, 20), targets, providers)
    if damage is not None:
        with pytest.raises(RuntimeError, match='changed its'):
            round_.bind_constraint_continuation(binding)
        assert copies == [] and round_._constraint_continuation_binding is None
    else:
        round_.bind_constraint_continuation(binding)
        assert [name for name, _ in copies] == ['target', 'provider', 'reset-index']
        assert copies[-1][1].tolist() == list(range(100, 100 + 2 * stride))
        assert round_._constraint_continuation_binding is binding


@pytest.mark.parametrize('attribute', ['target_constraint_forward_graph',
                                     'constraint_continuation_forward_graph'])
def test_target_constraint_forward_is_part_of_the_exclusive_portfolio_identity(attribute):
    from cubic_serve.runtime.quantized_qualification import portfolio_resources

    stream = SimpleNamespace(cuda_stream=7)
    stage = SimpleNamespace(rank=0, buffers=object(), arena=object(), stream=stream, graph=object())
    round_ = SimpleNamespace(graph=object(), candidate_graph=object(), pool=object(),
                             target=SimpleNamespace(slot=stage), target_constraint_graphs={False: object()},
                             target_constraint_forward_graph=object())
    executor = SimpleNamespace(slot=stage, body=object(), _pipelined_draft_rounds={1: round_},
                               _sampling=SimpleNamespace(greedy_plans=()))
    before = portfolio_resources({2: executor})
    setattr(round_, attribute, object())
    after = portfolio_resources({2: executor})
    assert before.graph_instance_id != after.graph_instance_id
    assert before.stream_lane_id == after.stream_lane_id


@pytest.mark.parametrize('budget', [False, True])
def test_shared_ordinary_forward_is_part_of_the_exclusive_portfolio_identity(budget):
    from cubic_serve.runtime.quantized_qualification import portfolio_resources

    stage = SimpleNamespace(rank=0, buffers=object(), arena=object(),
                            stream=SimpleNamespace(cuda_stream=7), graph=object())
    round_ = SimpleNamespace(graph=object(), candidate_graph=object(), pool=object(),
                            target=SimpleNamespace(slot=stage),
                            _forward_graphs={False: object(), True: object()})
    executor = SimpleNamespace(slot=stage, body=object(), _pipelined_draft_rounds={1: round_},
                               _sampling=SimpleNamespace(greedy_plans=()))
    before = portfolio_resources({2: executor})
    round_._forward_graphs[budget] = object()
    after = portfolio_resources({2: executor})
    assert before.graph_instance_id != after.graph_instance_id
    assert before.stream_lane_id == after.stream_lane_id


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sealed phase stream ordering")
@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("inflight", [False, True])
def test_constraint_phase_replay_consumes_late_upload_and_preserves_failure_boundary(failed, inflight):
    device = torch.device("cuda")
    stream = torch.cuda.Stream(device=device)
    current = torch.cuda.current_stream(device)
    inputs = torch.arange(257, dtype=torch.int64, device=device)
    hidden = torch.empty_like(inputs)
    mask = torch.zeros_like(inputs)
    output = torch.full_like(inputs, -991)
    host_mask = torch.arange(257, dtype=torch.int64, pin_memory=True) % 5
    stream.wait_stream(current)
    forward, tail = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(forward, stream=stream):
        hidden.copy_(inputs + 17)
    with torch.cuda.graph(tail, stream=stream):
        output.copy_(hidden * mask)
    stream.synchronize()
    output.fill_(-991)
    stream.wait_stream(current)
    round_ = CapturedDraftRound.__new__(CapturedDraftRound)
    round_._constraint_forward_inflight = inflight
    round_.target = SimpleNamespace(slot=SimpleNamespace(stream=stream), _batch_phase_failed=False)
    round_.target_constraint_forward_graph = forward
    round_.completed = torch.cuda.Event()
    if inflight:
        with torch.cuda.stream(stream):
            forward.replay()

    def prepare():
        if failed:
            raise ValueError("mask preparation failed after target launch")
        mask.copy_(host_mask, non_blocking=True)

    if failed:
        with pytest.raises(ValueError, match="mask preparation failed"):
            round_._replay_target_constraints(tail, prepare, current)
        assert round_.completed.query() and round_.target._batch_phase_failed
        assert torch.equal(output.cpu(), torch.full((257,), -991, dtype=torch.int64))
    else:
        round_._replay_target_constraints(tail, prepare, current)
        round_.completed.synchronize()
        assert torch.equal(output.cpu(), (torch.arange(257) + 17) * host_mask)
    assert torch.equal(hidden.cpu(), torch.arange(257) + 17)


@pytest.mark.parametrize("depth", [1, 2, 3, 7])
@pytest.mark.parametrize("invalid_accept", [False, True])
def test_runtime_masks_keep_request_rows_and_valid_prefix_witness_after_forward(depth, invalid_accept):
    from cubic_serve.runtime.performance import _execute_target_constrained_carried_round

    calls, bound, commits = [], [], []
    stride = depth + 1
    rows = tuple(SimpleNamespace(request_id=request, token_id=100 + row)
                 for request in ("plain", "schema") for row in range(stride))
    segments = tuple(SimpleNamespace(start=i * stride, stop=(i + 1) * stride) for i in range(2))
    sampling = SimpleNamespace(bind_constraints=lambda masks, owners: bound.append((masks, owners)),
                               constraint_bound=True)
    masks = tuple(torch.tensor([1 << i], dtype=torch.int32) for i in range(stride))

    def prefix(request, proposals):
        assert calls == ["forward"] and request == "schema"
        assert proposals == tuple(101 + d for d in range(depth))
        calls.append("mask")
        return masks, depth - int(invalid_accept)

    def run(actual_rows, actual_segments, *, target_constraints, prepare_constraints):
        assert actual_rows is rows and actual_segments is segments and target_constraints
        calls.append("forward")
        prepare_constraints()
        calls.append("verify")
        return ((depth, 31, 0), (depth, 37, 0))

    round_ = SimpleNamespace(depth=depth, target=SimpleNamespace(_sampling=sampling),
                             head=SimpleNamespace(_sampling=SimpleNamespace(constraint_bound=True)), run=run)
    grammar = SimpleNamespace(bindings={"schema": object()}, pool=SimpleNamespace(batch_mask_workers=1),
                               target_prefix_masks_many=lambda values, **kw: {
                                   key: prefix(key, tokens) for key, tokens in values.items()},
                               commit=lambda outputs: commits.append(outputs))
    if invalid_accept:
        with pytest.raises(RuntimeError, match="invalid grammar proposal"):
            _execute_target_constrained_carried_round(round_, rows, segments, grammar)
        assert not commits
    else:
        _execute_target_constrained_carried_round(round_, rows, segments, grammar)
        assert commits == [{"schema": (*range(101, 101 + depth), 37)}]
        assert not sampling.constraint_bound and not round_.head._sampling.constraint_bound
    actual_masks, owners = bound[0]
    assert owners == [False] * stride + [True] * stride
    assert actual_masks[:stride] == [None] * stride
    assert all(a is b for a, b in zip(actual_masks[stride:], masks))
    assert calls == ["forward", "mask", "verify"]


def test_composition_fixture_preserves_actual_carried_tokens_when_rows_are_permuted():
    from cubic_serve.generation.sampler import GREEDY_SAMPLING
    from cubic_serve.runtime.quantized_qualification import _draft_window_audit_rows

    round_ = SimpleNamespace(depth=2, stride=3, target=SimpleNamespace(
        vocab_size=128, _block_size=16, _scratch_states=tuple(range(10, 16)),
        body=SimpleNamespace(arena=SimpleNamespace())))
    pending = [(f"r{i}", 3, 20 + i, GREEDY_SAMPLING, 7, None, (30 + i, 40 + i))
               for i in range(2)]
    rows = _draft_window_audit_rows(round_, pending, 2, 1, owner_order=(1, 0))
    assert tuple(row.token_id for row in rows) == (21, 31, 41, 20, 30, 40)
    assert tuple(row.state_source_slot for row in rows) == (1, 1, 1, 0, 0, 0)


@pytest.mark.parametrize("damage", [None, "target", "pool", "missing"])
def test_retained_rounds_require_exact_bound_portfolio_without_new_capture(damage):
    from cubic_serve.executor.draft_graph import capture_draft_rounds

    from cubic_serve.executor.draft_graph import DraftPendingPool
    pool = DraftPendingPool(2, 2, [None, None], torch.empty(2, 1, 2, 4),
                            torch.empty(2, 1, 2, 4, dtype=torch.int64),
                            torch.empty(2, 2, dtype=torch.int64), {})
    providers = {
        m: SimpleNamespace(body=SimpleNamespace(topology=SimpleNamespace(pp_size=1),
                                               launch_projection=lambda: None),
                           _pending_draft_pool=pool)
        for m in (1, 2, 4, 8)
    }
    targets = {m: object() for m in (4, 8)}
    rounds = {
        (review, 2): SimpleNamespace(target=targets[review], align=providers[review],
                                head=providers[head], pool=pool, depth=2)
        for review, head in ((4, 1), (8, 2))
    }
    if damage == "target":
        rounds[(4, 2)].target = object()
    elif damage == "pool":
        providers[8]._pending_draft_pool = object()
    elif damage == "missing":
        del rounds[(8, 2)]
    if damage is None:
        assert capture_draft_rounds(providers, targets, retained_rounds=rounds) is rounds
    else:
        with pytest.raises(ValueError, match="retained draft"):
            capture_draft_rounds(providers, targets, retained_rounds=rounds)


@pytest.mark.parametrize("capacities,ranks", [
    ((2, 4, 1), (0, 0, 0)),
    ((4, 4, 1), (0, 0, 0)),
    ((6, 6, 3), (0, 0, 0)),
    ((0, 0, 0), (0, 0, 0)),
    ((2, 2, 1), (0, 1, 0)),
])
def test_review_provider_abi_rejects_before_device_allocation(capacities, ranks):
    stages = [SimpleNamespace(captured_m=m, _rank=rank)
              for m, rank in zip(capacities, ranks)]
    with pytest.raises(ValueError, match="ABI"):
        CapturedDraftRound(*stages, None)


def test_missing_projection_is_a_typed_rejection_before_capture():
    stages = [SimpleNamespace(captured_m=m, _rank=0, body=SimpleNamespace())
              for m in (2, 2, 1)]
    with pytest.raises(ValueError, match="head-only projection"):
        CapturedDraftRound(*stages, None)


def test_native_provider_projection_consumes_stored_input_and_preserves_raw_state():
    body = Qwen3_5TPMTPBody.__new__(Qwen3_5TPMTPBody)
    norm = torch.tensor([[.125, -.25, .5]], dtype=torch.bfloat16)
    hidden = torch.full_like(norm, float("nan"))
    weight = torch.tensor([[1., 2., 3.], [-4., 5., -6.]], dtype=norm.dtype)
    logits = torch.empty(1, 2, dtype=norm.dtype)
    body.buffers = SimpleNamespace(norm=norm, hidden=hidden, logits=logits)
    body._weights = {"lm_head.weight": weight}
    body._lm_head = object()
    initial = hidden.view(torch.uint8).clone()

    def linear(binding, value, matrix, output):
        assert binding is body._lm_head and value is norm and matrix is weight
        output.copy_(value @ matrix.T)

    body._launch_linear = linear
    body.normalize_feedback = lambda *args: pytest.fail("head phase reran its producer")
    body.launch_projection()
    assert torch.equal(logits, norm @ weight.T)
    assert torch.equal(hidden.view(torch.uint8), initial)


def test_typed_provider_projection_does_not_reenter_feedback_or_state():
    stage = FlashNextDecodeStage.__new__(FlashNextDecodeStage)
    stage._closed = False
    calls = []
    stage.program = SimpleNamespace(
        logits_step=lambda: calls.append("projection"),
        launch_feedback=lambda: pytest.fail("head phase advanced provider state"),
    )
    stage.launch_projection()
    assert calls == ["projection"]
    stage._closed = True
    with pytest.raises(ValueError, match="projection"):
        stage.launch_projection()


@pytest.mark.parametrize("depth", [2, 3, 7])
def test_depth_geometry_accepts_power_of_two_review_with_ownerless_tail(depth):
    import triton
    capacities = (triton.next_power_of_2(depth + 1),) * 2 + (1,)
    stages = [SimpleNamespace(captured_m=m, _rank=0, body=SimpleNamespace())
              for m in capacities]
    # The geometry is valid; the next explicit provider capability rejects it.
    with pytest.raises(ValueError, match="head-only projection"):
        CapturedDraftRound(*stages, None, depth=depth)


@pytest.mark.parametrize("depth", [0, -1, True, 1.5])
def test_depth_rejects_before_device_allocation(depth):
    stages = [SimpleNamespace(captured_m=m, _rank=0) for m in (2, 2, 1)]
    with pytest.raises(ValueError, match="ABI"):
        CapturedDraftRound(*stages, None, depth=depth)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native draft coordination")
@pytest.mark.parametrize("depth", [1, 2, 3, 7])
def test_prefix_merge_obeys_first_rejection_and_joint_result_masks_padding(depth):
    import triton
    from cubic_serve.executor.draft_graph import merge_prefix, joint_result

    # A late acceptance cannot make an earlier rejected proposal live.
    logical = depth + 1
    capacity = triton.next_power_of_2(logical + 1)
    ctx = torch.tensor([logical], dtype=torch.int32, device="cuda")
    decisions = torch.full((depth, capacity, 2), -1, dtype=torch.int64, device="cuda")
    for row in range(logical):
        for step in range(depth):
            decisions[step, row, 0] = int(step != row)
            decisions[step, row, 1] = 100 + 10 * row + step
    merged = torch.empty((capacity, 2), dtype=torch.int64, device="cuda")
    merge_prefix[(1,)](ctx, decisions, merged, capacity, depth, capacity)
    expected = torch.full((capacity, 2), -1, dtype=torch.int64)
    for row in range(logical):
        expected[row] = torch.tensor((row, 100 + 10 * row + min(row, depth - 1)))
    assert torch.equal(merged.cpu(), expected)
    q = torch.arange(depth * capacity, dtype=torch.int64, device="cuda").reshape(depth, capacity)
    guard = torch.tensor(True, device="cuda")
    result = torch.empty((capacity, depth + 3), dtype=torch.int64, device="cuda")
    joint_result[(1,)](ctx, merged, q, guard, result, capacity, 1, 1024, capacity, depth)
    actual = result.cpu()
    assert torch.equal(actual[:, :2], expected)
    assert torch.equal(actual[:, 2:-1], q.cpu().T)
    assert actual[:logical, -1].all() and not actual[logical:, -1].any()
    # An invalid intermediate guard rejects the whole transaction.
    guard.fill_(False)
    joint_result[(1,)](ctx, merged, q, guard, result, capacity, 1, 1024, capacity, depth)
    assert not result[:, -1].cpu().any()


@pytest.mark.parametrize("engine_fields", ["none", "partial", "complete"])
def test_pending_pool_distinguishes_bootstrap_and_complete_engine_schemas(engine_fields):
    from cubic_serve.executor.draft_graph import DraftPendingPool
    from cubic_serve.runtime.cache import CacheArena, CacheArenaSpec, KVLayout, StateSlotSpec
    from cubic_serve.runtime.cache.shared_pool import SharedCacheField, SharedCacheStorage

    specs = (StateSlotSpec("counter", torch.int64, (1,)),
             StateSlotSpec("rounded", torch.bfloat16, (3,)))
    schema = CacheArenaSpec("provider", 8, 4, specs, scratch_capacity=8)
    layout = KVLayout((0,), 2, 1, 4)
    provider = SimpleNamespace(
        _sampling=SimpleNamespace(workspace=SimpleNamespace(
            gathered_scores=torch.empty(2, 8, 20),
            gathered_token_ids=torch.empty(2, 8, 20, dtype=torch.int64))),
        _max_draft_tokens=3,
        state_fields={s.state_id: torch.empty((5, *s.shape), dtype=s.dtype) for s in specs},
    )
    extras = (() if engine_fields == "none" else
              (SharedCacheField("feedback", torch.bfloat16, (3,)),))
    if engine_fields == "complete":
        extras += DraftPendingPool.fields(provider)
    storage = SharedCacheStorage(((schema, layout),), budget_bytes=65536,
                                state_capacity=4, extra_state_fields=extras, device="cpu")
    arena = CacheArena(storage.arena_specs[0], layout, "cpu", rank_id=0, shared_storage=storage)
    provider.body = SimpleNamespace(arena=arena, hidden=torch.empty(8, 3))
    provider.state_fields = arena.states
    if engine_fields == "partial":
        with pytest.raises(ValueError, match="omits declared provider fields"):
            DraftPendingPool.allocate(provider)
        return
    pool = DraftPendingPool.allocate(provider)
    assert pool.scores.shape == (3, 2, 4, 20)
    assert pool.tokens.shape == (3, 4)
    assert (pool.scores.untyped_storage().data_ptr() == storage.backing.data_ptr()
            ) == (engine_fields == "complete")
    for name, value in pool.backup.items():
        assert value.shape == arena.states[name][:4].shape
        assert value.dtype == arena.states[name].dtype
        if engine_fields == "none":
            assert value.is_contiguous()


def test_pending_pool_prices_only_durable_owners_and_restores_every_typed_field():
    from cubic_serve.executor.draft_graph import DraftPendingPool

    fields = {"rounded": torch.arange(30).reshape(10, 3).bfloat16(),
              "counter": torch.arange(10, dtype=torch.int64)}
    workspace = SimpleNamespace(gathered_scores=torch.empty(2, 8, 20),
                                gathered_token_ids=torch.empty(2, 8, 20, dtype=torch.int64))
    provider = SimpleNamespace(
        _sampling=SimpleNamespace(workspace=workspace), _max_draft_tokens=3,
        body=SimpleNamespace(arena=SimpleNamespace(spec=SimpleNamespace(state_slot_capacity=4)),
                             hidden=torch.empty(8, 3)), state_fields=fields,
    )
    pool = DraftPendingPool.allocate(provider)
    assert pool.scores.shape == (3, 2, 4, 20)
    assert pool.tokens.shape == (3, 4)
    assert all(pool.backup[name].shape == field[:4].shape for name, field in fields.items())
    for name, field in fields.items():
        pool.backup[name].copy_(field[:4])
    pool.metadata[1] = ("request",)
    fields["rounded"][1].zero_()
    fields["counter"][1] = -1
    pool.restore(provider, (SimpleNamespace(request_id="request", state_source_slot=1),))
    assert torch.equal(fields["rounded"][1], torch.tensor([3, 4, 5]).bfloat16())
    assert fields["counter"][1] == 1 and pool.metadata[1] is None
    # A recycled owner with another request identity never loads stale state.
    pool.metadata[1] = ("old-request",)
    fields["counter"][1] = 99
    pool.restore(provider, (SimpleNamespace(request_id="new-request", state_source_slot=1),))
    assert fields["counter"][1] == 99 and pool.metadata[1] is None


def test_ownerless_pricing_preserves_pending_geometry_without_live_owner_state():
    from cubic_serve.executor.draft_graph import DraftPendingPool, capture_draft_rounds
    provider = SimpleNamespace(
        _sampling=SimpleNamespace(workspace=SimpleNamespace(
            gathered_scores=torch.empty(1, 1024, 20),
            gathered_token_ids=torch.empty(1, 1024, 20, dtype=torch.int64))),
        _max_draft_tokens=3,
        body=SimpleNamespace(hidden=torch.empty(1024, 3), arena=SimpleNamespace(
            spec=SimpleNamespace(state_slot_capacity=1),
            shared_storage=SimpleNamespace(extra_state_fields=()))),
        state_fields={"feedback": torch.empty(2, 3, dtype=torch.bfloat16)},
    )
    pool = DraftPendingPool.allocate(provider, pricing_owner_capacity=256)
    assert pool.owners == 256 and pool.tokens.shape == (3, 256)
    assert pool.backup["feedback"].shape == (256, 3)
    assert provider.state_fields["feedback"].shape == (2, 3)
    with pytest.raises(ValueError, match="cannot bind a serving portfolio"):
        capture_draft_rounds({}, {}, pricing_owner_capacity=256)
    provider.body.arena.shared_storage.extra_state_fields = (object(),)
    with pytest.raises(ValueError, match="bootstrap schema"):
        DraftPendingPool.allocate(provider, pricing_owner_capacity=256)


def test_round_portfolio_selection_is_a_provider_capability():
    from cubic_serve.executor.draft_graph import capture_draft_rounds

    assert capture_draft_rounds({}, {}) == {}
    provider = SimpleNamespace(body=SimpleNamespace(topology=SimpleNamespace(pp_size=1)))
    assert capture_draft_rounds({1: provider}, {}) == {}
    provider.body.launch_projection = lambda: None
    provider.body.topology.pp_size = 2
    assert capture_draft_rounds({1: provider}, {}) == {}


@pytest.mark.parametrize("missing_intermediate", [False, True])
def test_dynamic_retained_portfolio_covers_intermediate_depth_without_recapture(
    monkeypatch, missing_intermediate,
):
    from cubic_serve.executor import draft_graph

    pool = draft_graph.DraftPendingPool(
        3, 2, [None, None], torch.empty(3, 1, 2, 20),
        torch.empty(3, 1, 2, 20, dtype=torch.int64),
        torch.empty(3, 2, dtype=torch.int64), {})
    providers = {m: SimpleNamespace(
        body=SimpleNamespace(topology=SimpleNamespace(pp_size=1),
                             launch_projection=lambda: None),
        _pending_draft_pool=pool,
    ) for m in (1, 2, 4, 8)}
    targets = {m: object() for m in (2, 4, 8)}
    # The declared K3 portfolio must also serve K2; owner capacity remains two,
    # even though the existing provider graphs can hold more physical rows.
    bindings = ((2, 1, 1), (4, 1, 2), (4, 2, 1),
                (8, 2, 2), (4, 3, 1), (8, 3, 2))
    rounds = {(review, depth): SimpleNamespace(
        target=targets[review], align=providers[review], head=providers[head],
        pool=pool.prefix(depth), depth=depth,
    ) for review, depth, head in bindings}
    monkeypatch.setattr(draft_graph, "CapturedDraftRound",
                        lambda *a, **kw: pytest.fail("retained portfolio recaptured"))
    if missing_intermediate:
        del rounds[(4, 2)]
        del rounds[(8, 2)]
        with pytest.raises(ValueError, match="retained draft"):
            draft_graph.capture_draft_rounds(
                providers, targets, dynamic_depth=True, retained_rounds=rounds)
    else:
        assert draft_graph.capture_draft_rounds(
            providers, targets, dynamic_depth=True, retained_rounds=rounds) is rounds


@pytest.mark.parametrize('owners', [1, 2, 3])
@pytest.mark.parametrize('depth', [1, 2, 3])
@pytest.mark.parametrize('dynamic', [False, True])
@pytest.mark.parametrize('request_limit', [None,1,4])
def test_prompt_token_capacity_does_not_create_draft_owners(monkeypatch, owners, depth, dynamic, request_limit):
    from cubic_serve.executor import draft_graph

    pool = SimpleNamespace(owners=owners, depth=depth)
    pool.prefix = lambda d: SimpleNamespace(owners=owners, depth=d)
    monkeypatch.setattr(draft_graph.DraftPendingPool, 'allocate', lambda _: pool)
    built = []

    def build(target, align, head, pending, **kwargs):
        assert head.captured_m <= pending.owners
        built.append(head.captured_m)
        return SimpleNamespace()

    monkeypatch.setattr(draft_graph, 'CapturedDraftRound', build)
    providers = {m: SimpleNamespace(captured_m=m, body=SimpleNamespace(
        topology=SimpleNamespace(pp_size=1), launch_projection=lambda: None,
    )) for m in (1, 2, 4, 8, 16, 32)}
    if request_limit is not None:
        for provider in providers.values():provider._max_request_owners=request_limit
    targets = {m: SimpleNamespace() for m in (1, 2, 4, 8, 16, 32, 64, 128)}
    progress = []
    def report(completed, total, *, phase):
        # A build is complete only after its constructor returned successfully.
        assert completed == len(built)
        progress.append((completed, total, phase))
    draft_graph.capture_draft_rounds(providers, targets, dynamic_depth=dynamic, progress=report)
    assert built and max(built)<=min(owners,request_limit or owners)
    assert progress == [(done, len(built), "rounds") for done in range(len(built) + 1)]


@pytest.mark.parametrize("enforce_capacity", [False, True])
def test_round_portfolio_uses_state_and_page_bytes_without_limiting_prefill_graphs(
    monkeypatch, enforce_capacity,
):
    from cubic_serve.executor import draft_graph
    from cubic_serve.runtime.cache.shared_pool import SharedCachePool

    allocator = SharedCachePool(pool_id="cold", budget_bytes=384, page_bytes=4, state_bytes=32)
    pool = SimpleNamespace(owners=32, depth=3)
    pool.prefix = lambda d: SimpleNamespace(owners=32, depth=d)
    monkeypatch.setattr(draft_graph.DraftPendingPool, "allocate", lambda _: pool)
    built = []
    def build(target, align, head, pending, **kwargs):
        built.append((head.captured_m, kwargs["depth"]))
        return SimpleNamespace()
    monkeypatch.setattr(draft_graph, "CapturedDraftRound", build)
    storage = SimpleNamespace(allocator=allocator)
    providers = {m: SimpleNamespace(captured_m=m, body=SimpleNamespace(
        topology=SimpleNamespace(pp_size=1), launch_projection=lambda: None,
        arena=SimpleNamespace(shared_storage=storage,
                              layout=SimpleNamespace(block_size_tokens=32)),
    )) for m in (1, 2, 4, 8, 16, 32, 64, 128)}
    targets = {m: SimpleNamespace() for m in providers}
    draft_graph.capture_draft_rounds(providers, targets, dynamic_depth=True,
                                    enforce_capacity=enforce_capacity)
    assert set(targets) == {1, 2, 4, 8, 16, 32, 64, 128}
    if enforce_capacity:
        assert set(built) == {(1, 1), (2, 1), (4, 1), (1, 2), (2, 2), (1, 3), (2, 3)}
    else:
        assert (32, 3) in built


@pytest.mark.parametrize('dynamic', [False, True])
@pytest.mark.parametrize('fallback', [None, 'depth', 'constraint', 'context'])
def test_carried_round_preserves_dynamic_depth_and_fallback_checkpoint(dynamic, fallback):
    from cubic_serve.runtime import performance as p
    from cubic_serve.runtime.continuous_batch import BatchCommand, BatchRow

    depth = 3
    current = 2 if fallback == 'depth' else depth
    rows = tuple(BatchRow(
        'request', 9 if index == 0 else -1, 10 + index, 0, 0, 0, 0, (0, 1),
        segment_length=current + 1 if index == 0 else 0,
        sample_ordinal=20 + index, mtp_position=10 + index,
        mtp_page_destination=0, mtp_page_table=(0, 1), is_decode=True,
    ) for index in range(current + 1))
    restored, reserved, replays = [], [], []
    root = rows[0]
    key = (root.request_id, root.mtp_position, root.token_id, root.sampling,
           root.sample_ordinal, root.position_axes)
    pool = SimpleNamespace(depth=depth, metadata=[(*key, (11, 12, 13))],
                           restore=lambda provider, value: restored.append(value))
    provider = SimpleNamespace(_pending_draft_pool=pool, reserve_state_slots=reserved.append)

    def run(value, segments):
        replays.append(value)
        return ((2, 23, 24, 25, 26, 1),)

    round_ = SimpleNamespace(depth=depth, C=1, run=run,
                             active_draft_window=SimpleNamespace(qualified=True))
    executor = SimpleNamespace(_pipelined_draft_round=round_, _block_size=16,
                               replay_elapsed_ms=lambda: 5.)
    admission = SimpleNamespace(dynamic_speculation=dynamic,
                                max_sequence_tokens=16 if fallback == 'context' else 32)
    command = BatchCommand(0, 4, rows, constraints=('grammar',) if fallback == 'constraint' else ())
    result = p._execute_pipelined_draft_batch(
        command, executor, {1: provider}, None, admission, (1, 2, 4), rank=0, sampler_rank=0,
    )
    if fallback is not None:
        assert result is None and restored == [rows] and not replays and not reserved
    else:
        assert not restored and len(replays) == 1
        assert tuple(row.token_id for row in replays[0]) == (9, 11, 12, 13)
        assert result.segment_results[0].output_token_ids == (11, 12, 23)
        assert result.draft_ingress is False
        assert pool.metadata[0] == ('request', 13, 23, root.sampling, 23, None, (24, 25, 26))


@pytest.mark.parametrize('chunk', [0, -1, True, 0.5])
def test_prefill_token_budget_is_validated_before_checkpoint_access(chunk):
    from cubic_serve.runtime.quantized_graph import TextGraphExecutor

    with pytest.raises(ValueError, match='prefill_chunk must be positive'):
        TextGraphExecutor('/missing-checkpoint', devices=(0,), tp=1, pp=1, ep=1,
            max_requests=1, max_context=None, prefill_chunk=chunk,
            operator_config=None, eos_token_ids=())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="device draft ingress")
@pytest.mark.parametrize("depth", [1, 2, 3, 7])
@pytest.mark.parametrize("capacity,logical,embedding", [(1, 1, True), (4, 1, True),
    (4, 3, True), (8, 3, True), (32, 3, True), (32, 17, True), (4, 4, False)])
def test_round_ingress_preserves_ownership_and_typed_ordinals(depth, capacity, logical, embedding):
    import triton
    from cubic_serve.executor.draft_graph import advance_round_context

    stride = depth + 1
    physical = triton.next_power_of_2(capacity * stride)
    page_size, table_width, layers = 5, 20, 2
    table_column = 9 + 2 * layers
    axes_column = table_column + table_width
    columns = axes_column + 3
    context_stride = 2 + depth + table_width
    ctx = torch.full((1 + capacity * context_stride,), -91, dtype=torch.int32)
    ctx[0] = logical
    result = torch.full((capacity, depth + 3), -1, dtype=torch.int64)
    metadata = torch.full((physical, columns), -37, dtype=torch.int32)
    ordinals = torch.full((capacity,), 2**54 + 7, dtype=torch.int64)
    base = torch.arange(capacity, dtype=torch.int64) + 2**54 + 19
    limits = torch.full((capacity,), page_size * table_width, dtype=torch.int64)
    for owner in range(logical):
        start = 3 + owner
        ctx[1 + owner * context_stride:3 + owner * context_stride] = torch.tensor((start, logical-1-owner))
        ctx[3 + owner * context_stride:3 + owner * context_stride + depth] = 5
        accepted = owner % (depth + 1)
        result[owner] = torch.tensor((accepted, 19 + owner, *range(21, 21 + depth), 1))
        for j in range(stride):
            row = owner * stride + j
            metadata[row, :5] = torch.tensor((start+j, 71, 71, (start+j)%page_size, start+j+1))
            metadata[row, 7] = stride if j == 0 else 0
            # A permutation of physical pages; coordinates are not page indices.
            metadata[row, table_column:axes_column] = torch.arange(table_width).flip(0) + 100*owner
            metadata[row, axes_column:] = torch.tensor((11 + j, 31 + 2*j, 17))
    expected_ctx, expected_meta, expected_ord = ctx.clone(), metadata.clone(), ordinals.clone()
    for owner in range(logical):
        delta = int(result[owner, 0]) + 1
        expected_ctx[1+owner*context_stride] += delta
        expected_ctx[3+owner*context_stride:3+owner*context_stride+depth] = result[owner, 2:-1].to(torch.int32)
        expected_ord[owner] = base[owner] + delta
        for j in range(stride):
            row = owner*stride+j
            position = int(metadata[row, 0])+delta
            page = metadata[row, table_column + position//page_size]
            expected_meta[row, :5] = torch.tensor((position, page, page, position%page_size, position+1))
            token = int(result[owner, 1+j])
            if embedding:
                expected_meta[row, 5:7] = torch.tensor((token-20 if 20 <= token < 24 else 0,
                                                       int(20 <= token < 24)))
            expected_meta[row, axes_column:] += torch.tensor((delta, 2*delta, 0))
    tensors = [x.cuda() for x in (ctx, result, metadata, ordinals, base, limits, limits)]
    ready = torch.ones(capacity, dtype=torch.bool, device="cuda")
    advance_round_context[(capacity,)](
        *tensors, ready, capacity, physical, context_stride, columns,
        axes_column, table_column, page_size, 20, 24, 64, depth,
        triton.next_power_of_2(stride), embedding,
    )
    assert torch.equal(tensors[0].cpu(), expected_ctx)
    assert torch.equal(tensors[2].cpu(), expected_meta)
    assert torch.equal(tensors[3].cpu(), expected_ord)
    assert ready.cpu().tolist() == [r < logical for r in range(capacity)]
    # Repeated device-fed rounds retain the committed integer coordinates;
    # each independent CPU step uses the preceding committed integer state.
    # Stable logical inputs occur in several padded physical capacities above.
    for turn in range(3):
        tensors[4].copy_(tensors[3])  # The real prepare_pending ordinal boundary.
        for owner in range(logical):
            delta = int(result[owner, 0]) + 1
            expected_ctx[1 + owner * context_stride] += delta
            expected_ord[owner] += delta
            for j in range(stride):
                row = owner * stride + j
                position = int(expected_meta[row, 0]) + delta
                page = expected_meta[row, table_column + position // page_size]
                expected_meta[row, :5] = torch.tensor(
                    (position, page, page, position % page_size, position + 1))
                expected_meta[row, axes_column:] += torch.tensor((delta, 2 * delta, 0))
        advance_round_context[(capacity,)](
            *tensors, ready, capacity, physical, context_stride, columns,
            axes_column, table_column, page_size, 20, 24, 64, depth,
            triton.next_power_of_2(stride), embedding,
        )
        assert torch.equal(tensors[0].cpu(), expected_ctx)
        assert torch.equal(tensors[2].cpu(), expected_meta)
        assert torch.equal(tensors[3].cpu(), expected_ord)
        assert ready.cpu().tolist() == [r < logical for r in range(capacity)]
    # Capacity exhaustion and a failed producer guard are non-mutating stops.
    for boundary in ("target", "provider", "producer", "proposal", "ordinal", "axes", "position"):
        before = [x.clone() for x in tensors]
        if boundary == "target": tensors[5].zero_()
        elif boundary == "provider": tensors[6].zero_()
        elif boundary == "producer": tensors[1][:, -1].zero_()
        elif boundary == "proposal": tensors[1][:, 2].fill_(-1)
        elif boundary == "ordinal": tensors[4].fill_(2**63-1)
        elif boundary == "position":
            tensors[2][:, 0].fill_(2**31-1)
            tensors[5].fill_(2**40)
        else:
            for owner in range(logical):
                tensors[2][owner*stride, axes_column] = 2**31-2
                tensors[2][owner*stride+1, axes_column] = 2**31-1
        protected = {index: tensors[index].clone() for index in (0, 2, 3)}
        advance_round_context[(capacity,)](
            *tensors, ready, capacity, physical, context_stride, columns,
            axes_column, table_column, page_size, 20, 24, 64, depth,
            triton.next_power_of_2(stride), embedding,
        )
        assert not ready.any().item()
        for index in (0, 2, 3):
            assert torch.equal(tensors[index], protected[index])
        tensors = before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native draft journal guard")
def test_native_draft_journal_rejects_invalid_prefix_and_counter_without_wrap():
    from pathlib import Path
    from cuda.bindings import runtime as rt
    from cubic_serve.executor.resident_window import check
    from cubic_serve.ops.cuda.native import CudaScalar
    from cubic_serve.ops.cuda.native_compiler import compile_native
    import cubic_serve.executor.draft_graph as graph_module

    device = torch.device("cuda", torch.cuda.current_device())
    _context_owner = torch.empty(1, device=device)
    source = (Path(graph_module.__file__).parents[1] /
              "ops/cuda/sampling/resident_draft_window.cu").read_bytes()
    module, entry = compile_native((("entry.cu", source),), device, "resident_draft_window")
    for fault in ("negative", "maximum", "minimum", "too_deep", "bonus", "proposal",
                  "producer", "budget", "ordinal"):
        meta = torch.tensor([1, 1, 0, 0, 0], dtype=torch.int32, device=device)
        ctx = torch.tensor([1, 3, 0, 2, 0], dtype=torch.int32, device=device)
        result = torch.tensor([[1, 4, 3, 1]], dtype=torch.int64, device=device)
        ordinal = torch.tensor([2**54+3], dtype=torch.int64, device=device)
        metadata = torch.zeros((2, 16), dtype=torch.int32, device=device)
        metadata[1, 13:] = 1
        ready = torch.ones(1, dtype=torch.bool, device=device)
        remaining = torch.tensor([100], dtype=torch.int64, device=device)
        offsets = torch.zeros(2, dtype=torch.int32, device=device)
        eos = torch.zeros(1, dtype=torch.int64, device=device)
        clocks = torch.zeros(4, dtype=torch.int64, device=device)
        policy = torch.zeros(2, dtype=torch.uint64, device=device)
        host = torch.zeros(80, dtype=torch.int64, pin_memory=True)
        address = torch.tensor([int(check(rt.cudaHostGetDevicePointer(host.data_ptr(), 0))[0])],
                               dtype=torch.uint64, device=device)
        if fault in ("negative", "maximum", "minimum", "too_deep"):
            result[0, 0] = {"negative": -1, "maximum": 2**63-1,
                            "minimum": -2**63, "too_deep": 2}[fault]
        elif fault == "bonus": result[0, 1] = 16
        elif fault == "proposal": result[0, 2] = -1
        elif fault == "producer": result[0, 3] = 0
        elif fault == "budget": remaining.fill_(1)
        else: ordinal.fill_(2**63-1)
        before = remaining.clone()
        call = module.bind(
            entry, (meta, ctx, result, ordinal, metadata, ready, remaining, offsets, eos,
                    clocks, policy, address, *(CudaScalar("i32", x) for x in
                    (1, 1, 5, 16, 13, 16, 0)), CudaScalar("i64", 0), CudaScalar("i32", 3)),
            operator_id="audit.draft_journal_rejected_prefix", grid=(1, 1, 1), block=(128, 1, 1),
        )
        call.launch()
        torch.cuda.synchronize(device)
        assert meta.cpu().tolist()[3:] == [1, 2], fault
        assert host.numpy()[64+9] == 0, fault
        assert host.numpy().view("int32")[0] == 0, "invalid data was published before rank join"
        assert torch.equal(remaining, before), fault


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native draft control transition")
@pytest.mark.parametrize("depth,capacity,logical,stop_reason", [
    (1, 1, 1, "decision"),
    (3, 4, 4, None),
    (2, 8, 5, "eos"),
    (3, 256, 129, "step_limit"),
])
def test_native_draft_journal_matches_independent_integer_transition(
    depth, capacity, logical, stop_reason,
):
    """Prove the control boundary without executing or snapshotting a model.

    These are four distinct contracts, not a model x M x K x policy matrix:
    C1/decision exhaustion, every accepted-prefix length, padded EOS, and a
    second journal record with more live rows than native threads. Sampling
    probabilities and model numerical correctness belong to other proofs.
    """
    from pathlib import Path
    from cuda.bindings import runtime as rt
    from cubic_serve.executor.resident_window import check
    from cubic_serve.ops.cuda.native import CudaScalar
    from cubic_serve.ops.cuda.native_compiler import compile_native
    import cubic_serve.executor.draft_graph as graph_module

    device = torch.device("cuda", torch.cuda.current_device())
    _context_owner = torch.empty(1, device=device)
    source = (Path(graph_module.__file__).parents[1] /
              "ops/cuda/sampling/resident_draft_window.cu").read_bytes()
    module, entry = compile_native((("entry.cu", source),), device, "resident_draft_window")
    index, steps = (1, 2) if stop_reason == "step_limit" else (0, 3)
    context_stride, columns, axes_column = depth + 2, 16, 13
    vocabulary, sentinel = 4096, -97
    record_width = capacity * (2 * depth + 8) + 6
    meta = torch.tensor([logical, steps, index, 0, 0], dtype=torch.int32)
    ctx = torch.full((1 + capacity * context_stride,), -91, dtype=torch.int32)
    ctx[0] = logical
    result = torch.full((capacity, depth + 3), -1, dtype=torch.int64)
    ordinals = torch.full((capacity,), -1, dtype=torch.int64)
    metadata = torch.full((capacity * (depth + 1), columns), -37, dtype=torch.int32)
    remaining = torch.full((capacity, 3), -59, dtype=torch.int64)
    expected_host = torch.full((64 + steps * record_width,), sentinel, dtype=torch.int64)
    expected_host[:64] = 0
    expected_remaining = remaining.clone()
    offsets = torch.zeros(capacity + 1, dtype=torch.int32)
    eos = torch.zeros(1, dtype=torch.int64)
    for row in range(logical):
        owner, accepted = logical - row - 1, row % (depth + 1)
        position, ordinal = 100 + owner, 2**54 + owner
        proposals = [20 + owner + j for j in range(depth)]
        bonus, next_proposals = 1000 + owner, list(range(2000 + owner, 2000 + owner + depth))
        begin = 1 + row * context_stride
        ctx[begin:begin + context_stride] = torch.tensor([position, owner, *proposals])
        result[row] = torch.tensor([accepted, bonus, *next_proposals, 1])
        ordinals[row] = ordinal
        axis0, delta = [11 + owner, 31, 17], [1, 2, 0]
        for step in range(depth + 1):
            metadata[row * (depth + 1) + step, axes_column:] = torch.tensor(
                [a + step * d for a, d in zip(axis0, delta, strict=True)])
        decision_tokens = 0 if stop_reason == "decision" else -1
        remaining[row] = torch.tensor([1000, decision_tokens, 1])
        # Independent integer protocol: commit accepted draft prefix + bonus;
        # keep the proposal RNG ordinal distinct from the output-token count.
        committed = proposals[:accepted] + [bonus]
        payload = ([len(committed)] + committed + [-1] * (depth + 1 - len(committed))
                   + next_proposals + [position + len(committed), ordinal + accepted]
                   + [a + len(committed) * d for a, d in zip(axis0, delta, strict=True)] + [1])
        start = 64 + index * record_width + row * (2 * depth + 8)
        expected_host[start:start + len(payload)] = torch.tensor(payload)
        expected_remaining[row] = torch.tensor(
            [1000 - len(committed), decision_tokens, 0 if decision_tokens >= 0 else 1])
        if stop_reason == "eos" and row == logical - 1:
            eos[0] = bonus
            offsets[row + 1:] = 1
    host = expected_host.clone().pin_memory()
    host[64:] = sentinel
    readonly = [ctx, result, ordinals, metadata, offsets, eos]
    ctx_d, result_d, ordinals_d, metadata_d, offsets_d, eos_d = [t.to(device) for t in readonly]
    meta_d, remaining_d = meta.to(device), remaining.to(device)
    clocks = torch.zeros(steps * 4, dtype=torch.int64, device=device)
    policy = torch.zeros(2, dtype=torch.uint64, device=device)
    ready = torch.ones(capacity, dtype=torch.bool, device=device)
    address = torch.tensor([int(check(rt.cudaHostGetDevicePointer(host.data_ptr(), 0))[0])],
                           dtype=torch.uint64, device=device)
    call = module.bind(
        entry, (meta_d, ctx_d, result_d, ordinals_d, metadata_d, ready, remaining_d,
                offsets_d, eos_d, clocks, policy, address,
                *(CudaScalar("i32", x) for x in
                  (capacity, depth, context_stride, columns, axes_column, vocabulary, 0)),
                CudaScalar("i64", 0), CudaScalar("i32", 3)),
        operator_id="audit.draft_journal_integer_transition", grid=(1, 1, 1), block=(128, 1, 1),
    )
    call.launch()
    torch.cuda.synchronize(device)
    expected_meta = meta.clone()
    expected_meta[4] = int(stop_reason is not None)
    assert torch.equal(meta_d.cpu(), expected_meta)
    assert torch.equal(host, expected_host), "journal payload, inactive rows, or publication header"
    assert torch.equal(remaining_d.cpu(), expected_remaining)
    for actual, expected in zip(
        (ctx_d, result_d, ordinals_d, metadata_d, offsets_d, eos_d), readonly, strict=True,
    ):
        assert torch.equal(actual.cpu(), expected), "control step mutated a read-only binding"
    expected_clocks = torch.zeros(steps * 4, dtype=torch.int64)
    expected_clocks[index * 4 + 3] = clocks[index * 4 + 3].item()
    assert expected_clocks[index * 4 + 3] > 0
    assert torch.equal(clocks.cpu(), expected_clocks)
    assert torch.equal(policy.cpu(), torch.zeros(2, dtype=torch.uint64))


def test_composition_plan_follows_bound_calls_and_actual_compiled_choices():
    from cubic_serve.executor.text_graph import BoundTextBlock
    from cubic_serve.runtime.quantized_qualification import _bound_composition_plan

    class Binding:
        def __init__(self, image):
            self.image = image
        def project_artifacts(self):
            return (SimpleNamespace(artifact_digest=self.image),)
        def launch(self):
            raise AssertionError("fingerprinting must never launch an operator")

    def body(image):
        return SimpleNamespace(program=BoundTextBlock(torch.empty(1), (Binding(image).launch,)))

    assert _bound_composition_plan(body("A")) == _bound_composition_plan(body("A"))
    assert _bound_composition_plan(body("A")) != _bound_composition_plan(body("B"))
    with pytest.raises(TypeError):
        _bound_composition_plan(SimpleNamespace(program=object()))


@pytest.mark.parametrize("nested", (False, True))
def test_composition_plan_binds_single_artifact_even_with_identical_launch_signature(nested):
    from cubic_serve.executor.text_graph import BoundTextBlock
    from cubic_serve.runtime.quantized_qualification import _bound_composition_plan

    class Primitive:
        __slots__ = ("image",)

        def __init__(self, image):
            self.image = image

        @property
        def launch_signature(self):
            return "unchanged-geometry-and-arithmetic"

        def project_artifact(self):
            return SimpleNamespace(artifact_digest=self.image)

        def launch(self):
            raise AssertionError("fingerprinting must not execute the primitive")

    def body(image):
        call = Primitive(image).launch
        if nested:
            def launch(bound=call):
                bound()
            call = launch
        return SimpleNamespace(program=BoundTextBlock(torch.empty(1), (call,)))

    assert _bound_composition_plan(body("image-A")) == _bound_composition_plan(body("image-A"))
    assert _bound_composition_plan(body("image-A")) != _bound_composition_plan(body("image-B"))


def test_composition_receipt_checks_integrity_and_check_namespace(monkeypatch, tmp_path):
    import json
    from cubic_serve.runtime import performance as module

    monkeypatch.setenv("CUBIC_SERVE_CACHE_DIR", str(tmp_path))
    key = "b2s256:" + "a" * 64
    module._write_execution_audit_receipt(key, 0, "full-proof", check="mtp-composition")
    assert module._read_execution_audit_receipt(key, 0, check="mtp-composition") == "full-proof"
    assert module._read_execution_audit_receipt(key, 1, check="mtp-composition") is None
    assert module._read_execution_audit_receipt(key, 0, check="migration") is None
    path = next((tmp_path / "qualification").glob("*.json"))
    raw = json.loads(path.read_text())
    raw["evidence_id"] = "damaged"
    path.write_text(json.dumps(raw))
    assert module._read_execution_audit_receipt(key, 0, check="mtp-composition") is None


@pytest.mark.parametrize("peer_miss", [False, True])
def test_composition_hit_keeps_live_policies_owners_and_peer_miss_runs_full(
    monkeypatch, tmp_path, peer_miss,
):
    from cubic_serve.runtime import quantized_qualification as module

    monkeypatch.setenv("CUBIC_SERVE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(module.dist, "all_reduce", lambda value, **kw:
                        value.fill_(0) if peer_miss else None)
    monkeypatch.setattr(module.dist, "barrier", lambda: None)
    monkeypatch.setattr(torch.accelerator, "empty_host_cache", lambda: None)
    key = "b2s256:" + "b" * 64
    monkeypatch.setattr(module, "_draft_composition_receipt_key", lambda *args: key)
    calls = []
    def check(capacity, round_, window, *args, **kwargs):
        order = args[4]
        calls.append((window.requires_greedy, order, kwargs["sampling"]))
        return (capacity, order, window.requires_greedy)
    monkeypatch.setattr(module, "_qualify_draft_window_case", check)
    portfolio = {m: SimpleNamespace(body=SimpleNamespace(hidden=torch.empty(1)))
                 for m in (1, 2, 4, 8, 16, 32)}
    def rounds():
        return {(32, 2): SimpleNamespace(
            C=8, depth=2, stride=3, target=SimpleNamespace(_block_size=16),
            _resident_window=SimpleNamespace(requires_greedy=True, qualified=False),
            _candidate_window=SimpleNamespace(requires_greedy=False, qualified=False))}

    module.qualify_draft_windows(rounds(), portfolio, portfolio, None, None,
                                 rank=0, sampler_rank=0)
    full_count = len(calls)
    assert full_count == 3 * len(module._draft_window_audit_owner_maps(8, (6, 7, 8)))
    calls.clear()
    live = rounds()
    module.qualify_draft_windows(live, portfolio, portfolio, None, None,
                                 rank=0, sampler_rank=0)
    assert len(calls) == (full_count if peer_miss else 6)
    assert any(sampling is not None for _, _, sampling in calls)
    if not peer_miss:
        assert {order for _, order, _ in calls} == {tuple(range(8)), (5, 4, 3, 2, 1, 0)}
    assert live[(32, 2)]._resident_window.qualified and live[(32, 2)]._candidate_window.qualified
    def corrupt(*args, **kwargs):
        raise ValueError("live state corruption")
    monkeypatch.setattr(module, "_qualify_draft_window_case", corrupt)
    with pytest.raises(ValueError, match="live state corruption"):
        module.qualify_draft_windows(rounds(), portfolio, portfolio, None, None,
                                     rank=0, sampler_rank=0)


def test_composition_manifest_does_not_walk_checkpoint_or_allocator_factories():
    from cubic_serve.executor.text_graph import BoundTextBlock
    from cubic_serve.runtime.quantized import QuantizedWeights
    from cubic_serve.runtime.quantized_qualification import _bound_composition_plan

    class Binding:
        def project_artifacts(self):
            return (SimpleNamespace(artifact_digest="actual-bound-image"),)
        def launch(self):
            raise AssertionError("no execution while fingerprinting")
    def body():
        weights = object.__new__(QuantizedWeights)
        def forbidden():
            raise AssertionError("weight inventories belong to the checkpoint key, not execution")
        weights.project_artifacts = forbidden
        call = Binding().launch
        def launch(factory=weights, bound=call):
            bound()
        return SimpleNamespace(program=BoundTextBlock(torch.empty(1), (launch,)))
    assert _bound_composition_plan(body()) == _bound_composition_plan(body())


def test_composition_manifest_binds_scalar_closures_defaults_and_keyword_names():
    from functools import partial
    from cubic_serve.executor.text_graph import BoundTextBlock
    from cubic_serve.runtime.quantized_qualification import _bound_composition_plan

    def plan(call):
        return _bound_composition_plan(SimpleNamespace(
            program=BoundTextBlock(torch.empty(1), (call,))))
    def closure(scale):
        def launch():
            return scale
        return launch
    def defaults(scale):
        def launch(value=scale):
            return value
        return launch
    def keyword(**kwargs):
        return kwargs
    assert plan(closure(.5)) == plan(closure(.5))
    assert plan(closure(.5)) != plan(closure(.75))
    assert plan(defaults(.5)) != plan(defaults(.75))
    assert plan(partial(keyword, scale=2)) != plan(partial(keyword, partitions=2))


def test_composition_code_identity_is_independent_of_references_and_hash_seed():
    import os
    import subprocess
    import sys
    from cubic_serve.runtime.quantized_qualification import _composition_callback_code

    def launch(value):
        return value + 17 if value in {"A", "B", "C"} else value * 2
    code = launch.__code__
    first = _composition_callback_code(code)
    held = (code.co_consts, code.co_names, code.co_varnames, code.co_code)
    _composition_callback_code.cache_clear()
    assert _composition_callback_code(code) == first
    assert held[0] is code.co_consts
    assert _composition_callback_code(code.replace(co_filename="other-install.py",
                                                   co_firstlineno=999)) == first
    assert _composition_callback_code(code.replace(co_consts=(*code.co_consts, 99))) != first
    command = '''from cubic_serve.runtime.quantized_qualification import _composition_callback_code
from cubic_serve.ops.cuda.native import NativeLaunch
print(_composition_callback_code(NativeLaunch.launch.__code__))
'''
    results = [subprocess.check_output([sys.executable, "-c", command],
                                      env=os.environ | {"PYTHONHASHSEED": seed}, timeout=15)
               for seed in ("1", "42")]
    assert results[0] == results[1]


def test_pending_depth_views_share_owner_state_and_restore_one_checkpoint():
    import torch
    from cubic_serve.executor.draft_graph import DraftPendingPool
    metadata = [("r", 0, 3, None, 0, None, (4, 5, 6))]
    pool = DraftPendingPool(3, 1, metadata, torch.zeros(3, 1, 1, 2),
                            torch.zeros(3, 1, 1, 2, dtype=torch.int64),
                            torch.zeros(3, 1, dtype=torch.int64), {"rnn": torch.tensor([[7.]])})
    lower = pool.prefix(1)
    assert lower.metadata is metadata and lower.backup is pool.backup
    assert lower.is_prefix_of(pool)
    alien = pool.prefix(1)
    alien.tokens = alien.tokens.clone()
    assert not alien.is_prefix_of(pool)
    lower.tokens[0, 0] = 11
    assert pool.tokens[0, 0] == 11
    state = {"rnn": torch.tensor([[99.]])}
    lower.restore(SimpleNamespace(state_fields=state),
                  (SimpleNamespace(state_source_slot=0, request_id="r"),))
    assert metadata == [None] and state["rnn"].item() == 7
    assert pool.prefix(3) is pool
    with pytest.raises(ValueError):
        pool.prefix(4)


def test_depth_portfolio_selects_k1_with_shared_maximum_pending_storage():
    from cubic_serve.runtime import performance as p
    from cubic_serve.runtime.continuous_batch import BatchCommand, BatchRow
    rows = tuple(BatchRow(
        "r", 9 if i == 0 else -1, 10 + i, 0, 0, 0, 0, (0, 1),
        segment_length=2 if i == 0 else 0, sample_ordinal=20 + i,
        mtp_position=10 + i, mtp_page_destination=0, mtp_page_table=(0, 1), is_decode=True,
    ) for i in range(2))
    root = rows[0]
    key = ("r", 10, 9, root.sampling, 20, None)
    calls = []
    pool = SimpleNamespace(depth=3, metadata=[(*key, (11,))],
                           restore=lambda *args: pytest.fail("valid K1 pending must not restore"))
    provider = SimpleNamespace(_pending_draft_pool=pool, reserve_state_slots=lambda _: None)
    def run(value, segments):
        calls.append(value)
        return ((1, 23, 24, 1),)
    lower = SimpleNamespace(depth=1, C=1, run=run,
                            active_draft_window=SimpleNamespace(qualified=True))
    maximum = SimpleNamespace(depth=3)
    executor = SimpleNamespace(_pipelined_draft_round=maximum,
                               _pipelined_draft_rounds={1: lower, 3: maximum},
                               _block_size=16, replay_elapsed_ms=lambda: 5.)
    result = p._execute_pipelined_draft_batch(
        BatchCommand(0, 2, rows), executor, {1: provider}, None,
        SimpleNamespace(max_sequence_tokens=32), (1, 2, 4), rank=0, sampler_rank=0)
    assert executor._pipelined_draft_round is lower
    assert result.resident_draft_depth == 1 and not result.draft_ingress
    assert tuple(row.token_id for row in calls[0]) == (9, 11)
    assert result.segment_results[0].output_token_ids == (11, 23)
    assert pool.metadata[0] == ("r", 12, 23, root.sampling, 22, None, (24,))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='shared review CUDA graph')
def test_ordinary_and_temporal_parents_clone_one_sealed_review(monkeypatch):
    from cubic_serve.executor import draft_graph
    from cuda.bindings import runtime as rt
    from cubic_serve.executor.resident_window import check
    from cubic_serve.guardrail.graph_observer import graph_nodes, graph_identity

    # This single-process proof isolates graph cloning from the collective ABI.
    monkeypatch.setattr(draft_graph, 'barrier', lambda: None)
    round_ = CapturedDraftRound.__new__(CapturedDraftRound)
    round_._capture_candidates = round_._capture_budget = False
    round_._capture_review_only = round_._capture_target_constraints = False
    round_._review_graphs, round_._review_capture_open = {}, True
    round_._review_graph_pool = torch.cuda.graph_pool_handle()
    round_.target = SimpleNamespace(_sampling=SimpleNamespace(constraint_bound=False))
    round_.head = SimpleNamespace(_sampling=SimpleNamespace(constraint_bound=False))
    x = torch.tensor([17], device='cuda:0', dtype=torch.int64)
    y = torch.zeros_like(x)
    round_.finish_review = lambda: y.copy_(x + 1)
    round_.finish_review()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    round_.capture_review_graph(stream)
    round_._review_capture_open = False
    core = round_.captured_review_graph()
    parents = []
    for _ in range(2):
        parent = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(parent, stream=stream):
            round_.append_review_graph()
        parent.instantiate()
        parents.append(parent)
    # Both parents contain a clone of the actual same source graph. Tensor
    # temporaries retain identical pointers; they are not recaptured bodies.
    expected = graph_identity(core.raw_cuda_graph())
    for parent, value in zip(parents, (29, 41), strict=True):
        nodes = graph_nodes(parent.raw_cuda_graph())
        assert len(nodes) == 1
        child = check(rt.cudaGraphChildGraphNodeGetGraph(nodes[0]))[0]
        actual = graph_identity(child)
        mapping = {n: int(check(rt.cudaGraphNodeFindInClone(rt.cudaGraphNode_t(n), child))[0])
                   for n, _, _ in expected[0]}
        assert tuple(sorted((mapping[n], kind, args) for n, kind, args in expected[0])) == actual[0]
        assert tuple(sorted((mapping[a], mapping[b], *rest)
                            for a, b, *rest in expected[1])) == actual[1]
        x.fill_(value)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            parent.replay()
        stream.synchronize()
        assert y.item() == value + 1
    with pytest.raises(RuntimeError, match='closed after construction'):
        round_.capture_review_graph(stream)
    with pytest.raises(RuntimeError, match='startup parent capture'):
        round_.append_review_graph()
    round_._capture_candidates = True
    with pytest.raises(RuntimeError, match='no sealed graph'):
        round_.captured_review_graph()
    for parent in parents:
        parent.reset()
    core.reset()


@pytest.mark.skipif(os.environ.get('CUBIC_TEST_QUANT_CUDA') != '1',
                    reason='explicit CUDA checkpoint row-map proof')
@pytest.mark.parametrize('depth', [1, 2, 3])
@pytest.mark.parametrize('owners,logical', [
    (1, 1), (4, 1), (4, 3), (4, 4),
    (256, 1), (256, 129), (256, 255), (256, 256),
])
@pytest.mark.parametrize('permuted', [False, True])
@pytest.mark.parametrize('shared', [False, True])
def test_native_provider_row_maps_and_feedback_preserve_checkpoint_ownership(
        depth, owners, logical, permuted, shared):
    """Every native map field, continuation and private padding has a CPU oracle.

    This integer/byte transport proof does not qualify model arithmetic or
    authorize dropping padding, state, or end-to-end composition witnesses.
    """
    import triton
    from cubic_serve.executor.draft_graph import (
        align_feedback, align_meta, autoregressive_meta,
    )

    device = torch.device('cuda:0')
    layers, page, guard = 3, 16, 17
    tables = max(4, math.ceil((page + owners + 2 * depth) / page))
    stride = depth + 1
    physical = triton.next_power_of_2(owners * stride)
    columns, axes = 24, 18
    context_width = 2 + depth + tables
    checkpoint_base = max(17, owners + 1)
    scratch_state = max(73, checkpoint_base + 3 * owners * depth + 1)
    scratch_page = max(211, 37 + tables * owners + 1)
    vocabulary = max(128, 73 + owners + 31 * depth)
    start, stop = 16, 96
    order = list(reversed(range(owners))) if permuted else list(range(owners))
    ctx = torch.zeros(1 + owners * context_width, dtype=torch.int32)
    ctx[0] = logical
    context = ctx[1:].view(owners, context_width)
    metadata = torch.full((physical, columns), -937, dtype=torch.int32)
    for r in range(owners):
        context[r, :2] = torch.tensor([page - 2 + r, order[r]])
        context[r, 2:2 + depth] = torch.arange(depth) * 31 + 13 + r
        context[r, 2 + depth:] = torch.arange(tables) + 37 + tables * order[r]
        for j in range(stride):
            metadata[r * stride + j, axes:axes + 3] = torch.tensor(
                [101 + 7 * r + (axis + 1) * j for axis in range(3)])
    original_metadata = metadata.clone()
    ctx_gpu, meta_gpu = ctx.to(device), metadata.to(device)
    checkpoint_cs = 3 + depth
    checkpoint = torch.zeros(1 + owners * checkpoint_cs, dtype=torch.int32)
    checkpoint[0] = logical
    private = checkpoint[1:].view(owners, checkpoint_cs)[:, 3:]
    private[:] = checkpoint_base + 3 * torch.arange(owners * depth).flip(0).view(owners, depth)
    checkpoint_gpu = checkpoint.to(device)
    extra = dict(checkpoint_ctx=checkpoint_gpu, CHECKPOINT_CS=checkpoint_cs,
                 PADDING_ROWS=1, PADDING_PAGES=1) if shared else {}

    def guarded(shape, dtype=torch.int32):
        count = math.prod(shape)
        storage = torch.full((count + 2 * guard,), -937, dtype=dtype, device=device)
        return storage, storage[guard:-guard].view(shape)

    def check_storage(storage):
        assert bool((storage[:guard] == -937).all())
        assert bool((storage[-guard:] == -937).all())

    for accepted in [*range(depth + 1), None]:
        decisions = torch.tensor([
            [((r % (depth + 1)) if accepted is None else accepted), 72 + r]
            for r in range(owners)], dtype=torch.int64)
        decision_gpu = decisions.to(device)
        for step in (None, *range(1, depth)):
            rows = physical if step is None else owners
            vectors = [guarded((rows,)) for _ in range(8)]
            ids, mask, source_page, dest_page, offset, lengths, segments, activity = (
                item[1] for item in vectors)
            position_storage, positions = guarded((3, rows))
            table_storage, table = guarded((rows, tables))
            map_buffers = [guarded((2, layers, rows)) for _ in range(3)]
            maps = tuple(item[1] for item in map_buffers)
            buffers=SimpleNamespace(token_ids=ids,position_ids=positions,page_source=source_page,
                page_destination=dest_page,token_offsets=offset,sequence_lengths=lengths,
                segment_lengths=segments,row_activity=activity,state_slot_maps=maps,page_table=table)
            body=SimpleNamespace(buffers=buffers,embedding_mask=mask,
                embedding_span=SimpleNamespace(start=start,stop=stop),
                arena=SimpleNamespace(layout=SimpleNamespace(block_size_tokens=page)))
            target=SimpleNamespace(_metadata_device=meta_gpu,_metadata_columns=columns,
                _position_axes_slice=slice(axes,axes+3),vocab_size=vocabulary)
            fixture=SimpleNamespace(ctx=ctx_gpu,decisions=decision_gpu,target=target,
                align=SimpleNamespace(body=body),head=SimpleNamespace(body=body),M=physical,C=owners,
                T=tables,CS=context_width,depth=depth,scratch_page=scratch_page,scratch_state=scratch_state,
                _checkpoint_arguments=lambda:extra)
            if step is None:
                CapturedDraftRound.bind_provider_alignment_metadata(fixture)
            else:
                proposals=torch.tensor([25+r for r in range(owners)],device=device)
                fixture.head._token_ids=proposals
                CapturedDraftRound.bind_provider_proposal_metadata(fixture,step)
            expected_vectors = [[] for _ in vectors]
            expected_positions, expected_tables, expected_maps = [], [], []
            for row in range(rows):
                r, j = divmod(row, stride) if step is None else (row, step)
                owned = r < logical
                a = int(decisions[r, 0]) if owned else -1
                live = owned and (j <= a if step is None else True)
                owner = order[r] if owned else 0
                root = int(context[r, 0]) if owned else 0
                token = (int(context[r, 2 + j]) if j < a else int(decisions[r, 1])) \
                    if step is None and live else 25 + r if step is not None else 0
                local = live and start <= token < stop
                cache_position = root + (j + 1 if step is None else a + 1 + step) if live else 0
                page_id = int(context[r, 2 + depth + cache_position // page]) \
                    if live else scratch_page if shared else scratch_page + row
                source = owner if live and (step is None or step == 1) else \
                    scratch_state + row if step is None else \
                    scratch_state + r * stride + (step - 2 if live else 0)
                destination = (owner if live and j == a else scratch_state + row) \
                    if step is None else scratch_state + r * stride + step - 1
                if shared:
                    source = (owner if step is None or step == 1 else int(private[r, step - 2])) if live else scratch_state
                    destination = ((owner if j == a else int(private[r, j])) if step is None
                                   else int(private[r, step - 1])) if live else scratch_state
                values = (token - start if local else 0, int(local), page_id, page_id,
                          cache_position % page, cache_position + 1 if live else 1,
                          a + 1 if live and j == 0 and step is None else int(live) if step is not None else 0,
                          int(live))
                for column, value in zip(expected_vectors, values, strict=True):
                    column.append(value)
                expected_positions.append([
                    101 + 7 * r + (axis + 1) * (j + 1 if step is None else a + 1 + step)
                    if live else 0 for axis in range(3)])
                expected_tables.append(context[r, 2 + depth:].tolist() if live else
                                       [scratch_page if shared else scratch_page + row] * tables)
                expected_maps.append([[source * layers + layer for layer in range(layers)],
                                      [destination * layers + layer for layer in range(layers)]])
            for (storage, tensor), values in zip(vectors, expected_vectors, strict=True):
                assert torch.equal(tensor.cpu(), torch.tensor(values, dtype=tensor.dtype))
                check_storage(storage)
            assert torch.equal(positions.cpu(), torch.tensor(expected_positions).T)
            assert torch.equal(table.cpu(), torch.tensor(expected_tables))
            for storage, tensor in map_buffers:
                assert torch.equal(tensor.cpu(), torch.tensor(expected_maps).permute(1, 2, 0))
                check_storage(storage)
            if step is not None:
                assert torch.equal(proposals.cpu(), torch.arange(owners) + 25)
            check_storage(position_storage)
            check_storage(table_storage)
        assert torch.equal(ctx_gpu.cpu(), ctx)
        assert torch.equal(meta_gpu.cpu(), original_metadata)
        assert torch.equal(decision_gpu.cpu(), decisions)

        width = 259
        target = torch.randint(0, 65536, (physical, width), device=device,
                               dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        original = target.cpu()
        output_storage, output = guarded((physical, width), torch.bfloat16)
        align_feedback[(physical, triton.cdiv(width, 256))](
            ctx_gpu, decision_gpu, target, output, width, context_width, 256, depth)
        expected = torch.zeros_like(target, device='cpu')
        for row in range(physical):
            r, j = divmod(row, stride)
            if r < logical and j <= int(decisions[r, 0]):
                expected[row] = original[r * stride + min(j, int(decisions[r, 0]))]
        assert torch.equal(output.cpu().view(torch.uint8), expected.view(torch.uint8))
        assert torch.equal(target.cpu().view(torch.uint8), original.view(torch.uint8))
        check_storage(output_storage)


@pytest.mark.skipif(os.environ.get('CUBIC_TEST_QUANT_CUDA') != '1', reason='explicit CUDA byte transport proof')
def test_shared_pending_and_feedback_preserve_all_unowned_bytes():
    from cubic_serve.executor.draft_graph import pending_copy, publish_feedback
    from cubic_serve.runtime.cache.shared_pool import SharedCacheLayout, SharedCacheField
    layout = SharedCacheLayout((SharedCacheField('guard',torch.int64,(4,)),
        SharedCacheField('scores',torch.float32,(2,2,17)),
        SharedCacheField('ids',torch.int64,(2,2,17)),
        SharedCacheField('tokens',torch.int64,(2,)),
        SharedCacheField('feedback',torch.bfloat16,(259,))))
    owners, rows = 7, 4
    raw = torch.full((layout.entry_bytes*owners,),0x31,dtype=torch.uint8,device='cuda')
    fields=layout.views(raw,owners)
    ctx=torch.tensor([3,0,6,0,2,0,4,0,999],dtype=torch.int32,device='cuda')
    score=torch.arange(2*rows*17,dtype=torch.float32,device='cuda').view(2,rows,17)
    ids=torch.arange(2*rows*17,dtype=torch.int64,device='cuda').view(2,rows,17)+2**54+1
    tokens=torch.arange(rows,dtype=torch.int64,device='cuda')+2**54+3
    for depth in range(2):
        before=raw.cpu();expected=before.clone();ref=layout.views(expected,owners)
        for r,owner in enumerate((6,2,4)):
            ref['scores'][owner,depth]=score[:,r].cpu()
            ref['ids'][owner,depth]=ids[:,r].cpu()
            ref['tokens'][owner,depth]=tokens[r].cpu()
        ps=fields['scores'][:,depth].permute(1,0,2)
        pi=fields['ids'][:,depth].permute(1,0,2)
        pt=fields['tokens'][:,depth]
        kwargs=dict(SCORE_OWNER=ps.stride(1),SCORE_TP=ps.stride(0),ID_OWNER=pi.stride(1),
                    ID_TP=pi.stride(0),TOKEN_OWNER=pt.stride(0))
        pending_copy[(rows,)](ctx,score,ids,tokens,ps,pi,pt,2,rows,owners,2,17,32,False,**kwargs)
        assert torch.equal(raw.cpu(),expected)
        outscore=torch.empty_like(score);outids=torch.empty_like(ids);outtokens=torch.empty_like(tokens)
        pending_copy[(rows,)](ctx,outscore,outids,outtokens,ps,pi,pt,2,rows,owners,2,17,32,True,**kwargs)
        assert torch.equal(outscore[:,:3],score[:,:3])
        assert torch.equal(outids[:,:3],ids[:,:3])
        assert torch.equal(outtokens[:3],tokens[:3])
        assert torch.isnan(outscore[:,3]).all() and (outids[:,3]==-1).all() and outtokens[3]==-1
    activity=torch.tensor([1,0,1,0],dtype=torch.int32,device='cuda')
    indices=torch.tensor([4,999,1,999],dtype=torch.int64,device='cuda')
    source=torch.arange(rows*259,device='cuda').view(rows,259).to(torch.bfloat16)
    expected=raw.cpu();refs=layout.views(expected,owners)
    refs['feedback'][4]=source[0].cpu();refs['feedback'][1]=source[2].cpu()
    def launch():
        publish_feedback[(rows,2)](activity,indices,source,fields['feedback'],259,1,fields['feedback'].stride(0),256)
    launch()
    graph=torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):launch()
        graph.replay()
        assert torch.equal(raw.cpu(),expected)
    finally:graph.reset()
