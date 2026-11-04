import os
import threading
from dataclasses import replace
from multiprocessing import shared_memory

import pytest

from cubic_serve.executor.resident_window import resident_audit_rows
from cubic_serve.runtime.continuous_batch import (
    RESIDENT_STEP_CAPACITY,
    BatchProtocolError,
    ContinueBatchCommand,
    InputEncoderExecutionLease,
    SharedPublicationInterrupt,
    completed_resident_pages,
    deserialize_worker_command,
    resident_gpu_step_intervals,
    serialize_worker_command,
)


class _NoncanonicalWireString(str):
    pass


def test_worker_wire_reuses_only_immutable_definitions_within_one_command(monkeypatch):
    from dataclasses import FrozenInstanceError
    from cubic_serve.contracts.request import ReasoningControl, StructuredOutputMeta
    from cubic_serve.generation.structured_sampling import ConstraintBinding
    from cubic_serve.runtime import continuous_batch as wire

    schema = StructuredOutputMeta.json_schema({'type': 'integer'})
    formatted = StructuredOutputMeta.json_schema(
        {'type': 'integer'}, options={'disable_any_whitespace': True})
    grammar = StructuredOutputMeta.grammar('root ::= "a"')
    # JSON Schema allows unknown keywords; the same canonical bytes have
    # different semantics when interpreted as a grammar declaration.
    schema_with_grammar_bytes = StructuredOutputMeta(
        schema.kind, grammar.canonical_payload)
    definitions = (schema, schema, formatted, grammar, schema_with_grammar_bytes)
    rows = tuple(wire.BatchRow(str(i), 2, 32, i, i, i, i, (i, i, i),
                               is_decode=True) for i in range(len(definitions)))
    bindings = tuple(ConstraintBinding(
        str(i), meta, i, 100-i,
        reasoning_control=ReasoningControl((3,), (4,), i+1, True),
    ) for i, meta in enumerate(definitions))
    command = wire.BatchCommand(7, 8, rows, bindings)
    payload = serialize_worker_command(command)
    calls = []

    def construct(*args):
        calls.append(args)
        return StructuredOutputMeta(*args)

    monkeypatch.setattr(wire, 'StructuredOutputMeta', construct)
    first = deserialize_worker_command(payload)
    second = deserialize_worker_command(payload)
    assert first == second == command
    assert len(calls) == 2 * len(set(definitions))
    assert first.constraints[0].meta is first.constraints[1].meta
    assert first.constraints[0] is not first.constraints[1]
    assert first.constraints[0].reasoning_control != first.constraints[1].reasoning_control
    assert first.constraints[0].meta is not second.constraints[0].meta
    assert len({id(b.meta) for b in first.constraints}) == len(set(definitions))
    with pytest.raises(FrozenInstanceError):
        first.constraints[0].meta.canonical_payload = 'null'


def test_worker_wire_definition_reuse_cannot_hide_invalid_equal_typed_payload():
    import pickle
    from cubic_serve.contracts.request import StructuredOutputMeta
    from cubic_serve.generation.structured_sampling import ConstraintBinding
    from cubic_serve.runtime import continuous_batch as wire

    meta = StructuredOutputMeta.json_schema({'type': 'integer'})
    rows = tuple(wire.BatchRow(str(i), 2, 0, i, i, i, i, (i,),
                               is_decode=True) for i in range(2))
    command = wire.BatchCommand(7, 2, rows, tuple(
        ConstraintBinding(str(i), meta, 0, 10) for i in range(2)))
    encoded = list(pickle.loads(serialize_worker_command(command)))
    bindings = list(encoded[5])
    invalid = list(bindings[1])
    invalid[2] = _NoncanonicalWireString(invalid[2])
    bindings[1] = tuple(invalid)
    encoded[5] = tuple(bindings)
    with pytest.raises(BatchProtocolError):
        deserialize_worker_command(pickle.dumps(tuple(encoded)))


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_draft_window_fixture_preserves_every_rejectable_prefix(depth):
    from types import SimpleNamespace

    from cubic_serve.runtime.quantized_qualification import _draft_window_audit_rows

    stride = depth + 1
    # Deliberately use an independent scratch arena offset, not durable owner+M.
    scratch = tuple(range(73, 73 + 2 * stride))
    round_ = SimpleNamespace(depth=depth, stride=stride, C=2, target=SimpleNamespace(
        body=SimpleNamespace(arena=SimpleNamespace(shared_storage=None)), _block_size=16, _scratch_states=scratch, vocab_size=256))
    pending = [None, ("ignored", 15, 203, None, 9, None, tuple(range(17, 17 + depth)))]
    rows = _draft_window_audit_rows(round_, pending, 2, 2)
    for owner in range(2):
        segment = rows[owner * stride:(owner + 1) * stride]
        assert {row.state_source_slot for row in segment} == {owner}
        assert segment[-1].state_destination_slot == owner
        for accepted in range(depth):
            checkpoint = segment[accepted].state_destination_slot
            assert checkpoint not in (0, 1)
            assert checkpoint != segment[-1].state_destination_slot
            assert sum(row.state_destination_slot == checkpoint for row in rows) == 1
        assert [row.segment_length for row in segment] == [stride] + [0] * depth
        assert [row.mtp_position for row in segment] == list(range(
            0 if owner == 0 else 15, (0 if owner == 0 else 15) + stride))
    assert rows[stride].token_id == 203
    assert tuple(row.token_id for row in rows[stride + 1:]) == tuple(range(17, 17 + depth))
    assert rows[stride].sample_ordinal == 9
    assert rows[stride + 1].page_destination == 3  # Cross-page review ownership.


def test_draft_window_fixture_seed_uses_declared_vocabulary():
    from types import SimpleNamespace

    from cubic_serve.runtime.quantized_qualification import _draft_window_audit_rows

    round_ = SimpleNamespace(depth=1, stride=2, C=1, target=SimpleNamespace(
        body=SimpleNamespace(arena=SimpleNamespace(shared_storage=None)), _block_size=16, _scratch_states=(7, 8), vocab_size=2))
    rows = _draft_window_audit_rows(round_, [None], 1, 1)
    assert rows[0].token_id == 1
    assert all(0 <= row.token_id < round_.target.vocab_size for row in rows)


def test_draft_window_fixture_respects_typed_initial_values_and_protected_rows():
    from types import SimpleNamespace

    import torch

    from cubic_serve.runtime.cache import CacheArena, CacheArenaSpec, KVLayout, StateSlotSpec
    from cubic_serve.runtime.quantized_qualification import _initialize_draft_window_fixture

    layout = KVLayout((0,), 4, 1, 2)
    target = CacheArena(CacheArenaSpec("target", 2, 2, (
        StateSlotSpec("history", torch.int64, (2,), initial_value=19),
    ), scratch_capacity=2), layout, "cpu", rank_id=0)
    provider = CacheArena(CacheArenaSpec("provider", 2, 2, (
        StateSlotSpec("history", torch.int64, (2,), initial_value=23),
    ), scratch_capacity=2), layout, "cpu", rank_id=0)
    target.states["history"].fill_(77)
    provider.states["history"].fill_(78)
    pool = SimpleNamespace(backup={"history": torch.full((2, 2), 79, dtype=torch.int64)})
    target_states, owners = torch.tensor([0, 2]), torch.tensor([0])
    feedback = torch.full((4, 2), 81, dtype=torch.bfloat16)
    fields = [("target.history", target.states["history"], target_states),
              ("provider.history", provider.states["history"], owners),
              ("backup.history", pool.backup["history"], owners),
              ("feedback", feedback, target_states)]
    _initialize_draft_window_fixture(fields, target, provider, target_states, owners, pool)
    assert target.states["history"].tolist() == [[19, 19], [77, 77], [19, 19], [77, 77]]
    assert provider.states["history"].tolist() == [[23, 23], [78, 78], [78, 78], [78, 78]]
    assert pool.backup["history"].tolist() == [[23, 23], [79, 79]]
    assert feedback.tolist() == [[0, 0], [81, 81], [0, 0], [81, 81]]



@pytest.mark.parametrize("capacity", [1, 2, 4, 8, 16])
def test_draft_composition_covers_full_padding_and_permuted_ownership(capacity):
    from types import SimpleNamespace
    from cubic_serve.runtime.quantized_qualification import (
        _draft_window_audit_owner_maps, _draft_window_audit_rows,
    )
    cases = _draft_window_audit_owner_maps(capacity)
    assert tuple(range(capacity)) in cases
    assert tuple(reversed(range(capacity))) in cases
    assert (0,) in cases
    assert tuple(range(capacity // 2 + 1)) in cases
    if capacity > 2:
        assert any(1 < len(order) < capacity for order in cases)
    round_ = SimpleNamespace(depth=2, stride=3, C=capacity, target=SimpleNamespace(
        body=SimpleNamespace(arena=SimpleNamespace(shared_storage=None)), _block_size=16, _scratch_states=tuple(range(100,100+3*capacity)), vocab_size=256))
    pending = [(str(owner), owner+1, 200+owner, None, 11+owner, None, (51+owner, 71+owner))
               for owner in range(capacity)]
    for order in cases:
        rows = _draft_window_audit_rows(round_, pending, len(order), 2, owner_order=order)
        for physical_owner, owner in enumerate(order):
            segment = rows[3*physical_owner:3*physical_owner+3]
            assert {r.state_source_slot for r in segment} == {owner}
            assert segment[-1].state_destination_slot == owner
            assert [r.state_destination_slot for r in segment[:-1]] == [100+3*physical_owner,101+3*physical_owner]
            assert segment[0].token_id == 200+owner
            assert segment[0].sample_ordinal == 11+owner
            assert segment[0].page_table == (2*owner,2*owner+1)


def test_draft_composition_rejects_nonpermutation_before_constructing_rows():
    from cubic_serve.runtime.quantized_qualification import _draft_window_audit_rows
    for order in [(0,0),(1,2),(0,),[0,1]]:
        with pytest.raises(ValueError, match="permute"):
            _draft_window_audit_rows(None, None, 2, 1, owner_order=order)


def test_draft_fixture_initializes_private_provider_checkpoints_without_touching_other_rows():
    from types import SimpleNamespace
    import torch
    from cubic_serve.runtime.cache import CacheArena, CacheArenaSpec, KVLayout, StateSlotSpec
    from cubic_serve.runtime.quantized_qualification import _initialize_draft_window_fixture
    layout=KVLayout((0,),4,1,2)
    spec=lambda name,initial:CacheArenaSpec(name,2,2,(StateSlotSpec("history",torch.int64,(2,),initial_value=initial),),scratch_capacity=2)
    target=CacheArena(spec("target",19),layout,"cpu",rank_id=0)
    provider=CacheArena(spec("provider",23),layout,"cpu",rank_id=0)
    target.states["history"].fill_(77);provider.states["history"].fill_(78)
    owners=torch.tensor([0]);private=torch.tensor([0,2]);pool=SimpleNamespace(backup={"history":torch.full((2,2),79,dtype=torch.int64)})
    fields=[("target.history",target.states["history"],private),("provider.history",provider.states["history"],private),("backup.history",pool.backup["history"],owners)]
    _initialize_draft_window_fixture(fields,target,provider,private,owners,pool,provider_states=private)
    assert provider.states["history"].tolist()==[[23,23],[78,78],[23,23],[78,78]]
    assert pool.backup["history"].tolist()==[[23,23],[79,79]]


@pytest.mark.parametrize("dtype", ["bfloat16", "float32", "int64", "bool"])
def test_draft_fixture_host_snapshot_is_immutable_and_restores_only_named_rows(dtype):
    import torch
    from cubic_serve.runtime.quantized_qualification import (
        _snapshot_draft_fixture_fields, _restore_draft_fixture_fields,
    )
    tensor=torch.arange(24).reshape(4,2,3).to(getattr(torch,dtype))
    indices=torch.tensor([3,1]);fields=[("state",tensor,indices)]
    expected=tensor.clone();saved=_snapshot_draft_fixture_fields(fields)
    assert saved["state"].device.type=="cpu"
    tensor[3].fill_(0);tensor[1].fill_(0);tensor[0].fill_(1)
    assert torch.equal(saved["state"],expected[indices])
    _restore_draft_fixture_fields(fields,saved)
    assert torch.equal(tensor[indices],expected[indices])
    assert torch.equal(tensor[0],torch.ones_like(tensor[0]))
    assert torch.equal(tensor[2],expected[2])


def test_provider_prompt_grants_are_independent_and_wire_roundtrip():
    command=ContinueBatchCommand(2,1,4,("a","b"),max_steps=2,eos_token_ids=((),()),
        prompt_remaining=(9,9),prompt_tokens=((10,11,12,13),(20,21)),
        prompt_positions=(((3,5,7),(4,8,9),(5,11,13),(6,14,17)),((3,3,3),(4,4,4))),
        owner_lengths=(2,1),source_pages=(5,11),
        page_program=(((5,6,1,0),(6,6,1,0),(11,11,1,0)),((6,7,1,0),(7,7,1,0),(11,11,1,0))),
        provider_tables=((19,23),(29,31)))
    args=dict(previous_batch_id=1,captured_m=4,request_ids=command.request_ids,offsets=(1,2),page_counts=(2,2),block_size=16)
    assert command.resident_updates(**args)=={}
    assert deserialize_worker_command(serialize_worker_command(command))==command
    for tables in (((19,19),(29,31)),((19,-1),(29,31)),((19,23),),([19,23],(29,31))):
        assert replace(command,provider_tables=tables).resident_updates(**args) is None
    # Maintenance never crosses from a pure prompt program into target decode.
    final=replace(command,prompt_remaining=(9,2))
    assert final.resident_updates(**args) is None


def test_draft_prompt_segments_require_matching_explicit_provider_positions():
    from cubic_serve.runtime.continuous_batch import BatchRow,owned_segment_rows
    rows=tuple(BatchRow("a",10+i,4+i,0,0,2,3,(1,3),segment_length=2 if i==0 else 0,
                       mtp_position=4+i,mtp_page_destination=7,mtp_page_table=(5,7)) for i in range(2))
    assert owned_segment_rows(rows)==()
    assert owned_segment_rows(rows,allow_draft_prompt=True)==(2,)
    for bad in (replace(rows[1],mtp_position=9),replace(rows[1],is_decode=True),replace(rows[1],input_feature=b"x")):
        assert owned_segment_rows((rows[0],bad),allow_draft_prompt=True)==()


@pytest.mark.parametrize("fault", [False,True])
def test_prompt_page_program_rolls_back_both_heaps_and_preserves_live_ownership(fault):
    import heapq
    from types import SimpleNamespace
    from cubic_serve.runtime.performance import _ProcessController,_ScheduledRequest
    from cubic_serve.runtime.continuous_batch import BatchRow
    record=_ScheduledRequest(SimpleNamespace(input_ids=(1,)*100),None,0.,prefill_cursor=1,
        state_slot=0,pages=[0],mtp_pages=[7],state_initialized=True)
    target_heap=list(range(10,40));provider_heap=list(range(50,70));calls=[]
    controller=SimpleNamespace(_num_speculative_tokens=1,_block_size=2,max_sequence_tokens=100,
        _shared_state_slots=set(),_shared_pages={},_free_pages=target_heap,_free_mtp_pages=provider_heap,
        _next_input_position=lambda r:r.prefill_cursor)
    def reserve(r,width,*,running):
        first=r.prefill_cursor//2;last=(r.prefill_cursor+width-1)//2
        new=[]
        while len(r.pages)<=last:
            page=heapq.heappop(controller._free_pages);r.pages.append(page);new.append(page)
        while len(r.mtp_pages)<=last:r.mtp_pages.append(heapq.heappop(controller._free_mtp_pages))
        calls.append(1)
        if fault and len(calls)==2:raise RuntimeError("injected allocation failure")
        table=list(r.pages);dest={j:heapq.heappop(controller._free_pages) for j in range(first,last+1)}
        for j,page in dest.items():table[j]=page
        return tuple(BatchRow("a",1,p,0,0,r.pages[p//2],dest[p//2],tuple(table),new_pages=tuple(new) if i==0 else (),segment_length=width if i==0 else 0,mtp_position=p,mtp_page_destination=r.mtp_pages[p//2],mtp_page_table=tuple(r.mtp_pages)) for i,p in enumerate(range(r.prefill_cursor,r.prefill_cursor+width)))
    controller._reserve_segment_locked=reserve
    run=lambda:_ProcessController._stage_segment_page_program_locked(controller,(record,),(2,),2)
    if fault:
        with pytest.raises(RuntimeError,match="injected"):run()
        assert controller._free_pages is target_heap and controller._free_mtp_pages is provider_heap
        assert record.mtp_pages==[7] and record.pending_cow_pages==()
    else:
        assert len(run())==2
        assert record.mtp_pages==[7,50,51]
        assert sorted(controller._free_mtp_pages)==list(range(52,70))
        assert record.pending_cow_pages
    assert record.pages==[0] and record.prefill_cursor==1 and record.state_slot==0

def test_prompt_program_wire_preserves_truncated_prompt_and_distinct_position_axes():
    command = ContinueBatchCommand(
        2, 1, 2, ("decode", "prompt"), max_steps=3,
        eos_token_ids=((), (99,)), max_duration_seconds=1.0,
        prompt_remaining=(0, 8), prompt_tokens=((), (10, 11, 12)),
        prompt_positions=((), ((3, 7, 9), (4, 11, 9), (5, 15, 10))),
    )
    actual = deserialize_worker_command(serialize_worker_command(command))
    assert actual == command
    assert command.resident_updates(
        previous_batch_id=1, captured_m=2, request_ids=("decode", "prompt"),
        offsets=(0, 1), page_counts=(1, 1), block_size=16,
    ) == {}
    for changes in (
        dict(prompt_tokens=((), (10, 11))),
        dict(prompt_positions=((), ((3, 7, 9), (4, 11, 9), (5, 15)))),
        dict(prompt_remaining=(0, -1)),
        dict(prompt_remaining=(0, True)),
        dict(prompt_remaining=(0, 2**31)),
        dict(prompt_positions=((), (1, 2, 3))),
    ):
        invalid = replace(command, **changes)
        assert invalid.resident_updates(
            previous_batch_id=1, captured_m=2, request_ids=("decode", "prompt"),
            offsets=(0, 1), page_counts=(1, 1), block_size=16,
        ) is None


def test_unqualified_prompt_program_fails_before_any_compatibility_replay():
    from types import SimpleNamespace

    from cubic_serve.runtime.continuous_batch import run_resident_continuation

    command = ContinueBatchCommand(
        2, 1, 1, ("prompt",), eos_token_ids=((),),
        prompt_remaining=(1,), prompt_tokens=((17,),),
        prompt_positions=(((1, 3, 5),),),
    )
    with pytest.raises(BatchProtocolError, match="no qualified graph"):
        run_resident_continuation(
            command, SimpleNamespace(captured_m=1), previous_batch_id=1,
            request_ids=("prompt",), offsets=(0,), page_counts=(1,), block_size=16,
            rank=0, sampler_rank=0, completion_statuses=[0, 0],
            completion_barrier=None, timeout=1.0, interrupt=None,
            publish=lambda _: pytest.fail("unqualified program published output"),
        )


def test_gpu_periods_include_intervening_device_work_without_host_time():
    # GPU model bodies take 2/3/2 ms; the same GPU boundary recurs every
    # 5/6 ms after sampler, communication and ingress work. CPU publication
    # may be delayed arbitrarily and is intentionally absent from this input.
    clocks = [[1_000_000, 3_000_000], [5_000_000, 8_000_000], [12_000_000, 14_000_000]]
    assert resident_gpu_step_intervals(clocks) == (5.0, 6.0)
    shift = 2**54 + 17
    assert resident_gpu_step_intervals([[a + shift, b + shift] for a, b in clocks]) == (5.0, 6.0)
    assert resident_gpu_step_intervals([]) == ()
    assert resident_gpu_step_intervals(clocks[:1]) == ()


@pytest.mark.parametrize("clocks", [[[-1, 3]], [[4, 3]], [[1, 4], [2, 4]], [[1, 5], [2, 4]]])
def test_invalid_gpu_clock_trajectory_is_a_protocol_failure(clocks):
    with pytest.raises(BatchProtocolError, match="GPU clock order"):
        resident_gpu_step_intervals(clocks)


@pytest.mark.parametrize(
    "capacity,domain,expected",
    [
        (1, (1,), (1,)),
        (2, (1, 2), (1, 2)),
        (4, (3, 4), (3, 4)),
        (8, (5, 6, 7, 8), (5, 7, 8)),
        (256, tuple(range(129, 257)), (129, 193, 256)),
    ],
)
def test_temporal_audit_never_replays_outside_bucket_domain(capacity, domain, expected):
    assert resident_audit_rows(capacity, domain) == expected


@pytest.mark.parametrize(
    "capacity,domain",
    [
        (4, ()),
        (4, (0, 3)),
        (4, (3, 5)),
        (4, (3, 3)),
        (4, (4, 3)),
        (4, (True, 3)),
    ],
)
def test_invalid_audit_domain_fails_closed(capacity, domain):
    with pytest.raises(ValueError):
        resident_audit_rows(capacity, domain)


@pytest.mark.parametrize("block_size", [16, 32, 48, 64, 256])
@pytest.mark.parametrize("logical_m", [1, 3, 5])
def test_multiple_page_grants_match_independent_step_transitions(block_size, logical_m):
    ids = tuple(str(row) for row in range(logical_m))
    offsets = tuple((block_size - 1 + row * 5) % block_size for row in range(logical_m))
    current_offsets = list(offsets)
    page_counts = (3,) * logical_m
    counts = list(page_counts)
    grants = []
    for step in range(RESIDENT_STEP_CAPACITY):
        for row in range(logical_m):
            current_offsets[row] += 1
            if current_offsets[row] == block_size:
                current_offsets[row] = 0
                grants.append((step, row, counts[row], 100 + row * 100 + counts[row]))
                counts[row] += 1
    first = tuple(entry[1:] for entry in grants if entry[0] == 0)
    future = tuple(entry for entry in grants if entry[0] > 0)
    captured_m = 1 << (logical_m - 1).bit_length()
    command = ContinueBatchCommand(
        2, 1, captured_m, ids, first, RESIDENT_STEP_CAPACITY, ((),) * logical_m, 1.0, future
    )
    arguments = dict(
        previous_batch_id=1,
        captured_m=captured_m,
        request_ids=ids,
        offsets=offsets,
        page_counts=page_counts,
        block_size=block_size,
    )
    assert command.resident_updates(**arguments) == {
        row: (index, page) for row, index, page in first
    }
    if future:
        step, row, index, page = future[-1]
        invalid = replace(
            command, future_page_updates=(*future[:-1], (step, row, index + 1, page))
        )
        assert invalid.resident_updates(**arguments) is None
    oversized = replace(command, max_steps=RESIDENT_STEP_CAPACITY + 1)
    assert oversized.resident_updates(**arguments) is None


@pytest.mark.parametrize("block_size", [16, 32, 48, 64, 256])
@pytest.mark.parametrize("logical_m", [1, 3, 5, 128, 256])
def test_completed_grants_match_chronological_state_with_permuted_storage(block_size, logical_m):
    offsets = tuple((block_size - 1 + row * 5) % block_size for row in range(logical_m))
    counts = tuple(3 + row % 11 for row in range(logical_m))
    live_offsets, live_counts = list(offsets), list(counts)
    history = {0: (offsets, counts)}
    grants = []
    for step in range(RESIDENT_STEP_CAPACITY):
        for row in range(logical_m):
            live_offsets[row] += 1
            if live_offsets[row] == block_size:
                live_offsets[row] = 0
                grants.append((step, row, live_counts[row], 100 + len(grants)))
                live_counts[row] += 1
        history[step + 1] = tuple(live_offsets), tuple(live_counts)
    # CPU command order is independent of the GPU's indexed temporal schedule.
    for plan in (tuple(grants), tuple(reversed(grants)), tuple(grants[::2] + grants[1::2])):
        for completed in (0, 1, 17, 32, 63, 127, RESIDENT_STEP_CAPACITY):
            assert completed_resident_pages(offsets, counts, plan, completed) == history[completed]


@pytest.mark.parametrize(
    "offsets,counts,schedule,steps",
    [
        ((0,), (), (), 1),
        ((0,), (1,), (), -1),
        ((0,), (1,), (), True),
        ((0,), (1,), ((0, -1, 1, 7),), 1),
        ((0,), (1,), ((0, 1, 1, 7),), 1),
        ((0,), (1,), ((-1, 0, 1, 7),), 1),
    ],
)
def test_completed_page_state_rejects_invalid_owner_or_prefix(offsets, counts, schedule, steps):
    with pytest.raises(BatchProtocolError):
        completed_resident_pages(offsets, counts, schedule, steps)


@pytest.mark.parametrize("rank", [0, 1])
def test_page_state_consumption_precedes_topology_receipt(monkeypatch, rank):
    from types import SimpleNamespace

    import numpy as np

    from cubic_serve.executor.resident_window import ResidentDecodeWindow
    from cubic_serve.runtime import continuous_batch

    events = []
    original = continuous_batch.completed_resident_pages

    def reconcile(*args):
        result = original(*args)
        assert result == ((0, 1), (5, 8))
        events.append("reconcile")
        return result

    monkeypatch.setattr(continuous_batch, "completed_resident_pages", reconcile)
    window = ResidentDecodeWindow.__new__(ResidentDecodeWindow)
    window.draft_maintenance = None
    window.e = SimpleNamespace(captured_m=4, slot=SimpleNamespace(synchronize=lambda: None))
    window.supports = lambda command: True
    window.bind = lambda *args, **kwargs: None
    window.replay = lambda: None
    outputs = [[10 + step, 20 + step] for step in range(3)]
    window.consume = lambda rows: (outputs, [1.0] * 3)
    window._gpu_step_intervals_ms = (2.0, 3.0)
    window.host_flags = np.array([3, 0, 0], dtype=np.int32)
    window.host_array = np.zeros(64 + 3 * 7, dtype=np.int64)
    for step, tokens in enumerate(outputs):
        begin = 64 + step * 7
        window.host_array[begin : begin + 2] = tokens
        window.host_array[begin + 4] = 1_000_000
        window.host_array[begin + 5] = int(step == 2)

    class Barrier:
        def wait(self, timeout):
            assert events[-1] == "reconcile"
            events.append("barrier")

    def publish(result):
        if result.continuation_complete:
            assert events[-1] == "barrier"
            assert result.gpu_step_intervals_ms == (2.0, 3.0)
            events.append("final")
        else:
            assert result.gpu_step_intervals_ms == ()
            events.append("intermediate")

    command = ContinueBatchCommand(
        2, 1, 4, ("a", "b"), ((0, 3, 100),), 5, ((), ()), 1.0,
        ((1, 1, 7, 200), (2, 0, 4, 101), (4, 0, 5, 102)),
    )
    actual = window.run(
        command, request_ids=("a", "b"), offsets=(1, 0), page_counts=(3, 7),
        rank=rank, sampler_rank=0, completion_barrier=Barrier(), timeout=1.0,
        interrupt=None, publish=publish,
    )
    assert actual == ((0, 1), (5, 8))
    assert events == (["intermediate", "intermediate", "reconcile", "barrier", "final"]
                      if rank == 0 else ["reconcile", "barrier"])


@pytest.mark.parametrize("capacity,rows", [(1, 1), (8, 5), (128, 65), (256, 256)])
@pytest.mark.parametrize("shared,qualified", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize("with_program", [False, True])
def test_pinned_integer_staging_preserves_transport_and_unowned_tail(
    capacity, rows, shared, qualified, with_program,
):
    from types import SimpleNamespace

    import numpy as np
    import torch

    from cubic_serve.executor.resident_window import ResidentDecodeWindow

    window = ResidentDecodeWindow.__new__(ResidentDecodeWindow)
    window.draft_maintenance = None
    window.e = SimpleNamespace(captured_m=capacity, vocab_size=131071)
    window.output = torch.empty((RESIDENT_STEP_CAPACITY, capacity), dtype=torch.int64)
    window.host_flags = np.zeros(8, dtype=np.int32)
    window.host_flags[2] = 1  # A cancellation arrived before worker input fill.
    window._shared_journal = object() if shared else None
    window.qualified = qualified
    window._output_map_bound = False
    window._logical_steps_bound = False
    fields = {
        "policy_host": torch.zeros(3, dtype=torch.uint64),
        "meta_host": torch.full((5,), -99, dtype=torch.int32),
        "prompt_steps_host": torch.full((capacity,), -99, dtype=torch.int32),
        "input_tokens_host": torch.full((RESIDENT_STEP_CAPACITY, capacity), -99, dtype=torch.int64),
        "input_positions_host": torch.full((RESIDENT_STEP_CAPACITY, 3, capacity), -99,
                                           dtype=torch.int64),
        "page_schedule_host": torch.full((RESIDENT_STEP_CAPACITY, capacity, 3), -99,
                                         dtype=torch.int32),
        "eos_offsets_host": torch.full((capacity + 1,), -99, dtype=torch.int32),
        "eos_values_host": torch.full((capacity * 2,), -99, dtype=torch.int64),
    }
    for name, value in fields.items():
        setattr(window, name, value)
    window._host_staging_arrays = {name: value.numpy() for name, value in fields.items()}
    for target, source in (
        ("policy", "policy_host"), ("meta", "meta_host"),
        ("page_schedule", "page_schedule_host"), ("eos_offsets", "eos_offsets_host"),
        ("eos_values", "eos_values_host"),
        ("prompt_steps", "prompt_steps_host"),
        ("input_tokens", "input_tokens_host"),
        ("input_positions", "input_positions_host"),
    ):
        setattr(window, target, fields[source].clone())
    template = torch.tensor([rows, 0], dtype=torch.int32)
    window.sampler = SimpleNamespace(
        metadata=torch.zeros(2, dtype=torch.int32), _metadata_hosts={(rows, 0): template},
        _metadata_key=None,
    )
    eos = tuple((17, 2**54 + row) for row in range(rows))
    plan = tuple((step, row, 1 + step // 16, 1000 + step * capacity + row)
                 for step in range(15, RESIDENT_STEP_CAPACITY, 16)
                 for row in range(rows))
    pointers = {name: value.data_ptr() for name, value in fields.items()}
    counts = tuple(row % 3 for row in range(rows))
    tokens = tuple(tuple(17 + row + step for step in range(count))
                   for row, count in enumerate(counts))
    points = tuple(tuple((step + 1, step + 3, step + 5) for step in range(count))
                   for count in counts)
    program = dict(prompt_remaining=counts, prompt_tokens=tokens, prompt_positions=points)
    window.bind(
        rows, RESIDENT_STEP_CAPACITY, eos, tuple(reversed(plan)), duration=8.0,
        **(program if with_program else {}),
    )
    assert window.host_flags[2] == int(shared and qualified)

    # Build the submitted byte contracts independently of the staging views.
    expected = torch.full_like(window.page_schedule, -1)
    for step, row, index, page in plan:
        expected[step, row] = torch.tensor((row, index, page), dtype=torch.int32)
    assert torch.equal(window.page_schedule, expected)
    assert torch.equal(window.policy, torch.tensor((8_000_000_000, 0, 0), dtype=torch.uint64))
    assert torch.equal(window.meta, torch.tensor((rows, RESIDENT_STEP_CAPACITY, 0, 0, 0)))
    assert torch.equal(window.eos_offsets[: rows + 1], torch.arange(rows + 1) * 2)
    assert bool((window.eos_offsets[rows + 1 :] == -99).all())
    assert torch.equal(window.eos_values[: rows * 2],
                       torch.tensor([token for ids in eos for token in ids], dtype=torch.int64))
    assert bool((window.eos_values[rows * 2 :] == -99).all())
    assert torch.equal(window.sampler.metadata, template)
    assert window.sampler._metadata_key == (rows, False)
    assert pointers == {name: value.data_ptr() for name, value in fields.items()}
    assert window.prompt_steps[:rows].tolist() == (list(counts) if with_program else [0] * rows)
    assert bool((window.prompt_steps[rows:] == 0).all())
    if with_program:
        for row, count in enumerate(counts):
            assert window.input_tokens[:count, row].tolist() == list(tokens[row])
            assert window.input_positions[:count, :, row].tolist() == [list(p) for p in points[row]]
        assert bool((window.input_tokens[:, rows:] == 0).all())
        assert bool((window.input_positions[:, :, rows:] == 0).all())


def test_shared_interrupt_changes_only_stop_words_and_retains_process_event():
    memories = [shared_memory.SharedMemory(create=True, size=512) for _ in range(2)]
    signal = SharedPublicationInterrupt(threading.Event())
    try:
        for memory in memories:
            memory.buf[:] = bytes([0xA5]) * memory.size
        signal.attach(tuple(memory.name for memory in memories))
        for enabled in (True, False, True, False):
            (signal.set if enabled else signal.clear)()
            assert signal.is_set() is enabled
            for memory in memories:
                assert bytes(memory.buf[:8]) == bytes([0xA5]) * 8
                assert int.from_bytes(memory.buf[8:12], "little") == int(enabled)
                assert bytes(memory.buf[12:]) == bytes([0xA5]) * (memory.size - 12)
        signal.close()
        signal.close()
        # Closing a controller's views never unlinks the worker-owned journal.
        for memory in memories:
            attached = shared_memory.SharedMemory(name=memory.name)
            attached.close()
    finally:
        signal.close()
        for memory in memories:
            memory.close()
            memory.unlink()


@pytest.mark.parametrize("names", [None, [], (["invalid"],), ("same", "same")])
def test_shared_interrupt_rejects_invalid_descriptors(names):
    signal = SharedPublicationInterrupt(threading.Event())
    with pytest.raises(BatchProtocolError, match="descriptors"):
        signal.attach(names)
    signal.close()


def test_input_encoder_revokes_then_waits_for_completion_without_dispatch_overtaking():
    interrupted = threading.Event()
    encoder_entered = threading.Event()
    release_encoder = threading.Event()
    dispatch_entered = threading.Event()
    lease = InputEncoderExecutionLease(interrupted)
    order = []
    lease.acquire()  # The language graph still owns buffers and state.

    def encode():
        with lease:
            order.append("encoder")
            encoder_entered.set()
            assert release_encoder.wait(2)

    def dispatch():
        lease.acquire()
        try:
            order.append("dispatch")
            dispatch_entered.set()
        finally:
            lease.release()

    encoder = threading.Thread(target=encode)
    dispatcher = threading.Thread(target=dispatch)
    encoder.start()
    assert interrupted.wait(2)
    assert not encoder_entered.is_set()
    dispatcher.start()
    lease.release()  # Only after durable language graph completion.
    try:
        assert encoder_entered.wait(2)
        assert not dispatch_entered.is_set()
    finally:
        release_encoder.set()
        encoder.join(2)
        dispatcher.join(2)
    assert not encoder.is_alive() and not dispatcher.is_alive()
    assert order == ["encoder", "dispatch"]


def test_input_encoder_failed_revocation_does_not_leak_a_lease_ticket():
    class BrokenInterrupt:
        def set(self):
            raise OSError("revocation unavailable")

    lease = InputEncoderExecutionLease(BrokenInterrupt())
    with pytest.raises(OSError, match="revocation unavailable"), lease:
        pytest.fail("encoder ran without revocation")
    lease.acquire()
    lease.release()
    with pytest.raises(RuntimeError, match="not owned"):
        lease.release()




def test_stochastic_continuation_selects_qualified_window_without_changing_sampling_mode():
    from types import SimpleNamespace

    from cubic_serve.runtime.continuous_batch import run_resident_continuation

    requests = ('a', 'b', 'c')
    command = ContinueBatchCommand(2, 1, 4, requests)
    observed = []

    def run(actual, **kwargs):
        observed.append((actual, kwargs['request_ids'], kwargs['offsets']))
        return 'stochastic-result'

    executor = SimpleNamespace(
        captured_m=4, _sampling=SimpleNamespace(all_greedy=False),
        _resident_window=SimpleNamespace(supports=lambda _: False),
        _resident_stochastic_window=SimpleNamespace(supports=lambda _: True, run=run),
    )
    result = run_resident_continuation(
        command, executor, previous_batch_id=1, request_ids=requests,
        offsets=(1, 2, 3), page_counts=(1, 1, 1), block_size=16,
        rank=0, sampler_rank=0, completion_statuses=[0, 0],
        completion_barrier=None, timeout=1, interrupt=None, publish=lambda _: None,
    )
    assert result == 'stochastic-result'
    assert observed == [(command, requests, (1, 2, 3))]
    assert executor._sampling.all_greedy is False


def test_survivor_wire_and_owned_projection_precede_graph_replay():
    from types import SimpleNamespace

    from cubic_serve.runtime.continuous_batch import run_resident_continuation

    events = []
    command = ContinueBatchCommand(2, 1, 4, ('d', 'a', 'c'), source_rows=(3, 0, 2))
    assert deserialize_worker_command(serialize_worker_command(command)) == command
    def run(actual, **kwargs):
        events.append(('run', kwargs['request_ids'], kwargs['offsets']))
        assert actual.source_rows == ()
        return kwargs['offsets'], kwargs['page_counts']
    executor = SimpleNamespace(
        captured_m=4, _sampling=SimpleNamespace(all_greedy=True),
        _resident_window=SimpleNamespace(supports=lambda _: True, run=run),
        _resident_compaction=SimpleNamespace(
            qualified=True,
            bind=lambda rows, **kw: events.append(('bind', rows, kw['source_limit'])),
            replay=lambda: events.append(('replay',)),
        ),
    )
    arguments = dict(previous_batch_id=1, request_ids=('a','b','c','d'),
                     offsets=(1,2,3,4), page_counts=(1,1,1,1), block_size=16,
                     rank=0, sampler_rank=0, completion_statuses=[0,0],
                     completion_barrier=None, timeout=1, interrupt=None, publish=lambda _: None)
    assert run_resident_continuation(command, executor, **arguments) == ((4,1,3),(1,1,1))
    assert events == [('bind',(3,0,2),4), ('replay',), ('run',('d','a','c'),(4,1,3))]
    assert executor._bound_active_rows == 3
    for rows, ids in (((0,0,2),('a','a','c')), ((True,0,2),('b','a','c')),
                      ((4,0,2),('x','a','c')), ((3,0,2),('a','d','c')),
                      ((0,2),('a','c'))):
        events.clear()
        with pytest.raises(BatchProtocolError):
            run_resident_continuation(replace(command, source_rows=rows, request_ids=ids),
                                      executor, **arguments)
        assert events == []
    executor._features_bound = True
    with pytest.raises(BatchProtocolError, match='no qualified graph'):
        run_resident_continuation(command, executor, **arguments)
    assert events == []
    executor._features_bound = False
    executor._resident_compaction.qualified = False
    with pytest.raises(BatchProtocolError, match='no qualified graph'):
        run_resident_continuation(command, executor, **arguments)
    assert events == []


@pytest.mark.skipif(os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1",
                    reason="explicit CUDA audit required")
def test_compaction_scalar_padding_is_captured_and_alias_safe():
    import torch

    from cubic_serve.executor.resident_window import CapturedRowCompaction

    device = torch.device("cuda:0")
    ids = torch.arange(4, dtype=torch.int64, device=device) + 2**54
    positions = torch.arange(12, dtype=torch.int64, device=device).view(3, 4)
    activity = torch.full((4,), -99, dtype=torch.int32, device=device)
    fields = ((ids, ids, 0, 0), (positions, positions, 1, -17))
    original_ids, original_positions = ids.cpu().clone(), positions.cpu().clone()
    graph = CapturedRowCompaction(fields, activity, stream=torch.cuda.Stream(device=device))
    try:
        ids.copy_(original_ids)
        positions.copy_(original_positions)
        graph.bind((3, 0, 2), source_limit=4)
        graph.replay()
        torch.cuda.synchronize(device)
        assert ids.cpu().tolist() == [2**54+3, 2**54, 2**54+2, 0]
        expected = torch.full((3, 4), -17, dtype=torch.int64)
        expected[:, :3] = original_positions[:, [3, 0, 2]]
        assert torch.equal(positions.cpu(), expected)
        assert activity.cpu().tolist() == [1, 1, 1, 0]
        with pytest.raises(ValueError):
            graph.bind((0, 0), source_limit=4)
    finally:
        graph.close()


def test_segmented_continuation_keeps_request_count_distinct_from_logical_m():
    command = ContinueBatchCommand(
        2, 1, 4, ("prefill", "decode"), max_steps=2, eos_token_ids=((), ()),
        prompt_remaining=(9, 0), prompt_tokens=((10, 11, 12, 13), ()),
        prompt_positions=(((3, 5, 7), (4, 8, 9), (5, 11, 13), (6, 14, 17)), ()),
        owner_lengths=(2, 1), source_pages=(5, 11),
        page_program=(((5, 6, 1, 0), (6, 6, 1, 0), (11, 11, 1, 0)),
                      ((6, 7, 1, 0), (7, 7, 1, 0), (11, 11, 1, 0))),
    )
    assert command.logical_m == 3 and len(command.request_ids) == 2
    assert deserialize_worker_command(serialize_worker_command(command)) == command
    arguments = dict(previous_batch_id=1, captured_m=4, request_ids=command.request_ids,
                     offsets=(1, 2), page_counts=(2, 2), block_size=16)
    assert command.resident_updates(**arguments) == {}
    for changes in (
        dict(owner_lengths=(True, 1)), dict(owner_lengths=(4, 1)),
        dict(owner_lengths=(2, 2)), dict(page_program=()),
        dict(source_pages=(5, 5)),
        dict(source_pages=(6, 11)),
        dict(source_pages=(-1, 11)),
        dict(source_rows=(0, 1)), dict(max_steps=5),
        dict(page_updates=((0, 2, 12),)),
    ):
        assert replace(command, **changes).resident_updates(**arguments) is None
    assert command.resident_updates(**dict(arguments, offsets=(15, 2))) is None


def test_segmented_owner_witness_rejects_shared_state_and_draft_rows():
    from cubic_serve.runtime.continuous_batch import BatchCommand, BatchRow, owned_segment_rows

    rows = (
        BatchRow('p', 10, 17, 3, 3, 7, 8, (2, 8), reset_state=True, segment_length=2),
        BatchRow('p', 11, 18, 3, 3, 8, 8, (2, 8), segment_length=0),
        BatchRow('d', 12, 2, 4, 4, 9, 9, (9,), is_decode=True),
    )
    assert owned_segment_rows(rows) == (2, 1)
    assert BatchCommand(1, 4, rows).logical_m == 3
    assert owned_segment_rows((*rows[:2], replace(rows[2], state_source_slot=3,
                                                 state_destination_slot=3))) == ()
    assert owned_segment_rows((*rows[:2], replace(rows[2], mtp_position=4))) == ()
    with pytest.raises(BatchProtocolError, match='not contiguous'):
        owned_segment_rows((rows[0], replace(rows[1], state_source_slot=5), rows[2]))
    assert owned_segment_rows((rows[0], replace(rows[1], input_feature=b'xx'), rows[2])) == ()


def test_draft_continuation_wire_preserves_real_grants_and_review_rows():
    from cubic_serve.runtime.continuous_batch import DraftWindowBinding

    binding = DraftWindowBinding(3, (7, 2), (19, 23), (64, 64), (64, 64), (100, 60),
                                 ((10, 11), (12, 13)), ((20, 21), (22, 23)))
    command = ContinueBatchCommand(9, 8, 8, ('a', 'b'), max_steps=4,
                                   eos_token_ids=((99,), ()), draft=binding)
    assert command.logical_m == 8
    assert deserialize_worker_command(serialize_worker_command(command)) == command
    budgeted = replace(binding, decision_tokens=(0, 7), decision_rounds=(2, 0))
    assert deserialize_worker_command(serialize_worker_command(
        replace(command, draft=budgeted))).draft == budgeted
    for changes in (dict(owners=(7, 7)), dict(target_positions=(True, 23)),
                    dict(target_limits=(22, 64)), dict(remaining=(100, 0)),
                    dict(depth=True), dict(provider_limits=(64,)),
                    dict(decision_tokens=(1, 2)),
                    dict(decision_tokens=(0, 0), decision_rounds=(0, 0)),
                    dict(decision_tokens=(True, 1), decision_rounds=(1, 1)),
                    dict(decision_tokens=[], decision_rounds=[]),
                    dict(decision_tokens=(1, 2), decision_rounds=(1,))):
        with pytest.raises(BatchProtocolError):
            replace(binding, **changes)


def test_draft_window_publishes_intermediate_but_holds_final_for_all_ranks():
    from types import SimpleNamespace
    from cubic_serve.runtime.continuous_batch import (
        BatchResult, DraftWindowBinding, SegmentResult, run_draft_continuation,
    )

    trace = []
    binding = DraftWindowBinding(1, (7,), (19,), (64,), (64,), (100,))
    command = ContinueBatchCommand(9, 8, 2, ('a',), max_steps=2,
                                   eos_token_ids=((),), draft=binding)
    receipts = tuple(BatchResult(0, 9, ('a',), (), (SegmentResult('a', 2, (5, 6)),),
                                continuation_index=i, continuation_complete=i == 1)
                     for i in range(2))
    window = SimpleNamespace(
        qualified=True, bind=lambda **kw: trace.append(('bind', kw)),
        replay=lambda: trace.append('replay'), records=lambda: receipts,
        receipt=lambda i, **kw: receipts[i], consume=lambda: trace.append('consume'),
    )
    executor = SimpleNamespace(captured_m=2,
                               _pipelined_draft_round=SimpleNamespace(depth=1, active_draft_window=window))
    barrier = SimpleNamespace(wait=lambda timeout: trace.append('all-ranks-complete'))
    kwargs = dict(previous_batch_id=8, request_ids=('a',), rank=0, sampler_rank=0,
                  completion_barrier=barrier, timeout=1., interrupt=None,
                  publish=lambda result: trace.append(('publish', result)))
    run_draft_continuation(command, executor, **kwargs)
    assert [event if isinstance(event, str) else event[0] for event in trace] == [
        'bind', 'replay', 'publish', 'consume', 'all-ranks-complete', 'publish',
    ]
    assert trace[-1][1].resident_draft_depth == 1
    assert trace[0][1]['target_limits'] == (64,)
    for changes in (dict(previous_batch_id=7), dict(request_ids=('b',))):
        with pytest.raises(BatchProtocolError):
            run_draft_continuation(command, executor, **dict(kwargs, **changes))
    with pytest.raises(BatchProtocolError):
        run_draft_continuation(replace(command, page_updates=((0, 1, 7),)), executor, **kwargs)


@pytest.mark.parametrize('depth,count', [(1, 1), (2, 3), (3, 4)])
@pytest.mark.parametrize('random_sampling', [False, True])
@pytest.mark.parametrize('host_grammar', [False, True])
@pytest.mark.parametrize('age', [0., 86400.])
def test_draft_window_planner_uses_owned_pages_and_remaining_output(
    depth, count, random_sampling, host_grammar, age,
):
    import time
    from types import SimpleNamespace
    from cubic_serve.runtime.continuous_batch import GenerateCommand
    from cubic_serve.runtime.performance import _ProcessController, _ScheduledRequest

    from cubic_serve.generation.sampler import GREEDY_SAMPLING, SamplingParameters

    controller = _ProcessController.__new__(_ProcessController)
    selected = tuple(_ScheduledRequest(
        GenerateCommand(str(i), (11, 12), 1024, (99,), sampling=(
            SamplingParameters(1., .95, 20, 0x123456789+i)
            if random_sampling and i % 2 == 0 else GREEDY_SAMPLING)), None, time.monotonic() - age,
        prefill_cursor=2, generated_count=10, last_token_id=13, state_slot=10 + i,
        pages=[20 + i], mtp_pages=[30 + i], state_initialized=True,
    ) for i in range(count))
    capacity = 1 << (count * (depth + 1) - 1).bit_length()
    controller._resident_draft = (7, capacity, depth, selected)
    controller._resident_draft_capacities = (capacity,)
    controller._grammar_draft_continuation_capacities = (capacity,)
    controller._resident_grammar_draft = host_grammar
    controller._num_speculative_tokens = depth
    controller._requests = {record.command.request_id: record for record in selected}
    controller._shared_state_slots = {}; controller._shared_pages = {}; controller._shared_mtp_pages = {}
    controller._worker_timeout = 100.; controller._block_size = 32
    controller.max_sequence_tokens = 4096
    controller._decode_window_seconds = 1.; controller._resident_draft_step_seconds = .01
    controller._batch_id = 8
    command, rows, owners = controller._next_draft_continuation_locked()
    assert not rows and owners == selected
    assert command.logical_m == count * (depth + 1)
    assert command.draft.target_limits == (32,) * count
    assert command.draft.provider_limits == (32,) * count
    # Even the largest accepted prefix of the last granted round stays in its
    # real provider/cache allocation. Table padding contributes no capacity.
    assert 11 + (command.max_steps - 1) * (depth + 1) + 2 * depth < 32
    controller._shared_mtp_pages[30] = 2
    assert controller._next_draft_continuation_locked() is None
    controller._shared_mtp_pages.clear()
    import heapq
    controller._free_pages = list(range(1000, 2000))
    controller._free_mtp_pages = list(range(2000, 3000))
    first_steps = command.max_steps
    expanded, _, _ = controller._next_draft_continuation_locked()
    if host_grammar:
        assert expanded.max_steps == first_steps
        assert len(controller._free_pages) == len(controller._free_mtp_pages) == 1000
    else:
        assert expanded.max_steps > first_steps
    assert expanded.draft.target_tables == tuple(tuple(r.pages) for r in selected)
    assert expanded.draft.provider_tables == tuple(tuple(r.mtp_pages) for r in selected)
    assert all(r.pages[0] == 20 + i and r.mtp_pages[0] == 30 + i
               for i, r in enumerate(selected))
    assert len({page for r in selected for page in r.pages}) == sum(len(r.pages) for r in selected)
    assert len(controller._free_pages) + sum(len(r.pages) - 1 for r in selected) == 1000
    for i, record in enumerate(selected):
        for page in record.pages[1:]: heapq.heappush(controller._free_pages, page)
        for page in record.mtp_pages[1:]: heapq.heappush(controller._free_mtp_pages, page)
        record.pages = [20 + i]; record.mtp_pages = [30 + i]
    controller._free_pages.clear(); controller._free_mtp_pages.clear()
    for i, record in enumerate(selected):
        record.generated_count = 42
        record.pages = [100 + i, 20 + i]
        record.mtp_pages = [200 + i, 30 + i]
        controller._shared_pages[100 + i] = 1
        controller._shared_mtp_pages[200 + i] = 1
    assert controller._next_draft_continuation_locked() is not None
    controller._shared_pages[20] = 1
    assert controller._next_draft_continuation_locked() is None
    controller._shared_pages.clear(); controller._shared_mtp_pages.clear()
    for i, record in enumerate(selected):
        record.generated_count = 10
        record.pages = [20 + i]; record.mtp_pages = [30 + i]
    selected[0].generated_count = 1022
    assert controller._next_draft_continuation_locked() is None
    selected[0].generated_count = 10
    controller._requests['new'] = SimpleNamespace(cancelled=False)
    assert controller._next_draft_continuation_locked() is None


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("age", [0., 86400.])
@pytest.mark.parametrize("coherent_budget", [False, True])
def test_dynamic_draft_window_stops_before_complete_cohort_decision(depth, age, coherent_budget):
    import time

    from cubic_serve.runtime.continuous_batch import GenerateCommand
    from cubic_serve.runtime.performance import _ProcessController, _ScheduledRequest
    from cubic_serve.runtime.speculative import DynamicDraftPolicy

    c = _ProcessController.__new__(_ProcessController)
    records = tuple(_ScheduledRequest(
        GenerateCommand(str(i), (11, 12), 1024, ()), None, time.monotonic() - age,
        prefill_cursor=2, generated_count=10, last_token_id=13, state_slot=i,
        pages=[20+i], mtp_pages=[30+i], state_initialized=True,
        draft_tokens=depth, draft_policy=DynamicDraftPolicy(depth, 3),
    ) for i in range(2))
    for _ in range(2):
        records[1].draft_policy.observe(accepted=depth, proposed=depth)
    capacity = 1 << (2 * (depth+1)-1).bit_length()
    c._resident_draft = (7, capacity, depth, records)
    c._resident_draft_capacities = (capacity,)
    c._num_speculative_tokens = 1
    c._dynamic_speculation = True
    c._requests = {r.command.request_id:r for r in records}
    c._shared_state_slots = {}
    c._shared_pages = {}
    c._shared_mtp_pages = {}
    c._worker_timeout = 100.
    c._block_size = 4096
    c.max_sequence_tokens = 16384
    c._decode_window_seconds = None
    c._resident_draft_step_seconds = .01
    c._batch_id = 8
    if coherent_budget:
        from cubic_serve.runtime.speculative import DynamicDraftPlanner
        c._graph_capacities = (1, 2, 4, 8)
        c._draft_planner = DynamicDraftPlanner(c._graph_capacities)
        c._draft_planner.seed_compound_costs(((4, 2, 1.), (6, 3, 1.), (8, 4, 1.)))
        c._draft_cost_planner = lambda *args: c._draft_planner
    result = c._next_draft_continuation_locked()
    assert result is not None
    command = result[0]
    horizon = max(r.draft_policy.rounds_until_decision for r in records)
    assert command.max_steps == (max(r.draft_policy.maximum_rounds_until_decision for r in records)
                                 if coherent_budget else horizon)
    assert bool(command.draft.decision_tokens) == coherent_budget
    for _ in range(command.max_steps):
        assert not all(r.draft_policy.ready for r in records)
        for r in records:
            r.draft_policy.observe(accepted=depth, proposed=depth)
        if coherent_budget and all(r.draft_policy.ready for r in records):
            break
    assert all(r.draft_policy.ready for r in records)
    next_window = c._next_draft_continuation_locked()
    if coherent_budget:
        assert next_window is not None
        assert next_window[0].draft.decision_tokens == (24, 24)
        assert all(not r.draft_policy.ready for r in records)
    else:
        assert next_window is None
    for r in records:
        r.draft_policy.move_toward(depth)
    records[1].draft_tokens = max(0, depth-1)
    assert c._next_draft_continuation_locked() is None


def test_draft_receipt_reuses_publication_policy_and_defers_cancelled_free():
    from types import SimpleNamespace
    from cubic_serve.runtime.continuous_batch import BatchResult, DraftWindowBinding, SegmentResult
    from cubic_serve.runtime.performance import _ProcessController, Qwen3_5TP2PP2ServingError

    controller = _ProcessController.__new__(_ProcessController)
    records = tuple(SimpleNamespace(command=SimpleNamespace(request_id=request, max_tokens=20,
                                                            eos_token_ids=(99,)),
                                    generated_count=2, cancelled=False)
                    for request in ('a', 'b'))
    controller._requests = {r.command.request_id: r for r in records}
    controller._serving_stats = SimpleNamespace(speculate=lambda *args: None)
    calls = []
    def commit(selected, tokens, wakes, **kw):
        calls.append((tuple(r.command.request_id for r in selected), tokens, kw['defer_cancelled_release']))
        for record, token in zip(selected, tokens):
            if record.cancelled:
                if not kw['defer_cancelled_release']:
                    controller._requests.pop(record.command.request_id)
            else:
                record.generated_count += 1
                if token == 99 or record.generated_count == record.command.max_tokens:
                    controller._requests.pop(record.command.request_id)
        return 0, 0
    controller._commit_steady_records_locked = commit
    batch = ContinueBatchCommand(1, 0, 8, ('a', 'b'), max_steps=3,
        draft=DraftWindowBinding(3, (1, 2), (3, 3), (64, 64), (64, 64), (18, 18)))
    result = BatchResult(0, 1, ('a', 'b'), (),
                         (SegmentResult('a', 3, (5, 6, 7)), SegmentResult('b', 1, (8,))),
                         continuation_complete=False)
    from cubic_serve.runtime.speculative import DynamicDraftPolicy
    controller._dynamic_speculation = True
    for record in records:
        record.draft_policy = DynamicDraftPolicy(3, 3)
    records[1].cancelled = True
    controller._commit_draft_receipt_locked(batch, records, result, None, profile=None)
    assert calls == [(('a', 'b'), (5, 8), True), (('a',), (6,), True), (('a',), (7,), True)]
    assert 'b' in controller._requests
    assert records[0].draft_policy._rounds == 1
    assert records[0].draft_policy._window_tokens == 3
    assert records[1].draft_policy._rounds == 0
    final = replace(result, segment_results=(SegmentResult('a', 3, (99, 9, 10)),
                                            SegmentResult('b', 1, (8,))),
                    continuation_complete=True)
    controller._commit_draft_receipt_locked(batch, records, final, None, profile=None)
    assert not controller._requests  # Post-EOS draft tokens were never exposed.
    assert records[0].draft_policy._rounds == 1  # Terminal rounds do not alter policy.
    with pytest.raises(Qwen3_5TP2PP2ServingError, match='terminal receipt'):
        controller._commit_draft_receipt_locked(batch, records, replace(final, continuation_complete=False),
                                                None, profile=None)


@pytest.mark.parametrize('terminal_token,max_tokens,eos', [(6, 4, ()), (99, 5, (99,))])
def test_draft_final_receipt_preserves_deferred_token_before_terminal(terminal_token, max_tokens, eos):
    import time
    from types import SimpleNamespace
    from cubic_serve.runtime.continuous_batch import (
        BatchResult, DraftWindowBinding, GenerateCommand, SegmentResult,
    )
    from cubic_serve.runtime.performance import _ProcessController, _ScheduledRequest
    from cubic_serve.runtime.token_stream import TokenStream, TokenWakeBatch

    controller = _ProcessController.__new__(_ProcessController)
    sink = TokenStream()
    record = _ScheduledRequest(GenerateCommand('a', (11,), max_tokens, eos), sink,
                               time.monotonic(), prefill_cursor=1, generated_count=2, state_slot=1,
                               state_initialized=True)
    controller._requests = {'a': record}
    controller._serving_stats = SimpleNamespace(speculate=lambda *args: None,
                                                publish=lambda *args, **kw: None)
    controller._finish_request_stats_locked = lambda record: None
    controller._release_request_locked = lambda request: controller._requests.pop(request)
    batch = ContinueBatchCommand(1, 0, 2, ('a',), max_steps=2,
        draft=DraftWindowBinding(1, (1,), (1,), (32,), (32,), (4,)))
    result = BatchResult(0, 1, ('a',), (),
                         (SegmentResult('a', 2, (5, terminal_token)),))
    wakes = TokenWakeBatch()
    controller._commit_draft_receipt_locked(batch, (record,), result, wakes, profile=None)
    wakes.flush()
    assert list(sink) == [5, terminal_token]
    assert record.generated_count == 4


@pytest.mark.parametrize('capacity,states,context,fits', [
    (64, 4, 32, False), (64, 4, 77, False), (64, 4, 78, True),
    (16, 8, 21, False), (16, 8, 22, True), (32, 16, 32, True),
])
def test_prompt_qualification_checks_actual_width_before_storage(capacity, states, context, fits):
    from types import SimpleNamespace
    from cubic_serve.executor.resident_window import ResidentDecodeWindow

    class ReachedStorage(Exception):
        pass

    def reached(_):
        raise ReachedStorage

    arena = SimpleNamespace(
        spec=SimpleNamespace(state_slot_capacity=states, page_capacity=10000),
        device='cpu', reset_page_rows=reached,
    )
    executor = SimpleNamespace(
        body=SimpleNamespace(buffers=SimpleNamespace(), arena=arena),
        _block_size=16, max_sequence_tokens=context, captured_m=capacity,
        vocab_size=4096, _target_hidden_slots=object(),
    )
    window = SimpleNamespace(
        e=executor,
        draft_maintenance=SimpleNamespace(provider=SimpleNamespace(
            body=SimpleNamespace(arena=arena))),
        qualified=False, program_qualified=False, segmented_program_qualified=False,
    )
    if fits:
        with pytest.raises(ReachedStorage):
            ResidentDecodeWindow.qualify_segments(window, (capacity,))
    else:
        ResidentDecodeWindow.qualify_segments(window, (capacity,))
        assert not window.qualified
        assert not window.program_qualified
        assert not window.segmented_program_qualified


@pytest.mark.parametrize('depth', [1, 3])
def test_draft_fixture_random_sampling_and_rng_follow_permuted_logical_owner(depth):
    from types import SimpleNamespace

    from cubic_serve.generation.sampler import SamplingParameters
    from cubic_serve.runtime.quantized_qualification import _draft_window_audit_rows

    sampling = SamplingParameters(1., .95, 20, 0x123456789)
    round_ = SimpleNamespace(depth=depth, stride=depth + 1, C=2, target=SimpleNamespace(
        body=SimpleNamespace(arena=SimpleNamespace(shared_storage=None)), _block_size=16, _scratch_states=tuple(range(100, 100 + 2*(depth+1))), vocab_size=256))
    pending = [(str(owner), 4, 101 + owner, None, (1 << 34) + owner, None,
                tuple(range(33, 33 + depth))) for owner in range(2)]
    rows = _draft_window_audit_rows(round_, pending, 2, 2, owner_order=(1, 0),
                                   sampling=sampling)
    for physical, owner in enumerate((1, 0)):
        segment = rows[physical*(depth+1):(physical+1)*(depth+1)]
        assert all(row.sampling.temperature == 1. and row.sampling.top_k == 20
                   and row.sampling.top_p == .95 and row.sampling.seed == sampling.seed+owner
                   for row in segment)
        assert [row.sample_ordinal for row in segment] == [
            (1 << 34)+owner+j for j in range(depth+1)]


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_draft_candidate_sentinel_allowance_never_hides_owned_score_changes(dtype):
    import torch
    from cubic_serve.runtime.quantized_qualification import _compare_draft_candidate_scores

    kind = getattr(torch, dtype)
    expected = torch.tensor([1., float('nan')], dtype=kind)
    actual = expected.clone()
    # Different quiet NaN payloads are legal only at the exact -1 ID.
    actual.view(torch.int16)[1] = 0x7fff
    ids = torch.tensor([7, -1])
    _compare_draft_candidate_scores(actual, expected, ids)
    actual[0] = torch.nextafter(actual[0], torch.tensor(float('inf'), dtype=kind))
    with pytest.raises(ValueError, match='finite candidates'):
        _compare_draft_candidate_scores(actual, expected, ids)
    actual.copy_(expected)
    with pytest.raises(ValueError, match='sentinel validity'):
        _compare_draft_candidate_scores(actual, expected, torch.tensor([7, 8]))
    actual[1] = 0.
    with pytest.raises(ValueError, match='sentinel validity'):
        _compare_draft_candidate_scores(actual, expected, ids)


@pytest.mark.skipif(os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1",
                    reason="explicit CUDA audit required")
@pytest.mark.parametrize("fault", [None, "last_token", "last_padding", "last_control"])
def test_full_native_journal_checks_every_slot_and_protected_tail(monkeypatch, fault):
    """Real phase2 kernel; no synthetic model or sampler is a numerical oracle."""
    from types import SimpleNamespace

    import torch
    import torch.distributed as dist
    from cuda.bindings import runtime as rt

    from cubic_serve.executor.resident_window import SOURCE, ResidentDecodeWindow, check
    from cubic_serve.ops.cuda.native import CudaScalar
    from cubic_serve.ops.cuda.native_compiler import compile_native

    device, capacity, rows, steps = torch.device("cuda:0"), 8, 3, RESIDENT_STEP_CAPACITY
    def integers(*shape):
        return torch.zeros(shape, dtype=torch.int32, device=device)
    window = ResidentDecodeWindow.__new__(ResidentDecodeWindow)
    tokens = torch.full((capacity,), -1, dtype=torch.int64, device=device)
    window.e = SimpleNamespace(
        captured_m=capacity, vocab_size=31, _token_ids=tokens, _rank=0, _world_group=None,
        body=SimpleNamespace(hidden=tokens),
        slot=SimpleNamespace(synchronize=lambda: torch.cuda.synchronize(device)),
    )
    window.meta = torch.tensor([rows, steps, 0, 0, 0], dtype=torch.int32, device=device)
    window.output = torch.empty((steps, capacity), dtype=torch.int64, device=device)
    clocks = torch.empty((steps, 2), dtype=torch.int64, device=device)
    window.policy = torch.zeros(3, dtype=torch.uint64, device=device)
    valid = torch.ones((), dtype=torch.bool, device=device)
    window.sampler = SimpleNamespace(final_valid=valid)
    host = torch.zeros(64 + steps * (capacity + 3), dtype=torch.int64, pin_memory=True)
    window.host_array = host.numpy()
    window.host_flags = window.host_array.view("int32")
    mapped = check(rt.cudaHostGetDevicePointer(host.data_ptr(), 0))[0]
    address = torch.tensor([int(mapped)], dtype=torch.uint64, device=device)
    activity = integers(capacity)
    activity[:rows] = 1
    arguments = (
        window.meta, tokens, valid, integers(capacity + 1), tokens,
        window.output, clocks, address, integers(steps, capacity, 3), integers(capacity, 3),
        window.policy, activity, integers(capacity),
        torch.arange(capacity, dtype=torch.int32, device=device),
        torch.zeros(capacity, dtype=torch.bool, device=device),
        integers(steps, capacity, 4), integers(capacity), integers(capacity), integers(capacity),
        integers(capacity), integers(capacity), integers(capacity, 1), integers(capacity),
        CudaScalar("i32", 64), CudaScalar("i32", 1), CudaScalar("i32", capacity),
        CudaScalar("i32", 31), CudaScalar("i32", 1),
        *(CudaScalar("i64", 0) for _ in range(4)), CudaScalar("i32", 2),
    )
    module, entry = compile_native((("entry.cu", SOURCE),), device, "resident_window")
    native = module.bind(entry, arguments, operator_id="runtime.resident_window_coordinate",
                         grid=(1, 1, 1), block=(128, 1, 1))
    calls = 0
    def launch():
        nonlocal calls
        native.launch()
        if calls == steps - 1:
            if fault == "last_token":
                window.output[-1, 0].fill_(-2)
            elif fault == "last_padding":
                window.output[-1, rows].fill_(0)
            elif fault == "last_control":
                window.meta[4].fill_(9)
        calls += 1
    window.calls = (None, None, SimpleNamespace(launch=launch))
    monkeypatch.setattr(dist, "barrier", lambda **kwargs: None)
    if fault is None:
        window._qualify_journal_storage(rows)
    else:
        with pytest.raises(RuntimeError, match="journal storage or control write-set"):
            window._qualify_journal_storage(rows)
    assert calls == steps
    if fault is None:
        # Across these base-vocabulary digits every (step,row) has a unique
        # signature, even when the test vocabulary is smaller than the journal.
        # This independently detects permutations between any journal slots.
        factor = 1
        ordinal = torch.arange(steps * rows).reshape(steps, rows)
        while factor < steps * rows:
            expected = (ordinal // factor).remainder(window.e.vocab_size)
            device_expected = expected.to(device)
            window.output.fill_(-99)
            window.host_array[64:] = -99
            for index in range(steps):
                tokens[:rows].copy_(device_expected[index])
                window.meta[2].fill_(index)
                native.launch()
            torch.cuda.synchronize(device)
            mapped_output = torch.from_numpy(window.host_array[64:]).reshape(
                steps, capacity + 3,
            )
            assert torch.equal(window.output[:, :rows].cpu(), expected)
            assert torch.equal(mapped_output[:, :rows], expected)
            assert bool((window.output[:, rows:] == -99).all())
            assert bool((mapped_output[:, rows:capacity] == -99).all())
            factor *= window.e.vocab_size


def test_draft_ingress_transport_keeps_the_qualified_continuation_capability():
    import pickle

    from cubic_serve.runtime.continuous_batch import BatchResult

    result = BatchResult(0, 7, ("owner",), (), resident_draft_depth=3, draft_ingress=True)
    restored = pickle.loads(pickle.dumps(result))
    assert restored == result and restored.draft_ingress is True
    assert restored.resident_draft_depth == 3
    assert BatchResult(0, 7, ("owner",), ()).draft_ingress is False


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_masked_continuation_advances_canonical_rows_without_new_grants(depth, order):
    from types import SimpleNamespace
    from cubic_serve.runtime.continuous_batch import advance_carried_draft_rows
    from cubic_serve.runtime.quantized_qualification import _draft_window_audit_rows

    round_ = SimpleNamespace(depth=depth, stride=depth + 1, C=2, target=SimpleNamespace(
        body=SimpleNamespace(arena=SimpleNamespace(shared_storage=None)), _block_size=16, _scratch_states=tuple(range(10, 30)), vocab_size=256))
    previous = [(f"audit:draft-window:{owner}", 14, 70, None, 11, None, (71,) * depth)
                for owner in range(2)]
    rows = _draft_window_audit_rows(round_, previous, 2, 3, owner_order=order)
    # Use the canonical immutable sampling instances, as real pending records do.
    pending = [(f"audit:draft-window:{owner}", 14 + owner + 1, 80 + owner,
                rows[order.index(owner) * (depth + 1)].sampling, 12 + owner,
                None, tuple(range(90, 90 + depth))) for owner in range(2)]
    actual = advance_carried_draft_rows(rows, pending, depth=depth, block_size=16)
    expected = _draft_window_audit_rows(round_, pending, 2, 3, owner_order=order)
    assert actual == expected
    axes_rows = tuple(replace(row, position_axes=(row.position, 2*row.position, 7)) for row in rows)
    axes_pending = [(*entry[:5], (entry[1], 2*entry[1], 7), entry[6]) for entry in pending]
    actual = advance_carried_draft_rows(axes_rows, axes_pending, depth=depth, block_size=16)
    assert all(row.position_axes == (row.position, 2*row.position, 7) for row in actual)
    for invalid in (
        [None, pending[1]],
        [(*pending[0][:1], 99, *pending[0][2:]), pending[1]],
        [("other", *pending[0][1:]), pending[1]],
    ):
        with pytest.raises(BatchProtocolError):
            advance_carried_draft_rows(rows, invalid, depth=depth, block_size=16)


@pytest.mark.parametrize("terminal", ["count", "decision", "interrupt", "late_interrupt", "eos", "grammar_end"])
@pytest.mark.parametrize("forward_ahead", [False, True])
def test_masked_continuation_rendezvous_and_immutable_publication(terminal, forward_ahead):
    from types import SimpleNamespace
    from cubic_serve.contracts.request import StructuredOutputKind, StructuredOutputMeta
    from cubic_serve.generation.structured_sampling import ConstraintBinding
    from cubic_serve.runtime.continuous_batch import (
        BatchCommand, BatchResult, BatchRow, DraftWindowBinding, SegmentResult,
        run_grammar_draft_continuation,
    )

    row = BatchRow("a", 1, 0, 0, 7, 2, 2, (2, 3), segment_length=2,
                   mtp_position=0, mtp_page_destination=4, mtp_page_table=(4, 5),
                   is_decode=True)
    rows = (row, replace(row, token_id=2, position=1, mtp_position=1,
                         sample_ordinal=1, state_destination_slot=0, segment_length=0))
    constraint = ConstraintBinding("a", StructuredOutputMeta(StructuredOutputKind.REGEX, '".*"'), 0, 20)
    template = BatchCommand(7, 2, rows, (constraint,))
    pending = [("a", 2, 8, row.sampling, 2, None, (9,))]
    round_ = SimpleNamespace(depth=1, grammar_continuation_qualified=True,
                             _last_review_only=True, pool=SimpleNamespace(metadata=pending))
    binding = DraftWindowBinding(1, (0,), (2,), (32,), (32,), (20,), ((2, 3),), ((4, 5),))
    if terminal == "decision":
        # Both token and round observations must finish, before forward-ahead
        # may launch another target; verify the identical stop on both ranks.
        binding = replace(binding, decision_tokens=(3,), decision_rounds=(2,))
    command = ContinueBatchCommand(8, 7, 2, ("a",), max_steps=3,
                                   eos_token_ids=((99,),), draft=binding)
    barrier = threading.Barrier(2)
    statuses, receipts, errors, progressed = [0] * 4, [], [], [0, 0]
    events = [[], []]

    def worker(rank):
        local = SimpleNamespace(**round_.__dict__)
        local.pool = SimpleNamespace(metadata=list(pending))
        local.constraint_continuation_qualified = forward_ahead
        local._last_target_constraints = forward_ahead
        local.bind_constraint_continuation = lambda _: events[rank].append('bind')
        local.start_constraint_forward = lambda: events[rank].append('forward')
        local.close_constraint_continuation = lambda: events[rank].append('close')
        pool = SimpleNamespace(sessions={"a": SimpleNamespace(revision=2)})
        executor = SimpleNamespace(captured_m=2, _pipelined_draft_round=local, _block_size=16,
                                   slot=SimpleNamespace(synchronize=lambda: None))
        def execute(next_command):
            events[rank].append('execute')
            root = next_command.rows[0]
            assert next_command.constraints[0].revision == root.sample_ordinal
            assert next_command.constraints[0].remaining == 22 - root.sample_ordinal
            progressed[rank] += 1
            tokens = (8, 99) if terminal == "eos" else (8, 9)
            local.pool.metadata[0] = ("a", root.mtp_position + 2, 8, row.sampling,
                                      root.sample_ordinal + 2, None, (9,))
            pool.sessions["a"].revision += 2
            if terminal == "grammar_end":
                pool.sessions.clear()
            return BatchResult(rank, 8, ("a", "a"), (),
                               (SegmentResult("a", 2, tokens),) if rank == 0 else (),
                               pipelined_draft_depth=1)
        try:
            run_grammar_draft_continuation(
                command, executor, template=template, execute=execute,
                previous_batch_id=7, request_ids=("a",), rank=rank, sampler_rank=0,
                completion_statuses=statuses, completion_barrier=barrier, timeout=3.,
                interrupt=SimpleNamespace(is_set=lambda: terminal == "interrupt"
                    or terminal == "late_interrupt" and bool(receipts)),
                publish=lambda result: (events[rank].append('publish'), receipts.append(result)),
                grammar_pool=pool)
        except BaseException as error:
            errors.append(error)
            barrier.abort()
    workers = [threading.Thread(target=worker, args=(rank,)) for rank in range(2)]
    for worker_thread in workers: worker_thread.start()
    for worker_thread in workers: worker_thread.join(5.)
    assert not errors, errors
    assert not any(worker_thread.is_alive() for worker_thread in workers)
    count = 3 if terminal == "count" else 2 if terminal in ("late_interrupt", "decision") else 1
    assert progressed == [count, count]
    assert [receipt.continuation_index for receipt in receipts] == list(range(count))
    assert [receipt.continuation_complete for receipt in receipts] == [False] * (count - 1) + [True]
    assert all(receipt.request_ids == ("a",) and receipt.pipelined_draft_depth == 1 for receipt in receipts)
    if forward_ahead:
        assert events[0] == (['bind', 'forward', 'execute']
                            + ['forward', 'publish', 'execute'] * (count - 1)
                            + ['publish', 'close'])
        assert events[1] == ['bind'] + ['forward', 'execute'] * count + ['close']


@pytest.mark.parametrize("terminal", ["count", "eos", "interrupt", "late_interrupt"])
@pytest.mark.parametrize("owner_count", [1, 3])
@pytest.mark.parametrize("cohort_masks", [False, True])
def test_masked_k0_continuation_preserves_prefix_and_all_rank_completion(terminal, owner_count, cohort_masks):
    import torch
    from types import SimpleNamespace as S
    from cubic_serve.contracts.request import ReasoningControl, StructuredOutputMeta
    from cubic_serve.generation.structured_sampling import ConstraintBinding, GrammarPool
    from cubic_serve.runtime.continuous_batch import run_grammar_continuation
    from cubic_serve.runtime.performance import _constraint_prefix_masks_with_overlap

    ids = tuple(f"owner:{i}" for i in range(owner_count))
    control = ReasoningControl((7,), (9, 10), 1, True)
    bindings = tuple(ConstraintBinding(key, StructuredOutputMeta.identity(), 0, 20,
                                       reasoning_control=control) for key in ids)
    command = ContinueBatchCommand(8, 7, 4, ids, max_steps=3,
        eos_token_ids=tuple((9,) if terminal == 'eos' else (1,) for _ in ids),
        max_duration_seconds=10.)
    barrier, statuses = threading.Barrier(2), [0] * 4
    receipts, errors, final, events = [], [], [], [[], []]

    def worker(rank):
        pool = GrammarPool(None, vocab_size=16, stop_token_ids=(1,))
        initial = pool.begin(bindings)
        for key in ids:
            assert initial.mask(key, 0) is None
        initial.commit({key: (2,) for key in ids})
        progress = 0
        def bind(masks, owners):
            assert owners == [True] * owner_count
            expected = (9, 10, None)[progress]
            for mask in masks:
                assert (mask is None if expected is None else int(mask[0]) == 1 << expected)
        executor = S(captured_m=4, _sampling=S(bind_constraints=bind, constraint_bound=False,
                                              masked_plans=(S(qualified=True),)),
                     _input_update_graph=object(),
                     _token_ids=torch.zeros(4, dtype=torch.int64),
                     slot=S(synchronize=lambda: None), replay_elapsed_ms=lambda: 1.)
        def replay(count, updates):
            assert count == owner_count and not updates
            events[rank].append('forward')
        def sample(count):
            nonlocal progress
            token = (9, 10, 2)[progress]
            executor._token_ids[:count].fill_(token)
            progress += 1
            events[rank].append('sample')
            return (token,) * count if rank == 0 else ()
        executor.replay_steady_continuation, executor.sample_steady_continuation = replay, sample
        try:
            outcome = run_grammar_continuation(
                command, executor, bindings=bindings, previous_batch_id=7,
                request_ids=ids, offsets=(0,) * owner_count, page_counts=(1,) * owner_count,
                block_size=16, rank=rank, sampler_rank=0,
                completion_statuses=statuses, completion_barrier=barrier, timeout=3.,
                interrupt=S(is_set=lambda: terminal == 'interrupt'
                            or terminal == 'late_interrupt' and bool(receipts)),
                publish=lambda result: (events[rank].append('publish'), receipts.append(result)),
                grammar_pool=pool, prepare_masks=(_constraint_prefix_masks_with_overlap
                                                 if cohort_masks else None))
            final.append((rank, progress, outcome, tuple(pool.sessions[key].revision for key in ids)))
        except BaseException as error:
            errors.append(error);barrier.abort()
    workers = [threading.Thread(target=worker, args=(rank,)) for rank in range(2)]
    for thread in workers:thread.start()
    for thread in workers:thread.join(5.)
    assert not errors, errors
    assert not any(thread.is_alive() for thread in workers)
    count = 3 if terminal == 'count' else 2 if terminal == 'late_interrupt' else 1
    assert len(final) == 2
    assert all(progress == count and revisions == (count + 1,) * owner_count
               and result[:2] == ((count,) * owner_count, (1,) * owner_count)
               for _, progress, result, revisions in final)
    assert [r.continuation_index for r in receipts] == list(range(count))
    assert [r.continuation_complete for r in receipts] == [False] * (count - 1) + [True]
    assert all(r.grammar_cost.valid() and r.request_ids == ids for r in receipts)
    assert events[0] == ['forward', 'sample'] + ['forward', 'publish', 'sample'] * (count - 1) + ['publish']
    assert events[1] == ['forward', 'sample'] * count


def test_masked_k0_rejects_unowned_and_cross_page_windows_before_forward():
    from types import SimpleNamespace as S
    from cubic_serve.contracts.request import ReasoningControl, StructuredOutputMeta
    from cubic_serve.generation.structured_sampling import ConstraintBinding
    from cubic_serve.runtime.continuous_batch import run_grammar_continuation
    binding = ConstraintBinding('a', StructuredOutputMeta.identity(), 0, 20,
                                reasoning_control=ReasoningControl((7,), (9,), 4, True))
    command = ContinueBatchCommand(8, 7, 4, ('a',), max_steps=3, eos_token_ids=((1,),))
    for damage in ('owner', 'page', 'remaining', 'previous', 'compaction', 'future', 'unqualified'):
        kwargs = dict(bindings=(binding,), previous_batch_id=7, request_ids=('a',),
                      offsets=(0,), page_counts=(1,), block_size=16, rank=0, sampler_rank=0,
                      completion_statuses=[0, 0], completion_barrier=None, timeout=1.,
                      interrupt=None, publish=None, grammar_pool=S())
        altered = command
        if damage == 'owner':kwargs['bindings'] = (replace(binding, request_id='foreign'),)
        elif damage == 'page':kwargs['offsets'] = (14,)
        elif damage == 'remaining':kwargs['bindings'] = (replace(binding, remaining=3),)
        elif damage == 'previous':kwargs['previous_batch_id'] = 6
        elif damage == 'compaction':altered = replace(command, source_rows=(0,))
        elif damage == 'future':altered = replace(command, future_page_updates=((1, 0, 1, 4),))
        executor = S(captured_m=4, _input_update_graph=object(),
                     _sampling=S(masked_plans=(S(qualified=damage != 'unqualified'),)))
        with pytest.raises(BatchProtocolError):
            run_grammar_continuation(altered, executor, **kwargs)


def test_publication_exports_complete_depth_portfolio_not_last_selected_round():
    from types import SimpleNamespace as S
    from cubic_serve.executor.resident_window import resident_publication_names
    def round_(name):
        journal = S(name=name)
        return S(_resident_window=S(_shared_journal=journal),
                 _candidate_window=S(_shared_journal=journal))
    low, high = round_("lower"), round_("maximum")
    executor = S(_pipelined_draft_round=high, _pipelined_draft_rounds={1: low, 3: high})
    assert resident_publication_names({8: executor}) == ("lower", "maximum")
    executor._pipelined_draft_round = low
    assert resident_publication_names({8: executor}) == ("lower", "maximum")


def _grammar_append_fixture(depth=1, count=4, *, block_size=16):
    import time
    from cubic_serve.contracts.request import StructuredOutputMeta
    from cubic_serve.runtime.continuous_batch import BatchRow, GenerateCommand
    from cubic_serve.runtime.performance import _ProcessController, _ScheduledRequest

    c = _ProcessController.__new__(_ProcessController)
    c._block_size = block_size
    c.max_sequence_tokens = 4096
    c._max_speculative_tokens = c._num_speculative_tokens = depth
    c._dynamic_speculation = False
    c._shared_state_slots = {}; c._shared_pages = {}; c._shared_mtp_pages = {}
    c._free_pages = list(range(10000, 20000))
    c._free_mtp_pages = list(range(20000, 30000))
    c._graph_capacities = tuple(1 << i for i in range(11))
    capacity = 1 << (count * (depth + 1) - 1).bit_length()
    c._grammar_draft_continuation_capacities = (capacity,)
    c._resident_draft_capacities = (capacity,)
    c._worker_timeout = 100.
    c._decode_window_seconds = None
    c._resident_draft_step_seconds = .01
    c._batch_id = 8
    records, rows = [], []
    stride = depth + 1
    for i in range(count):
        position = 32 + i % 16
        target_count = (position + stride + block_size - 1) // block_size
        provider_count = (position + stride + depth + block_size - 1) // block_size
        record = _ScheduledRequest(
            GenerateCommand(str(i), (11, 12), 1024, (99,),
                            structured_output=StructuredOutputMeta.regex(".*")),
            None, time.monotonic(), prefill_cursor=2, generated_count=position - 1,
            last_token_id=13, state_slot=i, state_initialized=True,
            pages=list(range(i * 128, i * 128 + target_count)),
            mtp_pages=list(range(4096 + i * 128, 4096 + i * 128 + provider_count)),
        )
        records.append(record)
        # Distinct COW destinations exercise the real normal-batch table view.
        replay = tuple(page + 50000 if first <= j <= last else page
                       for j, page in enumerate(record.pages)
                       for first, last in [(position // block_size,
                                            (position + depth) // block_size)])
        for d in range(stride):
            page = (position + d) // block_size
            rows.append(BatchRow(
                str(i), 13, position + d, i, count + i * depth + d if d < depth else i,
                record.pages[page], replay[page], replay,
                segment_length=stride if d == 0 else 0, mtp_position=position + d,
                mtp_page_destination=record.mtp_pages[page],
                mtp_page_table=tuple(record.mtp_pages), is_decode=True,
            ))
    c._requests = {r.command.request_id: r for r in records}
    return c, tuple(rows), tuple(records)


@pytest.mark.parametrize('depth', [1, 2, 3, 7])
@pytest.mark.parametrize('count', [1, 4, 32])
def test_grammar_append_grants_preserve_cow_and_fund_existing_continuation(depth, count):
    c, rows, records = _grammar_append_fixture(depth, count)
    original_pages = tuple(tuple(r.pages) for r in records)
    original_provider = tuple(tuple(r.mtp_pages) for r in records)
    target_free, provider_free = len(c._free_pages), len(c._free_mtp_pages)
    granted = c._grant_grammar_draft_lookahead_locked(rows, records)
    stride = depth + 1
    fresh = set()
    for i, record in enumerate(records):
        start = i * stride
        root, original = granted[start], rows[start]
        assert tuple(record.pages[:len(original_pages[i])]) == original_pages[i]
        assert tuple(record.mtp_pages[:len(original_provider[i])]) == original_provider[i]
        assert root.page_table[:len(original.page_table)] == original.page_table
        assert root.mtp_page_table[:len(original.mtp_page_table)] == original.mtp_page_table
        added = tuple(record.pages[len(original_pages[i]):])
        assert root.new_pages == (*original.new_pages, *added)
        assert not fresh.intersection(added)
        fresh.update(added)
        for offset, row in enumerate(granted[start:start + stride]):
            assert row.page_source == rows[start + offset].page_source
            assert row.page_destination == rows[start + offset].page_destination
            assert row.state_source_slot == rows[start + offset].state_source_slot
            assert row.state_destination_slot == rows[start + offset].state_destination_slot
        # Complete the current maximal prefix exactly as the normal commit does.
        record.pages[:] = root.page_table
        record.generated_count += stride
    assert target_free - len(c._free_pages) == len(fresh)
    assert provider_free - len(c._free_mtp_pages) == sum(
        len(r.mtp_pages) - len(old) for r, old in zip(records, original_provider))
    c._resident_draft = (7, c._grammar_draft_continuation_capacities[0], depth, records)
    c._resident_grammar_draft = True
    result = c._next_draft_continuation_locked()
    assert result is not None and result[0].max_steps >= 2
    assert result[0].draft.target_tables == tuple(tuple(r.pages) for r in records)


@pytest.mark.parametrize('depth,maximum', [(2, 3), (4, 7), (6, 7)])
@pytest.mark.parametrize('block_size', [16, 32])
@pytest.mark.parametrize('qualified', [False, True])
def test_grammar_append_grants_admit_declared_intermediate_dynamic_depth(
        depth, maximum, block_size, qualified):
    from cubic_serve.runtime.speculative import DynamicDraftPolicy

    c, rows, records = _grammar_append_fixture(depth, count=4, block_size=block_size)
    c._max_speculative_tokens = maximum
    c._dynamic_speculation = True
    for record in records:
        record.draft_policy = DynamicDraftPolicy(depth, maximum)
    if not qualified:
        c._grammar_draft_continuation_capacities = ()
    before = (tuple(c._free_pages), tuple(c._free_mtp_pages),
              tuple((tuple(r.pages), tuple(r.mtp_pages)) for r in records))
    granted = c._grant_grammar_draft_lookahead_locked(rows, records)
    if not qualified:
        assert granted is rows
        assert before == (tuple(c._free_pages), tuple(c._free_mtp_pages),
                          tuple((tuple(r.pages), tuple(r.mtp_pages)) for r in records))
        return
    assert granted is not rows
    for original, expanded in zip(rows, granted, strict=True):
        assert replace(expanded, page_table=original.page_table,
                       mtp_page_table=original.mtp_page_table,
                       new_pages=original.new_pages) == original


@pytest.mark.parametrize('damage', ['ordinary', 'target_pressure', 'provider_pressure',
                                   'unqualified', 'shared_state', 'remaining', 'context',
                                   'position_transition', 'policy_boundary'])
def test_optional_grammar_append_grants_leave_ineligible_admission_unchanged(damage):
    c, rows, records = _grammar_append_fixture()
    if damage == 'ordinary':
        for record in records:
            record.command = replace(record.command, structured_output=None)
    elif damage == 'target_pressure': c._free_pages.clear()
    elif damage == 'provider_pressure': c._free_mtp_pages.clear()
    elif damage == 'unqualified': c._grammar_draft_continuation_capacities = ()
    elif damage == 'shared_state': c._shared_state_slots[records[0].state_slot] = 2
    elif damage == 'remaining': records[0].generated_count = 1022
    elif damage == 'context': c.max_sequence_tokens = 35
    elif damage == 'position_transition':
        records[0].command = replace(records[0].command, decode_position_step=(1, 2, 3))
    elif damage == 'policy_boundary':
        from types import SimpleNamespace as S
        c._dynamic_speculation = True
        for record in records:
            record.draft_policy = S(stopped=False, current_tokens=1, rounds_until_decision=2)
    before = (tuple(c._free_pages), tuple(c._free_mtp_pages),
              tuple((tuple(r.pages), tuple(r.mtp_pages)) for r in records))
    assert c._grant_grammar_draft_lookahead_locked(rows, records) is rows
    assert before == (tuple(c._free_pages), tuple(c._free_mtp_pages),
                      tuple((tuple(r.pages), tuple(r.mtp_pages)) for r in records))


def test_grammar_append_grants_obey_existing_reset_index_capacity():
    c, rows, records = _grammar_append_fixture(block_size=1)
    granted = c._grant_grammar_draft_lookahead_locked(rows, records)
    capacity = c._grammar_draft_continuation_capacities[0]
    count = sum(len(row.new_pages) for row in granted)
    assert count > capacity  # Extra input initialization uses existing staging.
    assert count <= capacity * c.max_sequence_tokens
    assert len({page for row in granted for page in row.new_pages}) == count


@pytest.mark.parametrize('budget', ['duration', 'unpriced'])
def test_grammar_append_grants_bound_optional_pages_by_the_actual_window(budget):
    c, rows, records = _grammar_append_fixture(count=1, block_size=4)
    if budget == 'duration':
        c._decode_window_seconds = .021
    else:
        c._resident_draft_step_seconds = 0.
    granted = c._grant_grammar_draft_lookahead_locked(rows, records)
    # Only the existing three-round minimum is funded by a short/unpriced window.
    root = rows[0]
    expected_target = (root.position + 3 * 2 + 3) // 4
    expected_provider = (root.mtp_position + 3 * 2 + 1 + 3) // 4
    assert len(records[0].pages) == expected_target
    assert len(records[0].mtp_pages) == expected_provider
    assert len(granted[0].new_pages) <= c._grammar_draft_continuation_capacities[0]


@pytest.mark.parametrize('age', [600.01, 3600.01, 86400.])
def test_grammar_append_grants_are_independent_of_request_age(age):
    import time
    young, rows, records = _grammar_append_fixture(count=1, block_size=4)
    old, old_rows, old_records = _grammar_append_fixture(count=1, block_size=4)
    for record in old_records:
        record.admitted_at = time.monotonic() - age
    expected = young._grant_grammar_draft_lookahead_locked(rows, records)
    actual = old._grant_grammar_draft_lookahead_locked(old_rows, old_records)
    assert actual == expected
    assert tuple(old._free_pages) == tuple(young._free_pages)
    assert tuple(old._free_mtp_pages) == tuple(young._free_mtp_pages)
    assert tuple(old_records[0].pages) == tuple(records[0].pages)
    assert tuple(old_records[0].mtp_pages) == tuple(records[0].mtp_pages)


def test_grammar_append_grants_follow_permuted_logical_owners_without_other_row_changes():
    c, rows, records = _grammar_append_fixture()
    order = (2, 0, 3, 1)
    rows = tuple(row for owner in order for row in rows[2 * owner:2 * owner + 2])
    records = tuple(records[owner] for owner in order)
    granted = c._grant_grammar_draft_lookahead_locked(rows, records)
    for i, record in enumerate(records):
        root = granted[2 * i]
        assert root.request_id == record.command.request_id
        assert root.state_source_slot == record.state_slot
        assert root.page_table[len(rows[2 * i].page_table):] == tuple(
            record.pages[len(rows[2 * i].page_table):])
    for before, after in zip(rows, granted, strict=True):
        assert replace(after, page_table=before.page_table, mtp_page_table=before.mtp_page_table,
                       new_pages=before.new_pages) == before


def test_trimmed_grammar_tables_require_new_normal_bind_before_reuse():
    from types import SimpleNamespace as S
    from cubic_serve.runtime.continuous_batch import DraftWindowBinding
    c, rows, records = _grammar_append_fixture()
    capacity = c._grammar_draft_continuation_capacities[0]
    binding = DraftWindowBinding(1, tuple(r.state_slot for r in records), (32,) * len(records),
                                 (64,) * len(records), (64,) * len(records), (100,) * len(records),
                                 tuple(tuple(r.pages) for r in records),
                                 tuple(tuple(r.mtp_pages) for r in records))
    command = ContinueBatchCommand(8, 7, capacity, tuple(r.command.request_id for r in records),
                                   max_steps=2, draft=binding)
    result = S(pipelined_draft_depth=1, resident_draft_depth=0, step_ms=10.)
    c._remember_draft_continuation_locked(command, records, result)
    assert c._resident_draft is not None
    records[0].pages.pop()
    c._remember_draft_continuation_locked(command, records, result)
    assert c._resident_draft is None


@pytest.mark.parametrize('hold', [False, True])
@pytest.mark.parametrize('grammar', [False, True])
def test_completed_grammar_window_keeps_grants_until_the_actual_reuse_decision(hold, grammar):
    from types import SimpleNamespace as S
    from cubic_serve.runtime.continuous_batch import (
        BatchResult, SegmentResult, DraftWindowBinding,
    )
    from cubic_serve.runtime.speculative import DynamicDraftPlanner, DynamicDraftPolicy
    from cubic_serve.runtime.token_stream import TokenStream, TokenWakeBatch

    c, _, records = _grammar_append_fixture(count=4, block_size=64)
    c._dynamic_speculation = True
    c._decode_window_seconds = .5
    planner = DynamicDraftPlanner(c._graph_capacities)
    planner.seed_costs(target_samples=((4, 10.), (8, 16.)), proposal_samples=((4, 1.),))
    planner.observe_compound_cost(
        target_rows=8, segment_depth=2, milliseconds=12. if hold else 1000.)
    c._draft_cost_planner = lambda *a: planner
    c._serving_stats = S(speculate=lambda *a, **k: None, publish=lambda *a, **k: None)
    c._finish_request_stats_locked = lambda *a: None
    for i, record in enumerate(records):
        record.output = TokenStream()
        record.generated_count += 48
        record.draft_tokens = 1
        record.draft_policy = DynamicDraftPolicy(1, 1)
        for _ in range(24):
            record.draft_policy.observe(accepted=1, proposed=1)
        record.pages = list(range(i * 128, i * 128 + 8))
        record.mtp_pages = list(range(4096 + i * 128, 4096 + i * 128 + 8))
    ids = tuple(r.command.request_id for r in records)
    before = tuple((tuple(r.pages), tuple(r.mtp_pages)) for r in records)
    binding = DraftWindowBinding(
        1, tuple(r.state_slot for r in records),
        tuple(c._next_input_position(r) for r in records),
        (512,) * 4, (512,) * 4, (900,) * 4,
        tuple(pages for pages, _ in before), tuple(pages for _, pages in before))
    batch = ContinueBatchCommand(7, 6, 8, ids, max_steps=2, draft=binding)
    result = BatchResult(
        0, 7, ids, (), tuple(SegmentResult(i, 2, (5, 6)) for i in ids),
        pipelined_draft_depth=int(grammar), resident_draft_depth=int(not grammar),
        continuation_complete=True, step_ms=10.)
    c._commit_draft_receipt_locked(batch, records, result, TokenWakeBatch(), profile=None)
    kept = tuple((tuple(r.pages), tuple(r.mtp_pages)) for r in records)
    if grammar:
        assert before == kept
    else:
        assert all(len(r.pages) == len(r.mtp_pages) == 2 for r in records)
    c._remember_draft_continuation_locked(batch, records, result)

    class OrdinaryPlannerReached(Exception):
        pass

    def ordinary():
        raise OrdinaryPlannerReached

    c._next_segment_continuation_locked = lambda: None
    c._resident_survivor_rows_locked = ordinary
    if hold:
        command, _, selected = c._next_dispatch_locked()
        assert command.max_steps >= 2 and selected == records
        assert command.draft.target_tables == tuple(p for p, _ in kept)
        assert kept == tuple((tuple(r.pages), tuple(r.mtp_pages)) for r in records)
    else:
        with pytest.raises(OrdinaryPlannerReached):
            c._next_dispatch_locked()
        # These concrete positions need two pages. Every unused grant is freed
        # before the ordinary allocator can reuse storage for another owner.
        assert all(len(r.pages) == len(r.mtp_pages) == 2 for r in records)
        assert all(set(pages[2:]) <= set(c._free_pages) for pages, _ in before)
        assert all(set(pages[2:]) <= set(c._free_mtp_pages) for _, pages in before)


@pytest.mark.skipif(os.environ.get('CUBIC_TEST_QUANT_CUDA') != '1', reason='explicit CUDA control audit required')
def test_native_draft_decision_budget_matches_actual_work_and_protects_padding():
    import json, random
    from pathlib import Path
    import torch
    from cuda.bindings import runtime as rt
    from cubic_serve.executor.resident_window import check
    from cubic_serve.ops.cuda.native import CudaScalar
    from cubic_serve.ops.cuda.native_compiler import compile_native
    rng = random.Random(2171)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    source = (Path(__file__).parents[2] / 'src/cubic_serve/ops/cuda/sampling/resident_draft_window.cu').read_bytes()
    module, entry = compile_native((('entry.cu', source),), device, 'resident_draft_window')
    checked = 0
    for depth in (1, 2, 3):
        C = 4
        CS = depth + 8
        V = 128
        meta = torch.tensor([3, 100, 0, 0, 0], dtype=torch.int32, device=device)
        ctx = torch.zeros(1 + C * CS, dtype=torch.int32, device=device)
        ctx[1:].view(C, CS)[:, 2:2 + depth] = 11
        result = torch.full((C, depth + 3), 11, dtype=torch.int64, device=device)
        result[:, depth + 2] = 1
        ordinals = torch.zeros(C, dtype=torch.int64, device=device)
        metadata = torch.zeros((C * (depth + 1), 3), dtype=torch.int32, device=device)
        ready = torch.ones(C, dtype=torch.bool, device=device)
        remaining = torch.zeros((C, 3), dtype=torch.int64, device=device)
        offsets = torch.zeros(C + 1, dtype=torch.int32, device=device)
        eos = torch.zeros(1, dtype=torch.int64, device=device)
        clocks = torch.zeros((100, 4), dtype=torch.int64, device=device)
        policy = torch.zeros(2, dtype=torch.uint64, device=device)
        host = torch.zeros(64 + 100 * (C * (2 * depth + 8) + 6), dtype=torch.int64, pin_memory=True)
        mapped = int(check(rt.cudaHostGetDevicePointer(host.data_ptr(), 0))[0])
        addr = torch.tensor([mapped], dtype=torch.uint64, device=device)
        args = (meta, ctx, result, ordinals, metadata, ready, remaining, offsets, eos, clocks, policy, addr, CudaScalar('i32', C), CudaScalar('i32', depth), CudaScalar('i32', CS), CudaScalar('i32', 3), CudaScalar('i32', 0), CudaScalar('i32', V), CudaScalar('i32', 1), CudaScalar('i64', 0))
        calls = [module.bind(entry, (*args, CudaScalar('i32', phase)), operator_id='runtime.resident_draft_control', grid=(1, 1, 1), block=(128, 1, 1)) for phase in (3, 4)]
        for enabled in (False, True):
            for trial in range(40):
                expected = [[10000, rng.randrange(25), rng.randrange(1, 9)] if enabled else [10000, -1, -1] for _ in range(3)] + [[971, 972, 973]]
                remaining.copy_(torch.tensor(expected, device=device))
                meta[2:] = 0
                host.zero_()
                for step in range(40):
                    accepted = [rng.randrange(depth + 1) for _ in range(3)]
                    result[:3, 0] = torch.tensor(accepted, device=device)
                    meta[2] = step
                    for row, count in zip(expected, accepted):
                        row[0] -= count + 1
                        if enabled:
                            row[1] = max(0, row[1] - count - 1)
                            row[2] = max(0, row[2] - 1)
                    calls[0].launch()
                    torch.cuda.synchronize()
                    actual = remaining.cpu().tolist()
                    assert actual == expected, (depth, trial, actual, expected)
                    stop = enabled and (not any((x for row in expected[:3] for x in row[1:])))
                    assert int(meta[4]) == int(stop)
                    calls[1].launch()
                    torch.cuda.synchronize()
                    assert remaining.cpu().tolist() == expected
                    checked += 1
                    if stop:
                        break
                    if not enabled and step == 4:
                        break
    assert checked > 1000
