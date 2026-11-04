from __future__ import annotations

import unittest
from dataclasses import replace

import torch
from torch import Tensor

from cubic_serve.contracts import execution as ec
from cubic_serve.contracts.request import TokenPosition
from cubic_serve.guardrail.graph_padding import (
    GraphQualificationRun,
    GraphQualificationStateFactory,
)
from cubic_serve.models.qwen3_5 import model as qc
from cubic_serve.runtime import cache as rc
from cubic_serve.runtime import model_state as ps

_BLOCK = 2
_CAPACITY = (("context_pages", 2), ("request_slots", 4))
_STARTS = (0, 2, 3, 1)
_ROWS: dict[int, tuple[ec.LaunchRow, ...]] = {
    1: (
        ec.PaddingLaunchRow(0),
        ec.ActiveLaunchRow(0),
        ec.PaddingLaunchRow(1),
        ec.PaddingLaunchRow(2),
    ),
    2: (
        ec.ActiveLaunchRow(1),
        ec.PaddingLaunchRow(0),
        ec.ActiveLaunchRow(0),
        ec.PaddingLaunchRow(1),
    ),
    4: (ec.ActiveLaunchRow(2), ec.ActiveLaunchRow(0), ec.ActiveLaunchRow(3), ec.ActiveLaunchRow(1)),
}


def _config() -> qc.Qwen3_5Config:
    schedule = (
        qc.AttentionKind.GATED_DELTA_NET,
        qc.AttentionKind.FULL_ATTENTION,
        qc.AttentionKind.GATED_DELTA_NET,
        qc.AttentionKind.FULL_ATTENTION,
    )
    text = qc.Qwen3_5TextConfig(
        "qwen3_5_text",
        "bfloat16",
        16,
        8,
        len(schedule),
        64,
        "silu",
        1e-6,
        schedule,
        2,
        16,
        qc.FullAttentionConfig(2, 1, 2, True),
        qc.GatedDeltaNetConfig(3, 2, 3, 2, 1, "float32", "swish"),
        None,
        qc.MRoPEConfig(True, (1, 1, 1), 1.0, 10_000.0, "default"),
        qc.MTPMetadata(qc.MTPStatus.METADATA_ONLY, 1, False),
    )
    graph = qc.Qwen3_5GraphConfig(
        qc.Qwen3_5GraphVariant.DENSE,
        "qwen3_5",
        "fixture",
        text,
        qc.VisionConfig("qwen3_5", 1, 2, 4, 8, 1, 3, 2, 1, 2, 4, "gelu", ()),
        qc.ModalityTokenConfig(12, 13, 14, 15),
    )
    return qc.Qwen3_5Config(
        qc.ModelIdentity("fixture", "tiny", "qwen3_5", "qwen3_5", "fixture"), graph
    )


def _state() -> ps.DecodeState:
    schema = ps.build_decode_state_schema(
        _config(),
        arena_id="qwen:rank0",
        page_capacity=16,
        state_capacity=8,
        scratch_capacity=3,
        scratch_arena_ids=("scratch:0",),
        max_pages=2,
        block_size=_BLOCK,
    )
    return ps.DecodeState(schema, "cpu")


def _pages(row: int, start: int) -> tuple[int, ...]:
    count = (start + 1 + _BLOCK - 1) // _BLOCK
    return tuple(4 + row * 3 + index for index in range(count))


def _contract(segment: ec.PlanSegment, revision: int = 0) -> ec.StateTxn:
    owner, page = segment.request_id, segment.cache_pages[-1]
    kv = f"kv:{owner}"
    state_ids = tuple(f"state:{state}:{owner}" for state in ps.STATE_IDS)
    reads = (
        ec.StateAccess(kv, 0, pages=tuple(sorted(segment.cache_pages))),
        *(ec.StateAccess(state_id, 0) for state_id in state_ids),
    )
    writes = (
        ec.StateAccess(kv, 0, segment.cache_slots, (page,)),
        *(ec.StateAccess(state_id, 0) for state_id in state_ids),
    )
    return ec.StateTxn(
        f"txn:{owner}:{revision}", owner, revision, revision + 1, reads, writes, (0,)
    )


def _batch(logical_m: int) -> ec.BatchPlan:
    segments = []
    for row, start in enumerate(_STARTS[:logical_m]):
        owner, pages = f"request:{row}", _pages(row, start)
        axes = (start + 7,) if row % 2 == 0 else (start + 7, start + 8, start + 9)
        segments.append(
            ec.PlanSegment(
                owner,
                f"b2s256:{row:x}".ljust(71, "0"),
                f"b2s256:{row + 4:x}".ljust(71, "0"),
                ec.BatchPhase.DECODE,
                True,
                start,
                1,
                (row + 1,),
                (TokenPosition(axes),),
                (pages[-1] * _BLOCK + start % _BLOCK,),
                pages,
            )
        )
    frozen = tuple(segments)
    return ec.BatchPlan(
        f"plan:{logical_m}",
        ec.BatchPhase.DECODE,
        tuple(segment.request_id for segment in frozen),
        frozen,
        ec.BatchShape(logical_m, _CAPACITY),
        "topology:0",
        "partition:0",
        tuple(_contract(segment) for segment in frozen),
    )


def _launch(batch: ec.BatchPlan) -> ec.LaunchPlan:
    logical_m = batch.shape.logical_m
    bucket = ec.GraphBucket(
        "bucket:decode",
        "template:qwen",
        "scope:qwen",
        "abi:qwen",
        batch.topology_id,
        batch.partition_id,
        ec.BatchPhase.DECODE,
        _CAPACITY,
        (1, 2, 4),
        4,
        "rows:v1",
        "scratch:v1",
        ("padding-private",),
        "evidence:fixture",
    )
    launch_id, execution_id = f"launch:{logical_m}", f"execution:{logical_m}"
    resources = (
        ec.RankReplayResources(0, "graph:0", "stream:0", "buffers:0", "workspace:0", "scratch:0"),
    )
    lease = ec.ReplayLease(
        f"lease:{logical_m}",
        1,
        launch_id,
        execution_id,
        bucket.bucket_id,
        bucket.template_id,
        bucket.topology_id,
        bucket.partition_id,
        resources,
    )
    return ec.LaunchPlan(
        launch_id,
        execution_id,
        batch.plan_id,
        logical_m,
        bucket,
        _ROWS[logical_m],
        lease,
    )


def _key(row: int, field: str) -> str:
    return f"row.{row:04d}.{field}"


def _seed(state: ps.DecodeState, batch: ec.BatchPlan) -> dict[str, Tensor]:
    result = {}
    layout, specs = state.schema.kv_layout, state.arena.state_specs
    for row, segment in enumerate(batch.segments):
        shape = (len(segment.cache_pages), *layout.page_shape)
        key = torch.zeros(shape, dtype=torch.bfloat16)
        value = torch.zeros_like(key)
        written = torch.zeros(shape[:-2], dtype=torch.bool)
        for logical_page in range(len(segment.cache_pages)):
            count = min(_BLOCK, max(0, segment.token_start - logical_page * _BLOCK))
            key[logical_page, :, :count].fill_(row + 1)
            value[logical_page, :, :count].fill_(-(row + 1))
            written[logical_page, :, :count] = True
        local = {
            ps.CONV_STATE_ID: torch.full(
                specs[ps.CONV_STATE_ID].shape, row + 1, dtype=torch.bfloat16
            ),
            ps.RECURRENT_STATE_ID: torch.full(
                specs[ps.RECURRENT_STATE_ID].shape, row + 1, dtype=torch.float32
            ),
            "kv_key": key,
            "kv_value": value,
            "kv_written": written,
            "position": torch.tensor(segment.token_start, dtype=torch.int64),
        }
        result.update({_key(row, field): tensor for field, tensor in local.items()})
    return result


def _emulate(session: ps.DecodeSession) -> dict[int, float]:
    backing = session.bindings[0]
    states = dict(backing.state_tensors)
    scratch_states = dict(backing.scratch_state_slots)
    for index, slot in enumerate(backing.scratch_page_slots):
        backing.key[slot].fill_(70 + index)
        backing.value[slot].fill_(-(70 + index))
        backing.written[slot].fill_(True)
        for state_id, slab in states.items():
            slab[scratch_states[state_id][index]].fill_(3 + index)
    values = {}
    for physical, row in enumerate(session.rows):
        value = float(20 + (physical if row.logical_index is None else row.logical_index))
        source, destination = row.page_cow
        backing.key[destination].copy_(backing.key[source])
        backing.value[destination].copy_(backing.value[source])
        backing.written[destination].copy_(backing.written[source])
        backing.key[destination, :, row.token_offset].fill_(value)
        backing.value[destination, :, row.token_offset].fill_(-value)
        backing.written[destination, :, row.token_offset] = True
        for state_id, (source, destination) in zip(ps.STATE_IDS, row.state_cow, strict=True):
            states[state_id][destination].copy_(states[state_id][source])
            states[state_id][destination].add_(value)
        page_index = (row.sequence_length - 1) // _BLOCK
        output = backing.key[row.page_table[page_index], :, 0].float().mean()
        assert bool(torch.isfinite(output).item())
        if row.logical_index is not None:
            values[row.logical_index] = value
    return values


class Qwen3_5PerformanceStateTests(unittest.TestCase):
    def test_schema_has_full_attention_kv_and_two_aggregate_gdn_slabs(self) -> None:
        state = _state()
        self.assertEqual(state.schema.kv_layout.layer_ids, (1, 3))
        self.assertEqual(state.schema.gdn_layer_ids, (0, 2))
        self.assertEqual(state.store.device, state.device)
        specs = state.arena.state_specs
        self.assertEqual(tuple(specs), ps.STATE_IDS)
        self.assertEqual(specs[ps.CONV_STATE_ID].shape, (2, 11, 2))
        self.assertEqual(specs[ps.RECURRENT_STATE_ID].shape, (2, 1, 3, 2))

    def test_m1_m2_m4_permutations_use_exact_cow_and_separate_commit(self) -> None:
        for logical_m in (1, 2, 4):
            with self.subTest(logical_m=logical_m):
                state, batch = _state(), _batch(logical_m)
                launch, seed = _launch(batch), _seed(state, batch)
                factory = ps.QualificationFactory(state, batch)
                self.assertIsInstance(factory, GraphQualificationStateFactory)
                run = factory.create(seed)
                self.assertIsInstance(run, GraphQualificationRun)
                session = run.begin(batch, launch)
                self.assertEqual(tuple(tx.plan.owner_id for tx in session.transactions), run.owners)
                self.assertEqual(len(set(run.owners)), logical_m)
                for physical, row in enumerate(session.rows):
                    source = launch.rows[physical]
                    if isinstance(source, ec.PaddingLaunchRow):
                        self.assertIsNone(row.logical_index)
                        self.assertIsNone(row.position)
                        self.assertEqual((row.sequence_length, row.token_offset), (1, 0))
                        self.assertEqual(row.page_cow[0], row.page_cow[1])
                        self.assertEqual(len(set(row.page_table)), 1)
                        self.assertTrue(all(a == b for a, b in row.state_cow))
                    else:
                        segment = batch.segments[source.logical_index]
                        self.assertEqual(row.logical_index, source.logical_index)
                        self.assertIs(row.position, segment.positions[0])
                        self.assertEqual(row.sequence_length, segment.token_start + 1)
                        self.assertNotEqual(row.page_cow[0], row.page_cow[1])
                with self.assertRaises(rc.TransactionStateError):
                    session.commit()
                values = _emulate(session)
                session.accept_and_verify()
                self.assertEqual(session.status, "verified")
                self.assertTrue(
                    all(state.store.current_revision(owner) == 0 for owner in run.owners)
                )
                with self.assertRaises(RuntimeError):
                    run.committed_snapshot()
                with self.assertRaises(rc.TransactionStateError):
                    state.snapshot(run.owners[0])
                consumed = tuple(
                    session.bindings[0].key[row.page_cow[1], :, row.token_offset].clone()
                    for row in session.rows
                    if row.logical_index is not None
                )
                self.assertTrue(all(bool(torch.isfinite(value).all().item()) for value in consumed))
                self.assertEqual(session.commit(), (1,) * logical_m)
                snapshot = run.committed_snapshot()
                self.assertEqual(
                    set(snapshot),
                    {_key(row, field) for row in range(logical_m) for field in ps.SEED_FIELDS},
                )
                for logical, segment in enumerate(batch.segments):
                    delta = values[logical]
                    self.assertTrue(
                        torch.equal(
                            snapshot[_key(logical, ps.CONV_STATE_ID)],
                            seed[_key(logical, ps.CONV_STATE_ID)] + delta,
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            snapshot[_key(logical, ps.RECURRENT_STATE_ID)],
                            seed[_key(logical, ps.RECURRENT_STATE_ID)] + delta,
                        )
                    )
                    self.assertEqual(
                        int(snapshot[_key(logical, "position")].item()), segment.token_start + 1
                    )
                    self.assertTrue(
                        bool(
                            snapshot[_key(logical, "kv_written")][
                                -1, :, segment.token_start % _BLOCK
                            ]
                            .all()
                            .item()
                        )
                    )
                run.close()
                retry = factory.create(seed)
                retry.abort()

    def test_verified_abort_releases_exact_pages_and_scratch(self) -> None:
        state, batch = _state(), _batch(2)
        factory, launch = ps.QualificationFactory(state, batch), _launch(batch)
        run = factory.create(_seed(state, batch))
        session = run.begin(batch, launch)
        _emulate(session)
        session.accept_and_verify()
        run.abort()
        retry = factory.create(_seed(state, batch))
        retry.abort()

    def test_page_slot_position_and_contract_mismatches_fail_closed(self) -> None:
        state, batch = _state(), _batch(1)
        run = ps.QualificationFactory(state, batch).create(_seed(state, batch))
        segment = batch.segments[0]
        cases = (
            replace(segment, cache_pages=(segment.cache_pages[0] + 1,)),
            replace(segment, cache_slots=(segment.cache_slots[0] + _BLOCK,)),
            replace(segment, token_start=1, cache_slots=(segment.cache_pages[-1] * _BLOCK + 1,)),
        )
        for bad in cases:
            rejected = replace(batch, segments=(bad,), state_transactions=(_contract(bad),))
            with self.assertRaises(ValueError):
                state.begin(rejected, _launch(rejected))
        bad_read = ec.StateAccess(f"kv:{segment.request_id}", 0)
        bad_contract = replace(
            batch.state_transactions[0], reads=(bad_read, *batch.state_transactions[0].reads[1:])
        )
        with self.assertRaises(ValueError):
            state.begin(replace(batch, state_transactions=(bad_contract,)), _launch(batch))
        run.abort()


if __name__ == "__main__":
    unittest.main()
