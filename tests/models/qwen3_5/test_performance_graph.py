from __future__ import annotations

import unittest
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import torch

from cubic_serve.contracts import execution as ec
from cubic_serve.executor import model_runner as pf
from cubic_serve.executor.cuda_graph import (
    AcquireDeadline,
    CudaGraphRejected,
    GraphProgram,
    GraphRuntimeBinding,
    ReplayIntent,
)
from cubic_serve.guardrail.graph_padding import GraphPaddingBuffers
from cubic_serve.models.qwen3_5.model import Qwen3_5ReferenceModel
from cubic_serve.runtime import model_state as ps
from tests.guardrail.operator_evidence_fixture import (
    sealed_test_composite_performance_plan,
    sealed_test_performance_plan,
)
from tests.models.qwen3_5.test_performance_state import (
    _batch,
    _config,
    _launch,
    _seed,
    _state,
)


class _Weights:
    device = torch.device("cpu")

    def __getitem__(self, name: str) -> torch.Tensor:
        raise KeyError(name)


class _Stream:
    def __init__(self) -> None:
        self.waits = self.syncs = 0

    def wait_stream(self, other: object) -> None:
        self.waits += 1

    def synchronize(self) -> None:
        self.syncs += 1


class _Graph:
    def __init__(self, callback: Any) -> None:
        self.callback = callback
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1
        self.callback()


def _program() -> tuple[pf.Qwen3_5DecodeGraphProgram, ps.DecodeState]:
    state = _state()
    model = Qwen3_5ReferenceModel(config=_config(), weights=cast(Any, _Weights()))
    return pf.Qwen3_5DecodeGraphProgram(model, state), state


def _one_axis_batch(logical_m: int) -> ec.BatchPlan:
    batch = _batch(logical_m)
    segments = tuple(
        replace(
            segment, positions=(replace(segment.positions[0], axes=(segment.token_start + 7,)),)
        )
        for segment in batch.segments
    )
    return replace(batch, segments=segments)


def _padding(state: ps.DecodeState, count: int) -> GraphPaddingBuffers:
    layout, specs = state.schema.kv_layout, state.arena.state_specs
    page = (count, *layout.page_shape)
    return GraphPaddingBuffers(
        "poison",
        {
            "token_ids": torch.arange(5, 5 + count, dtype=torch.int64),
            "position_ids": torch.arange(3 * count, dtype=torch.int64).reshape(3, count),
        },
        {
            "kv_key": torch.full(page, 3, dtype=torch.bfloat16),
            "kv_value": torch.full(page, -3, dtype=torch.bfloat16),
            "kv_written": torch.ones(page[:-2], dtype=torch.bool),
            ps.CONV_STATE_ID: torch.full(
                (count, *specs[ps.CONV_STATE_ID].shape), 4, dtype=torch.bfloat16
            ),
            ps.RECURRENT_STATE_ID: torch.full(
                (count, *specs[ps.RECURRENT_STATE_ID].shape), 5, dtype=torch.float32
            ),
        },
        {"canary": torch.tensor((11, 13), dtype=torch.int64)},
    )


def _body(state: ps.DecodeState, captured_m: int) -> Any:
    layers = state.schema.gdn_layer_ids
    buffers = SimpleNamespace(
        token_ids=torch.empty(captured_m, dtype=torch.int64),
        position_ids=torch.empty((3, captured_m), dtype=torch.int64),
        page_source=torch.empty(captured_m, dtype=torch.int32),
        page_destination=torch.empty(captured_m, dtype=torch.int32),
        token_offsets=torch.empty(captured_m, dtype=torch.int32),
        page_table=torch.empty((captured_m, state.schema.max_pages), dtype=torch.int32),
        sequence_lengths=torch.empty(captured_m, dtype=torch.int32),
        segment_lengths=torch.empty(captured_m, dtype=torch.int32),
        row_activity=torch.empty(captured_m, dtype=torch.int32),
        conv_slots=torch.empty((2, len(layers), captured_m), dtype=torch.int32),
        recurrent_slots=torch.empty((2, len(layers), captured_m), dtype=torch.int32),
        guard=torch.empty(2, dtype=torch.int64),
        logits=torch.empty(
            (captured_m, _config().graph.text.vocabulary_size), dtype=torch.bfloat16
        ),
    )
    return SimpleNamespace(arena=state.arena, buffers=buffers, gdn_layers=layers)


def _host(body: Any) -> pf._HostBuffers:
    return pf._HostBuffers(
        *(torch.empty_like(getattr(body.buffers, name)) for name in pf._HostBuffers.__annotations__)
    )


class QwenPerformanceGraphTests(unittest.TestCase):
    def test_packed_capture_uses_constant_padding_state_at_large_physical_m(self) -> None:
        from cubic_serve.runtime.quantized_graph import _bootstrap_cache_arenas

        _, state = _program()
        spec = replace(state.arena.spec, scratch_capacity=16,
                       scratch_arena_ids=("scratch:0", "scratch:1"))
        storage, (arena,) = _bootstrap_cache_arenas(((spec, state.arena.layout),), "cpu", 0)
        storage.backing.fill_(37)
        # Protect every byte except the exact named padding field payloads.
        expected = storage.backing.clone()
        dummy = arena.spec.state_slot_capacity + 1
        first_page = arena.spec.storage_page_capacity + 16
        for index in range(first_page, first_page + 16):
            for field in storage.page_layout.fields:
                start = index * storage.page_layout.entry_bytes + storage.page_layout.offsets[field.name]
                expected[start:start + field.nbytes].zero_()
        for field in storage.state_layout.fields:
            start = (storage.allocator.state_offset_bytes + dummy * storage.state_layout.entry_bytes
                     + storage.state_layout.offsets[field.name])
            expected[start:start + field.nbytes].zero_()
        fixture = SimpleNamespace(arena=arena, schema=state.schema)
        body = cast(Any, object.__new__(pf._DecodeBody))
        body.arena, body.gdn_layers = arena, state.schema.gdn_layer_ids
        body.buffers = _body(fixture, 16).buffers
        resource = ec.RankReplayResources(0, "g", "s", "b", "w", "scratch:1")
        body.prepare_capture_scratch(resource)
        self.assertTrue(torch.equal(storage.backing, expected))
        wanted = torch.arange(len(body.gdn_layers)) + dummy * len(body.gdn_layers)
        self.assertTrue(torch.equal(body.buffers.conv_slots[0], wanted[:, None].expand(-1, 16)))
        self.assertTrue(torch.equal(body.buffers.conv_slots, body.buffers.recurrent_slots))
        self.assertFalse(bool(body.buffers.row_activity.any()))
        self.assertTrue(bool((body.buffers.segment_lengths == 1).all()))
        storage.backing.fill_(51)
        body.bind_capture_scratch(resource)
        self.assertTrue(bool((storage.backing == 51).all()))

    def test_capture_warmup_uses_only_named_scratch(self) -> None:
        _, state = _program()
        arena, captured_m = state.arena, 2
        arena.key[: arena.spec.page_capacity].fill_(9)
        for tensor in arena.states.values():
            tensor[: arena.spec.state_slot_capacity].fill_(9)
        owner_key = arena.key[: arena.spec.page_capacity].clone()
        owner_states = {
            name: tensor[: arena.spec.state_slot_capacity].clone()
            for name, tensor in arena.states.items()
        }
        body = cast(Any, object.__new__(pf._DecodeBody))
        body.arena, body.gdn_layers, body.buffers = arena, (0, 2), _body(state, captured_m).buffers
        resource = ec.RankReplayResources(0, "g", "s", "b", "w", "scratch:0")
        body.prepare_capture_scratch(resource)
        self.assertTrue(torch.equal(owner_key, arena.key[: arena.spec.page_capacity]))
        self.assertTrue(
            all(
                torch.equal(value, arena.states[name][: arena.spec.state_slot_capacity])
                for name, value in owner_states.items()
            )
        )
        self.assertEqual(
            tuple(body.buffers.page_source.tolist()),
            (arena.spec.page_capacity, arena.spec.page_capacity + 1),
        )
        self.assertTrue(torch.equal(body.buffers.conv_slots, body.buffers.recurrent_slots))
        with self.assertRaises(pf.Qwen3_5PerformanceRejected):
            body.prepare_capture_scratch(replace(resource, scratch_arena_id="foreign"))

    def test_ordinary_typed_seal_cannot_admit_composite_graph(self) -> None:
        program, state = _program()
        topology = ec.ParallelTopology.single_node(
            topology_id="topology:0", device_indices=(0,), tp_size=1, pp_size=1
        )
        architecture = program.model.config.architecture_plan()
        nodes = (
            architecture.token_embedding_node_id,
            *(layer.node_id for layer in architecture.decoder_layers),
            architecture.final_norm_node_id,
            architecture.lm_head_node_id,
        )
        partition = ec.PartitionPlan(
            "partition:0",
            topology.topology_id,
            tuple(ec.GraphAssignment(node, 0, (0,)) for node in nodes),
        )
        bucket = replace(
            _launch(_batch(1)).bucket,
            replay_abi_id=pf.REPLAY_ABI_ID,
            capacity_items=pf._capacity_items(state, program.model.config.graph.text, 4),
            row_map_schema_id=pf.ROW_MAP_SCHEMA_ID,
            scratch_schema_id=pf.SCRATCH_SCHEMA_ID,
            graph_safety_contract_ids=(pf.GRAPH_SAFETY_CONTRACT_ID,),
        )
        safety = ec.GraphSafetyContract(
            pf.GRAPH_SAFETY_CONTRACT_ID,
            pf.ROW_MAP_SCHEMA_ID,
            ec.PaddingIsolation.PRIVATE_SCRATCH,
            "graph-safety",
        )
        candidate = ec.CandidateBinding(
            pf.CANDIDATE_ID,
            pf.PROGRAM_ID,
            ec.CandidateBackend.CUDA,
            pf.REPLAY_ABI_ID,
            0,
            "evidence:ordinary-leaf",
            safety,
        )
        signature = pf._decode_signature(program.model.config.graph.text, topology.topology_id)
        bindings = (ec.OpBinding(signature, candidate=candidate),)
        closure = program.child_operator_closure
        self.assertEqual(len(closure), len(set(closure)))
        self.assertGreater(len(closure), 10)
        execution = sealed_test_performance_plan("execution:qwen", bindings, (bucket,))
        sealed_bucket = execution.graph_buckets[0]
        binding = GraphRuntimeBinding(execution, topology, partition, sealed_bucket)
        with self.assertRaisesRegex(CudaGraphRejected, "ordinary leaf.*composite"):
            program.validate_binding(binding)
        children = tuple(
            ec.OpBinding(
                replace(signature, semantic_op=semantic),
                candidate=replace(
                    candidate,
                    candidate_id=f"candidate:{semantic}",
                    implementation_id=f"implementation:{semantic}",
                    evidence_id="tests-only-unsealed",
                    graph_safety=None,
                ),
            )
            for semantic in ("attention", "linear")
        )
        composite_execution = sealed_test_composite_performance_plan(
            "execution:qwen:composite", bindings[0], children, (bucket,)
        )
        composite_bucket = composite_execution.graph_buckets[0]
        composite_binding = GraphRuntimeBinding(
            composite_execution, topology, partition, composite_bucket
        )
        with self.assertRaisesRegex(CudaGraphRejected, "occurrence manifest"):
            program.validate_binding(composite_binding)
        self.assertIsInstance(program, GraphProgram)

    def test_fill_replay_consume_and_commit_follow_physical_row_map(self) -> None:
        program, state = _program()
        batch, launch = _one_axis_batch(2), None
        launch = _launch(batch)
        intent = ReplayIntent(batch, launch.rows, AcquireDeadline.after(1.0))
        body, stream = _body(state, launch.captured_m), _Stream()

        def emulate_graph() -> None:
            b, arena = body.buffers, body.arena
            for physical in range(launch.captured_m):
                source, destination = (
                    int(b.page_source[physical]),
                    int(b.page_destination[physical]),
                )
                arena.key[destination].copy_(arena.key[source])
                arena.value[destination].copy_(arena.value[source])
                arena.written[destination].copy_(arena.written[source])
                arena.written[destination, :, int(b.token_offsets[physical])] = True
                b.logits[physical].fill_(physical + 1)
            for slots_name, state_id in (
                ("conv_slots", ps.CONV_STATE_ID),
                ("recurrent_slots", ps.RECURRENT_STATE_ID),
            ):
                slots = getattr(b, slots_name)
                source = slots[0, 0] // len(body.gdn_layers)
                destination = slots[1, 0] // len(body.gdn_layers)
                for src, dst in zip(source.tolist(), destination.tolist(), strict=True):
                    arena.states[state_id][dst].copy_(arena.states[state_id][src])

        graph = _Graph(emulate_graph)
        slot = pf.Qwen3_5DecodeGraphSlot(
            launch.replay_lease.rank_resources,
            body,
            _host(body),
            cast(Any, stream),
            cast(Any, graph),
        )
        program._binding = cast(GraphRuntimeBinding, SimpleNamespace(bucket=launch.bucket))
        program._slots.add(id(slot))
        run = ps.QualificationFactory(state, batch).create(_seed(state, batch))
        positions = torch.tensor([segment.positions[0].axes[0] for segment in batch.segments])
        padding = _padding(state, launch.captured_m - launch.logical_m)
        with (
            patch("torch.cuda.current_stream", return_value=object()),
            patch("torch.cuda.device", return_value=nullcontext()),
            patch("torch.cuda.stream", return_value=nullcontext()),
        ):
            session = program.fill(
                slot,
                launch,
                intent,
                {"token_ids": torch.tensor((1, 2)), "position_ids": positions},
                run,
                padding,
            )
            session.replay()
            output = session.consume()["logits"]
            self.assertTrue(torch.all(output[0] == 3))
            self.assertTrue(torch.all(output[1] == 1))
            body.buffers.logits.zero_()
            self.assertTrue(torch.all(output[0] == 3), "logical output must be durable")
            session.commit()
        self.assertEqual(tuple(state.store.current_revision(owner) for owner in run.owners), (1, 1))
        run.close()
        self.assertEqual((stream.waits, stream.syncs, graph.replays), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
