from __future__ import annotations

import gc
import unittest
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from cubic_serve.contracts.execution import RankReplayResources
from cubic_serve.executor import model_runner as parallel
from cubic_serve.executor.model_runner import _Buffers, _DecodeBody
from cubic_serve.executor.mtp_runner import _mtp_graph_capacity_supported
from cubic_serve.models.parallel_weights import AxisSpan
from cubic_serve.ops.cuda.attention import (
    BF16PagedGQADecodePrimitive,
    PageCOWCopyPrimitive,
    PageTokenWritePrimitive,
)
from cubic_serve.ops.cuda.linear_attention import (
    CausalConv1DDecodePrimitive,
    GatedDeltaDecodePrimitive,
)
from cubic_serve.ops.cuda.position import MRoPETableEvidence
from cubic_serve.runtime.cache import CacheArena, CacheArenaSpec, KVLayout


class _Primitive:
    def __init__(self, *operands: torch.Tensor, **_options: object) -> None:
        self.operands = operands


class MTPGraphCapacityPolicyTests(unittest.TestCase):
    def test_low_row_whitelist_does_not_admit_arbitrary_exact_shapes(self) -> None:
        self.assertFalse(_mtp_graph_capacity_supported(3, 3))
        self.assertTrue(_mtp_graph_capacity_supported(4, 3))
        self.assertFalse(_mtp_graph_capacity_supported(5, 3))
        self.assertTrue(_mtp_graph_capacity_supported(8, 7))
        self.assertFalse(_mtp_graph_capacity_supported(9, 7))
        self.assertFalse(_mtp_graph_capacity_supported(9, 8))


class _FakeArena:
    def __init__(
        self,
        spec: CacheArenaSpec,
        layout: KVLayout,
        device: torch.device | str,
        *,
        rank_id: int,
    ) -> None:
        self.spec, self.layout = spec, layout
        self.device, self.rank_id = torch.device(device), rank_id
        self.key = torch.empty(2)
        self.value = torch.empty(2)
        self.written = torch.empty(2, dtype=torch.bool)
        self.states: dict[str, torch.Tensor] = {}


class _FakeDecodeBody:
    def __init__(
        self,
        model: object,
        arena: object,
        *,
        captured_m: int,
        table_pages: int,
        mrope_tables: object,
    ) -> None:
        self.model = model
        self.arena = arena
        self.mrope_tables = mrope_tables
        self.buffers = SimpleNamespace(
            page_table=torch.zeros((captured_m, table_pages), dtype=torch.int32),
            hidden=torch.empty((captured_m, 4), dtype=torch.bfloat16),
            norm=torch.empty((captured_m, 4), dtype=torch.bfloat16),
            mlp_active=torch.empty((captured_m, 3), dtype=torch.bfloat16),
            gdn_a=torch.empty((captured_m, 2), dtype=torch.bfloat16),
            gdn_b=torch.empty((captured_m, 3), dtype=torch.bfloat16),
            gdn_gate=torch.empty((captured_m, 4), dtype=torch.bfloat16),
        )
        cache = arena if isinstance(arena, parallel._ArenaView) else None
        if cache is not None:
            self.conv_state = cache.key
            self.recurrent_state = cache.value
            self.page_copy = _Primitive(cache.key, cache.value, cache.written)
            self.conv = _Primitive(cache.key)
            self.gdn = _Primitive(cache.value)
            self.page_writes = (_Primitive(cache.key),)
            self.attentions = (_Primitive(cache.value),)


def _cache_binding(
    _body: _DecodeBody,
    arena: parallel._ArenaView,
) -> parallel._CacheRuntimeBinding:
    return parallel._CacheRuntimeBinding(
        arena=arena,
        conv_state=arena.key,
        recurrent_state=arena.value,
        page_copy=cast(PageCOWCopyPrimitive, _Primitive(arena.key)),
        conv=cast(CausalConv1DDecodePrimitive, _Primitive(arena.key)),
        gdn=cast(GatedDeltaDecodePrimitive, _Primitive(arena.value)),
        page_writes=(cast(PageTokenWritePrimitive, _Primitive(arena.key)),),
        attentions=(cast(BF16PagedGQADecodePrimitive, _Primitive(arena.value)),),
    )


def _primary(*, scratch_capacity: int = 128) -> parallel.Qwen3_5TP2PP2DecodeBody:
    captured_m, table_pages = 4, 8
    body = object.__new__(parallel.Qwen3_5TP2PP2DecodeBody)
    body.rank, body.stage, body.tensor_rank = 0, 0, 0
    body.layer_ids, body.full_layer_ids, body.gdn_layer_ids = (0,), (0,), (1,)
    body.captured_m = captured_m
    body.lane_id = 0
    body.bucket_logical_ms = tuple(range(1, captured_m + 1))
    body.embedding_span, body.vocab_span = cast(AxisSpan, object()), None
    body._vocab_size = 32
    body._global_weight_shapes = {"weight": (4, 4)}
    body._mrope_tables = (
        torch.empty(1),
        torch.empty(1),
        MRoPETableEvidence(1, 2, (1, 1, 1), "0x1.0p+0", "b2s256:test"),
    )
    body._packed_gate_up_weights = torch.empty((2, 6, 4), dtype=torch.bfloat16)
    body._packed_gdn_abz_weights = torch.empty((2, 9, 4), dtype=torch.bfloat16)
    body._local_intermediate = 3
    body._gdn_abz_widths = (2, 3, 4)
    body._tp_group = cast(ProcessGroup, object())
    body._pp_group = cast(ProcessGroup, object())
    body._pp_peer = 2
    body._is_arena_primary = True
    body._bootstrap_siblings = []
    body._graph_capture_started = False
    body._arena_rebind_failed = False
    body._arena_replacement_allowed = True
    spec = CacheArenaSpec(
        "rank:0",
        page_capacity=128 * table_pages,
        state_slot_capacity=128,
        state_specs=(),
        scratch_capacity=scratch_capacity,
        scratch_arena_ids=("graph",),
    )
    layout = KVLayout((0,), 16, 1, 2)
    arena = _FakeArena(spec, layout, torch.device("cpu"), rank_id=0)
    model = SimpleNamespace(weights=object())
    local_arena = parallel._ArenaView(
        spec,
        layout,
        arena.device,
        arena.rank_id,
        arena.key,
        arena.value,
        arena.written,
        arena.states,
    )
    inner = _FakeDecodeBody(
        model,
        local_arena,
        captured_m=captured_m,
        table_pages=table_pages,
        mrope_tables=body._mrope_tables,
    )
    body.arena = cast(CacheArena, arena)
    body._body = cast(_DecodeBody, inner)
    body._packed_mlp = torch.empty((captured_m, 6), dtype=torch.bfloat16)
    body._packed_gdn_abz = torch.empty((captured_m, 9), dtype=torch.bfloat16)
    body.embedding_mask = torch.empty((captured_m, 1), dtype=torch.bfloat16)
    return body


def _resource(rank: int = 0) -> RankReplayResources:
    return RankReplayResources(rank, "graph", "stream", "buffers", "workspace", "graph")


def _wave_bodies(
    *, stage: int = 1, micro_m: int = 4
) -> tuple[
    parallel.Qwen3_5TP2PP2DecodeBody,
    parallel.Qwen3_5TP2PP2DecodeBody,
]:
    table_pages = 2
    shared_arena = SimpleNamespace(
        rank_id=0,
        device=torch.device("cpu"),
        spec=SimpleNamespace(
            page_capacity=parallel.minimum_kv_page_capacity(
                table_pages, 2 * micro_m
            ),
            state_slot_capacity=2 * micro_m,
            scratch_capacity=micro_m,
            scratch_arena_ids=("graph",),
        ),
        key=torch.empty(1),
        value=torch.empty(1),
        written=torch.empty(1, dtype=torch.bool),
        states={},
    )
    shared_model = SimpleNamespace(weights=object())
    shared_gate_up = torch.empty(1)
    shared_gdn_abz = torch.empty(1)
    shared_tables = (
        torch.empty(1),
        torch.empty(1),
        MRoPETableEvidence(1, 2, (1, 1, 1), "0x1.0p+0", "b2s256:wave"),
    )
    shared_tp, shared_pp = object(), object()
    embedding_span = AxisSpan(0, 8) if stage == 0 else None
    vocab_span = AxisSpan(0, 5) if stage == 1 else None

    def make(lane_id: int) -> parallel.Qwen3_5TP2PP2DecodeBody:
        body = object.__new__(parallel.Qwen3_5TP2PP2DecodeBody)
        body.rank, body.stage, body.tensor_rank = 0, stage, 0
        body.layer_ids, body.full_layer_ids, body.gdn_layer_ids = (0,), (0,), (1,)
        body.captured_m, body.lane_id = micro_m, lane_id
        body.bucket_logical_ms = tuple(range(1, micro_m + 1))
        body.embedding_span, body.vocab_span = embedding_span, vocab_span
        body._vocab_size = 8
        body._mrope_tables = shared_tables
        body._packed_gate_up_weights = shared_gate_up
        body._packed_gdn_abz_weights = shared_gdn_abz
        body._tp_group = cast(ProcessGroup, shared_tp)
        body._pp_group = cast(ProcessGroup, shared_pp)
        body._pp_peer = 2
        body._graph_capture_started = False
        body._arena_rebind_failed = False
        body.arena = cast(CacheArena, shared_arena)
        values = {
            name: torch.empty(1, dtype=torch.bfloat16)
            for name in _Buffers.__annotations__
        }
        values["hidden"] = torch.empty((micro_m, 3), dtype=torch.bfloat16)
        values["logits"] = torch.empty((micro_m, 5), dtype=torch.bfloat16)
        values["page_table"] = torch.empty(
            (micro_m, table_pages), dtype=torch.int32
        )
        buffers = SimpleNamespace(**values)
        body._body = cast(
            _DecodeBody,
            SimpleNamespace(model=shared_model, arena=shared_arena, buffers=buffers),
        )
        body._packed_mlp = torch.empty(1)
        body._packed_gdn_abz = torch.empty(1)
        body.embedding_mask = (
            torch.empty((micro_m, 1), dtype=torch.bfloat16) if stage == 0 else None
        )
        return body

    return make(0), make(1)


class Qwen3_5ParallelBucketTests(unittest.TestCase):
    def test_logical_m_keeps_dense_math_on_the_full_physical_bucket(self) -> None:
        body = object.__new__(parallel.Qwen3_5TP2PP2DecodeBody)
        body.captured_m = 2
        body._active_rows = torch.ones((1,), dtype=torch.int32)

        body.set_active_rows(1)

        self.assertEqual(body._active_rows.item(), 2)
        with self.assertRaises(parallel.Qwen3_5ParallelPerformanceError):
            body.set_active_rows(0)

    def test_page_capacity_covers_one_full_context_plus_active_cow_owners(self) -> None:
        self.assertEqual(parallel.minimum_kv_page_capacity(16_384, 256), 16_896)
        self.assertLess(
            parallel.minimum_kv_page_capacity(16_384, 256),
            256 * 16_384,
        )
        for table_pages, active_owners in ((0, 1), (1, 0), (True, 1)):
            with self.subTest(
                table_pages=table_pages, active_owners=active_owners
            ), self.assertRaises(ValueError):
                parallel.minimum_kv_page_capacity(table_pages, active_owners)

    def test_siblings_share_weights_tables_and_arena_but_not_bucket_runtime(self) -> None:
        primary = _primary()
        with (
            patch.object(parallel, "_DecodeBody", _FakeDecodeBody),
            patch.object(parallel, "bind_bf16_linear_kernel", _Primitive),
            patch.object(parallel, "BF16PackedSwiGLUPrimitive", _Primitive),
            patch.object(parallel, "BF16PackedGDNProjectionSplitPrimitive", _Primitive),
            patch.object(
                parallel,
                "_pack_gate_up_weights",
                side_effect=AssertionError("sibling repacked checkpoint weights"),
            ),
            patch.object(
                parallel,
                "_pack_gdn_abz_weights",
                side_effect=AssertionError("sibling repacked checkpoint weights"),
            ),
        ):
            siblings = tuple(
                primary.build_sibling(captured_m=captured_m)
                for captured_m in (16, 64, 128)
            )

        self.assertTrue(primary._arena_replacement_allowed)
        for sibling, captured_m in zip(siblings, (16, 64, 128), strict=True):
            self.assertEqual(sibling.captured_m, captured_m)
            self.assertIs(sibling.arena, primary.arena)
            self.assertIs(sibling._body.model, primary._body.model)
            self.assertIs(sibling._body.arena.key, primary.arena.key)
            self.assertIs(sibling._body.arena.value, primary.arena.value)
            self.assertIs(sibling._body.arena.written, primary.arena.written)
            self.assertIs(sibling._body.arena.states, primary.arena.states)
            self.assertIs(sibling._packed_gate_up_weights, primary._packed_gate_up_weights)
            self.assertIs(sibling._packed_gdn_abz_weights, primary._packed_gdn_abz_weights)
            self.assertIs(sibling._mrope_tables, primary._mrope_tables)
            self.assertIsNot(sibling._packed_mlp, primary._packed_mlp)
            self.assertIsNot(sibling._packed_gdn_abz, primary._packed_gdn_abz)
            self.assertIsNot(sibling.buffers.hidden, primary.buffers.hidden)
            self.assertIsNot(sibling.embedding_mask, primary.embedding_mask)
            self.assertEqual(sibling._packed_mlp.shape, (captured_m, 6))
            self.assertEqual(sibling._packed_gdn_abz.shape, (captured_m, 9))
        self.assertEqual(
            len({item._packed_mlp.untyped_storage().data_ptr() for item in siblings}),
            len(siblings),
        )
        self.assertEqual(len({id(item._body) for item in siblings}), len(siblings))
        self.assertEqual(
            len({id(item._linear_mlp_gate_up) for item in siblings}),
            len(siblings),
        )
        self.assertEqual(
            len({id(item._packed_gdn_abz_split) for item in siblings}),
            len(siblings),
        )

    def test_external_final_spec_must_cover_the_largest_sibling(self) -> None:
        primary = _primary(scratch_capacity=64)
        with (
            patch.object(parallel, "_DecodeBody", _FakeDecodeBody),
            patch.object(parallel, "bind_bf16_linear_kernel", _Primitive),
            patch.object(parallel, "BF16PackedSwiGLUPrimitive", _Primitive),
            patch.object(parallel, "BF16PackedGDNProjectionSplitPrimitive", _Primitive),
        ):
            sibling = primary.build_sibling(captured_m=128)
        with self.assertRaisesRegex(
            parallel.Qwen3_5ParallelPerformanceError,
            "final shared arena",
        ):
            primary.replace_shared_cache_arena(primary.arena.spec, (sibling,))

    def test_low_m_portfolio_accepts_a_static_non_power_of_two_shape(self) -> None:
        primary = _primary()
        with (
            patch.object(parallel, "_DecodeBody", _FakeDecodeBody),
            patch.object(parallel, "bind_bf16_linear_kernel", _Primitive),
            patch.object(parallel, "BF16PackedSwiGLUPrimitive", _Primitive),
            patch.object(parallel, "BF16PackedGDNProjectionSplitPrimitive", _Primitive),
        ):
            sibling = primary.build_sibling(captured_m=3)

        self.assertEqual(sibling.captured_m, 3)
        self.assertEqual(sibling.bucket_logical_ms, (1, 2, 3))

    def test_same_m_second_activation_lane_has_a_unique_family_key(self) -> None:
        primary = _primary()
        with (
            patch.object(parallel, "_DecodeBody", _FakeDecodeBody),
            patch.object(parallel, "bind_bf16_linear_kernel", _Primitive),
            patch.object(parallel, "BF16PackedSwiGLUPrimitive", _Primitive),
            patch.object(parallel, "BF16PackedGDNProjectionSplitPrimitive", _Primitive),
        ):
            lane_b = primary.build_sibling(captured_m=primary.captured_m, lane_id=1)

        self.assertEqual((primary.captured_m, primary.lane_id), (4, 0))
        self.assertEqual((lane_b.captured_m, lane_b.lane_id), (4, 1))
        self.assertIs(lane_b.arena, primary.arena)
        self.assertIs(lane_b._body.model, primary._body.model)
        self.assertIsNot(lane_b.buffers.hidden, primary.buffers.hidden)
        with self.assertRaisesRegex(
            parallel.Qwen3_5ParallelPerformanceError,
            "bootstrap arena",
        ):
            primary.build_sibling(captured_m=primary.captured_m, lane_id=1)

    def test_shared_replacement_requires_both_same_m_lanes(self) -> None:
        primary = _primary()
        with (
            patch.object(parallel, "_DecodeBody", _FakeDecodeBody),
            patch.object(parallel, "bind_bf16_linear_kernel", _Primitive),
            patch.object(parallel, "BF16PackedSwiGLUPrimitive", _Primitive),
            patch.object(parallel, "BF16PackedGDNProjectionSplitPrimitive", _Primitive),
        ):
            lane_b = primary.build_sibling(captured_m=primary.captured_m, lane_id=1)

        with self.assertRaisesRegex(
            parallel.Qwen3_5ParallelPerformanceError,
            "complete uncaptured bucket family",
        ):
            primary.replace_shared_cache_arena(primary.arena.spec, ())
        with (
            patch.object(parallel, "CacheArena", _FakeArena),
            patch.object(parallel, "_build_cache_runtime", _cache_binding),
        ):
            primary.replace_shared_cache_arena(primary.arena.spec, (lane_b,))
        self.assertIs(primary.arena, lane_b.arena)

    def test_coordinated_replace_preserves_private_runtime_and_drops_old_arena(self) -> None:
        primary = _primary()
        with (
            patch.object(parallel, "_DecodeBody", _FakeDecodeBody),
            patch.object(parallel, "bind_bf16_linear_kernel", _Primitive),
            patch.object(parallel, "BF16PackedSwiGLUPrimitive", _Primitive),
            patch.object(parallel, "BF16PackedGDNProjectionSplitPrimitive", _Primitive),
        ):
            sibling = primary.build_sibling(captured_m=16)
        old_ref = weakref.ref(primary.arena)
        primary_workspace = primary._packed_mlp
        sibling_workspace = sibling._packed_mlp
        primary_hidden = primary.buffers.hidden
        sibling_hidden = sibling.buffers.hidden
        spec = primary.arena.spec
        with (
            patch.object(parallel, "CacheArena", _FakeArena),
            patch.object(parallel, "_build_cache_runtime", _cache_binding),
        ):
            primary.replace_shared_cache_arena(spec, (sibling,))

        self.assertIs(primary.arena, sibling.arena)
        self.assertIs(primary._body.arena.key, primary.arena.key)
        self.assertIs(sibling._body.arena.key, primary.arena.key)
        self.assertIs(primary._packed_mlp, primary_workspace)
        self.assertIs(sibling._packed_mlp, sibling_workspace)
        self.assertIs(primary.buffers.hidden, primary_hidden)
        self.assertIs(sibling.buffers.hidden, sibling_hidden)
        self.assertFalse(primary._arena_replacement_allowed)
        self.assertFalse(sibling._arena_replacement_allowed)
        gc.collect()
        self.assertIsNone(old_ref())

        with self.assertRaisesRegex(
            parallel.Qwen3_5ParallelPerformanceError,
            "complete uncaptured bucket family",
        ):
            primary.replace_shared_cache_arena(spec, (sibling,))

    def test_rebind_failure_poison_detaches_the_complete_family(self) -> None:
        primary = _primary()
        with (
            patch.object(parallel, "_DecodeBody", _FakeDecodeBody),
            patch.object(parallel, "bind_bf16_linear_kernel", _Primitive),
            patch.object(parallel, "BF16PackedSwiGLUPrimitive", _Primitive),
            patch.object(parallel, "BF16PackedGDNProjectionSplitPrimitive", _Primitive),
        ):
            siblings = tuple(
                primary.build_sibling(captured_m=captured_m)
                for captured_m in (16, 64)
            )
        family = (primary, *siblings)
        old_ref = weakref.ref(primary.arena)
        calls = 0

        def fail_second(
            body: _DecodeBody,
            arena: parallel._ArenaView,
        ) -> parallel._CacheRuntimeBinding:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic rebind failure")
            return _cache_binding(body, arena)

        with (
            patch.object(parallel, "CacheArena", _FakeArena),
            patch.object(parallel, "_build_cache_runtime", fail_second),
            self.assertRaisesRegex(
                parallel.Qwen3_5ParallelPerformanceError,
                "synthetic rebind failure",
            ),
        ):
            primary.replace_shared_cache_arena(primary.arena.spec, siblings)

        for item in family:
            self.assertTrue(item._arena_rebind_failed)
            self.assertFalse(item._arena_replacement_allowed)
            self.assertFalse(hasattr(item, "arena"))
            for name in parallel._CACHE_RUNTIME_ATTRIBUTES:
                self.assertFalse(hasattr(item._body, name))
        gc.collect()
        self.assertIsNone(old_ref())

    def test_capture_start_permanently_rejects_shared_replacement(self) -> None:
        primary = _primary()
        primary._mark_graph_capture_started()
        with self.assertRaisesRegex(
            parallel.Qwen3_5ParallelPerformanceError,
            "complete uncaptured bucket family",
        ):
            primary.replace_shared_cache_arena(primary.arena.spec, ())

    def test_wavefront_requires_independent_matching_lanes(self) -> None:
        lane_a, lane_b = _wave_bodies()
        parallel.Qwen3_5TP2PP2WavefrontGraphSlot._validate_lanes(
            lane_a, lane_b, _resource()
        )

        lane_b.buffers.hidden = lane_a.buffers.hidden
        with self.assertRaisesRegex(
            parallel.Qwen3_5ParallelPerformanceError,
            "alias private activation",
        ):
            parallel.Qwen3_5TP2PP2WavefrontGraphSlot._validate_lanes(
                lane_a, lane_b, _resource()
            )

        lane_a, lane_b = _wave_bodies()
        lane_b.lane_id = 0
        with self.assertRaisesRegex(
            parallel.Qwen3_5ParallelPerformanceError,
            "lane 0/lane 1",
        ):
            parallel.Qwen3_5TP2PP2WavefrontGraphSlot._validate_lanes(
                lane_a, lane_b, _resource()
            )

    def test_wavefront_capture_launches_a_then_b_and_concatenates_logits(self) -> None:
        lane_a, lane_b = _wave_bodies(stage=1, micro_m=4)
        launches: list[str] = []

        def launch_a() -> None:
            launches.append("a")
            assert lane_a.logits is not None
            lane_a.logits.fill_(1)

        def launch_b() -> None:
            launches.append("b")
            assert lane_b.logits is not None
            lane_b.logits.fill_(2)

        lane_a.launch = launch_a  # type: ignore[method-assign]
        lane_b.launch = launch_b  # type: ignore[method-assign]
        scratch: list[int] = []
        lane_a.prepare_scratch = (  # type: ignore[method-assign]
            lambda resource: scratch.append(resource.rank)
        )
        lane_b.prepare_scratch = (  # type: ignore[method-assign]
            lambda resource: scratch.append(resource.rank)
        )

        class FakeStream:
            def __init__(self, *, device: torch.device) -> None:
                self.device = device

            def wait_stream(self, _stream: object) -> None:
                pass

            def synchronize(self) -> None:
                pass

        class FakeGraph:
            def replay(self) -> None:
                pass

        with (
            patch.object(parallel, "Stream", FakeStream),
            patch.object(parallel, "CUDAGraph", FakeGraph),
            patch.object(parallel, "capture_graph", lambda *_args, **_kwargs: nullcontext()),
            patch.object(dist, "barrier"),
            patch.object(torch.cuda, "current_stream", return_value=object()),
            patch.object(torch.cuda, "stream", lambda _stream: nullcontext()),
        ):
            slot = parallel.Qwen3_5TP2PP2WavefrontGraphSlot.capture(
                lane_a, lane_b, _resource()
            )

        self.assertEqual(scratch, [0, 0])
        self.assertEqual(launches, ["a", "b", "a", "b", "a", "b"])
        self.assertEqual(slot.micro_m, 4)
        self.assertEqual(slot.captured_m, 8)
        self.assertIsNotNone(slot.logits)
        assert slot.logits is not None
        self.assertTrue(slot.logits.is_contiguous())
        torch.testing.assert_close(slot.logits[:4], torch.ones((4, 5), dtype=torch.bfloat16))
        torch.testing.assert_close(
            slot.logits[4:], torch.full((4, 5), 2, dtype=torch.bfloat16)
        )

    def test_first_stage_wavefront_exposes_no_logits(self) -> None:
        lane_a, lane_b = _wave_bodies(stage=0)
        parallel.Qwen3_5TP2PP2WavefrontGraphSlot._validate_lanes(
            lane_a, lane_b, _resource()
        )
        launches: list[int] = []
        lane_a.launch = lambda: launches.append(0)  # type: ignore[method-assign]
        lane_b.launch = lambda: launches.append(1)  # type: ignore[method-assign]
        parallel.Qwen3_5TP2PP2WavefrontGraphSlot._launch_waves(
            lane_a, lane_b, None
        )
        self.assertEqual(launches, [0, 1])


if __name__ == "__main__":
    unittest.main()
