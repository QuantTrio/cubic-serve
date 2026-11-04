from __future__ import annotations

import asyncio
import heapq
import io
import multiprocessing
import pickle
import struct
import threading
import time
import unittest
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from itertools import pairwise
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import torch

from cubic_serve.executor.replay_lease import ReplayLeasePool
from cubic_serve.runtime import performance as serving
from cubic_serve.runtime.model_state import (
    CONV_STATE_ID,
    RECURRENT_STATE_ID,
)


def test_budget_only_pool_does_not_load_model_frontend():
    from cubic_serve.contracts.request import ReasoningControl, StructuredOutputMeta
    from cubic_serve.generation.structured_sampling import ConstraintBinding

    binding = ConstraintBinding("budget", StructuredOutputMeta.identity(), 0, 8,
                                reasoning_control=ReasoningControl((2,), (3,), 0, True))
    command = SimpleNamespace(constraints=(binding,), released_constraints=())
    admission = SimpleNamespace(constraint_stop_token_ids=(7,))
    with patch("cubic_serve.models.builtin.resolve_model_definition",
               side_effect=AssertionError("budget loaded the grammar frontend")):
        pool = serving._request_grammar_pool(None, command, admission, 8)
        round_ = pool.begin((binding,))
        assert round_.mask("budget", 0)[0] == 1 << 3
        round_.commit({"budget": (3,)})
        assert serving._request_grammar_pool(pool, command, admission, 8) is pool
        assert pool.factory is None


def test_steady_audit_uses_declared_physical_page_geometry():
    for page_size in (16, 32, 64):
        body = SimpleNamespace(captured_m=8, arena=SimpleNamespace(
            layout=SimpleNamespace(block_size_tokens=page_size)))
        for position in (0, 1, page_size - 1, page_size):
            rows = serving._steady_audit_rows(body, 3, position=position)
            for row_index, row in enumerate(rows):
                assert row.position == position
                assert row.page_table[position // page_size] == row.page_destination
                assert row.page_destination == 2 * row_index + position // page_size
                assert row.reset_state == (position in (0, page_size - 1))
                assert bool(row.new_pages) == (position in (0, page_size - 1, page_size))


def test_padding_poison_uses_byte_storage_and_includes_format_fields():
    for poison in (0, 1):
        key = torch.zeros((4, 2, 8), dtype=torch.uint8)
        value = torch.zeros_like(key)
        scales = torch.zeros((4, 2))
        states = {name: torch.zeros((4, 2, 8))
                  for name in (CONV_STATE_ID, RECURRENT_STATE_ID, "kv_key_tail")}
        arena = SimpleNamespace(key=key, value=value, states=states,
                                written=torch.zeros((4, 2), dtype=torch.bool),
                                page_fields={"kv_key_scales": scales})
        executor = SimpleNamespace(_scratch_pages=(1,), _scratch_states=(1,),
                                   body=SimpleNamespace(arena=arena, hidden=torch.zeros(1)))
        serving._poison_production_audit_scratch(executor, poison)
        for tensor in (key, value, scales, *states.values()):
            assert bool(tensor[1].any())
            assert not bool(tensor[[0, 2, 3]].any())


class _FakeController:
    vocab_size = 128
    max_sequence_tokens = 8
    rank_cache_capacities: tuple[serving.Qwen3_5TP2PP2RankCacheCapacity, ...] = ()
    rank_graph_evidence_ids = (
        "b2s256:" + "5" * 64,
        "b2s256:" + "6" * 64,
        "b2s256:" + "7" * 64,
        "b2s256:" + "8" * 64,
    )

    def __init__(self, output: tuple[int, ...] = (20, 21)) -> None:
        self.output = output
        self.commands: list[serving._GenerateCommand] = []
        self.close_count = 0

    @property
    def ready(self) -> bool:
        return self.close_count == 0

    def submit(
        self,
        command: serving._GenerateCommand,
        output: serving._TokenSink,
    ) -> None:
        self.commands.append(command)
        for token_id in self.output:
            output.publish(token_id)
        output.finish()

    def cancel(self, request_id: str) -> None:
        del request_id

    def close(self) -> None:
        self.close_count += 1


@dataclass(frozen=True, slots=True)
class _Metadata:
    request_id: str


class StartupProgressTests(unittest.TestCase):
    def test_numerical_progress_requires_new_completed_witnesses(self) -> None:
        progress = serving._StartupProgress(2)
        self.assertTrue(progress.update(serving._LoadProgress(
            0, "numerical_audit", 1, 33, "witnesses",
        )))
        self.assertFalse(progress.update(serving._LoadProgress(
            0, "numerical_audit", 1, 33, "witnesses",
        )))
        self.assertTrue(progress.update(serving._LoadProgress(
            1, "numerical_audit", 1, 33, "witnesses",
        )))
        for completed, total, unit in ((0, 33, "witnesses"), (2, 34, "witnesses"),
                                        (2, 33, "buckets"), (34, 33, "witnesses")):
            with self.subTest(completed=completed, total=total, unit=unit), self.assertRaises(
                serving.Qwen3_5TP2PP2ServingError,
            ):
                progress.update(serving._LoadProgress(
                    0, "numerical_audit", completed, total, unit,
                ))

    def test_only_strictly_advancing_validated_startup_work_renews_progress(self) -> None:
        progress = serving._StartupProgress(2)
        self.assertFalse(progress.update(serving._LoadProgress(
            0, "graph_measurement", 0, 9, "buckets",
        )))
        self.assertTrue(progress.update(serving._LoadProgress(
            0, "graph_measurement", 1, 9, "buckets",
        )))
        self.assertFalse(progress.update(serving._LoadProgress(
            0, "graph_measurement", 1, 9, "buckets",
        )))
        for completed, total in ((0, 9), (2, 10), (True, 9)):
            with self.subTest(completed=completed, total=total), self.assertRaises(
                serving.Qwen3_5TP2PP2ServingError,
            ):
                progress.update(serving._LoadProgress(
                    0, "graph_measurement", completed, total, "buckets",
                ))

    def test_startup_deadline_tracks_each_rank_work_without_extending_stalled_peers(self) -> None:
        from collections import deque

        class Channel:
            def __init__(self) -> None:
                self.items: deque[Any] = deque()

            def recv(self) -> Any:
                return self.items.popleft()

        for stalled in (False, True):
            with self.subTest(stalled=stalled):
                controller = serving._ProcessController.__new__(serving._ProcessController)
                channels = (Channel(), Channel())
                controller._results = channels
                controller._processes = (SimpleNamespace(exitcode=None),) * 2
                controller._startup_progress = serving._StartupProgress(2)
                clock = [0.0]
                schedule = deque(
                    [(3, "base", 10, 10), (8, "graph_measurement", 1, 9),
                     (16, "graph_measurement", 2, 9), (24, "graph_capture", 1, 9),
                     (32, "ready", 0, 0)]
                )

                def wait(pending: tuple[Any, ...], timeout: float) -> list[Any]:
                    when, section, completed, total = schedule[0]
                    if when > clock[0] + timeout:
                        clock[0] += timeout
                        return []
                    schedule.popleft()
                    clock[0] = float(when)
                    for rank, channel in enumerate(channels):
                        if section == "ready":
                            channel.items.append(serving._Ready(
                                rank, 128, 2048, 256, "graphs", "states", cast(Any, None), (), (),
                            ))
                        elif stalled and rank == 1 and section != "base":
                            channel.items.append(serving._LoadProgress(rank, "base", 10, 10))
                        else:
                            unit = "bytes" if section == "base" else "buckets"
                            channel.items.append(serving._LoadProgress(
                                rank, section, completed, total, unit,
                            ))
                    return list(channels)

                with patch.object(serving, "wait", wait), patch.object(
                    serving.time, "monotonic", lambda: clock[0],
                ):
                    if stalled:
                        with self.assertRaisesRegex(
                            serving.Qwen3_5TP2PP2ServingError, "startup timed out",
                        ):
                            controller._receive_ready(10.0)
                        self.assertEqual(clock[0], 13.0)
                    else:
                        self.assertEqual(len(controller._receive_ready(10.0)), 2)
                        self.assertEqual(clock[0], 32.0)

    def test_quantized_tensor_progress_has_real_units_and_no_absent_mtp(self) -> None:
        output = io.StringIO()
        progress = serving._StartupProgress(1)
        with patch("cubic_serve.runtime.performance.sys.stderr", output):
            progress.update(serving._LoadProgress(0, "base", 0, 123, "tensors"))
            self.assertIn("0/123 tensors", output.getvalue())
            progress.update(serving._LoadProgress(0, "base", 123, 123, "tensors"))
            progress.finish()
        self.assertIn("123/123 tensors", output.getvalue())
        self.assertNotIn("GiB", output.getvalue())
        self.assertNotIn("weights/mtp", output.getvalue())

    def test_rank_failure_reaches_controller_before_distributed_teardown(self) -> None:
        events = []
        error = RuntimeError("rank cache capacity rejected")

        @contextmanager
        def distributed_scope():
            try:
                yield
            finally:
                # A real cleanup can wait indefinitely for a peer collective.
                self.assertEqual(events, ["reported"])
                events.append("teardown")

        with (
            self.assertRaises(RuntimeError) as caught,
            distributed_scope(),
            serving._report_before_teardown(lambda: events.append("reported")),
        ):
            raise error
        self.assertIs(caught.exception, error)
        self.assertEqual(events, ["reported", "teardown"])
        report = Mock()
        with serving._report_before_teardown(report):
            pass
        report.assert_not_called()

    def test_overlap_memory_retains_only_selected_lanes_and_preserves_other_owners(self) -> None:
        selected = tuple(
            SimpleNamespace(captured_m=32, _decode_overlap_estimated=True,
                            _decode_overlap_collective_threads=256) for _ in range(2)
        )
        rejected = SimpleNamespace(captured_m=16, _decode_overlap_estimated=False)
        regular, wave = object(), object()
        primary = SimpleNamespace(
            _decode_overlap_candidates=(),
            _decode_overlap_auxiliary=(*selected, rejected),
            _bootstrap_siblings=[regular, selected[0], rejected, wave, selected[1]],
        )
        body = SimpleNamespace(_decode_overlap_candidates=(selected,))
        plans = serving._retain_profiled_decode_overlap({1: primary, 64: body})
        self.assertEqual(plans, {64: serving.DecodeOverlapPlan((32, 32), True, 256)})
        self.assertEqual(primary._decode_overlap_auxiliary, selected)
        self.assertEqual(primary._bootstrap_siblings, [regular, selected[0], wave, selected[1]])
        body._decode_overlap_candidates = ()
        self.assertEqual(serving._retain_profiled_decode_overlap({1: primary, 64: body}), {})
        self.assertEqual(primary._decode_overlap_auxiliary, ())
        self.assertEqual(primary._bootstrap_siblings, [regular, wave])
        # Empty qualified plans must not reopen GPU calibration after KV sizing.
        with patch.object(serving, "collective_parameters") as policy:
            serving._prepare_decode_overlap_family(
                SimpleNamespace(_tp_size=4, _pp_size=1), {}, {}
            )
        policy.assert_not_called()

    def test_later_resident_bodies_do_not_hide_earlier_calibration_peak(self) -> None:
        primary = SimpleNamespace(
            captured_m=1,
            build_sibling=lambda **kw: SimpleNamespace(captured_m=kw["captured_m"]),
        )
        with (
            patch.object(serving, "_build_body", return_value=(primary, object())),
            patch.object(serving, "_prepare_decode_overlap_family"),
            patch.object(serving.torch.cuda, "reset_peak_memory_stats"),
            patch.object(
                serving.torch.cuda, "max_memory_allocated", side_effect=[150, 800, 950, 960]
            ),
            patch.object(serving.torch.cuda, "memory_allocated", side_effect=[100, 300, 900, 950]),
        ):
            body, _, _ = serving._build_body_family(
                object(), object(), SimpleNamespace(device=torch.device("cuda:0")), (1, 2, 4),
                max_sequence_tokens=16, page_capacity=16, state_capacity=4,
                scratch_capacity=4, wave_enabled=False,
            )
        # Whole-family peak-minus-live would report only ten bytes.
        self.assertEqual(body._construction_transient_bytes, 500)

    def test_continuation_witness_supports_small_graph_pools(self) -> None:
        body = SimpleNamespace(
            captured_m=4, buffers=SimpleNamespace(page_table=torch.empty((4, 1))),
            arena=SimpleNamespace(layout=SimpleNamespace(block_size_tokens=16)),
        )
        for length in (1, 2, 3, 4):
            rows = serving._continuation_audit_rows(body, length)
            self.assertEqual(len(rows), length)
            self.assertEqual(
                tuple(row.segment_length for row in rows), (length, *(0,) * (length - 1))
            )
        for length in (0, 5):
            with self.assertRaisesRegex(ValueError, "graph capacity"):
                serving._continuation_audit_rows(body, length)

    def test_single_rank_topology_progress_and_audit_consensus(self) -> None:
        topology = serving._topology((3,), tp_size=1, pp_size=1, ep_size=1)
        self.assertEqual(topology.world_size, 1)
        self.assertEqual(tuple((g.group_id, g.ranks) for g in topology.groups), (("tp:0", (0,)),))
        output = io.StringIO()
        with patch("cubic_serve.runtime.performance.sys.stderr", output):
            progress = serving._StartupProgress(1)
            progress.update(serving._LoadProgress(0, "base", 100, 100))
            progress.finish()
        self.assertIn("rank=0", output.getvalue())
        self.assertNotIn("rank=1", output.getvalue())
        context = multiprocessing.get_context("spawn")
        serving._wave_audit_consensus(
            context.Array("i", 2, lock=True), context.Barrier(1),
            rank=0, phase_id=1, failure_code=0, timeout_seconds=1,
        )

    def test_four_rank_updates_render_one_in_place_line_per_section(self) -> None:
        output = io.StringIO()
        progress = serving._StartupProgress()
        with patch("cubic_serve.runtime.performance.sys.stderr", output):
            for rank in range(4):
                progress.update(serving._LoadProgress(rank, "base", 0, 100))
            for rank in range(4):
                progress.update(serving._LoadProgress(rank, "base", 100, 100))
            progress.finish()

        rendered = output.getvalue()
        self.assertEqual(rendered.count("\n"), 8)
        self.assertIn("\x1b[4A", rendered)
        self.assertIn("weights/main_model", rendered)
        self.assertNotIn("module=", rendered)
        self.assertIn("rank=3", rendered)
        self.assertNotIn("weights/mtp", rendered)
        self.assertIn("100.0%", rendered)


class _FailingController(_FakeController):
    def submit(
        self,
        command: serving._GenerateCommand,
        output: serving._TokenSink,
    ) -> None:
        self.commands.append(command)
        output.fail(serving.Qwen3_5TP2PP2ServingError("replay failed"))
        self.close()


class _BlockingController(_FakeController):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.cancelled = threading.Event()
        self.submit_count = 0
        self.outputs: dict[str, serving._TokenSink] = {}

    def submit(
        self,
        command: serving._GenerateCommand,
        output: serving._TokenSink,
    ) -> None:
        self.commands.append(command)
        self.submit_count += 1
        if self.submit_count > 1:
            output.publish(22)
            output.finish()
            return
        self.outputs[command.request_id] = output
        self.entered.set()

    def cancel(self, request_id: str) -> None:
        self.cancelled.set()
        output = self.outputs.pop(request_id, None)
        if output is not None:
            output.finish()


class _GatedController(_FakeController):
    def __init__(self) -> None:
        super().__init__()
        self.second_started = threading.Event()
        self.release_second = threading.Event()
        self.outputs: list[serving._TokenSink] = []
        self.broker = threading.Thread(target=self._publish_second, daemon=True)
        self.broker.start()

    def submit(
        self,
        command: serving._GenerateCommand,
        output: serving._TokenSink,
    ) -> None:
        self.commands.append(command)
        self.outputs.append(output)
        output.publish(20)
        self.second_started.set()

    def _publish_second(self) -> None:
        if not self.release_second.wait(2):
            return
        for output in tuple(self.outputs):
            output.publish(21)
            output.finish()


class _FanoutController(_FakeController):
    def __init__(self) -> None:
        super().__init__(())
        self.outputs: dict[str, serving._TokenSink] = {}

    def submit(
        self,
        command: serving._GenerateCommand,
        output: serving._TokenSink,
    ) -> None:
        self.commands.append(command)
        self.outputs[command.request_id] = output

    def release(self) -> None:
        for output in tuple(self.outputs.values()):
            output.publish(20)
            output.finish()


class Qwen3_5TP2PP2PerformanceServingTests(unittest.TestCase):
    def test_speculative_graph_growth_preserves_owner_budget_and_runtime_reserve(self) -> None:
        attempted: list[tuple[int, int, int]] = []

        def plan(**kwargs: Any) -> tuple[Mock, int]:
            attempted.append((kwargs["scratch_capacity"], kwargs["state_slot_capacity"],
                              kwargs["reserve_bytes"]))
            if kwargs["scratch_capacity"] > 256:
                raise serving.Qwen3_5TP2PP2ServingError("extra review scratch does not fit")
            return Mock(), 0

        with patch.object(serving, "_plan_replacement_cache_capacity", side_effect=plan):
            selected = serving._largest_fitting_physical_graph(
                snapshot=Mock(), layout=Mock(), bootstrap_spec=Mock(),
                model_table_pages=16, auto_max_sequence_tokens=True,
                max_running_requests=256, state_slot_capacity=256,
                max_speculative_tokens=5, pipeline_parallel_size=1,
                reserve_bytes=100,
            )
        self.assertEqual(selected, 256)
        self.assertEqual(attempted, [(512, 256, 100), (256, 256, 100)])

    def test_wide_review_transactions_use_owner_disjoint_physical_scratch(self) -> None:
        for owners, capacity in ((4, 8), (32, 64), (256, 512)):
            body = cast(Any, SimpleNamespace(
                captured_m=capacity,
                arena=SimpleNamespace(spec=SimpleNamespace(state_slot_capacity=owners)),
                buffers=SimpleNamespace(page_table=torch.empty(capacity, 3)),
            ))
            for logical_m in (owners + 1, capacity - 1, capacity):
                rows = serving._review_audit_rows(body, logical_m)
                segments = serving._batch_segments(rows)
                self.assertLessEqual(len(segments), owners)
                destinations: set[int] = set()
                for start, stop in segments:
                    owner = rows[start].state_source_slot
                    self.assertLess(owner, owners)
                    self.assertEqual(rows[stop - 1].state_destination_slot, owner)
                    for physical in range(start, stop - 1):
                        slot = rows[physical].state_destination_slot
                        self.assertGreaterEqual(slot, owners)
                        self.assertLess(slot, owners + logical_m)
                        self.assertNotIn(slot, destinations)
                        destinations.add(slot)
                self.assertTrue(all(len(row.page_table) == 3 for row in rows))

    def test_memory_planner_selects_largest_fitting_graph_without_losing_mtp_depth(
        self,
    ) -> None:
        attempted: list[int] = []

        def plan(**kwargs: Any) -> tuple[Mock, int]:
            capacity = kwargs["scratch_capacity"]
            attempted.append(capacity)
            if capacity > 64:
                raise serving.Qwen3_5TP2PP2ServingError("does not fit")
            return Mock(), 0

        with patch.object(serving, "_plan_replacement_cache_capacity", side_effect=plan):
            selected = serving._largest_fitting_physical_graph(
                snapshot=Mock(),
                layout=Mock(),
                bootstrap_spec=Mock(),
                model_table_pages=16,
                auto_max_sequence_tokens=True,
                max_running_requests=256,
                state_slot_capacity=256,
                max_speculative_tokens=7,
                pipeline_parallel_size=2,
                reserve_bytes=123,
            )

        self.assertEqual(selected, 64)
        self.assertEqual(attempted, [128, 64])

    def test_pp1_memory_planner_does_not_demote_required_physical_graph(self) -> None:
        attempted: list[int] = []

        def reject(**kwargs: Any) -> tuple[Mock, int]:
            attempted.append(kwargs["scratch_capacity"])
            raise serving.Qwen3_5TP2PP2ServingError("does not fit")

        with (
            patch.object(serving, "_plan_replacement_cache_capacity", side_effect=reject),
            self.assertRaisesRegex(
                serving.Qwen3_5TP2PP2ServingError,
                "cannot fit required physical C256 graph",
            ),
        ):
            serving._largest_fitting_physical_graph(
                snapshot=Mock(),
                layout=Mock(),
                bootstrap_spec=Mock(),
                model_table_pages=16,
                auto_max_sequence_tokens=True,
                max_running_requests=256,
                state_slot_capacity=256,
                max_speculative_tokens=0,
                pipeline_parallel_size=1,
                reserve_bytes=0,
            )

        self.assertEqual(attempted, [256])

    def test_pp1_request_admission_extends_power_of_two_anchors(self) -> None:
        for requests in (1, 33, 255, 256, 257, 511, 512, 513, 1000, 2049):
            with self.subTest(requests=requests):
                anchors = serving._physical_graph_capacities(
                    pipeline_parallel_size=1, max_running_requests=requests
                )
                self.assertEqual(anchors[0], 1)
                self.assertTrue(all(b == 2 * a for a, b in pairwise(anchors)))
                self.assertGreaterEqual(anchors[-1], requests)
                if requests > 256:
                    self.assertLess(anchors[-2], requests)
                self.assertIn(
                    serving._bucket_for_logical_m(requests, anchors),
                    serving._serving_graph_capacities(
                        anchors, requests, pipeline_parallel_size=1
                    ),
                )
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                serving._physical_graph_capacities(
                    pipeline_parallel_size=1, max_running_requests=invalid
                )

    def test_pp1_extended_anchor_cannot_silently_demote_to_fitting_smaller_graph(self) -> None:
        attempted: list[int] = []

        def plan(**kwargs: Any) -> tuple[Mock, int]:
            capacity = kwargs["scratch_capacity"]
            attempted.append(capacity)
            if capacity > 256:
                raise serving.Qwen3_5TP2PP2ServingError("does not fit")
            return Mock(), 0

        with (
            patch.object(serving, "_plan_replacement_cache_capacity", side_effect=plan),
            self.assertRaisesRegex(
                serving.Qwen3_5TP2PP2ServingError,
                "cannot fit required physical C512 graph",
            ),
        ):
            serving._largest_fitting_physical_graph(
                snapshot=Mock(), layout=Mock(), bootstrap_spec=Mock(),
                model_table_pages=16, auto_max_sequence_tokens=True,
                max_running_requests=257, state_slot_capacity=1024,
                max_speculative_tokens=0, pipeline_parallel_size=1, reserve_bytes=0,
            )
        self.assertEqual(attempted, [512])

    def test_graph_pool_does_not_capture_above_admitted_concurrency(self) -> None:
        physical = serving._physical_graph_capacities()

        self.assertEqual(
            serving._serving_graph_capacities(physical, 128, pipeline_parallel_size=2),
            physical,
        )
        self.assertEqual(
            serving._serving_graph_capacities(
                serving._physical_graph_capacities(pipeline_parallel_size=1),
                256,
                pipeline_parallel_size=1,
            ),
            (*physical, 256),
        )
        self.assertEqual(
            serving._serving_graph_capacities(physical, 256, pipeline_parallel_size=2),
            (*physical, 256),
        )
        pp1 = serving._physical_graph_capacities(pipeline_parallel_size=1)
        self.assertFalse(serving._uses_wavefront(pp1, pp1, 1))
        self.assertTrue(serving._uses_wavefront(physical, (*physical, 256), 2))

    def test_pp1_speculative_pool_rotates_within_physical_token_budget(self) -> None:
        self.assertEqual(
            serving._physical_graph_capacities(0, pipeline_parallel_size=1)[-1],
            256,
        )
        physical = serving._physical_graph_capacities(1, pipeline_parallel_size=1)
        self.assertEqual(
            physical[-1],
            256,
        )
        self.assertEqual(
            serving._serving_graph_capacities(
                physical,
                256,
                pipeline_parallel_size=1,
                max_speculative_tokens=1,
            ),
            physical,
        )
        self.assertEqual(
            serving._serving_graph_capacities(
                physical,
                64,
                pipeline_parallel_size=1,
                max_speculative_tokens=1,
            ),
            physical,
        )

    def test_auto_context_uses_cache_left_after_graph_cow_pages(self) -> None:
        self.assertEqual(
            serving._effective_max_sequence_tokens(262_144, 4_311, 256),
            60_784,
        )
        self.assertEqual(
            serving._effective_max_sequence_tokens(262_144, 29_433, 128),
            262_144,
        )
        self.assertEqual(
            serving._effective_max_sequence_tokens(262_144, 4_311, 256, block_size=32),
            121_568,
        )
        # The shared byte pool has already excluded private graph scratch and
        # the live request's state. A physical M1024 graph does not own 1024
        # persistent requests and cannot impose another 2048-page deduction.
        self.assertEqual(
            serving._effective_max_sequence_tokens(
                262_144, 37, 1024, block_size=32, scratch_pages=0),
            1184,
        )
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            serving._effective_max_sequence_tokens(262_144, 37, 1024, scratch_pages=-1)

    def test_qualification_records_effective_context_and_resident_graphs(self) -> None:
        digests = tuple("b2s256:" + str(index) * 64 for index in range(1, 9))
        graph_buckets = tuple(
            (
                captured_m,
                serving._logical_m_values(captured_m, serving._PHYSICAL_GRAPH_BUCKET_CAPACITIES),
            )
            for captured_m in serving._PHYSICAL_GRAPH_BUCKET_CAPACITIES
        )
        material = {
            "model_id": "test-model",
            "inventory_digest": digests[0],
            "semantic_plan_digest": digests[1],
            "topology_digest": digests[2],
            "partition_digest": digests[3],
            "max_sequence_tokens": 72_144,
            "rank_graph_evidence_ids": digests[4:8],
            "graph_buckets": graph_buckets,
        }
        evidence_id = serving._graph_evidence_id(**material)

        qualification = serving.Qwen3_5TP2PP2ServingQualification(
            **material,
            graph_evidence_id=evidence_id,
        )

        self.assertEqual(qualification.max_sequence_tokens, 72_144)
        self.assertEqual(qualification.graph_buckets[-1][0], 128)

    def test_context_limit_defaults_to_model_and_allows_explicit_smaller_value(self) -> None:
        config = SimpleNamespace(
            graph=SimpleNamespace(text=SimpleNamespace(max_position_embeddings=262_144))
        )

        typed_config = cast(Any, config)
        self.assertEqual(serving._resolve_max_sequence_tokens(typed_config, None), 262_144)
        self.assertEqual(serving._resolve_max_sequence_tokens(typed_config, 8_192), 8_192)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            serving._resolve_max_sequence_tokens(typed_config, 0)
        with self.assertRaisesRegex(ValueError, "position table"):
            serving._resolve_max_sequence_tokens(typed_config, 262_145)

    @staticmethod
    def _batch_controller(
        *commands: serving._GenerateCommand, block_size: int = 16,
    ) -> serving._ProcessController:
        controller = serving._ProcessController.__new__(serving._ProcessController)
        controller._block_size = block_size
        controller._prefix_block_packer = struct.Struct(f"<{block_size}I")
        controller._worker_timeout = 60.0
        controller._max_running_requests = 16
        controller._requests = {
            command.request_id: serving._ScheduledRequest(
                command, serving._TokenStream(), time.monotonic()
            )
            for command in commands
        }
        controller._request_order = [command.request_id for command in commands]
        controller._serving_stats = serving._ServingStats()
        for record in controller._requests.values():
            controller._serving_stats.admit(record.admitted_at)
        controller._next_request_index = 0
        controller._state_slot_capacity = 32
        controller._state_scratch_capacity = serving._MAX_PHYSICAL_CAPTURED_M
        controller._free_state_slots = list(range(32))
        controller._page_capacity = 128
        controller.max_sequence_tokens = controller._page_capacity * block_size
        controller._free_pages = list(range(128))
        controller._free_mtp_pages = list(range(128))
        controller._shared_state_slots = {}
        controller._shared_pages = {}
        controller._shared_mtp_pages = {}
        controller._prefix_cache = OrderedDict()
        controller._prefix_cache_enabled = False
        controller._num_speculative_tokens = 0
        controller._physical_graph_capacities = serving._PHYSICAL_GRAPH_BUCKET_CAPACITIES
        controller._graph_capacities = serving._PHYSICAL_GRAPH_BUCKET_CAPACITIES
        heapq.heapify(controller._free_state_slots)
        heapq.heapify(controller._free_pages)
        heapq.heapify(controller._free_mtp_pages)
        return controller

    def test_nondefault_pages_preserve_prefill_prefix_and_continuation_ownership(self) -> None:
        for size in (32, 48):
            with self.subTest(page_tokens=size):
                tokens = tuple(range(2 * size + 1))
                command = serving._GenerateCommand("cold", tokens, 4, ())
                prefill = self._batch_controller(command, block_size=size)
                rows, _ = prefill._next_batch_locked()
                self.assertEqual(len(rows), len(tokens))
                self.assertEqual(len(rows[0].new_pages), 3)
                for start in range(0, len(rows), size):
                    chunk = rows[start:start + size]
                    self.assertEqual(len({row.page_destination for row in chunk}), 1)
                    self.assertNotEqual(chunk[0].page_source, chunk[0].page_destination)
                    self.assertTrue(all(row.page_source == row.page_destination
                                        for row in chunk[1:]))

                controller = self._batch_controller(command, block_size=size)
                controller._prefix_cache_enabled = True
                cold = controller._requests["cold"]
                cold.prefill_cursor = 2 * size
                cold.state_slot = heapq.heappop(controller._free_state_slots)
                cold.pages.extend(heapq.heappop(controller._free_pages) for _ in range(2))
                cold.state_initialized = True
                controller._publish_prefix_locked(cold)
                snapshot = next(iter(controller._prefix_cache.values()))
                controller._release_request_locked("cold")
                warm = serving._ScheduledRequest(
                    serving._GenerateCommand("warm", tokens, 4, ()),
                    serving._TokenStream(), time.monotonic(),
                )
                controller._serving_stats.admit(warm.admitted_at)
                controller._restore_prefix_locked(warm)
                self.assertEqual(warm.prefill_cursor, 2 * size)
                self.assertEqual(tuple(warm.pages), snapshot.pages)
                resumed = controller._reserve_segment_locked(warm, 1, running=0)
                self.assertIsNotNone(resumed)
                self.assertEqual(resumed[0].position, 2 * size)
                self.assertEqual(resumed[0].page_table[:2], snapshot.pages)

                command = serving._GenerateCommand("page", (10,), 2 * size, ())
                controller = self._batch_controller(command, block_size=size)
                record = controller._requests["page"]
                record.prefill_cursor = 1
                record.generated_count = size - 1
                record.last_token_id = 20
                record.state_slot = heapq.heappop(controller._free_state_slots)
                record.pages.append(heapq.heappop(controller._free_pages))
                record.state_initialized = True
                controller._batch_id = 0
                _, rows, selected = controller._next_dispatch_locked()
                controller._commit_steady_decode_locked(
                    selected, rows, (21,), serving.TokenWakeBatch(), profile=None,
                )
                continuation, rows, _ = controller._next_dispatch_locked()
                self.assertIsInstance(continuation, serving._ContinueBatchCommand)
                self.assertEqual(rows, ())
                self.assertEqual(continuation.page_updates, ((0, 1, record.pages[1]),))

    def test_serving_stats_use_real_busy_time_and_exclude_idle_gaps(self) -> None:
        stats = serving._ServingStats()

        with patch.object(serving._LOGGER, "info") as info:
            stats.admit(100.0)
            stats.step(40, 0, 1.0)
            stats.prefix_lookup(974, 1000)
            stats.publish(1, is_prefill=True)
            stats.step(0, 1, 3.0)
            stats.publish(3)
            stats.terminal(104.0, kv_usage_percent=97.375)

            self.assertEqual(info.call_count, 1)
            first = info.call_args.args[0]
            self.assertEqual(
                first,
                "[STATUS] n_req=1 kv=97.4% prefix=97.4% prefill=10.0 tok/s decode=1.0 tok/s",
            )
            self.assertEqual(stats.active_seconds_closed, 4.0)

            stats.admit(200.0)
            stats.step(0, 1, 2.0)
            stats.publish(1)
            stats.terminal(202.0)

            second = info.call_args.args[0]
            self.assertEqual(
                second,
                "[STATUS] n_req=1 kv=0.0% prefix=0.0% prefill=0.0 tok/s decode=0.5 tok/s",
            )
            self.assertEqual(stats.active_seconds_closed, 6.0)

    def test_serving_stats_merge_concurrent_requests_and_batch_mtp_logs(self) -> None:
        stats = serving._ServingStats()

        with patch.object(serving._LOGGER, "info") as info:
            for offset in range(128):
                stats.admit(float(offset) / 128)
            stats.step(0, 128, 5.0)
            for _ in range(128):
                stats.publish(1)
                stats.speculate(3, 5, 1)
            for _ in range(63):
                stats.terminal(4.0)
            self.assertEqual(info.call_count, 0)

            stats.terminal(4.0)
            self.assertEqual(info.call_count, 0)

            for _ in range(64):
                stats.terminal(5.0)
            self.assertEqual(info.call_count, 1)
            final = info.call_args.args[0]
            self.assertEqual(
                final,
                "[STATUS] n_req=128 kv=0.0% prefix=0.0% "
                "prefill=0.0 tok/s draft=5.0 accept=3.0 "
                "decode=25.6 tok/s",
            )
            self.assertEqual(stats.active_seconds_closed, 5.0)

    def test_serving_stats_periodic_denominator_uses_event_time(self) -> None:
        stats = serving._ServingStats()

        with patch.object(serving._LOGGER, "info") as info:
            stats.admit(0.0)
            stats.step(0, 1, 10.25)
            stats.publish(10)
            stats.maybe_report(10.25)
            first = info.call_args.args[0]
            self.assertIn("decode=1.0 tok/s", first)
            self.assertEqual(stats.last_report_active_seconds, 10.25)

            stats.step(0, 1, 3.75)
            stats.publish(4)
            stats.terminal(14.0)
            final = info.call_args.args[0]
            self.assertIn("decode=1.1 tok/s", final)
            self.assertEqual(stats.active_seconds_closed, 14.0)

    def test_serving_stats_mixed_batch_has_no_invented_phase_split(self) -> None:
        stats = serving._ServingStats()
        with patch.object(serving._LOGGER, "info") as info:
            stats.admit(0.0)
            stats.step(100, 2, 2.0)
            stats.publish(1, is_prefill=True)
            stats.publish(6)
            stats.maybe_report(10.0)
            line = info.call_args.args[0]
            self.assertIn("prefill=10.0 tok/s decode=0.7 tok/s", line)
            self.assertNotIn("mixed_prefill", line)
            self.assertNotIn("mixed_decode", line)
            stats.step(0, 1, 1.0)
            stats.publish(80)
            stats.terminal(20.0)
            self.assertIn("decode=8.0 tok/s", info.call_args.args[0])
            self.assertNotIn("mixed_", info.call_args.args[0])

    def test_decode_cannot_republish_advanced_state_as_an_evicted_prompt_prefix(self):
        command = serving._GenerateCommand("decoded", tuple(range(33)), 16, ())
        controller = self._batch_controller(command)
        controller._prefix_cache_enabled = True
        record = controller._requests["decoded"]
        record.prefill_cursor = 32
        record.state_slot = heapq.heappop(controller._free_state_slots)
        record.pages.extend(heapq.heappop(controller._free_pages) for _ in range(2))
        record.state_initialized = True
        record.generated_count = 3
        controller._publish_prefix_locked(record)
        self.assertFalse(controller._prefix_cache)
        self.assertFalse(controller._shared_state_slots)

    def test_aligned_prefix_snapshot_restores_target_mtp_and_recurrent_ownership(
        self,
    ) -> None:
        command = serving._GenerateCommand("cold", tuple(range(33)), 4, ())
        controller = self._batch_controller(command)
        controller._prefix_cache_enabled = True
        controller._num_speculative_tokens = 1
        cold = controller._requests["cold"]
        cold.prefill_cursor = 32
        cold.state_slot = heapq.heappop(controller._free_state_slots)
        cold.pages.extend(heapq.heappop(controller._free_pages) for _ in range(2))
        cold.mtp_pages.extend(heapq.heappop(controller._free_mtp_pages) for _ in range(2))
        cold.state_initialized = True

        controller._publish_prefix_locked(cold)
        snapshot = next(iter(controller._prefix_cache.values()))
        self.assertEqual(snapshot.token_ids, command.input_ids[:32])
        self.assertEqual(controller._shared_state_slots, {cold.state_slot: 1})

        controller._release_request_locked("cold")
        warm = serving._ScheduledRequest(
            serving._GenerateCommand("warm", command.input_ids, 4, ()),
            serving._TokenStream(),
            time.monotonic(),
        )
        controller._serving_stats.admit(warm.admitted_at)
        controller._restore_prefix_locked(warm)

        self.assertEqual(warm.prefill_cursor, 32)
        self.assertEqual(warm.state_slot, snapshot.state_slot)
        self.assertEqual(tuple(warm.pages), snapshot.pages)
        self.assertEqual(tuple(warm.mtp_pages), snapshot.mtp_pages)
        row = controller._reserve_segment_locked(warm, 1, running=0)
        assert row is not None
        self.assertEqual(row[0].position, 32)
        self.assertFalse(row[0].reset_state)
        self.assertEqual(row[0].mtp_position, 32)
        self.assertEqual(row[0].page_table[:2], snapshot.pages)
        self.assertEqual(row[0].mtp_page_table[:2], snapshot.mtp_pages)
        self.assertEqual(
            controller._serving_stats.prefix_hit_tokens_total,
            32,
        )

    def test_fixed_mtp_depths_share_one_segment_contract(self) -> None:
        for draft_count in (1, 2, 3):
            with self.subTest(draft_count=draft_count):
                command = serving._GenerateCommand(f"mtp:{draft_count}", (10,), draft_count + 2, ())
                controller = self._batch_controller(command)
                controller._num_speculative_tokens = draft_count
                record = controller._requests[command.request_id]
                record.prefill_cursor = 1
                record.generated_count = 1
                record.last_token_id = 20
                record.state_slot = heapq.heappop(controller._free_state_slots)
                record.pages.append(heapq.heappop(controller._free_pages))
                record.mtp_pages.append(heapq.heappop(controller._free_mtp_pages))
                record.state_initialized = True

                rows, selected = controller._next_batch_locked()

                self.assertEqual(selected, (record,))
                self.assertEqual(len(rows), draft_count + 1)
                self.assertEqual(rows[0].segment_length, draft_count + 1)
                self.assertEqual(
                    tuple(row.sample_ordinal for row in rows),
                    tuple(range(1, draft_count + 2)),
                )
                self.assertEqual(
                    len({row.state_destination_slot for row in rows}),
                    draft_count + 1,
                )
                self.assertEqual(rows[-1].state_destination_slot, record.state_slot)
                self.assertTrue(
                    all(
                        row.state_destination_slot >= controller._state_slot_capacity
                        for row in rows[:-1]
                    )
                )
                self.assertEqual(len(controller._free_state_slots), 31)

    def test_fixed_mtp_schedules_when_every_durable_owner_is_live(self) -> None:
        commands = tuple(
            serving._GenerateCommand(f"saturated:{index}", (10,), 4, ()) for index in range(256)
        )
        controller = self._batch_controller(*commands)
        controller._max_running_requests = 256
        controller._state_slot_capacity = 256
        controller._state_scratch_capacity = 256
        controller._free_state_slots = list(range(256))
        controller._free_pages = list(range(1024))
        controller._free_mtp_pages = list(range(1024))
        controller._num_speculative_tokens = 1
        for record in controller._requests.values():
            record.prefill_cursor = 1
            record.generated_count = 1
            record.last_token_id = 20
            record.state_slot = heapq.heappop(controller._free_state_slots)
            record.pages.append(heapq.heappop(controller._free_pages))
            record.mtp_pages.append(heapq.heappop(controller._free_mtp_pages))
            record.state_initialized = True

        rows, selected = controller._next_batch_locked()

        self.assertEqual(len(rows), serving._MAX_PHYSICAL_CAPTURED_M)
        self.assertEqual(len(selected), serving._MAX_PHYSICAL_CAPTURED_M // 2)
        self.assertEqual(controller._free_state_slots, [])
        self.assertEqual(
            {row.state_destination_slot for row in rows[::2]},
            set(range(256, 256 + len(selected))),
        )
        self.assertTrue(all(row.state_destination_slot < 256 for row in rows[1::2]))

    def test_fixed_mtp_rotates_requests_within_physical_token_budget(self) -> None:
        commands = tuple(
            serving._GenerateCommand(f"wave:{index}", (10,), 4, ()) for index in range(256)
        )
        controller = self._batch_controller(*commands)
        controller._max_running_requests = 256
        controller._physical_graph_capacities = serving._PP1_PHYSICAL_GRAPH_BUCKET_CAPACITIES
        controller._graph_capacities = serving._PP1_PHYSICAL_GRAPH_BUCKET_CAPACITIES
        controller._state_slot_capacity = 256
        controller._state_scratch_capacity = 256
        controller._free_state_slots = list(range(256))
        controller._free_pages = list(range(1024))
        controller._free_mtp_pages = list(range(1024))
        controller._num_speculative_tokens = 1
        for record in controller._requests.values():
            record.prefill_cursor = 1
            record.generated_count = 1
            record.last_token_id = 20
            record.state_slot = heapq.heappop(controller._free_state_slots)
            record.pages.append(heapq.heappop(controller._free_pages))
            record.mtp_pages.append(heapq.heappop(controller._free_mtp_pages))
            record.state_initialized = True

        rows, selected = controller._next_batch_locked()

        self.assertEqual((len(rows), len(selected)), (256, 128))
        self.assertEqual(
            tuple(row.state_destination_slot for row in rows[::2]),
            tuple(range(256, 384)),
        )

    def test_partial_mtp_acceptance_repairs_durable_state_from_scratch(self) -> None:
        executor = serving._CapturedGraphRank.__new__(serving._CapturedGraphRank)
        first = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        second = torch.arange(12, dtype=torch.bfloat16).reshape(6, 2)
        executor.body = cast(
            Any,
            SimpleNamespace(arena=SimpleNamespace(states={"a": first, "b": second})),
        )
        hidden = torch.arange(18, dtype=torch.bfloat16).reshape(6, 3)
        rows = (
            serving._BatchRow("partial", 1, 0, 0, 4, 0, 0, (0,), segment_length=2),
            serving._BatchRow("partial", 2, 1, 0, 0, 0, 0, (0,)),
            serving._BatchRow("full", 3, 0, 1, 5, 1, 1, (1,), segment_length=2),
            serving._BatchRow("full", 4, 1, 1, 1, 1, 1, (1,)),
        )
        segments = (
            serving._RaggedSpeculativeSegment(0, 0, 2),
            serving._RaggedSpeculativeSegment(1, 2, 4),
        )
        expected_first = first[4].clone()
        expected_second = second[4].clone()
        expected_hidden = hidden[4].clone()

        serving._repair_speculative_state(executor, hidden, rows, segments, (0, 1))

        torch.testing.assert_close(first[0], expected_first, rtol=0, atol=0)
        torch.testing.assert_close(second[0], expected_second, rtol=0, atol=0)
        torch.testing.assert_close(hidden[0], expected_hidden, rtol=0, atol=0)

    def test_mtp_repair_covers_every_partial_acceptance_depth(self) -> None:
        executor = serving._CapturedGraphRank.__new__(serving._CapturedGraphRank)
        states = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        executor.body = cast(
            Any,
            SimpleNamespace(arena=SimpleNamespace(states={"recurrent": states})),
        )
        hidden = torch.arange(24, dtype=torch.bfloat16).reshape(8, 3)
        rows = tuple(
            serving._BatchRow(
                "depth-five",
                token,
                offset,
                0,
                2 + offset if offset < 5 else 0,
                0,
                0,
                (0,),
                segment_length=6 if offset == 0 else 0,
            )
            for offset, token in enumerate(range(10, 16))
        )
        segment = serving._RaggedSpeculativeSegment(0, 0, 6)
        baseline_states = states.clone()
        baseline_hidden = hidden.clone()
        owners = states.shape[0]
        draft_states = {
            "raw_keys": torch.arange(owners * 8).reshape(owners, 2, 4).bfloat16(),
            "positions": torch.arange(owners * 6).reshape(owners, 2, 3),
        }
        baseline_draft = {name: value.clone() for name, value in draft_states.items()}

        for accepted in range(5):
            states.copy_(baseline_states)
            hidden.copy_(baseline_hidden)
            for name, value in draft_states.items():
                value.copy_(baseline_draft[name])
            source = rows[accepted].state_destination_slot

            serving._repair_speculative_state(
                executor, hidden, rows, (segment,), (accepted,), draft_state_fields=draft_states,
            )

            torch.testing.assert_close(states[0], baseline_states[source], rtol=0, atol=0)
            torch.testing.assert_close(hidden[0], baseline_hidden[source], rtol=0, atol=0)
            for name, value in draft_states.items():
                self.assertTrue(torch.equal(value[0], baseline_draft[name][source]))
                self.assertTrue(torch.equal(value[1:], baseline_draft[name][1:]))

    def test_dynamic_mtp_packs_independent_request_depths_in_one_batch(self) -> None:
        commands = tuple(
            serving._GenerateCommand(f"dynamic:{depth}", (10,), 16, ()) for depth in (0, 1, 3)
        )
        controller = self._batch_controller(*commands)
        controller._num_speculative_tokens = 1
        controller._max_speculative_tokens = 3
        for depth, record in zip((0, 1, 3), controller._requests.values(), strict=True):
            record.draft_tokens = depth
            record.prefill_cursor = 1
            record.generated_count = 1
            record.last_token_id = 20
            record.state_slot = heapq.heappop(controller._free_state_slots)
            record.pages.append(heapq.heappop(controller._free_pages))
            record.mtp_pages.append(heapq.heappop(controller._free_mtp_pages))
            record.state_initialized = True

        rows, selected = controller._next_batch_locked()

        self.assertEqual(
            tuple(record.command.request_id for record in selected),
            ("dynamic:0", "dynamic:1", "dynamic:3"),
        )
        self.assertEqual(len(rows), 1 + (1 + 1) + (3 + 1))
        self.assertEqual(serving._batch_segments(rows), ((0, 1), (1, 3), (3, 7)))
        speculative, levels = serving._ragged_speculative_plan(rows)
        self.assertEqual(tuple(segment.draft_count for segment in speculative), (1, 3))
        self.assertEqual(
            tuple(tuple(rows[segment.start].request_id for segment in level) for level in levels),
            (
                ("dynamic:1", "dynamic:3"),
                ("dynamic:3",),
                ("dynamic:3",),
            ),
        )
        self.assertEqual(rows[0].request_id, "dynamic:0")
        self.assertEqual(rows[0].segment_length, 1)
        self.assertEqual(serving._mtp_maintenance_pairs(rows, speculative), ())
        self.assertEqual(len({row.state_destination_slot for row in rows}), len(rows))
        for start, stop in serving._batch_segments(rows):
            request_mtp_pages = rows[start].mtp_page_table
            self.assertEqual(bool(request_mtp_pages), rows[start].request_id != "dynamic:0")
            self.assertTrue(
                all(row.mtp_page_table == request_mtp_pages for row in rows[start:stop])
            )

    def test_complete_cohort_waits_then_restores_one_common_depth(self):
        from cubic_serve.runtime.speculative import DraftPolicyConfig, DynamicDraftPlanner, DynamicDraftPolicy
        config = DraftPolicyConfig(decision_tokens=1, decision_min_rounds=1)
        commands = tuple(serving._GenerateCommand(key, (10,), 64, ()) for key in ("a", "b"))
        controller = self._batch_controller(*commands)
        controller._num_speculative_tokens = 1
        controller._max_speculative_tokens = 3
        planner = controller._draft_planner = DynamicDraftPlanner((1, 2, 4, 8, 16), config=config)
        planner.seed_costs(target_samples=((1, 2.), (2, 3.), (4, 4.), (8, 20.), (16, 35.)),
                           proposal_samples=((1, 2.), (2, 3.), (4, 5.)))
        planner.seed_compound_costs(((8, 4, 3.),))
        records = tuple(controller._requests.values())
        for record in records:
            record.draft_policy = DynamicDraftPolicy(1, 3, config)
            record.draft_tokens = 1
            record.prefill_cursor = record.generated_count = 1
            record.last_token_id = 20
            record.state_slot = heapq.heappop(controller._free_state_slots)
            record.pages.append(heapq.heappop(controller._free_pages))
            record.mtp_pages.append(heapq.heappop(controller._free_mtp_pages))
            record.state_initialized = True
        # Use the measured planner in both calls; readiness differs per request.
        controller._draft_trace = False
        controller._draft_cost_planner = lambda *a, **kw: planner
        records[0].draft_policy.observe(accepted=1, proposed=1)
        rows, _ = controller._next_batch_locked()
        self.assertEqual(tuple(stop-start for start, stop in serving._batch_segments(rows)), (2, 2))
        self.assertTrue(records[0].draft_policy.ready)
        self.assertEqual(tuple(r.draft_tokens for r in records), (1, 1))
        records[1].draft_policy.observe(accepted=1, proposed=1)
        rows, _ = controller._next_batch_locked()
        self.assertEqual(tuple(stop-start for start, stop in serving._batch_segments(rows)), (4, 4))
        self.assertEqual(tuple(r.draft_tokens for r in records), (3, 3))
        self.assertFalse(any(r.draft_policy.ready or r.draft_policy.stopped for r in records))
        self.assertLessEqual(len(rows), controller._state_scratch_capacity)

    def test_terminal_cohort_exit_preserves_ready_window_then_stops_all_owners(self):
        from cubic_serve.runtime.speculative import (
            DraftPolicyConfig,
            DynamicDraftPlanner,
            DynamicDraftPolicy,
        )
        for seeded, ready_owner in ((False, 0), (False, 1), (True, 0), (True, 1)):
            with self.subTest(compound_profile=seeded, ready_owner=ready_owner):
                config = DraftPolicyConfig(decision_tokens=1, decision_min_rounds=1)
                commands = tuple(serving._GenerateCommand(key, (10,), 64, ()) for key in ("a", "b"))
                controller = self._batch_controller(*commands)
                controller._num_speculative_tokens = 1
                controller._max_speculative_tokens = 3
                planner = controller._draft_planner = DynamicDraftPlanner(
                    (1, 2, 4, 8, 16), config=config,
                )
                planner.seed_costs(target_samples=((n, 1.) for n in planner.graph_capacities),
                                   proposal_samples=((1, 10.), (2, 10.), (4, 10.)))
                if seeded:
                    planner.seed_compound_costs(((4, 2, 100.),))
                records = tuple(controller._requests.values())
                for record in records:
                    record.draft_policy = DynamicDraftPolicy(1, 3, config)
                    record.draft_tokens = 1
                    record.prefill_cursor = record.generated_count = 1
                    record.last_token_id = 20
                    record.state_slot = heapq.heappop(controller._free_state_slots)
                    record.pages.append(heapq.heappop(controller._free_pages))
                    record.mtp_pages.append(heapq.heappop(controller._free_mtp_pages))
                    record.state_initialized = True
                controller._draft_trace = False
                controller._draft_cost_planner = lambda *a, planner=planner, **kw: planner
                records[ready_owner].draft_policy.observe(accepted=1, proposed=1)
                rows, _ = controller._next_batch_locked()
                self.assertEqual(
                    tuple(stop-start for start, stop in serving._batch_segments(rows)), (2, 2),
                )
                self.assertTrue(records[ready_owner].draft_policy.ready)
                self.assertFalse(any(r.draft_policy.stopped for r in records))
                records[1-ready_owner].draft_policy.observe(accepted=1, proposed=1)
                rows, _ = controller._next_batch_locked()
                self.assertEqual(tuple(row.request_id for row in rows), ("a", "b"))
                self.assertEqual(
                    tuple(stop-start for start, stop in serving._batch_segments(rows)), (1, 1),
                )
                self.assertEqual(tuple(r.draft_tokens for r in records), (0, 0))
                self.assertTrue(all(r.draft_policy.stopped and r.draft_stopped for r in records))
                for record in records:
                    with self.assertRaises(RuntimeError):
                        record.draft_policy.move_toward(1)

    def test_ragged_verification_reduces_each_request_commit_independently(self) -> None:
        commands = tuple(
            serving._GenerateCommand(f"commit:{depth}", (10,), 16, ()) for depth in (1, 3, 2)
        )
        controller = self._batch_controller(*commands)
        controller._num_speculative_tokens = 1
        controller._max_speculative_tokens = 3
        for depth, record in zip((1, 3, 2), controller._requests.values(), strict=True):
            record.draft_tokens = depth
            record.prefill_cursor = 1
            record.generated_count = 1
            record.last_token_id = 20
            record.state_slot = heapq.heappop(controller._free_state_slots)
            record.pages.append(heapq.heappop(controller._free_pages))
            record.mtp_pages.append(heapq.heappop(controller._free_mtp_pages))
            record.state_initialized = True
        rows, _ = controller._next_batch_locked()
        speculative, levels = serving._ragged_speculative_plan(rows)
        self.assertEqual(serving._ragged_proposal_buckets(levels), (4, 2, 1))

        reduced = serving._ragged_acceptance_rows(
            speculative,
            levels,
            (
                ((1, 100), (1, 200), (0, 300)),
                ((0, 201), (1, 301)),
                ((1, 202),),
            ),
        )

        self.assertEqual(reduced, ((1, 100), (1, 201), (0, 300)))

    def test_ragged_proposal_hidden_compacts_only_surviving_request_rows(self) -> None:
        executor = serving._CapturedMTPRank.__new__(serving._CapturedMTPRank)
        executor.captured_m = 4
        norm = torch.arange(12, dtype=torch.bfloat16).view(4, 3)
        executor.body = SimpleNamespace(buffers=SimpleNamespace(norm=norm))
        executor._feedback = norm
        executor._proposal_hidden = torch.empty_like(norm)

        compact = executor.chain_hidden((1, 3))

        torch.testing.assert_close(compact, norm[(1, 3), :])

    def test_depth_executor_stashes_candidates_in_stable_verifier_history(self) -> None:
        owner = serving._CapturedMTPRank.__new__(serving._CapturedMTPRank)
        owner.captured_m = 4
        owner._max_draft_tokens = 3
        owner._verification_q_scores = torch.zeros((3, 2, 4, 1), dtype=torch.bfloat16)
        owner._verification_q_token_ids = torch.zeros((3, 2, 4, 1), dtype=torch.int64)
        owner._verification_draft_tokens = torch.zeros((3, 4), dtype=torch.int64)
        owner._verification_ordinals = torch.zeros((3, 4), dtype=torch.int64)
        owner._verification_temperatures = torch.zeros((3, 4), dtype=torch.float32)
        owner._verification_top_ks = torch.zeros((3, 4), dtype=torch.int64)
        owner._verification_top_ps = torch.zeros((3, 4), dtype=torch.float32)
        owner._verification_seeds = torch.zeros((3, 4), dtype=torch.int64)

        source = serving._CapturedMTPRank.__new__(serving._CapturedMTPRank)
        source.captured_m = 2
        workspace = SimpleNamespace(
            gathered_scores=torch.arange(4, dtype=torch.bfloat16).view(2, 2, 1),
            gathered_token_ids=torch.arange(10, 14, dtype=torch.int64).view(2, 2, 1),
        )
        source._sampling = SimpleNamespace(
            workspace=workspace,
            constraint_bound=False,
            ordinals=torch.tensor((20, 21)),
            temperatures=torch.tensor((0.5, 0.75)),
            top_ks=torch.tensor((4, 5)),
            top_ps=torch.tensor((0.8, 0.9)),
            seeds=torch.tensor((30, 31)),
        )
        source._token_ids = torch.tensor((40, 41))

        owner.stash_proposal(1, 2, source=source)

        torch.testing.assert_close(
            owner._verification_q_scores[1, :, :2], workspace.gathered_scores
        )
        torch.testing.assert_close(owner._verification_draft_tokens[1, :2], source._token_ids)
        torch.testing.assert_close(
            owner._verification_temperatures[1, :2], source._sampling.temperatures
        )

    def test_request_k_can_drop_from_one_to_zero_without_more_drafting(self) -> None:
        command = serving._GenerateCommand("dynamic:down", (10,), 16, ())
        controller = self._batch_controller(command)
        controller._num_speculative_tokens = 1
        controller._max_speculative_tokens = 1
        record = controller._requests[command.request_id]
        record.draft_tokens = 1
        record.prefill_cursor = 1
        record.generated_count = 1
        record.last_token_id = 20
        record.state_slot = heapq.heappop(controller._free_state_slots)
        record.pages.append(heapq.heappop(controller._free_pages))
        record.mtp_pages.append(heapq.heappop(controller._free_mtp_pages))
        record.state_initialized = True

        first_rows, _ = controller._next_batch_locked()
        self.assertEqual(len(first_rows), 2)
        for slot in {row.state_destination_slot for row in first_rows}:
            heapq.heappush(controller._free_state_slots, slot)
        for page in {row.page_destination for row in first_rows}:
            heapq.heappush(controller._free_pages, page)

        record.draft_tokens = 0
        second_rows, _ = controller._next_batch_locked()
        speculative, levels = serving._ragged_speculative_plan(second_rows)

        self.assertEqual(len(second_rows), 1)
        self.assertEqual((speculative, levels), ((), ()))
        self.assertEqual(serving._mtp_maintenance_pairs(second_rows, speculative), ())
        self.assertTrue(record.draft_stopped)
        record.draft_tokens = 1
        with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, "cannot resume"):
            serving._effective_request_draft_tokens(record, default=1, maximum=1)

    def test_dynamic_request_observations_advance_its_scheduled_depth(self) -> None:
        record = serving._ScheduledRequest(
            serving._GenerateCommand("dynamic:advance", (10,), 256, ()),
            serving._TokenStream(),
            time.monotonic(),
            draft_policy=serving.DynamicDraftPolicy(
                1,
                3,
                serving.DraftPolicyConfig(decision_tokens=4, decision_min_rounds=2),
            ),
            draft_tokens=1,
        )

        serving._advance_dynamic_draft(
            record,
            accepted=1,
            proposed=1,
            terminal=False,
        )
        serving._advance_dynamic_draft(
            record,
            accepted=1,
            proposed=1,
            terminal=False,
        )

        assert record.draft_policy is not None
        decision = record.draft_policy.move_toward(2)
        record.draft_tokens = decision.next_tokens
        self.assertEqual(record.draft_tokens, 2)
        self.assertFalse(record.draft_stopped)

    def test_dynamic_request_can_terminally_demote_to_target_only(self) -> None:
        record = serving._ScheduledRequest(
            serving._GenerateCommand("dynamic:stop", (10,), 256, ()),
            serving._TokenStream(),
            time.monotonic(),
            draft_policy=serving.DynamicDraftPolicy(
                1,
                3,
                serving.DraftPolicyConfig(decision_tokens=2, decision_min_rounds=2),
            ),
            draft_tokens=1,
        )

        for _ in range(2):
            serving._advance_dynamic_draft(
                record,
                accepted=0,
                proposed=1,
                terminal=False,
            )

        assert record.draft_policy is not None
        decision = record.draft_policy.move_toward(0)
        record.draft_tokens = decision.next_tokens
        record.draft_stopped = record.draft_policy.stopped
        self.assertEqual(record.draft_tokens, 0)
        self.assertTrue(record.draft_stopped)
        self.assertEqual(
            serving._effective_request_draft_tokens(record, default=1, maximum=3),
            0,
        )

    def test_prefix_eviction_preserves_a_live_warm_owner(self) -> None:
        command = serving._GenerateCommand("cold", tuple(range(33)), 4, ())
        controller = self._batch_controller(command)
        controller._prefix_cache_enabled = True
        controller._num_speculative_tokens = 1
        cold = controller._requests["cold"]
        cold.prefill_cursor = 32
        cold.state_slot = heapq.heappop(controller._free_state_slots)
        cold.pages.extend(heapq.heappop(controller._free_pages) for _ in range(2))
        cold.mtp_pages.extend(heapq.heappop(controller._free_mtp_pages) for _ in range(2))
        cold.state_initialized = True
        controller._publish_prefix_locked(cold)
        snapshot = next(iter(controller._prefix_cache.values()))
        controller._release_request_locked("cold")

        warm = serving._ScheduledRequest(
            serving._GenerateCommand("warm", command.input_ids, 4, ()),
            serving._TokenStream(),
            time.monotonic(),
        )
        controller._serving_stats.admit(warm.admitted_at)
        controller._restore_prefix_locked(warm)
        controller._evict_prefix_locked(
            len(controller._free_state_slots) + 1,
            len(controller._free_pages) + 1,
            len(controller._free_mtp_pages) + 1,
        )

        self.assertFalse(controller._prefix_cache)
        self.assertNotIn(snapshot.state_slot, controller._free_state_slots)
        self.assertTrue(set(snapshot.pages).isdisjoint(controller._free_pages))
        self.assertTrue(set(snapshot.mtp_pages).isdisjoint(controller._free_mtp_pages))
        controller._requests["warm"] = warm
        controller._request_order.append("warm")
        controller._release_request_locked("warm")
        self.assertIn(snapshot.state_slot, controller._free_state_slots)
        self.assertTrue(set(snapshot.pages).issubset(controller._free_pages))
        self.assertTrue(set(snapshot.mtp_pages).issubset(controller._free_mtp_pages))

    def test_concurrent_prefix_hits_cow_target_mtp_and_recurrent_destinations(
        self,
    ) -> None:
        command = serving._GenerateCommand("cold", tuple(range(33)), 4, ())
        controller = self._batch_controller(command)
        controller._prefix_cache_enabled = True
        controller._num_speculative_tokens = 1
        cold = controller._requests["cold"]
        cold.prefill_cursor = 32
        cold.state_slot = heapq.heappop(controller._free_state_slots)
        cold.pages.extend(heapq.heappop(controller._free_pages) for _ in range(2))
        cold.mtp_pages.extend(heapq.heappop(controller._free_mtp_pages) for _ in range(2))
        cold.state_initialized = True
        controller._publish_prefix_locked(cold)
        snapshot = next(iter(controller._prefix_cache.values()))
        controller._release_request_locked("cold")

        rows = []
        for request_id in ("warm:a", "warm:b"):
            warm = serving._ScheduledRequest(
                serving._GenerateCommand(request_id, command.input_ids, 4, ()),
                serving._TokenStream(),
                time.monotonic(),
            )
            controller._restore_prefix_locked(warm)
            segment = controller._reserve_segment_locked(warm, 1, running=0)
            assert segment is not None
            rows.append(segment[0])

        self.assertEqual({row.state_source_slot for row in rows}, {snapshot.state_slot})
        self.assertEqual(len({row.state_destination_slot for row in rows}), 2)
        self.assertEqual(len({row.page_destination for row in rows}), 2)
        self.assertEqual(len({row.mtp_page_destination for row in rows}), 2)
        for row in rows:
            self.assertEqual(row.page_table[:2], snapshot.pages)
            self.assertEqual(row.mtp_page_table[:2], snapshot.mtp_pages)

    def test_last_prefix_branch_defers_then_reuses_source_at_owner_capacity(self) -> None:
        command = serving._GenerateCommand("cold", tuple(range(33)), 4, ())
        controller = self._batch_controller(command)
        controller._prefix_cache_enabled = True
        cold = controller._requests["cold"]
        cold.prefill_cursor = 32
        cold.state_slot = heapq.heappop(controller._free_state_slots)
        cold.pages.extend(heapq.heappop(controller._free_pages) for _ in range(2))
        cold.state_initialized = True
        controller._publish_prefix_locked(cold)
        source = cold.state_slot
        assert source is not None
        controller._release_request_locked("cold")

        # Model a two-request owner pool: the cached source plus one free COW
        # destination.  The second branch must wait, not be failed.
        controller._free_state_slots = [heapq.heappop(controller._free_state_slots)]
        warm = []
        for request_id in ("warm:a", "warm:b"):
            record = serving._ScheduledRequest(
                serving._GenerateCommand(request_id, command.input_ids, 4, ()),
                serving._TokenStream(),
                time.monotonic(),
            )
            controller._restore_prefix_locked(record)
            warm.append(record)

        first = controller._reserve_segment_locked(warm[0], 1, running=2)
        assert first is not None
        self.assertIsNone(controller._reserve_segment_locked(warm[1], 1, running=2))
        self.assertFalse(warm[1].cancelled)
        self.assertFalse(warm[1].output_closed)

        controller._release_shared(
            controller._free_state_slots,
            controller._shared_state_slots,
            source,
        )
        second = controller._reserve_segment_locked(warm[1], 1, running=2)
        assert second is not None
        self.assertEqual(second[0].state_source_slot, source)
        self.assertEqual(second[0].state_destination_slot, source)

    def test_duplicate_prefix_publication_keeps_one_snapshot_owner(self) -> None:
        first_command = serving._GenerateCommand("first", tuple(range(33)), 4, ())
        second_command = serving._GenerateCommand("second", tuple(range(33)), 4, ())
        controller = self._batch_controller(first_command, second_command)
        controller._prefix_cache_enabled = True
        for request_id in ("first", "second"):
            record = controller._requests[request_id]
            record.prefill_cursor = 32
            record.state_slot = heapq.heappop(controller._free_state_slots)
            record.pages.extend(heapq.heappop(controller._free_pages) for _ in range(2))
            record.state_initialized = True

        first = controller._requests["first"]
        second = controller._requests["second"]
        controller._publish_prefix_locked(first)
        snapshot = next(iter(controller._prefix_cache.values()))
        controller._publish_prefix_locked(second)

        self.assertEqual(len(controller._prefix_cache), 1)
        self.assertEqual(controller._shared_state_slots, {snapshot.state_slot: 1})
        self.assertEqual(controller._shared_pages, {page: 1 for page in snapshot.pages})
        controller._release_request_locked("second")
        self.assertIn(snapshot.state_slot, controller._shared_state_slots)
        self.assertTrue(all(page in controller._shared_pages for page in snapshot.pages))

    def test_submit_rejects_before_prefix_hit_can_exceed_running_capacity(self) -> None:
        commands = tuple(
            serving._GenerateCommand(f"running:{index}", tuple(range(33)), 4, ())
            for index in range(16)
        )
        controller = self._batch_controller(*commands)
        controller._closed = False
        controller._poisoned = False
        controller._condition = threading.Condition()
        controller._prefix_cache_enabled = True

        with self.assertRaisesRegex(
            serving.Qwen3_5TP2PP2Overloaded,
            "maximum running request",
        ):
            controller.submit(
                serving._GenerateCommand("running:16", tuple(range(33)), 4, ()),
                serving._TokenStream(),
            )

        self.assertNotIn("running:16", controller._requests)

    def test_exclusive_request_updates_recurrent_state_in_place(self) -> None:
        commands = tuple(
            serving._GenerateCommand(f"request:{index}", (10 + index,), 2, ()) for index in range(5)
        )
        controller = self._batch_controller(*commands)

        rows, selected = controller._next_batch_locked()

        self.assertEqual(len(rows), 5)
        self.assertEqual(
            tuple(row.request_id for row in rows),
            tuple(command.request_id for command in commands),
        )
        self.assertEqual(len({row.state_source_slot for row in rows}), 5)
        self.assertEqual(len({row.state_destination_slot for row in rows}), 5)
        self.assertEqual(
            tuple(row.state_source_slot for row in rows),
            tuple(row.state_destination_slot for row in rows),
        )
        self.assertEqual(len({row.page_source for row in rows}), 5)
        self.assertEqual(len({row.page_destination for row in rows}), 5)
        self.assertEqual(len(selected), 5)
        self.assertEqual(controller._next_request_index, 0)

    def test_prefill_is_one_segment_transaction_across_kv_pages(self) -> None:
        command = serving._GenerateCommand("request:chunk", tuple(range(34)), 2, ())
        controller = self._batch_controller(command)

        rows, selected = controller._next_batch_locked()

        self.assertEqual(len(rows), 34)
        self.assertEqual(selected, (controller._requests[command.request_id],))
        self.assertEqual(serving._batch_segments(rows), ((0, 34),))
        self.assertEqual(tuple(row.segment_length for row in rows), (34,) + (0,) * 33)
        self.assertEqual(len({row.state_source_slot for row in rows}), 1)
        self.assertEqual(len({row.state_destination_slot for row in rows}), 1)
        self.assertEqual(len({row.page_destination for row in rows}), 3)
        self.assertEqual(len(rows[0].new_pages), 3)
        for page_index in range(3):
            page_rows = rows[
                page_index * serving._BLOCK_SIZE : (page_index + 1) * serving._BLOCK_SIZE
            ]
            if not page_rows:
                continue
            self.assertNotEqual(page_rows[0].page_source, page_rows[0].page_destination)
            self.assertTrue(all(row.page_source == row.page_destination for row in page_rows[1:]))

    def test_graph_pool_selects_smallest_qualified_multi_m_bucket(self) -> None:
        expected = {
            1: 1,
            2: 2,
            3: 4,
            4: 4,
            5: 8,
            8: 8,
            9: 16,
            16: 16,
            17: 32,
            32: 32,
            33: 64,
            64: 64,
            65: 128,
            128: 128,
            129: 256,
            256: 256,
        }
        self.assertEqual(
            {logical_m: serving._bucket_for_logical_m(logical_m) for logical_m in expected},
            expected,
        )
        with self.assertRaises(serving.Qwen3_5TP2PP2Overloaded):
            serving._bucket_for_logical_m(257)

    def test_speculative_depth_uses_shared_multi_m_anchors(self) -> None:
        physical = serving._physical_graph_capacities(7)
        serving_capacities = (*physical, 256)

        self.assertEqual(physical[:4], (1, 2, 4, 8))
        self.assertEqual(serving._logical_m_values(1, physical), (1,))
        for capacity in physical[1:]:
            self.assertGreater(len(serving._logical_m_values(capacity, physical)), 1)
        self.assertEqual(
            tuple(
                serving._bucket_for_logical_m(value, serving_capacities) for value in range(1, 9)
            ),
            (1, 2, 4, 4, 8, 8, 8, 8),
        )
        self.assertEqual(serving._bucket_for_logical_m(9, serving_capacities), 16)

    def test_worker_command_bytes_fan_out_and_close_are_strict(self) -> None:
        row = serving._BatchRow("request:ipc", 7, 0, 0, 1, 0, 1, (1,))
        batch = serving._BatchCommand(3, 4, (row,))
        payload = serving._serialize_worker_command(batch)
        context = multiprocessing.get_context("spawn")
        pipes = tuple(context.Pipe(duplex=False) for _ in range(serving._WORLD_SIZE))
        try:
            for _, sender in pipes:
                sender.send_bytes(payload)
            self.assertEqual(
                tuple(
                    serving._deserialize_worker_command(receiver.recv_bytes())
                    for receiver, _ in pipes
                ),
                (batch,) * serving._WORLD_SIZE,
            )

            continuation = serving._ContinueBatchCommand(
                4, batch.batch_id, batch.captured_m, (row.request_id,)
            )
            self.assertEqual(
                serving._deserialize_worker_command(
                    serving._serialize_worker_command(continuation)
                ),
                continuation,
            )

            close_payload = serving._serialize_worker_command(serving._CloseCommand())
            self.assertIsInstance(
                serving._deserialize_worker_command(close_payload),
                serving._CloseCommand,
            )
            with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, "unknown"):
                serving._deserialize_worker_command(
                    pickle.dumps(serving._GenerateCommand("wrong", (1,), 1, ()))
                )
            with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, "invalid"):
                serving._deserialize_worker_command(b"not a pickle")
        finally:
            for receiver, sender in pipes:
                receiver.close()
                sender.close()

    def test_constraint_wire_is_optional_and_preserves_request_ownership(self) -> None:
        from cubic_serve.contracts.request import StructuredOutputMeta
        from cubic_serve.generation.structured_sampling import ConstraintBinding

        row = serving._BatchRow("grammar", 7, 0, 0, 1, 0, 1, (1,))
        plain = serving._BatchCommand(3, 4, (row,))
        original = serving._serialize_worker_command(plain)
        wire = pickle.loads(original)
        self.assertEqual(len(wire), 5)
        self.assertEqual(len(wire[4][0]), 19)
        binding = ConstraintBinding("grammar", StructuredOutputMeta.regex("[ab]+"), 2, 9, (7,))
        constrained = replace(plain, constraints=(binding,), released_constraints=("old",))
        self.assertEqual(
            serving._deserialize_worker_command(serving._serialize_worker_command(constrained)),
            constrained,
        )
        self.assertEqual(
            serving._serialize_worker_command(
                replace(constrained, constraints=(), released_constraints=())
            ),
            original,
        )
        foreign = replace(constrained, constraints=(replace(binding, request_id="foreign"),))
        with self.assertRaises(serving.Qwen3_5TP2PP2ServingError):
            serving._deserialize_worker_command(serving._serialize_worker_command(foreign))

    def test_budget_wire_roundtrip_preserves_controls_and_rejects_version_downgrade(self):
        from cubic_serve.contracts.request import ReasoningControl, StructuredOutputMeta
        from cubic_serve.generation.structured_sampling import ConstraintBinding

        row = serving._BatchRow("budget", 7, 0, 0, 1, 0, 1, (1,))
        control = ReasoningControl((10, 11), (7, 8), 4, True, 3, initial_tail=(7,))
        for meta in (StructuredOutputMeta.identity(), StructuredOutputMeta.regex("[ab]+")):
            binding = ConstraintBinding("budget", meta, 2, 9, (7, 8), control, ())
            command = serving._BatchCommand(3, 4, (row,), (binding,))
            payload = serving._serialize_worker_command(command)
            self.assertEqual(serving._deserialize_worker_command(payload), command)
            wire = pickle.loads(payload)
            self.assertEqual(wire[0], "cubic.worker.v2")
            self.assertEqual(len(wire[5][0]), 9)
            downgraded = ("cubic.worker.v1", *wire[1:])
            with self.assertRaises(serving.Qwen3_5TP2PP2ServingError):
                serving._deserialize_worker_command(pickle.dumps(downgraded))
            truncated = (*wire[:5], (wire[5][0][:7],), wire[6])
            with self.assertRaises(serving.Qwen3_5TP2PP2ServingError):
                serving._deserialize_worker_command(pickle.dumps(truncated))

    def test_steady_mtp0_decode_commit_preserves_owned_state_and_output(self) -> None:
        command = serving._GenerateCommand("steady", (10,), 4, ())
        controller = self._batch_controller(command)
        record = controller._requests[command.request_id]
        record.prefill_cursor = 1
        record.generated_count = 1
        record.last_token_id = 20
        record.state_slot = heapq.heappop(controller._free_state_slots)
        record.pages.append(heapq.heappop(controller._free_pages))
        record.state_initialized = True
        state_slot = record.state_slot
        pages = tuple(record.pages)
        rows, selected = controller._next_batch_locked()
        wakes = serving.TokenWakeBatch()

        metrics = controller._commit_steady_decode_locked(
            selected,
            rows,
            (21,),
            wakes,
            profile=None,
        )
        wakes.flush()

        self.assertEqual(metrics, (0, 0))
        self.assertEqual(record.generated_count, 2)
        self.assertEqual(record.last_token_id, 21)
        self.assertEqual(record.state_slot, state_slot)
        self.assertEqual(tuple(record.pages), pages)
        self.assertEqual(next(record.output), 21)

    def test_auto_prefill_uses_graph_budget_and_keeps_mixed_decode_quota(self) -> None:
        prompt = serving._GenerateCommand("long", tuple(range(600)), 4, ())
        for cap in (None, 16):
            controller = self._batch_controller(prompt)
            controller._prefill_chunk_tokens = cap
            rows, _ = controller._next_batch_locked()
            self.assertEqual(len(rows), cap or controller._graph_capacities[-1])
        decode = serving._GenerateCommand("decode", (10,), 32, ())
        controller = self._batch_controller(decode, prompt)
        controller._prefill_chunk_tokens = None
        record = controller._requests[decode.request_id]
        record.prefill_cursor = 1
        record.generated_count = 1
        record.last_token_id = 20
        record.state_slot = heapq.heappop(controller._free_state_slots)
        record.pages.append(heapq.heappop(controller._free_pages))
        record.state_initialized = True
        rows, _ = controller._next_batch_locked()
        self.assertTrue(any(row.request_id == decode.request_id for row in rows))
        self.assertEqual(
            sum(not row.is_decode for row in rows),
            controller._graph_capacities[-1] - 1,
        )

    def test_mixed_prefill_preserves_decode_owners_and_physical_wave_segments(self) -> None:
        cases = (
            (16, 16, 3, 2, 0),
            (256, 256, 1, 1, 0),
            (256, 256, 8, 3, 0),
            (512, 256, 1, 1, 0),
            (512, 256, 3, 2, 0),
            (16, 16, 3, 2, 3),
        )
        for budget, wave, decoded, pending, depth in cases:
            with self.subTest(budget=budget, wave=wave, decoded=decoded, depth=depth):
                commands = tuple(
                    serving._GenerateCommand(f"decode:{i}", (10 + i,), 1024, ())
                    for i in range(decoded)
                ) + tuple(
                    serving._GenerateCommand(f"prefill:{i}", tuple(range(600)), 1024, ())
                    for i in range(pending)
                )
                controller = self._batch_controller(*commands)
                controller._graph_capacities = (1, budget)
                controller._physical_graph_capacities = (1, wave)
                controller._num_speculative_tokens = depth
                for i in range(decoded):
                    record = controller._requests[f"decode:{i}"]
                    record.prefill_cursor = record.generated_count = 1
                    record.last_token_id = 20
                    record.state_slot = heapq.heappop(controller._free_state_slots)
                    record.pages.append(heapq.heappop(controller._free_pages))
                    record.state_initialized = True
                rows, selected = controller._next_batch_locked()
                self.assertLessEqual(len(rows), budget)
                self.assertEqual(
                    {row.request_id for row in rows if row.is_decode},
                    {f"decode:{i}" for i in range(decoded)},
                )
                self.assertTrue(any(not row.is_decode for row in rows))
                self.assertTrue(all(row.segment_length <= wave for row in rows))
                self.assertEqual(
                    len({row.state_destination_slot for row in rows if row.segment_length}),
                    len(selected),
                )
                for i in range(pending):
                    segment = [row for row in rows if row.request_id == f"prefill:{i}"]
                    if not segment:
                        continue  # An atomic segment can wait for the next physical wave.
                    self.assertEqual([row.token_id for row in segment], list(range(len(segment))))
                    self.assertEqual([row.position for row in segment], list(range(len(segment))))
                    self.assertEqual(segment[0].segment_length, len(segment))
                    self.assertTrue(all(row.segment_length == 0 for row in segment[1:]))

    def test_prefill_consumes_old_hidden_before_owned_slot_publication(self) -> None:
        executor = serving._CapturedGraphRank.__new__(serving._CapturedGraphRank)
        executor._target_hidden_slots = torch.tensor([[11.0, 12.0]])
        executor.body = SimpleNamespace(
            buffers=SimpleNamespace(norm=torch.tensor([[21.0, 22.0], [31.0, 32.0]]))
        )
        executor.bind_batch = Mock()
        executor.replay_bound = Mock()
        rows = tuple(
            serving._BatchRow(
                "owned", 17 + i, i, 0, 0, 0, 0, (0,), segment_length=2 if i == 0 else 0
            )
            for i in range(2)
        )
        executor.replay_batch(rows, commit_hidden=False)
        draft_input = torch.stack((executor._target_hidden_slots[0], executor.body.buffers.norm[0]))
        executor._commit_target_hidden(rows)
        self.assertTrue(torch.equal(draft_input, torch.tensor([[11.0, 12.0], [21.0, 22.0]])))
        self.assertTrue(torch.equal(executor._target_hidden_slots, torch.tensor([[31.0, 32.0]])))

    def test_owned_decode_uses_continuation_until_the_next_page_boundary(self) -> None:
        command = serving._GenerateCommand("continue", (10,), 32, ())
        controller = self._batch_controller(command)
        record = controller._requests[command.request_id]
        record.prefill_cursor = 1
        record.generated_count = 1
        record.last_token_id = 20
        record.state_slot = heapq.heappop(controller._free_state_slots)
        record.pages.append(heapq.heappop(controller._free_pages))
        record.state_initialized = True
        controller._batch_id = 0

        batch, rows, selected = controller._next_dispatch_locked()
        self.assertIsInstance(batch, serving._BatchCommand)
        controller._commit_steady_decode_locked(
            selected, rows, (21,), serving.TokenWakeBatch(), profile=None
        )
        continuation, next_rows, next_selected = controller._next_dispatch_locked()

        self.assertIsInstance(continuation, serving._ContinueBatchCommand)
        self.assertEqual(next_rows, ())
        self.assertEqual(next_selected, selected)
        self.assertEqual(continuation.page_updates, ())

    def test_steady_continuation_allocates_only_the_crossing_page(self) -> None:
        command = serving._GenerateCommand("page", (10,), 32, ())
        controller = self._batch_controller(command)
        record = controller._requests[command.request_id]
        record.prefill_cursor = 1
        record.generated_count = 15
        record.last_token_id = 20
        record.state_slot = heapq.heappop(controller._free_state_slots)
        record.pages.append(heapq.heappop(controller._free_pages))
        record.state_initialized = True
        controller._batch_id = 0

        _batch, rows, selected = controller._next_dispatch_locked()
        controller._commit_steady_decode_locked(
            selected, rows, (21,), serving.TokenWakeBatch(), profile=None
        )
        continuation, next_rows, _ = controller._next_dispatch_locked()

        self.assertIsInstance(continuation, serving._ContinueBatchCommand)
        self.assertEqual(next_rows, ())
        self.assertEqual(len(record.pages), 2)
        self.assertEqual(continuation.page_updates, ((0, 1, record.pages[1]),))

    def test_terminal_k0_owned_decode_uses_continuation_until_the_next_page_boundary(self) -> None:
        command = serving._GenerateCommand("continue", (10,), 32, ())
        controller = self._batch_controller(command)
        record = controller._requests[command.request_id]
        controller._num_speculative_tokens = 7
        record.draft_stopped = True
        record.prefill_cursor = 1
        record.generated_count = 1
        record.last_token_id = 20
        record.state_slot = heapq.heappop(controller._free_state_slots)
        record.pages.append(heapq.heappop(controller._free_pages))
        record.state_initialized = True
        controller._batch_id = 0

        batch, rows, selected = controller._next_dispatch_locked()
        self.assertIsInstance(batch, serving._BatchCommand)
        controller._commit_steady_decode_locked(
            selected, rows, (21,), serving.TokenWakeBatch(), profile=None
        )
        continuation, next_rows, next_selected = controller._next_dispatch_locked()

        self.assertIsInstance(continuation, serving._ContinueBatchCommand)
        self.assertEqual(next_rows, ())
        self.assertEqual(next_selected, selected)
        self.assertEqual(continuation.page_updates, ())

    def test_terminal_k0_continuation_allocates_only_the_crossing_page(self) -> None:
        command = serving._GenerateCommand("page", (10,), 32, ())
        controller = self._batch_controller(command)
        record = controller._requests[command.request_id]
        controller._num_speculative_tokens = 7
        record.draft_stopped = True
        record.prefill_cursor = 1
        record.generated_count = 15
        record.last_token_id = 20
        record.state_slot = heapq.heappop(controller._free_state_slots)
        record.pages.append(heapq.heappop(controller._free_pages))
        record.state_initialized = True
        controller._batch_id = 0

        _batch, rows, selected = controller._next_dispatch_locked()
        controller._commit_steady_decode_locked(
            selected, rows, (21,), serving.TokenWakeBatch(), profile=None
        )
        continuation, next_rows, _ = controller._next_dispatch_locked()

        self.assertIsInstance(continuation, serving._ContinueBatchCommand)
        self.assertEqual(next_rows, ())
        self.assertEqual(len(record.pages), 2)
        self.assertEqual(continuation.page_updates, ((0, 1, record.pages[1]),))

    def test_steady_decode_commit_rejects_shared_ownership(self) -> None:
        command = serving._GenerateCommand("shared", (10,), 4, ())
        controller = self._batch_controller(command)
        record = controller._requests[command.request_id]
        record.prefill_cursor = 1
        record.generated_count = 1
        record.last_token_id = 20
        record.state_slot = heapq.heappop(controller._free_state_slots)
        record.pages.append(heapq.heappop(controller._free_pages))
        record.state_initialized = True
        rows, selected = controller._next_batch_locked()
        controller._shared_state_slots[record.state_slot] = 1

        self.assertIsNone(
            controller._commit_steady_decode_locked(
                selected,
                rows,
                (21,),
                serving.TokenWakeBatch(),
                profile=None,
            )
        )
        self.assertEqual(record.generated_count, 1)

    def test_cold_audit_covers_lower_interior_and_full_bucket(self) -> None:
        self.assertEqual(serving._cold_audit_logical_ms(1), (1,))
        self.assertEqual(serving._cold_audit_logical_ms(2), (1, 2))
        self.assertEqual(serving._cold_audit_logical_ms(4), (3, 4))
        self.assertEqual(serving._cold_audit_logical_ms(8), (5, 6, 7, 8))
        self.assertEqual(serving._cold_audit_logical_ms(16), (9, 12, 16))
        self.assertEqual(serving._cold_audit_logical_ms(32), (17, 24, 32))
        self.assertEqual(serving._cold_audit_logical_ms(64), (33, 48, 64))
        self.assertEqual(serving._cold_audit_logical_ms(128), (65, 96, 128))
        self.assertEqual(serving._cold_audit_logical_ms(256), (129, 192, 256))

    def test_feeding_profile_is_opt_in_and_retains_bounded_aggregates(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(serving._new_step_feeding_profile("controller"))
        with patch.dict("os.environ", {"CUBIC_SERVE_FEEDING_PROFILE": "1"}):
            profile = serving._new_step_feeding_profile("controller")
        assert profile is not None
        for ordinal in range(serving._FEEDING_PROFILE_SAMPLE_LIMIT + 3):
            profile.record("result_wait_cpu_ms", float(ordinal))

        summary = cast(dict[str, Any], profile.summary())
        metrics = cast(dict[str, dict[str, int | float]], summary["metrics"])
        result_wait = metrics["result_wait_cpu_ms"]
        self.assertEqual(result_wait["count"], serving._FEEDING_PROFILE_SAMPLE_LIMIT + 3)
        self.assertEqual(result_wait["retained"], serving._FEEDING_PROFILE_SAMPLE_LIMIT)
        self.assertEqual(summary["role"], "controller")

    def test_feeding_profile_emits_and_stops_at_role_anchor_window(self) -> None:
        environment = {
            "CUBIC_SERVE_FEEDING_PROFILE": "1",
            "CUBIC_SERVE_FEEDING_PROFILE_STEPS": "2",
        }
        with patch.dict("os.environ", environment, clear=True):
            profile = serving._new_step_feeding_profile("worker-rank-0")
        assert profile is not None
        with patch.object(serving._LOGGER, "warning") as warning:
            profile.record("validity_host_sync_cpu_ms", 1.0)
            self.assertTrue(profile.active)
            profile.record("validity_host_sync_cpu_ms", 2.0)
            self.assertFalse(profile.active)
            profile.record("validity_host_sync_cpu_ms", 1000.0)

        warning.assert_called_once()
        summary = cast(dict[str, Any], profile.summary())
        metrics = cast(dict[str, dict[str, int | float]], summary["metrics"])
        self.assertEqual(metrics["validity_host_sync_cpu_ms"]["count"], 2)
        self.assertEqual(summary["sample_steps"], 2)

    def test_audit_cow_pairs_never_alias_and_ping_pong_across_steps(self) -> None:
        forward = tuple(serving._audit_cow_pair(row, 0) for row in range(128))
        reverse = tuple(serving._audit_cow_pair(row, 1) for row in range(128))

        self.assertTrue(all(source != destination for source, destination in forward))
        self.assertEqual(
            {slot for pair in forward for slot in pair},
            set(range(256)),
        )
        self.assertEqual(
            reverse,
            tuple((destination, source) for source, destination in forward),
        )
        with self.assertRaises(ValueError):
            serving._audit_cow_pair(-1, 0)
        with self.assertRaises(ValueError):
            serving._audit_cow_pair(0, 2)

    def test_sampler_warmup_owns_finite_inputs_after_partial_model_audits(self) -> None:
        observed = []
        executors = {}
        for capacity, domain in ((1, (1,)), (4, (3, 4))):
            logits = torch.full((capacity, 16), float("nan"))

            def sample(rows, logits=logits):
                self.assertTrue(torch.isfinite(logits[:len(rows)]).all())
                observed.append(len(rows))

            executors[capacity] = SimpleNamespace(
                slot=SimpleNamespace(logits=logits),
                body=SimpleNamespace(bucket_logical_ms=domain, hidden=torch.zeros(1)),
                sample_tokens=sample,
            )
        with patch.object(torch.cuda, "current_stream"):
            serving._warm_sampling(cast(Any, executors))
        self.assertEqual(observed, [1, 3, 4])

    def test_masked_sampling_prior_reuses_qualified_heads_and_restores_warm_state(self):
        calls = []
        state = SimpleNamespace(constraint_bound=False)
        def bind(masks, owners):
            self.assertEqual((masks, owners), ([None], [True]))
            state.constraint_bound = True
        state.bind_constraints = bind
        def sample(rows):
            calls.append((len(rows), rows[0].sampling.temperature, state.constraint_bound))
        executor = SimpleNamespace(
            _sampling=state, slot=SimpleNamespace(logits=torch.zeros(1, 8)),
            body=SimpleNamespace(bucket_logical_ms=(1,), hidden=torch.zeros(1)),
            sample_tokens=sample,
        )
        with patch.object(torch.cuda, 'current_stream'), patch.object(torch.cuda, 'synchronize'):
            prices = serving._warm_sampling({1: executor}, price_sampling_through=1)
        self.assertEqual([(rows, greedy) for rows, greedy, _ms in prices], [(1, False), (1, True)])
        self.assertTrue(all(ms > 0 for _rows, _greedy, ms in prices))
        self.assertFalse(state.constraint_bound)
        self.assertEqual(calls[0], calls[-1])
        self.assertEqual(sum(masked and temperature == 0 for _, temperature, masked in calls), 4)
        self.assertEqual(sum(masked and temperature == 1 for _, temperature, masked in calls), 4)

    def test_batch_row_staging_is_host_only_and_bulk_restores_padding(self) -> None:
        rank = serving._CapturedGraphRank.__new__(serving._CapturedGraphRank)
        rank.captured_m = 4
        rank.max_sequence_tokens = 64
        rank.vocab_size = 128
        rank._table_pages = 3
        rank._bound_active_rows = 3
        rank._local_state_layers = torch.arange(2, dtype=torch.int32)
        rank._state_destination_column = 8
        rank._metadata_columns = 9 + 4 + rank._table_pages
        rank._source_slots_slice = slice(9, 11)
        rank._destination_slots_slice = slice(11, 13)
        rank._page_table_slice = slice(13, 16)
        rank._metadata_hosts = tuple(torch.full((4, 16), -777, dtype=torch.int32) for _ in range(2))
        rank._metadata_host_arrays = tuple(item.numpy() for item in rank._metadata_hosts)
        rank._index_page_offset = 4
        rank._index_capacity = 16
        rank._index_hosts = tuple(torch.empty(16, dtype=torch.int64) for _ in range(2))
        rank._index_host_arrays = tuple(item.numpy() for item in rank._index_hosts)
        rank._scratch_pages = (100, 101, 102, 103)
        rank._scratch_states = (50, 51, 52, 53)
        rank.body = cast(
            Any,
            SimpleNamespace(
                stage=0,
                embedding_span=SimpleNamespace(start=32, stop=96),
                embedding_mask=torch.empty((4, 1), dtype=torch.bfloat16),
                arena=SimpleNamespace(
                    spec=SimpleNamespace(
                        page_capacity=100,
                        state_slot_capacity=50,
                        scratch_capacity=4,state_scratch_capacity=None,state_scratch_capacity_per_arena=4,
                        scratch_arena_ids=("test",),
                    )
                ),
            ),
        )
        rows = (
            serving._BatchRow(
                "request:a",
                41,
                17,
                3,
                50,
                7,
                8,
                (2, 5),
                reset_state=True,
                new_pages=(8,),
            ),
            serving._BatchRow("request:b", 113, 2, 9, 10, 11, 12, (6,), new_pages=(12, 13)),
        )

        transfer_rows, reset_count, new_page_count = rank._stage_batch_rows(rows, host_slot=0)

        metadata = rank._metadata_hosts[0]
        indices = rank._index_hosts[0]
        self.assertEqual((transfer_rows, reset_count, new_page_count), (3, 1, 3))
        self.assertEqual(metadata[0, :7].tolist(), [17, 7, 8, 1, 18, 9, 1])
        self.assertEqual(metadata[0, 7:13].tolist(), [1, 50, 6, 7, 100, 101])
        self.assertEqual(metadata[0, 13:].tolist(), [2, 5, -777])
        self.assertEqual(metadata[1, 5:7].tolist(), [0, 0])
        self.assertEqual(metadata[1, 13:].tolist(), [6, 12, -777])
        self.assertEqual(
            metadata[2].tolist(),
            [0, 102, 102, 0, 1, 0, 0, 0, 51, 102, 103, 102, 103, 102, 102, -777],
        )
        self.assertEqual(rank._staged_page_table_pages, 2)
        self.assertEqual(indices[0].item(), 3)
        self.assertEqual(indices[4:7].tolist(), [8, 12, 13])

        buffers = SimpleNamespace(
            position_ids=torch.empty((3, 4), dtype=torch.int64),
            page_source=torch.empty(4, dtype=torch.int32),
            page_destination=torch.empty(4, dtype=torch.int32),
            token_offsets=torch.empty(4, dtype=torch.int32),
            sequence_lengths=torch.empty(4, dtype=torch.int32),
            segment_lengths=torch.empty(4, dtype=torch.int32),
            page_table=torch.full((4, 3), 555, dtype=torch.int32),
            conv_slots=torch.empty((2, 2, 4), dtype=torch.int32),
            recurrent_slots=torch.empty((2, 2, 4), dtype=torch.int32),
            token_ids=torch.empty(4, dtype=torch.int64),
        )
        body = cast(Any, rank.body)
        buffers.state_slot_maps = (buffers.conv_slots, buffers.recurrent_slots)
        body.buffers = buffers
        rank._bind_staged_metadata(
            metadata[:transfer_rows], transfer_rows, rank._staged_page_table_pages
        )

        self.assertEqual(tuple(body.embedding_mask.shape), (4, 1))
        self.assertEqual(body.embedding_mask[:3].tolist(), [[1.0], [0.0], [0.0]])
        self.assertEqual(buffers.position_ids[:, :3].tolist(), [[17, 2, 0]] * 3)
        self.assertEqual(buffers.conv_slots[:, :, 2].tolist(), [[102, 103]] * 2)
        self.assertEqual(buffers.page_table[0].tolist(), [2, 5, 555])

        # Independent typed state slabs need no GDN-specific attribute names.
        expected_slots = buffers.conv_slots[:, :, :transfer_rows].clone()
        buffers.state_slot_maps = tuple(
            torch.full((2, 2, 4), -99, dtype=torch.int32) for _ in range(3)
        )
        del buffers.conv_slots
        del buffers.recurrent_slots
        rank._bind_staged_metadata(metadata[:transfer_rows], transfer_rows, 2)
        for state_map in buffers.state_slot_maps:
            self.assertTrue(torch.equal(state_map[:, :, :transfer_rows], expected_slots))
            self.assertTrue((state_map[:, :, transfer_rows:] == -99).all())

        # A startup witness can leave inactive rows pointing at real pages.
        # First binding covers the entire bucket, so even an unmasked cache
        # writer cannot overwrite those old owners from a padding row.
        buffers.page_destination.copy_(torch.arange(4, dtype=torch.int32))
        rank._bound_active_rows = rank.captured_m
        transfer_rows, _, _ = rank._stage_batch_rows(rows[:1], host_slot=0)
        self.assertEqual(transfer_rows, rank.captured_m)
        rank._bind_staged_metadata(
            metadata[:transfer_rows], transfer_rows, rank._staged_page_table_pages
        )
        pages = torch.full((104,), -1, dtype=torch.int32)
        pages[buffers.page_destination.long()] = torch.arange(4, dtype=torch.int32)
        self.assertEqual(pages[8].item(), 0)
        self.assertTrue((pages[:8] == -1).all())
        self.assertTrue((pages[9:100] == -1).all())
        self.assertEqual(pages[101:].tolist(), [1, 2, 3])

    def test_batch_row_staging_rejects_before_mutating_device_state(self) -> None:
        rank = serving._CapturedGraphRank.__new__(serving._CapturedGraphRank)
        rank.captured_m = 1
        rank.max_sequence_tokens = 8
        rank.vocab_size = 16
        rank._table_pages = 1
        rank._bound_active_rows = 0
        rank._local_state_layers = torch.arange(1, dtype=torch.int32)
        rank._metadata_columns = 10
        rank._source_slots_slice = slice(7, 8)
        rank._destination_slots_slice = slice(8, 9)
        rank._page_table_slice = slice(9, 10)
        rank._metadata_hosts = (torch.full((1, 10), -1, dtype=torch.int32),)
        rank._metadata_host_arrays = (rank._metadata_hosts[0].numpy(),)
        rank._index_page_offset = 1
        rank._index_capacity = 2
        rank._index_hosts = (torch.full((2,), -1, dtype=torch.int64),)
        rank._index_host_arrays = (rank._index_hosts[0].numpy(),)
        rank._scratch_pages = (10,)
        rank._scratch_states = (20,)
        rank.body = cast(
            Any,
            SimpleNamespace(
                stage=1,
                embedding_span=None,
                embedding_mask=None,
                arena=SimpleNamespace(
                    spec=SimpleNamespace(
                        page_capacity=4,
                        state_slot_capacity=4,
                        scratch_capacity=1,state_scratch_capacity=None,state_scratch_capacity_per_arena=1,
                        scratch_arena_ids=("test",),
                    )
                ),
            ),
        )

        with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, "capacity"):
            rank._stage_batch_rows(
                (serving._BatchRow("bad", 1, 0, 0, 1, 0, 9, (0,)),),
                host_slot=0,
            )

        self.assertTrue(torch.equal(rank._metadata_hosts[0], torch.full((1, 10), -1)))
        self.assertTrue(torch.equal(rank._index_hosts[0], torch.full((2,), -1)))

    def test_private_state_sources_require_binding_permission_and_exclude_padding(self) -> None:
        rank = serving._CapturedGraphRank.__new__(serving._CapturedGraphRank)
        rank.captured_m = rank._bound_active_rows = 4
        rank.max_sequence_tokens, rank.vocab_size, rank._table_pages = 64, 128, 1
        rank._local_state_layers = torch.arange(1, dtype=torch.int32)
        rank._state_destination_column = 8
        rank._source_slots_slice, rank._destination_slots_slice = slice(9, 10), slice(10, 11)
        rank._page_table_slice = slice(11, 12)
        rank._metadata_hosts = (torch.empty((4, 12), dtype=torch.int32),)
        rank._metadata_host_arrays = (rank._metadata_hosts[0].numpy(),)
        rank._index_hosts = (torch.empty(8, dtype=torch.int64),)
        rank._index_host_arrays = (rank._index_hosts[0].numpy(),)
        rank._index_page_offset = 4
        rank._scratch_states, rank._scratch_pages = (4, 5, 6, 7), (12, 13, 14, 15)
        rank.body = cast(Any, SimpleNamespace(stage=1, arena=SimpleNamespace(spec=SimpleNamespace(
            state_slot_capacity=4,scratch_capacity=4,state_scratch_capacity=None,state_scratch_capacity_per_arena=4,
            scratch_arena_ids=("owned",),
            page_capacity=16,
        ))))
        row = serving._BatchRow("draft", 7, 1, 4, 0, 0, 0, (0,))
        with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, "capacity"):
            rank._stage_batch_rows((row,), host_slot=0)
        rank._private_state_sources = True
        count, _, _ = rank._stage_batch_rows((row,), host_slot=0)
        self.assertEqual(count, 4)
        self.assertEqual(rank._metadata_hosts[0][:, 9].tolist(), [4, 5, 6, 7])
        self.assertEqual(rank._metadata_hosts[0][:, 10].tolist(), [0, 5, 6, 7])
        for invalid in (replace(row, state_source_slot=8),
                        replace(row, state_source_slot=-1), replace(row, reset_state=True)):
            with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, "capacity"):
                rank._stage_batch_rows((invalid,), host_slot=0)
        # A compact later proposal wave reads a checkpoint written by a
        # larger target bucket in the same leased scratch arena.
        rank.body.arena.spec.scratch_capacity = 8
        rank.body.arena.spec.state_scratch_capacity_per_arena = 8
        rank._stage_batch_rows((replace(row, state_source_slot=10),), host_slot=0)
        self.assertEqual(rank._metadata_hosts[0][0, 9].item(), 10)
        with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, "capacity"):
            rank._stage_batch_rows((replace(row, state_source_slot=12),), host_slot=0)
        # Checkpoints not read in the current ragged wave still belong to the
        # transaction. Borrow padding from the full leased arena, preserving
        # earlier/future depth checkpoints in the compact bucket's prefix.
        rank._reserved_scratch_states = frozenset((4, 5, 6, 7, 10))
        rank._stage_batch_rows((replace(row, state_source_slot=10),), host_slot=0)
        self.assertEqual(rank._metadata_hosts[0][1:, 10].tolist(), [8, 9, 11])
        rank._reserved_scratch_states = frozenset(range(4, 12))
        with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, "padding scratch"):
            rank._stage_batch_rows((replace(row, state_source_slot=10),), host_slot=0)

    def test_private_source_graph_rejects_missing_joint_scratch_budget(self) -> None:
        for capacity in (2, 4, 8, 16, 32, 64):
            body = SimpleNamespace(captured_m=capacity, arena=SimpleNamespace(
                layout=SimpleNamespace(block_size_tokens=16),
                spec=SimpleNamespace(scratch_capacity=2 * capacity - 2),
            ))
            with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError,
                                        "live checkpoints and graph padding"):
                serving._CapturedGraphRank(
                    body, None, None, None, SimpleNamespace(block_size_tokens=16),
                    private_state_sources=True,
                )

    def test_compact_draft_padding_preserves_full_transaction_checkpoint_set(self) -> None:
        for capacity in (1, 2, 4, 8, 16, 32, 64):
            for logical in sorted({1, capacity // 2 + 1, capacity}):
                rank = serving._CapturedGraphRank.__new__(serving._CapturedGraphRank)
                rank.captured_m = rank._bound_active_rows = capacity
                rank.max_sequence_tokens, rank.vocab_size, rank._table_pages = 64, 128, 1
                rank._block_size = 16
                rank._local_state_layers = torch.arange(1, dtype=torch.int32)
                rank._state_destination_column = 8
                rank._source_slots_slice = slice(9, 10)
                rank._destination_slots_slice = slice(10, 11)
                rank._page_table_slice = slice(11, 12)
                metadata = torch.empty((capacity, 12), dtype=torch.int32)
                indices = torch.empty(2 * capacity, dtype=torch.int64)
                rank._metadata_hosts, rank._metadata_host_arrays = (metadata,), (metadata.numpy(),)
                rank._index_hosts, rank._index_host_arrays = (indices,), (indices.numpy(),)
                rank._index_page_offset = capacity
                rank._scratch_states = tuple(range(capacity, 2 * capacity))
                rank._scratch_pages = tuple(range(4 * capacity, 5 * capacity))
                rank._private_state_sources = True
                rank._reserved_scratch_states = frozenset(rank._scratch_states)
                rank.body = cast(Any, SimpleNamespace(stage=1, arena=SimpleNamespace(
                    spec=SimpleNamespace(state_slot_capacity=capacity,
                                         scratch_capacity=2*capacity,state_scratch_capacity=None,state_scratch_capacity_per_arena=2*capacity,
                                         scratch_arena_ids=("lease",),
                                         page_capacity=4 * capacity),
                )))
                rows = tuple(serving._BatchRow(
                    str(i), 7, 1, capacity + i, i, i, i, (i,),
                ) for i in reversed(range(logical)))
                count, _, _ = rank._stage_batch_rows(rows, host_slot=0)
                self.assertEqual(count, capacity)
                self.assertEqual(metadata[:logical, 9].tolist(),
                                 [row.state_source_slot for row in rows])
                self.assertEqual(metadata[:logical, 10].tolist(),
                                 [row.state_destination_slot for row in rows])
                padding = metadata[logical:, 10].tolist()
                self.assertEqual(len(set(padding)), capacity - logical)
                self.assertTrue(set(padding).isdisjoint(rank._reserved_scratch_states))
                self.assertTrue(all(2 * capacity <= slot < 3 * capacity for slot in padding))

    def test_batch_bind_and_replay_are_distinct_single_use_phases(self) -> None:
        class FakeStream:
            def __init__(self) -> None:
                self.waits: list[object] = []

            def wait_stream(self, stream: object) -> None:
                self.waits.append(stream)

        class FakeSlot:
            def __init__(self) -> None:
                self.stream = FakeStream()
                self.replays = 0

            def select_decode_overlap(self, independent_rows: bool) -> None:
                self.independent_rows = independent_rows

            def replay(self) -> None:
                self.replays += 1

        class FakeEvent:
            def __init__(self) -> None:
                self.records: list[object] = []
                self.synchronizes = 0

            def record(self, stream: object) -> None:
                self.records.append(stream)

            def synchronize(self) -> None:
                self.synchronizes += 1

        def make_rank() -> tuple[serving._CapturedGraphRank, FakeSlot, FakeStream]:
            from cubic_serve.runtime.cache import (
                CacheArena,
                CacheArenaSpec,
                KVLayout,
                StateSlotSpec,
            )

            rank = serving._CapturedGraphRank.__new__(serving._CapturedGraphRank)
            slot = FakeSlot()
            current = FakeStream()
            rank.slot = cast(Any, slot)
            rank.body = cast(
                Any,
                SimpleNamespace(
                    hidden=torch.empty(1),
                    buffers=SimpleNamespace(row_activity=torch.zeros(3, dtype=torch.int32)),
                    arena=CacheArena(
                        CacheArenaSpec("bind-test", 4, 4,
                            (StateSlotSpec("state", torch.bfloat16, (2,)),)),
                        KVLayout((0,), 16, 1, 2), "cpu", rank_id=0,
                    ),
                    set_active_rows=Mock(),
                ),
            )
            rank._feeding_profiler = None
            rank._batch_bound = False
            rank._batch_phase_failed = False
            rank._next_host_slot = 0
            rank._host_event_recorded = [False]
            rank._host_events = cast(Any, (FakeEvent(),))
            rank._metadata_hosts = (torch.ones((1, 1), dtype=torch.int32),)
            rank._metadata_device = torch.zeros((1, 1), dtype=torch.int32)
            rank._index_hosts = (torch.zeros(1, dtype=torch.int64),)
            rank._index_device = torch.zeros(1, dtype=torch.int64)
            rank._bound_active_rows = 0
            return rank, slot, current

        rows = (
            serving._BatchRow("request:phase", 1, 0, 0, 1, 0, 1, (1,), segment_length=2),
            serving._BatchRow("request:phase", 2, 1, 0, 1, 1, 1, (1,), segment_length=0),
        )
        rank, slot, current = make_rank()
        with (
            patch.object(rank, "_stage_batch_rows", return_value=(3, 0, 0)),
            patch.object(rank, "_bind_staged_metadata") as bind_metadata,
            patch("torch.cuda.current_stream", return_value=current),
        ):
            rank.bind_batch(rows)
            self.assertEqual(slot.replays, 0)
            self.assertEqual(rank.body.buffers.row_activity.tolist(), [1, 1, 0])
            bind_metadata.assert_called_once()

            rank.replay_bound()
            self.assertEqual(slot.replays, 1)
            self.assertEqual(slot.stream.waits, [current])
            self.assertEqual(current.waits, [slot.stream])

            with self.assertRaisesRegex(
                serving.Qwen3_5TP2PP2ServingError, "requires one successful"
            ):
                rank.replay_bound()
            self.assertEqual(slot.replays, 1)
            with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, "poisoned"):
                rank.bind_batch(rows)

        rank, slot, current = make_rank()
        with (
            patch.object(rank, "_stage_batch_rows", return_value=(3, 0, 0)),
            patch.object(rank, "_bind_staged_metadata"),
            patch("torch.cuda.current_stream", return_value=current),
        ):
            rank.bind_batch(rows)
            with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, "must replay"):
                rank.bind_batch(rows)
            with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, "poisoned"):
                rank.replay_bound()
            self.assertEqual(slot.replays, 0)

        rank = serving._CapturedGraphRank.__new__(serving._CapturedGraphRank)
        order: list[str] = []
        with (
            patch.object(rank, "bind_batch", side_effect=lambda _: order.append("bind")),
            patch.object(rank, "replay_bound", side_effect=lambda: order.append("replay")),
        ):
            rank.replay_batch(rows)
        self.assertEqual(order, ["bind", "replay"])

        rank = serving._CapturedGraphRank.__new__(serving._CapturedGraphRank)
        rank._target_hidden_slots = torch.zeros((4, 2), dtype=torch.bfloat16)
        rank._state_destination_column = 0
        rank._metadata_device = torch.tensor([[1], [3]], dtype=torch.int32)
        rank._state_destinations = torch.empty(2, dtype=torch.int64)
        rank.body = cast(
            Any,
            SimpleNamespace(
                buffers=SimpleNamespace(
                    norm=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
                )
            ),
        )
        unique = (
            serving._BatchRow("request:a", 1, 0, 0, 1, 0, 1, (1,)),
            serving._BatchRow("request:b", 1, 0, 0, 3, 0, 1, (1,)),
        )
        with (
            patch.object(rank, "bind_batch"),
            patch.object(rank, "replay_bound"),
        ):
            rank.replay_batch(unique)
        self.assertEqual(rank._target_hidden_slots[1].tolist(), [1.0, 2.0])
        self.assertEqual(rank._target_hidden_slots[3].tolist(), [3.0, 4.0])

        rank._target_hidden_slots.zero_()
        duplicate = (unique[0], replace(unique[1], state_destination_slot=1))
        with (
            patch.object(rank, "bind_batch"),
            patch.object(rank, "replay_bound"),
        ):
            rank.replay_batch(duplicate)
        self.assertEqual(rank._target_hidden_slots[1].tolist(), [3.0, 4.0])

        rank, slot, current = make_rank()
        with (
            patch.object(rank, "_stage_batch_rows", return_value=(1, 0, 0)),
            patch.object(rank, "_bind_staged_metadata"),
            patch("torch.cuda.current_stream", return_value=current),
        ):
            rank.bind_batch(rows)
            rank.complete_bound(replay=False)
            self.assertEqual(slot.replays, 0)
            rank.bind_batch(rows)
            rank.replay_bound()
            self.assertEqual(slot.replays, 1)

        # A tiny live write-set must not transfer the maximum-context tail.
        rank, _, current = make_rank()
        rank._index_hosts = (torch.full((16,), 99, dtype=torch.int64),)
        rank._index_hosts[0][0] = 2
        rank._index_hosts[0][4:6] = torch.tensor([1, 3])
        rank._index_device = torch.full((16,), -7, dtype=torch.int64)
        rank._index_page_offset = 4
        rank._target_hidden_slots = None
        arena = rank.body.arena
        arena.states["state"] = torch.ones((4, 2), dtype=torch.bfloat16)
        arena.key = torch.ones((4, 2), dtype=torch.bfloat16)
        arena.value = torch.ones_like(arena.key)
        arena.written = torch.ones(4, dtype=torch.bool)
        with (
            patch.object(rank, "_stage_batch_rows", return_value=(3, 1, 2)),
            patch.object(rank, "_bind_staged_metadata"),
            patch("torch.cuda.current_stream", return_value=current),
        ):
            rank.bind_batch(rows)
        expected = torch.full((16,), -7, dtype=torch.int64)
        expected[0], expected[4], expected[5] = 2, 1, 3
        self.assertTrue(torch.equal(rank._index_device, expected))
        self.assertEqual(arena.states["state"].sum(dim=1).tolist(), [2, 2, 0, 2])
        self.assertEqual(arena.key.sum(dim=1).tolist(), [2, 0, 2, 0])
        self.assertEqual(arena.written.tolist(), [True, False, True, False])

    def test_failed_batch_bind_cannot_replay_or_recover(self) -> None:
        rank = serving._CapturedGraphRank.__new__(serving._CapturedGraphRank)
        rank._feeding_profiler = None
        rank._batch_bound = False
        rank._batch_phase_failed = False
        rank._next_host_slot = 0
        rank._host_event_recorded = [False]
        rank._host_events = ()
        slot = SimpleNamespace(replays=0)
        rank.slot = cast(Any, slot)
        rows = (serving._BatchRow("request:failed", 1, 0, 0, 1, 0, 1, (1,)),)

        with (
            patch.object(rank, "_stage_batch_rows", side_effect=ValueError("invalid bind")),
            self.assertRaisesRegex(ValueError, "invalid bind"),
        ):
            rank.bind_batch(rows)
        with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, "poisoned"):
            rank.replay_bound()
        with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, "poisoned"):
            rank.bind_batch(rows)

    def test_wave_eager_oracle_completes_lane_a_before_lane_b(self) -> None:
        events: list[str] = []

        class FakeBody:
            hidden = torch.empty(1)
            logits = None

            def __init__(self, lane: str) -> None:
                self.lane = lane

            def bind_scratch(self, resource: object) -> None:
                del resource

            def launch(self) -> None:
                events.append(f"launch:{self.lane}")

        lane_a = SimpleNamespace(body=FakeBody("a"), resource=object())
        lane_b = SimpleNamespace(body=FakeBody("b"), resource=object())
        executor = SimpleNamespace(
            body=lane_a.body,
            slot=SimpleNamespace(logits=None),
            _lane_a=lane_a,
            _lane_b=lane_b,
            micro_m=4,
            captured_m=8,
            _split_rows=lambda values: (values[:3], values[3:]),
        )
        rows = tuple(
            serving._BatchRow(f"audit:{index}", 1, 0, 0, 1, 0, 1, (1,)) for index in range(5)
        )
        current = SimpleNamespace(synchronize=lambda: events.append("bind-sync"))
        with (
            patch.object(serving, "_reset_wave_audit_executor"),
            patch.object(serving, "_reset_wave_audit_owners"),
            patch.object(serving, "_wave_audit_rows", return_value=rows),
            patch.object(serving, "_dirty_production_audit_destinations"),
            patch.object(serving, "_poison_wave_audit_scratch"),
            patch.object(serving, "_padding_is_bound_to_scratch", return_value=True),
            patch.object(serving, "_poison_wave_audit_padding_inputs"),
            patch.object(serving, "_direct_bind_production_audit_rows"),
            patch.object(serving, "_bind_direct_padding_to_unused_scratch"),
            patch.object(serving, "_wave_active_snapshot", return_value={}),
            patch.object(serving, "_wave_padding_snapshot", return_value={}),
            patch.object(serving, "_wave_sampling_layout_is_exact", return_value=True),
            patch("torch.cuda.current_stream", return_value=current),
            patch(
                "torch.cuda.synchronize",
                side_effect=lambda _device: events.append("cuda-sync"),
            ),
            patch(
                "cubic_serve.runtime.performance.barrier",
                side_effect=lambda: events.append("barrier"),
            ),
        ):
            serving._run_wave_audit_once(cast(Any, executor), 5, graph=False, poison=0)

        self.assertEqual(
            events,
            [
                "bind-sync",
                "barrier",
                "launch:a",
                "cuda-sync",
                "barrier",
                "launch:b",
                "cuda-sync",
                "barrier",
                "bind-sync",
            ],
        )

    def test_wave_audit_consensus_rejects_all_rank_local_failures(self) -> None:
        context = multiprocessing.get_context("spawn")
        statuses = context.Array("i", (0, 0, 1, 2, 1, 0, 1, 4), lock=True)
        barrier = context.Barrier(1)

        with self.assertRaisesRegex(
            serving.Qwen3_5TP2PP2ServingError,
            r"rank/code \(\(1, 2\), \(3, 4\)\)",
        ):
            serving._wave_audit_consensus(
                statuses,
                barrier,
                rank=0,
                phase_id=1,
                failure_code=0,
                timeout_seconds=1,
            )

    def test_wavefront_balances_two_binders_and_replays_once(self) -> None:
        class FakeStream:
            def __init__(self, events: list[tuple[object, ...]]) -> None:
                self.events = events

            def wait_stream(self, stream: object) -> None:
                self.events.append(("slot-wait", stream))

        class FakeSlot:
            def __init__(self, events: list[tuple[object, ...]]) -> None:
                self.stream = FakeStream(events)
                self.events = events
                self.replays = 0

            def replay(self) -> None:
                self.replays += 1
                self.events.append(("replay",))

        class FakeBinder:
            def __init__(self, lane: str, events: list[tuple[object, ...]]) -> None:
                self.lane = lane
                self.events = events
                self.bound: tuple[serving._BatchRow, ...] = ()
                self._feeding_profiler = None

            def bind_batch(self, rows: tuple[serving._BatchRow, ...]) -> None:
                self.bound = rows
                self.events.append(("bind", self.lane, len(rows)))

            def complete_bound(self, *, replay: bool) -> None:
                self.events.append(("complete", self.lane, replay))

            def _commit_target_hidden(self, rows: tuple[serving._BatchRow, ...]) -> None:
                self.events.append(("commit", self.lane, len(rows)))

        class FakeCurrent(FakeStream):
            pass

        def make_wave() -> tuple[
            serving._CapturedWavefrontRank,
            FakeBinder,
            FakeBinder,
            FakeSlot,
            FakeCurrent,
            list[tuple[object, ...]],
        ]:
            events: list[tuple[object, ...]] = []
            wave = serving._CapturedWavefrontRank.__new__(serving._CapturedWavefrontRank)
            lane_a = FakeBinder("a", events)
            lane_b = FakeBinder("b", events)
            slot = FakeSlot(events)
            current = FakeCurrent(events)
            wave._lane_a = cast(Any, lane_a)
            wave._lane_b = cast(Any, lane_b)
            wave.slot = cast(Any, slot)
            wave.body = cast(Any, SimpleNamespace(hidden=torch.empty(1)))
            wave.micro_m = 4
            wave.captured_m = 8
            wave._feeding_profiler = None
            wave._phase_failed = False
            wave._active_rows = None
            return wave, lane_a, lane_b, slot, current, events

        def rows(count: int) -> tuple[serving._BatchRow, ...]:
            return tuple(
                serving._BatchRow(f"request:wave:{index}", 1, 0, 0, 1, 0, 1, (1,))
                for index in range(count)
            )

        for logical_m, expected in ((5, (3, 2)), (6, (3, 3)), (8, (4, 4))):
            wave, lane_a, lane_b, slot, current, events = make_wave()
            with patch("torch.cuda.current_stream", return_value=current):
                wave.replay_batch(rows(logical_m))

            self.assertEqual((len(lane_a.bound), len(lane_b.bound)), expected)
            self.assertEqual(
                tuple(row.request_id for row in lane_a.bound + lane_b.bound),
                tuple(row.request_id for row in rows(logical_m)),
            )
            self.assertEqual(slot.replays, 1)
            self.assertIn(("commit", "a", expected[0]), events)
            self.assertIn(("commit", "b", expected[1]), events)
            self.assertEqual(
                events[:5],
                [
                    ("bind", "a", expected[0]),
                    ("bind", "b", expected[1]),
                    ("complete", "a", False),
                    ("complete", "b", False),
                    ("slot-wait", current),
                ],
            )
            self.assertEqual(wave._active_rows, logical_m)

    def test_wavefront_binding_and_replay_fail_closed(self) -> None:
        class FakeStream:
            def wait_stream(self, stream: object) -> None:
                del stream

        class FakeSlot:
            def __init__(self, *, fail: bool = False) -> None:
                self.stream = FakeStream()
                self.fail = fail
                self.replays = 0

            def replay(self) -> None:
                self.replays += 1
                if self.fail:
                    raise RuntimeError("wave replay failed")

        class FakeBinder:
            def __init__(self, *, fail: bool = False) -> None:
                self.fail = fail
                self.completions: list[bool] = []
                self._feeding_profiler = None

            def bind_batch(self, rows: tuple[serving._BatchRow, ...]) -> None:
                del rows
                if self.fail:
                    raise RuntimeError("lane bind failed")

            def complete_bound(self, *, replay: bool) -> None:
                self.completions.append(replay)

        def make_wave(
            *, lane_b_fail: bool = False, replay_fail: bool = False
        ) -> tuple[serving._CapturedWavefrontRank, FakeBinder, FakeBinder, FakeSlot]:
            wave = serving._CapturedWavefrontRank.__new__(serving._CapturedWavefrontRank)
            lane_a = FakeBinder()
            lane_b = FakeBinder(fail=lane_b_fail)
            slot = FakeSlot(fail=replay_fail)
            wave._lane_a = cast(Any, lane_a)
            wave._lane_b = cast(Any, lane_b)
            wave.slot = cast(Any, slot)
            wave.body = cast(Any, SimpleNamespace(hidden=torch.empty(1)))
            wave.micro_m = 4
            wave.captured_m = 8
            wave._feeding_profiler = None
            wave._phase_failed = False
            wave._active_rows = None
            return wave, lane_a, lane_b, slot

        rows = tuple(
            serving._BatchRow(f"request:failure:{index}", 1, 0, 0, 1, 0, 1, (1,))
            for index in range(5)
        )
        wave, lane_a, _, slot = make_wave(lane_b_fail=True)
        with self.assertRaisesRegex(RuntimeError, "lane bind failed"):
            wave.replay_batch(rows)
        self.assertEqual(lane_a.completions, [False])
        self.assertEqual(slot.replays, 0)
        with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, "poisoned"):
            wave.replay_batch(rows)

        wave, lane_a, lane_b, slot = make_wave(replay_fail=True)
        with (
            patch("torch.cuda.current_stream", return_value=FakeStream()),
            self.assertRaisesRegex(RuntimeError, "wave replay failed"),
        ):
            wave.replay_batch(rows)
        self.assertEqual(lane_a.completions, [False])
        self.assertEqual(lane_b.completions, [False])
        self.assertEqual(slot.replays, 1)
        with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, "poisoned"):
            wave.replay_batch(rows)

        wave, lane_a, lane_b, slot = make_wave()
        duplicate = (*rows[:-1], rows[0])
        with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, "request-disjoint"):
            wave.replay_batch(duplicate)
        self.assertEqual((lane_a.completions, lane_b.completions, slot.replays), ([], [], 0))

        for invalid_count in (4, 9):
            wave, lane_a, lane_b, slot = make_wave()
            invalid = tuple(
                serving._BatchRow(f"request:range:{index}", 1, 0, 0, 1, 0, 1, (1,))
                for index in range(invalid_count)
            )
            with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, "above M"):
                wave.replay_batch(invalid)
            self.assertEqual((lane_a.completions, lane_b.completions, slot.replays), ([], [], 0))

    def test_wavefront_sampling_uses_combined_logits_and_workspace(self) -> None:
        wave = serving._CapturedWavefrontRank.__new__(serving._CapturedWavefrontRank)
        combined_logits = torch.arange(24, dtype=torch.bfloat16).view(8, 3)
        wave.slot = cast(
            Any,
            SimpleNamespace(
                logits=combined_logits,
                vocab_span=SimpleNamespace(start=0, stop=3),
            ),
        )
        wave.body = cast(Any, SimpleNamespace(hidden=torch.empty(1)))
        wave.micro_m = 4
        wave.captured_m = 8
        wave.speculative = False
        wave.vocab_size = 3
        wave._rank = 2
        wave._sampler_rank = 2
        wave._final_ranks = (2,)
        wave._world_group = cast(Any, object())
        wave._final_spans = cast(Any, (SimpleNamespace(start=0, stop=3),))
        wave._final_tp_group = cast(Any, object())
        wave._token_ids = torch.arange(8, dtype=torch.int64)
        wave._local_max_scores = torch.empty(8, dtype=torch.bfloat16)
        wave._gathered_max_scores = torch.empty((1, 8), dtype=torch.bfloat16)
        wave._gathered_token_ids = torch.empty((1, 8), dtype=torch.int64)
        wave._logical_logits = torch.empty_like(combined_logits)
        wave._feeding_profiler = None
        wave._phase_failed = False
        for active_rows in (5, 6, 7, 8):
            wave._active_rows = active_rows
            wave._lane_a_rows = (active_rows + 1) // 2
            with patch.object(
                serving,
                "gather_greedy_broadcast_tp",
                return_value=wave._token_ids,
            ) as gather:
                self.assertEqual(wave.sample_tokens(active_rows), tuple(range(active_rows)))

            kwargs = gather.call_args.kwargs
            local_logits = cast(torch.Tensor, kwargs["local_logits"])
            first = (active_rows + 1) // 2
            second = active_rows - first
            expected = torch.cat((combined_logits[:first], combined_logits[4 : 4 + second]))
            self.assertTrue(torch.equal(local_logits[:active_rows], expected))
            self.assertIs(
                local_logits,
                combined_logits if active_rows == 8 else wave._logical_logits,
            )
            self.assertEqual(kwargs["active_rows"], active_rows)
            self.assertEqual(tuple(kwargs["token_ids"].shape), (8,))
            self.assertIsNone(wave._active_rows)

    def test_wave_audit_resets_only_current_owners_and_padding_scratch(self) -> None:
        wave = serving._CapturedWavefrontRank.__new__(serving._CapturedWavefrontRank)
        wave.micro_m = 4
        wave.captured_m = 8
        wave.speculative = False
        padding_buffers = SimpleNamespace(
            page_destination=torch.tensor([20, 21, 22, 23], dtype=torch.int32),
            conv_slots=torch.tensor([[[30, 31, 32, 33]], [[30, 31, 32, 33]]], dtype=torch.int32),
        )
        lane_a = SimpleNamespace(
            _scratch_pages=(20, 21, 22, 23),
            _scratch_states=(30, 31, 32, 33),
            captured_m=4,
            body=SimpleNamespace(buffers=padding_buffers),
        )
        lane_b = SimpleNamespace(
            captured_m=4,
            body=SimpleNamespace(buffers=padding_buffers),
        )
        wave._lane_a = cast(Any, lane_a)
        wave._lane_b = cast(Any, lane_b)
        arena = SimpleNamespace(
            spec=SimpleNamespace(
                page_capacity=20,
                state_slot_capacity=30,
                scratch_capacity=1,state_scratch_capacity=None,state_scratch_capacity_per_arena=1,
                scratch_arena_ids=("test",),
            ),
            key=torch.full((24, 1), 7.0),
            value=torch.full((24, 1), 7.0),
            written=torch.ones((24, 1), dtype=torch.bool),
            states={
                CONV_STATE_ID: torch.full((34, 1), 7.0),
                RECURRENT_STATE_ID: torch.full((34, 1), 7.0),
            },
        )
        wave.body = cast(Any, SimpleNamespace(hidden=torch.empty(1), arena=arena))

        pages, states = serving._wave_audit_scratch_indices(wave, 5)
        self.assertEqual((pages.tolist(), states.tolist()), ([22, 23], [32, 33]))
        pages, states = serving._wave_audit_scratch_indices(wave, 6)
        self.assertEqual((pages.tolist(), states.tolist()), ([23], [33]))
        pages, states = serving._wave_audit_scratch_indices(wave, 8)
        self.assertEqual((pages.tolist(), states.tolist()), ([], []))

        serving._reset_wave_audit_owners(wave, 5)

        self.assertTrue(torch.equal(arena.key[:10], torch.zeros_like(arena.key[:10])))
        self.assertTrue(torch.equal(arena.key[10:22], torch.full_like(arena.key[10:22], 7)))
        self.assertTrue(torch.equal(arena.key[22:], torch.zeros_like(arena.key[22:])))
        conv = arena.states[CONV_STATE_ID]
        self.assertTrue(torch.equal(conv[:10], torch.zeros_like(conv[:10])))
        self.assertTrue(torch.equal(conv[10:32], torch.full_like(conv[10:32], 7)))
        self.assertTrue(torch.equal(conv[32:], torch.zeros_like(conv[32:])))

        serving._poison_wave_audit_scratch(wave, (3, 2), poison=0)

        self.assertTrue(torch.equal(arena.key[20:22], torch.full_like(arena.key[20:22], 7)))
        self.assertTrue(torch.equal(arena.key[22:], torch.full_like(arena.key[22:], 37)))
        self.assertTrue(torch.equal(arena.value[22:], torch.full_like(arena.value[22:], -37)))
        self.assertTrue(torch.all(arena.written[22:]))
        self.assertTrue(torch.equal(conv[32:], torch.full_like(conv[32:], 37 / 4)))
        recurrent = arena.states[RECURRENT_STATE_ID]
        self.assertTrue(torch.equal(recurrent[32:], torch.full_like(recurrent[32:], 37 / 16)))

    def test_wave_audit_executor_rebinds_without_full_scratch_clear(self) -> None:
        events: list[str] = []

        def binder(label: str) -> SimpleNamespace:
            return SimpleNamespace(
                _bound_active_rows=4,
                _next_host_slot=1,
                _host_event_recorded=[True, True],
                _batch_bound=True,
                _batch_phase_failed=True,
                resource=object(),
                body=SimpleNamespace(
                    bind_scratch=lambda _resource: events.append(f"bind:{label}"),
                    buffers=SimpleNamespace(
                        segment_lengths=torch.ones(4, dtype=torch.int32),
                        row_activity=torch.ones(4, dtype=torch.int32),
                    ),
                ),
            )

        lane_a, lane_b = binder("a"), binder("b")
        wave = SimpleNamespace(
            slot=SimpleNamespace(synchronize=lambda: events.append("synchronize")),
            _lane_a=lane_a,
            _lane_b=lane_b,
            _phase_failed=True,
            _active_rows=5,
        )

        serving._reset_wave_audit_executor(cast(Any, wave))

        self.assertEqual(events, ["synchronize", "bind:a", "bind:b"])
        for item in (lane_a, lane_b):
            self.assertEqual(item._bound_active_rows, 0)
            self.assertEqual(item._next_host_slot, 0)
            self.assertEqual(item._host_event_recorded, [False, False])
            self.assertFalse(item._batch_bound)
            self.assertFalse(item._batch_phase_failed)
            self.assertFalse(bool(torch.count_nonzero(item.body.buffers.segment_lengths)))
        self.assertFalse(wave._phase_failed)
        self.assertIsNone(wave._active_rows)

    def test_audit_row_binding_preserves_migrating_source_and_reverses_cow(self) -> None:
        captured_m = 1
        buffers = SimpleNamespace(
            position_ids=torch.zeros((3, captured_m), dtype=torch.int64),
            page_source=torch.zeros(captured_m, dtype=torch.int32),
            page_destination=torch.zeros(captured_m, dtype=torch.int32),
            token_offsets=torch.zeros(captured_m, dtype=torch.int32),
            page_table=torch.zeros((captured_m, 2), dtype=torch.int32),
            sequence_lengths=torch.ones(captured_m, dtype=torch.int32),
            segment_lengths=torch.ones(captured_m, dtype=torch.int32),
            row_activity=torch.ones(captured_m, dtype=torch.int32),
            conv_slots=torch.zeros((2, 1, captured_m), dtype=torch.int32),
            recurrent_slots=torch.zeros((2, 1, captured_m), dtype=torch.int32),
        )
        buffers.state_slot_maps = (buffers.conv_slots, buffers.recurrent_slots)
        arena = SimpleNamespace(
            spec=SimpleNamespace(
                page_capacity=8,
                state_slot_capacity=8,
                scratch_capacity=1,state_scratch_capacity=None,state_scratch_capacity_per_arena=1,
                scratch_arena_ids=("test",),
            ),
            layout=SimpleNamespace(block_size_tokens=16),
            key=torch.zeros((8, 2), dtype=torch.bfloat16),
            value=torch.zeros((8, 2), dtype=torch.bfloat16),
            written=torch.zeros((8, 2), dtype=torch.bool),
            states={
                CONV_STATE_ID: torch.zeros((8, 2), dtype=torch.bfloat16),
                RECURRENT_STATE_ID: torch.zeros((8, 2), dtype=torch.float32),
            },
        )
        body = cast(
            Any,
            SimpleNamespace(
                buffers=buffers,
                captured_m=captured_m,
                arena=arena,
                hidden=torch.empty((captured_m, 1), dtype=torch.bfloat16),
                stage=1,
                set_active_rows=Mock(),
            ),
        )

        serving._bind_audit_rows(body, 1, position=0, parity=0, preserve_first=False)
        self.assertEqual(
            (buffers.page_source[0].item(), buffers.page_destination[0].item()),
            (0, 1),
        )
        arena.key[1].fill_(7)
        arena.states[RECURRENT_STATE_ID][1].fill_(9)

        serving._bind_audit_rows(body, 1, position=1, parity=1, preserve_first=True)
        self.assertEqual(
            (buffers.page_source[0].item(), buffers.page_destination[0].item()),
            (1, 0),
        )
        self.assertTrue(torch.equal(arena.key[1], torch.full_like(arena.key[1], 7)))
        self.assertTrue(
            torch.equal(
                arena.states[RECURRENT_STATE_ID][1],
                torch.full_like(arena.states[RECURRENT_STATE_ID][1], 9),
            )
        )

    def test_migration_audit_uses_each_bucket_dispatch_boundary(self) -> None:
        logical_ms = serving._migration_logical_ms()

        self.assertEqual(logical_ms, (1, 2, 3, 5, 9, 17, 33, 65))
        self.assertEqual(
            tuple(serving._bucket_for_logical_m(value) for value in logical_ms),
            serving._PHYSICAL_GRAPH_BUCKET_CAPACITIES,
        )

    def test_migration_chain_accepts_largest_first_capture_insertion_order(self) -> None:
        for order in ((1, 8, 4, 2), (8, 4, 2, 1)):
            with self.subTest(order=order):
                schedule = []
                executors = {m: SimpleNamespace(
                    body=SimpleNamespace(captured_m=m, hidden=torch.empty(1),
                                         buffers=SimpleNamespace(page_table=torch.empty(1, 2)),
                                         bind_scratch=Mock(), launch=Mock()),
                    resource=object(),
                ) for m in order}

                def rows(body, logical_m, schedule=schedule, **kwargs):
                    # Exercise the actual dispatch-domain obligation that the
                    # former dict-order traversal violated (M8 -> M4 at M9).
                    capacity = body.captured_m
                    self.assertTrue(0 < logical_m <= capacity)
                    self.assertEqual(serving._bucket_for_logical_m(logical_m, (1, 2, 4, 8)),
                                     capacity)
                    schedule.append((capacity, logical_m))
                    return ()

                with (
                    patch.object(serving, "_reset_owner"),
                    patch.object(serving, "_reset_production_audit_executor"),
                    patch.object(serving, "_production_audit_rows", side_effect=rows),
                    patch.object(serving, "_dirty_production_audit_destinations"),
                    patch.object(serving, "_poison_production_audit_scratch"),
                    patch.object(serving, "_direct_bind_production_audit_rows"),
                    patch.object(serving, "_tracked_snapshot", return_value={}),
                    patch.object(serving.torch.cuda, "current_stream",
                                 return_value=SimpleNamespace(synchronize=lambda: None)),
                    patch.object(serving.torch.cuda, "synchronize"),
                    patch.object(serving, "barrier"),
                ):
                    result, _ = serving._run_bucket_migration_chain(
                        executors, graph=False, poison=0, snapshot_capacity=None,
                    )
                self.assertEqual(schedule, [(1, 1), (2, 2), (4, 3), (8, 5)])
                self.assertEqual(tuple(result), (1, 2, 4, 8))

    def test_partial_migration_cache_hit_reruns_audit_on_every_rank(self) -> None:
        executors = {1: SimpleNamespace(body=SimpleNamespace(rank=0, hidden=torch.empty(1)))}

        def expose_peer_miss(value: torch.Tensor, *, op: object) -> None:
            self.assertEqual(op, serving.dist.ReduceOp.MIN)
            value.zero_()

        with (
            patch.object(serving, "_migration_audit_cache_key", return_value="b2s256:key"),
            patch.object(serving, "_read_migration_audit_receipt", return_value="cached"),
            patch.object(serving.dist, "all_reduce", side_effect=expose_peer_miss),
            patch.object(serving, "_run_bucket_migration_audit", return_value="fresh") as run_audit,
            patch.object(serving, "_write_migration_audit_receipt") as write_receipt,
        ):
            evidence = serving._cold_start_bucket_migration_audit(
                cast(Any, executors), cast(Any, object())
            )

        self.assertEqual(evidence, "fresh")
        run_audit.assert_called_once()
        write_receipt.assert_called_once_with("b2s256:key", 0, "fresh")

    def test_migration_plan_binds_launches_and_ordered_overlap_lanes(self) -> None:
        def body(m: int, signature: str = "tile32") -> SimpleNamespace:
            return SimpleNamespace(
                captured_m=m,
                _linear_down=SimpleNamespace(launch_signature=signature),
            )

        primary, left, right = body(8), body(2), body(6)
        slot = SimpleNamespace(_overlap=(object(), object(), (left, right)))
        executors = cast(Any, {8: SimpleNamespace(body=primary, slot=slot)})
        original = serving._migration_execution_plan(executors)
        # Recreated graph/event handles must not prevent legitimate cache reuse.
        slot._overlap = (object(), object(), (body(2), body(6)))
        self.assertEqual(original, serving._migration_execution_plan(executors))
        slot._overlap = (object(), object(), (right, left))
        self.assertNotEqual(original, serving._migration_execution_plan(executors))
        slot._overlap = (object(), object(), (left, right))
        right._linear_down.launch_signature = "tile64"
        self.assertNotEqual(original, serving._migration_execution_plan(executors))
        right._linear_down.launch_signature = "tile32"
        slot._overlap = None
        self.assertNotEqual(original, serving._migration_execution_plan(executors))
        # Packed stages expose the same ordered ownership through their own ABI.
        slot._overlap_resources = (object(), (left, right), object())
        self.assertEqual(original, serving._migration_execution_plan(executors))

    def test_migration_plan_binds_expert_and_nested_stateful_launches(self) -> None:
        expert = SimpleNamespace(launch_signature="expert-up16-down64")
        gdn = SimpleNamespace(launch_signature="gdn-value64")
        body = SimpleNamespace(captured_m=32, _routed_experts=expert,
                               _body=SimpleNamespace(gdn=gdn))
        executors = cast(Any, {32: SimpleNamespace(body=body, slot=SimpleNamespace())})
        original = serving._migration_execution_plan(executors)
        expert.launch_signature = "expert-up32-down128"
        self.assertNotEqual(original, serving._migration_execution_plan(executors))
        expert.launch_signature = "expert-up16-down64"
        gdn.launch_signature = "gdn-value16"
        self.assertNotEqual(original, serving._migration_execution_plan(executors))
        gdn.launch_signature = "gdn-value64"
        self.assertEqual(original, serving._migration_execution_plan(executors))

    def test_migration_plan_binds_transport_callback_owner(self) -> None:
        class Transport:
            launch_signature = "gather"

            def launch(self) -> None:
                pass

        transport = Transport()
        inner = SimpleNamespace(_tp_reduce=transport.launch)
        body = SimpleNamespace(captured_m=16, _body=inner)
        executors = cast(Any, {16: SimpleNamespace(body=body, slot=SimpleNamespace())})
        original = serving._migration_execution_plan(executors)
        transport.launch_signature = "partition"
        self.assertNotEqual(original, serving._migration_execution_plan(executors))
        # Recreated process-group/primitive instances with the same selected
        # plan can reuse a receipt; timing samples and object IDs cannot.
        inner._tp_reduce = Transport().launch
        self.assertEqual(original, serving._migration_execution_plan(executors))

    def test_migration_plan_cache_changes_when_only_peer_plan_changes(self) -> None:
        executors = cast(Any, {1: SimpleNamespace(
            body=SimpleNamespace(captured_m=1), slot=SimpleNamespace(),
        )})
        peer: object = ("peer-plan-a",)

        def gather(plans: list[object], local: object) -> None:
            plans[:] = [local, peer]

        with (
            patch.object(serving.dist, "get_world_size", return_value=2),
            patch.object(serving.dist, "all_gather_object", side_effect=gather),
        ):
            before = serving._migration_topology_plan_digest(executors)
            self.assertEqual(before, serving._migration_topology_plan_digest(executors))
            peer = ("peer-plan-b",)
            self.assertNotEqual(before, serving._migration_topology_plan_digest(executors))

    def test_production_audit_rows_cover_cow_growth_and_bulk_resets(self) -> None:
        body = cast(
            Any,
            SimpleNamespace(
                captured_m=8,
                arena=SimpleNamespace(spec=SimpleNamespace(state_slot_capacity=16)),
                buffers=SimpleNamespace(page_table=torch.empty((8, 3))),
            ),
        )

        rows = serving._production_audit_rows(
            body,
            5,
            position=2,
            parity=1,
            initialize_from=3,
        )

        self.assertEqual(
            tuple(row.request_id for row in rows),
            tuple(f"audit:row:{physical}" for physical in range(5)),
        )
        self.assertEqual(
            tuple((row.state_source_slot, row.state_destination_slot) for row in rows),
            ((1, 0), (3, 2), (5, 4), (7, 6), (9, 8)),
        )
        self.assertTrue(all(row.page_table == (row.page_destination,) * 3 for row in rows))
        self.assertEqual(
            tuple(row.reset_state for row in rows),
            (False, False, False, True, True),
        )
        self.assertEqual(tuple(row.new_pages for row in rows), ((), (), (), (6,), (8,)))
        with self.assertRaisesRegex(ValueError, "outside"):
            serving._production_audit_rows(body, 4, position=0, parity=0, initialize_from=0)

    def test_batch_selection_is_decode_first_with_bounded_physical_m(self) -> None:
        prompt = serving._GenerateCommand("prefill", (10, 11), 2, ())
        decode = serving._GenerateCommand("decode", (12,), 2, ())
        controller = self._batch_controller(prompt, decode)
        decode_record = controller._requests["decode"]
        decode_record.prefill_cursor = 1
        decode_record.generated_count = 1
        decode_record.last_token_id = 21

        rows, _ = controller._next_batch_locked()

        self.assertEqual(tuple(row.request_id for row in rows), ("decode", "prefill", "prefill"))
        self.assertEqual(rows[0].token_id, 21)
        self.assertEqual(rows[0].position, 1)
        self.assertLessEqual(len(rows), serving._MAX_PHYSICAL_CAPTURED_M)

    def test_full_decode_load_reserves_a_bounded_prefill_lane(self) -> None:
        commands = (
            *(
                serving._GenerateCommand(f"decode:{index}", (10,), 4, ())
                for index in range(serving._MAX_CAPTURED_M)
            ),
            serving._GenerateCommand("prefill", (11, 12), 4, ()),
        )
        controller = self._batch_controller(*commands)
        controller._max_running_requests = 256
        controller._free_state_slots = list(range(512))
        controller._free_pages = list(range(1024))
        heapq.heapify(controller._free_state_slots)
        heapq.heapify(controller._free_pages)
        for command in commands[:-1]:
            record = controller._requests[command.request_id]
            record.prefill_cursor = 1
            record.generated_count = 1
            record.last_token_id = 21

        rows, _ = controller._next_batch_locked()

        self.assertEqual(len(rows), serving._MAX_PHYSICAL_CAPTURED_M)
        self.assertIn("prefill", {row.request_id for row in rows})

    def test_mixed_prefill_backlog_scales_without_starving_decode(self) -> None:
        decode_commands = tuple(
            serving._GenerateCommand(f"decode:{index}", (10,), 4, ())
            for index in range(serving._MAX_PHYSICAL_CAPTURED_M)
        )
        prefill_commands = tuple(
            serving._GenerateCommand(f"prefill:{index}", (11, 12), 4, ())
            for index in range(serving._MAX_PHYSICAL_CAPTURED_M // 2)
        )
        controller = self._batch_controller(*decode_commands, *prefill_commands)
        controller._max_running_requests = 512
        controller._free_state_slots = list(range(1024))
        controller._free_pages = list(range(2048))
        heapq.heapify(controller._free_state_slots)
        heapq.heapify(controller._free_pages)
        for command in decode_commands:
            record = controller._requests[command.request_id]
            record.prefill_cursor = 1
            record.generated_count = 1
            record.last_token_id = 21

        rows, _ = controller._next_batch_locked()
        decode_rows = sum(row.is_decode for row in rows)

        self.assertEqual(len(rows), serving._MAX_PHYSICAL_CAPTURED_M)
        self.assertEqual(decode_rows, serving._MAX_PHYSICAL_CAPTURED_M // 2)
        self.assertEqual(len(rows) - decode_rows, serving._MAX_PHYSICAL_CAPTURED_M // 2)

    def test_continuing_prefill_rotates_capacity_excluded_decodes(self) -> None:
        for capacity in (8, 128, 256):
            for prefill_count in (1, 3):
                with self.subTest(capacity=capacity, prefill_count=prefill_count):
                    decodes = tuple(
                        serving._GenerateCommand(f"decode:{i}", (10,), 1024, ())
                        for i in range(capacity - 1)
                    )
                    prefills = tuple(
                        serving._GenerateCommand(f"prefill:{i}", (11,) * 2048, 16, ())
                        for i in range(prefill_count)
                    )
                    controller = self._batch_controller(*decodes, *prefills)
                    controller._max_running_requests = capacity + prefill_count
                    anchors = tuple(1 << i for i in range(capacity.bit_length()))
                    controller._physical_graph_capacities = anchors
                    controller._graph_capacities = anchors
                    for command in decodes:
                        record = controller._requests[command.request_id]
                        record.prefill_cursor = record.generated_count = 1
                        record.last_token_id = 21

                    # Isolate scheduling from the separately tested cache
                    # allocator. Keep the prompt pending across both steps.
                    def reserve(record, length, **kwargs):
                        return tuple(SimpleNamespace(request_id=record.command.request_id)
                                     for _ in range(length))

                    seen = set()
                    with patch.object(controller, "_reserve_segment_locked", side_effect=reserve):
                        for _ in range(2):
                            rows, selected = controller._next_batch_locked()
                            self.assertLessEqual(len(rows), capacity)
                            ids = {record.command.request_id for record in selected}
                            self.assertTrue({command.request_id for command in prefills} <= ids)
                            seen.update(ids)
                    self.assertTrue({command.request_id for command in decodes} <= seen)

    def test_concurrency_above_graph_capacity_stays_microbatched(self) -> None:
        for request_count in (129, 192, 223, 224, 256):
            commands = tuple(
                serving._GenerateCommand(f"decode:{index}", (10,), 2, ())
                for index in range(request_count)
            )
            controller = self._batch_controller(*commands)
            controller._max_running_requests = 256
            controller._free_state_slots = list(range(512))
            controller._free_pages = list(range(1024))
            heapq.heapify(controller._free_state_slots)
            heapq.heapify(controller._free_pages)
            for command in commands:
                record = controller._requests[command.request_id]
                record.prefill_cursor = 1
                record.generated_count = 1
                record.last_token_id = 21

            rows, _ = controller._next_batch_locked()

            self.assertEqual(len(rows), serving._MAX_PHYSICAL_CAPTURED_M)

    def test_single_token_prefill_fits_one_private_page(self) -> None:
        command = serving._GenerateCommand("request:capacity", (10,), 1, ())
        controller = self._batch_controller(command)
        controller._free_pages = [7]
        record = controller._requests[command.request_id]

        rows, selected = controller._next_batch_locked()

        self.assertEqual(selected, (record,))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].page_source, 7)
        self.assertEqual(rows[0].page_destination, 7)
        self.assertEqual(rows[0].new_pages, (7,))
        self.assertFalse(rows[0].is_decode)
        self.assertTrue(rows[0].reset_state)
        self.assertEqual(controller._free_pages, [])

    def test_resident_prompt_program_consumes_inputs_without_publishing_midprompt_logits(self) -> None:
        commands = (
            serving._GenerateCommand("decode", (10, 11), 4, ()),
            serving._GenerateCommand("prompt", (20, 21, 22, 23, 24, 25), 4, (99,)),
        )
        controller = self._batch_controller(*commands)
        decoded, prompt = (controller._requests[c.request_id] for c in commands)
        for record in (decoded, prompt):
            record.prefill_cursor = 2
            record.state_slot = heapq.heappop(controller._free_state_slots)
            record.pages.append(heapq.heappop(controller._free_pages))
            record.state_initialized = True
        decoded.generated_count = 1
        decoded.last_token_id = 50
        controller._resident_steady = (decoded, prompt)
        controller._resident_steady_batch_id = 1
        controller._resident_steady_captured_m = 2
        controller._resident_input_program_capacities = (2,)
        controller._resident_step_seconds = 0.01
        controller._decode_window_seconds = None
        controller._batch_id = 2

        command, rows, selected = controller._next_dispatch_locked()

        self.assertIsInstance(command, serving._ContinueBatchCommand)
        assert isinstance(command, serving._ContinueBatchCommand)
        self.assertEqual(rows, ())
        self.assertEqual(selected, (decoded, prompt))
        self.assertEqual(command.max_steps, 3)
        self.assertEqual(command.prompt_remaining, (0, 4))
        self.assertEqual(command.prompt_tokens, ((), (22, 23, 24)))
        self.assertEqual(command.prompt_positions, ((), ((2, 2, 2), (3, 3, 3), (4, 4, 4))))
        self.assertEqual((decoded.generated_count, prompt.prefill_cursor), (1, 2))

        wakes = serving.TokenWakeBatch()
        for step in range(2):
            controller._commit_steady_records_locked(
                selected, (51 + step, 99), wakes, profile=None,
                allow_prefill=True, defer_cancelled_release=True,
            )
        wakes.flush()
        self.assertEqual(prompt.prefill_cursor, 4)
        self.assertEqual(prompt.generated_count, 0)
        self.assertIsNone(prompt.last_token_id)
        self.assertFalse(prompt.output_closed)
        self.assertTrue(cast(Any, prompt.output)._items.empty())
        self.assertEqual((next(cast(Iterator[int], decoded.output)),
                          next(cast(Iterator[int], decoded.output))), (51, 52))

    def test_stochastic_prompt_program_keeps_policy_and_output_boundaries(self) -> None:
        from cubic_serve.executor.resident_window import ResidentDecodeWindow

        for temperatures in ((0.9, 0.7), (0.0, 0.7)):
            with self.subTest(temperatures=temperatures):
                commands = (
                    serving._GenerateCommand("decode", (10, 11), 8, ()),
                    serving._GenerateCommand("prompt", (20, 21, 22, 23, 24, 25), 4, (99,)),
                )
                commands = tuple(replace(command, sampling=replace(
                    command.sampling, temperature=temperature, seed=ordinal + 17))
                    for ordinal, (command, temperature) in enumerate(
                        zip(commands, temperatures, strict=True)))
                controller = self._batch_controller(*commands)
                selected = tuple(controller._requests[c.request_id] for c in commands)
                decoded, prompt = selected
                for record in selected:
                    record.prefill_cursor = 2
                    record.state_slot = heapq.heappop(controller._free_state_slots)
                    record.pages.append(heapq.heappop(controller._free_pages))
                    record.state_initialized = True
                decoded.generated_count = 1
                decoded.last_token_id = 50
                controller._resident_steady = selected
                controller._resident_steady_batch_id = 1
                controller._resident_steady_captured_m = 2
                controller._resident_input_program_capacities = ()
                controller._resident_stochastic_input_program_capacities = (2,)
                controller._resident_step_seconds = 0.01
                controller._decode_window_seconds = None
                controller._batch_id = 2

                command, rows, owners = controller._next_dispatch_locked()

                self.assertIsInstance(command, serving._ContinueBatchCommand)
                assert isinstance(command, serving._ContinueBatchCommand)
                self.assertEqual((rows, owners), ((), selected))
                self.assertEqual(command.prompt_remaining, (0, 4))
                self.assertEqual(command.prompt_tokens, ((), (22, 23, 24, 25)))
                self.assertEqual(tuple(r.command.sampling for r in selected),
                                 tuple(c.sampling for c in commands))
                window = ResidentDecodeWindow.__new__(ResidentDecodeWindow)
                window.e = SimpleNamespace(_sampling=SimpleNamespace(all_greedy=False),
                                           _target_hidden_slots=None)
                window.qualified = True
                window.stochastic = True
                window.program_qualified = False
                window.eos_values = torch.empty(2)
                self.assertFalse(window.supports(command))
                window.program_qualified = True
                self.assertTrue(window.supports(command))

                for step in range(4):
                    wakes = serving.TokenWakeBatch()
                    controller._commit_steady_records_locked(
                        selected, (51 + step, 99 if step < 3 else 77), wakes,
                        profile=None, allow_prefill=True, defer_cancelled_release=step < 3,
                    )
                    wakes.flush()
                    self.assertEqual(prompt.generated_count, int(step == 3))
                    self.assertFalse(prompt.output_closed)
                self.assertEqual(prompt.prefill_cursor, 6)
                self.assertEqual(next(cast(Iterator[int], prompt.output)), 77)
                self.assertEqual(tuple(r.command.sampling for r in selected),
                                 tuple(c.sampling for c in commands))

    def test_input_program_advertisement_keeps_sampling_and_draft_roles_separate(self) -> None:
        from cubic_serve.executor.resident_window import resident_input_program_capacities

        def executor(greedy, stochastic, *, draft=False):
            return SimpleNamespace(
                _resident_window=SimpleNamespace(program_qualified=greedy),
                _resident_stochastic_window=SimpleNamespace(program_qualified=stochastic),
                _target_hidden_slots=object() if draft else None,
            )

        executors = {1: executor(True, False), 2: executor(False, True),
                     4: executor(True, True, draft=True)}
        self.assertEqual(resident_input_program_capacities(executors), (1,))
        self.assertEqual(resident_input_program_capacities(executors, stochastic=True), (2,))

    def test_resident_prompt_program_requires_all_rank_qualification_and_supported_inputs(self) -> None:
        for cause in ("qualification", "stochastic", "future_feature"):
            with self.subTest(cause=cause):
                kwargs = ({"feature_rows": (b"", b"", b"media", b"")}
                          if cause == "future_feature" else {})
                command = serving._GenerateCommand("prompt", (10, 11, 12, 13), 4, (), **kwargs)
                controller = self._batch_controller(command)
                record = controller._requests["prompt"]
                record.prefill_cursor = 1
                record.state_slot = heapq.heappop(controller._free_state_slots)
                record.pages.append(heapq.heappop(controller._free_pages))
                record.state_initialized = True
                if cause == "stochastic":
                    record.command = replace(command, sampling=replace(command.sampling, temperature=0.7))
                controller._resident_steady = (record,)
                controller._resident_steady_batch_id = 1
                controller._resident_steady_captured_m = 1
                controller._resident_input_program_capacities = () if cause == "qualification" else (1,)
                self.assertIsNone(controller._steady_continuation_page_updates_locked())

    def test_resident_prompt_stops_at_cache_snapshot_before_state_is_mutated_again(self) -> None:
        for temperature in (0.0, 0.7):
            with self.subTest(temperature=temperature):
                self._check_resident_prompt_cache_snapshot(temperature)

    def _check_resident_prompt_cache_snapshot(self, temperature: float) -> None:
        command = serving._GenerateCommand("cached-program", tuple(range(17)), 4, ())
        command = replace(command, sampling=replace(command.sampling, temperature=temperature))
        controller = self._batch_controller(command)
        record = controller._requests[command.request_id]
        record.prefill_cursor = 1
        record.state_slot = heapq.heappop(controller._free_state_slots)
        record.pages.append(heapq.heappop(controller._free_pages))
        record.state_initialized = True
        controller._prefix_cache_enabled = True
        controller._resident_steady = (record,)
        controller._resident_steady_batch_id = 1
        controller._resident_steady_captured_m = 1
        controller._resident_input_program_capacities = (1,)
        controller._resident_stochastic_input_program_capacities = (1,)
        controller._resident_step_seconds = 0.01
        controller._decode_window_seconds = None
        controller._batch_id = 2

        batch, _, selected = controller._next_dispatch_locked()

        assert isinstance(batch, serving._ContinueBatchCommand)
        self.assertEqual(batch.max_steps, 15)
        self.assertEqual(batch.prompt_remaining, (16,))
        for step in range(batch.max_steps):
            wakes = serving.TokenWakeBatch()
            controller._commit_steady_records_locked(
                selected, (77,), wakes, profile=None, allow_prefill=True,
                defer_cancelled_release=step + 1 < batch.max_steps,
            )
            wakes.flush()
            if step + 1 < batch.max_steps:
                self.assertFalse(controller._prefix_cache)
        self.assertEqual(record.prefill_cursor, 16)
        self.assertEqual(record.generated_count, 0)
        snapshot, = controller._prefix_cache.values()
        self.assertEqual(snapshot.token_ids, command.input_ids[:16])
        self.assertEqual(snapshot.state_slot, record.state_slot)
        self.assertIsNone(controller._steady_continuation_page_updates_locked())
        next_rows, _ = controller._next_batch_locked()
        self.assertNotEqual(next_rows[0].state_source_slot, next_rows[0].state_destination_slot)

    def test_evicted_last_cache_coowner_does_not_cause_an_empty_scheduler_turn(self) -> None:
        command = serving._GenerateCommand("sole-owner", tuple(range(17)), 4, ())
        controller = self._batch_controller(command)
        record = controller._requests[command.request_id]
        record.prefill_cursor = 16
        record.state_initialized = True
        record.state_slot = heapq.heappop(controller._free_state_slots)
        record.pages.append(heapq.heappop(controller._free_pages))
        controller._free_state_slots = []
        controller._prefix_cache_enabled = True
        controller._publish_prefix_locked(record)

        rows, selected = controller._next_batch_locked()

        self.assertEqual(selected, (record,))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].state_source_slot, record.state_slot)
        self.assertEqual(rows[0].state_destination_slot, record.state_slot)
        self.assertFalse(controller._prefix_cache)
        self.assertFalse(record.cancelled)

    def test_multi_token_cow_exhaustion_rejects_without_leaking_owner_page(self) -> None:
        command = serving._GenerateCommand("request:capacity", (10, 11), 1, ())
        controller = self._batch_controller(command)
        controller._free_pages = [7]
        record = controller._requests[command.request_id]

        rows, selected = controller._next_batch_locked()

        self.assertEqual((rows, selected), ((), ()))
        self.assertNotIn(command.request_id, controller._requests)
        self.assertEqual(controller._free_pages, [7])
        with self.assertRaises(serving.Qwen3_5TP2PP2Overloaded):
            next(cast(Iterator[int], record.output))

    def test_single_token_prefill_preserves_private_owner_and_media_metadata(self) -> None:
        command = serving._GenerateCommand(
            "request:append", (10, 11), 3, (),
            positions=((4, 8, 12), (5, 9, 13)),
            feature_rows=(b"first", b"second"),
        )
        controller = self._batch_controller(command)
        record = controller._requests[command.request_id]
        record.state_slot = heapq.heappop(controller._free_state_slots)
        record.pages.append(heapq.heappop(controller._free_pages))
        record.prefill_cursor = 1
        record.state_initialized = True
        before = (tuple(controller._free_state_slots), tuple(controller._free_pages))

        rows = controller._reserve_segment_locked(record, 1, running=1)

        assert rows is not None
        row, = rows
        self.assertEqual(row.token_id, 11)
        self.assertEqual(row.position, 1)
        self.assertEqual(row.position_axes, (5, 9, 13))
        self.assertEqual(row.input_feature, b"second")
        self.assertFalse(row.is_decode)
        self.assertFalse(row.reset_state)
        self.assertEqual(row.state_source_slot, record.state_slot)
        self.assertEqual(row.state_destination_slot, record.state_slot)
        self.assertEqual(row.page_source, record.pages[0])
        self.assertEqual(row.page_destination, record.pages[0])
        self.assertEqual(before, (tuple(controller._free_state_slots), tuple(controller._free_pages)))

    def test_single_token_prefill_does_not_overwrite_shared_prefix(self) -> None:
        command = serving._GenerateCommand("request:shared", (10, 11), 3, ())
        controller = self._batch_controller(command)
        record = controller._requests[command.request_id]
        record.state_slot = heapq.heappop(controller._free_state_slots)
        record.pages.append(heapq.heappop(controller._free_pages))
        record.prefill_cursor = 1
        record.state_initialized = True
        controller._shared_state_slots[record.state_slot] = 2
        controller._shared_pages[record.pages[0]] = 2

        rows = controller._reserve_segment_locked(record, 1, running=1)

        assert rows is not None
        row, = rows
        self.assertNotEqual(row.state_source_slot, row.state_destination_slot)
        self.assertNotEqual(row.page_source, row.page_destination)
        self.assertEqual(controller._shared_state_slots[record.state_slot], 2)
        self.assertEqual(controller._shared_pages[record.pages[0]], 2)

    def test_single_token_shared_prefill_capacity_failure_preserves_sources(self) -> None:
        command = serving._GenerateCommand("request:shared-full", (10, 11), 3, ())
        controller = self._batch_controller(command)
        record = controller._requests[command.request_id]
        record.state_slot = heapq.heappop(controller._free_state_slots)
        record.pages.append(heapq.heappop(controller._free_pages))
        record.prefill_cursor = 1
        record.state_initialized = True
        controller._shared_state_slots[record.state_slot] = 2
        controller._shared_pages[record.pages[0]] = 2
        controller._free_pages = []
        free_states = tuple(controller._free_state_slots)

        rows = controller._reserve_segment_locked(record, 1, running=1)

        self.assertIsNone(rows)
        self.assertTrue(record.cancelled)
        self.assertEqual(tuple(controller._free_state_slots), free_states)
        self.assertEqual(controller._shared_state_slots[record.state_slot], 2)
        self.assertEqual(controller._shared_pages[record.pages[0]], 2)
        self.assertEqual(controller._free_pages, [])

    def test_single_token_prefill_page_growth_reserves_only_new_page(self) -> None:
        for block_size in (16, 32, 48):
            with self.subTest(block_size=block_size):
                command = serving._GenerateCommand(
                    "request:grow", tuple(range(block_size + 1)), 3, ()
                )
                controller = self._batch_controller(command, block_size=block_size)
                record = controller._requests[command.request_id]
                record.state_slot = heapq.heappop(controller._free_state_slots)
                record.pages.append(heapq.heappop(controller._free_pages))
                record.prefill_cursor = block_size
                record.state_initialized = True
                controller._free_pages = [7]
                original_page = record.pages[0]

                rows = controller._reserve_segment_locked(record, 1, running=1)

                assert rows is not None
                row, = rows
                self.assertEqual(row.page_table, (original_page, 7))
                self.assertEqual(row.new_pages, (7,))
                self.assertEqual(row.page_source, 7)
                self.assertEqual(row.page_destination, 7)
                self.assertEqual(row.state_source_slot, row.state_destination_slot)
                self.assertEqual(controller._free_pages, [])

    def test_public_executor_validates_and_forwards_synchronous_request(self) -> None:
        controller = _FakeController()
        executor = serving.Qwen3_5TP2PP2PerformanceExecutor._for_testing(
            controller, eos_token_ids=(99,)
        )
        self.assertEqual(executor.max_sequence_tokens, controller.max_sequence_tokens)
        self.assertEqual(
            tuple(executor.generate((10, 11), 3, _Metadata("request:4"))),
            (20, 21),
        )
        self.assertEqual(
            controller.commands,
            [serving._GenerateCommand("request:4", (10, 11), 3, (99,))],
        )
        for invalid in ((), (128,), (True,)):
            with self.assertRaises(ValueError):
                executor.generate(invalid, 1)
        with self.assertRaisesRegex(ValueError, "capacity"):
            executor.generate((1,), 9)
        executor.close()
        executor.close()
        self.assertEqual(controller.close_count, 1)
        with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, "closed"):
            tuple(executor.generate((1,), 1))

    def test_tokens_are_visible_before_generation_finishes(self) -> None:
        controller = _GatedController()
        executor = serving.Qwen3_5TP2PP2PerformanceExecutor._for_testing(controller)
        stream = executor.generate((1,), 2, _Metadata("request:stream"))

        self.assertEqual(next(stream), 20)
        self.assertTrue(controller.second_started.wait(1))
        self.assertFalse(controller.release_second.is_set())
        controller.release_second.set()
        self.assertEqual(tuple(stream), (21,))
        self.assertTrue(executor.ready)

    def test_concurrent_requests_receive_independent_streams(self) -> None:
        controller = _GatedController()
        executor = serving.Qwen3_5TP2PP2PerformanceExecutor._for_testing(controller)
        first = executor.generate((1,), 2, _Metadata("request:first"))
        self.assertEqual(next(first), 20)
        self.assertTrue(controller.second_started.wait(1))

        second = executor.generate((1,), 2, _Metadata("request:second"))
        self.assertEqual(next(second), 20)

        controller.release_second.set()
        self.assertEqual(tuple(first), (21,))
        self.assertEqual(tuple(second), (21,))

    def test_replay_failure_poison_closes_topology(self) -> None:
        controller = _FailingController()
        executor = serving.Qwen3_5TP2PP2PerformanceExecutor._for_testing(controller)
        with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, "replay failed"):
            tuple(executor.generate((1,), 1))
        self.assertEqual(controller.close_count, 1)
        self.assertFalse(executor.ready)
        with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, "poisoned"):
            executor.generate((1,), 1)

    def test_cancel_stops_only_request_and_topology_remains_reusable(self) -> None:
        controller = _BlockingController()
        executor = serving.Qwen3_5TP2PP2PerformanceExecutor._for_testing(controller)
        errors: list[BaseException] = []

        def run() -> None:
            try:
                tuple(executor.generate((1,), 1, _Metadata("request:cancel")))
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=run)
        worker.start()
        self.assertTrue(controller.entered.wait(1))
        executor.cancel("request:cancel")
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(controller.close_count, 0)
        self.assertTrue(executor.ready)
        self.assertEqual(
            tuple(executor.generate((1,), 1, _Metadata("request:after-cancel"))),
            (22,),
        )

    def test_context_growth_charges_target_and_mtp_tables(self) -> None:
        capacities = (1, 2, 4)
        for mtp, per_page in ((False, 124), (True, 180)):
            self.assertEqual(
                serving._context_staging_growth_bytes(
                    capacities,
                    (),
                    1,
                    16384,
                    mtp=mtp,
                ),
                16383 * per_page,
            )
            self.assertEqual(
                serving._context_staging_growth_bytes(
                    capacities,
                    (),
                    16384,
                    16384,
                    mtp=mtp,
                ),
                0,
            )

    def test_context_growth_includes_both_wavefront_rank_binders(self) -> None:
        lanes = (SimpleNamespace(captured_m=4), SimpleNamespace(captured_m=4))
        self.assertEqual(
            serving._context_staging_growth_bytes(
                (1, 2, 4),
                cast(Any, lanes),
                1,
                2,
                mtp=False,
            ),
            124 + 2 * (4 * 16 + 4),
        )

    def test_cache_sizing_profiles_staging_for_every_graph_bucket(self) -> None:
        bodies = {m: object() for m in (1, 2, 4)}
        runtime, admission = object(), SimpleNamespace(
            num_speculative_tokens=0, max_running_requests=4, dynamic_max_speculative_tokens=0,
            parallel_plan=SimpleNamespace(topology=SimpleNamespace(pp_size=1)),
        )
        # Include a non-linear final bucket cost: extrapolating the first two
        # captures must not miss later executor allocations.
        samples = [(1000, 2000), (870, 2000), (870, 2000), (760, 2000), (760, 2000), (560, 2000)]
        sampler_close = Mock()
        with (
            patch.object(serving.torch.cuda, "synchronize"),
            patch.object(serving.torch.cuda, "empty_cache"),
            patch.object(serving.torch.cuda, "reset_peak_memory_stats"),
            patch.object(serving.torch.cuda, "max_memory_allocated", return_value=600),
            patch.object(serving.torch.cuda, "memory_allocated", return_value=560),
            patch.object(serving.torch.cuda, "graph_pool_handle"),
            patch.object(serving.torch.cuda, "Stream"),
            patch.object(serving.torch.cuda, "mem_get_info", side_effect=samples),
            patch.object(serving.Qwen3_5TP2PP2GraphSlot, "capture") as capture,
            patch.object(serving, "_CapturedGraphRank") as executor,
            patch.object(
                serving, "_prepare_rank_greedy_sampling",
                side_effect=lambda executor, lifetime, *, qualify_resident, constraint_arena: lifetime.callback(
                    sampler_close),
            ) as prepare_sampling,
            patch.object(serving, "_ConstraintSamplingArena") as scratch_arena,
        ):
            result = serving._profile_cuda_graph_memory(
                bodies,
                (4, 2, 1),
                rank=0,
                device=torch.device("cuda:0"),
                runtime=runtime,
                admission=admission,
                weights=object(),
            )
        self.assertEqual(result, (440, 40, {}))
        self.assertEqual(capture.call_count, 3)
        self.assertEqual(
            [c.args[0] for c in executor.call_args_list], [bodies[m] for m in (4, 2, 1)]
        )
        self.assertEqual(prepare_sampling.call_count, 3)
        scratch_arena.assert_called_once()
        self.assertTrue(all(call.kwargs["constraint_arena"] is scratch_arena.return_value
                            for call in prepare_sampling.call_args_list))
        self.assertEqual(sampler_close.call_count, 3)

    def test_cache_sizing_includes_proposal_graphs_and_staging(self) -> None:
        body = SimpleNamespace(
            arena=SimpleNamespace(spec=SimpleNamespace(page_capacity=16)),
            buffers=SimpleNamespace(page_table=torch.empty((1, 1))),
            hidden=torch.empty((1, 8)),
        )
        proposal = object()
        admission = SimpleNamespace(
            num_speculative_tokens=1, max_running_requests=1, dynamic_max_speculative_tokens=1,
            parallel_plan=SimpleNamespace(topology=SimpleNamespace(pp_size=1)),
        )
        sampler_close = Mock()
        with (
            patch.object(serving.torch.cuda, "synchronize"),
            patch.object(serving.torch.cuda, "empty_cache"),
            patch.object(serving.torch.cuda, "reset_peak_memory_stats"),
            patch.object(serving.torch.cuda, "max_memory_allocated", side_effect=[120, 800]),
            patch.object(serving.torch.cuda, "memory_allocated", side_effect=[100, 650]),
            patch.object(serving.torch.cuda, "graph_pool_handle"),
            patch.object(serving.torch.cuda, "Stream"),
            patch.object(
                serving.torch.cuda,
                "mem_get_info",
                side_effect=[
                    (1000, 2000),
                    (900, 2000),
                    (900, 2000),
                    (500, 2000),
                ],
            ),
            patch.object(serving.Qwen3_5TP2PP2GraphSlot, "capture"),
            patch.object(serving, "_CapturedGraphRank"),
            patch.object(
                serving, "_prepare_rank_greedy_sampling",
                side_effect=lambda executor, lifetime, *, qualify_resident, constraint_arena: lifetime.callback(
                    sampler_close),
            ) as prepare_sampling,
            patch.object(serving, "_ConstraintSamplingArena") as scratch_arena,
            patch.object(serving, "_build_mtp_body_family", return_value={1: proposal}),
            patch.object(serving.Qwen3_5TPMTPGraphSlot, "capture") as capture,
            patch.object(serving, "_CapturedMTPRank") as executor,
        ):
            result = serving._profile_cuda_graph_memory(
                {1: cast(Any, body)},
                (1,),
                rank=0,
                device=torch.device("cuda:0"),
                runtime=object(),
                admission=cast(Any, admission),
                weights=object(),
            )
        self.assertEqual(result, (500, 150, {}))
        capture.assert_called_once_with(proposal)
        self.assertIs(executor.call_args.args[0], proposal)
        self.assertEqual(executor.call_args.kwargs["request_page_capacity"], 16)
        self.assertEqual(prepare_sampling.call_count, 2)
        scratch_arena.assert_called_once()
        self.assertTrue(all(call.kwargs["constraint_arena"] is scratch_arena.return_value
                            for call in prepare_sampling.call_args_list))
        self.assertEqual(sampler_close.call_count, 2)

    def test_close_reaps_started_workers_and_skips_unstarted_workers(self) -> None:
        controller = Qwen3_5TP2PP2PerformanceServingTests._batch_controller()
        controller._closed = False
        controller._condition = threading.Condition()
        controller._close_lock = threading.Lock()
        controller._commands = ()
        controller._results = ()
        controller._directory = Mock()
        controller._publish_worker_command = Mock()
        controller._continuation_interrupt = Mock()
        stopped = Mock(pid=123)
        stopped.is_alive.side_effect = (True, True)
        unstarted = Mock(pid=None)
        controller._processes = (stopped, unstarted)
        controller.close()
        controller.close()
        stopped.terminate.assert_called_once()
        stopped.kill.assert_called_once()
        self.assertEqual(stopped.join.call_count, 3)
        unstarted.join.assert_not_called()
        controller._directory.cleanup.assert_called_once()
        controller._continuation_interrupt.set.assert_called_once()

    def test_cancelled_inflight_mtp_result_does_not_poison_dispatcher(self) -> None:
        request_id = "request:cancel-inflight-mtp"
        command = serving._GenerateCommand(request_id, (10,), 4, ())
        controller = Qwen3_5TP2PP2PerformanceServingTests._batch_controller(command)
        record = controller._requests[request_id]
        record.prefill_cursor = len(command.input_ids)
        record.generated_count = 1
        record.last_token_id = 20
        record.state_slot = heapq.heappop(controller._free_state_slots)
        record.pages.append(heapq.heappop(controller._free_pages))
        record.state_initialized = True

        controller._free_mtp_pages = list(range(128))
        record.mtp_pages.append(heapq.heappop(controller._free_mtp_pages))
        controller._num_speculative_tokens = 1
        controller._feeding_profile = None
        controller._batch_id = 0
        controller._closed = False
        controller._poisoned = False
        controller._condition = threading.Condition()
        controller._close_lock = threading.Lock()
        controller._command_send_lock = threading.Lock()
        controller._commands = ()
        controller._results = ()
        controller._processes = ()
        controller._directory = Mock()
        controller._replay_leases = ReplayLeasePool((tuple(
            serving.RankReplayResources(
                rank, "graphs", "streams", "buffers", "workspace", "scratch"
            )
            for rank in range(4)
        ),))
        controller._replay_execution_id = "audited-plan"
        controller._replay_topology_id = "test-topology"

        def receive(batch: serving._BatchCommand) -> serving._BatchResult:
            self.assertEqual(controller._replay_leases.in_flight, 1)
            self.assertEqual(controller._replay_leases.available, 0)
            controller.cancel(request_id)
            return serving._BatchResult(
                0,
                batch.batch_id,
                tuple(row.request_id for row in batch.rows),
                (),
                (serving._SegmentResult(request_id, 2, (21, 22)),),
            )

        controller._receive_batch = Mock(side_effect=receive)
        dispatcher = threading.Thread(target=controller._dispatch)
        controller._dispatcher = dispatcher
        with patch.object(serving._LOGGER, "warning"):
            dispatcher.start()
            deadline = time.monotonic() + 2
            while request_id in controller._requests and time.monotonic() < deadline:
                time.sleep(0.01)
            with controller._condition:
                controller._closed = True
                controller._condition.notify_all()
            dispatcher.join(2)

        self.assertFalse(dispatcher.is_alive())
        self.assertNotIn(request_id, controller._requests)
        self.assertFalse(controller._poisoned)
        self.assertEqual(controller._replay_leases.in_flight, 0)
        self.assertEqual(controller._replay_leases.available, 1)
        controller._directory.cleanup.assert_not_called()


class Qwen3_5TokenMailboxTests(unittest.IsolatedAsyncioTestCase):
    async def test_c64_async_waiters_need_no_request_producer_threads(self) -> None:
        controller = _FanoutController()
        executor = serving.Qwen3_5TP2PP2PerformanceExecutor._for_testing(controller)
        streams = tuple(
            executor.generate((1,), 1, _Metadata(f"request:async:{index}")) for index in range(64)
        )

        async def next_token(stream: AsyncIterator[int]) -> int:
            return await anext(stream)

        waiters: tuple[asyncio.Task[int], ...] = tuple(
            asyncio.create_task(next_token(cast(AsyncIterator[int], stream))) for stream in streams
        )
        await asyncio.sleep(0)

        self.assertEqual(len(controller.outputs), 64)
        self.assertFalse(
            any(thread.name.startswith("cubic-token-stream-") for thread in threading.enumerate())
        )
        self.assertFalse(any(waiter.done() for waiter in waiters))

        controller.release()

        self.assertEqual(await asyncio.gather(*waiters), [20] * 64)
        for stream in streams:
            with self.assertRaises(StopAsyncIteration):
                await anext(cast(AsyncIterator[int], stream))
        self.assertTrue(executor.ready)


if __name__ == "__main__":
    unittest.main()


class CapacityBoundaryTests(unittest.TestCase):
    def test_every_request_limit_preserves_graph_and_audit_resources(self) -> None:
        # Enumerate whole intervals, not a whitelist of user-reported values.
        for pp in (1, 2):
            for draft in (0, 1, 3, 7):
                capacities = serving._physical_graph_capacities(
                    draft, pipeline_parallel_size=pp,
                )
                for prefix in (False, True):
                    for requests in range(1, 514):
                        owners = serving._owner_state_capacity(
                            requests, capacities,
                            max_speculative_tokens=draft, use_prefix_cache=prefix,
                        )
                        self.assertGreaterEqual(owners, requests)
                        self.assertGreaterEqual(owners, 4 if prefix or draft else 2)
                        retained = tuple(m for m in capacities if m <= owners)
                        self.assertEqual(retained[0], 1)
                        self.assertGreaterEqual(retained[-1], draft + 1)
                        for logical in range(1, min(requests, retained[-1]) + 1):
                            captured = serving._bucket_for_logical_m(logical, retained)
                            self.assertGreaterEqual(captured, logical)
                            self.assertIn(logical, serving._logical_m_values(captured, retained))
                            self.assertLessEqual(captured, owners)
                        # Rounding never creates an arbitrary exact-M graph.
                        self.assertTrue(all(m & (m - 1) == 0 for m in retained))

    def test_invalid_request_or_draft_capacity_fails_before_allocation(self) -> None:
        for requests, draft in ((0, 0), (-1, 1), (True, 0), (1, -1), (1, True)):
            with self.assertRaises(ValueError):
                serving._owner_state_capacity(
                    requests, (1, 2, 4, 8),
                    max_speculative_tokens=draft, use_prefix_cache=True,
                )



class ResidentContinuationProtocolTests(unittest.TestCase):
    def _controller(self, *, max_tokens=64, eos=()):
        command = serving._GenerateCommand("a", (10,), max_tokens, eos)
        controller = Qwen3_5TP2PP2PerformanceServingTests._batch_controller(command)
        controller._condition = threading.Condition()
        controller._batch_id = 0
        record = controller._requests["a"]
        record.prefill_cursor = 1
        record.generated_count = 1
        record.last_token_id = 20
        record.state_slot = heapq.heappop(controller._free_state_slots)
        record.pages.append(heapq.heappop(controller._free_pages))
        record.state_initialized = True
        return controller, record

    def test_survivors_reuse_only_the_qualified_unchanged_bucket(self):
        commands = tuple(serving._GenerateCommand(str(i), (10,), 64, ()) for i in range(4))
        controller = Qwen3_5TP2PP2PerformanceServingTests._batch_controller(*commands)
        previous = tuple(controller._requests.values())
        for record in previous:
            record.prefill_cursor = record.generated_count = 1
            record.state_slot = heapq.heappop(controller._free_state_slots)
            record.pages.append(heapq.heappop(controller._free_pages))
            record.state_initialized = True
        controller._resident_steady = previous
        controller._resident_steady_captured_m = 4
        controller._resident_compaction_capacities = (4,)
        del controller._requests['1']
        controller._resident_features_bound = True
        self.assertEqual(controller._resident_survivor_rows_locked(), ())
        self.assertEqual(controller._resident_steady, previous)
        controller._resident_features_bound = False
        self.assertEqual(controller._resident_survivor_rows_locked(), (0, 2, 3))
        self.assertEqual(controller._resident_steady, (previous[0], previous[2], previous[3]))
        del controller._requests['2']
        self.assertEqual(controller._resident_survivor_rows_locked(), ())
        controller._requests['2'] = previous[2]
        controller._resident_compaction_capacities = ()
        controller._resident_steady = previous
        self.assertEqual(controller._resident_survivor_rows_locked(), ())

    def _segment_controller(self):
        commands = (
            serving._GenerateCommand("p", tuple(range(40)), 64, ()),
            serving._GenerateCommand("d", tuple(range(4)), 64, ()),
        )
        c = Qwen3_5TP2PP2PerformanceServingTests._batch_controller(*commands)
        c._graph_capacities = c._physical_graph_capacities = (1, 2, 4)
        c.vocab_size = 128
        selected = tuple(c._requests.values())
        for r in selected:
            r.prefill_cursor = 4
            r.state_slot = heapq.heappop(c._free_state_slots)
            r.pages.append(heapq.heappop(c._free_pages))
            r.state_initialized = True
        selected[1].generated_count = 3
        selected[1].last_token_id = 17
        c._resident_steady = selected
        c._resident_owner_lengths = (2, 1)
        c._resident_steady_batch_id = 1
        c._resident_steady_captured_m = 4
        c._resident_segment_program_capacities = (4,)
        c._resident_step_seconds = 0.01
        c._decode_window_seconds = None
        c._batch_id = 2
        return c, selected

    def test_mixed_segment_window_is_independent_of_request_age(self):
        young, _ = self._segment_controller()
        old, records = self._segment_controller()
        for record in records:
            record.admitted_at = time.monotonic() - 86400
        expected, _, _ = young._next_segment_continuation_locked()
        actual, _, _ = old._next_segment_continuation_locked()
        self.assertEqual(actual, expected)
        self.assertEqual(tuple(old._free_pages), tuple(young._free_pages))

    def test_cost_history_is_private_to_processing_and_sampling_mode(self):
        from types import SimpleNamespace
        from cubic_serve.runtime.speculative import DynamicDraftPlanner
        controller = serving._ProcessController.__new__(serving._ProcessController)
        base = controller._draft_planner = DynamicDraftPlanner((1, 2, 4))
        base.seed_costs(target_samples=((1, 2.), (2, 3.), (4, 5.)), proposal_samples=((1, 1.),))
        domains = []
        for constrained in (False, True):
            for temperature in (0., 1.):
                rows = (SimpleNamespace(command=SimpleNamespace(
                    structured_output=object() if constrained else None,
                    sampling=SimpleNamespace(temperature=temperature))),)
                planner = controller._draft_cost_planner(rows)
                self.assertIs(controller._draft_cost_planner(rows), planner)
                domains.append(planner)
        self.assertEqual(len({id(p) for p in domains}), 4)
        # Completion can remove the last constrained owner before timing is
        # recorded. The immutable dispatched binding still selects its domain.
        rows = (SimpleNamespace(command=SimpleNamespace(
            structured_output=None, sampling=SimpleNamespace(temperature=1.))),)
        self.assertIs(controller._draft_cost_planner(rows, constrained=True), domains[-1])
        self.assertIs(controller._draft_cost_planner(rows), base)

    def test_cold_round_costs_do_not_cross_sampling_or_grammar_domains(self):
        from types import SimpleNamespace

        from cubic_serve.runtime.speculative import DynamicDraftPlanner
        controller = serving._ProcessController.__new__(serving._ProcessController)
        base = controller._draft_planner = DynamicDraftPlanner((1, 2, 4))
        base.seed_costs(target_samples=((1, 2.), (2, 3.), (4, 5.)),
                        proposal_samples=((1, 1.),), compound_samples=((4, 4, 8.),))
        controller._cold_compound_samples = {False: ((4, 4, 8.),), True: ((4, 4, 6.),)}
        for constrained in (False, True):
            for temperature in (0., 1.):
                rows = (SimpleNamespace(command=SimpleNamespace(
                    structured_output=object() if constrained else None,
                    sampling=SimpleNamespace(temperature=temperature))),)
                planner = controller._draft_cost_planner(rows)
                self.assertEqual(planner._seed_compound,
                                 {} if constrained else {(4, 4): 6. if temperature == 0. else 8.})

    def test_masked_tail_priors_are_bound_to_the_actual_sampler_mode(self):
        from cubic_serve.contracts.request import StructuredOutputMeta
        from cubic_serve.runtime.speculative import DynamicDraftPlanner

        c = serving._ProcessController.__new__(serving._ProcessController)
        c._draft_planner = DynamicDraftPlanner((1, 2, 4))
        c._draft_planner.seed_costs(target_samples=((1, 2.), (2, 3.), (4, 5.)),
                                   proposal_samples=((1, 1.),))
        c._cold_constraint_sampling_samples = {False: ((1, 1.5),), True: ((1, 0.75),)}
        for temperature, expected in ((0., .75), (1., 1.5)):
            row = SimpleNamespace(command=SimpleNamespace(
                structured_output=StructuredOutputMeta.regex('[ab]+'),
                reasoning_control=None, sampling=SimpleNamespace(temperature=temperature)))
            planner = c._draft_cost_planner((row,))
            self.assertTrue(planner.require_complete_rounds)
            self.assertEqual(planner._seed_constraint_sampling, {1: expected})
        self.assertEqual(c._draft_planner._seed_constraint_sampling, {})

    def test_integer_budget_and_grammar_keep_separate_round_prices(self):
        from cubic_serve.contracts.request import ReasoningControl, StructuredOutputMeta
        from cubic_serve.generation.structured_sampling import ConstraintBinding
        from cubic_serve.runtime.speculative import DynamicDraftPlanner

        c = serving._ProcessController.__new__(serving._ProcessController)
        c._draft_planner = DynamicDraftPlanner((1, 2, 4))
        c._draft_planner.seed_costs(target_samples=((1, 2.), (2, 3.), (4, 5.)),
                                   proposal_samples=((1, 1.),))
        c._cold_constraint_compound_samples = {
            (2, False): ((4, 4, 12.),), (True, False): ((4, 4, 19.),),
        }
        control = ReasoningControl((2,), (3,), 8, True)
        identity, grammar = StructuredOutputMeta.identity(), StructuredOutputMeta.regex('[ab]+')
        def record(meta, budget):
            return SimpleNamespace(command=SimpleNamespace(
                structured_output=meta, reasoning_control=budget,
                sampling=SimpleNamespace(temperature=1.)))
        plain, budget, structured = record(None, None), record(identity, control), record(grammar, None)
        ledger = c._draft_cost_planner((budget,))
        host = c._draft_cost_planner((structured,))
        self.assertIsNot(ledger, host)
        self.assertIsNot(ledger, c._draft_planner)
        self.assertEqual(ledger._seed_compound, {(4, 4): 12.})
        self.assertEqual(host._seed_compound, {(4, 4): 19.})
        self.assertEqual(c._draft_planner._seed_compound, {})
        self.assertIs(c._draft_cost_planner((plain, budget)), ledger)
        self.assertIs(c._draft_cost_planner((budget, structured)), host)
        # An immutable dispatched binding still names the correct domain when
        # the final controlled owner has retired before the result is priced.
        binding = ConstraintBinding('r', identity, 0, 8, reasoning_control=control)
        self.assertIs(c._draft_cost_planner((plain,), constrained=True, bindings=(binding,)), ledger)
        for _ in range(host.config.cost_sample_window):
            host.observe_compound_cost(target_rows=4, segment_depth=4, milliseconds=80.)
        self.assertEqual(host.compound_cost[4].values[4], 80.)
        self.assertNotIn(4, ledger.compound_cost)
        self.assertIs(c._draft_cost_planner((budget,)), ledger)

    def test_constrained_prefill_cannot_arm_an_unmasked_segment_window(self):
        from cubic_serve.contracts.request import StructuredOutputMeta

        for constrained in (False, True):
            command = serving._GenerateCommand('s', tuple(range(40)), 64, (),
                structured_output=StructuredOutputMeta.regex('[ab]+') if constrained else None)
            c = Qwen3_5TP2PP2PerformanceServingTests._batch_controller(command)
            c._graph_capacities = c._physical_graph_capacities = (1, 2, 4)
            c._resident_segment_program_capacities = (4,)
            c._active_structured = int(constrained)
            c._batch_id = 0
            batch, rows, selected = c._next_dispatch_locked()
            self.assertIsInstance(batch, serving._BatchCommand)
            self.assertEqual(len(rows), 4)
            self.assertEqual(c._resident_owner_lengths, () if constrained else (4,))
            self.assertEqual(bool(c._resident_steady), not constrained)

    def test_segment_window_commits_private_pages_only_after_final_receipt(self):
        c, selected = self._segment_controller()
        original = selected[0].pages[-1]
        before = set(c._free_pages)
        command, rows, actual = c._next_dispatch_locked()
        self.assertIsInstance(command, serving._ContinueBatchCommand)
        self.assertEqual((rows, actual), ((), selected))
        self.assertEqual(command.owner_lengths, (2, 1))
        self.assertEqual(command.logical_m, 3)
        self.assertEqual(command.max_steps, 17)
        self.assertEqual(command.prompt_tokens[0], tuple(range(4, 38)))
        self.assertEqual(command.resident_updates(
            previous_batch_id=1, captured_m=4, request_ids=('p','d'), offsets=(3,5),
            page_counts=(1,1), block_size=16,
        ), {})
        reserved = {page for r in selected for page in r.pending_cow_pages}
        self.assertEqual(set(c._free_pages), before - reserved)
        self.assertTrue(any(row[3] for step in command.page_program for row in step))
        for _ in range(command.max_steps - 1):
            c._commit_steady_records_locked(selected, (18,19), Mock(), profile=None,
                allow_prefill=True, prefill_counts=(2,1), defer_cancelled_release=True)
        self.assertEqual(selected[0].pages[-1], original)
        self.assertEqual(selected[0].prefill_cursor, 36)
        c._finish_segment_pages_locked(command, selected, command.max_steps)
        self.assertEqual(selected[0].pages[-1], command.page_program[-1][1][1])
        self.assertEqual(selected[0].pending_cow_pages, ())
        self.assertIn(original, c._free_pages)
        kept = {page for r in selected for page in r.pages}
        self.assertTrue((reserved - kept) <= set(c._free_pages))
        self.assertFalse(kept & set(c._free_pages))
        c._commit_steady_records_locked(selected, (18,19), Mock(), profile=None,
            allow_prefill=True, prefill_counts=(2,1))
        self.assertEqual(selected[0].prefill_cursor, 38)
        next_command, _, _ = c._next_dispatch_locked()
        self.assertIsInstance(next_command, serving._BatchCommand)

    def test_segment_cancel_holds_all_pages_until_deferred_final_release(self):
        c, selected = self._segment_controller()
        command, _, _ = c._next_dispatch_locked()
        owner = selected[0]
        pages = (*owner.pages, *owner.pending_cow_pages)
        owner.cancelled = True
        c._commit_steady_records_locked(selected, (18,19), Mock(), profile=None,
            allow_prefill=True, prefill_counts=(2,1), defer_cancelled_release=True)
        self.assertFalse(set(pages) & set(c._free_pages))
        c._finish_segment_pages_locked(command, selected, 1)
        self.assertFalse(set(pages) & set(c._free_pages))
        c._release_request_locked('p')
        self.assertTrue(set(pages) <= set(c._free_pages))
        self.assertEqual(len(c._free_pages), len(set(c._free_pages)))

    def test_segment_resource_or_capability_absence_never_reserves_pages(self):
        for reason in ('capability', 'page_budget', 'eos_budget', 'feature', 'media', 'constraint', 'prompt_complete'):
            c, selected = self._segment_controller()
            before = set(c._free_pages)
            if reason == 'capability':c._resident_segment_program_capacities = ()
            elif reason == 'page_budget':c._free_pages = [];before = set()
            elif reason == 'eos_budget':
                c.vocab_size = 2
                for record in selected:
                    record.command = replace(record.command,
                        input_ids=tuple(token % 2 for token in record.command.input_ids),
                        eos_token_ids=(0, 1))
                selected[1].last_token_id = 1
            elif reason == 'feature':c._resident_features_bound = True
            elif reason == 'constraint':c._active_structured = 1
            elif reason == 'media':
                selected[0].command = replace(selected[0].command,
                    feature_rows=(b'',)*4+(b'xx',)+(b'',)*35)
            else:selected[0].prefill_cursor = 40
            self.assertIsNone(c._next_segment_continuation_locked())
            self.assertEqual(set(c._free_pages), before)
            self.assertEqual(selected[0].pending_cow_pages, ())

    def test_segment_prefill_owner_cannot_invent_decode_position_transition(self):
        for first, step, positions, allowed in (
            (None, (2, 1, 1), (), False),
            (None, None, ((0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)), False),
            ((9, 9, 9), (1, 1, 1), (), False),
            ((4, 4, 4), (1, 1, 1), (), True),
            ((7, 10, 13), (1, 1, 1),
             ((0, 0, 0), (2, 3, 4), (4, 6, 8), (6, 9, 12)), True),
        ):
            with self.subTest(first=first, step=step, allowed=allowed):
                c, selected = self._segment_controller()
                owner = selected[1]
                owner.prefill_cursor, owner.generated_count, owner.last_token_id = 3, 0, None
                owner.command = replace(owner.command, positions=positions,
                    decode_position_first=first, decode_position_step=step)
                before = set(c._free_pages)
                command = c._next_segment_continuation_locked()
                self.assertEqual(command is not None, allowed)
                if not allowed:
                    self.assertEqual(set(c._free_pages), before)
                    self.assertTrue(all(not r.pending_cow_pages for r in selected))
                self.assertEqual(c._resident_decode_positions_safe(owner), allowed)

    def test_crosspage_interruption_adopts_only_completed_page_writes(self):
        for completed in (1, 2, 3, 9, 12):
            with self.subTest(completed=completed):
                c, selected = self._segment_controller()
                selected[0].prefill_cursor = 14
                before = set(c._free_pages)
                original = tuple(tuple(r.pages) for r in selected)
                command, _, _ = c._next_segment_continuation_locked()
                self.assertEqual(command.max_steps, 12)
                reserved = {page for r in selected for page in r.pending_cow_pages}
                expected = [list(pages) for pages in original]
                for coordinates in command.page_program[:completed]:
                    first = 0
                    for owner, width in enumerate(command.owner_lengths):
                        for _, target, index, _ in coordinates[first:first + width]:
                            if index == len(expected[owner]):
                                expected[owner].append(target)
                            else:
                                expected[owner][index] = target
                        first += width
                c._finish_segment_pages_locked(command, selected, completed)
                self.assertEqual([r.pages for r in selected], expected)
                self.assertTrue(all(not r.pending_cow_pages for r in selected))
                live = {page for pages in expected for page in pages}
                old = {page for pages in original for page in pages}
                self.assertEqual(set(c._free_pages), (before | old | reserved) - live)
                self.assertEqual(len(c._free_pages), len(set(c._free_pages)))

    def test_crosspage_program_reserves_sources_without_adopting_live_pages(self):
        c, selected = self._segment_controller()
        selected[0].prefill_cursor = 16
        selected[1].generated_count = 13
        old_pages = tuple(tuple(r.pages) for r in selected)
        old_cursors = tuple((r.prefill_cursor, r.generated_count) for r in selected)
        before = set(c._free_pages)
        program = c._stage_segment_page_program_locked(selected, (2, 1), 3)
        self.assertEqual(len(program), 3)
        self.assertEqual([len(step) for step in program], [3, 3, 3])
        self.assertEqual(tuple(tuple(r.pages) for r in selected), old_pages)
        self.assertEqual(tuple((r.prefill_cursor, r.generated_count) for r in selected), old_cursors)
        reserved = {page for r in selected for page in r.pending_cow_pages}
        self.assertEqual(before - set(c._free_pages), reserved)
        self.assertEqual(len(reserved), 5)
        self.assertTrue(all(row[2] == 1 for step in program for row in step))
        self.assertEqual(sum(row[3] for step in program for row in step), 2)
        self.assertTrue(all(row[0] in reserved for row in program[0]))

    def test_crosspage_program_rolls_back_or_shortens_at_actual_page_budget(self):
        for budget, completed in ((0, 0), (3, 0), (4, 2), (5, 3)):
            with self.subTest(budget=budget):
                c, selected = self._segment_controller()
                selected[0].prefill_cursor = 16
                selected[1].generated_count = 13
                c._free_pages = c._free_pages[:budget]
                original_heap = c._free_pages
                before = set(original_heap)
                program = c._stage_segment_page_program_locked(selected, (2, 1), 3)
                if not completed:
                    self.assertIsNone(program)
                    self.assertIs(c._free_pages, original_heap)
                    self.assertTrue(all(not r.pending_cow_pages for r in selected))
                else:
                    self.assertEqual(len(program), completed)
                    self.assertEqual(before - set(c._free_pages),
                        {page for r in selected for page in r.pending_cow_pages})
                self.assertTrue(all(not r.output_closed for r in selected))

    def test_crosspage_program_rejects_unqualified_phase_or_context_crossing(self):
        for reason in ('multi_decode', 'prompt_end', 'context_end'):
            c, selected = self._segment_controller()
            if reason == 'multi_decode':
                selected[0].prefill_cursor = len(selected[0].command.input_ids)
                selected[0].generated_count = 1
                selected[0].last_token_id = 17
            elif reason == 'prompt_end':
                selected[0].prefill_cursor = len(selected[0].command.input_ids) - 2
            else:
                c.max_sequence_tokens = 8
            before = tuple(c._free_pages)
            states = tuple(c._free_state_slots)
            self.assertIsNone(c._stage_segment_page_program_locked(selected, (2, 1), 3))
            self.assertEqual(tuple(c._free_pages), before)
            self.assertEqual(tuple(c._free_state_slots), states)
            self.assertTrue(all(not r.pending_cow_pages for r in selected))

    def test_crosspage_program_exception_restores_entire_reservation(self):
        c, selected = self._segment_controller()
        selected[0].prefill_cursor = 16
        selected[1].generated_count = 13
        original_heap = c._free_pages
        before = tuple(original_heap)
        original = c._reserve_segment_locked
        calls = 0
        def fail_second(*args, **kwargs):
            nonlocal calls
            result = original(*args, **kwargs)
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated preparation failure")
            return result
        with patch.object(c, '_reserve_segment_locked', side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, 'simulated preparation failure'):
                c._stage_segment_page_program_locked(selected, (2, 1), 3)
        self.assertIs(c._free_pages, original_heap)
        self.assertEqual(tuple(c._free_pages), before)
        self.assertTrue(all(not r.pending_cow_pages for r in selected))

    def test_window_cost_rule_is_bounded_by_page_and_output_budget(self) -> None:
        for generated, budget, expected in ((2, 64, 5), (13, 64, 5), (13, 14, 1)):
            controller, record = self._controller(max_tokens=budget, eos=(99,))
            controller._next_dispatch_locked()
            record.generated_count = generated
            controller._resident_step_seconds = 0.02
            continuation, _, _ = controller._next_dispatch_locked()
            self.assertEqual(continuation.max_steps, expected)
            self.assertEqual(continuation.eos_token_ids, ((99,),))

    def test_automatic_window_uses_request_and_transaction_budgets(self) -> None:
        from cubic_serve.runtime.continuous_batch import RESIDENT_STEP_CAPACITY

        for tokens in (31, 1536):
            controller, _record = self._controller(max_tokens=tokens)
            controller._decode_window_seconds = None
            controller._next_dispatch_locked()
            controller._resident_step_seconds = 0.001
            batch, _, _ = controller._next_dispatch_locked()
            self.assertEqual(batch.max_steps, min(tokens - 1, RESIDENT_STEP_CAPACITY))
            self.assertEqual(batch.max_duration_seconds, controller._worker_timeout)

    def test_long_request_age_does_not_shorten_continuation_windows(self) -> None:
        for age in (600.01, 3600.01, 86400.0):
            for configured in (None, 0.25):
                controller, record = self._controller(max_tokens=1536)
                controller._decode_window_seconds = configured
                controller._next_dispatch_locked()
                controller._resident_step_seconds = 0.02
                record.admitted_at = time.monotonic() - age
                batch, _, selected = controller._next_dispatch_locked()
                self.assertEqual(selected, (record,))
                self.assertEqual(batch.max_duration_seconds, configured or controller._worker_timeout)
                self.assertFalse(record.cancelled or record.output_closed)
                if configured is not None:
                    self.assertEqual(batch.max_steps, 12)

    def test_long_lived_two_request_batch_keeps_survivor_after_peer_finishes(self) -> None:
        commands = (serving._GenerateCommand("a", (10,), 2, ()),
                    serving._GenerateCommand("b", (11,), 6, ()))
        controller = Qwen3_5TP2PP2PerformanceServingTests._batch_controller(*commands)
        for record in controller._requests.values():
            record.admitted_at = time.monotonic() - 86400
            record.prefill_cursor = record.generated_count = 1
            record.last_token_id = 20
            record.state_slot = heapq.heappop(controller._free_state_slots)
            record.pages.append(heapq.heappop(controller._free_pages))
            record.state_initialized = True
        rows, selected = controller._next_batch_locked()
        self.assertEqual(len(selected), 2)
        wakes = serving.TokenWakeBatch()
        self.assertIsNotNone(controller._commit_steady_decode_locked(
            selected, rows, (21, 22), wakes, profile=None))
        wakes.flush()
        self.assertNotIn("a", controller._requests)
        survivor = controller._requests["b"]
        source_slot, pages = survivor.state_slot, tuple(survivor.pages)
        rows, selected = controller._next_batch_locked()
        self.assertEqual(selected, (survivor,))
        self.assertEqual(rows[0].state_source_slot, source_slot)
        self.assertEqual(rows[0].page_table, pages)
        self.assertFalse(survivor.cancelled or survivor.output_closed)
        wakes = serving.TokenWakeBatch()
        controller._commit_steady_decode_locked(selected, rows, (23,), wakes, profile=None)
        wakes.flush()
        self.assertEqual(survivor.generated_count, 3)

    def test_cancellation_inside_a_long_window_retains_inflight_ownership(self) -> None:
        controller, record = self._controller()
        record.admitted_at -= 86400
        controller._continuation_interrupt = threading.Event()
        slot, page = record.state_slot, record.pages[0]
        batch = serving._ContinueBatchCommand(2, 1, 1, ("a",), (), 3, ((),))
        intermediate = serving._BatchResult(0, 2, ("a",), (21,), continuation_complete=False)
        final = replace(intermediate, continuation_index=1, continuation_complete=True)

        def receive(_batch, *, window_index):
            self.assertEqual(window_index, 1)
            controller.cancel("a")
            self.assertTrue(controller._continuation_interrupt.is_set())
            self.assertTrue(record.cancelled and record.output_closed)
            self.assertIs(controller._requests["a"], record)
            self.assertNotIn(slot, controller._free_state_slots)
            self.assertNotIn(page, controller._free_pages)
            return final

        controller._receive_batch = receive
        result, _ = controller._consume_resident_window(
            batch, (record,), intermediate, stats_started_at=time.monotonic(), profile=None,
        )
        self.assertIs(result, final)
        self.assertIs(controller._requests["a"], record)
        controller._commit_steady_records_locked(
            (record,), final.token_ids, serving.TokenWakeBatch(), profile=None,
        )
        self.assertNotIn("a", controller._requests)
        self.assertIn(slot, controller._free_state_slots)
        self.assertIn(page, controller._free_pages)

    def test_resident_draft_prices_observe_steady_receipts_not_terminal_drain(self):
        from cubic_serve.runtime.speculative import DynamicDraftPlanner

        for intervals in ((12., 12., 12.), (18., 6., 12.)):
            with self.subTest(intervals=intervals):
                controller, record = self._controller()
                planner = controller._draft_planner = DynamicDraftPlanner((1,))
                planner.seed_costs(target_samples=((1, 10.),), proposal_samples=((1, 1.),))
                planner = controller._draft_cost_planner((record,), constrained=False)
                batch = serving._ContinueBatchCommand(2, 1, 1, ("a",), (), 4, ((),))
                first = serving._BatchResult(
                    0, 2, ("a",), (21,), target_ms=10., step_ms=intervals[0],
                    continuation_complete=False,
                )
                receipts = [replace(first, continuation_index=i, step_ms=intervals[i])
                            for i in (1, 2)]
                receipts.append(replace(first, continuation_index=3,
                                        continuation_complete=True, step_ms=500.))
                controller._receive_batch = lambda _batch, **_kwargs: receipts.pop(0)
                final, _ = controller._consume_resident_window(
                    batch, (record,), first, stats_started_at=time.monotonic(), profile=None,
                )
                self.assertTrue(final.continuation_complete)
                self.assertEqual(planner._resident_zero_cost.values, {1: 12.})
                self.assertEqual(planner._resident_zero_overhead.values, {1: 2.})
                self.assertEqual(planner._zero_overhead.values, {})
                planner._target_calibration.values[1] = 100.
                self.assertEqual(planner._zero_score(1), 1. / 12.)
                self.assertEqual(record.generated_count, 4)

    def test_cancel_does_not_release_owned_storage_between_receipts(self) -> None:
        controller, record = self._controller()
        record.cancelled = True
        slot, page = record.state_slot, record.pages[0]
        batch = serving._ContinueBatchCommand(2, 1, 1, ("a",), (), 3, ((),))
        intermediate = serving._BatchResult(
            0, 2, ("a",), (21,), continuation_complete=False
        )
        final = replace(intermediate, continuation_index=1, continuation_complete=True)
        def receive(_batch, *, window_index):
            self.assertEqual(window_index, 1)
            self.assertIs(controller._requests["a"], record)
            self.assertNotIn(slot, controller._free_state_slots)
            self.assertNotIn(page, controller._free_pages)
            return final
        controller._receive_batch = receive
        returned, _ = controller._consume_resident_window(
            batch, (record,), intermediate, stats_started_at=time.monotonic(), profile=None
        )
        self.assertIs(returned, final)
        self.assertIs(controller._requests["a"], record)
        controller._commit_steady_records_locked(
            (record,), final.token_ids, serving.TokenWakeBatch(), profile=None
        )
        self.assertNotIn("a", controller._requests)
        self.assertIn(slot, controller._free_state_slots)
        self.assertIn(page, controller._free_pages)

    def test_nonfinal_terminal_receipt_fails_before_state_commit(self) -> None:
        controller, record = self._controller(eos=(99,))
        batch = serving._ContinueBatchCommand(2, 1, 1, ("a",), (), 3, ((99,),))
        receipt = serving._BatchResult(0, 2, ("a",), (99,), continuation_complete=False)
        with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, "terminal receipt"):
            controller._consume_resident_window(
                batch, (record,), receipt, stats_started_at=time.monotonic(), profile=None
            )
        self.assertEqual(record.generated_count, 1)
        self.assertIs(controller._requests["a"], record)

    def test_worker_watchdog_renews_for_each_result_not_the_request_age(self) -> None:
        controller, record = self._controller()
        record.admitted_at -= 86400
        controller._sampler_rank = 0
        controller._feeding_profile = None
        controller._processes = ()
        channel = Mock()
        channel.poll.return_value = True
        controller._results = (channel,)
        row = controller._next_batch_locked()[0][0]
        batch = serving._BatchCommand(2, 1, (row,))
        valid = serving._BatchResult(0, 2, ("a",), (21,))
        channel.recv.return_value = valid
        for now in (100., 1000., 10000.):
            with patch.object(serving.time, "monotonic", return_value=now):
                self.assertIs(controller._receive_batch(batch), valid)
        # A worker that stops producing results still fails explicitly.
        clock = [10000.]
        def stalled_poll(seconds):
            clock[0] += seconds
            return False
        channel.poll.side_effect = stalled_poll
        with patch.object(serving.time, "monotonic", side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError,
                                        "graph operation timed out"):
                controller._receive_batch(batch)
        self.assertFalse(record.cancelled or record.output_closed)

    def test_gpu_period_receipt_requires_complete_positive_device_intervals(self) -> None:
        controller, _ = self._controller()
        controller._sampler_rank = 0
        controller._feeding_profile = None
        channel = Mock()
        channel.poll.return_value = True
        controller._results = (channel,)
        batch = serving._ContinueBatchCommand(2, 1, 1, ("a",), (), 3, ((),))
        valid = serving._BatchResult(
            0, 2, ("a",), (21,), continuation_index=1,
            gpu_step_intervals_ms=(2.5,),
        )
        channel.recv.return_value = valid
        self.assertIs(controller._receive_batch(batch, window_index=1), valid)
        for values in ((), (2.0, 3.0), (float("nan"),), (float("inf"),), (0.0,), (-1.0,)):
            channel.recv.return_value = replace(valid, gpu_step_intervals_ms=values)
            with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, "differs from dispatch"):
                controller._receive_batch(batch, window_index=1)
        channel.recv.return_value = replace(valid, gpu_step_intervals_ms=(2.5,),
                                           continuation_complete=False, continuation_index=0)
        with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, "differs from dispatch"):
            controller._receive_batch(batch, window_index=0)

    def test_grammar_cost_receipt_binds_the_exact_immutable_prefix(self) -> None:
        from cubic_serve.contracts.request import StructuredOutputMeta
        from cubic_serve.generation.structured_sampling import ConstraintBinding, GrammarRoundCost

        controller, record = self._controller()
        controller._sampler_rank = 0
        controller._feeding_profile = None
        channel = Mock()
        channel.poll.return_value = True
        controller._results = (channel,)
        meta = StructuredOutputMeta.regex('[ab]+')
        record.command = replace(record.command, structured_output=meta)
        row = controller._owned_single_token_row_locked(record)
        binding = ConstraintBinding('a', meta, record.generated_count, 8)
        batch = serving._BatchCommand(2, 1, (row,), (binding,))
        cost = GrammarRoundCost((('a', record.generated_count),), 1., 2., 2., 0., 1., 1., 0., 1, 1)
        valid = serving._BatchResult(0, 2, ('a',), (21,), grammar_cost=cost)
        channel.recv.return_value = valid
        self.assertIs(controller._receive_batch(batch), valid)
        for changed in ({'owners': (('other', 1),)}, {'owners': (('a', 0),)},
                        {'begin_ms': float('nan')}, {'output_tokens': 0}):
            channel.recv.return_value = replace(valid, grammar_cost=replace(cost, **changed))
            with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, 'differs from dispatch'):
                controller._receive_batch(batch)
        continuation = serving._ContinueBatchCommand(2, 1, 1, ('a',), (), 3, ((),))
        channel.recv.return_value = valid
        self.assertIs(controller._receive_batch(continuation), valid)
        record.generated_count += 1
        with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, 'differs from dispatch'):
            controller._receive_batch(continuation)
        channel.recv.return_value = replace(valid, grammar_cost=replace(
            cost, owners=(('a', record.generated_count),)))
        self.assertIs(controller._receive_batch(continuation), channel.recv.return_value)

    def test_cancelled_grammar_cost_prefix_drains_from_receipts_not_public_count(self):
        from cubic_serve.contracts.request import StructuredOutputMeta
        from cubic_serve.generation.structured_sampling import GrammarRoundCost

        controller, record = self._controller()
        controller._sampler_rank = 0
        controller._feeding_profile = None
        record.command = replace(record.command, structured_output=StructuredOutputMeta.regex('[ab]+'))
        channel = Mock()
        channel.poll.return_value = True
        controller._results = (channel,)
        batch = serving._ContinueBatchCommand(2, 1, 1, ('a',), (), 3, ((),))
        cost = GrammarRoundCost((('a', 1),), 1., 2., 2., 0., 1., 1., 0., 1, 1)
        first = serving._BatchResult(0, 2, ('a',), (21,), continuation_complete=False,
                                     grammar_cost=cost)
        channel.recv.return_value = first
        self.assertIs(controller._receive_batch(batch), first)
        record.cancelled = True
        controller._commit_steady_records_locked(
            (record,), (21,), serving.TokenWakeBatch(), profile=None, defer_cancelled_release=True)
        self.assertEqual(record.generated_count, 1)
        second = replace(first, continuation_index=1, continuation_complete=True,
                         grammar_cost=replace(cost, owners=(('a', 2),)))
        for bad_cost in (None, cost):
            channel.recv.return_value = replace(second, grammar_cost=bad_cost)
            with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, 'differs from dispatch'):
                controller._receive_batch(batch, window_index=1)
        channel.recv.return_value = second
        self.assertIs(controller._receive_batch(batch, window_index=1), second)
        self.assertIsNone(controller._grammar_cost_cursor)

    def test_missing_and_reordered_receipts_fail_closed(self) -> None:
        controller, _ = self._controller()
        controller._sampler_rank = 0
        controller._feeding_profile = None
        channel = Mock()
        channel.poll.return_value = True
        controller._results = (channel,)
        batch = serving._ContinueBatchCommand(2, 1, 1, ("a",), (), 3, ((),))
        for index, complete in ((1, True), (2, False)):
            channel.recv.return_value = serving._BatchResult(
                0, 2, ("a",), (21,), continuation_index=index,
                continuation_complete=complete,
            )
            with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, "differs from dispatch"):
                controller._receive_batch(batch, window_index=0 if index == 1 else 2)

    def _run_window(self, *, tokens=((10,), (11,), (12,)), eos=(), interrupt=None,
                    offsets=(2,), page_updates=(), deadline=1.0):
        from cubic_serve.runtime.continuous_batch import run_resident_continuation
        command = serving._ContinueBatchCommand(
            2, 1, 1, ("a",), page_updates, 3, (eos,), deadline
        )
        executor = Mock(captured_m=1)
        executor.sample_steady_continuation.side_effect = tokens
        executor.replay_elapsed_ms.return_value = 3.0
        barrier = Mock()
        receipts = []
        new_offsets, new_pages = run_resident_continuation(
            command, executor, previous_batch_id=1, request_ids=("a",),
            offsets=offsets, page_counts=(1,), block_size=16, rank=0, sampler_rank=0,
            completion_statuses=[0, 0], completion_barrier=barrier, timeout=1,
            interrupt=interrupt, publish=receipts.append,
        )
        return executor, barrier, receipts, new_offsets, new_pages

    def test_window_completes_every_step_before_publishing(self) -> None:
        executor, barrier, receipts, offsets, pages = self._run_window()
        self.assertEqual([r.token_ids for r in receipts], [(10,), (11,), (12,)])
        self.assertEqual([r.continuation_index for r in receipts], [0, 1, 2])
        self.assertEqual([r.continuation_complete for r in receipts], [False, False, True])
        self.assertEqual(executor.slot.synchronize.call_count, 3)
        self.assertEqual(barrier.wait.call_count, 3)
        self.assertEqual((offsets, pages), ((5,), (1,)))

    def test_eos_stops_before_another_state_write(self) -> None:
        executor, barrier, receipts, offsets, _ = self._run_window(eos=(11,))
        self.assertEqual(executor.replay_steady_continuation.call_count, 2)
        self.assertEqual(barrier.wait.call_count, 3)
        self.assertTrue(receipts[-1].continuation_complete)
        self.assertEqual(offsets, (4,))

    def test_new_work_interrupts_at_the_completed_boundary(self) -> None:
        interrupt = threading.Event()
        interrupt.set()
        executor, barrier, receipts, offsets, _ = self._run_window(interrupt=interrupt)
        self.assertEqual(executor.replay_steady_continuation.call_count, 1)
        self.assertEqual(barrier.wait.call_count, 2)
        self.assertTrue(receipts[0].continuation_complete)
        self.assertEqual(offsets, (3,))

    def test_deadline_does_not_wait_for_the_estimated_window(self) -> None:
        with patch("cubic_serve.runtime.continuous_batch.time.monotonic",
                   side_effect=(0.0, 0.06, 0.12)):
            executor, _, receipts, offsets, _ = self._run_window(deadline=0.1)
        self.assertEqual(executor.replay_steady_continuation.call_count, 2)
        self.assertTrue(receipts[-1].continuation_complete)
        self.assertEqual(offsets, (4,))

    def test_crossing_page_is_installed_only_once(self) -> None:
        executor, _, receipts, offsets, pages = self._run_window(
            offsets=(15,), page_updates=((0, 1, 7),)
        )
        self.assertEqual(executor.replay_steady_continuation.call_args_list,
                         [unittest.mock.call(1, ((0, 1, 7),)),
                          unittest.mock.call(1, ()), unittest.mock.call(1, ())])
        self.assertEqual((offsets, pages), ((2,), (2,)))
        self.assertEqual(len(receipts), 3)

    def test_window_cannot_touch_an_unallocated_page(self) -> None:
        with self.assertRaisesRegex(ValueError, "resident batch"):
            self._run_window(offsets=(13,))

    def test_reserved_future_page_is_installed_at_its_own_step(self) -> None:
        from cubic_serve.runtime.continuous_batch import run_resident_continuation
        command = serving._ContinueBatchCommand(
            2, 1, 1, ("a",), (), 3, ((),), 1.0, ((1, 0, 1, 7),)
        )
        executor = Mock(captured_m=1)
        executor.sample_steady_continuation.side_effect = [(10,), (11,), (12,)]
        executor.replay_elapsed_ms.return_value = 1.0
        offsets, pages = run_resident_continuation(
            command, executor, previous_batch_id=1, request_ids=("a",),
            offsets=(14,), page_counts=(1,), block_size=16, rank=0, sampler_rank=0,
            completion_statuses=[0, 0], completion_barrier=Mock(), timeout=1,
            interrupt=None, publish=lambda _: None,
        )
        self.assertEqual(executor.replay_steady_continuation.call_args_list,
                         [unittest.mock.call(1, ()), unittest.mock.call(1, ((0, 1, 7),)),
                          unittest.mock.call(1, ())])
        self.assertEqual((offsets, pages), ((1,), (2,)))

    def test_early_exit_releases_only_unconsumed_reservations(self) -> None:
        controller, record = self._controller()
        controller._next_dispatch_locked()
        record.generated_count = 13
        controller._resident_step_seconds = 0.02
        batch, _, selected = controller._next_dispatch_locked()
        self.assertEqual(batch.future_page_updates[0][:3], (3, 0, 1))
        reserved = record.pages[-1]
        self.assertNotIn(reserved, controller._free_pages)
        receipt = serving._BatchResult(0, batch.batch_id, ("a",), (21,))
        controller._consume_resident_window(
            batch, selected, receipt, stats_started_at=time.monotonic(), profile=None
        )
        self.assertEqual(len(record.pages), 1)
        self.assertIn(reserved, controller._free_pages)

    def test_page_pressure_shortens_the_window_before_its_crossing(self) -> None:
        controller, record = self._controller()
        controller._next_dispatch_locked()
        record.generated_count = 13
        controller._resident_step_seconds = 0.02
        controller._free_pages.clear()
        batch, _, _ = controller._next_dispatch_locked()
        self.assertEqual(batch.max_steps, 3)
        self.assertEqual(batch.future_page_updates, ())
        self.assertEqual(len(record.pages), 1)

    def test_all_ranks_observe_the_same_early_terminal_step(self) -> None:
        from cubic_serve.runtime.continuous_batch import run_resident_continuation
        command = serving._ContinueBatchCommand(2, 1, 1, ("a",), (), 3, ((11,),), 1.0)
        barrier = threading.Barrier(4)
        statuses = [0] * 8
        finished, errors = {}, []
        def rank_thread(rank):
            try:
                executor = Mock(captured_m=1)
                executor.sample_steady_continuation.side_effect = (
                    [(10,), (11,)] if rank == 0 else [(), ()]
                )
                executor.replay_elapsed_ms.return_value = 1.0
                finished[rank] = run_resident_continuation(
                    command, executor, previous_batch_id=1, request_ids=("a",),
                    offsets=(2,), page_counts=(1,), block_size=16, rank=rank,
                    sampler_rank=0, completion_statuses=statuses,
                    completion_barrier=barrier, timeout=2, interrupt=None, publish=lambda _: None,
                )
            except BaseException as error:
                errors.append(error)
        threads = [threading.Thread(target=rank_thread, args=(rank,)) for rank in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(finished, {rank: ((4,), (1,)) for rank in range(4)})

    def test_completion_slots_survive_skewed_successive_windows(self) -> None:
        from cubic_serve.runtime.continuous_batch import run_resident_continuation
        barrier = threading.Barrier(4)
        statuses = multiprocessing.get_context("spawn").Array("i", 8)
        errors = []
        def rank_thread(rank):
            try:
                for window in range(24):
                    steps = 1 + window % 3
                    eos = (11,) if window % 2 else ()
                    command = serving._ContinueBatchCommand(
                        window + 1, window, 1, ("a",), (), steps, (eos,), 1.0
                    )
                    executor = Mock(captured_m=1)
                    executor.sample_steady_continuation.side_effect = (
                        [(10,), (11,), (12,)] if rank == 0 else [(), (), ()]
                    )
                    executor.replay_elapsed_ms.return_value = 1.0
                    def wait(timeout):
                        result = barrier.wait(timeout)
                        if rank == 3:
                            time.sleep(0.001)
                        return result
                    offsets, pages = run_resident_continuation(
                        command, executor, previous_batch_id=window, request_ids=("a",),
                        offsets=(2,), page_counts=(1,), block_size=16, rank=rank,
                        sampler_rank=0, completion_statuses=statuses,
                        completion_barrier=SimpleNamespace(wait=wait), timeout=2,
                        interrupt=None, publish=lambda _: None,
                    )
                    self.assertEqual(offsets, (2 + min(steps, 2 if eos else steps),))
                    self.assertEqual(pages, (1,))
            except BaseException as error:
                errors.append(error)
        threads = [threading.Thread(target=rank_thread, args=(rank,)) for rank in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=4)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])

    def test_sparse_updates_require_the_completed_resident_identity(self) -> None:
        command = serving._ContinueBatchCommand(11, 10, 4, ("a", "b", "c"), ((2, 1, 9),))
        state = dict(previous_batch_id=10, captured_m=4, request_ids=("a", "b", "c"),
                     offsets=(1, 14, 15), page_counts=(1, 1, 1), block_size=16)
        self.assertEqual(command.resident_updates(**state), {2: (1, 9)})
        for changed in (
            replace(command, previous_batch_id=9),
            replace(command, captured_m=8),
            replace(command, request_ids=("b", "a", "c")),
            replace(command, page_updates=()),
            replace(command, page_updates=((2, 2, 9),)),
            replace(command, page_updates=((2, 1, 9), (2, 1, 10))),
            replace(command, page_updates=((0, 1, 8), (2, 1, 9))),
            replace(command, page_updates=((3, 1, 9),)),
        ):
            with self.subTest(command=changed):
                self.assertIsNone(changed.resident_updates(**state))
        self.assertIsNone(command.resident_updates(**{**state, "offsets": (1, 15)}))
        self.assertIsNone(command.resident_updates(**{**state, "page_counts": (1, 1)}))
        ordinary = replace(command, page_updates=())
        self.assertEqual(ordinary.resident_updates(**{**state, "offsets": (1, 14, 14)}), {})

    def test_retained_rows_exclude_cow_prefill_and_reset_transactions(self) -> None:
        from cubic_serve.runtime.continuous_batch import owned_single_token_rows
        row = serving._BatchRow("a", 17, 3, 0, 0, 1, 1, (1,), is_decode=True)
        self.assertTrue(owned_single_token_rows((row,)))
        self.assertFalse(owned_single_token_rows(()))
        for changed in (
            replace(row, is_decode=False), replace(row, reset_state=True),
            replace(row, segment_length=2), replace(row, segment_length=0),
            replace(row, state_source_slot=1), replace(row, page_source=2),
            replace(row, input_feature=b"feature"),
        ):
            self.assertFalse(owned_single_token_rows((changed,)))


class WorkerMetadataTests(unittest.TestCase):
    def test_owned_source_lifetime_covers_worker_success_and_failure(self) -> None:
        import tempfile
        from pathlib import Path

        for fails in (False, True):
            with self.subTest(fails=fails):
                source = Mock()
                admission = SimpleNamespace(model_definition_source=source)

                def worker(*args, admission=admission, source=source, fails=fails):
                    self.assertIs(args[2], admission)
                    source.close.assert_not_called()
                    if fails:
                        raise RuntimeError("worker failed")

                with patch.object(serving.pickle, "load", return_value=admission), \
                        tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "admission.pkl"
                    path.write_bytes(b"controller-owned metadata")
                    args = (worker, str(path), 0, "file:///store", None, None, 0,
                            Mock(), 1, None, None)
                    if fails:
                        with self.assertRaisesRegex(RuntimeError, "worker failed"):
                            serving._worker_from_metadata(*args)
                    else:
                        serving._worker_from_metadata(*args)
                source.close.assert_called_once_with()

    def test_spawn_metadata_preserves_admission_and_live_handles(self) -> None:
        import tempfile
        from pathlib import Path

        admission = SimpleNamespace(payload=b"metadata" * 100000, topology=(4, 1, 4))
        handles = tuple(object() for _ in range(7))
        worker = Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "admission.pkl"
            path.write_bytes(pickle.dumps(admission, protocol=pickle.HIGHEST_PROTOCOL))
            serving._worker_from_metadata(worker, str(path), 2, "file:///store", *handles)
        args = worker.call_args.args
        self.assertEqual(args[:3], (2, "file:///store", admission))
        self.assertEqual(args[3:], handles)

    def test_metadata_failure_notifies_parent_without_starting_worker(self) -> None:
        import tempfile
        from pathlib import Path

        worker, results = Mock(), Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "admission.pkl"
            path.write_bytes(b"invalid metadata")
            serving._worker_from_metadata(
                worker, str(path), 2, "file:///store", None, None, 0, results, 1, None, None,
            )
        worker.assert_not_called()
        failure = results.send.call_args.args[0]
        self.assertIsInstance(failure, serving._Failure)
        self.assertEqual(failure.rank, 2)
        self.assertIn("UnpicklingError", failure.detail)


def test_shared_checkpoint_bank_follows_actual_work_reuses_ids_and_returns_idle_bytes():
    from cubic_serve.runtime.cache.shared_pool import SharedCachePool,SharedCacheFreeSlots
    command=serving._GenerateCommand('r',(10,),64,())
    c=Qwen3_5TP2PP2PerformanceServingTests._batch_controller(command)
    pool=SharedCachePool(pool_id='test',budget_bytes=8192,page_bytes=32,state_bytes=256)
    c._shared_cache_pool=pool
    c._state_slot_capacity=pool.state_capacity
    c._state_scratch_capacity=16
    c._free_state_slots=SharedCacheFreeSlots(pool,'state')
    c._free_pages=SharedCacheFreeSlots(pool,'page')
    c._free_mtp_pages=SharedCacheFreeSlots(pool,'page')
    c._checkpoint_bank=[];c._checkpoint_bank_set=set();c._checkpoint_bank_used=0
    c._num_speculative_tokens=3
    r=c._requests['r'];r.prefill_cursor=r.generated_count=1;r.last_token_id=17;r.state_initialized=True
    r.state_slot=c._free_state_slots.allocate()
    r.pages.append(c._free_pages.allocate());r.mtp_pages.append(c._free_mtp_pages.allocate())
    before=pool.live_bytes
    first=c._reserve_segment_locked(r,4,running=1)
    assert first is not None
    checkpoints=tuple(row.state_destination_slot for row in first[:-1])
    assert len(set(checkpoints))==3 and r.state_slot not in checkpoints
    assert all(c._private_checkpoint(slot) for slot in checkpoints)
    assert pool.live_bytes==before+3*256+32
    c._free_pages.release(first[0].page_destination)
    versions={i:c._free_state_slots._tickets[i].generation for i in checkpoints}
    c._checkpoint_bank_used=0
    second=c._reserve_segment_locked(r,4,running=1)
    assert tuple(row.state_destination_slot for row in second[:-1])==checkpoints
    assert {i:c._free_state_slots._tickets[i].generation for i in checkpoints}==versions
    c._free_pages.release(second[0].page_destination)
    c._trim_checkpoint_bank(0)
    assert pool.live_bytes==before
    assert not c._checkpoint_bank_set
    pool.validate(c._free_state_slots._tickets[r.state_slot])


def test_seed_trajectory_uses_granted_shared_ids_and_preserves_the_durable_root():
    from types import SimpleNamespace
    rows=tuple(SimpleNamespace(state_source_slot=7,state_destination_slot=slot)
               for slot in (57,14,91,9))
    expected=((7,9),(9,57),(57,14))
    for ordinal,pair in enumerate(expected):
        assert serving._seed_provider_state_pair(rows,0,4,ordinal,
            scratch_state=1000,physical=3,depth=3,shared=True)==pair
    legacy=((7,9),(9,1012),(1012,1013))
    for ordinal,pair in enumerate(legacy):
        assert serving._seed_provider_state_pair(rows,0,4,ordinal,
            scratch_state=1000,physical=3,depth=3,shared=False)==pair


def test_shared_active_rows_reject_padding_slots_before_any_metadata_access():
    import pytest
    rank=serving._CapturedGraphRank.__new__(serving._CapturedGraphRank)
    rank.captured_m=1
    rank.body=SimpleNamespace(arena=SimpleNamespace(shared_storage=object(),
                                                  spec=SimpleNamespace(state_slot_capacity=4)))
    for src,dst in ((4,0),(0,4),(4,4),(-1,0)):
        row=serving._BatchRow('request',17,0,src,dst,0,0,(0,))
        with pytest.raises(serving.Qwen3_5TP2PP2ServingError,match='padding-only'):
            rank._stage_batch_rows((row,),host_slot=0)


def _full_shared_cache_controller(*,dynamic,prefill=False):
    from cubic_serve.runtime.cache.shared_pool import SharedCachePool,SharedCacheFreeSlots
    command=serving._GenerateCommand('pressure',(10,)*895,64,())
    c=Qwen3_5TP2PP2PerformanceServingTests._batch_controller(command)
    pool=SharedCachePool(pool_id='pressure',budget_bytes=2048,page_bytes=32,state_bytes=256)
    c._shared_cache_pool=pool;c._state_slot_capacity=pool.state_capacity;c._state_scratch_capacity=7
    c._page_capacity=pool.page_capacity
    c._free_state_slots=SharedCacheFreeSlots(pool,'state')
    c._free_pages=SharedCacheFreeSlots(pool,'page');c._free_mtp_pages=SharedCacheFreeSlots(pool,'page')
    c._checkpoint_bank=[];c._checkpoint_bank_set=set();c._checkpoint_bank_used=0
    c._num_speculative_tokens=1;c._dynamic_speculation=dynamic
    r=c._requests['pressure'];r.prefill_cursor=890 if prefill else 895
    r.generated_count=0 if prefill else 1;r.last_token_id=17;r.state_initialized=True
    r.state_slot=c._free_state_slots.allocate()
    while len(c._free_pages):r.pages.append(c._free_pages.allocate())
    for _ in r.pages:r.mtp_pages.append(c._free_mtp_pages.allocate())
    return c,r,pool


def test_dynamic_shared_cache_pressure_advances_k0_and_retires_provider_ownership():
    c,r,pool=_full_shared_cache_controller(dynamic=True)
    from cubic_serve.runtime.speculative import DynamicDraftPolicy
    r.draft_policy=DynamicDraftPolicy(1,1)
    rows,selected=c._next_batch_locked()
    assert len(rows)==len(selected)==1 and rows[0].position==895
    assert rows[0].mtp_position==-1 and not rows[0].mtp_page_table
    assert r.draft_stopped and r.draft_policy.stopped and not r.mtp_pages
    assert not r.output_closed and not r.cancelled
    assert pool.live_bytes==2048 and c._checkpoint_bank_used==0
    assert c._kv_usage_percent_locked()==100.*56/64


def test_fixed_shared_cache_pressure_reports_capacity_and_releases_all_ownership():
    c,r,pool=_full_shared_cache_controller(dynamic=False)
    rows,selected=c._next_batch_locked()
    assert not rows and not selected
    assert r.output_closed and r.cancelled and not c._requests
    assert pool.live_bytes==0


def _submission_controller(*, full=False):
    c, resident, pool = _full_shared_cache_controller(dynamic=False)
    if not full:
        c._release_request_locked(resident.command.request_id)
    c._closed = c._poisoned = False
    c._condition = threading.Condition()
    c._draft_planner = None
    return c, resident, pool


def test_full_pool_rejects_admission_without_touching_live_request_and_recovers():
    import pytest
    c, resident, pool = _submission_controller(full=True)
    command = serving._GenerateCommand('new', (10,), 2, ())
    ownership = (resident.state_slot, tuple(resident.pages), tuple(resident.mtp_pages))
    with pytest.raises(serving.Qwen3_5TP2PP2Overloaded, match='admission capacity'):
        c.submit(command, serving._TokenStream())
    assert list(c._requests) == ['pressure']
    assert ownership == (resident.state_slot, tuple(resident.pages), tuple(resident.mtp_pages))
    assert not resident.cancelled and not resident.output_closed
    assert pool.live_bytes == 2048
    c._release_request_locked('pressure')
    c.submit(command, serving._TokenStream())
    assert list(c._requests) == ['new'] and pool.live_bytes == 0


def test_concurrent_admissions_account_for_pending_state_and_both_kv_pools():
    from concurrent.futures import ThreadPoolExecutor
    c, _, pool = _submission_controller()

    def submit(index):
        try:
            c.submit(serving._GenerateCommand(str(index), (10,), 2, ()), serving._TokenStream())
        except serving.Qwen3_5TP2PP2Overloaded:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as executor:
        accepted = list(executor.map(submit, range(8)))
    # One 256-byte state and one 32-byte page bundle per pending admission;
    # target/provider fields share each bundle, with separate ownership.
    assert sum(accepted) == 7 and len(c._requests) == 7
    assert pool.live_bytes == 0
    for request_id in tuple(c._requests):
        c._release_request_locked(request_id)
    assert submit('retry')


def test_cache_deadlock_rejects_new_prefill_before_existing_decode():
    c, resident, pool = _submission_controller(full=True)
    command = serving._GenerateCommand('new', (10,), 2, ())
    newcomer = serving._ScheduledRequest(command, serving._TokenStream(), resident.admitted_at + 1)
    c._requests['new'] = newcomer
    c._request_order.append('new')
    c._serving_stats.admit(newcomer.admitted_at)
    rows, selected = c._next_batch_locked()
    assert not rows and not selected
    assert newcomer.cancelled and newcomer.output_closed
    assert not resident.cancelled and not resident.output_closed
    assert list(c._requests) == ['pressure'] and pool.live_bytes == 2048


def test_shared_cache_prefill_pressure_uses_one_causal_token_without_draft_policy_change():
    c,r,pool=_full_shared_cache_controller(dynamic=True,prefill=True)
    rows,selected=c._next_batch_locked()
    assert len(rows)==len(selected)==1 and rows[0].position==890
    assert not r.draft_stopped and not r.output_closed
    assert pool.live_bytes==2048


def _cold_calibration_body(budget):
    from cubic_serve.runtime.cache.shared_pool import SharedCachePool
    pool = SharedCachePool(pool_id="calibration", budget_bytes=budget,
                          page_bytes=32, state_bytes=256)
    storage = SimpleNamespace(allocator=pool, budget_bytes=budget,
        page_layout=SimpleNamespace(entry_bytes=32),
        state_layout=SimpleNamespace(entry_bytes=256))
    return SimpleNamespace(captured_m=16, hidden=torch.empty(0), arena=SimpleNamespace(
        shared_storage=storage, spec=SimpleNamespace(state_slot_capacity=pool.state_capacity),
        layout=SimpleNamespace(block_size_tokens=2)))


def test_cold_calibration_rows_respect_actual_shared_bytes_and_segment_owners():
    body = _cold_calibration_body(2048)
    ordinary = serving._cold_start_calibration_signature("target-costs", 16, body, ())
    assert ordinary[1] == 6
    corpora = [serving._cold_start_owned_rows(body, ordinary[1], position=i, reset=i == 0)
               for i in range(4)]
    deep = serving._cold_start_calibration_signature("target-depth-costs", 16, body, 4)
    for depth, logical in deep[1]:
        rows = serving._cold_start_depth_rows(body, logical, depth)
        assert sum(r.segment_length > 0 for r in rows) == (logical + depth - 1) // depth
        assert sum(r.segment_length for r in rows) == logical
        corpora.append(rows)
    for rows in corpora:
        states = {i for r in rows for i in (r.state_source_slot, r.state_destination_slot)}
        pages = {i for r in rows for i in r.page_table}
        assert all(0 <= i < 8 for i in states)
        for page in pages:
            p = (page * 32, (page + 1) * 32)
            for state in states:
                s = (state * 256, (state + 1) * 256)
                assert p[1] <= s[0] or s[1] <= p[0]


def test_cold_price_identity_binds_changed_owned_extent(monkeypatch):
    from cubic_serve.runtime.preparation import StartupPreparation
    from cubic_serve.runtime import quantized_qualification
    monkeypatch.setattr(quantized_qualification, "_bound_composition_plan",
                        lambda *a, **kw: "unchanged-kernel-and-workload")
    preparation = object.__new__(StartupPreparation)
    preparation.topology = (2, 1, 1)
    small = SimpleNamespace(body=_cold_calibration_body(2048))
    large = SimpleNamespace(body=_cold_calibration_body(4096))
    a, ai = preparation.cost_identity("target-costs", 16, small, ())
    b, bi = preparation.cost_identity("target-costs", 16, large, ())
    assert ai == bi and a != b
    assert a[-1][0] == b[-1][0] == "exclusive-owned-append-v1"


def test_mtp_cost_cleanup_uses_touched_pages_instead_of_physical_graph_rows(monkeypatch):
    arena = SimpleNamespace(
        shared_storage=SimpleNamespace(state_capacity=6),
        key=torch.full((8, 2), 19.0, dtype=torch.bfloat16),
        value=torch.full((8, 2), 23.0, dtype=torch.bfloat16),
        written=torch.ones((8, 2), dtype=torch.bool),
        reset_state_rows=Mock(),
    )
    body = SimpleNamespace(rank=0, arena=arena, buffers=SimpleNamespace(
        hidden=torch.empty((1024, 2)), target_hidden=torch.empty((1024, 2))))
    replay = Mock()
    executor = SimpleNamespace(captured_m=1024, body=body, replay=replay)
    monkeypatch.setattr(serving, "_cold_start_owner_limit", lambda body: 6)
    monkeypatch.setattr(serving, "barrier", lambda: None)
    monkeypatch.setattr(serving.torch.cuda, "current_stream", lambda device: Mock())
    result = serving._cold_start_mtp_proposal_costs({1024: executor})
    assert result[0][0] == 1024 and result[0][1] > 0
    assert replay.call_count == 4
    rows, hidden = replay.call_args.args
    assert len(rows) == 6 and hidden.shape == (6, 2)
    assert tuple(row.state_source_slot for row in rows) == tuple(range(5, -1, -1))
    assert torch.equal(arena.key[6:], torch.full((2, 2), 19.0))
    assert torch.equal(arena.value[6:], torch.full((2, 2), 23.0))
    assert bool(arena.written[6:].all())
    assert not bool(arena.key[:6].any() or arena.value[:6].any() or arena.written[:6].any())


def test_padding_poison_uses_bound_compact_slots_and_preserves_other_lease():
    rows = 1024
    buffers = SimpleNamespace(
        position_ids=torch.zeros(3, rows, dtype=torch.int64),
        page_source=torch.full((rows,), 8, dtype=torch.int32),
        page_destination=torch.full((rows,), 8, dtype=torch.int32),
        token_offsets=torch.zeros(rows, dtype=torch.int32),
        page_table=torch.full((rows, 1), 8, dtype=torch.int32),
        sequence_lengths=torch.ones(rows, dtype=torch.int32),
        segment_lengths=torch.zeros(rows, dtype=torch.int32),
        row_activity=torch.zeros(rows, dtype=torch.int32),
        conv_slots=torch.full((2, 1, rows), 4, dtype=torch.int32),
        recurrent_slots=torch.full((2, 1, rows), 4, dtype=torch.int32),
    )
    arena = SimpleNamespace(shared_storage=object(),
        spec=SimpleNamespace(storage_page_capacity=8, state_slot_capacity=4),
        key=torch.zeros(10, 2, dtype=torch.uint8),
        value=torch.zeros(10, 2, dtype=torch.uint8),
        written=torch.zeros(10, 2, dtype=torch.bool),
        states={name: torch.zeros(6, 2, dtype=torch.bfloat16)
                for name in (CONV_STATE_ID, RECURRENT_STATE_ID, "history")},
        page_fields={"scale": torch.zeros(10, 2)},
    )
    body = SimpleNamespace(stage=1, captured_m=rows, hidden=torch.empty(rows, 2),
                           buffers=buffers, arena=arena)
    serving._set_padding_poison(body, 1, logical_m=1)
    for tensor, selected in ((arena.key, 8), (arena.value, 8),
                             *[(t, 4) for t in arena.states.values()]):
        assert bool(tensor[selected].any())
        protected = torch.arange(tensor.shape[0]) != selected
        assert not bool(tensor[protected].any())
    assert buffers.page_destination[1:].unique().tolist() == [8]
    assert buffers.conv_slots[:, :, 1:].unique().tolist() == [4]
