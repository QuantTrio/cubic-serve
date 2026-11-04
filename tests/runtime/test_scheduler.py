from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from cubic_serve.contracts.execution import (
    BatchPhase,
    DecodeExtensionMeta,
    DecodeProvider,
)
from cubic_serve.contracts.model import WeightVariantMeta
from cubic_serve.contracts.request import (
    ContentPart,
    DecodePositionPlan,
    InputPlan,
    TokenPosition,
)
from cubic_serve.runtime.scheduler import (
    ContinuousScheduler,
    PageAllocation,
    SegmentCompletion,
)


class FakePageAllocator:
    def __init__(self, block_size_tokens: int = 4) -> None:
        self._block_size = block_size_tokens
        self._page_tables: dict[str, dict[int, int]] = {}
        self._next_page_id = 0
        self.calls: list[tuple[str, int, int]] = []

    @property
    def block_size_tokens(self) -> int:
        return self._block_size

    def allocate(
        self, *, request_id: str, token_start: int, token_count: int
    ) -> PageAllocation:
        self.calls.append((request_id, token_start, token_count))
        table = self._page_tables.setdefault(request_id, {})
        end = token_start + token_count
        for logical_page in range((end + self._block_size - 1) // self._block_size):
            if logical_page not in table:
                table[logical_page] = self._next_page_id
                self._next_page_id += 1
        logical_write_pages = {
            position // self._block_size for position in range(token_start, end)
        }
        context_pages = tuple(table[index] for index in range((end - 1) // self._block_size + 1))
        write_pages = tuple(sorted(table[index] for index in logical_write_pages))
        slots = tuple(
            table[position // self._block_size] * self._block_size
            + position % self._block_size
            for position in range(token_start, end)
        )
        return PageAllocation(
            cache_slots=slots,
            context_pages=context_pages,
            write_pages=write_pages,
        )


def input_plan(request_id: str, token_count: int) -> InputPlan:
    return InputPlan(
        request_id=request_id,
        model_id="toy",
        processor_recipe_digest=f"b2s256:{'a' * 64}",
        weight_variant=WeightVariantMeta(
            base_model_id="toy",
            base_revision="revision:1",
        ),
        content_parts=(ContentPart(ordinal=0, modality="text", text="fixture"),),
        token_ids=tuple(range(token_count)),
        positions=tuple(TokenPosition((index,)) for index in range(token_count)),
        decode_positions=DecodePositionPlan(
            first=TokenPosition((token_count,)),
            step_axes=(1,),
        ),
    )


class SchedulerTests(unittest.TestCase):
    def scheduler(
        self,
        allocator: FakePageAllocator,
        *,
        budget: int = 4,
        chunk: int = 3,
        alignment: int = 4,
    ) -> ContinuousScheduler:
        return ContinuousScheduler(
            page_allocator=allocator,
            token_budget=budget,
            prefill_chunk_tokens=chunk,
            topology_id="single",
            partition_id="toy",
            owner_ranks=(0, 1),
            state_ids=("recurrent",),
            state_commit_alignment_tokens=alignment,
        )

    def test_chunked_prefill_reserves_each_span_once(self) -> None:
        allocator = FakePageAllocator()
        scheduler = self.scheduler(allocator, budget=3, chunk=3)
        state = scheduler.admit(input_plan("a", 7), max_new_tokens=2)

        first = scheduler.next_plan()
        self.assertIsNotNone(first)
        assert first is not None
        self.assertEqual(first.phase, BatchPhase.PREFILL)
        self.assertEqual((first.segments[0].token_start, first.segments[0].token_count), (0, 3))
        self.assertIsNone(scheduler.next_plan())  # outstanding rows are never described twice
        scheduler.complete("a")

        second = scheduler.next_plan()
        assert second is not None
        self.assertEqual((second.segments[0].token_start, second.segments[0].token_count), (3, 1))
        scheduler.complete("a")
        final = scheduler.next_plan()
        assert final is not None
        self.assertEqual((final.segments[0].token_start, final.segments[0].token_count), (4, 3))
        scheduler.complete("a", generated_token_id=90)
        self.assertEqual(state.prefill_cursor, 7)
        self.assertEqual(state.generated_token_ids, (90,))
        self.assertTrue(state.decode_pending)

    def test_mixed_batch_and_exact_m_one_two(self) -> None:
        allocator = FakePageAllocator()
        scheduler = self.scheduler(allocator, budget=3, chunk=2)
        scheduler.admit(input_plan("decode", 1), max_new_tokens=3)
        warmup = scheduler.next_plan()
        assert warmup is not None
        self.assertEqual(warmup.shape.logical_m, 1)
        scheduler.complete("decode", generated_token_id=10)
        scheduler.admit(input_plan("prefill", 5), max_new_tokens=1)

        mixed = scheduler.next_plan()
        assert mixed is not None
        self.assertEqual(mixed.phase, BatchPhase.MIXED)
        self.assertEqual(
            [
                (segment.request_id, segment.phase, segment.token_count)
                for segment in mixed.segments
            ],
            [
                ("decode", BatchPhase.DECODE, 1),
                ("prefill", BatchPhase.PREFILL, 2),
            ],
        )
        self.assertEqual(mixed.shape.logical_m, 3)
        scheduler.complete("decode", generated_token_id=11)
        scheduler.complete("prefill")
        scheduler.cancel("prefill")

        decode_only = scheduler.next_plan()
        assert decode_only is not None
        self.assertEqual(decode_only.segments[0].request_id, "decode")
        self.assertEqual(decode_only.segments[0].token_count, 1)
        self.assertEqual(decode_only.shape.logical_m, 1)

    def test_decode_first_and_independent_prefill_budget(self) -> None:
        allocator = FakePageAllocator(block_size_tokens=16)
        scheduler = ContinuousScheduler(
            page_allocator=allocator,
            token_budget=6,
            prefill_chunk_tokens=6,
            prefill_token_budget=1,
            topology_id="single",
            partition_id="toy",
            owner_ranks=(0,),
            state_ids=(),
            state_commit_alignment_tokens=1,
        )
        scheduler.admit(input_plan("prefill", 8), max_new_tokens=1)
        scheduler.admit(input_plan("decode", 1), max_new_tokens=3)

        first = scheduler.next_plan()
        assert first is not None
        self.assertEqual(first.input_plan_ids, ("prefill",))
        scheduler.complete("prefill")
        second = scheduler.next_plan()
        assert second is not None
        self.assertEqual(second.input_plan_ids, ("decode",))
        scheduler.complete("decode", generated_token_id=10)

        plan = scheduler.next_plan()
        assert plan is not None
        self.assertEqual(
            tuple(
                (segment.request_id, segment.phase, segment.token_count)
                for segment in plan.segments
            ),
            (
                ("decode", BatchPhase.DECODE, 1),
                ("prefill", BatchPhase.PREFILL, 1),
            ),
        )
        self.assertEqual(plan.shape.logical_m, 2)

    def test_api_concurrency_is_not_graph_logical_m(self) -> None:
        allocator = FakePageAllocator()
        scheduler = self.scheduler(allocator, budget=2, chunk=1, alignment=1)
        for request_id in ("a", "b", "c", "d", "e"):
            scheduler.admit(input_plan(request_id, 1), max_new_tokens=1)

        planned: list[str] = []
        while plan := scheduler.next_plan():
            self.assertLessEqual(plan.shape.logical_m, 2)
            planned.extend(plan.input_plan_ids)
            scheduler.complete_bundle(
                tuple(
                    SegmentCompletion(request_id, generated_token_id=10)
                    for request_id in plan.input_plan_ids
                )
            )

        self.assertEqual(planned, ["a", "b", "c", "d", "e"])

    def test_batch_plan_remains_immutable_after_scheduler_progress(self) -> None:
        scheduler = self.scheduler(FakePageAllocator(), budget=1, chunk=1)
        scheduler.admit(input_plan("a", 1), max_new_tokens=1)
        plan = scheduler.next_plan()
        assert plan is not None
        snapshot = plan.segments

        scheduler.complete("a", generated_token_id=10)
        with self.assertRaises(FrozenInstanceError):
            plan.plan_id = "changed"  # type: ignore[misc]
        self.assertIs(plan.segments, snapshot)

    def test_finish_during_inflight_suppresses_future_rows(self) -> None:
        scheduler = self.scheduler(FakePageAllocator(), budget=1, chunk=1)
        state = scheduler.admit(input_plan("a", 1), max_new_tokens=2)
        self.assertIsNotNone(scheduler.next_plan())

        scheduler.finish("a")
        scheduler.complete("a")
        self.assertTrue(state.finished)
        self.assertIsNone(scheduler.next_plan())

    def test_batch_plan_carries_configured_graph_capacity_identity(self) -> None:
        capacity = (("context_pages", 64), ("request_slots", 8))
        scheduler = ContinuousScheduler(
            page_allocator=FakePageAllocator(),
            token_budget=2,
            prefill_chunk_tokens=1,
            topology_id="single",
            partition_id="toy",
            owner_ranks=(0,),
            state_ids=(),
            state_commit_alignment_tokens=1,
            capacity_items=capacity,
        )
        scheduler.admit(input_plan("a", 1), max_new_tokens=1)
        plan = scheduler.next_plan()
        assert plan is not None
        self.assertEqual(plan.shape.capacity_items, capacity)

    def test_completion_bundle_rejects_one_invalid_member_without_partial_advance(self) -> None:
        allocator = FakePageAllocator()
        scheduler = self.scheduler(allocator, budget=2, chunk=1)
        first_state = scheduler.admit(input_plan("first", 1), max_new_tokens=1)
        second_state = scheduler.admit(input_plan("second", 2), max_new_tokens=1)
        plan = scheduler.next_plan()
        assert plan is not None
        first_inflight = first_state.inflight
        second_inflight = second_state.inflight

        with self.assertRaisesRegex(ValueError, "intermediate prefill"):
            scheduler.complete_bundle(
                (
                    SegmentCompletion("first", generated_token_id=10),
                    SegmentCompletion("second", generated_token_id=11),
                )
            )

        self.assertIs(first_state.inflight, first_inflight)
        self.assertIs(second_state.inflight, second_inflight)
        self.assertEqual((first_state.state_revision, second_state.state_revision), (0, 0))
        self.assertEqual(
            (first_state.generated_token_ids, second_state.generated_token_ids),
            ((), ()),
        )

        scheduler.complete_bundle(
            (
                SegmentCompletion("first", generated_token_id=10),
                SegmentCompletion("second"),
            )
        )
        self.assertEqual((first_state.state_revision, second_state.state_revision), (1, 1))
        self.assertEqual(first_state.generated_token_ids, (10,))
        self.assertEqual(second_state.generated_token_ids, ())

    def test_fair_rotation_batch_shrink_cancel_and_drain(self) -> None:
        allocator = FakePageAllocator()
        scheduler = self.scheduler(allocator, budget=2, chunk=1)
        scheduler.admit(input_plan("a", 2), max_new_tokens=1)
        scheduler.admit(input_plan("b", 2), max_new_tokens=1)
        scheduler.admit(input_plan("c", 2), max_new_tokens=1)

        first = scheduler.next_plan()
        assert first is not None
        self.assertEqual(first.input_plan_ids, ("a", "b"))
        scheduler.complete("a")
        scheduler.complete("b")
        scheduler.cancel("b")

        second = scheduler.next_plan()
        assert second is not None
        self.assertEqual(second.input_plan_ids, ("c", "a"))
        self.assertNotIn("b", second.input_plan_ids)
        scheduler.complete("c")
        scheduler.complete("a", generated_token_id=20)

        shrunk = scheduler.next_plan()
        assert shrunk is not None
        self.assertEqual(shrunk.input_plan_ids, ("c",))
        self.assertEqual(shrunk.shape.logical_m, 1)
        scheduler.complete("c", generated_token_id=21)
        self.assertIsNone(scheduler.next_plan())
        self.assertTrue(scheduler.drained)

    def test_cancelled_outstanding_request_never_produces_another_row(self) -> None:
        allocator = FakePageAllocator()
        scheduler = self.scheduler(allocator, budget=1, chunk=1)
        state = scheduler.admit(input_plan("a", 2), max_new_tokens=2)
        self.assertIsNotNone(scheduler.next_plan())

        scheduler.cancel("a")
        self.assertIsNone(scheduler.next_plan())
        scheduler.complete("a")
        self.assertFalse(state.decode_pending)
        self.assertEqual(state.generated_token_ids, ())
        self.assertTrue(scheduler.drained)

    def test_allocator_owns_page_boundary_mapping(self) -> None:
        allocator = FakePageAllocator(block_size_tokens=4)
        scheduler = self.scheduler(allocator, budget=3, chunk=3)
        scheduler.admit(input_plan("a", 6), max_new_tokens=1)
        first = scheduler.next_plan()
        assert first is not None
        scheduler.complete("a")
        second = scheduler.next_plan()
        assert second is not None
        scheduler.complete("a")
        third = scheduler.next_plan()
        assert third is not None

        self.assertEqual(allocator.calls, [("a", 0, 3), ("a", 3, 1), ("a", 4, 2)])
        self.assertEqual(third.segments[0].cache_slots, (4, 5))
        self.assertEqual(third.segments[0].cache_pages, (0, 1))

    def test_alignment_one_keeps_chunked_prefill_batch_width(self) -> None:
        allocator = FakePageAllocator(block_size_tokens=16)
        scheduler = self.scheduler(allocator, budget=16, chunk=16, alignment=1)
        scheduler.admit(input_plan("a", 11), max_new_tokens=1)
        scheduler.admit(input_plan("b", 7), max_new_tokens=1)

        plan = scheduler.next_plan()
        assert plan is not None
        self.assertEqual(plan.shape.logical_m, 16)
        self.assertEqual(
            tuple((segment.request_id, segment.token_count) for segment in plan.segments),
            (("a", 11), ("b", 5)),
        )

    def test_decode_reads_the_full_prefix_but_writes_only_current_page(self) -> None:
        allocator = FakePageAllocator(block_size_tokens=2)
        scheduler = self.scheduler(allocator, budget=4, chunk=4, alignment=2)
        scheduler.admit(input_plan("a", 4), max_new_tokens=2)
        prefill = scheduler.next_plan()
        assert prefill is not None
        scheduler.complete("a")
        prefill = scheduler.next_plan()
        assert prefill is not None
        scheduler.complete("a", generated_token_id=90)

        decode = scheduler.next_plan()
        assert decode is not None
        segment = decode.segments[0]
        txn = decode.state_transactions[0]
        self.assertEqual(segment.token_ids, (90,))
        self.assertEqual(segment.positions, (TokenPosition((4,)),))
        self.assertEqual(segment.cache_pages, (0, 1, 2))
        kv_reads = tuple(access for access in txn.reads if access.resource_id.startswith("kv:"))
        kv_writes = tuple(
            access for access in txn.writes if access.resource_id.startswith("kv:")
        )
        recurrent_reads = tuple(
            access for access in txn.reads if access.resource_id.startswith("state:recurrent:")
        )
        self.assertEqual({access.pages for access in kv_reads}, {(0, 1, 2)})
        self.assertEqual({access.pages for access in kv_writes}, {(2,)})
        self.assertEqual({access.owner_rank for access in recurrent_reads}, {0, 1})

    def test_prefix_publish_requires_complete_block_and_state_alignment(self) -> None:
        allocator = FakePageAllocator(block_size_tokens=4)
        scheduler = self.scheduler(allocator, budget=4, chunk=4, alignment=8)
        scheduler.admit(input_plan("a", 8), max_new_tokens=1)

        first = scheduler.next_plan()
        assert first is not None
        first_publication = first.state_transactions[0].prefix_publish
        self.assertIsNotNone(first_publication)
        assert first_publication is not None
        self.assertEqual(first_publication.state_commit_position, 0)
        self.assertEqual(first_publication.replay_token_count, 4)
        scheduler.complete("a")
        second = scheduler.next_plan()
        assert second is not None
        publication = second.state_transactions[0].prefix_publish
        self.assertIsNotNone(publication)
        assert publication is not None
        self.assertEqual(publication.prefix_token_count, 8)
        self.assertEqual(publication.state_commit_position, 8)

    def test_prefix_publication_bridges_block_and_state_boundaries(self) -> None:
        allocator = FakePageAllocator(block_size_tokens=4)
        scheduler = self.scheduler(allocator, budget=4, chunk=4, alignment=3)
        scheduler.admit(input_plan("a", 5), max_new_tokens=1)

        first = scheduler.next_plan()
        assert first is not None
        self.assertEqual((first.segments[0].token_start, first.segments[0].token_count), (0, 3))
        self.assertIsNone(first.state_transactions[0].prefix_publish)
        scheduler.complete("a")

        second = scheduler.next_plan()
        assert second is not None
        publication = second.state_transactions[0].prefix_publish
        self.assertIsNotNone(publication)
        assert publication is not None
        self.assertEqual(
            (
                publication.prefix_token_count,
                publication.state_commit_position,
                publication.replay_token_count,
            ),
            (4, 3, 1),
        )

    def test_each_segment_has_one_plan_only_transaction_with_explicit_owners(self) -> None:
        allocator = FakePageAllocator()
        scheduler = self.scheduler(allocator, budget=2, chunk=1)
        scheduler.admit(input_plan("a", 2), max_new_tokens=1, base_revision=7)
        scheduler.admit(input_plan("b", 2), max_new_tokens=1)
        plan = scheduler.next_plan()
        assert plan is not None

        self.assertEqual(len(plan.segments), len(plan.state_transactions))
        self.assertEqual(len({txn.transaction_id for txn in plan.state_transactions}), 2)
        first = plan.state_transactions[0]
        self.assertEqual((first.base_revision, first.target_revision), (7, 8))
        self.assertEqual(first.expected_owner_ranks, (0, 1))
        self.assertEqual(
            [(access.resource_id.split(":", 1)[0], access.owner_rank) for access in first.writes],
            [("kv", 0), ("state", 0), ("kv", 1), ("state", 1)],
        )
        self.assertEqual(
            [(access.resource_id.split(":", 1)[0], access.owner_rank) for access in first.reads],
            [("kv", 0), ("state", 0), ("kv", 1), ("state", 1)],
        )
        self.assertEqual(plan.decode_extension.provider, DecodeProvider.NONE)

    def test_non_none_decode_extension_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unavailable"):
            ContinuousScheduler(
                page_allocator=FakePageAllocator(),
                token_budget=1,
                prefill_chunk_tokens=1,
                topology_id="single",
                partition_id="toy",
                owner_ranks=(0,),
                state_ids=(),
                state_commit_alignment_tokens=1,
                decode_extension=DecodeExtensionMeta(
                    provider=DecodeProvider.MTP,
                    provider_version="v1",
                    max_proposal_tokens=1,
                    state_schema_id="mtp:1",
                ),
            )

    def test_prefill_budget_must_fit_total_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "prefill token budget"):
            ContinuousScheduler(
                page_allocator=FakePageAllocator(),
                token_budget=2,
                prefill_chunk_tokens=1,
                prefill_token_budget=3,
                topology_id="single",
                partition_id="toy",
                owner_ranks=(0,),
                state_ids=(),
                state_commit_alignment_tokens=1,
            )


if __name__ == "__main__":
    unittest.main()
