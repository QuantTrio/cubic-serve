from __future__ import annotations

import unittest
from unittest import mock

import torch

from cubic_serve.runtime.cache import (
    CacheArena,
    CacheCapacityError,
    CacheCapacityPlan,
    CacheCapacityUsage,
    DeviceMemoryBudget,
    DeviceMemorySnapshot,
    KVLayout,
    StateSlotSpec,
    estimate_cache_arena_bytes,
    plan_cache_capacity,
    read_cuda_memory_snapshot,
)


class CacheCapacityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = KVLayout((0, 2), 4, 2, 3)
        self.state_specs = (
            StateSlotSpec("conv:0", torch.bfloat16, (2, 3)),
            StateSlotSpec("recurrent:0", torch.float32, (2,)),
        )

    def test_absolute_gpu_budget_counts_encoder_and_language_residents_per_device(self):
        gib = 1 << 30
        budget = DeviceMemoryBudget(DeviceMemorySnapshot("cuda:0", 24*gib, 24*gib),
                                    limit_bytes=22*gib, reserve_bytes=gib)
        # Three GiB for the frontend and fourteen GiB for language weights.
        self.assertEqual(budget.remaining_bytes(7*gib), 4*gib)
        self.assertEqual(budget.withheld_bytes, 3*gib)
        budget.require_headroom(3*gib + 8, additional_bytes=8)
        with self.assertRaises(CacheCapacityError):
            budget.require_headroom(3*gib + 7, additional_bytes=8)
        # A peer's spare memory never increases this device's envelope.
        peer = DeviceMemoryBudget(DeviceMemorySnapshot("cuda:1", 24*gib, 24*gib),
                                  limit_bytes=22*gib, reserve_bytes=gib)
        self.assertEqual(peer.remaining_bytes(24*gib), 21*gib)
        self.assertEqual(budget.remaining_bytes(2*gib), 0)
        with self.assertRaises(CacheCapacityError):
            budget.require_headroom(2*gib)

    def test_gpu_budget_rejects_unavailable_capacity_before_loading(self):
        snapshot = DeviceMemorySnapshot("cuda:0", 20, 24)
        for limit, reserve in ((21, 1), (20, 20), (None, 20)):
            with self.subTest(limit=limit, reserve=reserve), self.assertRaises(CacheCapacityError):
                DeviceMemoryBudget(snapshot, limit, reserve)
        for limit in (0, -1, True, 1.0):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                DeviceMemoryBudget(snapshot, limit, 1)
        budget = DeviceMemoryBudget(snapshot, None, 2)
        self.assertEqual(budget.remaining_bytes(100), 18)

    def test_allocator_limit_credits_only_current_process_reserved_storage(self):
        from cubic_serve.runtime.cache import capacity

        budget = DeviceMemoryBudget(DeviceMemorySnapshot("cuda:0", 24, 24), 22, 1)
        with (
            mock.patch.object(capacity, "read_cuda_memory_snapshot",
                              return_value=DeviceMemorySnapshot("cuda:0", 18, 24)),
            mock.patch.object(capacity.torch.cuda, "memory_reserved", return_value=2),
            mock.patch.object(capacity.torch.cuda, "set_per_process_memory_fraction") as allocator,
        ):
            budget.apply_allocator_limit()
        # Four bytes in another process stay charged; this process owns two.
        allocator.assert_called_once_with(17/24, "cuda:0")

    def _plan(self, *, free_bytes: int = 2_000) -> CacheCapacityPlan:
        return plan_cache_capacity(
            snapshot=DeviceMemorySnapshot("cuda:0", free_bytes, 10_000),
            arena_id="rank:0",
            layout=self.layout,
            state_specs=self.state_specs,
            state_slot_capacity=2,
            reserve_bytes=160,
            scratch_capacity=1,
            scratch_arena_ids=("graph:0", "graph:1"),
            allocation_alignment_bytes=1,
        )

    def test_budget_fits_fixed_state_and_scratch_then_maximizes_pages(self) -> None:
        plan = self._plan()

        # Page = BF16 K/V (2 * 48 * 2 B) + 8-byte written mask = 200 B.
        # State bundle = BF16 conv (12 B) + FP32 recurrent (8 B) = 20 B.
        # Two scratch groups consume 2 * (200 + 20) = 440 B.
        self.assertEqual(plan.bytes_per_page, 200)
        self.assertEqual(plan.bytes_per_state_bundle, 20)
        self.assertEqual(plan.scratch_slot_count, 2)
        self.assertEqual(plan.budget_bytes, 1_840)
        self.assertEqual(plan.arena_spec.page_capacity, 6)
        self.assertEqual(plan.backing_bytes, 1_680)
        self.assertEqual(plan.unallocated_bytes, 160)

        next_spec = type(plan.arena_spec)(
            arena_id=plan.arena_spec.arena_id,
            page_capacity=7,
            state_slot_capacity=plan.arena_spec.state_slot_capacity,
            state_specs=plan.arena_spec.state_specs,
            scratch_capacity=plan.arena_spec.scratch_capacity,
            scratch_arena_ids=plan.arena_spec.scratch_arena_ids,
        )
        self.assertGreater(
            estimate_cache_arena_bytes(self.layout, next_spec, allocation_alignment_bytes=1),
            plan.budget_bytes,
        )

    def test_auxiliary_page_storage_shares_the_same_memory_budget(self) -> None:
        plan = plan_cache_capacity(
            snapshot=DeviceMemorySnapshot("cuda:0", 2_000, 10_000),
            arena_id="rank:0",
            layout=self.layout,
            state_specs=self.state_specs,
            state_slot_capacity=2,
            reserve_bytes=160,
            auxiliary_bytes_per_page=50,
            scratch_capacity=1,
            scratch_arena_ids=("graph:0", "graph:1"),
            allocation_alignment_bytes=1,
        )

        self.assertEqual(plan.arena_spec.page_capacity, 5)
        self.assertEqual(plan.auxiliary_bytes_per_page, 50)
        self.assertEqual(plan.unallocated_bytes, 110)

    def test_prefix_holds_do_not_double_charge_but_cow_destinations_do(self) -> None:
        plan = self._plan()
        usage = CacheCapacityUsage(
            resident_pages=5,
            resident_state_slots=(("conv:0", 1), ("recurrent:0", 2)),
            prefix_retained_pages=4,
            prefix_retained_state_slots=(("conv:0", 1), ("recurrent:0", 1)),
            cow_pages=1,
            cow_state_slots=(("conv:0", 1),),
        )
        headroom = plan.require_admissible(usage)
        self.assertEqual(headroom.pages, 0)
        self.assertEqual(headroom.state_slots, (("conv:0", 0), ("recurrent:0", 0)))

        exhausted = CacheCapacityUsage(
            resident_pages=6,
            resident_state_slots=(("conv:0", 2),),
            prefix_retained_pages=6,
            cow_pages=1,
        )
        with self.assertRaisesRegex(CacheCapacityError, "COW"):
            plan.require_admissible(exhausted)

    def test_planned_arena_reuses_prefix_refs_and_allocates_private_cow_slots(self) -> None:
        plan = self._plan()
        arena = CacheArena(plan.arena_spec, self.layout, "cpu", rank_id=0)

        source = arena.reserve_page(page_id=0, owner_rank=0, owner_id="source", zero=True)
        assert source.slot is not None
        prefix_hold = arena.retain_page(source.slot)
        destination = arena.reserve_page(page_id=1, owner_rank=0, owner_id="destination", zero=True)
        assert destination.slot is not None
        self.assertEqual(prefix_hold.index, source.slot.index)
        self.assertNotEqual(destination.slot.index, source.slot.index)

        arena.release_page(source)
        arena.validate_page_ref(prefix_hold)
        arena.release_page_ref(prefix_hold)
        arena.release_page(destination)

    def test_alignment_is_applied_per_backing_tensor(self) -> None:
        plan = plan_cache_capacity(
            snapshot=DeviceMemorySnapshot("cuda:0", 8_192, 16_384),
            arena_id="aligned",
            layout=self.layout,
            state_specs=self.state_specs,
            state_slot_capacity=1,
            reserve_bytes=0,
            allocation_alignment_bytes=512,
        )
        self.assertEqual(plan.backing_bytes % 512, 0)
        self.assertLessEqual(plan.backing_bytes, plan.budget_bytes)

    def test_insufficient_budget_fails_without_shrinking_fixed_state_capacity(self) -> None:
        with self.assertRaisesRegex(CacheCapacityError, "one owner KV page"):
            self._plan(free_bytes=700)

    def test_cuda_snapshot_is_read_only_and_uses_reported_free_memory(self) -> None:
        with mock.patch.object(torch.cuda, "mem_get_info", return_value=(1_234, 5_678)) as get:
            snapshot = read_cuda_memory_snapshot("cuda:3")
        self.assertEqual(snapshot, DeviceMemorySnapshot("cuda:3", 1_234, 5_678))
        get.assert_called_once_with(torch.device("cuda:3"))

    def test_prefix_annotations_must_name_resident_physical_slots(self) -> None:
        with self.assertRaisesRegex(ValueError, "resident physical pages"):
            CacheCapacityUsage(resident_pages=1, prefix_retained_pages=2)
        with self.assertRaisesRegex(ValueError, "resident slots"):
            CacheCapacityUsage(
                resident_pages=0,
                resident_state_slots=(("conv:0", 1),),
                prefix_retained_state_slots=(("conv:0", 2),),
            )


if __name__ == "__main__":
    unittest.main()


def test_token_history_and_fp16_state_capacity_match_physical_storage():
    from cubic_serve.runtime.cache import CacheArenaSpec

    layout = KVLayout((0,), 4, 1, 8)
    states = tuple(
        StateSlotSpec(name, dtype, (3,))
        for name, dtype in (
            ("a", torch.float16),
            ("b", torch.bfloat16),
            ("c", torch.float32),
            ("d", torch.int64),
        )
    )
    spec = CacheArenaSpec("typed-state", 2, 3, states)
    arena = CacheArena(spec, layout, "cpu", rank_id=0)
    actual = sum(
        t.numel() * t.element_size()
        for t in (arena.key, arena.value, arena.written, *arena.states.values())
    )
    assert (
        estimate_cache_arena_bytes(spec=spec, layout=layout, allocation_alignment_bytes=1) == actual
    )
    history = torch.tensor([2**54 + 1, 248044, 17], dtype=torch.int64)
    slot, value = arena.reserve_state("d", source=history)
    assert torch.equal(value, history)
    retained = arena.retain_state("d", slot)
    arena.release_state("d", slot)
    arena.validate_state_ref("d", retained)
    assert torch.equal(value, history)
    arena.release_state_ref("d", retained)


def test_separate_physical_state_padding_is_priced_from_actual_schema():
    from cubic_serve.runtime.cache import CacheArena,CacheArenaSpec,KVLayout,StateSlotSpec,estimate_cache_arena_bytes
    layout=KVLayout((0,),8,1,16)
    spec=CacheArenaSpec('padding',3,4,(StateSlotSpec('matrix',torch.bfloat16,(16,16)),),
                        scratch_capacity=128,state_scratch_capacity=1)
    arena=CacheArena(spec,layout,'cpu',rank_id=0)
    assert arena.states['matrix'].shape[0]==5
    assert estimate_cache_arena_bytes(layout,spec,allocation_alignment_bytes=1)==arena.backed_bytes
