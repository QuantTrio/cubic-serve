from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from cubic_serve.runtime import performance as serving
from cubic_serve.runtime.cache import (
    CacheArenaSpec,
    DeviceMemorySnapshot,
    KVLayout,
    StateSlotSpec,
    estimate_cache_arena_bytes,
)


class Qwen3_5PerformanceCapacityTests(unittest.TestCase):
    def test_vram_capacity_display_uses_tightest_rank_without_gpu_queries(self) -> None:
        controller = serving._ProcessController.__new__(serving._ProcessController)
        controller._block_size = 32
        controller.rank_cache_capacities = (
            SimpleNamespace(vram_page_capacity_estimate=400_000),
            SimpleNamespace(vram_page_capacity_estimate=300_000),
        )
        with patch("torch.cuda.mem_get_info", side_effect=AssertionError("new GPU query")):
            self.assertEqual(controller.kv_vram_capacity_tokens, 9_600_000)
        controller.rank_cache_capacities = (
            SimpleNamespace(vram_page_capacity_estimate=400_000),
            SimpleNamespace(vram_page_capacity_estimate=None),
        )
        self.assertIsNone(controller.kv_vram_capacity_tokens)

    def setUp(self) -> None:
        self.layout = KVLayout((0, 2), 16, 2, 8)
        self.bootstrap_spec = CacheArenaSpec(
            "rank:0",
            page_capacity=16,
            state_slot_capacity=4,
            state_specs=(
                StateSlotSpec("conv", torch.bfloat16, (2, 8, 3)),
                StateSlotSpec("recurrent", torch.float32, (2, 2, 8, 8)),
            ),
            scratch_capacity=4,
            scratch_arena_ids=("graph",),
        )

    def test_post_capture_plan_credits_only_replaced_bootstrap_arena(self) -> None:
        snapshot = DeviceMemorySnapshot("cuda:0", 8 << 20, 32 << 20)
        plan, bootstrap_bytes = serving._plan_replacement_cache_capacity(
            snapshot=snapshot,
            layout=self.layout,
            bootstrap_spec=self.bootstrap_spec,
            state_slot_capacity=12,
            minimum_page_capacity=16,
            reserve_bytes=2 << 20,
        )

        self.assertEqual(
            bootstrap_bytes,
            estimate_cache_arena_bytes(self.layout, self.bootstrap_spec),
        )
        self.assertEqual(plan.snapshot.free_bytes, snapshot.free_bytes + bootstrap_bytes)
        self.assertEqual(plan.reserve_bytes, 2 << 20)
        self.assertGreaterEqual(plan.arena_spec.page_capacity, 16)
        self.assertEqual(plan.arena_spec.state_slot_capacity, 12)
        self.assertEqual(plan.arena_spec.scratch_capacity, 4)
        self.assertLessEqual(plan.backing_bytes, plan.budget_bytes)

    def test_post_capture_plan_fails_instead_of_shrinking_c4_context(self) -> None:
        with self.assertRaisesRegex(serving.Qwen3_5TP2PP2ServingError, "graph-pool context"):
            serving._plan_replacement_cache_capacity(
                snapshot=DeviceMemorySnapshot("cuda:0", 4 << 20, 16 << 20),
                layout=self.layout,
                bootstrap_spec=self.bootstrap_spec,
                state_slot_capacity=12,
                minimum_page_capacity=1_000_000,
                reserve_bytes=1 << 20,
            )

    def test_replacement_credit_cannot_exceed_physical_total(self) -> None:
        with self.assertRaisesRegex(
            serving.Qwen3_5TP2PP2ServingError, "exceeds the observed device"
        ):
            serving._plan_replacement_cache_capacity(
                snapshot=DeviceMemorySnapshot("cuda:0", (32 << 20) - 1, 32 << 20),
                layout=self.layout,
                bootstrap_spec=self.bootstrap_spec,
                state_slot_capacity=12,
                minimum_page_capacity=16,
                reserve_bytes=0,
            )


if __name__ == "__main__":
    unittest.main()
