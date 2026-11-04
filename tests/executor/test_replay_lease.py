from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

from cubic_serve.contracts.execution import RankReplayResources, ReplayLease
from cubic_serve.executor.replay_lease import (
    ReplayLeaseOwnershipError,
    ReplayLeasePool,
    ReplayLeaseUnavailable,
)


def _resource_set(slot: int, ranks: tuple[int, ...] = (0, 1)) -> tuple[RankReplayResources, ...]:
    return tuple(
        RankReplayResources(
            rank=rank,
            graph_instance_id=f"graph:{slot}:{rank}",
            stream_lane_id=f"stream:{slot}:{rank}",
            buffer_set_id=f"buffers:{slot}:{rank}",
            workspace_arena_id=f"workspace:{slot}:{rank}",
            scratch_arena_id=f"scratch:{slot}:{rank}",
        )
        for rank in ranks
    )


def _acquire(pool: ReplayLeasePool, launch: str = "launch:0") -> ReplayLease:
    return pool.acquire(
        launch_plan_id=launch,
        execution_plan_id="execution:0",
        bucket_id="decode:m8",
        template_id="qwen3_5:decode",
        topology_id="tp2:pp1",
        partition_id="partition:0",
    )


class ReplayLeasePoolTests(unittest.TestCase):
    def test_acquire_owns_one_complete_cross_rank_set(self) -> None:
        first_resources = _resource_set(0)
        second_resources = _resource_set(1)
        pool = ReplayLeasePool((first_resources, second_resources))

        first = _acquire(pool, "launch:first")
        second = _acquire(pool, "launch:second")

        self.assertEqual(first.rank_resources, first_resources)
        self.assertEqual(second.rank_resources, second_resources)
        self.assertEqual(tuple(resource.rank for resource in first.rank_resources), (0, 1))
        self.assertEqual(pool.available, 0)
        self.assertEqual(pool.in_flight, 2)

    def test_full_pool_rejects_without_partial_rank_ownership(self) -> None:
        resources = _resource_set(0, (0, 1, 2, 3))
        pool = ReplayLeasePool((resources,))
        barrier = Barrier(2)

        def contend(launch: str) -> ReplayLease | None:
            barrier.wait()
            try:
                return _acquire(pool, launch)
            except ReplayLeaseUnavailable:
                return None

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = tuple(executor.map(contend, ("launch:a", "launch:b")))

        leases = tuple(result for result in results if result is not None)
        self.assertEqual(len(leases), 1)
        self.assertEqual(leases[0].rank_resources, resources)
        self.assertEqual(pool.available, 0)
        self.assertEqual(pool.in_flight, 1)
        with self.assertRaisesRegex(ReplayLeaseUnavailable, "complete topology-wide"):
            _acquire(pool, "launch:rejected")

    def test_release_reuses_resources_with_new_epoch_and_lease_id(self) -> None:
        pool = ReplayLeasePool((_resource_set(0),))
        first = _acquire(pool, "launch:first")
        pool.release(first)

        second = _acquire(pool, "launch:second")

        self.assertGreater(second.epoch, first.epoch)
        self.assertNotEqual(second.lease_id, first.lease_id)
        self.assertEqual(second.rank_resources, first.rank_resources)
        self.assertEqual(second.launch_plan_id, "launch:second")

    def test_stale_or_forged_lease_cannot_release_current_owner(self) -> None:
        pool = ReplayLeasePool((_resource_set(0),))
        first = _acquire(pool, "launch:first")
        pool.release(first)
        current = _acquire(pool, "launch:current")

        with self.assertRaisesRegex(ReplayLeaseOwnershipError, "not active"):
            pool.release(first)
        forged = replace(current, epoch=current.epoch + 1)
        with self.assertRaisesRegex(ReplayLeaseOwnershipError, "do not match"):
            pool.release(forged)

        self.assertEqual(pool.in_flight, 1)
        pool.release(current)
        with self.assertRaisesRegex(ReplayLeaseOwnershipError, "not active"):
            pool.release(current)

    def test_foreign_pool_lease_cannot_release_local_resources(self) -> None:
        local = ReplayLeasePool((_resource_set(0),))
        foreign = ReplayLeasePool((_resource_set(1),))
        local_lease = _acquire(local, "launch:local")
        foreign_lease = _acquire(foreign, "launch:foreign")

        with self.assertRaisesRegex(ReplayLeaseOwnershipError, "not active"):
            local.release(foreign_lease)
        self.assertEqual(local.in_flight, 1)
        local.release(local_lease)

    def test_pool_rejects_partial_or_overlapping_resource_sets(self) -> None:
        with self.assertRaisesRegex(ValueError, "same sorted ranks"):
            ReplayLeasePool((_resource_set(0, (0, 1)), _resource_set(1, (0,))))

        shared_graph = replace(_resource_set(1)[0], graph_instance_id="graph:0:0")
        with self.assertRaisesRegex(ValueError, "share a rank-local graph instance"):
            ReplayLeasePool((_resource_set(0), (shared_graph, _resource_set(1)[1])))

        shared_workspace = replace(_resource_set(1)[0], workspace_arena_id="workspace:0:0")
        with self.assertRaisesRegex(ValueError, "share a rank-local workspace arena"):
            ReplayLeasePool((_resource_set(0), (shared_workspace, _resource_set(1)[1])))


if __name__ == "__main__":
    unittest.main()
