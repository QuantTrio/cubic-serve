"""Atomic ownership for topology-wide CUDA Graph replay resources.

The pool is intentionally independent of CUDA.  It protects the static graph
instances, streams, input/output buffers, workspace arenas, and scratch arenas represented by the
execution contracts; the CUDA executor remains responsible for waiting for all
rank-local work before releasing a lease.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
from uuid import uuid4

from cubic_serve.contracts.execution import RankReplayResources, ReplayLease


class ReplayLeaseUnavailable(RuntimeError):
    """No complete cross-rank replay resource set is currently available."""


class ReplayLeaseOwnershipError(RuntimeError):
    """A release does not identify the exact active lease capability."""


@dataclass(frozen=True, slots=True)
class _ActiveLease:
    slot_index: int
    lease: ReplayLease


class ReplayLeasePool:
    """Thread-safe pool of indivisible, topology-wide replay resource sets.

    Each slot contains the resources for every participating rank.  ``acquire``
    moves exactly one complete slot to the in-flight set while holding one lock;
    there is no API for partial rank acquisition.  A process-unique pool nonce
    plus a monotonically increasing epoch prevents a released capability from
    being mistaken for a later owner of the same static resources.
    """

    def __init__(
        self,
        resource_sets: tuple[tuple[RankReplayResources, ...], ...],
    ) -> None:
        if not resource_sets:
            raise ValueError("replay lease pool requires at least one complete resource set")

        expected_ranks = tuple(resource.rank for resource in resource_sets[0])
        if not expected_ranks or tuple(sorted(set(expected_ranks))) != expected_ranks:
            raise ValueError("each replay resource set must have unique, sorted ranks")

        identities: dict[str, set[tuple[int, str]]] = {
            "graph instance": set(),
            "stream lane": set(),
            "buffer set": set(),
            "workspace arena": set(),
            "scratch arena": set(),
        }
        attributes = {
            "graph instance": "graph_instance_id",
            "stream lane": "stream_lane_id",
            "buffer set": "buffer_set_id",
            "workspace arena": "workspace_arena_id",
            "scratch arena": "scratch_arena_id",
        }
        for resources in resource_sets:
            ranks = tuple(resource.rank for resource in resources)
            if ranks != expected_ranks:
                raise ValueError("all replay resource sets must cover the same sorted ranks")
            for resource in resources:
                for label, attribute in attributes.items():
                    identity = (resource.rank, getattr(resource, attribute))
                    if identity in identities[label]:
                        raise ValueError(f"replay resource sets cannot share a rank-local {label}")
                    identities[label].add(identity)

        self._resource_sets = resource_sets
        self._available: deque[int] = deque(range(len(resource_sets)))
        self._active: dict[str, _ActiveLease] = {}
        self._epoch = 0
        self._pool_nonce = uuid4().hex
        self._lock = Lock()

    @property
    def capacity(self) -> int:
        return len(self._resource_sets)

    @property
    def available(self) -> int:
        with self._lock:
            return len(self._available)

    @property
    def in_flight(self) -> int:
        with self._lock:
            return len(self._active)

    def acquire(
        self,
        *,
        launch_plan_id: str,
        execution_plan_id: str,
        bucket_id: str,
        template_id: str,
        topology_id: str,
        partition_id: str,
    ) -> ReplayLease:
        """Atomically acquire one complete rank set and bind it to a launch."""

        with self._lock:
            if not self._available:
                raise ReplayLeaseUnavailable(
                    "no complete topology-wide replay resource set is available"
                )

            slot_index = self._available[0]
            epoch = self._epoch + 1
            lease = ReplayLease(
                lease_id=f"replay:{self._pool_nonce}:{epoch}",
                epoch=epoch,
                launch_plan_id=launch_plan_id,
                execution_plan_id=execution_plan_id,
                bucket_id=bucket_id,
                template_id=template_id,
                topology_id=topology_id,
                partition_id=partition_id,
                rank_resources=self._resource_sets[slot_index],
            )

            self._available.popleft()
            self._epoch = epoch
            self._active[lease.lease_id] = _ActiveLease(slot_index, lease)
            return lease

    def release(self, lease: ReplayLease) -> None:
        """Release only the exact active lease, including its epoch and bindings."""

        if not isinstance(lease, ReplayLease):
            raise TypeError("release requires a ReplayLease")
        with self._lock:
            active = self._active.get(lease.lease_id)
            if active is None:
                raise ReplayLeaseOwnershipError("replay lease is not active in this pool")
            if active.lease != lease:
                raise ReplayLeaseOwnershipError(
                    "replay lease identity, epoch, bindings, or resources do not match"
                )

            del self._active[lease.lease_id]
            self._available.append(active.slot_index)
