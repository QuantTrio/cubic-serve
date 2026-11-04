"""Byte-budgeted sizing for the fixed BF16 KV and recurrent-state arena.

The planner is deliberately independent of request context length and CUDA
Graph captured M.  It consumes a post-startup free-memory snapshot, reserves
explicit runtime headroom, fixes the recurrent-state concurrency envelope, and
uses every remaining byte for paged KV.
"""

from __future__ import annotations

from typing_extensions import TypeAliasType

from dataclasses import dataclass
from math import prod

import torch

from cubic_serve.runtime.cache.pages import CacheArenaSpec, KVLayout, StateSlotSpec


class CacheCapacityError(RuntimeError):
    """The requested fixed cache arena cannot fit its byte budget."""


@dataclass(frozen=True, slots=True)
class DeviceMemorySnapshot:
    """One rank's device memory after weights and fixed runtime allocations."""

    device: str
    free_bytes: int
    total_bytes: int

    def __post_init__(self) -> None:
        if type(self.device) is not str or not self.device:
            raise ValueError("device memory snapshot requires a device name")
        if type(self.free_bytes) is not int or type(self.total_bytes) is not int:
            raise TypeError("device memory byte counts must be integers")
        if self.free_bytes < 0 or self.total_bytes <= 0 or self.free_bytes > self.total_bytes:
            raise ValueError("device memory snapshot has invalid free/total bytes")


@dataclass(frozen=True, slots=True)
class DeviceMemoryBudget:
    """One application's byte envelope, witnessed before loading its weights.

    The initial snapshot excludes pre-existing allocations. The withheld bytes
    remain unavailable to all subsequent encoder, graph and cache admission.
    Reserve is inside the application limit; it covers unallocated runtime work.
    A budget is local to one physical device, never summed across TP ranks.
    """

    startup: DeviceMemorySnapshot
    limit_bytes: int | None = None
    reserve_bytes: int = 1 << 30

    def __post_init__(self) -> None:
        if not isinstance(self.startup, DeviceMemorySnapshot):
            raise TypeError("GPU budget requires a device memory snapshot")
        if type(self.reserve_bytes) is not int or self.reserve_bytes < 0:
            raise ValueError("GPU reserve must be non-negative integer bytes")
        if self.limit_bytes is not None and (
            type(self.limit_bytes) is not int or self.limit_bytes <= 0
        ):
            raise ValueError("GPU limit must be positive integer bytes or None")
        limit = self.startup.free_bytes if self.limit_bytes is None else self.limit_bytes
        if limit > self.startup.free_bytes:
            raise CacheCapacityError(
                f"GPU budget exceeds available memory on {self.startup.device}: "
                f"requested={limit / (1 << 30):.3f} GiB "
                f"available={self.startup.free_bytes / (1 << 30):.3f} GiB"
            )
        if self.reserve_bytes >= limit:
            raise CacheCapacityError("GPU reserve leaves no application allocation budget")

    @property
    def withheld_bytes(self) -> int:
        limit = self.startup.free_bytes if self.limit_bytes is None else self.limit_bytes
        return self.startup.free_bytes - limit + self.reserve_bytes

    def remaining_bytes(self, free_bytes: int) -> int:
        """Account all observed post-snapshot residents, including the frontend."""
        if type(free_bytes) is not int or free_bytes < 0:
            raise ValueError("current GPU free memory must be non-negative integer bytes")
        return max(0, min(free_bytes, self.startup.free_bytes) - self.withheld_bytes)

    def require_headroom(self, free_bytes: int, *, additional_bytes: int = 0) -> None:
        if type(additional_bytes) is not int or additional_bytes < 0:
            raise ValueError("additional GPU headroom must be non-negative integer bytes")
        if free_bytes < self.withheld_bytes + additional_bytes:
            raise CacheCapacityError(
                f"application GPU budget exhausted on {self.startup.device}: "
                f"free={free_bytes / (1 << 30):.3f} GiB "
                f"required_headroom={(self.withheld_bytes + additional_bytes) / (1 << 30):.3f} GiB"
            )

    def apply_allocator_limit(self) -> None:
        """Bound new Torch allocations in this process by the shared envelope."""
        snapshot = read_cuda_memory_snapshot(self.startup.device)
        self.require_headroom(snapshot.free_bytes)
        capacity = (torch.cuda.memory_reserved(self.startup.device)
                    + self.remaining_bytes(snapshot.free_bytes))
        # The CUDA allocator API accepts a ratio; the public policy stays bytes.
        torch.cuda.set_per_process_memory_fraction(
            min(capacity, snapshot.total_bytes) / snapshot.total_bytes,
            self.startup.device,
        )


StateSlotCounts = TypeAliasType("StateSlotCounts", tuple[tuple[str, int], ...])


@dataclass(frozen=True, slots=True)
class CacheCapacityUsage:
    """Projected physical occupancy at a prefix/COW admission boundary.

    ``resident_*`` counts unique physical slots, including slots retained by a
    prefix.  Prefix holds are annotations and therefore are not added again.
    ``cow_*`` counts tentative destinations, which coexist with their resident
    sources until commit and must be added at the peak.
    """

    resident_pages: int
    resident_state_slots: StateSlotCounts = ()
    prefix_retained_pages: int = 0
    prefix_retained_state_slots: StateSlotCounts = ()
    cow_pages: int = 0
    cow_state_slots: StateSlotCounts = ()

    def __post_init__(self) -> None:
        for label, value in (
            ("resident pages", self.resident_pages),
            ("prefix-retained pages", self.prefix_retained_pages),
            ("COW pages", self.cow_pages),
        ):
            if type(value) is not int:
                raise TypeError(f"{label} must be an integer")
            if value < 0:
                raise ValueError(f"{label} must be non-negative")
        if self.prefix_retained_pages > self.resident_pages:
            raise ValueError("prefix page holds must refer to resident physical pages")
        for label, counts in (
            ("resident state", self.resident_state_slots),
            ("prefix-retained state", self.prefix_retained_state_slots),
            ("COW state", self.cow_state_slots),
        ):
            _validate_state_counts(counts, label=label)
        resident = dict(self.resident_state_slots)
        for state_id, count in self.prefix_retained_state_slots:
            if count > resident.get(state_id, 0):
                raise ValueError(
                    f"prefix state holds for {state_id!r} must refer to resident slots"
                )


@dataclass(frozen=True, slots=True)
class CacheCapacityHeadroom:
    """Remaining owner slots after accounting for tentative COW destinations."""

    pages: int
    state_slots: StateSlotCounts

    @property
    def admissible(self) -> bool:
        return self.pages >= 0 and all(count >= 0 for _, count in self.state_slots)


@dataclass(frozen=True, slots=True)
class CacheCapacityPlan:
    """A tested byte plan that can be passed directly to :class:`CacheArena`."""

    snapshot: DeviceMemorySnapshot
    arena_spec: CacheArenaSpec
    reserve_bytes: int
    budget_bytes: int
    backing_bytes: int
    bytes_per_page: int
    auxiliary_bytes_per_page: int
    bytes_per_state_bundle: int
    scratch_slot_count: int
    allocation_alignment_bytes: int

    @property
    def unallocated_bytes(self) -> int:
        return (
            self.budget_bytes
            - self.backing_bytes
            - self.arena_spec.page_capacity * self.auxiliary_bytes_per_page
        )

    def headroom(self, usage: CacheCapacityUsage) -> CacheCapacityHeadroom:
        """Return physical headroom without double-charging retained prefixes."""

        if not isinstance(usage, CacheCapacityUsage):
            raise TypeError("cache capacity usage must be CacheCapacityUsage")
        state_ids = tuple(spec.state_id for spec in self.arena_spec.state_specs)
        known = set(state_ids)
        supplied = {
            state_id
            for counts in (usage.resident_state_slots, usage.cow_state_slots)
            for state_id, _ in counts
        }
        supplied.update(state_id for state_id, _ in usage.prefix_retained_state_slots)
        unknown = supplied - known
        if unknown:
            raise ValueError(f"capacity usage names unknown state ids: {sorted(unknown)!r}")

        resident = dict(usage.resident_state_slots)
        cow = dict(usage.cow_state_slots)
        return CacheCapacityHeadroom(
            pages=self.arena_spec.page_capacity - usage.resident_pages - usage.cow_pages,
            state_slots=tuple(
                (
                    state_id,
                    self.arena_spec.state_slot_capacity
                    - resident.get(state_id, 0)
                    - cow.get(state_id, 0),
                )
                for state_id in state_ids
            ),
        )

    def require_admissible(self, usage: CacheCapacityUsage) -> CacheCapacityHeadroom:
        headroom = self.headroom(usage)
        if not headroom.admissible:
            raise CacheCapacityError(
                "cache admission would exhaust resident plus tentative COW slots"
            )
        return headroom


def read_cuda_memory_snapshot(
    device: torch.device | str, *, initialize_cuda: bool = True,
) -> DeviceMemorySnapshot:
    """Read CUDA free memory without synchronizing or mutating allocator state.

    Startup budgets use ``initialize_cuda=False`` before loading weights, so a
    control process need not own a GPU context. Cache sizing uses the default
    worker query after fixed runtime allocations are resident. Future transient
    work belongs in the planner's explicit reserve.
    """

    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("cache capacity snapshots require a CUDA device")
    if not initialize_cuda:
        return _read_memory_without_context(resolved)
    free_bytes, total_bytes = torch.cuda.mem_get_info(resolved)
    return DeviceMemorySnapshot(str(resolved), int(free_bytes), int(total_bytes))


def _read_memory_without_context(device: torch.device) -> DeviceMemorySnapshot:
    """Resolve the CUDA-visible UUID, then query NVML without a primary context."""
    import ctypes
    from uuid import UUID
    from cuda.bindings import driver

    if device.index is None:
        raise ValueError("context-free memory queries require an explicit CUDA ordinal")

    def checked(result):
        if int(result[0]):
            raise CacheCapacityError(f"CUDA device query failed: {result[0]}")
        return result[1] if len(result) > 1 else None

    checked(driver.cuInit(0))
    ordinal = checked(driver.cuDeviceGet(device.index))
    uuid = str(UUID(bytes=checked(driver.cuDeviceGetUuid(ordinal)).bytes))
    total = int(checked(driver.cuDeviceTotalMem(ordinal)))
    nvml = ctypes.CDLL("libnvidia-ml.so.1")

    def require(status):
        if status:
            raise CacheCapacityError(f"NVML memory query failed on {device}: status={status}")

    class Memory(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in ("total", "free", "used")]

    nvml.nvmlDeviceGetHandleByUUID.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    nvml.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Memory)]
    require(nvml.nvmlInit_v2())
    try:
        handle = ctypes.c_void_p()
        # CUDA's UUID denotes the subscribed MIG instance when MIG is enabled.
        for prefix in ("GPU-", "MIG-"):
            status = nvml.nvmlDeviceGetHandleByUUID((prefix + uuid).encode(), ctypes.byref(handle))
            if status == 0:
                break
        require(status)
        memory = Memory()
        require(nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)))
        return DeviceMemorySnapshot(str(device), min(int(memory.free), total), total)
    finally:
        nvml.nvmlShutdown()


def estimate_cache_arena_bytes(
    layout: KVLayout,
    spec: CacheArenaSpec,
    *,
    allocation_alignment_bytes: int = 512,
) -> int:
    """Estimate the tensor payload requested by ``CacheArena``.

    Each backing tensor is rounded independently.  This mirrors the arena's
    key, value, written-mask, and per-state allocations without pretending to
    predict CUDA allocator segments; the caller's reserve covers that overhead.
    """

    if not isinstance(layout, KVLayout) or not isinstance(spec, CacheArenaSpec):
        raise TypeError("cache arena byte estimation requires KVLayout and CacheArenaSpec")
    _validate_alignment(allocation_alignment_bytes)
    return _arena_backing_bytes(
        layout=layout,
        page_capacity=spec.page_capacity,
        state_slot_capacity=spec.state_slot_capacity,
        state_specs=spec.state_specs,
        scratch_slot_count=spec.scratch_capacity * len(spec.scratch_arena_ids),
        alignment=allocation_alignment_bytes,
        state_scratch_slot_count=spec.state_scratch_capacity_per_arena*len(spec.scratch_arena_ids),
    )


def plan_cache_capacity(
    *,
    snapshot: DeviceMemorySnapshot,
    arena_id: str,
    layout: KVLayout,
    state_specs: tuple[StateSlotSpec, ...],
    state_slot_capacity: int,
    reserve_bytes: int,
    auxiliary_bytes_per_page: int = 0,
    scratch_capacity: int = 1,
    scratch_arena_ids: tuple[str, ...] = ("default",),
    allocation_alignment_bytes: int = 512,
) -> CacheCapacityPlan:
    """Fit the largest paged-KV arena after fixed GDN-state and scratch slabs.

    ``state_slot_capacity`` is an explicit scheduler/prefix concurrency
    envelope.  No model context limit or captured batch size participates in
    physical KV sizing.
    """

    if not isinstance(snapshot, DeviceMemorySnapshot):
        raise TypeError("cache capacity planning requires a DeviceMemorySnapshot")
    if not isinstance(layout, KVLayout):
        raise TypeError("cache capacity planning requires a KVLayout")
    if type(state_specs) is not tuple or any(
        not isinstance(spec, StateSlotSpec) for spec in state_specs
    ):
        raise TypeError("state specs must be a tuple of StateSlotSpec values")
    if (
        type(state_slot_capacity) is not int
        or type(reserve_bytes) is not int
        or type(auxiliary_bytes_per_page) is not int
    ):
        raise TypeError("state capacity and reserve bytes must be integers")
    if state_slot_capacity < 0 or reserve_bytes < 0 or auxiliary_bytes_per_page < 0:
        raise ValueError("state capacity and reserve bytes must be non-negative")
    if bool(state_specs) != (state_slot_capacity > 0):
        raise ValueError("state specs and positive state capacity must be provided together")
    _validate_alignment(allocation_alignment_bytes)

    # Validate arena-level ids and scratch geometry once, before the binary fit.
    probe_spec = CacheArenaSpec(
        arena_id=arena_id,
        page_capacity=1,
        state_slot_capacity=state_slot_capacity,
        state_specs=state_specs,
        scratch_capacity=scratch_capacity,
        scratch_arena_ids=scratch_arena_ids,
    )
    budget_bytes = snapshot.free_bytes - reserve_bytes
    if budget_bytes <= 0:
        raise CacheCapacityError("startup free memory does not exceed the runtime reserve")

    scratch_slot_count = scratch_capacity * len(scratch_arena_ids)
    bytes_per_page = _bytes_per_page(layout)
    bytes_per_state_bundle = sum(_state_spec_bytes(spec) for spec in state_specs)

    low = 0
    high = budget_bytes // (bytes_per_page + auxiliary_bytes_per_page) + 1
    while low + 1 < high:
        candidate = (low + high) // 2
        backing = _arena_backing_bytes(
            layout=layout,
            page_capacity=candidate,
            state_slot_capacity=state_slot_capacity,
            state_specs=state_specs,
            scratch_slot_count=scratch_slot_count,
            alignment=allocation_alignment_bytes,
        )
        if backing + candidate * auxiliary_bytes_per_page <= budget_bytes:
            low = candidate
        else:
            high = candidate
    if low == 0:
        raise CacheCapacityError(
            "startup memory cannot fit one owner KV page plus fixed state and scratch slabs"
        )

    arena_spec = CacheArenaSpec(
        arena_id=probe_spec.arena_id,
        page_capacity=low,
        state_slot_capacity=probe_spec.state_slot_capacity,
        state_specs=probe_spec.state_specs,
        scratch_capacity=probe_spec.scratch_capacity,
        scratch_arena_ids=probe_spec.scratch_arena_ids,
    )
    backing_bytes = estimate_cache_arena_bytes(
        layout,
        arena_spec,
        allocation_alignment_bytes=allocation_alignment_bytes,
    )
    return CacheCapacityPlan(
        snapshot=snapshot,
        arena_spec=arena_spec,
        reserve_bytes=reserve_bytes,
        budget_bytes=budget_bytes,
        backing_bytes=backing_bytes,
        bytes_per_page=bytes_per_page,
        auxiliary_bytes_per_page=auxiliary_bytes_per_page,
        bytes_per_state_bundle=bytes_per_state_bundle,
        scratch_slot_count=scratch_slot_count,
        allocation_alignment_bytes=allocation_alignment_bytes,
    )


def _validate_state_counts(counts: StateSlotCounts, *, label: str) -> None:
    if type(counts) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or not item[0]
        or type(item[1]) is not int
        or item[1] < 0
        for item in counts
    ):
        raise ValueError(f"{label} counts must be (state_id, non-negative integer) pairs")
    state_ids = tuple(state_id for state_id, _ in counts)
    if tuple(sorted(set(state_ids))) != state_ids:
        raise ValueError(f"{label} state ids must be unique and sorted")


def _validate_alignment(alignment: int) -> None:
    if type(alignment) is not int:
        raise TypeError("allocation alignment must be an integer")
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("allocation alignment must be a positive power of two")


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _bytes_per_page(layout: KVLayout) -> int:
    kv_bytes = len(layout.layer_ids) * (
        layout.key_encoding.payload_bytes + layout.value_encoding.payload_bytes
    )
    written_bytes = len(layout.layer_ids) * layout.block_size_tokens
    return (
        kv_bytes
        + written_bytes
        + sum(
            prod(spec.shape) * torch.empty((), dtype=spec.dtype).element_size()
            for spec in layout.page_fields
        )
    )


def _state_spec_bytes(spec: StateSlotSpec) -> int:
    element_bytes = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4, torch.int64: 8}[
        spec.dtype
    ]
    return prod(spec.shape) * element_bytes


def _arena_backing_bytes(
    *,
    layout: KVLayout,
    page_capacity: int,
    state_slot_capacity: int,
    state_specs: tuple[StateSlotSpec, ...],
    scratch_slot_count: int,
    alignment: int,
    state_scratch_slot_count: int | None = None,
) -> int:
    page_slots = page_capacity + scratch_slot_count
    key_bytes = page_slots * len(layout.layer_ids) * layout.key_encoding.payload_bytes
    value_bytes = page_slots * len(layout.layer_ids) * layout.value_encoding.payload_bytes
    written_bytes = page_slots * len(layout.layer_ids) * layout.block_size_tokens
    state_slots=state_slot_capacity+(scratch_slot_count if state_scratch_slot_count is None else state_scratch_slot_count)
    return (
        _align_up(key_bytes, alignment)
        + _align_up(value_bytes, alignment)
        + _align_up(written_bytes, alignment)
        + sum(
            _align_up(
                page_slots * prod(spec.shape) * torch.empty((), dtype=spec.dtype).element_size(),
                alignment,
            )
            for spec in layout.page_fields
        )
        + sum(_align_up(state_slots * _state_spec_bytes(spec), alignment) for spec in state_specs)
    )
