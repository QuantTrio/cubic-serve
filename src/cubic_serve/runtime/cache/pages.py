"""Logical paged attention geometry and format-owned physical storage.

One page stores two tensors with the exact axis order
``[layer, token_in_block, kv_head, head_dim]``.  The page id is the block
coordinate, so ``(page_id, layer_id, token_offset, head, dim)`` names every
element without an implicit layout convention.
"""

from __future__ import annotations

from typing_extensions import TypeAliasType

from dataclasses import dataclass, field, replace
from threading import RLock

import torch

from .quantization import CacheGroupAxis, GroupedCacheLayout, KVCacheStorageSpec


class CacheStateError(RuntimeError):
    """Base class for fail-closed cache/state failures."""


class ArenaExhausted(CacheStateError):
    """The fixed arena has no unreserved physical slot for an operation."""


class StaleArenaSlot(CacheStateError):
    """A physical slot reference names an earlier allocation generation."""


class ArenaSlotReleaseError(CacheStateError):
    """A physical slot was released twice or was never allocated."""


class ArenaPoisoned(CacheStateError):
    """The fixed arena is quarantined after an allocator invariant failure."""


class ScratchArenaLeaseError(CacheStateError):
    """Scratch ownership is missing, foreign, stale, or already active."""


@dataclass(frozen=True, slots=True)
class PageFieldSpec:
    """One typed payload sharing the main KV page's allocation and ownership."""

    field_id: str
    dtype: torch.dtype
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.field_id) is not str or not self.field_id:
            raise ValueError("page field requires a nonempty id")
        if self.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.int32,
            torch.int64,
            torch.bool,
        ):
            raise ValueError(
                "page field dtype must be a supported floating, integer or boolean type"
            )
        if (
            type(self.shape) is not tuple
            or not self.shape
            or any(type(size) is not int or size <= 0 for size in self.shape)
        ):
            raise ValueError("page field shape requires positive integer dimensions")

    def validate(
        self, value: torch.Tensor, device: torch.device, *, pages: int | None = None
    ) -> None:
        shape = self.shape if pages is None else (pages, *self.shape)
        if (
            value.shape != shape
            or value.dtype != self.dtype
            or value.device != device
            or not value.is_contiguous()
        ):
            raise ValueError(f"page field {self.field_id!r} differs from its declared payload")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"page field {self.field_id!r} contains nonfinite values")


@dataclass(frozen=True, slots=True)
class KVLayout:
    """Frozen per-rank KV page geometry.

    ``layer_ids`` are model layer ids owned by this store.  They need not be a
    dense range, which keeps PP and hybrid full/linear-attention partitioning
    explicit.
    """

    layer_ids: tuple[int, ...]
    block_size_tokens: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype = torch.bfloat16
    page_fields: tuple[PageFieldSpec, ...] = ()
    storage: KVCacheStorageSpec = field(default_factory=KVCacheStorageSpec)

    def __post_init__(self) -> None:
        if type(self.page_fields) is not tuple or any(
            not isinstance(spec, PageFieldSpec) for spec in self.page_fields
        ):
            raise TypeError("page fields must be a tuple of typed specs")
        names = tuple(spec.field_id for spec in self.page_fields)
        if tuple(sorted(set(names))) != names:
            raise ValueError("page fields require unique sorted ids")
        if not self.layer_ids:
            raise ValueError("KV layout requires at least one full-attention layer")
        if tuple(sorted(set(self.layer_ids))) != self.layer_ids:
            raise ValueError("KV layer ids must be non-negative, unique, and sorted")
        if self.layer_ids[0] < 0:
            raise ValueError("KV layer ids must be non-negative")
        if min(self.block_size_tokens, self.num_kv_heads, self.head_dim) <= 0:
            raise ValueError("KV block size, head count, and head dim must be positive")
        if self.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("attention KV computation requires FP16 or BF16")
        if not isinstance(self.storage, KVCacheStorageSpec):
            raise TypeError("KV layout requires a typed cache storage spec")
        object.__setattr__(self, "storage", self.storage.resolve(self.dtype))
        fields = {spec.field_id: spec for spec in self.page_fields}
        for name, grouped in (("key", self.key_encoding), ("value", self.value_encoding)):
            if not grouped.spec.dtype.quantized:
                continue
            if (grouped.spec.axis is CacheGroupAxis.TOKENS
                    and self.block_size_tokens % grouped.spec.group_size):
                raise ValueError("column cache groups must divide the physical page size")
            scale_spec = PageFieldSpec(
                f"kv_{name}_scales", torch.float32,
                (len(self.layer_ids), grouped.group_count),
            )
            if scale_spec.field_id in fields and fields[scale_spec.field_id] != scale_spec:
                raise ValueError("cache scale field differs from its encoding geometry")
            fields[scale_spec.field_id] = scale_spec
        object.__setattr__(self, "page_fields", tuple(fields[name] for name in sorted(fields)))

    @property
    def axis_order(self) -> tuple[str, ...]:
        return ("layer", "token_in_block", "kv_head", "head_dim")

    @property
    def page_shape(self) -> tuple[int, int, int, int]:
        return (
            len(self.layer_ids),
            self.block_size_tokens,
            self.num_kv_heads,
            self.head_dim,
        )

    @property
    def key_encoding(self) -> GroupedCacheLayout:
        return GroupedCacheLayout(self.storage.key, self.block_size_tokens,
                                  self.num_kv_heads, self.head_dim)

    @property
    def value_encoding(self) -> GroupedCacheLayout:
        return GroupedCacheLayout(self.storage.value, self.block_size_tokens,
                                  self.num_kv_heads, self.head_dim)

    def storage_shape(self, *, key: bool) -> tuple[int, ...]:
        grouped = self.key_encoding if key else self.value_encoding
        if not grouped.spec.dtype.quantized:
            return self.page_shape
        return (len(self.layer_ids), grouped.group_count, grouped.packed_group_bytes)

    def storage_dtype(self, *, key: bool) -> torch.dtype:
        grouped = self.key_encoding if key else self.value_encoding
        if grouped.spec.dtype.quantized:
            return torch.uint8
        return {"bfloat16": torch.bfloat16, "float16": torch.float16}[grouped.spec.dtype.value]

    @property
    def quantization_state_specs(self) -> tuple[StateSlotSpec, ...]:
        """Column tails use the common owner/checkpoint state allocator."""
        return tuple(
            StateSlotSpec(f"kv_{name}_tail", self.dtype,
                          (len(self.layer_ids), spec.group_size,
                           self.num_kv_heads, self.head_dim))
            for name, spec in (("key", self.storage.key), ("value", self.storage.value))
            if spec.dtype.quantized and spec.axis is CacheGroupAxis.TOKENS
        )

    def payload_shape(self, token_count: int) -> tuple[int, int, int]:
        if token_count <= 0:
            raise ValueError("a KV payload must contain at least one token")
        return (token_count, self.num_kv_heads, self.head_dim)

    def complete_group_mask(self, written: torch.Tensor, *, key: bool) -> torch.Tensor:
        """Closed code groups witnessed by logical token ownership."""
        grouped = self.key_encoding if key else self.value_encoding
        layers = len(self.layer_ids)
        if grouped.spec.axis is CacheGroupAxis.TOKENS:
            closed = written.reshape(layers, -1, grouped.spec.group_size).all(dim=-1)
            return closed[:, None, None, :].expand(
                layers, self.num_kv_heads, self.head_dim, -1).reshape(layers, grouped.group_count)
        return written[:, :, None, None].expand(
            layers, self.block_size_tokens, self.num_kv_heads,
            grouped.group_shape[-1]).reshape(layers, grouped.group_count)

    def storage_token_mask(self, offsets: tuple[int, ...], *, key: bool,
                           device: torch.device | str = "cpu") -> torch.Tensor:
        """Per-layer byte/element write authority, separate from logical rows."""
        grouped = self.key_encoding if key else self.value_encoding
        if not grouped.spec.dtype.quantized:
            if (type(offsets) is not tuple or any(type(t) is not int for t in offsets)
                    or tuple(sorted(set(offsets))) != offsets
                    or any(not 0 <= t < self.block_size_tokens for t in offsets)):
                raise ValueError("cache write offsets differ from the logical page")
            mask = torch.zeros(grouped.logical_shape, dtype=torch.bool, device=device)
            if offsets:
                mask[list(offsets)] = True
            return mask
        groups = grouped.groups_for_tokens(offsets, complete=True)
        mask = torch.zeros(grouped.group_count, dtype=torch.bool, device=device)
        if groups:
            mask[list(groups)] = True
        return mask[:, None].expand(-1, grouped.packed_group_bytes)

    def validate_page_storage(self, key: torch.Tensor, value: torch.Tensor,
                              written: torch.Tensor, page_fields: dict[str, torch.Tensor], *,
                              device: torch.device | str, require_full: bool = False) -> None:
        """Validate realized physical storage; never run inside a forward observer."""
        device = torch.device(device)
        if (device.type == "meta" or written.device != device or written.dtype is not torch.bool
                or tuple(written.shape) != (len(self.layer_ids), self.block_size_tokens)
                or not written.is_contiguous()):
            raise ValueError("KV written mask differs from its logical page")
        if tuple(page_fields) != tuple(spec.field_id for spec in self.page_fields):
            raise ValueError("KV page fields differ from their storage schema")
        for spec in self.page_fields:
            spec.validate(page_fields[spec.field_id], device)
        if require_full and not bool(written.all()):
            raise ValueError("only fully written KV pages may enter a prefix snapshot")
        for is_key, name, data, grouped in (
            (True, "key", key, self.key_encoding), (False, "value", value, self.value_encoding)
        ):
            if (data.device != device or data.dtype != self.storage_dtype(key=is_key)
                    or tuple(data.shape) != self.storage_shape(key=is_key)
                    or not data.is_contiguous()):
                raise ValueError("KV physical payload differs from its declared storage layout")
            if grouped.spec.dtype.quantized:
                complete = self.complete_group_mask(written, key=is_key)
                scale = page_fields[f"kv_{name}_scales"]
                if (bool((scale < 0).any()) or bool((complete & (scale <= 0)).any())
                        or bool((complete & ~grouped.valid_code_groups(data)).any())):
                    raise ValueError("written KV groups require valid codes and positive scales")
            elif not bool(torch.isfinite(data).all()):
                raise ValueError("KV physical payload contains nonfinite values")

    def layer_index(self, layer_id: int) -> int:
        try:
            return self.layer_ids.index(layer_id)
        except ValueError as error:
            raise ValueError(f"layer {layer_id} is not owned by this KV layout") from error


@dataclass(frozen=True, slots=True)
class StateSlotSpec:
    """One schema-declared floating state or exact token-history slab."""

    state_id: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    initial_value: int = 0

    def __post_init__(self) -> None:
        if type(self.state_id) is not str or not self.state_id:
            raise ValueError("state slot spec requires an id")
        if self.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.int64):
            raise ValueError("state slabs require FP16/BF16/FP32 values or INT64 token history")
        if (
            type(self.shape) is not tuple
            or not self.shape
            or any(type(size) is not int or size <= 0 for size in self.shape)
        ):
            raise ValueError("state slot shape must be a non-empty tuple of positive integers")
        if type(self.initial_value) is not int:
            raise TypeError("state initial value must be an exact integer scalar")
        if self.dtype is torch.int64:
            if not -(2**63) <= self.initial_value < 2**63:
                raise ValueError("initial token value exceeds INT64")
        elif abs(self.initial_value) > torch.finfo(self.dtype).max:
            raise ValueError("initial state value exceeds its floating storage range")


@dataclass(frozen=True, slots=True)
class CacheArenaSpec:
    """Owner capacity plus lease-private ownerless graph-padding groups."""

    arena_id: str
    page_capacity: int
    state_slot_capacity: int
    state_specs: tuple[StateSlotSpec, ...]
    scratch_capacity: int = 1
    scratch_arena_ids: tuple[str, ...] = ("default",)
    # Storage extent stays fixed while pre-readiness physical admission grows.
    reserved_page_capacity: int = 0
    # Optional physical padding-state count, independent of graph row capacity.
    state_scratch_capacity: int | None = None
    page_scratch_capacity: int | None = None

    @property
    def page_scratch_capacity_per_arena(self) -> int:
        return self.scratch_capacity if self.page_scratch_capacity is None else self.page_scratch_capacity

    @property
    def state_scratch_capacity_per_arena(self) -> int:
        return self.scratch_capacity if self.state_scratch_capacity is None else self.state_scratch_capacity

    @property
    def storage_page_capacity(self) -> int:
        return self.reserved_page_capacity or self.page_capacity

    def __post_init__(self) -> None:
        if type(self.arena_id) is not str or not self.arena_id:
            raise ValueError("cache arena requires an id")
        if any(
            type(value) is not int
            for value in (self.page_capacity, self.state_slot_capacity, self.scratch_capacity)
        ):
            raise TypeError("cache arena capacities must be integers")
        if min(self.page_capacity, self.scratch_capacity) <= 0 or self.state_slot_capacity < 0:
            raise ValueError("owner/scratch capacities are invalid")
        if self.state_scratch_capacity is not None and (
                type(self.state_scratch_capacity) is not int or self.state_scratch_capacity <= 0):
            raise ValueError("physical state scratch capacity must be positive")
        if self.page_scratch_capacity is not None and (
                type(self.page_scratch_capacity) is not int or self.page_scratch_capacity <= 0):
            raise ValueError("physical page scratch capacity must be positive")
        if (type(self.reserved_page_capacity) is not int or self.reserved_page_capacity < 0
                or (self.reserved_page_capacity and self.reserved_page_capacity < self.page_capacity)):
            raise ValueError("reserved page extent cannot be smaller than admitted capacity")
        if type(self.state_specs) is not tuple or any(
            not isinstance(spec, StateSlotSpec) for spec in self.state_specs
        ):
            raise TypeError("cache arena state specs must be a tuple of StateSlotSpec values")
        state_ids = tuple(spec.state_id for spec in self.state_specs)
        if tuple(sorted(set(state_ids))) != state_ids:
            raise ValueError("cache arena state specs must have unique sorted ids")
        if self.state_specs and self.state_slot_capacity == 0:
            raise ValueError("cache arena state schemas require owner-mapped state slots")
        if (
            type(self.scratch_arena_ids) is not tuple
            or not self.scratch_arena_ids
            or any(type(arena_id) is not str or not arena_id for arena_id in self.scratch_arena_ids)
            or tuple(sorted(set(self.scratch_arena_ids))) != self.scratch_arena_ids
        ):
            raise ValueError("scratch arena ids must be non-empty, unique, and sorted")


CacheArenaDomain = TypeAliasType("CacheArenaDomain", tuple[int, KVLayout, CacheArenaSpec, torch.device])


@dataclass(frozen=True, slots=True)
class ArenaSlotRef:
    index: int
    generation: int
    hold_id: int


_ArenaSlot = ArenaSlotRef


@dataclass(frozen=True, slots=True)
class ArenaReadSlot:
    """Generation-guarded committed physical source for one logical resource."""

    resource_id: int | str
    source_slot: int
    source_generation: int
    source_revision: int


@dataclass(frozen=True, slots=True)
class ArenaCOWSlots:
    """One logical resource's committed source and tentative destination."""

    resource_id: int | str
    source_slot: int
    source_generation: int
    source_revision: int
    destination_slot: int
    destination_generation: int
    destination_revision: int


@dataclass(frozen=True, slots=True)
class _ScratchLeaseClaim:
    scratch_arena_id: str
    lease_id: str
    lease_epoch: int
    launch_plan_id: str
    resource_identity: tuple[str, str, str, str]


@dataclass(frozen=True, slots=True)
class _ScratchParticipantHold:
    claim: _ScratchLeaseClaim
    transaction_id: str


@dataclass(slots=True)
class _ActiveScratchLease:
    claim: _ScratchLeaseClaim
    participants: set[str]


@dataclass(frozen=True, slots=True)
class _ArenaReleaseBatch:
    page_slots: tuple[_ArenaSlot, ...]
    state_slots: tuple[tuple[str, _ArenaSlot], ...]
    scratch_holds: tuple[_ScratchParticipantHold, ...]


@dataclass(frozen=True, slots=True, eq=False)
class CacheArenaBinding:
    """Stable backing tensors, tentative physical indices, and ownerless scratch."""

    arena_id: str
    scratch_arena_id: str
    scratch_lease_id: str
    scratch_lease_epoch: int
    scratch_launch_plan_id: str
    read_page_slots: tuple[ArenaReadSlot, ...]
    read_state_slots: tuple[ArenaReadSlot, ...]
    page_slots: tuple[ArenaCOWSlots, ...]
    state_slots: tuple[ArenaCOWSlots, ...]
    key: torch.Tensor
    value: torch.Tensor
    written: torch.Tensor
    state_tensors: tuple[tuple[str, torch.Tensor], ...]
    scratch_page_slots: tuple[int, ...]
    scratch_state_slots: tuple[tuple[str, tuple[int, ...]], ...]
    page_fields: tuple[tuple[str, torch.Tensor], ...] = ()


class _ArenaSlotPool:
    def __init__(self, capacity: int, *, label: str, shared_slots=None) -> None:
        self.label = label
        self._free = list(range(capacity - 1, -1, -1)) if shared_slots is None else []
        self._shared_slots = shared_slots
        self._generations = [0] * capacity
        self._holds: list[set[int] | None] = [None] * capacity
        self._next_hold_id = 1
        self._lock = RLock()

    def reserve(self) -> _ArenaSlot:
        with self._lock:
            if not (self._free if self._shared_slots is None else len(self._shared_slots)):
                raise ArenaExhausted(f"cache arena exhausted its {self.label} slots")
            index = self._free.pop() if self._shared_slots is None else self._shared_slots.allocate()
            self._generations[index] += 1
            hold_id = self._mint_hold()
            self._holds[index] = {hold_id}
            return _ArenaSlot(index, self._generations[index], hold_id)

    def retain(self, slot: _ArenaSlot) -> _ArenaSlot:
        with self._lock:
            self._validate_locked(slot)
            hold_id = self._mint_hold()
            assert self._holds[slot.index] is not None
            self._holds[slot.index].add(hold_id)
            return _ArenaSlot(slot.index, slot.generation, hold_id)

    def validate(self, slot: _ArenaSlot) -> None:
        with self._lock:
            self._validate_locked(slot)

    def _validate_locked(self, slot: _ArenaSlot) -> None:
        if not 0 <= slot.index < len(self._generations):
            raise StaleArenaSlot(f"{self.label} slot is outside the owner-mapped arena")
        current = self._generations[slot.index]
        if current != slot.generation:
            raise StaleArenaSlot(
                f"stale {self.label} slot generation {slot.generation}; current is {current}"
            )
        if slot.hold_id not in (self._holds[slot.index] or ()):
            raise ArenaSlotReleaseError(f"{self.label} hold is stale, foreign, or already released")

    def release(self, slot: _ArenaSlot) -> None:
        self.release_prevalidated((slot,))

    def validate_batch(self, slots: tuple[_ArenaSlot, ...]) -> None:
        with self._lock:
            capabilities = tuple((slot.index, slot.generation, slot.hold_id) for slot in slots)
            if len(set(capabilities)) != len(capabilities):
                raise ArenaSlotReleaseError(f"{self.label} release batch repeats a hold capability")
            for slot in slots:
                self._validate_locked(slot)

    def release_prevalidated(self, slots: tuple[_ArenaSlot, ...]) -> None:
        with self._lock:
            for slot in slots:
                self._validate_locked(slot)
            for slot in slots:
                holds = self._holds[slot.index]
                assert holds is not None
                holds.remove(slot.hold_id)
                if not holds:
                    self._holds[slot.index] = None
                    if self._shared_slots is None:
                        self._free.append(slot.index)
                    else:
                        self._shared_slots.release(slot.index)

    def extend_capacity(self, capacity: int) -> None:
        with self._lock:
            if self._shared_slots is not None:
                raise ValueError("shared cache capacity is sealed at allocation")
            if capacity < len(self._generations) or any(self._holds):
                raise ValueError("page admission requires monotone growth without live owners")
            extra = capacity - len(self._generations)
            self._generations.extend([0] * extra)
            self._holds.extend([None] * extra)
            self._free = list(range(capacity - 1, -1, -1))

    def _mint_hold(self) -> int:
        hold_id = self._next_hold_id
        self._next_hold_id += 1
        return hold_id


class CacheArena:
    """Preallocated tensor storage; only metadata pools change after construction."""

    def __init__(
        self,
        spec: CacheArenaSpec,
        layout: KVLayout,
        device: torch.device | str,
        *,
        rank_id: int,
        virtual_peers: tuple[int, ...] = (),
        shared_storage=None,
        shared_index: int = 0,
    ) -> None:
        if rank_id < 0:
            raise ValueError("cache arena rank id must be non-negative")
        self.spec = spec
        self.layout = layout
        self.device = torch.device(device)
        if self.device.type=="cuda" and self.device.index is None:
            self.device=torch.device("cuda",torch.cuda.current_device())
        self.rank_id = rank_id
        scratch_total = len(spec.scratch_arena_ids) * spec.page_scratch_capacity_per_arena
        state_scratch_total = len(spec.scratch_arena_ids) * spec.state_scratch_capacity_per_arena
        self.shared_storage = shared_storage
        self._virtual_backings = []
        self._capacity_finalized = not bool(spec.reserved_page_capacity)

        def page_tensor(shape, dtype):
            rows = spec.storage_page_capacity + scratch_total
            if spec.reserved_page_capacity and self.device.type == "cuda":
                from cubic_serve.runtime.cache.virtual_memory import VirtualAllocation
                import math

                row_bytes = math.prod(shape) * dtype.itemsize
                storage = VirtualAllocation(rows * row_bytes, self.device, peers=virtual_peers)
                storage.ensure_range(0, spec.page_capacity * row_bytes)
                storage.ensure_range(spec.storage_page_capacity * row_bytes,
                                     scratch_total * row_bytes)
                self._virtual_backings.append((storage, row_bytes))
                return storage.tensor((rows, *shape), dtype)
            return torch.zeros((rows, *shape), dtype=dtype, device=self.device)

        if shared_storage is None:
            self.key = page_tensor(layout.storage_shape(key=True), layout.storage_dtype(key=True))
            self.value = page_tensor(layout.storage_shape(key=False), layout.storage_dtype(key=False))
            self.written = page_tensor((len(layout.layer_ids), layout.block_size_tokens), torch.bool)
            self.page_fields = {
                field.field_id: page_tensor(field.shape, field.dtype) for field in layout.page_fields
            }
            self.states = {
                state_spec.state_id: torch.full(
                    (spec.state_slot_capacity + state_scratch_total, *state_spec.shape),
                    state_spec.initial_value, dtype=state_spec.dtype, device=self.device,
                ) for state_spec in spec.state_specs
            }
        else:
            if (spec != shared_storage.arena_specs[shared_index]
                    or layout != shared_storage.schemas[shared_index][1]
                    or self.device != shared_storage.backing.device):
                raise ValueError("shared arena differs from its physical storage schema")
            pages, self.states = shared_storage.views(shared_index)
            self.key, self.value, self.written = (pages[name] for name in ("key", "value", "written"))
            self.page_fields = {field.field_id: pages["page." + field.field_id]
                                for field in layout.page_fields}
            self._capacity_finalized = True
        self.state_specs = {state_spec.state_id: state_spec for state_spec in spec.state_specs}
        if shared_storage is not None:
            from .shared_pool import SharedCacheFreeSlots
        self._page_pool = _ArenaSlotPool(
            spec.page_capacity, label="KV page", shared_slots=(None if shared_storage is None else
                SharedCacheFreeSlots(shared_storage.allocator, "page")))
        self._state_pools = {
            state_spec.state_id: _ArenaSlotPool(
                spec.state_slot_capacity,
                label=f"state {state_spec.state_id!r}",
                shared_slots=(None if shared_storage is None else
                              SharedCacheFreeSlots(shared_storage.allocator, "state")),
            )
            for state_spec in spec.state_specs
        }
        self._scratch_indices = {
            scratch_arena_id: index for index, scratch_arena_id in enumerate(spec.scratch_arena_ids)
        }
        self._scratch_active: dict[str, _ActiveScratchLease] = {}
        self._scratch_resource_identity: dict[str, tuple[str, str, str, str]] = {}
        self._scratch_last_epoch: dict[str, int] = {}
        self._poison_reason: str | None = None

    def padding_page_slots(self, count: int, scratch_arena_id: str) -> tuple[int, ...]:
        """Map ownerless rows to their lease-private physical padding pages."""
        if not 0 <= count <= self.spec.scratch_capacity:
            raise ValueError("padding row count exceeds its graph scratch envelope")
        index = self._scratch_indices[scratch_arena_id]
        capacity = self.spec.page_scratch_capacity_per_arena
        start = self.spec.storage_page_capacity + index * capacity
        return tuple(start + row % capacity for row in range(count))

    def padding_state_slots(self, count: int, scratch_arena_id: str) -> tuple[int, ...]:
        """Metadata-only padding addresses; active checkpoints need a real lease."""
        if not 0 <= count <= self.spec.scratch_capacity:
            raise ValueError("padding row count exceeds its graph scratch envelope")
        index = self._scratch_indices[scratch_arena_id]
        capacity = self.spec.state_scratch_capacity_per_arena
        start = self.spec.state_slot_capacity + index * capacity
        return tuple(start + row % capacity for row in range(count))

    @property
    def backed_bytes(self) -> int:
        pages = (sum(storage.backed_bytes for storage, _ in self._virtual_backings)
                 if self._virtual_backings else sum(t.numel() * t.element_size() for t in
                     (self.key, self.value, self.written, *self.page_fields.values())))
        return pages + sum(t.numel() * t.element_size() for t in self.states.values())

    def additional_page_backing_bytes(self, capacity: int) -> int:
        if type(capacity) is not int or not self.spec.page_capacity <= capacity <= self.spec.storage_page_capacity:
            raise ValueError("page backing exceeds its pre-readiness storage range")
        return sum(storage.additional_bytes(0, capacity * row_bytes)
                   for storage, row_bytes in self._virtual_backings)

    def finalize_page_capacity(self, capacity: int) -> None:
        self._require_healthy()
        if (self._capacity_finalized or self._scratch_active or any(self._page_pool._holds)
                or any(any(pool._holds) for pool in self._state_pools.values())):
            raise ValueError("page admission requires an unsealed arena without live leases")
        self.additional_page_backing_bytes(capacity)
        for storage, row_bytes in self._virtual_backings:
            storage.ensure_range(0, capacity * row_bytes)
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        self._page_pool.extend_capacity(capacity)
        self.spec = replace(self.spec, page_capacity=capacity)
        for storage, _ in self._virtual_backings:
            storage.seal()
        self._capacity_finalized = True

    def _require_healthy(self) -> None:
        if self._poison_reason is not None:
            raise ArenaPoisoned(self._poison_reason)

    def _poison(self, operation: str, error: BaseException) -> ArenaPoisoned:
        if self._poison_reason is None:
            self._poison_reason = f"cache arena {self.spec.arena_id!r} poisoned during {operation}"
        return ArenaPoisoned(self._poison_reason)

    def reserve_page(
        self,
        *,
        page_id: int,
        owner_rank: int,
        owner_id: str,
        revision: int = 0,
        zero: bool = False,
    ) -> _KVPage:
        self._require_healthy()
        slot = self._page_pool.reserve()
        key = self.key[slot.index]
        value = self.value[slot.index]
        written = self.written[slot.index]
        page_fields = {name: value[slot.index] for name, value in self.page_fields.items()}
        if zero:
            try:
                key.zero_()
                value.zero_()
                written.zero_()
                for field_value in page_fields.values():
                    field_value.zero_()
            except Exception:
                self._page_pool.release(slot)
                raise
        return _KVPage(
            page_id=page_id,
            owner_rank=owner_rank,
            owner_id=owner_id,
            revision=revision,
            key=key,
            value=value,
            written=written,
            slot=slot,
            page_fields=page_fields,
        )

    def release_page(self, page: _KVPage) -> None:
        if page.slot is None:
            raise ArenaSlotReleaseError("dynamic KV storage is not an arena slot")
        batch = self.prepare_release((page.slot,), (), ())
        self.release_prevalidated(batch)

    def retain_page(self, slot: ArenaSlotRef) -> ArenaSlotRef:
        self._require_healthy()
        return self._page_pool.retain(slot)

    def validate_page_ref(self, slot: ArenaSlotRef) -> None:
        self._require_healthy()
        self._page_pool.validate(slot)

    def release_page_ref(self, slot: ArenaSlotRef) -> None:
        self._require_healthy()
        self._page_pool.release(slot)

    def validate_page(self, page: _KVPage) -> None:
        self._require_healthy()
        if page.slot is None:
            raise StaleArenaSlot("dynamic KV storage has no arena generation")
        self._page_pool.validate(page.slot)

    def state_spec(self, state_id: str) -> StateSlotSpec:
        try:
            return self.state_specs[state_id]
        except KeyError as error:
            raise ValueError(f"state {state_id!r} is absent from the cache arena schema") from error

    def reset_state_rows(self, slots: torch.Tensor) -> None:
        """Initialize caller-validated owned slots according to every state schema."""
        self._require_healthy()
        for state_id, tensor in self.states.items():
            tensor.index_fill_(0, slots, self.state_specs[state_id].initial_value)

    def reset_page_rows(self, pages: torch.Tensor) -> None:
        """Clear new owned pages, including all payloads sharing their lifecycle."""
        self._require_healthy()
        self.key.index_fill_(0, pages, 0)
        self.value.index_fill_(0, pages, 0)
        self.written.index_fill_(0, pages, False)
        for tensor in self.page_fields.values():
            tensor.index_fill_(0, pages, 0)

    def reserve_state(
        self,
        state_id: str,
        *,
        source: torch.Tensor | None = None,
    ) -> tuple[_ArenaSlot, torch.Tensor]:
        self._require_healthy()
        self.state_spec(state_id)
        pool = self._state_pools[state_id]
        slot = pool.reserve()
        value = self.states[state_id][slot.index]
        try:
            if source is not None:
                value.copy_(source)
            else:
                value.fill_(self.state_specs[state_id].initial_value)
        except Exception:
            pool.release(slot)
            raise
        return slot, value

    def release_state(self, state_id: str, slot: _ArenaSlot) -> None:
        batch = self.prepare_release((), ((state_id, slot),), ())
        self.release_prevalidated(batch)

    def retain_state(self, state_id: str, slot: ArenaSlotRef) -> ArenaSlotRef:
        self._require_healthy()
        self.state_spec(state_id)
        return self._state_pools[state_id].retain(slot)

    def validate_state_ref(self, state_id: str, slot: ArenaSlotRef) -> None:
        self.validate_state(state_id, slot)

    def release_state_ref(self, state_id: str, slot: ArenaSlotRef) -> None:
        self._require_healthy()
        self.state_spec(state_id)
        self._state_pools[state_id].release(slot)

    def validate_state(self, state_id: str, slot: _ArenaSlot) -> None:
        self._require_healthy()
        self.state_spec(state_id)
        self._state_pools[state_id].validate(slot)

    def claim_scratch(
        self,
        claim: _ScratchLeaseClaim,
        *,
        transaction_id: str,
        scratch_indices: tuple[int, ...],
    ) -> _ScratchParticipantHold:
        self._require_healthy()
        if claim.scratch_arena_id not in self._scratch_indices:
            raise ScratchArenaLeaseError(f"unknown scratch arena {claim.scratch_arena_id!r}")
        if any(index < 0 or index >= self.spec.scratch_capacity for index in scratch_indices):
            raise ScratchArenaLeaseError("launch padding exceeds its scratch arena capacity")
        active = self._scratch_active.get(claim.scratch_arena_id)
        if active is not None:
            if active.claim != claim:
                raise ScratchArenaLeaseError("scratch arena already has an active replay lease")
            active.participants.add(transaction_id)
            return _ScratchParticipantHold(claim, transaction_id)
        owner = self._scratch_resource_identity.get(claim.scratch_arena_id)
        if owner is not None and owner != claim.resource_identity:
            raise ScratchArenaLeaseError("scratch arena belongs to a foreign replay resource")
        if claim.lease_epoch <= self._scratch_last_epoch.get(claim.scratch_arena_id, 0):
            raise ScratchArenaLeaseError("scratch arena replay lease is stale")
        self._scratch_resource_identity.setdefault(claim.scratch_arena_id, claim.resource_identity)
        self._scratch_last_epoch[claim.scratch_arena_id] = claim.lease_epoch
        self._scratch_active[claim.scratch_arena_id] = _ActiveScratchLease(claim, {transaction_id})
        return _ScratchParticipantHold(claim, transaction_id)

    def prepare_release(
        self,
        page_slots: tuple[_ArenaSlot, ...],
        state_slots: tuple[tuple[str, _ArenaSlot], ...],
        scratch_holds: tuple[_ScratchParticipantHold, ...],
    ) -> _ArenaReleaseBatch:
        self._require_healthy()
        try:
            self._page_pool.validate_batch(page_slots)
            grouped_states: dict[str, list[_ArenaSlot]] = {}
            for state_id, slot in state_slots:
                self.state_spec(state_id)
                grouped_states.setdefault(state_id, []).append(slot)
            for state_id, slots in grouped_states.items():
                self._state_pools[state_id].validate_batch(tuple(slots))
            hold_ids = tuple(
                (hold.claim.scratch_arena_id, hold.transaction_id) for hold in scratch_holds
            )
            if len(set(hold_ids)) != len(hold_ids):
                raise ScratchArenaLeaseError("scratch release batch repeats a participant")
            for hold in scratch_holds:
                active = self._scratch_active.get(hold.claim.scratch_arena_id)
                if (
                    active is None
                    or active.claim != hold.claim
                    or hold.transaction_id not in active.participants
                ):
                    raise ScratchArenaLeaseError("scratch release capability is stale or foreign")
        except (CacheStateError, ValueError) as error:
            raise self._poison("release prevalidation", error) from error
        return _ArenaReleaseBatch(page_slots, state_slots, scratch_holds)

    def release_prevalidated(self, batch: _ArenaReleaseBatch) -> None:
        self._require_healthy()
        self._page_pool.release_prevalidated(batch.page_slots)
        grouped_states: dict[str, list[_ArenaSlot]] = {}
        for state_id, slot in batch.state_slots:
            grouped_states.setdefault(state_id, []).append(slot)
        for state_id, slots in grouped_states.items():
            self._state_pools[state_id].release_prevalidated(tuple(slots))
        for hold in batch.scratch_holds:
            active = self._scratch_active[hold.claim.scratch_arena_id]
            active.participants.remove(hold.transaction_id)
            if not active.participants:
                del self._scratch_active[hold.claim.scratch_arena_id]

    def binding(
        self,
        *,
        scratch_hold: _ScratchParticipantHold,
        read_page_slots: tuple[ArenaReadSlot, ...] = (),
        read_state_slots: tuple[ArenaReadSlot, ...] = (),
        page_slots: tuple[ArenaCOWSlots, ...] = (),
        state_slots: tuple[ArenaCOWSlots, ...] = (),
    ) -> CacheArenaBinding:
        self._require_healthy()
        scratch_claim = scratch_hold.claim
        active = self._scratch_active.get(scratch_claim.scratch_arena_id)
        if (
            active is None
            or active.claim != scratch_claim
            or (scratch_hold.transaction_id not in active.participants)
        ):
            raise ScratchArenaLeaseError("scratch binding requires its exact active capability")
        scratch_index = self._scratch_indices[scratch_claim.scratch_arena_id]
        state_start = (self.spec.state_slot_capacity
                       + scratch_index * self.spec.state_scratch_capacity_per_arena)
        page_scratch = self.padding_page_slots(self.spec.scratch_capacity, scratch_claim.scratch_arena_id)
        state_scratch = tuple(range(state_start, state_start + self.spec.state_scratch_capacity_per_arena))
        scratch_states = tuple((state_id, state_scratch) for state_id in self._state_pools)
        return CacheArenaBinding(
            arena_id=self.spec.arena_id,
            scratch_arena_id=scratch_claim.scratch_arena_id,
            scratch_lease_id=scratch_claim.lease_id,
            scratch_lease_epoch=scratch_claim.lease_epoch,
            scratch_launch_plan_id=scratch_claim.launch_plan_id,
            read_page_slots=read_page_slots,
            read_state_slots=read_state_slots,
            page_slots=page_slots,
            state_slots=state_slots,
            key=self.key,
            value=self.value,
            written=self.written,
            state_tensors=tuple(self.states.items()),
            scratch_page_slots=page_scratch,
            scratch_state_slots=scratch_states,
            page_fields=tuple(self.page_fields.items()),
        )

    @property
    def backing_identity(self) -> tuple[int, ...]:
        """Rank-global tensor addresses captured by every graph slot."""

        return (
            self.key.data_ptr(),
            self.value.data_ptr(),
            self.written.data_ptr(),
            *(tensor.data_ptr() for tensor in self.states.values()),
            *(tensor.data_ptr() for tensor in self.page_fields.values()),
        )

    @property
    def domain(self) -> CacheArenaDomain:
        return (self.rank_id, self.layout, self.spec, self.device)


_CacheArena = CacheArena


@dataclass(slots=True)
class _KVPage:
    page_id: int
    owner_rank: int
    owner_id: str
    revision: int
    key: torch.Tensor
    value: torch.Tensor
    written: torch.Tensor
    slot: _ArenaSlot | None = None
    page_fields: dict[str, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def zeroed(
        cls,
        *,
        page_id: int,
        owner_rank: int,
        owner_id: str,
        layout: KVLayout,
        device: torch.device,
    ) -> _KVPage:
        return cls(
            page_id=page_id,
            owner_rank=owner_rank,
            owner_id=owner_id,
            revision=0,
            key=torch.zeros(layout.storage_shape(key=True),
                            dtype=layout.storage_dtype(key=True), device=device),
            value=torch.zeros(layout.storage_shape(key=False),
                              dtype=layout.storage_dtype(key=False), device=device),
            written=torch.zeros(
                (len(layout.layer_ids), layout.block_size_tokens),
                dtype=torch.bool,
                device=device,
            ),
            slot=None,
            page_fields={
                spec.field_id: torch.zeros(spec.shape, dtype=spec.dtype, device=device)
                for spec in layout.page_fields
            },
        )

    def clone(self) -> _KVPage:
        return _KVPage(
            page_id=self.page_id,
            owner_rank=self.owner_rank,
            owner_id=self.owner_id,
            revision=self.revision,
            key=self.key.clone(),
            value=self.value.clone(),
            written=self.written.clone(),
            slot=None,
            page_fields={name: value.clone() for name, value in self.page_fields.items()},
        )


@dataclass(frozen=True, slots=True)
class KVPageSnapshot:
    """Detached diagnostic or prefix image of one committed page."""

    page_id: int
    owner_rank: int
    owner_id: str
    revision: int
    key: torch.Tensor
    value: torch.Tensor
    written: torch.Tensor

    page_fields: dict[str, torch.Tensor] = field(default_factory=dict)

    def clone(self) -> KVPageSnapshot:
        return KVPageSnapshot(
            page_id=self.page_id,
            owner_rank=self.owner_rank,
            owner_id=self.owner_id,
            revision=self.revision,
            key=self.key.clone(),
            value=self.value.clone(),
            written=self.written.clone(),
            page_fields={name: value.clone() for name, value in self.page_fields.items()},
        )
