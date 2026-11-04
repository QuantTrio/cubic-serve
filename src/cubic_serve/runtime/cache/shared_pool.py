"""Byte ownership for preallocated attention and recurrent-state storage.

Pool capacity is a byte budget and tensor geometry, independent of the request
ceiling. Pages prefer low addresses and states prefer high addresses. Interior free
regions are usable by either class without moving live payloads. This module performs
no device allocation, transfer, synchronization, capture or model arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod

import torch

from .pages import ArenaExhausted, ArenaSlotReleaseError, StaleArenaSlot


@dataclass(frozen=True, slots=True)
class SharedCacheField:
    """A format-owned, densely stored payload within one resource entry."""

    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if (type(self.name) is not str or not self.name or not self.shape
                or any(type(n) is not int or n <= 0 for n in self.shape)
                or self.dtype not in (torch.bool, torch.uint8, torch.int8,
                                      torch.float16, torch.bfloat16, torch.float32,
                                      torch.int32, torch.int64)):
            raise ValueError("shared cache field requires a named typed positive shape")

    @property
    def nbytes(self) -> int:
        return prod(self.shape) * self.dtype.itemsize


class SharedCacheLayout:
    """Array-of-entries layout with dense inner payloads and a fixed outer stride.

    Each resource class has its own views of the same backing. Only its owned
    entries may be read or written; the byte allocator proves live page/state
    extents disjoint. Field alignment is a vector-access contract, not a
    model/device/format dispatch rule.
    """

    def __init__(self, fields: tuple[SharedCacheField, ...], *, alignment: int = 16):
        if (not fields or len({field.name for field in fields}) != len(fields)
                or type(alignment) is not int or alignment < 8
                or alignment & (alignment - 1)):
            raise ValueError("shared cache layout requires unique fields and aligned entries")
        self.fields = fields
        self.alignment = alignment
        self.offsets: dict[str, int] = {}
        offset = 0
        for field in fields:
            offset = (offset + alignment - 1) // alignment * alignment
            self.offsets[field.name] = offset
            offset += field.nbytes
        self.entry_bytes = (offset + alignment - 1) // alignment * alignment

    def views(self, backing: torch.Tensor, capacity: int, *, offset: int = 0
              ) -> dict[str, torch.Tensor]:
        if (backing.dtype is not torch.uint8 or backing.ndim != 1
                or not backing.is_contiguous() or type(capacity) is not int or capacity < 1
                or type(offset) is not int or offset < 0 or offset % self.alignment
                or backing.data_ptr() % self.alignment
                or offset + capacity * self.entry_bytes > backing.numel()
                or backing.numel() % 8):
            raise ValueError("shared cache views exceed their contiguous byte backing")
        result = {}
        for field in self.fields:
            itemsize = field.dtype.itemsize
            strides = []
            stride = 1
            for dimension in reversed(field.shape):
                strides.append(stride)
                stride *= dimension
            result[field.name] = torch.as_strided(
                backing.view(field.dtype), (capacity, *field.shape),
                (self.entry_bytes // itemsize, *reversed(strides)),
                (offset + self.offsets[field.name]) // itemsize,
            )
        return result


class SharedCacheStorage:
    """Fixed backing and typed views for a topology-local cache byte budget.

    Every arena supplies its existing format-owned page and state schema.
    Target/provider fields share entries, without sharing their arithmetic.
    Only a constant dummy state per padding arena is outside the common pool;
    actual speculative checkpoints must be reserved from the byte allocator
    under the graph transaction's exclusive lease.
    """

    @staticmethod
    def _padding_pages(spec, layout):
        # The common encoded append and attention window both consume explicit
        # row activity. Inactive rows cannot write/read an owned cache entry;
        # their COW source equals destination in one lease-private dummy page.
        return 1 if layout.storage.quantized else spec.page_scratch_capacity_per_arena

    @staticmethod
    def layouts(schemas, extra_state_fields=()):
        page_fields, state_fields = [], []
        for index, (spec, layout) in enumerate(schemas):
            prefix = f"arena{index}."
            page_fields.extend((
                SharedCacheField(prefix + "key", layout.storage_dtype(key=True), layout.storage_shape(key=True)),
                SharedCacheField(prefix + "value", layout.storage_dtype(key=False), layout.storage_shape(key=False)),
                SharedCacheField(prefix + "written", torch.bool,
                                 (len(layout.layer_ids), layout.block_size_tokens)),
            ))
            page_fields.extend(SharedCacheField(prefix + "page." + f.field_id, f.dtype, f.shape)
                               for f in layout.page_fields)
            state_fields.extend(SharedCacheField(prefix + "state." + f.state_id, f.dtype, f.shape)
                                for f in spec.state_specs)
        state_fields.extend(SharedCacheField("engine." + f.name, f.dtype, f.shape)
                            for f in extra_state_fields)
        return (SharedCacheLayout(tuple(page_fields)),
                SharedCacheLayout(tuple(state_fields)) if state_fields else None)

    @staticmethod
    def allocation_bytes(budget_bytes, schemas, extra_state_fields=()):
        schemas = tuple(schemas)
        pages, states = SharedCacheStorage.layouts(schemas, extra_state_fields)
        dummy_states = max(len(spec.scratch_arena_ids) for spec, _ in schemas) if states else 0
        state_bytes = 0 if states is None else states.entry_bytes
        scratch_start = (budget_bytes + dummy_states * state_bytes + pages.entry_bytes - 1) // pages.entry_bytes
        scratch_pages = max(SharedCacheStorage._padding_pages(spec, layout) * len(spec.scratch_arena_ids)
                            for spec, layout in schemas)
        return (scratch_start + scratch_pages) * pages.entry_bytes

    def __init__(self, schemas, *, budget_bytes: int, device: torch.device | str,
                 state_capacity: int | None = None, extra_state_fields=()):
        from dataclasses import replace
        if type(budget_bytes) is not int or budget_bytes <= 0 or budget_bytes % 16:
            raise ValueError("shared cache budget requires aligned positive bytes")
        self.schemas = tuple((replace(spec, page_scratch_capacity=self._padding_pages(spec, layout)), layout)
                             for spec, layout in schemas)
        schemas = self.schemas
        self.extra_state_fields = tuple(extra_state_fields)
        self.page_layout, self.state_layout = self.layouts(self.schemas, self.extra_state_fields)
        self.budget_bytes = budget_bytes
        state_bytes = self.state_layout.entry_bytes if self.state_layout is not None else 0
        self.allocator = SharedCachePool(pool_id="shared-cache", budget_bytes=budget_bytes,
                                         page_bytes=self.page_layout.entry_bytes,
                                         state_bytes=state_bytes, state_capacity=state_capacity)
        self.state_capacity = self.allocator.state_capacity
        self.page_capacity = self.allocator.page_capacity
        if not self.page_capacity or (state_bytes and not self.state_capacity):
            raise ArenaExhausted("shared budget cannot hold a cache resource")
        dummy_count = max(len(spec.scratch_arena_ids) for spec, _ in schemas) if state_bytes else 0
        self.page_scratch_start = (
            budget_bytes + dummy_count * state_bytes + self.page_layout.entry_bytes - 1
        ) // self.page_layout.entry_bytes
        self.backing = torch.zeros(self.allocation_bytes(budget_bytes, self.schemas, self.extra_state_fields),
                                   dtype=torch.uint8, device=device)
        self.arena_specs = tuple(replace(
            spec, page_capacity=self.page_capacity, state_slot_capacity=self.state_capacity,
            reserved_page_capacity=self.page_scratch_start, state_scratch_capacity=1,
        ) for spec, _ in schemas)
        self._fields: dict[tuple, str] = {}
        self._arena_views = []
        for index, (spec, _) in enumerate(schemas):
            prefix = f"arena{index}."
            page_count = (self.page_scratch_start
                          + spec.page_scratch_capacity_per_arena * len(spec.scratch_arena_ids))
            page_views = self.page_layout.views(self.backing, page_count)
            state_views = ({} if self.state_layout is None else self.state_layout.views(
                self.backing, self.state_capacity + len(spec.scratch_arena_ids),
                offset=self.allocator.state_offset_bytes))
            selected_pages = {name[len(prefix):]: value for name, value in page_views.items()
                              if name.startswith(prefix)}
            selected_states = {name[len(prefix + "state."):]: value for name, value in state_views.items()
                               if name.startswith(prefix + "state.")}
            for value in selected_pages.values():
                self._fields[self._identity(value)] = "page"
            for value in selected_states.values():
                self._fields[self._identity(value)] = "state"
            for field in spec.state_specs:
                if field.initial_value:
                    selected_states[field.state_id].fill_(field.initial_value)
            self._arena_views.append((selected_pages, selected_states))
        engine_views = ({} if not self.extra_state_fields else self.state_layout.views(
            self.backing, self.state_capacity + dummy_count,
            offset=self.allocator.state_offset_bytes))
        self.extra_states = {f.name: engine_views["engine." + f.name] for f in self.extra_state_fields}
        for value in self.extra_states.values():
            self._fields[self._identity(value)] = "state"

    @classmethod
    def budget_from_available(cls, available_bytes: int, schemas, extra_state_fields=()) -> int:
        """Largest aligned owner pool after exact constant padding storage."""
        schemas = tuple(schemas)
        pages, states = cls.layouts(schemas, extra_state_fields)
        dummy_count = max(len(spec.scratch_arena_ids) for spec, _ in schemas) if states else 0
        scratch_pages = max(cls._padding_pages(spec, layout) * len(spec.scratch_arena_ids)
                            for spec, layout in schemas)
        state_bytes = 0 if states is None else states.entry_bytes
        return ((available_bytes // pages.entry_bytes - scratch_pages) * pages.entry_bytes
                - dummy_count * state_bytes)

    @staticmethod
    def _identity(value):
        return value.data_ptr(), tuple(value.shape), value.stride(), value.dtype, value.device

    def allows_domain_overlap(self, left, right) -> bool:
        """Static view registration only; replay still requires byte ownership.

        The views cover potential slots, not simultaneous live contents. Only
        the common cache's validated page/state transaction can authorize their
        actual indices. This does not grant readiness or a numerical receipt.
        """
        a, b = self._fields.get(self._identity(left)), self._fields.get(self._identity(right))
        return a is not None and b is not None and a != b

    def views(self, index: int):
        return self._arena_views[index]


@dataclass(frozen=True, slots=True)
class SharedPoolSlot:
    """One independently releasable hold on a generation-bound allocation."""

    pool_id: str
    kind: str
    index: int
    generation: int
    hold_id: int
    _owner: object = field(repr=False, compare=False)


class SharedCachePool:
    """Fixed-address byte ownership with exact overlap admission on every rank.

    Page and state grids may overlap as potential views. Live entries never
    overlap. Bitmaps retain interior holes, so releasing state makes all of its
    unoccupied bytes available to pages without moving another live state.
    """

    def __init__(self, *, pool_id: str, budget_bytes: int,
                 page_bytes: int, state_bytes: int, state_capacity: int | None = None,
                 rank_geometries: tuple[tuple[int, int, int], ...] = ()) -> None:
        if type(pool_id) is not str or not pool_id:
            raise ValueError("shared cache pool requires an identity")
        if (any(type(n) is not int or n <= 0 for n in (budget_bytes, page_bytes))
                or type(state_bytes) is not int or state_bytes < 0):
            raise ValueError("shared cache geometry requires positive integer bytes")
        self.pool_id = pool_id
        self.budget_bytes = budget_bytes
        self.page_bytes = page_bytes
        self.state_bytes = state_bytes
        self.rank_geometries = ((budget_bytes, page_bytes, state_bytes), *rank_geometries)
        if any(type(b) is not int or b <= 0 or type(p) is not int or p <= 0
               or type(s) is not int or s < 0 for b, p, s in self.rank_geometries):
            raise ValueError("invalid topology cache geometry")
        self.page_capacity = min(b // p for b, p, _ in self.rank_geometries)
        state_limits = [b // s for b, _, s in self.rank_geometries if s]
        maximum_states = min(state_limits) if state_limits else self.page_capacity
        if state_capacity is None:
            state_capacity = maximum_states
        if type(state_capacity) is not int or not 0 <= state_capacity <= maximum_states:
            raise ValueError("shared state extent exceeds its byte budget")
        self.state_capacity = state_capacity
        self.state_offset_bytes = budget_bytes - state_capacity * state_bytes
        self.live_bits = {"page": 0, "state": 0}
        self.blocked_bits = {"page": 0, "state": 0}
        self.blocked_version = {"page": 0, "state": 0}
        self._rank_blocked = {kind: [0] * len(self.rank_geometries)
                              for kind in ("page", "state")}
        self._live: dict[str, dict[int, int]] = {"page": {}, "state": {}}
        self._generation: dict[str, dict[int, int]] = {"page": {}, "state": {}}
        self._references: dict[str, dict[int, set[int]]] = {"page": {}, "state": {}}
        self._next_hold_id = 1
        self._token = object()

    @staticmethod
    def _mask(start: int, end: int, origin: int, stride: int, capacity: int) -> int:
        if not stride or start >= end:
            return 0
        lo = max(0, (start - origin) // stride)
        hi = min(capacity, (end - origin + stride - 1) // stride)
        return ((1 << (hi - lo)) - 1) << lo if hi > lo else 0

    def _overlap(self, kind: str, index: int, geometry: tuple[int, int, int]) -> int:
        budget, page, state = geometry
        origin = budget - self.state_capacity * state
        if kind == "page":
            return self._mask(index * page, (index + 1) * page,
                              origin, state, self.state_capacity)
        return self._mask(origin + index * state, origin + (index + 1) * state,
                          0, page, self.page_capacity)

    def _occupy(self, kind: str, index: int, *, release: bool = False) -> None:
        other = "state" if kind == "page" else "page"
        bit = 1 << index
        self.live_bits[kind] = (self.live_bits[kind] & ~bit if release
                                else self.live_bits[kind] | bit)
        union = 0
        for rank, geometry in enumerate(self.rank_geometries):
            mask = self._overlap(kind, index, geometry)
            blocked = self._rank_blocked[other][rank]
            if release:
                blocked &= ~mask
                if mask:
                    # Only the two boundary cells can intersect another live
                    # entry. Interior cells lie entirely inside the freed one.
                    for edge in ((mask & -mask).bit_length() - 1, mask.bit_length() - 1):
                        if self._overlap(other, edge, geometry) & self.live_bits[kind]:
                            blocked |= 1 << edge
            else:
                blocked |= mask
            self._rank_blocked[other][rank] = blocked
            union |= blocked
        if self.blocked_bits[other]!=union:
            self.blocked_version[other]+=1
            self.blocked_bits[other]=union

    @property
    def unassigned_bytes(self) -> int:
        return self.budget_bytes - self.live_bytes

    @property
    def live_bytes(self) -> int:
        return (len(self._live["page"]) * self.page_bytes
                + len(self._live["state"]) * self.state_bytes)

    def cold_owner_capacity(self, *, state_slots: int, pages: int) -> int:
        """Packed high-state/low-page owners in an empty pool on every rank.

        This sizes pre-readiness fixtures. Live admission still requires the
        exact occupancy/ownership checks, including fragmentation and references.
        """
        if any(type(n) is not int or n <= 0 for n in (state_slots, pages)):
            raise ValueError("cold owners require positive state/page counts")
        return min(self.state_capacity // state_slots, *(b // (state_slots * s + pages * p)
                   for b, p, s in self.rank_geometries))

    def capacity(self, kind: str) -> int:
        if kind not in ("page", "state"):
            raise ValueError("unknown shared cache resource class")
        return self.page_capacity if kind == "page" else self.state_capacity

    @staticmethod
    def _next(kind: str, bits: int) -> int:
        if not bits:
            raise ArenaExhausted("shared cache byte budget exhausted")
        return ((bits & -bits).bit_length() - 1 if kind == "page"
                else bits.bit_length() - 1)

    def allocate(self, kind: str) -> SharedPoolSlot:
        eligible = ((1 << self.capacity(kind)) - 1) & ~(
            self.live_bits[kind] | self.blocked_bits[kind])
        return self.reserve_at(kind, self._next(kind, eligible))

    def reserve_at(self, kind: str, index: int) -> SharedPoolSlot:
        """Claim one field-family slot; other fields may hold the same bundle."""
        if kind not in self._live or type(index) is not int or index < 0:
            raise ValueError("invalid shared cache resource index")
        if index in self._live[kind]:
            hold = self._next_hold_id
            self._next_hold_id += 1
            self._references[kind][index].add(hold)
            return SharedPoolSlot(self.pool_id, kind, index, self._live[kind][index], hold, self._token)
        if index >= self.capacity(kind) or self.blocked_bits[kind] & (1 << index):
            raise ArenaExhausted("shared cache byte budget exhausted")
        self._occupy(kind, index)
        generation = self._generation[kind].get(index, 0) + 1
        self._generation[kind][index] = generation
        self._live[kind][index] = generation
        hold = self._next_hold_id
        self._next_hold_id += 1
        self._references[kind][index] = {hold}
        return SharedPoolSlot(self.pool_id, kind, index, generation, hold, self._token)

    def validate(self, slot: SharedPoolSlot) -> None:
        if (not isinstance(slot, SharedPoolSlot) or slot.pool_id != self.pool_id or slot._owner is not self._token
                or slot.kind not in self._live
                or self._live[slot.kind].get(slot.index) != slot.generation
                or slot.hold_id not in self._references[slot.kind].get(slot.index, ())):
            raise StaleArenaSlot("shared cache allocation is foreign or stale")

    def release(self, slot: SharedPoolSlot) -> None:
        try:
            self.validate(slot)
        except StaleArenaSlot as error:
            raise ArenaSlotReleaseError("shared cache allocation is foreign or released") from error
        kind, index = slot.kind, slot.index
        self._references[kind][index].remove(slot.hold_id)
        if self._references[kind][index]:
            return
        del self._references[kind][index]
        del self._live[kind][index]
        self._occupy(kind, index, release=True)


class SharedCacheFreeSlots:
    """One field family's free indices over a shared physical byte allocator.

    Separate target/provider page families can use the same numeric page while
    occupying disjoint fields. The physical bundle is charged once. Prefix
    reference counts remain the caller's existing ownership mechanism.
    """

    def __init__(self, pool: SharedCachePool, kind: str):
        if kind not in ("page", "state"):
            raise ValueError("unknown shared cache resource class")
        self.pool, self.kind = pool, kind
        self._free_bits = (1 << pool.capacity(kind)) - 1
        self._tickets: dict[int, SharedPoolSlot] = {}
        self._count=pool.capacity(kind)
        self._count_version=-1
        self._next_candidate: int | None = None
        self._candidate_version = -1

    def __len__(self) -> int:
        version=self.pool.blocked_version[self.kind]
        if self._count_version!=version:
            self._count=(self._free_bits&~self.pool.blocked_bits[self.kind]).bit_count()
            self._count_version=version
        return self._count

    def candidates(self, count: int) -> tuple[int, ...]:
        if type(count) is not int or count < 0:
            raise ValueError("cache reservation count must be nonnegative")
        eligible = self._free_bits & ~self.pool.blocked_bits[self.kind]
        if count > eligible.bit_count():
            raise ArenaExhausted("shared cache byte budget exhausted")
        result = []
        for _ in range(count):
            index = self.pool._next(self.kind, eligible)
            result.append(index)
            eligible &= ~(1 << index)
        return tuple(result)

    def allocate(self) -> int:
        count=len(self)
        if not count:
            raise ArenaExhausted("shared cache byte budget exhausted")
        blocked = self.pool.blocked_bits[self.kind]
        version = self.pool.blocked_version[self.kind]
        index = self._next_candidate
        # Consecutive grants normally advance through adjacent free entries.
        # Any opposing ownership change invalidates that ordering witness;
        # releases below/above the cursor restore the appropriate preference.
        if (version != self._candidate_version or index is None
                or not 0 <= index < self.pool.capacity(self.kind)
                or not self._free_bits & (1 << index) or blocked & (1 << index)):
            # Both operands stay nonnegative. Inverting a serving-sized mask
            # otherwise creates a full-width two's-complement temporary.
            eligible = self._free_bits ^ (self._free_bits & blocked)
            index = self.pool._next(self.kind, eligible)
        ticket = self.pool.reserve_at(self.kind, index)
        self._free_bits ^= 1 << index
        self._tickets[index] = ticket
        self._count=count-1
        self._next_candidate = index + (1 if self.kind == "page" else -1)
        self._candidate_version = version
        return index

    def release(self, index: int) -> None:
        if index not in self._tickets:
            raise ArenaSlotReleaseError("shared cache slot is foreign or released")
        count=len(self)
        self.pool.release(self._tickets.pop(index))
        self._free_bits |= 1 << index
        self._count=count+1
        if (self._next_candidate is None
                or (index < self._next_candidate if self.kind == "page"
                    else index > self._next_candidate)):
            self._next_candidate = index


def shared_reservation_fits(reservations: tuple[tuple[SharedCacheFreeSlots, int], ...]) -> bool:
    """Simulate exact joint admission without copying live ownership records."""
    from copy import copy
    if not reservations:
        return True
    pool = reservations[0][0].pool
    if len({id(slots) for slots, _ in reservations}) != len(reservations):
        raise ValueError("joint reservation repeats a resource family")
    for slots,count in reservations:
        if slots.pool is not pool:raise ValueError("joint reservation crosses physical cache pools")
        if type(count) is not int or count<0:raise ValueError("cache reservation count must be nonnegative")
    if len({slots.kind for slots,count in reservations if count})<=1:
        return all(count<=len(slots) for slots,count in reservations)
    shadow = copy(pool)
    shadow.live_bits = pool.live_bits.copy()
    shadow.blocked_bits = pool.blocked_bits.copy()
    shadow.blocked_version = pool.blocked_version.copy()
    shadow._rank_blocked = {kind: values.copy() for kind, values in pool._rank_blocked.items()}
    try:
        for slots, count in reservations:
            if slots.pool is not pool:
                raise ValueError("joint reservation crosses physical cache pools")
            if type(count) is not int or count < 0:
                raise ValueError("cache reservation count must be nonnegative")
            free = slots._free_bits
            for _ in range(count):
                index = shadow._next(slots.kind, free & ~shadow.blocked_bits[slots.kind])
                bit = 1 << index
                free &= ~bit
                if not shadow.live_bits[slots.kind] & bit:
                    shadow._occupy(slots.kind, index)
    except ArenaExhausted:
        return False
    return True


class SharedCacheJournal:
    """Rollback only allocations made by one scheduler page-program attempt."""

    def __init__(self,slots):
        self.slots=slots
        self.pool,self.kind=slots.pool,slots.kind
        self.allocated=[]

    @property
    def _free_bits(self):
        return self.slots._free_bits

    def __len__(self):
        return len(self.slots)

    def allocate(self):
        index=self.slots.allocate()
        self.allocated.append(index)
        return index

    def release(self,index):
        self.slots.release(index)
        self.allocated.remove(index)

    def finish(self,*,commit):
        if not commit:
            for index in reversed(self.allocated):self.slots.release(index)
        self.allocated.clear()


class SharedCacheOwnership:
    """Host-side byte and row admission before any shared-pool replay fill.

    The existing topology lease and row/page transaction guards still apply.
    A potential page/state alias is safe only with this live ownership witness.
    No device work, synchronization or readiness receipt is issued here.
    """

    _tag=b'CSP1'

    def __init__(self,pool):
        self.pool=pool
        self.pages=self.states=0
        self._last_states=-1
        self._blocked_pages=0
        self._resident_states=0

    @classmethod
    def pack(cls,payload,pool):
        import struct
        blobs=tuple(pool.live_bits[k].to_bytes((pool.live_bits[k].bit_length()+7)//8,'little')
                    for k in ('page','state'))
        return cls._tag+struct.pack('<II',len(blobs[0]),len(blobs[1]))+b''.join(blobs)+payload

    def unpack(self,payload):
        import struct
        if not payload.startswith(self._tag) or len(payload)<12:
            raise ValueError('shared cache command lacks byte ownership')
        np,ns=struct.unpack_from('<II',payload,4)
        if np>(self.pool.page_capacity+7)//8 or ns>(self.pool.state_capacity+7)//8 or 12+np+ns>=len(payload):
            raise ValueError('shared cache ownership extent is invalid')
        pages=int.from_bytes(payload[12:12+np],'little')
        states=int.from_bytes(payload[12+np:12+np+ns],'little')
        if pages.bit_length()>self.pool.page_capacity or states.bit_length()>self.pool.state_capacity:
            raise ValueError('shared cache ownership exceeds physical capacity')
        if states!=self._last_states:
            blocked=0;bits=states
            while bits:
                bit=bits&-bits;bits^=bit
                blocked |= self.pool._overlap('state',bit.bit_length()-1,
                                             (self.pool.budget_bytes,self.pool.page_bytes,self.pool.state_bytes))
            self._blocked_pages=blocked
            self._last_states=states
        if pages&self._blocked_pages:
            raise ValueError('live KV pages overlap live recurrent state')
        self.pages,self.states=pages,states
        return payload[12+np+ns:]

    def _require(self,kind,indices):
        bits=self.pages if kind=='page' else self.states
        if any(type(i) is not int or i<0 or not (bits>>i)&1 for i in indices):
            raise ValueError('cache transaction references an unowned resource')

    def validate_command(self,command):
        from cubic_serve.runtime.continuous_batch import BatchCommand,ContinueBatchCommand
        if isinstance(command,BatchCommand):
            durable=0
            for row in command.rows:
                self._require('state',(row.state_source_slot,row.state_destination_slot))
                self._require('page',(row.page_source,row.page_destination,*row.page_table,*row.new_pages))
                if row.mtp_position>=0:
                    self._require('page',(row.mtp_page_destination,*row.mtp_page_table))
                durable |= 1<<row.state_destination_slot
            self._resident_states=durable
        elif isinstance(command,ContinueBatchCommand):
            if self._resident_states&~self.states:
                raise ValueError('continuation lost its state lease')
            self._require('page',(p for _,_,p in command.page_updates))
            self._require('page',(p for _,_,_,p in command.future_page_updates))
            self._require('page',command.source_pages)
            self._require('page',(p for step in command.page_program for entry in step for p in entry[:2]))
            self._require('page',(p for table in command.provider_tables for p in table))
            if command.draft is not None:
                self._require('page',(p for tables in (command.draft.target_tables,command.draft.provider_tables)
                                      for table in tables for p in table))
        else:
            raise ValueError('unexpected shared cache transaction')
