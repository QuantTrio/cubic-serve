"""Atomic copy-on-write transactions over attention KV and recurrent state."""

from __future__ import annotations

from typing_extensions import TypeAliasType

import hashlib
from dataclasses import dataclass
from cubic_serve._compat import StrEnum
from threading import RLock

import torch

from cubic_serve.contracts import execution as ec
from cubic_serve.runtime.cache.pages import (
    ArenaCOWSlots,
    ArenaReadSlot,
    CacheArena,
    CacheArenaBinding,
    CacheArenaSpec,
    KVLayout,
    KVPageSnapshot,
    ScratchArenaLeaseError,
    StaleArenaSlot,
    _ArenaReleaseBatch,
    _ArenaSlot,
    _CacheArena,
    _KVPage,
    _ScratchLeaseClaim,
    _ScratchParticipantHold,
)
from cubic_serve.runtime.cache.pages import CacheStateError as CacheStateError
from cubic_serve.runtime.cache.prefix import (
    PhysicalPageRef,
    PhysicalStateRef,
    PrefixRestore,
    PrefixSnapshot,
    RankPrefixResidence,
    RankPrefixSource,
    RecurrentStateSnapshot,
    _hash_tensor,
    seal_prefix_snapshot,
    seal_recurrent_state_snapshot,
)


class OwnershipError(CacheStateError):
    pass


class UnwrittenKVError(CacheStateError):
    pass


class AccessViolation(CacheStateError):
    pass


class RevisionConflict(CacheStateError):
    pass


class TransactionStateError(CacheStateError):
    pass


def _require_finite_floating_tensor(value: torch.Tensor, *, label: str) -> None:
    if value.device.type == "meta":
        return
    if (value.is_floating_point() or value.is_complex()) and not bool(
        torch.isfinite(value).all().item()
    ):
        raise ValueError(f"{label} must contain only finite values")


@dataclass(frozen=True, slots=True)
class KVRegion:
    page_id: int
    layer_id: int
    token_offsets: tuple[int, ...]
    revision: int

    def __post_init__(self) -> None:
        if min(self.page_id, self.layer_id, self.revision) < 0:
            raise ValueError("KV region ids and revision must be non-negative")
        if not self.token_offsets:
            raise ValueError("KV region must name at least one token offset")
        if tuple(sorted(set(self.token_offsets))) != self.token_offsets:
            raise ValueError("KV region token offsets must be unique and sorted")
        if self.token_offsets[0] < 0:
            raise ValueError("KV region token offsets must be non-negative")


@dataclass(frozen=True, slots=True)
class RecurrentStateRegion:
    state_id: str
    revision: int

    def __post_init__(self) -> None:
        if not self.state_id or self.revision < 0:
            raise ValueError("recurrent state region needs an id and non-negative revision")


ResourceRegion = TypeAliasType("ResourceRegion", KVRegion | RecurrentStateRegion)


@dataclass(frozen=True, slots=True)
class TxnPlan:
    """Immutable authority for one mutation; runtime status is intentionally absent."""

    transaction_id: str
    owner_id: str
    base_revision: int
    reads: tuple[ResourceRegion, ...]
    writes: tuple[ResourceRegion, ...]
    target_position: int | None = None

    def __post_init__(self) -> None:
        if not self.transaction_id or not self.owner_id or self.base_revision < 0:
            raise ValueError("transaction id, owner, and base revision are required")
        if not self.writes:
            raise ValueError("a state transaction requires an explicit non-empty write set")
        if self.target_position is not None and self.target_position < 0:
            raise ValueError("transaction target position must be non-negative")
        if (
            any(isinstance(region, RecurrentStateRegion) for region in self.writes)
            and self.target_position is None
        ):
            raise ValueError("recurrent writes require an explicit transaction target position")
        read_ids = tuple(_region_identity(region) for region in self.reads)
        write_ids = tuple(_region_identity(region) for region in self.writes)
        if len(set(read_ids)) != len(read_ids):
            raise ValueError("transaction read regions must be unique")
        if len(set(write_ids)) != len(write_ids):
            raise ValueError("transaction write regions must be unique")


class _TxnStatus(StrEnum):
    OPEN = "open"
    TENTATIVE = "tentative"
    VERIFIED = "verified"
    COMMITTED = "committed"
    ABORTED = "aborted"


@dataclass(slots=True)
class _RecurrentState:
    state_id: str
    owner_rank: int
    owner_id: str
    revision: int
    position: int
    value: torch.Tensor
    valid: bool
    slot: _ArenaSlot | None = None

    def clone(self) -> _RecurrentState:
        return _RecurrentState(
            state_id=self.state_id,
            owner_rank=self.owner_rank,
            owner_id=self.owner_id,
            revision=self.revision,
            position=self.position,
            value=self.value.clone(),
            valid=self.valid,
            slot=None,
        )


_ArenaResource = TypeAliasType("_ArenaResource", _KVPage | _RecurrentState)


def _physical_slot(resource: _ArenaResource) -> _ArenaSlot:
    if resource.slot is None:
        raise StaleArenaSlot("arena resource has no physical generation")
    return resource.slot


def _read_slot(resource_id: int | str, source: _ArenaResource) -> ArenaReadSlot:
    slot = _physical_slot(source)
    return ArenaReadSlot(resource_id, slot.index, slot.generation, source.revision)


def _cow_slots(
    resource_id: int | str, source: _ArenaResource, destination: _ArenaResource
) -> ArenaCOWSlots:
    source_slot, destination_slot = _physical_slot(source), _physical_slot(destination)
    return ArenaCOWSlots(
        resource_id,
        source_slot.index,
        source_slot.generation,
        source.revision,
        destination_slot.index,
        destination_slot.generation,
        source.revision + 1,
    )


@dataclass(frozen=True, slots=True)
class RecurrentStateProvision:
    """Detached initial value staged before an owner namespace is published."""

    state_id: str
    value: torch.Tensor
    position: int = 0
    valid: bool = True

    def __post_init__(self) -> None:
        if not self.state_id or self.position < 0:
            raise ValueError("state provision requires an id and non-negative position")
        if not isinstance(self.value, torch.Tensor) or self.value.layout is not torch.strided:
            raise TypeError("state provision requires a strided tensor")
        if self.value.numel() == 0:
            raise ValueError("state provision tensor cannot be empty")
        _require_finite_floating_tensor(self.value, label="state provision tensor")


@dataclass(frozen=True, slots=True)
class QualificationOwnerImage:
    """Detached canonical bytes used only to seed or compare qualification state."""

    owner_id: str
    page_ids: tuple[int, ...]
    key: torch.Tensor
    value: torch.Tensor
    written: torch.Tensor
    recurrent_states: tuple[RecurrentStateProvision, ...]
    revision: int = 0
    page_fields: tuple[tuple[str, torch.Tensor], ...] = ()


@dataclass(frozen=True, slots=True)
class QualificationOwnerLease:
    """Store-issued capability for one qualification-only owner generation."""

    owner_id: str
    page_ids: tuple[int, ...]
    state_ids: tuple[str, ...]
    _nonce: object


@dataclass(slots=True)
class _PreparedPrefixRestore:
    owner_id: str
    part: RankPrefixResidence
    pages: dict[int, _KVPage]
    states: dict[tuple[str, str], _RecurrentState]
    restore: PrefixRestore
    page_holds: tuple[_ArenaSlot, ...]
    state_holds: tuple[tuple[str, _ArenaSlot], ...]
    activated: bool = False


def _region_identity(region: ResourceRegion) -> tuple[object, ...]:
    if isinstance(region, KVRegion):
        return ("kv", region.page_id, region.layer_id)
    return ("state", region.state_id)


def _storage_identity(region: ResourceRegion) -> tuple[str, int | str]:
    if isinstance(region, KVRegion):
        return ("kv", region.page_id)
    return ("state", region.state_id)


class CacheStateStore:
    """Committed state arena shared by reference and performance execution.

    Transactions publish private dynamic objects or fixed-arena slot mappings
    under the read lock, so readers observe only complete bundles.
    """

    def __init__(
        self,
        layout: KVLayout,
        *,
        rank_id: int,
        device: torch.device | str = "cpu",
        arena_spec: CacheArenaSpec | None = None,
        arena: CacheArena | None = None,
    ) -> None:
        if rank_id < 0:
            raise ValueError("cache state rank id must be non-negative")
        self.layout = layout
        self.rank_id = rank_id
        self.device = torch.device(device)
        if arena is not None and arena_spec is not None:
            raise ValueError("pass either a shared cache arena or an arena spec, not both")
        if arena is not None and (
            arena.rank_id != rank_id or arena.layout != layout or arena.device != self.device
        ):
            raise ValueError("shared cache arena differs from the store rank/layout/device domain")
        self._arena = (
            arena
            if arena is not None
            else None
            if arena_spec is None
            else _CacheArena(arena_spec, layout, self.device, rank_id=rank_id)
        )
        self.arena_spec = None if self._arena is None else self._arena.spec
        self._pages: dict[int, _KVPage] = {}
        self._states: dict[tuple[str, str], _RecurrentState] = {}
        self._owner_pages: dict[str, set[int]] = {}
        self._owner_states: dict[str, set[str]] = {}
        self._owner_revisions: dict[str, int] = {}
        self._active_transactions: dict[str, str] = {}
        self._active_by_owner: dict[str, set[str]] = {}
        self._known_transaction_ids: set[str] = set()
        self._known_state_checkpoints: set[str] = set()
        self._prepared_owners: set[str] = set()
        self._qualification_leases: dict[str, QualificationOwnerLease] = {}
        self._next_page_id = 0
        self._lock = RLock()

    @property
    def prefix_lock(self) -> RLock:
        return self._lock

    @property
    def arena_backing_identity(self) -> tuple[int, ...] | None:
        return None if self._arena is None else self._arena.backing_identity

    def allocate_page(self, owner_id: str, *, page_id: int | None = None) -> int:
        self._require_owner_name(owner_id)
        with self._lock:
            self._require_owner_not_prepared(owner_id)
            self._require_no_pending(owner_id)
            resolved = self._next_page_id if page_id is None else page_id
            if resolved < 0 or resolved in self._pages:
                raise ValueError("page id must be non-negative and unused")
            page = (
                _KVPage.zeroed(
                    page_id=resolved,
                    owner_rank=self.rank_id,
                    owner_id=owner_id,
                    layout=self.layout,
                    device=self.device,
                )
                if self._arena is None
                else self._arena.reserve_page(
                    page_id=resolved,
                    owner_rank=self.rank_id,
                    owner_id=owner_id,
                    zero=True,
                )
            )
            self._pages[resolved] = page
            self._owner_pages.setdefault(owner_id, set()).add(resolved)
            self._owner_states.setdefault(owner_id, set())
            self._owner_revisions.setdefault(owner_id, 0)
            self._next_page_id = max(self._next_page_id, resolved + 1)
            return resolved

    def register_recurrent_state(
        self,
        owner_id: str,
        state_id: str,
        value: torch.Tensor,
        *,
        position: int,
        valid: bool = True,
    ) -> None:
        self._require_owner_name(owner_id)
        if not state_id:
            raise ValueError("state id is required")
        if position < 0:
            raise ValueError("initial recurrent state position must be non-negative")
        self._validate_state_tensor(value)
        self._validate_arena_state_tensor(state_id, value)
        with self._lock:
            self._require_owner_not_prepared(owner_id)
            self._require_no_pending(owner_id)
            key = (owner_id, state_id)
            if key in self._states:
                raise ValueError(f"state {state_id!r} already exists for owner {owner_id!r}")
            self._states[key] = self._new_state(owner_id, state_id, value, position, valid)
            self._owner_states.setdefault(owner_id, set()).add(state_id)
            self._owner_pages.setdefault(owner_id, set())
            self._owner_revisions.setdefault(owner_id, 0)

    def provision_owner(
        self,
        owner_id: str,
        recurrent_states: tuple[RecurrentStateProvision, ...] = (),
    ) -> None:
        """Atomically publish a fresh owner and all of its initial state."""

        self._require_owner_name(owner_id)
        if type(recurrent_states) is not tuple or any(
            not isinstance(state, RecurrentStateProvision) for state in recurrent_states
        ):
            raise TypeError("recurrent state provisions must be a tuple")
        state_ids = tuple(state.state_id for state in recurrent_states)
        if len(set(state_ids)) != len(state_ids):
            raise ValueError("provisioned recurrent state ids must be unique")
        for provision in recurrent_states:
            self._validate_state_tensor(provision.value)
            self._validate_arena_state_tensor(provision.state_id, provision.value)
        with self._lock:
            self._require_owner_not_prepared(owner_id)
            if owner_id in self._owner_revisions:
                raise OwnershipError("owner provision requires a fresh namespace")
            self._require_no_pending(owner_id)
            staged: dict[tuple[str, str], _RecurrentState] = {}
            try:
                for provision in recurrent_states:
                    staged[(owner_id, provision.state_id)] = self._new_state(
                        owner_id,
                        provision.state_id,
                        provision.value,
                        provision.position,
                        provision.valid,
                    )
            except Exception:
                release_batch = self._prepare_arena_release((), tuple(staged.values()))
                self._release_prevalidated(release_batch)
                raise
            self._states.update(staged)
            self._owner_revisions[owner_id] = 0
            self._owner_pages[owner_id] = set()
            self._owner_states[owner_id] = set(state_ids)

    def provision_qualification_owner(
        self, image: QualificationOwnerImage
    ) -> QualificationOwnerLease:
        """Publish one fixed-arena owner image without exposing physical slots."""

        with self._lock:
            return self._provision_qualification_owner_locked(image)

    def _provision_qualification_owner_locked(
        self, image: QualificationOwnerImage
    ) -> QualificationOwnerLease:
        if self._arena is None or image.revision != 0:
            raise ValueError("qualification images require a fixed arena at revision zero")
        if (
            not image.owner_id
            or not image.page_ids
            or len(set(image.page_ids)) != len(image.page_ids)
            or any(page_id < 0 for page_id in image.page_ids)
        ):
            raise ValueError("qualification image owner/pages are invalid")
        page_count = len(image.page_ids)
        expected_written = (page_count, len(self.layout.layer_ids), self.layout.block_size_tokens)
        if (
            image.written.dtype is not torch.bool
            or image.written.device != self.device
            or tuple(image.written.shape) != expected_written
            or tuple(image.key.shape) != (page_count, *self.layout.storage_shape(key=True))
            or tuple(image.value.shape) != (page_count, *self.layout.storage_shape(key=False))
        ):
            raise ValueError("qualification image KV differs from its canonical arena ABI")
        if tuple(name for name, _ in image.page_fields) != tuple(
            spec.field_id for spec in self.layout.page_fields
        ):
            raise ValueError("qualification page fields differ from the layout schema")
        for spec, (_, value) in zip(self.layout.page_fields, image.page_fields, strict=True):
            spec.validate(value, self.device, pages=len(image.page_ids))
        for page in range(page_count):
            fields = {name: value[page] for name, value in image.page_fields}
            self.layout.validate_page_storage(
                image.key[page], image.value[page], image.written[page], fields,
                device=self.device,
            )
            for is_key, data in ((True, image.key[page]), (False, image.value[page])):
                grouped = self.layout.key_encoding if is_key else self.layout.value_encoding
                if grouped.spec.dtype.quantized:
                    visible = self.layout.complete_group_mask(image.written[page], key=is_key)
                    mask = visible[..., None].expand_as(data)
                    name = "kv_key_scales" if is_key else "kv_value_scales"
                    if bool((fields[name][~visible] != 0).any()):
                        raise ValueError(
                            "qualification image violates canonical arena ABI: unwritten scales"
                        )
                else:
                    mask = image.written[page][..., None, None].expand_as(data)
                if bool((data[~mask] != 0).any()):
                    raise ValueError(
                        "qualification image violates canonical arena ABI: unwritten storage"
                    )
        state_ids = tuple(state.state_id for state in image.recurrent_states)
        if state_ids != tuple(self._arena.state_specs):
            raise ValueError("qualification image state ids differ from the arena schema")
        allocated: list[int] = []
        provisioned = False
        try:
            self.provision_owner(image.owner_id, image.recurrent_states)
            provisioned = True
            for page_index, page_id in enumerate(image.page_ids):
                allocated.append(self.allocate_page(image.owner_id, page_id=page_id))
                page = self._owned_page(image.owner_id, page_id)
                page.key.copy_(image.key[page_index])
                page.value.copy_(image.value[page_index])
                page.written.copy_(image.written[page_index])
                for name, value in image.page_fields:
                    page.page_fields[name].copy_(value[page_index])
            lease = QualificationOwnerLease(image.owner_id, image.page_ids, state_ids, object())
            self._qualification_leases[image.owner_id] = lease
            return lease
        except BaseException:
            if provisioned:
                self.retire_owner(
                    image.owner_id,
                    expected_revision=0,
                    expected_page_ids=tuple(allocated),
                )
            raise

    def snapshot_qualification_owner(
        self, lease: QualificationOwnerLease
    ) -> QualificationOwnerImage:
        with self._lock:
            if self._qualification_leases.get(lease.owner_id) is not lease:
                raise OwnershipError("qualification owner lease is stale or foreign")
            self._require_no_pending(lease.owner_id)
            pages = tuple(self._owned_page(lease.owner_id, page_id) for page_id in lease.page_ids)
            states = tuple(
                self._owned_state(lease.owner_id, state_id) for state_id in lease.state_ids
            )
            return QualificationOwnerImage(
                lease.owner_id,
                lease.page_ids,
                torch.stack(tuple(page.key for page in pages)).clone(),
                torch.stack(tuple(page.value for page in pages)).clone(),
                torch.stack(tuple(page.written for page in pages)).clone(),
                tuple(
                    RecurrentStateProvision(
                        state.state_id, state.value.clone(), state.position, state.valid
                    )
                    for state in states
                ),
                self.current_revision(lease.owner_id),
                tuple(
                    (
                        spec.field_id,
                        torch.stack(
                            tuple(page.page_fields[spec.field_id] for page in pages)
                        ).clone(),
                    )
                    for spec in self.layout.page_fields
                ),
            )

    def retire_qualification_owner(
        self, lease: QualificationOwnerLease, *, expected_revision: int
    ) -> None:
        with self._lock:
            if self._qualification_leases.get(lease.owner_id) is not lease:
                raise OwnershipError("qualification owner lease is stale or foreign")
            self.retire_owner(
                lease.owner_id,
                expected_revision=expected_revision,
                expected_page_ids=lease.page_ids,
            )

    def rollback_owner_provision(
        self,
        owner_id: str,
        *,
        expected_revision: int,
        expected_page_ids: tuple[int, ...] = (),
    ) -> None:
        """Remove only an exact, unpublished admission-time owner bundle."""
        if expected_revision < 0 or tuple(sorted(set(expected_page_ids))) != expected_page_ids:
            raise ValueError("rollback revision/pages must be non-negative, unique, and sorted")
        self.retire_owner(
            owner_id,
            expected_revision=expected_revision,
            expected_page_ids=expected_page_ids,
        )

    def current_revision(self, owner_id: str) -> int:
        with self._lock:
            try:
                return self._owner_revisions[owner_id]
            except KeyError as error:
                raise OwnershipError(f"unknown state owner {owner_id!r}") from error

    def page_revision(self, owner_id: str, page_id: int) -> int:
        with self._lock:
            return self._owned_page(owner_id, page_id).revision

    def state_revision(self, owner_id: str, state_id: str) -> int:
        with self._lock:
            return self._owned_state(owner_id, state_id).revision

    def state_position(self, owner_id: str, state_id: str) -> int:
        with self._lock:
            return self._owned_state(owner_id, state_id).position

    def kv_region(
        self,
        owner_id: str,
        page_id: int,
        layer_id: int,
        token_offsets: tuple[int, ...],
    ) -> KVRegion:
        with self._lock:
            page = self._owned_page(owner_id, page_id)
            region = KVRegion(page_id, layer_id, token_offsets, page.revision)
            self._validate_kv_region(region)
            return region

    def recurrent_region(self, owner_id: str, state_id: str) -> RecurrentStateRegion:
        with self._lock:
            state = self._owned_state(owner_id, state_id)
            return RecurrentStateRegion(state_id=state_id, revision=state.revision)

    def plan(
        self,
        transaction_id: str,
        owner_id: str,
        *,
        reads: tuple[ResourceRegion, ...] = (),
        writes: tuple[ResourceRegion, ...],
        target_position: int | None = None,
    ) -> TxnPlan:
        return TxnPlan(
            transaction_id=transaction_id,
            owner_id=owner_id,
            base_revision=self.current_revision(owner_id),
            reads=reads,
            writes=writes,
            target_position=target_position,
        )

    def begin(self, plan: TxnPlan) -> StateTransaction:
        with self._lock:
            if plan.transaction_id in self._known_transaction_ids:
                raise ValueError(f"transaction id {plan.transaction_id!r} was already used")
            self._validate_plan_current(plan)
            if self._arena is not None and plan.owner_id in self._active_by_owner:
                raise TransactionStateError(
                    "fixed cache arena permits one active mutation per owner"
                )
            self._known_transaction_ids.add(plan.transaction_id)
            self._activate_transaction(plan.transaction_id, plan.owner_id)
            transaction = StateTransaction(self, plan)
            try:
                transaction._reserve_arena_staging()
            except Exception:
                try:
                    transaction._release_arena_staging()
                finally:
                    self._deactivate_transaction(plan.transaction_id)
                    self._known_transaction_ids.discard(plan.transaction_id)
                raise
            return transaction

    def commit_bundle(self, transactions: tuple[StateTransaction, ...]) -> tuple[int, ...]:
        """Publish a verified rank-local batch as one all-or-none bundle.

        Dynamic storage publishes private COW objects. Arena storage publishes
        graph/eager-written physical slots. Readers observe the old or new
        complete bundle under this same lock.
        """

        if type(transactions) is not tuple or not transactions:
            raise ValueError("commit bundle requires a non-empty transaction tuple")
        if any(not isinstance(transaction, StateTransaction) for transaction in transactions):
            raise TypeError("commit bundle accepts only StateTransaction values")
        if any(transaction._store is not self for transaction in transactions):
            raise ValueError("all bundled transactions must belong to this store")
        transaction_ids = tuple(transaction.plan.transaction_id for transaction in transactions)
        owner_ids = tuple(transaction.plan.owner_id for transaction in transactions)
        if len(set(transaction_ids)) != len(transaction_ids):
            raise ValueError("commit bundle transaction ids must be unique")
        if len(set(owner_ids)) != len(owner_ids):
            raise ValueError("a commit bundle may mutate each owner at most once")

        with self._lock:
            for transaction in transactions:
                if transaction._status is not _TxnStatus.VERIFIED:
                    raise TransactionStateError(
                        "every bundled transaction must be verified before commit"
                    )
                if (
                    self._active_transactions.get(transaction.plan.transaction_id)
                    != transaction.plan.owner_id
                    or transaction.plan.transaction_id
                    not in self._active_by_owner.get(transaction.plan.owner_id, set())
                ):
                    raise TransactionStateError(
                        "bundled transaction is not the store's active owner mutation"
                    )
                self._validate_plan_current(transaction.plan)
                transaction._validate_staged_payload()

            old_pages = tuple(
                self._pages[page_id]
                for transaction in transactions
                for page_id in transaction._staged_pages
            )
            old_states = tuple(
                self._states[(transaction.plan.owner_id, state_id)]
                for transaction in transactions
                for state_id in transaction._staged_states
            )
            release_batch = self._prepare_arena_release(
                old_pages,
                old_states,
                tuple(
                    hold
                    for transaction in transactions
                    if (hold := transaction._scratch_hold) is not None
                ),
            )
            for transaction in transactions:
                for page_id, page in transaction._staged_pages.items():
                    page.revision += 1
                    self._pages[page_id] = page
                for state_id, state in transaction._staged_states.items():
                    state.revision += 1
                    self._states[(transaction.plan.owner_id, state_id)] = state
                self._owner_revisions[transaction.plan.owner_id] = (
                    transaction.plan.base_revision + 1
                )
                self._deactivate_transaction(transaction.plan.transaction_id)
            self._release_prevalidated(release_batch)
            for transaction in transactions:
                transaction._scratch_hold = None
                transaction._status = _TxnStatus.COMMITTED
                transaction._discard_staging()

        return tuple(transaction.plan.base_revision + 1 for transaction in transactions)

    def page_written_mask(self, owner_id: str, page_id: int) -> torch.Tensor:
        with self._lock:
            return self._owned_page(owner_id, page_id).written.clone()

    def unwritten_storage_is_zero(self, owner_id: str, page_id: int) -> bool:
        """Check allocation/rollback hygiene without exposing unreadable KV values."""

        with self._lock:
            page = self._owned_page(owner_id, page_id)
            unwritten = ~page.written
            return bool(
                torch.all(page.key[unwritten] == 0).item()
                and torch.all(page.value[unwritten] == 0).item()
            )

    def read_kv(
        self,
        owner_id: str,
        page_id: int,
        layer_id: int,
        token_offsets: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        region = KVRegion(page_id, layer_id, token_offsets, self.page_revision(owner_id, page_id))
        with self._lock:
            page = self._owned_page(owner_id, page_id)
            return self._read_page(page, region)

    def read_recurrent_state(
        self,
        owner_id: str,
        state_id: str,
        *,
        require_valid: bool = True,
    ) -> torch.Tensor:
        with self._lock:
            state = self._owned_state(owner_id, state_id)
            if require_valid and not state.valid:
                raise CacheStateError(f"recurrent state {state_id!r} is not arithmetically valid")
            return state.value.clone()

    def checkpoint_recurrent_state(self, owner_id: str, state_id: str) -> RecurrentStateSnapshot:
        """Issue a detached, tamper-evident checkpoint of committed state."""

        with self._lock:
            self._require_no_pending(owner_id)
            checkpoint = self._snapshot_state(self._owned_state(owner_id, state_id))
            self._known_state_checkpoints.add(checkpoint.content_digest)
            return checkpoint

    def capture_prefix(
        self,
        *,
        snapshot_id: str,
        prefix_key: str,
        owner_id: str,
        page_ids: tuple[int, ...],
        state_ids: tuple[str, ...],
        physical_block_boundary: int,
        arithmetic_state_commit_boundary: int,
        state_commit_alignment: int,
        replay_tokens: tuple[int, ...],
        recurrent_checkpoints: tuple[RecurrentStateSnapshot, ...] = (),
    ) -> PrefixSnapshot:
        with self._lock:
            self._require_no_pending(owner_id)
            if len(set(page_ids)) != len(page_ids):
                raise ValueError("prefix page ids must be unique")
            if len(set(state_ids)) != len(state_ids):
                raise ValueError("prefix state ids must be unique")
            checkpoint_ids = tuple(checkpoint.state_id for checkpoint in recurrent_checkpoints)
            if len(set(checkpoint_ids)) != len(checkpoint_ids):
                raise ValueError("prefix recurrent checkpoints must have unique state ids")
            if not set(checkpoint_ids) <= set(state_ids):
                raise ValueError("prefix checkpoint references an undeclared recurrent state")
            pages = tuple(
                self._snapshot_page(self._owned_page(owner_id, page_id)) for page_id in page_ids
            )
            checkpoint_by_id = {
                checkpoint.state_id: checkpoint for checkpoint in recurrent_checkpoints
            }
            states: list[RecurrentStateSnapshot] = []
            for state_id in state_ids:
                current = self._owned_state(owner_id, state_id)
                checkpoint = checkpoint_by_id.get(state_id)
                if checkpoint is None:
                    states.append(self._snapshot_state(current))
                    continue
                checkpoint.validate()
                if checkpoint.content_digest not in self._known_state_checkpoints:
                    raise ValueError("recurrent checkpoint was not issued by this store")
                if checkpoint.owner_rank != self.rank_id or checkpoint.owner_id != owner_id:
                    raise OwnershipError("recurrent checkpoint owner does not match capture")
                if checkpoint.revision > current.revision:
                    raise RevisionConflict("recurrent checkpoint revision is from the future")
                if checkpoint.position > current.position:
                    raise ValueError("recurrent checkpoint position is ahead of current state")
                if checkpoint.value.dtype is not current.value.dtype or tuple(
                    checkpoint.value.shape
                ) != tuple(current.value.shape):
                    raise ValueError("recurrent checkpoint dtype or shape changed")
                states.append(checkpoint.clone())
            source_revision = self.current_revision(owner_id)
        return seal_prefix_snapshot(
            snapshot_id=snapshot_id,
            prefix_key=prefix_key,
            source_rank=self.rank_id,
            source_owner_id=owner_id,
            source_revision=source_revision,
            kv_layout=self.layout,
            physical_block_boundary=physical_block_boundary,
            arithmetic_state_commit_boundary=arithmetic_state_commit_boundary,
            state_commit_alignment=state_commit_alignment,
            replay_tokens=replay_tokens,
            pages=pages,
            recurrent_states=tuple(states),
        )

    def restore_prefix(
        self,
        snapshot: PrefixSnapshot,
        *,
        expected_prefix_key: str,
        owner_id: str,
    ) -> PrefixRestore:
        """Atomically materialize a sealed bundle into an unused owner namespace."""

        self._require_owner_name(owner_id)
        snapshot.validate()
        if snapshot.prefix_key != expected_prefix_key:
            raise ValueError("prefix identity does not match the requested prefix key")
        if snapshot.source_rank != self.rank_id:
            raise OwnershipError(
                f"rank {self.rank_id} cannot restore rank {snapshot.source_rank} state"
            )
        if snapshot.kv_layout != self.layout:
            raise ValueError("prefix KV layout does not match this cache store")

        with self._lock:
            if owner_id in self._owner_revisions:
                raise OwnershipError("prefix restore requires a fresh owner namespace")
            self._require_no_pending(owner_id)
            if self._arena is not None:
                raise TransactionStateError(
                    "prefix restore remains a detached reference operation in arena stage"
                )

            next_page_id = self._next_page_id
            staged_pages: dict[int, _KVPage] = {}
            page_map: list[tuple[int, int]] = []
            staged_states: dict[tuple[str, str], _RecurrentState] = {}
            for page_image in snapshot.pages:
                restored_id = next_page_id
                next_page_id += 1
                restored_page = _KVPage(
                    page_id=restored_id,
                    owner_rank=self.rank_id,
                    owner_id=owner_id,
                    revision=page_image.revision,
                    key=page_image.key.to(device=self.device).clone(),
                    value=page_image.value.to(device=self.device).clone(),
                    written=page_image.written.to(device=self.device).clone(),
                    page_fields={
                        name: value.to(device=self.device).clone()
                        for name, value in page_image.page_fields.items()
                    },
                )
                self._validate_page_payload(restored_page, expected_owner_id=owner_id)
                staged_pages[restored_id] = restored_page
                page_map.append((page_image.page_id, restored_id))

            for state_image in snapshot.recurrent_states:
                value = state_image.value.to(device=self.device).clone()
                self._validate_state_tensor(value)
                staged_states[(owner_id, state_image.state_id)] = _RecurrentState(
                    state_id=state_image.state_id,
                    owner_rank=self.rank_id,
                    owner_id=owner_id,
                    revision=state_image.revision,
                    position=state_image.position,
                    value=value,
                    valid=state_image.valid,
                )

            self._pages.update(staged_pages)
            self._states.update(staged_states)
            self._owner_revisions[owner_id] = snapshot.source_revision
            self._owner_pages[owner_id] = set(staged_pages)
            self._owner_states[owner_id] = {state.state_id for state in snapshot.recurrent_states}
            self._next_page_id = next_page_id

        return PrefixRestore(
            source_rank=snapshot.source_rank,
            target_rank=self.rank_id,
            owner_id=owner_id,
            owner_revision=snapshot.source_revision,
            source_to_restored_pages=tuple(page_map),
            state_ids=tuple(state.state_id for state in snapshot.recurrent_states),
            state_positions=tuple(
                (state.state_id, state.position) for state in snapshot.recurrent_states
            ),
            state_valid_mask=snapshot.state_valid_mask,
            physical_block_boundary=snapshot.physical_block_boundary,
            arithmetic_state_commit_boundary=snapshot.arithmetic_state_commit_boundary,
            replay_tokens=snapshot.replay_tokens,
        )

    def pin_prefix_part(
        self,
        source: RankPrefixSource,
        *,
        token_boundary: int,
    ) -> RankPrefixResidence:
        """Retain one complete, joint-boundary arena part for directory ownership."""

        if self._arena is None:
            raise TransactionStateError("physical prefix residence requires a cache arena")
        if source.rank_id != self.rank_id or source.owner_revision < 0:
            raise OwnershipError("physical prefix source rank/revision differs from its store")
        if token_boundary <= 0 or token_boundary % self.layout.block_size_tokens:
            raise ValueError("physical prefix boundary must be positive and block aligned")
        if len(source.page_ids) != token_boundary // self.layout.block_size_tokens:
            raise ValueError("physical prefix page count differs from its exact boundary")
        if len(set(source.page_ids)) != len(source.page_ids):
            raise ValueError("physical prefix page ids must be unique")
        expected_states = tuple(self._arena.state_specs)
        if source.state_ids != expected_states:
            raise ValueError("physical prefix must include every arena state component")
        with self._lock:
            self._require_no_pending(source.owner_id)
            if self.current_revision(source.owner_id) != source.owner_revision:
                raise RevisionConflict("physical prefix source revision changed")
            pages = tuple(self._owned_page(source.owner_id, page_id) for page_id in source.page_ids)
            states = tuple(
                self._owned_state(source.owner_id, state_id) for state_id in source.state_ids
            )
            if any(not bool(torch.all(page.written).item()) for page in pages):
                raise UnwrittenKVError("physical prefix requires fully written KV pages")
            if any(not state.valid or state.position != token_boundary for state in states):
                raise ValueError("physical prefix requires every state at the same exact boundary")
            content_digest = self._physical_prefix_digest(pages, states, token_boundary)
            page_holds, state_holds = self._retain_prefix_refs(
                tuple(_physical_slot(page) for page in pages),
                tuple((state.state_id, _physical_slot(state)) for state in states),
            )
            return RankPrefixResidence(
                rank_id=self.rank_id,
                arena_id=self._arena.spec.arena_id,
                arena_domain=self._arena.domain,
                backing_identity=self._arena.backing_identity,
                source_owner_id=source.owner_id,
                source_revision=source.owner_revision,
                token_boundary=token_boundary,
                content_digest=content_digest,
                pages=tuple(
                    PhysicalPageRef(page.page_id, page.revision, hold)
                    for page, hold in zip(pages, page_holds, strict=True)
                ),
                states=tuple(
                    PhysicalStateRef(
                        state.state_id,
                        state.revision,
                        state.position,
                        state.valid,
                        hold,
                    )
                    for state, (_, hold) in zip(states, state_holds, strict=True)
                ),
            )

    def release_prefix_part(self, part: RankPrefixResidence) -> None:
        arena = self._require_prefix_part_domain(part)
        with self._lock:
            arena.release_prevalidated(
                arena.prepare_release(
                    tuple(page.slot for page in part.pages),
                    tuple((state.state_id, state.slot) for state in part.states),
                    (),
                )
            )

    def validate_prefix_part(self, part: RankPrefixResidence) -> None:
        arena = self._require_prefix_part_domain(part)
        with self._lock:
            for page in part.pages:
                arena.validate_page_ref(page.slot)
            for state in part.states:
                arena.validate_state_ref(state.state_id, state.slot)
            if any(
                not bool(torch.all(arena.written[page.slot.index]).item()) for page in part.pages
            ):
                raise UnwrittenKVError("physical prefix part contains partial KV")

    def prepare_prefix_restore(
        self,
        part: RankPrefixResidence,
        owner_id: str,
    ) -> _PreparedPrefixRestore:
        arena = self._require_prefix_part_domain(part)
        self._require_owner_name(owner_id)
        with self._lock:
            if owner_id in self._owner_revisions or owner_id in self._prepared_owners:
                raise OwnershipError("physical prefix restore requires a fresh owner namespace")
            self.validate_prefix_part(part)
            page_holds, state_holds = self._retain_prefix_refs(
                tuple(page.slot for page in part.pages),
                tuple((state.state_id, state.slot) for state in part.states),
            )

            next_page_id = self._next_page_id
            staged_pages: dict[int, _KVPage] = {}
            page_map: list[tuple[int, int]] = []
            for source_page, hold in zip(part.pages, page_holds, strict=True):
                target_page_id = next_page_id
                next_page_id += 1
                staged_pages[target_page_id] = _KVPage(
                    target_page_id,
                    self.rank_id,
                    owner_id,
                    source_page.revision,
                    arena.key[hold.index],
                    arena.value[hold.index],
                    arena.written[hold.index],
                    hold,
                    {name: value[hold.index] for name, value in arena.page_fields.items()},
                )
                page_map.append((source_page.source_page_id, target_page_id))
            staged_states = {
                (owner_id, source_state.state_id): _RecurrentState(
                    source_state.state_id,
                    self.rank_id,
                    owner_id,
                    source_state.revision,
                    source_state.position,
                    arena.states[source_state.state_id][hold.index],
                    source_state.valid,
                    hold,
                )
                for source_state, (_, hold) in zip(part.states, state_holds, strict=True)
            }
            prepared = _PreparedPrefixRestore(
                owner_id=owner_id,
                part=part,
                pages=staged_pages,
                states=staged_states,
                restore=PrefixRestore(
                    source_rank=part.rank_id,
                    target_rank=self.rank_id,
                    owner_id=owner_id,
                    owner_revision=part.source_revision,
                    source_to_restored_pages=tuple(page_map),
                    state_ids=tuple(state.state_id for state in part.states),
                    state_positions=tuple(
                        (state.state_id, state.position) for state in part.states
                    ),
                    state_valid_mask=tuple(state.valid for state in part.states),
                    physical_block_boundary=part.token_boundary,
                    arithmetic_state_commit_boundary=part.token_boundary,
                    replay_tokens=(),
                ),
                page_holds=tuple(page_holds),
                state_holds=tuple(state_holds),
            )
            self._next_page_id = next_page_id
            self._prepared_owners.add(owner_id)
            return prepared

    def validate_prepared_prefix_restore(self, prepared: _PreparedPrefixRestore) -> None:
        with self._lock:
            if prepared.activated or prepared.owner_id not in self._prepared_owners:
                raise TransactionStateError("physical prefix restore preparation is stale")
            if prepared.owner_id in self._owner_revisions:
                raise OwnershipError("prepared physical prefix owner became visible early")
            arena = self._require_prefix_part_domain(prepared.part)
            for slot in prepared.page_holds:
                arena._page_pool.validate(slot)
            for state_id, slot in prepared.state_holds:
                arena.validate_state(state_id, slot)

    def activate_prefix_restore(self, prepared: _PreparedPrefixRestore) -> PrefixRestore:
        """Publish only pointer swaps after all topology ranks were prevalidated."""

        with self._lock:
            self.validate_prepared_prefix_restore(prepared)
            try:
                self._publish_prepared_delta(prepared)
            except Exception:
                self._rollback_prepared_metadata(prepared)
                raise
            self._prepared_owners.remove(prepared.owner_id)
            prepared.activated = True
            return prepared.restore

    def abort_prefix_restore(self, prepared: _PreparedPrefixRestore) -> None:
        with self._lock:
            if prepared.activated:
                self.retire_owner(
                    prepared.owner_id,
                    expected_revision=prepared.restore.owner_revision,
                    expected_page_ids=tuple(
                        target for _, target in prepared.restore.source_to_restored_pages
                    ),
                )
                return
            if prepared.owner_id not in self._prepared_owners:
                return
            arena = self._require_prefix_part_domain(prepared.part)
            self._rollback_prepared_metadata(prepared)
            batch = arena.prepare_release(prepared.page_holds, prepared.state_holds, ())
            self._prepared_owners.remove(prepared.owner_id)
            arena.release_prevalidated(batch)

    def _publish_prepared_delta(self, prepared: _PreparedPrefixRestore) -> None:
        self._pages.update(prepared.pages)
        self._states.update(prepared.states)
        self._owner_revisions[prepared.owner_id] = prepared.part.source_revision
        self._owner_pages[prepared.owner_id] = set(prepared.pages)
        self._owner_states[prepared.owner_id] = {state_id for _, state_id in prepared.states}

    def _rollback_prepared_metadata(self, prepared: _PreparedPrefixRestore) -> None:
        for page_id, page in prepared.pages.items():
            if self._pages.get(page_id) is page:
                del self._pages[page_id]
        for key, state in prepared.states.items():
            if self._states.get(key) is state:
                del self._states[key]
        expected_pages = set(prepared.pages)
        expected_states = {state_id for _, state_id in prepared.states}
        if self._owner_pages.get(prepared.owner_id) == expected_pages:
            del self._owner_pages[prepared.owner_id]
        if self._owner_states.get(prepared.owner_id) == expected_states:
            del self._owner_states[prepared.owner_id]
        if self._owner_revisions.get(prepared.owner_id) == prepared.part.source_revision:
            del self._owner_revisions[prepared.owner_id]

    def retire_owner(
        self,
        owner_id: str,
        *,
        expected_revision: int,
        expected_page_ids: tuple[int, ...],
    ) -> None:
        """Release an exact owner bundle after its external completion barrier."""

        self._require_owner_name(owner_id)
        if expected_revision < 0 or len(set(expected_page_ids)) != len(expected_page_ids):
            raise ValueError("owner retirement revision/pages are invalid")
        with self._lock:
            self._require_owner_not_prepared(owner_id)
            self._require_no_pending(owner_id)
            if self.current_revision(owner_id) != expected_revision:
                raise RevisionConflict("owner revision changed before retirement")
            page_ids = self._owner_pages[owner_id]
            if page_ids != set(expected_page_ids):
                raise OwnershipError("owner pages differ from the exact retirement set")
            owned_pages = tuple(self._pages[page_id] for page_id in expected_page_ids)
            state_ids = self._owner_states[owner_id]
            owned_states = tuple(
                self._states[(owner_id, state_id)] for state_id in sorted(state_ids)
            )
            release_batch = self._prepare_arena_release(owned_pages, owned_states)
            for page_id in expected_page_ids:
                del self._pages[page_id]
            for state_id in state_ids:
                del self._states[(owner_id, state_id)]
            del self._owner_pages[owner_id]
            del self._owner_states[owner_id]
            del self._owner_revisions[owner_id]
            self._qualification_leases.pop(owner_id, None)
            self._release_prevalidated(release_batch)

    def _require_prefix_part_domain(self, part: RankPrefixResidence) -> CacheArena:
        if self._arena is None:
            raise TransactionStateError("physical prefix requires a cache arena")
        if (
            part.rank_id != self.rank_id
            or part.arena_id != self._arena.spec.arena_id
            or part.arena_domain != self._arena.domain
            or part.backing_identity != self._arena.backing_identity
        ):
            raise OwnershipError("physical prefix part differs from its rank-global arena")
        if part.token_boundary <= 0 or part.token_boundary % self.layout.block_size_tokens:
            raise ValueError("physical prefix part is not exact-block aligned")
        if len(part.pages) != part.token_boundary // self.layout.block_size_tokens or len(
            {page.source_page_id for page in part.pages}
        ) != len(part.pages):
            raise ValueError("physical prefix part has an incomplete or aliased KV component")
        if tuple(state.state_id for state in part.states) != tuple(self._arena.state_specs):
            raise ValueError("physical prefix part is missing an arena state component")
        if any(not state.valid or state.position != part.token_boundary for state in part.states):
            raise ValueError("physical prefix state is not valid at the joint exact boundary")
        return self._arena

    def _retain_prefix_refs(
        self,
        page_refs: tuple[_ArenaSlot, ...],
        state_refs: tuple[tuple[str, _ArenaSlot], ...],
    ) -> tuple[tuple[_ArenaSlot, ...], tuple[tuple[str, _ArenaSlot], ...]]:
        assert self._arena is not None
        pages: list[_ArenaSlot] = []
        states: list[tuple[str, _ArenaSlot]] = []
        try:
            pages.extend(self._arena.retain_page(ref) for ref in page_refs)
            states.extend(
                (state_id, self._arena.retain_state(state_id, ref)) for state_id, ref in state_refs
            )
        except Exception:
            batch = self._arena.prepare_release(tuple(pages), tuple(states), ())
            self._arena.release_prevalidated(batch)
            raise
        return tuple(pages), tuple(states)

    def _physical_prefix_digest(
        self,
        pages: tuple[_KVPage, ...],
        states: tuple[_RecurrentState, ...],
        boundary: int,
    ) -> str:
        digest = hashlib.blake2s()
        digest.update(repr((self.rank_id, self.layout, self.arena_spec, boundary)).encode())
        for ordinal, page in enumerate(pages):
            digest.update(repr(ordinal).encode())
            _hash_tensor(digest, page.key)
            _hash_tensor(digest, page.value)
            _hash_tensor(digest, page.written)
            for name, value in sorted(page.page_fields.items()):
                digest.update(name.encode())
                _hash_tensor(digest, value)
        for state in states:
            digest.update(repr((state.state_id, state.position, state.valid)).encode())
            _hash_tensor(digest, state.value)
        return f"b2s256:{digest.hexdigest()}"

    def _validate_plan_current(self, plan: TxnPlan) -> None:
        try:
            owner_revision = self._owner_revisions[plan.owner_id]
        except KeyError as error:
            raise OwnershipError(f"unknown state owner {plan.owner_id!r}") from error
        if owner_revision != plan.base_revision:
            raise RevisionConflict(
                f"owner revision changed: planned {plan.base_revision}, current {owner_revision}"
            )
        seen_storage_revisions: dict[tuple[str, int | str], int] = {}
        for region in (*plan.reads, *plan.writes):
            if isinstance(region, KVRegion):
                self._validate_kv_region(region)
                current = self._owned_page(plan.owner_id, region.page_id).revision
            else:
                current = self._owned_state(plan.owner_id, region.state_id).revision
            if current != region.revision:
                raise RevisionConflict(
                    f"resource {_storage_identity(region)!r} changed: "
                    f"planned {region.revision}, current {current}"
                )
            storage_id = _storage_identity(region)
            previous = seen_storage_revisions.setdefault(storage_id, region.revision)
            if previous != region.revision:
                raise ValueError("one transaction declared conflicting resource revisions")

    def _validate_kv_region(self, region: KVRegion) -> None:
        self.layout.layer_index(region.layer_id)
        if region.token_offsets[-1] >= self.layout.block_size_tokens:
            raise ValueError("KV region token offset exceeds its physical block")

    def _read_page(self, page: _KVPage, region: KVRegion) -> tuple[torch.Tensor, torch.Tensor]:
        if self.layout.storage.quantized:
            raise ValueError("encoded KV reads require an explicit native-tail visibility window")
        layer_index = self.layout.layer_index(region.layer_id)
        offsets = torch.tensor(region.token_offsets, dtype=torch.long, device=self.device)
        if not bool(torch.all(page.written[layer_index, offsets]).item()):
            raise UnwrittenKVError("unwritten KV positions are never readable")
        return (
            page.key[layer_index, offsets].clone(),
            page.value[layer_index, offsets].clone(),
        )

    def _owned_page(self, owner_id: str, page_id: int) -> _KVPage:
        try:
            page = self._pages[page_id]
        except KeyError as error:
            raise OwnershipError(f"unknown KV page {page_id}") from error
        if page.owner_id != owner_id:
            raise OwnershipError(f"owner {owner_id!r} does not own KV page {page_id}")
        if page.owner_rank != self.rank_id:
            raise OwnershipError(f"rank {self.rank_id} does not own KV page {page_id}")
        if tuple(page.page_fields) != tuple(spec.field_id for spec in self.layout.page_fields):
            raise ValueError("page fields differ from the layout schema")
        for spec in self.layout.page_fields:
            spec.validate(page.page_fields[spec.field_id], self.device)
        if self._arena is not None:
            self._arena.validate_page(page)
        return page

    def _owned_state(self, owner_id: str, state_id: str) -> _RecurrentState:
        try:
            state = self._states[(owner_id, state_id)]
        except KeyError as error:
            raise OwnershipError(
                f"owner {owner_id!r} does not own recurrent state {state_id!r}"
            ) from error
        if state.owner_rank != self.rank_id:
            raise OwnershipError(f"rank {self.rank_id} does not own recurrent state {state_id!r}")
        if self._arena is not None:
            if state.slot is None:
                raise StaleArenaSlot("arena state mapping has no physical generation")
            self._arena.validate_state(state_id, state.slot)
        return state

    def _require_no_pending(self, owner_id: str) -> None:
        if owner_id in self._active_by_owner:
            raise TransactionStateError(
                f"owner {owner_id!r} has tentative state and cannot publish or provision"
            )

    def _require_owner_not_prepared(self, owner_id: str) -> None:
        if owner_id in self._prepared_owners:
            raise TransactionStateError(
                f"owner {owner_id!r} is reserved by a physical prefix restore"
            )

    def _activate_transaction(self, transaction_id: str, owner_id: str) -> None:
        self._active_transactions[transaction_id] = owner_id
        self._active_by_owner.setdefault(owner_id, set()).add(transaction_id)

    def _deactivate_transaction(self, transaction_id: str) -> None:
        owner_id = self._active_transactions.pop(transaction_id)
        owner_transactions = self._active_by_owner[owner_id]
        owner_transactions.remove(transaction_id)
        if not owner_transactions:
            del self._active_by_owner[owner_id]

    @staticmethod
    def _require_owner_name(owner_id: str) -> None:
        if not owner_id:
            raise ValueError("state owner id is required")

    def _validate_state_tensor(self, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor) or value.layout is not torch.strided:
            raise TypeError("recurrent state must be a strided torch.Tensor")
        if value.device != self.device:
            raise ValueError(
                f"recurrent state device {value.device} does not match store {self.device}"
            )
        if value.numel() == 0:
            raise ValueError("recurrent state tensors must be non-empty")

        _require_finite_floating_tensor(value, label="recurrent state tensor")

    def _validate_arena_state_tensor(self, state_id: str, value: torch.Tensor) -> None:
        if self._arena is None:
            return
        spec = self._arena.state_spec(state_id)
        if value.dtype is not spec.dtype or tuple(value.shape) != spec.shape:
            raise ValueError(f"state {state_id!r} differs from its cache arena schema")

    def _new_state(
        self,
        owner_id: str,
        state_id: str,
        value: torch.Tensor,
        position: int,
        valid: bool,
    ) -> _RecurrentState:
        slot: _ArenaSlot | None = None
        if self._arena is None:
            owned = value.detach().clone()
        else:
            slot, owned = self._arena.reserve_state(state_id, source=value.detach())
        self._validate_state_tensor(owned)
        return _RecurrentState(state_id, self.rank_id, owner_id, 0, position, owned, valid, slot)

    def _prepare_arena_release(
        self,
        pages: tuple[_KVPage, ...],
        states: tuple[_RecurrentState, ...],
        scratch_holds: tuple[_ScratchParticipantHold, ...] = (),
    ) -> _ArenaReleaseBatch | None:
        if self._arena is None:
            if scratch_holds:
                raise TransactionStateError("dynamic cache storage cannot own graph scratch")
            return None
        try:
            page_slots = tuple(_physical_slot(page) for page in pages)
            state_slots = tuple((state.state_id, _physical_slot(state)) for state in states)
        except StaleArenaSlot as error:
            raise self._arena._poison("release plan", error) from error
        return self._arena.prepare_release(page_slots, state_slots, scratch_holds)

    def _release_prevalidated(self, batch: _ArenaReleaseBatch | None) -> None:
        if batch is not None:
            assert self._arena is not None
            self._arena.release_prevalidated(batch)

    def _validate_page_payload(self, page: _KVPage, *, expected_owner_id: str) -> None:
        if page.owner_rank != self.rank_id or page.owner_id != expected_owner_id:
            raise OwnershipError("KV page payload owner differs from its store namespace")
        if page.page_id < 0 or page.revision < 0:
            raise ValueError("KV page payload ids and revision must be non-negative")
        if (
            page.key.layout is not torch.strided
            or page.value.layout is not torch.strided
            or page.written.layout is not torch.strided
        ):
            raise TypeError("KV page payloads must be strided tensors")
        self.layout.validate_page_storage(
            page.key, page.value, page.written, page.page_fields, device=self.device,
        )
        if page.written.dtype is not torch.bool:
            raise ValueError("KV page written mask must remain boolean")
        if page.key.device != self.device or page.value.device != self.device:
            raise ValueError("KV page payload device must match the cache store")
        if page.written.device != self.device:
            raise ValueError("KV page written-mask device must match the cache store")
        if (
            tuple(page.key.shape) != self.layout.storage_shape(key=True)
            or tuple(page.value.shape) != self.layout.storage_shape(key=False)
        ):
            raise ValueError("KV page payload shape differs from the store layout")
        expected_mask_shape = (len(self.layout.layer_ids), self.layout.block_size_tokens)
        if tuple(page.written.shape) != expected_mask_shape:
            raise ValueError("KV page written-mask shape differs from the store layout")
        if not bool(torch.isfinite(page.key).all().item()) or not bool(
            torch.isfinite(page.value).all().item()
        ):
            raise ValueError("committable attention KV must contain only finite values")
        if self._arena is not None:
            self._arena.validate_page(page)

    def _snapshot_page(self, page: _KVPage) -> KVPageSnapshot:
        self._validate_page_payload(page, expected_owner_id=page.owner_id)
        return KVPageSnapshot(
            page_id=page.page_id,
            owner_rank=page.owner_rank,
            owner_id=page.owner_id,
            revision=page.revision,
            key=page.key.clone(),
            value=page.value.clone(),
            written=page.written.clone(),
            page_fields={name: value.clone() for name, value in page.page_fields.items()},
        )

    def _snapshot_state(self, state: _RecurrentState) -> RecurrentStateSnapshot:
        self._validate_state_tensor(state.value)
        return seal_recurrent_state_snapshot(
            state_id=state.state_id,
            owner_rank=state.owner_rank,
            owner_id=state.owner_id,
            revision=state.revision,
            position=state.position,
            value=state.value.clone(),
            valid=state.valid,
        )


class StateTransaction:
    """Private COW state plus a read-only runtime-owned status machine."""

    def __init__(self, store: CacheStateStore, plan: TxnPlan) -> None:
        self._store = store
        self.plan = plan
        self._status = _TxnStatus.OPEN
        self._staged_pages: dict[int, _KVPage] = {}
        self._staged_states: dict[str, _RecurrentState] = {}
        self._written_offsets: dict[tuple[object, ...], set[int]] = {}
        self._written_states: set[str] = set()
        self._initialized_pages: set[int] = set()
        self._graph_completed = False
        self._scratch_hold: _ScratchParticipantHold | None = None

    @property
    def status(self) -> str:
        return self._status.value

    def graph_binding(self, launch_plan: ec.LaunchPlan) -> CacheArenaBinding:
        """Bind all reserved tentative slots while the transaction is still open."""

        if self._status is not _TxnStatus.OPEN:
            raise TransactionStateError("graph binding is available only for an open transaction")
        if not isinstance(launch_plan, ec.LaunchPlan):
            raise TypeError("graph binding requires a typed LaunchPlan")
        with self._store._lock:
            self._ensure_current_or_abort()
            arena = self._store._arena
            if arena is None:
                raise TransactionStateError("graph binding requires fixed cache arena mode")
            rank_resources = tuple(
                resource
                for resource in launch_plan.replay_lease.rank_resources
                if resource.rank == self._store.rank_id
            )
            if len(rank_resources) != 1:
                raise ScratchArenaLeaseError(
                    "launch replay lease does not exactly cover the cache-store rank"
                )
            rank_resource = rank_resources[0]
            claim = _ScratchLeaseClaim(
                scratch_arena_id=rank_resource.scratch_arena_id,
                lease_id=launch_plan.replay_lease.lease_id,
                lease_epoch=launch_plan.replay_lease.epoch,
                launch_plan_id=launch_plan.launch_plan_id,
                resource_identity=(
                    rank_resource.graph_instance_id,
                    rank_resource.stream_lane_id,
                    rank_resource.buffer_set_id,
                    rank_resource.workspace_arena_id,
                ),
            )
            if self._scratch_hold is not None and self._scratch_hold.claim != claim:
                raise ScratchArenaLeaseError("transaction is already bound to another replay lease")
            self._validate_graph_staging()
            page_slots = tuple(
                _cow_slots(
                    page_id,
                    self._store._owned_page(self.plan.owner_id, page_id),
                    page,
                )
                for page_id, page in sorted(self._staged_pages.items())
            )
            state_slots = tuple(
                _cow_slots(
                    state_id,
                    self._store._owned_state(self.plan.owner_id, state_id),
                    state,
                )
                for state_id, state in sorted(self._staged_states.items())
            )
            write_page_ids = set(self._staged_pages)
            write_state_ids = set(self._staged_states)
            read_page_slots = tuple(
                _read_slot(page_id, self._store._owned_page(self.plan.owner_id, page_id))
                for page_id in sorted(
                    {
                        region.page_id
                        for region in self.plan.reads
                        if isinstance(region, KVRegion) and region.page_id not in write_page_ids
                    }
                )
            )
            read_state_slots = tuple(
                _read_slot(state_id, self._store._owned_state(self.plan.owner_id, state_id))
                for state_id in sorted(
                    {
                        region.state_id
                        for region in self.plan.reads
                        if isinstance(region, RecurrentStateRegion)
                        and region.state_id not in write_state_ids
                    }
                )
            )
            scratch_indices = tuple(
                row.scratch_index
                for row in launch_plan.rows
                if isinstance(row, ec.PaddingLaunchRow)
            )
            hold = arena.claim_scratch(
                claim,
                transaction_id=self.plan.transaction_id,
                scratch_indices=scratch_indices,
            )
            try:
                binding = arena.binding(
                    scratch_hold=hold,
                    read_page_slots=read_page_slots,
                    read_state_slots=read_state_slots,
                    page_slots=page_slots,
                    state_slots=state_slots,
                )
            except Exception:
                if self._scratch_hold is None:
                    arena.release_prevalidated(arena.prepare_release((), (), (hold,)))
                raise
            self._scratch_hold = hold
            return binding

    def accept_graph_writes(self) -> None:
        """Close the sealed graph write set after its external completion barrier."""

        if self._status is not _TxnStatus.OPEN:
            raise TransactionStateError("graph writes can be accepted only while open")
        with self._store._lock:
            self._ensure_current_or_abort()
            if self._scratch_hold is None:
                raise TransactionStateError("graph writes require a sealed graph binding")
            self._validate_graph_staging()
            for region in self.plan.writes:
                if isinstance(region, KVRegion):
                    self._written_offsets[_region_identity(region)] = set(region.token_offsets)
                    self._initialized_pages.add(region.page_id)
                else:
                    state = self._staged_states[region.state_id]
                    current = self._store._owned_state(self.plan.owner_id, region.state_id)
                    assert self.plan.target_position is not None
                    if self.plan.target_position < current.position:
                        raise ValueError("recurrent state position cannot move backwards")
                    state.position = self.plan.target_position
                    state.valid = True
                    self._written_states.add(region.state_id)
            self._graph_completed = True
            self._status = _TxnStatus.TENTATIVE

    def read_kv(
        self, page_id: int, layer_id: int, token_offsets: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_mutable()
        declared = self._authorized_kv_region(
            self.plan.reads, page_id, layer_id, token_offsets, operation="read"
        )
        with self._store._lock:
            self._ensure_current_or_abort()
            page = self._staged_pages.get(page_id)
            if page is None or (
                self._store._arena is not None and page_id not in self._initialized_pages
            ):
                page = self._store._owned_page(self.plan.owner_id, page_id)
            return self._store._read_page(
                page,
                KVRegion(page_id, layer_id, token_offsets, declared.revision),
            )

    def write_kv(
        self,
        page_id: int,
        layer_id: int,
        token_offsets: tuple[int, ...],
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self._require_mutable()
        declared = self._authorized_kv_region(
            self.plan.writes, page_id, layer_id, token_offsets, operation="write"
        )
        self._validate_kv_payload(token_offsets, key, value)
        identity = _region_identity(declared)
        already_written = self._written_offsets.setdefault(identity, set())
        overlap = already_written.intersection(token_offsets)
        if overlap:
            raise AccessViolation(f"KV offsets may be staged only once: {sorted(overlap)}")

        with self._store._lock:
            self._ensure_current_or_abort()
            page = self._staged_pages.get(page_id)
            if page is None:
                page = self._store._owned_page(self.plan.owner_id, page_id).clone()
                self._staged_pages[page_id] = page
            elif self._store._arena is not None:
                self._store._arena.validate_page(page)
                if page_id not in self._initialized_pages:
                    committed = self._store._owned_page(self.plan.owner_id, page_id)
                    page.key.copy_(committed.key)
                    page.value.copy_(committed.value)
                    page.written.copy_(committed.written)
                    for name, field_value in committed.page_fields.items():
                        page.page_fields[name].copy_(field_value)
                    self._initialized_pages.add(page_id)
            layer_index = self._store.layout.layer_index(layer_id)
            offsets = torch.tensor(token_offsets, dtype=torch.long, device=self._store.device)
            page.key[layer_index, offsets] = key.detach()
            page.value[layer_index, offsets] = value.detach()
            page.written[layer_index, offsets] = True
        already_written.update(token_offsets)
        self._status = _TxnStatus.TENTATIVE

    def read_recurrent_state(self, state_id: str) -> torch.Tensor:
        self._require_mutable()
        self._authorized_state_region(self.plan.reads, state_id, operation="read")
        with self._store._lock:
            self._ensure_current_or_abort()
            state = self._staged_states.get(state_id)
            if state is None or (
                self._store._arena is not None and state_id not in self._written_states
            ):
                state = self._store._owned_state(self.plan.owner_id, state_id)
            if not state.valid:
                raise CacheStateError(f"recurrent state {state_id!r} is not valid")
            return state.value.clone()

    def write_recurrent_state(
        self,
        state_id: str,
        value: torch.Tensor,
        *,
        target_position: int,
        valid: bool = True,
    ) -> None:
        self._require_mutable()
        self._authorized_state_region(self.plan.writes, state_id, operation="write")
        if state_id in self._written_states:
            raise AccessViolation("a recurrent state may be staged only once")
        self._store._validate_state_tensor(value)
        self._store._validate_arena_state_tensor(state_id, value)
        with self._store._lock:
            self._ensure_current_or_abort()
            current = self._store._owned_state(self.plan.owner_id, state_id)
            if self.plan.target_position != target_position:
                raise AccessViolation(
                    "recurrent write position differs from the transaction target"
                )
            if target_position < current.position:
                raise ValueError("recurrent state position cannot move backwards")
            if value.dtype is not current.value.dtype or tuple(value.shape) != tuple(
                current.value.shape
            ):
                raise ValueError("recurrent state dtype and shape are immutable")
            if self._store._arena is None:
                staged = current.clone()
                staged.value = value.detach().clone()
                self._staged_states[state_id] = staged
            else:
                staged = self._staged_states[state_id]
                if staged.slot is None:
                    raise StaleArenaSlot("tentative state has no physical generation")
                self._store._arena.validate_state(state_id, staged.slot)
                staged.value.copy_(value.detach())
            staged.position = target_position
            staged.valid = valid
        self._written_states.add(state_id)
        self._status = _TxnStatus.TENTATIVE

    def verify(self) -> None:
        if self._status is not _TxnStatus.TENTATIVE:
            raise TransactionStateError("only tentative state can be verified")
        for region in self.plan.writes:
            if isinstance(region, KVRegion):
                actual = self._written_offsets.get(_region_identity(region), set())
                if actual != set(region.token_offsets):
                    raise AccessViolation(
                        f"KV write set was not fulfilled for page={region.page_id}, "
                        f"layer={region.layer_id}"
                    )
            elif region.state_id not in self._written_states:
                raise AccessViolation(
                    f"recurrent state write set was not fulfilled for {region.state_id!r}"
                )
        with self._store._lock:
            self._ensure_current_or_abort()
            self._validate_staged_payload()
        self._status = _TxnStatus.VERIFIED

    def verify_kv_matches_committed(self) -> None:
        """Prove replayed KV is bitwise identical before replacing cached rows."""

        if self._status is not _TxnStatus.TENTATIVE:
            raise TransactionStateError("KV replay comparison requires tentative state")
        with self._store._lock:
            self._ensure_current_or_abort()
            for region in self.plan.writes:
                if not isinstance(region, KVRegion):
                    continue
                staged = self._staged_pages.get(region.page_id)
                if staged is None:
                    raise AccessViolation("KV replay region has no staged page")
                committed = self._store._owned_page(self.plan.owner_id, region.page_id)
                layer_index = self._store.layout.layer_index(region.layer_id)
                offsets = torch.tensor(
                    region.token_offsets,
                    dtype=torch.long,
                    device=self._store.device,
                )
                if not bool(torch.all(committed.written[layer_index, offsets]).item()):
                    raise UnwrittenKVError("prefix replay requires committed KV at every row")
                if not torch.equal(
                    staged.key[layer_index, offsets], committed.key[layer_index, offsets]
                ) or not torch.equal(
                    staged.value[layer_index, offsets], committed.value[layer_index, offsets]
                ):
                    raise CacheStateError(
                        "prefix replay recomputed KV differs from the cached BF16 state"
                    )

            for page_id, staged in self._staged_pages.items():
                committed = self._store._owned_page(self.plan.owner_id, page_id)
                if any(
                    not torch.equal(staged.page_fields[name], value)
                    for name, value in committed.page_fields.items()
                ):
                    raise CacheStateError(
                        "prefix replay recomputed page fields differ from cached state"
                    )

    def commit(self) -> int:
        return self._store.commit_bundle((self,))[0]

    def abort(self) -> None:
        if self._status in (_TxnStatus.COMMITTED, _TxnStatus.ABORTED):
            raise TransactionStateError("a terminal transaction cannot be aborted")
        with self._store._lock:
            self._abort_locked()

    def __enter__(self) -> StateTransaction:
        self._require_mutable()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._status not in (_TxnStatus.COMMITTED, _TxnStatus.ABORTED):
            self.abort()

    def _authorized_kv_region(
        self,
        regions: tuple[ResourceRegion, ...],
        page_id: int,
        layer_id: int,
        token_offsets: tuple[int, ...],
        *,
        operation: str,
    ) -> KVRegion:
        requested = set(token_offsets)
        if not requested or len(requested) != len(token_offsets):
            raise AccessViolation("KV operation offsets must be non-empty and unique")
        for region in regions:
            if (
                isinstance(region, KVRegion)
                and region.page_id == page_id
                and region.layer_id == layer_id
                and requested <= set(region.token_offsets)
            ):
                return region
        raise AccessViolation(
            f"KV {operation} is outside the transaction's declared resource region"
        )

    @staticmethod
    def _authorized_state_region(
        regions: tuple[ResourceRegion, ...], state_id: str, *, operation: str
    ) -> RecurrentStateRegion:
        for region in regions:
            if isinstance(region, RecurrentStateRegion) and region.state_id == state_id:
                return region
        raise AccessViolation(
            f"recurrent state {operation} is outside the transaction's declared set"
        )

    def _validate_kv_payload(
        self,
        token_offsets: tuple[int, ...],
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        expected = self._store.layout.payload_shape(len(token_offsets))
        if tuple(key.shape) != expected or tuple(value.shape) != expected:
            raise ValueError(f"KV payload shape must be {expected}")
        if self._store.layout.storage.quantized:
            raise ValueError("encoded KV writes require explicit group storage authority")
        if (key.dtype != self._store.layout.storage_dtype(key=True)
                or value.dtype != self._store.layout.storage_dtype(key=False)):
            raise ValueError("attention KV writes differ from their declared storage dtype")
        if key.device != self._store.device or value.device != self._store.device:
            raise ValueError("KV payload device must match the cache store")
        if key.layout is not torch.strided or value.layout is not torch.strided:
            raise TypeError("KV payloads must be strided tensors")
        if not bool(torch.isfinite(key).all().item()) or not bool(
            torch.isfinite(value).all().item()
        ):
            raise ValueError("attention KV writes must contain only finite values")

    def _validate_staged_payload(self) -> None:
        declared_page_ids = {
            region.page_id for region in self.plan.writes if isinstance(region, KVRegion)
        }
        declared_state_ids = {
            region.state_id
            for region in self.plan.writes
            if isinstance(region, RecurrentStateRegion)
        }
        if set(self._staged_pages) != declared_page_ids:
            raise AccessViolation("staged KV pages differ from the declared write set")
        if set(self._staged_states) != declared_state_ids:
            raise AccessViolation("staged recurrent states differ from the declared write set")
        if self._graph_completed:
            self._validate_graph_staging()
            return
        for page_id, page in self._staged_pages.items():
            if page.page_id != page_id:
                raise AccessViolation("staged KV page identity differs from its mapping key")
            self._store._validate_page_payload(
                page,
                expected_owner_id=self.plan.owner_id,
            )
        for state_id, state in self._staged_states.items():
            if state.state_id != state_id or state.owner_id != self.plan.owner_id:
                raise AccessViolation(
                    "staged recurrent state identity differs from its mapping key"
                )
            if state.owner_rank != self._store.rank_id:
                raise OwnershipError("staged recurrent state rank differs from the cache store")
            if state.position != self.plan.target_position:
                raise AccessViolation(
                    "staged recurrent state position differs from the transaction target"
                )
            self._store._validate_state_tensor(state.value)
            self._store._validate_arena_state_tensor(state_id, state.value)
            if self._store._arena is not None:
                self._store._arena.validate_state(state_id, _physical_slot(state))

    def _validate_graph_staging(self) -> None:
        arena = self._store._arena
        if arena is None:
            raise TransactionStateError("graph staging requires fixed cache arena mode")
        for page_id, page in self._staged_pages.items():
            current_page = self._store._owned_page(self.plan.owner_id, page_id)
            if (page.page_id, page.owner_rank, page.owner_id, page.revision) != (
                page_id,
                self._store.rank_id,
                self.plan.owner_id,
                current_page.revision,
            ):
                raise AccessViolation("tentative graph page metadata changed")
            arena.validate_page(page)
        for state_id, state in self._staged_states.items():
            current_state = self._store._owned_state(self.plan.owner_id, state_id)
            if (state.state_id, state.owner_rank, state.owner_id, state.revision) != (
                state_id,
                self._store.rank_id,
                self.plan.owner_id,
                current_state.revision,
            ):
                raise AccessViolation("tentative graph state metadata changed")
            arena.validate_state(state_id, _physical_slot(state))

    def _reserve_arena_staging(self) -> None:
        arena = self._store._arena
        if arena is None:
            return
        page_ids = sorted(
            {region.page_id for region in self.plan.writes if isinstance(region, KVRegion)}
        )
        state_ids = sorted(
            {
                region.state_id
                for region in self.plan.writes
                if isinstance(region, RecurrentStateRegion)
            }
        )
        for page_id in page_ids:
            current_page = self._store._owned_page(self.plan.owner_id, page_id)
            self._staged_pages[page_id] = arena.reserve_page(
                page_id=page_id,
                owner_rank=self._store.rank_id,
                owner_id=self.plan.owner_id,
                revision=current_page.revision,
            )
        for state_id in state_ids:
            current_state = self._store._owned_state(self.plan.owner_id, state_id)
            slot, value = arena.reserve_state(state_id)
            self._staged_states[state_id] = _RecurrentState(
                state_id,
                self._store.rank_id,
                self.plan.owner_id,
                current_state.revision,
                current_state.position,
                value,
                current_state.valid,
                slot,
            )

    def _release_arena_staging(self) -> None:
        arena = self._store._arena
        if arena is None:
            return
        release_batch = self._store._prepare_arena_release(
            tuple(self._staged_pages.values()),
            tuple(self._staged_states.values()),
            () if self._scratch_hold is None else (self._scratch_hold,),
        )
        self._store._release_prevalidated(release_batch)
        self._scratch_hold = None

    def _ensure_current_or_abort(self) -> None:
        try:
            self._store._validate_plan_current(self.plan)
        except (RevisionConflict, OwnershipError):
            self._abort_locked()
            raise

    def _require_mutable(self) -> None:
        if self._status not in (_TxnStatus.OPEN, _TxnStatus.TENTATIVE):
            raise TransactionStateError(
                f"transaction is {self._status.value} and can no longer be accessed"
            )

    def _abort_locked(self) -> None:
        try:
            self._release_arena_staging()
        finally:
            if self.plan.transaction_id in self._store._active_transactions:
                self._store._deactivate_transaction(self.plan.transaction_id)
            self._status = _TxnStatus.ABORTED
            self._scratch_hold = None
            self._discard_staging()

    def _discard_staging(self) -> None:
        self._staged_pages.clear()
        self._staged_states.clear()
        self._written_offsets.clear()
        self._written_states.clear()
        self._initialized_pages.clear()
