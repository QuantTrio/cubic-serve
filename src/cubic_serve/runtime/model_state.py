"""Qwen3.5 decode-state schema and model-specific state transactions."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from cubic_serve.contracts import execution as ec
from cubic_serve.contracts.request import TokenPosition
from cubic_serve.models.qwen3_5.model import (
    CONV_STATE_ID,
    RECURRENT_STATE_ID,
    Qwen3_5Config,
)
from cubic_serve.models.qwen3_5.state import dense_decode_state_layout
from cubic_serve.runtime import cache as rc

STATE_IDS = (CONV_STATE_ID, RECURRENT_STATE_ID)
STATE_PATH_ID = "cubic.qwen3_5.decode.physical-state.v1"
SEED_FIELDS = (*STATE_IDS, "kv_key", "kv_value", "kv_written", "position")


@dataclass(frozen=True, slots=True)
class DecodeStateSchema:
    kv_layout: rc.KVLayout
    arena_spec: rc.CacheArenaSpec
    gdn_layer_ids: tuple[int, ...]
    max_pages: int

    @property
    def digest(self) -> str:
        return f"b2s256:{hashlib.blake2s(repr((STATE_PATH_ID, self)).encode()).hexdigest()}"


def build_decode_state_schema(
    config: Qwen3_5Config,
    *,
    arena_id: str,
    page_capacity: int,
    state_capacity: int,
    scratch_capacity: int,
    scratch_arena_ids: tuple[str, ...],
    max_pages: int,
    block_size: int = 16,
) -> DecodeStateSchema:
    layout, states, linear = dense_decode_state_layout(
        config, max_pages=max_pages, block_size=block_size,
    )
    arena = rc.CacheArenaSpec(
        arena_id, page_capacity, state_capacity, states, scratch_capacity, scratch_arena_ids
    )
    return DecodeStateSchema(layout, arena, linear, max_pages)


@dataclass(frozen=True, slots=True)
class DecodeRow:
    logical_index: int | None
    position: TokenPosition | None
    page_cow: tuple[int, int]
    page_table: tuple[int, ...]
    sequence_length: int
    token_offset: int
    state_cow: tuple[tuple[int, int], ...]


@dataclass(slots=True)
class DecodeSession:
    store: rc.CacheStateStore
    transactions: tuple[rc.StateTransaction, ...]
    bindings: tuple[rc.CacheArenaBinding, ...]
    rows: tuple[DecodeRow, ...]

    @property
    def status(self) -> str:
        statuses = {transaction.status for transaction in self.transactions}
        return statuses.pop() if len(statuses) == 1 else "invalid"

    def accept_and_verify(self) -> None:
        if self.status != "open":
            raise rc.TransactionStateError("graph writes may be accepted exactly once")
        try:
            for transaction in self.transactions:
                transaction.accept_graph_writes()
                transaction.verify()
        except BaseException:
            _abort(self.transactions)
            raise

    def commit(self) -> tuple[int, ...]:
        if self.status != "verified":
            raise rc.TransactionStateError("consume durable graph outputs before state commit")
        return self.store.commit_bundle(self.transactions)

    def abort(self) -> None:
        if self.status in ("committed", "aborted"):
            raise rc.TransactionStateError("decode session is already terminal")
        _abort(self.transactions)


class DecodeState:
    def __init__(self, schema: DecodeStateSchema, device: torch.device | str) -> None:
        self.schema, self.device = schema, torch.device(device)
        self.arena = rc.CacheArena(schema.arena_spec, schema.kv_layout, self.device, rank_id=0)
        self.store = rc.CacheStateStore(
            schema.kv_layout,
            rank_id=0,
            device=self.device,
            arena=self.arena,
        )
        self._owners: dict[str, rc.QualificationOwnerLease] = {}

    def create_owner(
        self,
        owner: str,
        *,
        position: int,
        page_ids: tuple[int, ...],
        seed: Mapping[str, Tensor] | None = None,
    ) -> None:
        if not owner or position < 0 or len(page_ids) > self.schema.max_pages:
            raise ValueError("owner identity, position, or exact pages are invalid")
        values = self._seed(position, page_ids, seed)
        states = tuple(
            rc.RecurrentStateProvision(state, values[state], position) for state in STATE_IDS
        )
        image = rc.QualificationOwnerImage(
            owner,
            page_ids,
            values["kv_key"],
            values["kv_value"],
            values["kv_written"],
            states,
        )
        self._owners[owner] = self.store.provision_qualification_owner(image)

    def retire_owner(self, owner: str, *, expected_revision: int) -> None:
        lease = self._owners.get(owner)
        if lease is None:
            raise ValueError("decode owner is stale or foreign")
        self.store.retire_qualification_owner(lease, expected_revision=expected_revision)
        del self._owners[owner]

    def begin(self, batch: ec.BatchPlan, launch: ec.LaunchPlan) -> DecodeSession:
        self._validate_batch(batch, launch)
        transactions: list[rc.StateTransaction] = []
        bindings: list[rc.CacheArenaBinding] = []
        try:
            for segment, contract in zip(batch.segments, batch.state_transactions, strict=True):
                transaction = self.store.begin(self._plan(segment, contract))
                transactions.append(transaction)
                bindings.append(transaction.graph_binding(launch))
            rows = self._rows(launch, batch.segments, tuple(bindings))
        except BaseException:
            _abort(tuple(transactions))
            raise
        return DecodeSession(self.store, tuple(transactions), tuple(bindings), rows)

    def snapshot(self, owner: str) -> dict[str, Tensor]:
        lease = self._owners.get(owner)
        if lease is None:
            raise ValueError("decode owner is stale or foreign")
        image = self.store.snapshot_qualification_owner(lease)
        return {state.state_id: state.value for state in image.recurrent_states} | {
            "kv_key": image.key,
            "kv_value": image.value,
            "kv_written": image.written,
            "position": torch.tensor(
                image.recurrent_states[0].position, dtype=torch.int64, device=self.device
            ),
        }

    def _empty(self, position: int, pages: tuple[int, ...]) -> dict[str, Tensor]:
        layout, states = self.schema.kv_layout, self.arena.state_specs
        kv_shape = (len(pages), *layout.page_shape)
        return {
            state_id: torch.zeros(spec.shape, dtype=spec.dtype, device=self.device)
            for state_id, spec in states.items()
        } | {
            "kv_key": torch.zeros(kv_shape, dtype=torch.bfloat16, device=self.device),
            "kv_value": torch.zeros(kv_shape, dtype=torch.bfloat16, device=self.device),
            "kv_written": torch.zeros(kv_shape[:-2], dtype=torch.bool, device=self.device),
            "position": torch.tensor(position, dtype=torch.int64, device=self.device),
        }

    def _seed(
        self, position: int, pages: tuple[int, ...], source: Mapping[str, Tensor] | None
    ) -> dict[str, Tensor]:
        if source is None:
            return self._empty(position, pages)
        if set(source) != set(SEED_FIELDS):
            raise ValueError("decode seed keys differ from the durable schema")
        values = {name: tensor.detach().clone() for name, tensor in source.items()}
        marker = values["position"]
        if (
            marker.device != self.device
            or marker.dtype is not torch.int64
            or marker.shape
            or int(marker.item()) != position
        ):
            raise ValueError("decode seed position differs from its fixed ABI")
        return values

    def _validate_batch(self, batch: ec.BatchPlan, launch: ec.LaunchPlan) -> None:
        identity = (batch.plan_id, batch.topology_id, batch.partition_id, batch.shape.logical_m)
        expected = (
            launch.source_plan_id,
            launch.bucket.topology_id,
            launch.bucket.partition_id,
            launch.logical_m,
        )
        if (
            identity != expected
            or batch.phase is not ec.BatchPhase.DECODE
            or launch.bucket.phase is not ec.BatchPhase.DECODE
            or batch.decode_extension.provider is not ec.DecodeProvider.NONE
        ):
            raise ValueError("batch is outside the single-rank qwen3_5 decode ABI")
        for segment in batch.segments:
            lease = self._owners.get(segment.request_id)
            if segment.token_count != 1 or lease is None or lease.page_ids != segment.cache_pages:
                raise ValueError("each decode row requires one token and its exact owner pages")

    def _plan(self, segment: ec.PlanSegment, contract: ec.StateTxn) -> rc.TxnPlan:
        owner, block = segment.request_id, self.schema.kv_layout.block_size_tokens
        target = segment.token_start + 1
        page, offset = divmod(segment.cache_slots[0], block)
        if (
            len(segment.cache_pages) != (target + block - 1) // block
            or page != segment.cache_pages[-1]
            or offset != segment.token_start % block
            or any(
                self.store.state_position(owner, state) != segment.token_start
                for state in STATE_IDS
            )
        ):
            raise ValueError("decode physical state differs from the immutable segment")
        self._validate_contract(segment, contract, page)
        reads = tuple(
            self.store.kv_region(owner, physical, layer, offsets)
            for logical, physical in enumerate(segment.cache_pages)
            for offsets in (
                tuple(range(min(block, max(0, segment.token_start - logical * block)))),
            )
            if offsets
            for layer in self.schema.kv_layout.layer_ids
        ) + tuple(self.store.recurrent_region(owner, state) for state in STATE_IDS)
        writes = tuple(
            self.store.kv_region(owner, page, layer, (offset,))
            for layer in self.schema.kv_layout.layer_ids
        ) + tuple(self.store.recurrent_region(owner, state) for state in STATE_IDS)
        return rc.TxnPlan(
            contract.transaction_id, owner, contract.base_revision, reads, writes, target
        )

    def _validate_contract(self, segment: ec.PlanSegment, contract: ec.StateTxn, page: int) -> None:
        owner, kv = segment.request_id, f"kv:{segment.request_id}"
        expected = {kv, *(f"state:{state}:{owner}" for state in STATE_IDS)}
        reads = {access.resource_id: access for access in contract.reads}
        writes = {access.resource_id: access for access in contract.writes}
        accesses = (*contract.reads, *contract.writes)
        valid = (
            contract.request_id == owner
            and contract.expected_owner_ranks == (0,)
            and contract.base_revision == self.store.current_revision(owner)
            and contract.prefix_publish is None
            and set(reads) == expected == set(writes)
            and reads[kv].pages == tuple(sorted(segment.cache_pages))
            and (writes[kv].pages, writes[kv].slots)
            == ((page,), tuple(sorted(segment.cache_slots)))
            and all(
                access.owner_rank == 0
                and (access.resource_id == kv or not (access.pages or access.slots))
                for access in accesses
            )
        )
        if not valid:
            raise ValueError("decode transaction differs from the canonical physical access")

    def _rows(
        self,
        launch: ec.LaunchPlan,
        segments: tuple[ec.PlanSegment, ...],
        bindings: tuple[rc.CacheArenaBinding, ...],
    ) -> tuple[DecodeRow, ...]:
        first, rows = bindings[0], []
        scratch_states = dict(first.scratch_state_slots)
        for row in launch.rows:
            if isinstance(row, ec.PaddingLaunchRow):
                page = first.scratch_page_slots[row.scratch_index]
                state = tuple(
                    (scratch_states[state_id][row.scratch_index],) * 2 for state_id in STATE_IDS
                )
                rows.append(
                    DecodeRow(
                        None, None, (page, page), (page,) * self.schema.max_pages, 1, 0, state
                    )
                )
                continue
            segment, binding = segments[row.logical_index], bindings[row.logical_index]
            cow = _cow(binding.page_slots, segment.cache_pages[-1])
            sources = {slot.resource_id: slot.source_slot for slot in binding.read_page_slots}
            sources[cow.resource_id] = cow.destination_slot
            table = tuple(sources[item] for item in segment.cache_pages)
            table += (binding.scratch_page_slots[0],) * (self.schema.max_pages - len(table))
            state = tuple(_pair(_cow(binding.state_slots, state_id)) for state_id in STATE_IDS)
            rows.append(
                DecodeRow(
                    row.logical_index,
                    segment.positions[0],
                    _pair(cow),
                    table,
                    segment.token_start + 1,
                    segment.cache_slots[0] % self.schema.kv_layout.block_size_tokens,
                    state,
                )
            )
        return tuple(rows)


def _cow(slots: tuple[rc.ArenaCOWSlots, ...], resource: int | str) -> rc.ArenaCOWSlots:
    found = tuple(slot for slot in slots if slot.resource_id == resource)
    if len(found) != 1:
        raise RuntimeError(f"resource {resource!r} does not have exactly one COW mapping")
    return found[0]


def _pair(slot: rc.ArenaCOWSlots) -> tuple[int, int]:
    return (slot.source_slot, slot.destination_slot)


def _abort(transactions: tuple[rc.StateTransaction, ...]) -> None:
    for transaction in transactions:
        if transaction.status not in ("committed", "aborted"):
            transaction.abort()


def _row_key(row: int, field: str) -> str:
    return f"row.{row:04d}.{field}"


class QualificationRun:
    def __init__(self, state: DecodeState, batch: ec.BatchPlan, owners: tuple[str, ...]) -> None:
        self.state, self.batch, self.owners = state, batch, owners
        self.production_state: object = self
        self.session: DecodeSession | None = None

    def begin(self, batch: ec.BatchPlan, launch: ec.LaunchPlan) -> DecodeSession:
        if not self.owners or self.session is not None or batch != self.batch:
            raise RuntimeError("qualification run is closed, reused, or bound to another batch")
        self.session = self.state.begin(batch, launch)
        return self.session

    def committed_snapshot(self) -> Mapping[str, Tensor]:
        if not self.owners or self.session is None or self.session.status != "committed":
            raise RuntimeError("qualification snapshot requires committed graph state")
        return {
            _row_key(row, field): value
            for row, owner in enumerate(self.owners)
            for field, value in self.state.snapshot(owner).items()
        }

    def close(self) -> None:
        self._cleanup(abort=False)

    def abort(self) -> None:
        self._cleanup(abort=True)

    def _cleanup(self, *, abort: bool) -> None:
        if not self.owners:
            raise RuntimeError("qualification run was already cleaned up")
        if self.session is not None and self.session.status not in ("committed", "aborted"):
            if not abort:
                raise RuntimeError("cannot close qualification with tentative state")
            self.session.abort()
        for owner in self.owners:
            revision = self.state.store.current_revision(owner)
            self.state.retire_owner(owner, expected_revision=revision)
        self.owners = ()


class QualificationFactory:
    state_path_id = STATE_PATH_ID

    def __init__(self, state: DecodeState, batch: ec.BatchPlan) -> None:
        self.state, self.batch = state, batch
        self.state_path_digest = state.schema.digest

    def create(self, seed: Mapping[str, Tensor]) -> QualificationRun:
        expected = {
            _row_key(row, field) for row in range(len(self.batch.segments)) for field in SEED_FIELDS
        }
        if set(seed) != expected:
            raise ValueError("qualification seed lacks a stable logical-row state schema")
        owners: list[str] = []
        try:
            for row, segment in enumerate(self.batch.segments):
                local = {field: seed[_row_key(row, field)] for field in SEED_FIELDS}
                self.state.create_owner(
                    segment.request_id,
                    position=segment.token_start,
                    page_ids=segment.cache_pages,
                    seed=local,
                )
                owners.append(segment.request_id)
        except BaseException:
            for owner in owners:
                self.state.retire_owner(
                    owner, expected_revision=self.state.store.current_revision(owner)
                )
            raise
        return QualificationRun(self.state, self.batch, tuple(owners))
