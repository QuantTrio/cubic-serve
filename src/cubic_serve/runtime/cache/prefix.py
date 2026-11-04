"""Immutable prefix-state bundles and a lease-aware in-process cache."""

from __future__ import annotations

from typing_extensions import TypeAliasType

import hashlib
import json
from contextlib import ExitStack
from dataclasses import dataclass, replace
from threading import RLock
from typing import Any, Protocol

import torch

from cubic_serve.runtime.cache.pages import (
    ArenaSlotRef,
    CacheArenaDomain,
    CacheStateError,
    KVLayout,
    KVPageSnapshot,
)


class PhysicalPrefixConflict(CacheStateError):
    """One semantic prefix identity resolved to conflicting physical content."""


class _Digest(Protocol):
    def update(self, data: bytes, /) -> None: ...


def _hash_tensor(digest: _Digest, tensor: torch.Tensor) -> None:
    detached = tensor.detach().contiguous().cpu()
    header = {
        "dtype": str(detached.dtype),
        "shape": list(detached.shape),
    }
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    digest.update(detached.view(torch.uint8).numpy().tobytes())


@dataclass(frozen=True, slots=True)
class RecurrentStateSnapshot:
    state_id: str
    owner_rank: int
    owner_id: str
    revision: int
    position: int
    value: torch.Tensor
    valid: bool
    content_digest: str

    def validate(self) -> None:
        if not self.state_id or not self.owner_id:
            raise ValueError("recurrent checkpoint identity is required")
        if min(self.owner_rank, self.revision, self.position) < 0:
            raise ValueError(
                "recurrent checkpoint rank, revision, and position must be non-negative"
            )
        if (
            not isinstance(self.value, torch.Tensor)
            or self.value.layout is not torch.strided
            or self.value.numel() == 0
        ):
            raise ValueError("recurrent checkpoint must contain a non-empty strided tensor")
        if self.value.device.type == "meta":
            raise ValueError("recurrent checkpoint must have realized tensor storage")
        if (self.value.is_floating_point() or self.value.is_complex()) and not bool(
            torch.isfinite(self.value).all().item()
        ):
            raise ValueError("recurrent checkpoint must contain only finite values")
        if self.content_digest != self.compute_digest():
            raise ValueError("recurrent checkpoint content digest mismatch")

    def compute_digest(self) -> str:
        digest = hashlib.blake2s()
        metadata = {
            "state_id": self.state_id,
            "owner_rank": self.owner_rank,
            "owner_id": self.owner_id,
            "revision": self.revision,
            "position": self.position,
            "valid": self.valid,
        }
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
        _hash_tensor(digest, self.value)
        return f"b2s256:{digest.hexdigest()}"

    def clone(self) -> RecurrentStateSnapshot:
        clone = replace(self, value=self.value.clone())
        clone.validate()
        return clone


def seal_recurrent_state_snapshot(
    *,
    state_id: str,
    owner_rank: int,
    owner_id: str,
    revision: int,
    position: int,
    value: torch.Tensor,
    valid: bool,
) -> RecurrentStateSnapshot:
    snapshot = RecurrentStateSnapshot(
        state_id=state_id,
        owner_rank=owner_rank,
        owner_id=owner_id,
        revision=revision,
        position=position,
        value=value.detach().clone(),
        valid=valid,
        content_digest="",
    )
    snapshot = replace(snapshot, content_digest=snapshot.compute_digest())
    snapshot.validate()
    return snapshot


@dataclass(frozen=True, slots=True)
class PrefixSnapshot:
    """All model state needed to resume one physically aligned prefix.

    ``physical_block_boundary`` is where reusable KV ends.
    ``arithmetic_state_commit_boundary`` is the earlier (or equal) position at
    which recurrent arithmetic state is valid.  The exact intervening token ids
    are carried by ``replay_tokens``; callers must replay them before decoding.
    """

    snapshot_id: str
    prefix_key: str
    source_rank: int
    source_owner_id: str
    source_revision: int
    kv_layout: KVLayout
    physical_block_boundary: int
    arithmetic_state_commit_boundary: int
    state_commit_alignment: int
    replay_tokens: tuple[int, ...]
    pages: tuple[KVPageSnapshot, ...]
    recurrent_states: tuple[RecurrentStateSnapshot, ...]
    state_valid_mask: tuple[bool, ...]
    content_digest: str

    def validate(self) -> None:
        if not self.snapshot_id or not self.prefix_key or not self.source_owner_id:
            raise ValueError("prefix snapshot identity is required")
        if min(self.source_rank, self.source_revision) < 0:
            raise ValueError("prefix source rank and revision must be non-negative")
        if self.physical_block_boundary <= 0:
            raise ValueError("prefix must contain at least one physical KV block")
        if self.physical_block_boundary % self.kv_layout.block_size_tokens:
            raise ValueError("physical prefix boundary must be KV-block aligned")
        if self.state_commit_alignment <= 0:
            raise ValueError("state commit alignment must be positive")
        if not 0 <= self.arithmetic_state_commit_boundary <= self.physical_block_boundary:
            raise ValueError("arithmetic state boundary must lie within the cached prefix")
        if self.arithmetic_state_commit_boundary % self.state_commit_alignment:
            raise ValueError("arithmetic state boundary violates its commit alignment")
        if self.arithmetic_state_commit_boundary % self.kv_layout.storage.column_group_alignment:
            raise ValueError("prefix replay must start at a closed column cache-group boundary")
        replay_count = self.physical_block_boundary - self.arithmetic_state_commit_boundary
        if len(self.replay_tokens) != replay_count:
            raise ValueError("replay tokens must exactly bridge state and physical boundaries")
        if any(token_id < 0 for token_id in self.replay_tokens):
            raise ValueError("replay token ids must be non-negative")

        expected_pages = self.physical_block_boundary // self.kv_layout.block_size_tokens
        if len(self.pages) != expected_pages:
            raise ValueError("prefix page count does not match its physical boundary")
        source_page_ids = tuple(page.page_id for page in self.pages)
        if len(set(source_page_ids)) != len(source_page_ids):
            raise ValueError("prefix page ids must be unique")
        for page in self.pages:
            if page.owner_rank != self.source_rank:
                raise ValueError("prefix page rank differs from snapshot source rank")
            if page.owner_id != self.source_owner_id:
                raise ValueError("prefix page owner differs from snapshot owner")
            if page.revision < 0:
                raise ValueError("prefix page revision must be non-negative")
            self.kv_layout.validate_page_storage(
                page.key, page.value, page.written, page.page_fields,
                device=page.key.device, require_full=True,
            )
        state_ids = tuple(state.state_id for state in self.recurrent_states)
        if len(set(state_ids)) != len(state_ids) or any(not state_id for state_id in state_ids):
            raise ValueError("prefix recurrent state ids must be non-empty and unique")
        if len(self.state_valid_mask) != len(self.recurrent_states):
            raise ValueError("state_valid_mask must have one entry per recurrent state")
        if self.state_valid_mask != tuple(state.valid for state in self.recurrent_states):
            raise ValueError("state_valid_mask must match the recurrent state images")
        if (
            not self.recurrent_states
            and self.arithmetic_state_commit_boundary != self.physical_block_boundary
        ):
            raise ValueError("a KV-only prefix cannot require arithmetic state replay")
        for state in self.recurrent_states:
            state.validate()
            if state.owner_rank != self.source_rank:
                raise ValueError("prefix recurrent state rank differs from snapshot source rank")
            if state.owner_id != self.source_owner_id:
                raise ValueError("prefix recurrent state owner differs from snapshot owner")
            if state.valid:
                if state.position != self.arithmetic_state_commit_boundary:
                    raise ValueError(
                        "valid recurrent state position must equal the arithmetic boundary"
                    )
            elif (
                self.arithmetic_state_commit_boundary != 0
                or state.position != 0
                or len(self.replay_tokens) != self.physical_block_boundary
            ):
                raise ValueError(
                    "invalid recurrent state requires replay of the complete prefix from zero"
                )

        if self.content_digest != self.compute_digest():
            raise ValueError("prefix snapshot content digest mismatch")

    def compute_digest(self) -> str:
        digest = hashlib.blake2s()
        metadata = {
            "snapshot_id": self.snapshot_id,
            "prefix_key": self.prefix_key,
            "source_rank": self.source_rank,
            "source_owner_id": self.source_owner_id,
            "source_revision": self.source_revision,
            "layer_ids": self.kv_layout.layer_ids,
            "block_size_tokens": self.kv_layout.block_size_tokens,
            "num_kv_heads": self.kv_layout.num_kv_heads,
            "head_dim": self.kv_layout.head_dim,
            "kv_dtype": str(self.kv_layout.dtype),
            "physical_block_boundary": self.physical_block_boundary,
            "arithmetic_state_commit_boundary": self.arithmetic_state_commit_boundary,
            "state_commit_alignment": self.state_commit_alignment,
            "replay_tokens": self.replay_tokens,
            "state_valid_mask": self.state_valid_mask,
        }
        if self.kv_layout.page_fields:
            metadata["page_fields"] = [
                (spec.field_id, str(spec.dtype), spec.shape) for spec in self.kv_layout.page_fields
            ]
        if self.kv_layout.storage.quantized or self.kv_layout.storage.enable_rotation:
            metadata["cache_storage"] = self.kv_layout.storage.identity
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
        for page in self.pages:
            digest.update(f"page:{page.owner_rank}:{page.page_id}:{page.revision}".encode())
            _hash_tensor(digest, page.key)
            _hash_tensor(digest, page.value)
            _hash_tensor(digest, page.written)
            for name, value in sorted(page.page_fields.items()):
                digest.update(name.encode())
                _hash_tensor(digest, value)
        for state in self.recurrent_states:
            digest.update(f"state:{state.content_digest}:{state.position}".encode())
            _hash_tensor(digest, state.value)
        return f"b2s256:{digest.hexdigest()}"

    def compute_semantic_digest(self) -> str:
        """Hash resumable state while excluding allocator/provenance identities."""

        digest = hashlib.blake2s()
        metadata = {
            "prefix_key": self.prefix_key,
            "layer_ids": self.kv_layout.layer_ids,
            "block_size_tokens": self.kv_layout.block_size_tokens,
            "num_kv_heads": self.kv_layout.num_kv_heads,
            "head_dim": self.kv_layout.head_dim,
            "kv_dtype": str(self.kv_layout.dtype),
            "physical_block_boundary": self.physical_block_boundary,
            "arithmetic_state_commit_boundary": self.arithmetic_state_commit_boundary,
            "state_commit_alignment": self.state_commit_alignment,
            "replay_tokens": self.replay_tokens,
            "state_valid_mask": self.state_valid_mask,
        }
        if self.kv_layout.page_fields:
            metadata["page_fields"] = [
                (spec.field_id, str(spec.dtype), spec.shape) for spec in self.kv_layout.page_fields
            ]
        if self.kv_layout.storage.quantized or self.kv_layout.storage.enable_rotation:
            metadata["cache_storage"] = self.kv_layout.storage.identity
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
        for logical_page, page in enumerate(self.pages):
            digest.update(f"logical-page:{logical_page}".encode())
            _hash_tensor(digest, page.key)
            _hash_tensor(digest, page.value)
            _hash_tensor(digest, page.written)
            for name, value in sorted(page.page_fields.items()):
                digest.update(name.encode())
                _hash_tensor(digest, value)
        for state in self.recurrent_states:
            digest.update(
                f"state:{state.state_id}:position:{state.position}:valid:{state.valid}".encode()
            )
            _hash_tensor(digest, state.value)
        return f"b2s256:{digest.hexdigest()}"

    def clone(self) -> PrefixSnapshot:
        pages = tuple(page.clone() for page in self.pages)
        states = tuple(state.clone() for state in self.recurrent_states)
        clone = replace(self, pages=pages, recurrent_states=states)
        clone.validate()
        return clone


def seal_prefix_snapshot(
    *,
    snapshot_id: str,
    prefix_key: str,
    source_rank: int,
    source_owner_id: str,
    source_revision: int,
    kv_layout: KVLayout,
    physical_block_boundary: int,
    arithmetic_state_commit_boundary: int,
    state_commit_alignment: int,
    replay_tokens: tuple[int, ...],
    pages: tuple[KVPageSnapshot, ...],
    recurrent_states: tuple[RecurrentStateSnapshot, ...],
) -> PrefixSnapshot:
    """Build and integrity-seal a prefix snapshot from detached images."""

    snapshot = PrefixSnapshot(
        snapshot_id=snapshot_id,
        prefix_key=prefix_key,
        source_rank=source_rank,
        source_owner_id=source_owner_id,
        source_revision=source_revision,
        kv_layout=kv_layout,
        physical_block_boundary=physical_block_boundary,
        arithmetic_state_commit_boundary=arithmetic_state_commit_boundary,
        state_commit_alignment=state_commit_alignment,
        replay_tokens=replay_tokens,
        pages=tuple(page.clone() for page in pages),
        recurrent_states=tuple(state.clone() for state in recurrent_states),
        state_valid_mask=tuple(state.valid for state in recurrent_states),
        content_digest="",
    )
    snapshot = replace(snapshot, content_digest=snapshot.compute_digest())
    snapshot.validate()
    return snapshot


_PrefixCacheKey = TypeAliasType("_PrefixCacheKey", tuple[str, int])


def _require_content_digest(value: str, *, label: str) -> None:
    if type(value) is not str or not value.startswith("b2s256:"):
        raise ValueError(f"{label} must be a content_digest digest")
    hexadecimal = value[7:]
    if len(hexadecimal) != 64 or any(
        character not in "0123456789abcdef" for character in hexadecimal
    ):
        raise ValueError(f"{label} must be a lowercase content_digest digest")


class PrefixLease:
    """A lease prevents eviction; exposed tensors are always detached copies."""

    def __init__(
        self, cache: PrefixCache, cache_key: _PrefixCacheKey, snapshot: PrefixSnapshot
    ) -> None:
        self._cache = cache
        self.prefix_key, self.source_rank = cache_key
        self._cache_key = cache_key
        self._snapshot = snapshot
        self._released = False

    @property
    def snapshot(self) -> PrefixSnapshot:
        if self._released:
            raise RuntimeError("prefix lease has been released")
        return self._snapshot.clone()

    def release(self) -> None:
        if not self._released:
            self._cache._release(self._cache_key)
            self._released = True

    def __enter__(self) -> PrefixLease:
        if self._released:
            raise RuntimeError("prefix lease has been released")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


class PrefixCache:
    """Small correctness-first prefix repository with explicit leases."""

    def __init__(self) -> None:
        self._entries: dict[_PrefixCacheKey, PrefixSnapshot] = {}
        self._lease_counts: dict[_PrefixCacheKey, int] = {}
        self._quarantined: dict[_PrefixCacheKey, str] = {}
        self._lock = RLock()

    def publish(self, snapshot: PrefixSnapshot) -> bool:
        snapshot.validate()
        private_copy = snapshot.clone()
        cache_key = (snapshot.prefix_key, snapshot.source_rank)
        with self._lock:
            quarantine_reason = self._quarantined.get(cache_key)
            if quarantine_reason is not None:
                raise RuntimeError(
                    f"prefix key is quarantined after a correctness failure: {quarantine_reason}"
                )
            existing = self._entries.get(cache_key)
            if existing is not None:
                if existing.compute_semantic_digest() != snapshot.compute_semantic_digest():
                    self._quarantined[cache_key] = (
                        "one prefix key identified two different semantic state bundles"
                    )
                    raise ValueError("one prefix key cannot identify two state bundles")
                return False
            self._entries[cache_key] = private_copy
            self._lease_counts[cache_key] = 0
            return True

    def acquire(self, prefix_key: str, *, source_rank: int) -> PrefixLease | None:
        cache_key = (prefix_key, source_rank)
        with self._lock:
            quarantine_reason = self._quarantined.get(cache_key)
            if quarantine_reason is not None:
                raise RuntimeError(
                    f"prefix key is quarantined after a correctness failure: {quarantine_reason}"
                )
            snapshot = self._entries.get(cache_key)
            if snapshot is None:
                return None
            snapshot.validate()
            self._lease_counts[cache_key] += 1
            return PrefixLease(self, cache_key, snapshot)

    def evict(self, prefix_key: str, *, source_rank: int) -> bool:
        cache_key = (prefix_key, source_rank)
        with self._lock:
            if cache_key not in self._entries:
                return False
            if self._lease_counts[cache_key]:
                return False
            del self._entries[cache_key]
            del self._lease_counts[cache_key]
            return True

    def quarantine_reason(self, prefix_key: str, *, source_rank: int) -> str | None:
        """Return the persistent correctness quarantine for one cache identity."""

        with self._lock:
            return self._quarantined.get((prefix_key, source_rank))

    def _release(self, cache_key: _PrefixCacheKey) -> None:
        with self._lock:
            count = self._lease_counts.get(cache_key)
            if count is None or count <= 0:
                raise RuntimeError("prefix lease accounting underflow")
            self._lease_counts[cache_key] = count - 1


@dataclass(frozen=True, slots=True)
class PrefixRestore:
    source_rank: int
    target_rank: int
    owner_id: str
    owner_revision: int
    source_to_restored_pages: tuple[tuple[int, int], ...]
    state_ids: tuple[str, ...]
    state_positions: tuple[tuple[str, int], ...]
    state_valid_mask: tuple[bool, ...]
    physical_block_boundary: int
    arithmetic_state_commit_boundary: int
    replay_tokens: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PhysicalPageRef:
    source_page_id: int
    revision: int
    slot: ArenaSlotRef


@dataclass(frozen=True, slots=True)
class PhysicalStateRef:
    state_id: str
    revision: int
    position: int
    valid: bool
    slot: ArenaSlotRef


@dataclass(frozen=True, slots=True)
class RankPrefixSource:
    rank_id: int
    owner_id: str
    owner_revision: int
    page_ids: tuple[int, ...]
    state_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RankPrefixResidence:
    rank_id: int
    arena_id: str
    arena_domain: CacheArenaDomain
    backing_identity: tuple[int, ...]
    source_owner_id: str
    source_revision: int
    token_boundary: int
    content_digest: str
    pages: tuple[PhysicalPageRef, ...]
    states: tuple[PhysicalStateRef, ...]


@dataclass(frozen=True, slots=True)
class PrefixExtensionComponentSpec:
    """One schema-declared sequence or point-state boundary relation."""

    component_id: str
    form: str
    boundary_lag_tokens: int
    block_size_tokens: int = 1
    position_start: int = 0

    def validate(self) -> None:
        if (
            type(self.component_id) is not str
            or not self.component_id
            or type(self.form) is not str
            or self.form not in ("sequence", "boundary")
        ):
            raise ValueError("prefix extension component identity/form is invalid")
        if any(
            type(value) is not int
            for value in (
                self.boundary_lag_tokens,
                self.block_size_tokens,
                self.position_start,
            )
        ):
            raise ValueError("prefix extension component arithmetic must use exact integers")
        if min(self.boundary_lag_tokens, self.position_start) < 0:
            raise ValueError("prefix extension component lag/position must be non-negative")
        if self.block_size_tokens <= 0:
            raise ValueError("prefix extension component block size must be positive")


@dataclass(frozen=True, slots=True)
class PrefixExtensionComponent:
    """One rank's immutable extension blob with explicit length and alignment."""

    component_id: str
    form: str
    target_token_boundary: int
    boundary_lag_tokens: int
    token_count: int
    position_start: int
    position_end: int
    block_size_tokens: int
    valid_tail_tokens: int
    storage_kind: str
    content_digest: str

    def validate(self, spec: PrefixExtensionComponentSpec) -> None:
        if type(spec) is not PrefixExtensionComponentSpec:
            raise ValueError("prefix extension blob requires an exact component schema")
        spec.validate()
        if (
            type(self.component_id) is not str
            or type(self.form) is not str
            or self.component_id != spec.component_id
            or self.form != spec.form
        ):
            raise ValueError("prefix extension blob differs from its schema component")
        if any(
            type(value) is not int
            for value in (
                self.target_token_boundary,
                self.boundary_lag_tokens,
                self.token_count,
                self.position_start,
                self.position_end,
                self.block_size_tokens,
                self.valid_tail_tokens,
            )
        ):
            raise ValueError("prefix extension blob arithmetic must use exact integers")
        if self.target_token_boundary <= 0:
            raise ValueError("prefix extension target boundary must be positive")
        if (
            self.boundary_lag_tokens != spec.boundary_lag_tokens
            or self.block_size_tokens != spec.block_size_tokens
            or (self.form == "sequence" and self.position_start != spec.position_start)
        ):
            raise ValueError("prefix extension blob lag/layout differs from its schema")
        state_boundary = self.target_token_boundary - self.boundary_lag_tokens
        expected_count = state_boundary - self.position_start if self.form == "sequence" else 1
        if state_boundary < 0 or self.token_count != expected_count:
            raise ValueError("prefix extension blob length violates its declared lag")
        if self.position_end != self.position_start + self.token_count:
            raise ValueError("prefix extension blob positions do not cover its token length")
        if self.form == "boundary" and self.position_start != state_boundary:
            raise ValueError("prefix extension point state is stored at the wrong boundary")
        expected_tail = (
            0 if self.token_count == 0 else (self.token_count - 1) % self.block_size_tokens + 1
        )
        if self.valid_tail_tokens != expected_tail:
            raise ValueError("prefix extension blob valid tail is inconsistent")
        if type(self.storage_kind) is not str or self.storage_kind != "private_immutable_blob":
            raise ValueError("prefix extension storage must be private and immutable")
        _require_content_digest(self.content_digest, label="prefix extension content")


@dataclass(frozen=True, slots=True)
class PrefixExtensionSchema:
    """Model-independent identity and boundary relations for extension state."""

    kind: str
    schema_id: str
    semantic_digest: str
    components: tuple[PrefixExtensionComponentSpec, ...]

    def validate(self) -> None:
        if (
            type(self.kind) is not str
            or not self.kind
            or self.kind == "none"
            or type(self.schema_id) is not str
            or not self.schema_id
        ):
            raise ValueError("prefix extension kind/schema identity is required")
        _require_content_digest(self.semantic_digest, label="prefix extension semantic identity")
        if type(self.components) is not tuple or any(
            type(component) is not PrefixExtensionComponentSpec for component in self.components
        ):
            raise ValueError("prefix extension schema components must be an exact tuple")
        for component in self.components:
            component.validate()
        component_ids = tuple(component.component_id for component in self.components)
        if not component_ids or len(set(component_ids)) != len(component_ids):
            raise ValueError("prefix extension component identities must be non-empty and unique")


@dataclass(frozen=True, slots=True)
class RankPrefixExtension:
    """Complete extension component manifest for one topology rank."""

    rank_id: int
    target_token_boundary: int
    components: tuple[PrefixExtensionComponent, ...]

    def validate(self, schema: PrefixExtensionSchema) -> None:
        if type(schema) is not PrefixExtensionSchema:
            raise ValueError("rank extension requires an exact extension schema")
        schema.validate()
        if (
            type(self.rank_id) is not int
            or type(self.target_token_boundary) is not int
            or self.rank_id < 0
            or self.target_token_boundary <= 0
        ):
            raise ValueError("rank extension identity/boundary is invalid")
        if type(self.components) is not tuple or any(
            type(component) is not PrefixExtensionComponent for component in self.components
        ):
            raise ValueError("rank extension components must be an exact tuple")
        if tuple(component.component_id for component in self.components) != tuple(
            component.component_id for component in schema.components
        ):
            raise ValueError("rank extension is missing or reordering a component")
        for component, spec in zip(self.components, schema.components, strict=True):
            if component.target_token_boundary != self.target_token_boundary:
                raise ValueError("rank extension component has a different target boundary")
            component.validate(spec)


def prefix_extension_content_digest(component_id: str, tensors: tuple[torch.Tensor, ...]) -> str:
    if type(component_id) is not str or not component_id or type(tensors) is not tuple:
        raise ValueError("prefix extension payload identity/tensors are invalid")
    digest = hashlib.blake2s()
    digest.update(component_id.encode())
    for tensor in tensors:
        if (
            type(tensor) is not torch.Tensor
            or tensor.layout is not torch.strided
            or tensor.numel() == 0
            or tensor.device.type == "meta"
        ):
            raise ValueError("prefix extension payload requires realized strided tensors")
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all().item()
        ):
            raise ValueError("prefix extension payload tensors must be finite")
        _hash_tensor(digest, tensor)
    return f"b2s256:{digest.hexdigest()}"


PrefixExtensionTensorBundle = TypeAliasType("PrefixExtensionTensorBundle", tuple[str, tuple[torch.Tensor, ...]])


@dataclass(frozen=True, slots=True)
class RankPrefixExtensionPayload:
    """One rank's private tensor blobs; cache and consumer each receive clones."""

    rank_id: int
    components: tuple[PrefixExtensionTensorBundle, ...]

    def validate(self, manifest: RankPrefixExtension) -> None:
        if type(self.rank_id) is not int or self.rank_id != manifest.rank_id:
            raise ValueError("prefix extension payload rank differs from its manifest")
        if type(self.components) is not tuple or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not tuple
            for item in self.components
        ):
            raise ValueError("prefix extension payload components must be exact tuples")
        if tuple(item[0] for item in self.components) != tuple(
            component.component_id for component in manifest.components
        ):
            raise ValueError("prefix extension payload is missing or reordering a component")
        for (component_id, tensors), component in zip(
            self.components, manifest.components, strict=True
        ):
            if prefix_extension_content_digest(component_id, tensors) != component.content_digest:
                raise ValueError("prefix extension payload content differs from its manifest")

    def clone(self) -> RankPrefixExtensionPayload:
        return RankPrefixExtensionPayload(
            self.rank_id,
            tuple(
                (component_id, tuple(tensor.detach().clone() for tensor in tensors))
                for component_id, tensors in self.components
            ),
        )


@dataclass(frozen=True, slots=True)
class PrefixBundleManifest:
    """Joint, block-aligned target state; partial-boundary replay is excluded."""

    prefix_key: str
    topology_id: str
    ranks: tuple[int, ...]
    token_boundary: int
    extension_kind: str = "none"
    extensions: tuple[RankPrefixExtension, ...] = ()
    extension_schema: PrefixExtensionSchema | None = None

    def validate(self) -> None:
        if (
            type(self.prefix_key) is not str
            or not self.prefix_key
            or type(self.topology_id) is not str
            or not self.topology_id
            or type(self.token_boundary) is not int
            or self.token_boundary <= 0
        ):
            raise ValueError("physical prefix identity and positive boundary are required")
        if (
            type(self.ranks) is not tuple
            or not self.ranks
            or any(type(rank) is not int for rank in self.ranks)
            or tuple(sorted(set(self.ranks))) != self.ranks
        ):
            raise ValueError("physical prefix ranks must be non-empty, unique, and sorted")
        if self.ranks[0] < 0:
            raise ValueError("physical prefix ranks must be non-negative")
        if type(self.extension_kind) is not str:
            raise ValueError("physical prefix extension kind must be a string")
        if type(self.extensions) is not tuple or any(
            type(extension) is not RankPrefixExtension for extension in self.extensions
        ):
            raise ValueError("physical prefix rank extensions must be an exact tuple")
        if self.extension_kind == "none":
            if self.extension_schema is not None or self.extensions:
                raise ValueError("NONE physical prefix cannot carry extension state")
            return
        if self.extension_schema is None or self.extension_schema.kind != self.extension_kind:
            raise ValueError("physical prefix extension schema is missing or has the wrong kind")
        self.extension_schema.validate()
        if tuple(extension.rank_id for extension in self.extensions) != self.ranks:
            raise ValueError("physical prefix extensions must cover every rank in order")
        for extension in self.extensions:
            if extension.target_token_boundary != self.token_boundary:
                raise ValueError("physical prefix extension has a different target boundary")
            extension.validate(self.extension_schema)

    def extension_identity(self) -> PrefixExtensionSchema | None:
        """Return the validated cache-variant identity without inspecting payloads."""

        if type(self.extension_kind) is not str:
            raise ValueError("physical prefix extension kind must be a string")
        if type(self.extensions) is not tuple:
            raise ValueError("physical prefix rank extensions must be an exact tuple")
        if self.extension_kind == "none":
            if self.extension_schema is not None or self.extensions:
                raise ValueError("NONE physical prefix cannot carry extension state")
            return None
        if (
            type(self.extension_schema) is not PrefixExtensionSchema
            or self.extension_schema.kind != self.extension_kind
        ):
            raise ValueError("physical prefix extension schema is missing or has the wrong kind")
        self.extension_schema.validate()
        return self.extension_schema


class _PhysicalPrefixStore(Protocol):
    rank_id: int
    prefix_lock: RLock

    def pin_prefix_part(
        self, source: RankPrefixSource, *, token_boundary: int
    ) -> RankPrefixResidence: ...

    def release_prefix_part(self, part: RankPrefixResidence) -> None: ...

    def validate_prefix_part(self, part: RankPrefixResidence) -> None: ...

    def prepare_prefix_restore(self, part: RankPrefixResidence, owner_id: str) -> Any: ...

    def validate_prepared_prefix_restore(self, prepared: Any) -> None: ...

    def activate_prefix_restore(self, prepared: Any) -> PrefixRestore: ...

    def abort_prefix_restore(self, prepared: Any) -> None: ...

    def retire_owner(
        self, owner_id: str, *, expected_revision: int, expected_page_ids: tuple[int, ...]
    ) -> None: ...


@dataclass(slots=True)
class _PhysicalPrefixEntry:
    manifest: PrefixBundleManifest
    generation: int
    parts: tuple[RankPrefixResidence, ...]
    extension_payloads: tuple[RankPrefixExtensionPayload, ...] = ()
    leases: int = 0
    draining: bool = False
    quarantined: bool = False


class PrefixBundleLease:
    def __init__(self, cache: PrefixBundleCache, entry: _PhysicalPrefixEntry) -> None:
        self._cache = cache
        self._entry = entry
        self._closed = False
        self._restored = False

    @property
    def generation(self) -> int:
        return self._entry.generation

    @property
    def extension(self) -> tuple[PrefixExtensionSchema, tuple[RankPrefixExtension, ...]] | None:
        """Return immutable extension metadata selected with this exact lease."""

        if self._closed:
            raise RuntimeError("physical prefix lease is closed")
        schema = self._entry.manifest.extension_schema
        if schema is None:
            return None
        return schema, self._entry.manifest.extensions

    @property
    def extension_payloads(self) -> tuple[RankPrefixExtensionPayload, ...]:
        if self._closed:
            raise RuntimeError("physical prefix lease is closed")
        return tuple(payload.clone() for payload in self._entry.extension_payloads)

    def restore(self, owner_ids: tuple[tuple[int, str], ...]) -> tuple[PrefixRestore, ...]:
        if self._entry.manifest.extension_schema is not None:
            raise RuntimeError("extension prefixes require atomic restore_bundle")
        return self._restore_target(owner_ids)

    def restore_bundle(
        self, owner_ids: tuple[tuple[int, str], ...]
    ) -> tuple[tuple[PrefixRestore, ...], tuple[RankPrefixExtensionPayload, ...]]:
        payloads = self.extension_payloads
        return self._restore_target(owner_ids), payloads

    def _restore_target(self, owner_ids: tuple[tuple[int, str], ...]) -> tuple[PrefixRestore, ...]:
        if self._closed or self._restored:
            raise RuntimeError("physical prefix lease is closed or already restored")
        with self._cache._lock:
            if self._entry.quarantined:
                raise PhysicalPrefixConflict("physical prefix entry is quarantined")
        expected_ranks = self._entry.manifest.ranks
        if tuple(rank for rank, _ in owner_ids) != expected_ranks:
            raise ValueError("restore owners must cover every topology rank in order")
        stores = tuple(self._cache._stores[rank] for rank in expected_ranks)
        prepared: list[tuple[_PhysicalPrefixStore, Any]] = []
        with ExitStack() as locks:
            for store in stores:
                locks.enter_context(store.prefix_lock)
            try:
                for store, part, (_, owner_id) in zip(
                    stores, self._entry.parts, owner_ids, strict=True
                ):
                    prepared.append((store, store.prepare_prefix_restore(part, owner_id)))
                for store, item in prepared:
                    store.validate_prepared_prefix_restore(item)
                restores = tuple(store.activate_prefix_restore(item) for store, item in prepared)
            except Exception:
                for store, item in reversed(prepared):
                    store.abort_prefix_restore(item)
                raise
        self._restored = True
        return restores

    def release(self) -> None:
        if not self._closed:
            self._cache._release(self._entry)
            self._closed = True

    def __enter__(self) -> PrefixBundleLease:
        if self._closed:
            raise RuntimeError("physical prefix lease is closed")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


_PhysicalPrefixKey = TypeAliasType("_PhysicalPrefixKey", tuple[str, str, PrefixExtensionSchema | None])


class PrefixBundleCache:
    """Topology-wide directory over immutable physical arena holds."""

    def __init__(self, stores: tuple[_PhysicalPrefixStore, ...]) -> None:
        ranks = tuple(store.rank_id for store in stores)
        if not stores or tuple(sorted(set(ranks))) != ranks:
            raise ValueError("physical prefix stores must have unique sorted ranks")
        self._stores = dict(zip(ranks, stores, strict=True))
        self._entries: dict[_PhysicalPrefixKey, _PhysicalPrefixEntry] = {}
        self._quarantined: dict[_PhysicalPrefixKey, str] = {}
        self._next_generation: dict[_PhysicalPrefixKey, int] = {}
        self._lock = RLock()

    def publish(
        self,
        manifest: PrefixBundleManifest,
        sources: tuple[RankPrefixSource, ...],
        *,
        extension_payloads: tuple[RankPrefixExtensionPayload, ...] = (),
    ) -> bool:
        key: _PhysicalPrefixKey | None = None
        extension_identity: PrefixExtensionSchema | None = None
        try:
            extension_identity = manifest.extension_identity()
            key = (manifest.prefix_key, manifest.topology_id, extension_identity)
            manifest.validate()
        except ValueError as error:
            if (
                key is not None
                and extension_identity is not None
                and manifest.prefix_key
                and manifest.topology_id
            ):
                with self._lock:
                    self._quarantined[key] = f"invalid prefix extension: {error}"
                    existing = self._entries.pop(key, None)
                    if existing is not None:
                        existing.draining = existing.quarantined = True
                        finalize = existing.leases == 0
                if existing is not None and finalize:
                    self._drop_directory_holds(existing)
            raise
        if (
            type(extension_payloads) is not tuple
            or tuple(
                payload.rank_id
                for payload in extension_payloads
                if type(payload) is RankPrefixExtensionPayload
            )
            != tuple(extension.rank_id for extension in manifest.extensions)
            or any(
                type(payload) is not RankPrefixExtensionPayload for payload in extension_payloads
            )
        ):
            raise ValueError("prefix extension payloads must cover every rank in order")
        for payload, extension in zip(extension_payloads, manifest.extensions, strict=True):
            payload.validate(extension)
        private_payloads = tuple(payload.clone() for payload in extension_payloads)
        if manifest.ranks != tuple(source.rank_id for source in sources):
            raise ValueError("publish sources must cover every manifest rank in order")
        if set(manifest.ranks) != set(self._stores):
            raise ValueError("manifest topology differs from the bundle cache stores")
        assert key is not None
        with self._lock:
            if key in self._quarantined:
                raise PhysicalPrefixConflict(self._quarantined[key])
        parts: list[RankPrefixResidence] = []
        stores = tuple(self._stores[rank] for rank in manifest.ranks)
        with ExitStack() as locks:
            for store in stores:
                locks.enter_context(store.prefix_lock)
            try:
                for source in sources:
                    parts.append(
                        self._stores[source.rank_id].pin_prefix_part(
                            source, token_boundary=manifest.token_boundary
                        )
                    )
            except Exception:
                for part in reversed(parts):
                    self._stores[part.rank_id].release_prefix_part(part)
                raise
        with self._lock:
            quarantine_reason = self._quarantined.get(key)
            existing = None if quarantine_reason is not None else self._entries.get(key)
            if existing is not None:
                same = existing.manifest == manifest and tuple(
                    part.content_digest for part in existing.parts
                ) == tuple(part.content_digest for part in parts)
                if same:
                    for part in reversed(parts):
                        self._stores[part.rank_id].release_prefix_part(part)
                    return False
                quarantine_reason = "one physical prefix key names conflicting state content"
                self._quarantined[key] = quarantine_reason
                del self._entries[key]
                existing.draining = existing.quarantined = True
                finalize = existing.leases == 0
            elif quarantine_reason is None:
                generation = self._next_generation.get(key, 0) + 1
                self._next_generation[key] = generation
                self._entries[key] = _PhysicalPrefixEntry(
                    manifest,
                    generation,
                    tuple(parts),
                    private_payloads,
                )
                return True
        for part in reversed(parts):
            self._stores[part.rank_id].release_prefix_part(part)
        if existing is not None and finalize:
            self._drop_directory_holds(existing)
        assert quarantine_reason is not None
        raise PhysicalPrefixConflict(quarantine_reason)

    def acquire(
        self,
        prefix_key: str,
        *,
        topology_id: str,
        extension_schema: PrefixExtensionSchema | None = None,
    ) -> PrefixBundleLease | None:
        if extension_schema is not None:
            extension_schema.validate()
        with self._lock:
            key = (prefix_key, topology_id, extension_schema)
            if key in self._quarantined:
                raise PhysicalPrefixConflict(self._quarantined[key])
            entry = self._entries.get(key)
            if entry is None or entry.manifest.extension_schema != extension_schema:
                return None
            entry.leases += 1
            return PrefixBundleLease(self, entry)

    def evict(
        self,
        prefix_key: str,
        *,
        topology_id: str,
        extension_schema: PrefixExtensionSchema | None = None,
    ) -> bool:
        if extension_schema is not None:
            extension_schema.validate()
        with self._lock:
            entry = self._entries.pop((prefix_key, topology_id, extension_schema), None)
            if entry is None:
                return False
            entry.draining = True
            finalize = entry.leases == 0
        if finalize:
            self._drop_directory_holds(entry)
        return True

    def _release(self, entry: _PhysicalPrefixEntry) -> None:
        with self._lock:
            if entry.leases <= 0:
                raise RuntimeError("physical prefix lease accounting underflow")
            entry.leases -= 1
            finalize = entry.draining and entry.leases == 0
        if finalize:
            self._drop_directory_holds(entry)

    def _drop_directory_holds(self, entry: _PhysicalPrefixEntry) -> None:
        stores = tuple(self._stores[part.rank_id] for part in entry.parts)
        with ExitStack() as locks:
            for store in stores:
                locks.enter_context(store.prefix_lock)
            for store, part in zip(stores, entry.parts, strict=True):
                store.validate_prefix_part(part)
            for store, part in reversed(tuple(zip(stores, entry.parts, strict=True))):
                store.release_prefix_part(part)
