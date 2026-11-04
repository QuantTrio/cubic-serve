"""First end-to-end eager reference engine over real scheduler and cache plans."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Protocol

import torch

from cubic_serve.contracts.execution import BatchPlan, PlanSegment, StateTxn
from cubic_serve.contracts.model import PrefixReplayPolicy, ScalarDType
from cubic_serve.contracts.request import InputPlan, TokenEvent, TokenEventKind, TokenPlacement
from cubic_serve.executor.runner import (
    ReferenceExecutionContext,
    ReferenceMathPolicy,
    ReferenceOpExecutor,
    ReferenceSegmentIdentity,
    configure_reference_math,
    require_reference_execution,
)
from cubic_serve.generation import GreedySampler, LogitConstraint
from cubic_serve.inputs import FeatureStore
from cubic_serve.models import (
    CheckpointWeightSource,
    FullBF16WeightPolicy,
    LoadedModelAdapter,
    ModelBuildContext,
    ModelProgram,
    ModelSegment,
    ModelStateIO,
    ModelStepResult,
    PreparedInput,
    ProgramExecutionRecipe,
    WeightEvidence,
    WeightIntegrityError,
    seal_attested_full_bf16_weight_set,
    seal_full_bf16_weight_set,
)
from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope
from cubic_serve.runtime.cache import (
    CacheStateStore,
    KVLayout,
    PrefixCache,
    PrefixSnapshot,
    RankLocalPageAllocator,
    RecurrentStateProvision,
    RecurrentStateSnapshot,
    StateTransaction,
    TxnPlan,
)
from cubic_serve.runtime.scheduler import ContinuousScheduler, SegmentCompletion


@dataclass(frozen=True, slots=True)
class PrefixSnapshotIdentity:
    """Stable prefix identity with allocator-local page ids deliberately omitted."""

    prefix_key: str
    source_rank: int
    source_owner_id: str
    source_revision: int
    physical_block_boundary: int
    arithmetic_state_commit_boundary: int
    state_commit_alignment: int
    semantic_digest: str

    def __post_init__(self) -> None:
        for name, value in (
            ("prefix key", self.prefix_key),
            ("prefix semantic digest", self.semantic_digest),
        ):
            algorithm, separator, digest = value.partition(":")
            if algorithm != "b2s256" or separator != ":" or len(digest) != 64:
                raise ValueError(f"{name} must use b2s256:<64 hex chars>")
            try:
                int(digest, 16)
            except ValueError as error:
                raise ValueError(f"{name} contains non-hex characters") from error
        if not self.source_owner_id or min(self.source_rank, self.source_revision) < 0:
            raise ValueError("prefix source identity and non-negative revision are required")
        if self.physical_block_boundary <= 0 or self.state_commit_alignment <= 0:
            raise ValueError("prefix physical boundary and state alignment must be positive")
        if not 0 <= self.arithmetic_state_commit_boundary <= self.physical_block_boundary:
            raise ValueError("prefix arithmetic boundary must lie within physical KV")
        if self.arithmetic_state_commit_boundary % self.state_commit_alignment:
            raise ValueError("prefix arithmetic boundary violates state alignment")


@dataclass(frozen=True, slots=True)
class PrefixAdmission:
    hit: bool
    token_count: int
    resume_token_count: int
    prefix_key: str | None
    candidate_keys: tuple[tuple[int, str], ...] = ()
    snapshot: PrefixSnapshotIdentity | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.resume_token_count <= self.token_count:
            raise ValueError("prefix resume boundary must lie within physical KV")
        if self.hit != (self.prefix_key is not None):
            raise ValueError("prefix hit and key presence must agree")
        boundaries = tuple(boundary for boundary, _ in self.candidate_keys)
        if boundaries != tuple(sorted(set(boundaries), reverse=True)) or any(
            boundary <= 0 for boundary in boundaries
        ):
            raise ValueError("prefix candidate boundaries must be positive, unique, descending")
        for _, key in self.candidate_keys:
            algorithm, separator, digest = key.partition(":")
            if algorithm != "b2s256" or separator != ":" or len(digest) != 64:
                raise ValueError("prefix candidate key must use b2s256:<64 hex chars>")
            try:
                int(digest, 16)
            except ValueError as error:
                raise ValueError("prefix candidate key contains non-hex characters") from error
        if not self.hit:
            if self.token_count or self.resume_token_count or self.snapshot is not None:
                raise ValueError("prefix miss cannot claim restored state")
            return
        if self.snapshot is None:
            raise ValueError("prefix hit requires stable snapshot identity")
        if (
            self.prefix_key != self.snapshot.prefix_key
            or self.token_count != self.snapshot.physical_block_boundary
            or self.resume_token_count != self.snapshot.arithmetic_state_commit_boundary
            or (self.token_count, self.prefix_key) not in self.candidate_keys
        ):
            raise ValueError("prefix admission differs from its selected snapshot identity")

    @property
    def base_revision(self) -> int:
        return 0 if self.snapshot is None else self.snapshot.source_revision


@dataclass(frozen=True, slots=True)
class PrefixPublication:
    """Post-commit prefix evidence without allocator-local physical page ids."""

    request_id: str
    plan_id: str
    segment_index: int
    snapshot: PrefixSnapshotIdentity
    inserted: bool

    def __post_init__(self) -> None:
        if not self.request_id or not self.plan_id or self.segment_index < 0:
            raise ValueError("prefix publication requires request/plan/segment identity")
        if self.snapshot.source_owner_id != self.request_id:
            raise ValueError("prefix publication source owner differs from its request")
        if type(self.inserted) is not bool:
            raise TypeError("prefix publication inserted must be a bool")


def _prefix_snapshot_identity(snapshot: PrefixSnapshot) -> PrefixSnapshotIdentity:
    snapshot.validate()
    return PrefixSnapshotIdentity(
        prefix_key=snapshot.prefix_key,
        source_rank=snapshot.source_rank,
        source_owner_id=snapshot.source_owner_id,
        source_revision=snapshot.source_revision,
        physical_block_boundary=snapshot.physical_block_boundary,
        arithmetic_state_commit_boundary=snapshot.arithmetic_state_commit_boundary,
        state_commit_alignment=snapshot.state_commit_alignment,
        semantic_digest=snapshot.compute_semantic_digest(),
    )


@dataclass(frozen=True, slots=True)
class EngineLogitsRow:
    """Detached real logits row emitted by one committed segment."""

    request_id: str
    plan_id: str
    segment_index: int
    row_index: int
    tensor: torch.Tensor = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.request_id or not self.plan_id:
            raise ValueError("engine logits row requires request and plan identities")
        if self.segment_index < 0 or self.row_index < 0:
            raise ValueError("engine logits row indices must be non-negative")
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError("engine logits row must contain a tensor")
        if self.tensor.ndim != 1 or self.tensor.numel() == 0:
            raise ValueError("engine logits row must be non-empty and rank one")
        if not self.tensor.is_floating_point() or not bool(
            torch.isfinite(self.tensor).all().item()
        ):
            raise ValueError("engine logits row must contain finite floating values")
        object.__setattr__(self, "tensor", self.tensor.detach().clone())

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (self.request_id, self.plan_id, self.segment_index, self.row_index)


@dataclass(frozen=True, slots=True)
class PrefixPublicationFailure:
    """Visible evidence that an optional post-commit prefix side effect failed."""

    request_id: str
    prefix_token_count: int
    error_type: str
    message: str

    def __post_init__(self) -> None:
        if not self.request_id or self.prefix_token_count <= 0:
            raise ValueError("prefix failure requires request identity and a positive boundary")
        if not self.error_type or not self.message:
            raise ValueError("prefix failure requires an explicit error type and message")


@dataclass(frozen=True, slots=True)
class EngineStep:
    batch_plan: BatchPlan
    events: tuple[TokenEvent, ...]
    published_prefix_keys: tuple[str, ...]
    logits_rows: tuple[EngineLogitsRow, ...]
    prefix_publications: tuple[PrefixPublication, ...] = ()
    prefix_publication_failures: tuple[PrefixPublicationFailure, ...] = ()

    def __post_init__(self) -> None:
        keys = tuple(row.key for row in self.logits_rows)
        if len(set(keys)) != len(keys):
            raise ValueError("engine step logits row keys must be unique")
        for row in self.logits_rows:
            if row.plan_id != self.batch_plan.plan_id:
                raise ValueError("engine logits row plan differs from EngineStep")
            try:
                segment = self.batch_plan.segments[row.segment_index]
            except IndexError as error:
                raise ValueError("engine logits row references an absent segment") from error
            if segment.request_id != row.request_id or row.row_index >= segment.token_count:
                raise ValueError("engine logits row differs from its BatchPlan segment")
        inserted_keys = tuple(
            publication.snapshot.prefix_key
            for publication in self.prefix_publications
            if publication.inserted
        )
        if inserted_keys != self.published_prefix_keys:
            raise ValueError("published prefix keys differ from publication evidence")
        publication_keys = tuple(
            (publication.plan_id, publication.segment_index)
            for publication in self.prefix_publications
        )
        if len(set(publication_keys)) != len(publication_keys):
            raise ValueError("prefix publication evidence must be unique per segment")
        for publication in self.prefix_publications:
            if publication.plan_id != self.batch_plan.plan_id:
                raise ValueError("prefix publication plan differs from EngineStep")
            try:
                segment = self.batch_plan.segments[publication.segment_index]
                transaction = self.batch_plan.state_transactions[publication.segment_index]
            except IndexError as error:
                raise ValueError("prefix publication references an absent segment") from error
            planned = transaction.prefix_publish
            snapshot = publication.snapshot
            if (
                segment.request_id != publication.request_id
                or planned is None
                or snapshot.source_revision != transaction.target_revision
                or snapshot.physical_block_boundary != planned.prefix_token_count
                or snapshot.arithmetic_state_commit_boundary != planned.state_commit_position
                or snapshot.state_commit_alignment != planned.commit_alignment_tokens
            ):
                raise ValueError("prefix publication differs from its committed segment plan")

    @property
    def post_commit_ok(self) -> bool:
        return not self.prefix_publication_failures


@dataclass(frozen=True, slots=True)
class CommittedStateDigest:
    request_id: str
    owner_rank: int
    state_revision: int
    cache_digest: str
    state_digest: str


@dataclass(frozen=True, slots=True)
class ReferenceExecutionFingerprint:
    """Core-derived identity used to bind deterministic replay evidence."""

    executor_id: str
    contract_digest: str
    model_digest: str
    implementation_digest: str
    topology_digest: str
    evidence_digest: str


class _TransactionStateIO(ModelStateIO):
    def __init__(
        self,
        transaction: StateTransaction,
        *,
        identity: ReferenceSegmentIdentity,
        logical_to_physical: tuple[int, ...],
        block_size_tokens: int,
    ) -> None:
        if transaction.plan.owner_id != identity.segment.request_id:
            raise ValueError("state capability transaction owner differs from its segment")
        expected_position = identity.segment.token_start + identity.segment.token_count
        if transaction.plan.target_position not in (None, expected_position):
            raise ValueError("state capability transaction position differs from its segment")
        self._transaction = transaction
        self._identity = identity
        self._logical_to_physical = logical_to_physical
        self._block_size = block_size_tokens
        self._active = True

    def _require_active(self) -> None:
        require_reference_execution()
        if not self._active:
            raise RuntimeError("segment state capability was revoked")

    def revoke(self) -> None:
        self._active = False

    def _groups(
        self, token_ordinals: tuple[int, ...]
    ) -> tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]:
        self._require_active()
        if not token_ordinals or tuple(sorted(set(token_ordinals))) != token_ordinals:
            raise ValueError("logical KV ordinals must be non-empty, unique, and sorted")
        grouped: list[tuple[int, list[int], list[int]]] = []
        for row_index, ordinal in enumerate(token_ordinals):
            logical_page, offset = divmod(ordinal, self._block_size)
            try:
                physical_page = self._logical_to_physical[logical_page]
            except IndexError as error:
                raise ValueError("logical KV ordinal is outside the bound page table") from error
            if not grouped or grouped[-1][0] != physical_page:
                grouped.append((physical_page, [], []))
            grouped[-1][1].append(offset)
            grouped[-1][2].append(row_index)
        return tuple((page, tuple(offsets), tuple(indices)) for page, offsets, indices in grouped)

    def read_kv(
        self, layer_id: int, token_ordinals: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for page_id, offsets, _ in self._groups(token_ordinals):
            key, value = self._transaction.read_kv(page_id, layer_id, offsets)
            keys.append(key)
            values.append(value)
        return torch.cat(keys, dim=0), torch.cat(values, dim=0)

    def write_kv(
        self,
        layer_id: int,
        token_ordinals: tuple[int, ...],
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        if key.shape[0] != len(token_ordinals) or value.shape[0] != len(token_ordinals):
            raise ValueError("logical KV payload row count does not match token ordinals")
        for page_id, offsets, indices in self._groups(token_ordinals):
            selection = torch.tensor(indices, dtype=torch.long, device=key.device)
            self._transaction.write_kv(
                page_id,
                layer_id,
                offsets,
                key.index_select(0, selection),
                value.index_select(0, selection),
            )

    def read_recurrent_state(self, state_id: str) -> torch.Tensor:
        self._require_active()
        return self._transaction.read_recurrent_state(state_id)

    def write_recurrent_state(
        self,
        state_id: str,
        value: torch.Tensor,
        *,
        target_position: int,
        valid: bool = True,
    ) -> None:
        self._require_active()
        self._transaction.write_recurrent_state(
            state_id,
            value,
            target_position=target_position,
            valid=valid,
        )


class ReferenceEngine:
    """Synchronous correctness engine; performance execution is intentionally absent."""

    def __init__(
        self,
        loaded_adapter: LoadedModelAdapter,
        *,
        kv_layout: KVLayout,
        weight_policy: FullBF16WeightPolicy,
        device: torch.device | str = "cpu",
        token_budget: int = 4,
        prefill_chunk_tokens: int = 4,
        topology_id: str = "single",
        partition_id: str = "single:rank0",
        owned_layer_ids: tuple[int, ...] | None = None,
        model_root: Path | None = None,
        checkpoint_weights: CheckpointWeightSource | None = None,
        prefix_cache: PrefixCache | None = None,
        eos_token_ids: tuple[int, ...] = (),
        operator_config: OperatorConfig | None = None,
    ) -> None:
        self._operator_config = operator_config or OperatorConfig()
        self._operator_digest = self._operator_config.digest
        self._reference_policy: ReferenceMathPolicy = configure_reference_math()
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        descriptor = loaded_adapter.descriptor
        if kv_layout.block_size_tokens != descriptor.model_spec.cache.page_size_tokens:
            raise ValueError("runtime KV block size differs from the sealed ModelSpec")
        if descriptor.model_spec.layer_count is not None and any(
            layer >= descriptor.model_spec.layer_count for layer in kv_layout.layer_ids
        ):
            raise ValueError("runtime KV layout owns a layer outside the model graph")
        resolved_owned_layers = kv_layout.layer_ids if owned_layer_ids is None else owned_layer_ids
        if tuple(sorted(set(resolved_owned_layers))) != resolved_owned_layers:
            raise ValueError("owned model layer ids must be non-negative, unique, and sorted")
        if not set(kv_layout.layer_ids) <= set(resolved_owned_layers):
            raise ValueError("every rank-local KV layer must belong to the model partition")
        if descriptor.model_spec.layer_count is not None and any(
            layer < 0 or layer >= descriptor.model_spec.layer_count
            for layer in resolved_owned_layers
        ):
            raise ValueError("model partition owns a layer outside the model graph")
        if tuple(sorted(set(eos_token_ids))) != eos_token_ids or any(
            token_id < 0 for token_id in eos_token_ids
        ):
            raise ValueError("EOS token ids must be non-negative, unique, and sorted")

        self.loaded_adapter = loaded_adapter
        self._topology_id = topology_id
        self._partition_id = partition_id
        self._owned_layer_ids = resolved_owned_layers
        self.store = CacheStateStore(kv_layout, rank_id=0, device=self.device)
        self.allocator = RankLocalPageAllocator(self.store)
        self.prefix_cache = prefix_cache if prefix_cache is not None else PrefixCache()
        self.features = FeatureStore()
        self._ops = ReferenceOpExecutor(
            loaded_adapter.reference_plan,
            loaded_adapter.reference_catalog,
        )
        if any(signature.topology_id != topology_id for signature in self._ops.signatures):
            raise ValueError("reference operator plan differs from the engine topology")
        with (
            ReferenceExecutionContext(),
            operator_strategy_scope(self._operator_config, device=str(self.device), model=""),
        ):
            self._program: ModelProgram = loaded_adapter.build_program(
                ModelBuildContext(
                    device=self.device,
                    rank_id=0,
                    world_size=1,
                    owned_layer_ids=resolved_owned_layers,
                    model_root=model_root,
                )
            )
            self._identity_program = self._program
            try:
                recipe = self._identity_program.execution_recipe()
            except Exception as error:
                raise WeightIntegrityError("model program recipe could not be sealed") from error
            require_reference_execution()
            if not isinstance(recipe, ProgramExecutionRecipe):
                raise WeightIntegrityError(
                    "ModelProgram.execution_recipe() must return ProgramExecutionRecipe"
                )
            if checkpoint_weights is None:
                sealed_weights = seal_full_bf16_weight_set(
                    self._identity_program.named_weight_tensors,
                    policy=weight_policy,
                )
            else:
                if model_root is None or checkpoint_weights.inventory.model_root != model_root:
                    raise ValueError(
                        "checkpoint weight evidence must identify the active model_root"
                    )
                sealed_weights = seal_attested_full_bf16_weight_set(
                    self._identity_program.named_weight_tensors,
                    inventory=checkpoint_weights.inventory,
                    policy=weight_policy,
                    semantic_plan_digest=checkpoint_weights.semantic_plan_digest,
                )
        self._program_recipe = recipe
        self._program_recipe_digest = _program_recipe_digest(
            self._identity_program,
            recipe,
            adapter_source_digest=loaded_adapter.source_digest,
        )
        self._implementation_digest = _reference_implementation_digest(loaded_adapter)
        self._sealed_weights = sealed_weights
        # This engine is deliberately world-size one, so its local executable
        # inventory must equal the global descriptor. Distributed engines must
        # instead have a coordinator prove the rank-local union before issuing
        # the global digest used by cross-rank cache/replay identities.
        if self._sealed_weights.evidence.manifest != descriptor.model_spec.weights:
            raise WeightIntegrityError(
                "single-rank descriptor WeightManifest differs from core-issued "
                "full-model tensor evidence"
            )
        state_ids = tuple(sorted(state.state_id for state in descriptor.model_spec.states))
        state_alignment = (
            math.lcm(*(state.commit_alignment_tokens for state in descriptor.model_spec.states))
            if descriptor.model_spec.states
            else 1
        )
        self.scheduler = ContinuousScheduler(
            page_allocator=self.allocator,
            token_budget=token_budget,
            prefill_chunk_tokens=prefill_chunk_tokens,
            topology_id=topology_id,
            partition_id=partition_id,
            owner_ranks=(0,),
            state_ids=state_ids,
            state_commit_alignment_tokens=state_alignment,
            decode_extension=descriptor.decode_extension,
        )
        self._state_ids = state_ids
        self._state_alignment = state_alignment
        self._input_plans: dict[str, InputPlan] = {}
        self._prepared_inputs: dict[str, PreparedInput] = {}
        self._event_sequence: dict[str, int] = {}
        self._prefix_admissions: dict[str, PrefixAdmission] = {}
        self._prefix_enabled: dict[str, bool] = {}
        self._ignore_eos: set[str] = set()
        self._constraints: dict[str, LogitConstraint] = {}
        self._sampler = GreedySampler()
        self._eos_token_ids = frozenset(eos_token_ids)
        self._lock = RLock()

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        """The exact terminal-token set used by sampling completion."""

        return tuple(sorted(self._eos_token_ids))

    def cancel(self, request_id: str) -> bool:
        """Stop future scheduling for one admitted request.

        Cancellation is deliberately not retirement or resource reclamation.
        The current store has no atomic owner-teardown ABI, so committed KV,
        recurrent state, allocator mappings, and any feature snapshot remain
        owned by ``request_id``.  The serving driver must independently stop
        publishing output for the cancelled request.

        Returns ``True`` only for the transition from active to cancelled.
        """

        with self._lock:
            state = self.scheduler.request_state(request_id)
            if state.cancelled or state.finished:
                return False
            self.scheduler.cancel(request_id)
            return True

    def preview_prefix_admission(
        self,
        prepared_input: PreparedInput,
        *,
        use_prefix_cache: bool = True,
    ) -> PrefixAdmission:
        """Inspect the exact admission outcome without initializing request state."""

        with self._lock:
            self._verify_execution_identity("prefix admission preview")
            if not isinstance(prepared_input, PreparedInput):
                raise TypeError("reference engine admission requires PreparedInput")
            self._validate_prepared_input(prepared_input)
            if prepared_input.input_plan.request_id in self._input_plans:
                raise ValueError(
                    f"request {prepared_input.input_plan.request_id!r} was already admitted"
                )
            if not use_prefix_cache:
                return PrefixAdmission(False, 0, 0, None)
            admission, _ = self._find_longest_prefix(prepared_input)
            return admission

    def admit(
        self,
        prepared_input: PreparedInput,
        *,
        max_new_tokens: int,
        use_prefix_cache: bool = True,
        expected_prefix_admission: PrefixAdmission | None = None,
        constraint: LogitConstraint | None = None,
        ignore_eos: bool = False,
    ) -> PrefixAdmission:
        with self._lock:
            self._verify_execution_identity("request admission")
            if not isinstance(prepared_input, PreparedInput):
                raise TypeError("reference engine admission requires PreparedInput")
            input_plan = prepared_input.input_plan
            self._validate_prepared_input(prepared_input)
            if prepared_input.feature_bundle is not None and any(
                binding.feature.device != self.device
                for binding in prepared_input.feature_bundle.bindings
            ):
                raise ValueError("encoded feature device differs from the reference engine")
            if max_new_tokens <= 0:
                raise ValueError("max_new_tokens must be positive")
            if type(ignore_eos) is not bool:
                raise TypeError("ignore_eos must be a boolean")
            if input_plan.request_id in self._input_plans:
                raise ValueError(f"request {input_plan.request_id!r} was already admitted")
            if constraint is not None and (
                constraint.request_id != input_plan.request_id
                or constraint.finished
                or constraint.failed
            ):
                raise ValueError("request constraint identity or lifecycle is invalid")

            admission, snapshot = (
                self._find_longest_prefix(prepared_input)
                if use_prefix_cache
                else (PrefixAdmission(False, 0, 0, None), None)
            )
            if expected_prefix_admission is not None and admission != expected_prefix_admission:
                raise RuntimeError("prefix admission changed after its fail-closed preview")
            base_revision = admission.base_revision
            features_published = False
            scheduler_admitted = False
            owner_published = False
            restored_pages: tuple[int, ...] = ()
            prefix_bound = False
            owned_prepared_input = prepared_input
            try:
                if prepared_input.feature_bundle is not None:
                    owned_bundle = self.features.publish(input_plan, prepared_input.feature_bundle)
                    owned_prepared_input = PreparedInput(input_plan, owned_bundle)
                    features_published = True
                provisions = () if snapshot is not None else self._prepare_request_state()
                self._verify_execution_identity("request initialization")
                self.scheduler.admit(
                    input_plan,
                    max_new_tokens=max_new_tokens,
                    execution_input_digest=owned_prepared_input.semantic_digest,
                    base_revision=base_revision,
                    prefill_cursor=admission.resume_token_count,
                )
                scheduler_admitted = True
                if snapshot is None:
                    self.store.provision_owner(input_plan.request_id, provisions)
                    owner_published = True
                else:
                    assert admission.prefix_key is not None
                    restore = self.store.restore_prefix(
                        snapshot,
                        expected_prefix_key=admission.prefix_key,
                        owner_id=input_plan.request_id,
                    )
                    owner_published = True
                    restored_pages = tuple(target for _, target in restore.source_to_restored_pages)
                    self.allocator.bind_restored_prefix(
                        request_id=input_plan.request_id,
                        restore=restore,
                    )
                    prefix_bound = True
            except BaseException:
                if prefix_bound:
                    self.allocator.rollback_restored_binding(
                        input_plan.request_id,
                        expected_pages=restored_pages,
                    )
                if owner_published:
                    self.store.rollback_owner_provision(
                        input_plan.request_id,
                        expected_revision=base_revision,
                        expected_page_ids=tuple(sorted(restored_pages)),
                    )
                if scheduler_admitted:
                    self.scheduler.rollback_admission(input_plan.request_id)
                if features_published:
                    self.features.release(input_plan.request_id)
                raise
            self._input_plans[input_plan.request_id] = input_plan
            self._prepared_inputs[input_plan.request_id] = owned_prepared_input
            self._event_sequence[input_plan.request_id] = 0
            self._prefix_admissions[input_plan.request_id] = admission
            self._prefix_enabled[input_plan.request_id] = use_prefix_cache
            if ignore_eos:
                self._ignore_eos.add(input_plan.request_id)
            if constraint is not None:
                self._constraints[input_plan.request_id] = constraint
            return admission

    def step(self) -> EngineStep | None:
        with self._lock:
            self._verify_execution_identity("step planning")
            batch_plan = self.scheduler.next_plan()
            if batch_plan is None:
                return None
            transactions: list[StateTransaction] = []
            results: list[ModelStepResult] = []
            sampled_tokens: list[int | None] = []
            logits_rows: tuple[EngineLogitsRow, ...] = ()
            completions: tuple[SegmentCompletion, ...] = ()
            prefix_checkpoints: dict[str, tuple[RecurrentStateSnapshot, ...]] = {}
            try:
                for segment, contract_txn in zip(
                    batch_plan.segments,
                    batch_plan.state_transactions,
                    strict=True,
                ):
                    publication = contract_txn.prefix_publish
                    if publication is not None and publication.replay_token_count:
                        if publication.state_commit_position != segment.token_start:
                            raise ValueError(
                                "prefix replay checkpoint must equal the segment start"
                            )
                        prefix_checkpoints[segment.request_id] = tuple(
                            self.store.checkpoint_recurrent_state(segment.request_id, state_id)
                            for state_id in self._state_ids
                        )
                    transaction = self._begin_transaction(segment, contract_txn)
                    transactions.append(transaction)
                with (
                    ReferenceExecutionContext(),
                    operator_strategy_scope(
                        self._operator_config, device=str(self.device), model=""
                    ),
                ):
                    for segment, transaction in zip(batch_plan.segments, transactions, strict=True):
                        prepared_input = self._prepared_inputs[segment.request_id]
                        input_plan = prepared_input.input_plan
                        feature_reader = self.features.reader_for_segment(
                            input_plan,
                            segment,
                        )
                        identity = ReferenceSegmentIdentity(
                            plan_id=batch_plan.plan_id,
                            topology_id=batch_plan.topology_id,
                            partition_id=batch_plan.partition_id,
                            segment=segment,
                        )
                        state_io = _TransactionStateIO(
                            transaction,
                            identity=identity,
                            logical_to_physical=self.allocator.page_table(
                                segment.request_id
                            ).logical_to_physical,
                            block_size_tokens=self.allocator.block_size_tokens,
                        )
                        op_capability = self._ops.bind_segment(identity)
                        try:
                            require_reference_execution()
                            result = self._program.execute_segment(
                                ModelSegment.from_plan(segment),
                                op_capability,
                                state_io,
                                feature_reader,
                            )
                            op_capability.verify_used()
                            require_reference_execution()
                        finally:
                            op_capability.revoke()
                            state_io.revoke()
                            feature_reader.revoke()
                        self._validate_result(segment, result)
                        admission = self._prefix_admissions[segment.request_id]
                        token_end = segment.token_start + segment.token_count
                        if (
                            admission.hit
                            and segment.token_start < admission.token_count
                            and token_end <= admission.token_count
                        ):
                            transaction.verify_kv_matches_committed()
                        results.append(result)
                    sampled_tokens = [
                        self._sample_if_ready(segment, result)
                        for segment, result in zip(batch_plan.segments, results, strict=True)
                    ]
                    logits_rows = tuple(
                        EngineLogitsRow(
                            request_id=segment.request_id,
                            plan_id=batch_plan.plan_id,
                            segment_index=segment_index,
                            row_index=segment.token_count - 1,
                            tensor=result.logits,
                        )
                        for segment_index, (segment, result) in enumerate(
                            zip(batch_plan.segments, results, strict=True)
                        )
                        if segment.produces_output
                    )
                    completions = tuple(
                        SegmentCompletion(
                            request_id=segment.request_id,
                            generated_token_id=token_id,
                            finished=(
                                token_id in self._eos_token_ids
                                if token_id is not None
                                and (
                                    segment.request_id not in self._ignore_eos
                                    or segment.request_id in self._constraints
                                )
                                else False
                            ),
                        )
                        for segment, token_id in zip(
                            batch_plan.segments, sampled_tokens, strict=True
                        )
                    )
                    for completion in completions:
                        self.scheduler.validate_completion(
                            completion.request_id,
                            generated_token_id=completion.generated_token_id,
                            finished=completion.finished,
                        )
                    self._finish_terminal_constraints(completions)
                self._verify_execution_identity("transaction pre-commit")
                for transaction in transactions:
                    transaction.verify()
                self.store.commit_bundle(tuple(transactions))
            except BaseException:
                for transaction in transactions:
                    if transaction.status not in ("committed", "aborted"):
                        transaction.abort()
                for segment in batch_plan.segments:
                    if self.scheduler.request_state(segment.request_id).inflight is not None:
                        self.scheduler.fail(segment.request_id)
                raise

            self.scheduler.complete_bundle(completions)
            events: list[TokenEvent] = []
            published: list[str] = []
            publications: list[PrefixPublication] = []
            publication_failures: list[PrefixPublicationFailure] = []
            for segment_index, (segment, contract_txn, token_id) in enumerate(
                zip(
                    batch_plan.segments,
                    batch_plan.state_transactions,
                    sampled_tokens,
                    strict=True,
                )
            ):
                if token_id is not None:
                    events.append(self._event(segment.request_id, TokenEventKind.TOKEN, token_id))
                    if self.scheduler.request_state(segment.request_id).finished:
                        events.append(self._event(segment.request_id, TokenEventKind.FINISH))
                if (
                    contract_txn.prefix_publish is not None
                    and self._prefix_enabled[segment.request_id]
                ):
                    publication = contract_txn.prefix_publish
                    try:
                        publication_evidence = self._publish_prefix(
                            self._prepared_inputs[segment.request_id],
                            plan_id=batch_plan.plan_id,
                            segment_index=segment_index,
                            token_count=publication.prefix_token_count,
                            state_commit_position=publication.state_commit_position,
                            commit_alignment=publication.commit_alignment_tokens,
                            recurrent_checkpoints=prefix_checkpoints.get(segment.request_id, ()),
                        )
                    except Exception as error:
                        publication_failures.append(
                            PrefixPublicationFailure(
                                request_id=segment.request_id,
                                prefix_token_count=publication.prefix_token_count,
                                error_type=type(error).__name__,
                                message=str(error) or "prefix publication failed without detail",
                            )
                        )
                        continue
                    if publication_evidence is not None:
                        publications.append(publication_evidence)
                        if publication_evidence.inserted:
                            published.append(publication_evidence.snapshot.prefix_key)
            return EngineStep(
                batch_plan=batch_plan,
                events=tuple(events),
                published_prefix_keys=tuple(published),
                logits_rows=logits_rows,
                prefix_publications=tuple(publications),
                prefix_publication_failures=tuple(publication_failures),
            )

    def run_until_drained(self, *, max_steps: int = 10_000) -> tuple[EngineStep, ...]:
        steps: list[EngineStep] = []
        for _ in range(max_steps):
            step = self.step()
            if step is None:
                if self.scheduler.drained:
                    return tuple(steps)
                raise RuntimeError("reference engine stalled with active unschedulable requests")
            steps.append(step)
        raise RuntimeError("reference engine exceeded its explicit step limit")

    def prefix_admission(self, request_id: str) -> PrefixAdmission:
        return self._prefix_admissions[request_id]

    @property
    def weight_evidence(self) -> WeightEvidence:
        """Return startup provenance after a cheap live-storage integrity check."""

        with self._lock:
            self._verify_execution_identity("weight evidence access")
            return self._sealed_weights.evidence

    def _verify_execution_identity(self, boundary: str) -> None:
        if self._operator_config.digest != self._operator_digest:
            raise RuntimeError("reference activation policy changed after admission")
        # Weight inventory and recipe providers are adapter callbacks too; keep
        # them inside the same eager policy boundary as model execution.
        with (
            ReferenceExecutionContext(),
            operator_strategy_scope(self._operator_config, device=str(self.device), model=""),
        ):
            self._sealed_weights.verify_unchanged(boundary=boundary)
            require_reference_execution()
            try:
                recipe = self._identity_program.execution_recipe()
            except Exception as error:
                raise WeightIntegrityError(
                    f"model program recipe cannot be verified at {boundary}"
                ) from error
            require_reference_execution()
        if recipe != self._program_recipe:
            raise WeightIntegrityError(f"model program recipe drifted at {boundary}")

    @property
    def execution_fingerprint(self) -> ReferenceExecutionFingerprint:
        """Derive replay identity from the admitted engine, never caller metadata."""

        self._verify_execution_identity("execution fingerprint")

        descriptor = self.loaded_adapter.descriptor
        contract_digest = _digest_payload(
            {
                "schema": "cubic.reference-contract.v1",
                "model_spec": repr(descriptor.model_spec),
                "weight_variant": repr(descriptor.weight_variant),
                "decode_extension": repr(descriptor.decode_extension),
                "required_ops": repr(descriptor.required_ops),
                "kv_layout": repr(self.store.layout),
            }
        )
        model_digest = _digest_payload(
            {
                "adapter": (descriptor.adapter_id, descriptor.adapter_revision),
                "adapter_source": self.loaded_adapter.source_digest,
                "model": (descriptor.model_spec.model_id, descriptor.model_spec.revision),
                "weight_manifest": repr(descriptor.model_spec.weights),
                "actual_weight_semantics": self._sealed_weights.evidence.semantic_digest,
                "program_type": type(self._identity_program).__qualname__,
                "program_recipe": self._program_recipe_digest,
            }
        )
        implementation_digest = self._implementation_digest
        topology_digest = _digest_payload(
            {
                "topology": self._topology_id,
                "partition": self._partition_id,
                "rank": self.store.rank_id,
                "device": str(self.device),
                "owned_layer_ids": self._owned_layer_ids,
                "kv_layout": repr(self.store.layout),
            }
        )
        evidence_digest = _digest_payload(
            {
                "mode": "reference",
                "math_policy": repr(self._reference_policy),
                "contract": contract_digest,
                "model": model_digest,
                "implementation": implementation_digest,
                "topology": topology_digest,
            }
        )
        return ReferenceExecutionFingerprint(
            executor_id="cubic.reference.engine.v1",
            contract_digest=contract_digest,
            model_digest=model_digest,
            implementation_digest=implementation_digest,
            topology_digest=topology_digest,
            evidence_digest=evidence_digest,
        )

    def terminal_digest(self, request_id: str) -> CommittedStateDigest:
        """Hash committed logical KV/state without exposing physical page identity."""

        with self._lock:
            state = self.scheduler.request_state(request_id)
            if state.inflight is not None or not (state.finished or state.cancelled):
                raise RuntimeError("terminal digest requires a settled request")
            table = self.allocator.page_table(request_id)
            cache_hash = hashlib.blake2s()
            for logical_page, physical_page in enumerate(table.logical_to_physical):
                cache_hash.update(f"logical-page:{logical_page}".encode())
                written = self.store.page_written_mask(request_id, physical_page)
                _hash_tensor(cache_hash, written)
                for layer_index, layer_id in enumerate(self.store.layout.layer_ids):
                    offsets = tuple(
                        int(index)
                        for index in torch.nonzero(written[layer_index], as_tuple=False)
                        .flatten()
                        .tolist()
                    )
                    if not offsets:
                        continue
                    key, value = self.store.read_kv(request_id, physical_page, layer_id, offsets)
                    cache_hash.update(f"layer:{layer_id}:offsets:{offsets}".encode())
                    _hash_tensor(cache_hash, key)
                    _hash_tensor(cache_hash, value)

            state_hash = hashlib.blake2s()
            for state_id in self._state_ids:
                checkpoint = self.store.checkpoint_recurrent_state(request_id, state_id)
                state_hash.update(
                    (
                        f"state:{state_id}:position:{checkpoint.position}:valid:{checkpoint.valid}"
                    ).encode()
                )
                _hash_tensor(state_hash, checkpoint.value)
            return CommittedStateDigest(
                request_id=request_id,
                owner_rank=0,
                state_revision=self.store.current_revision(request_id),
                cache_digest=f"b2s256:{cache_hash.hexdigest()}",
                state_digest=f"b2s256:{state_hash.hexdigest()}",
            )

    def prefix_key(self, prepared_input: PreparedInput, token_count: int) -> str:
        self._verify_execution_identity("prefix identity")
        self._validate_prepared_input(prepared_input)
        input_plan = prepared_input.input_plan
        if not 0 < token_count <= len(input_plan.token_ids):
            raise ValueError("prefix key token count must fall within the realized input")
        affected_parts = sorted(
            {
                placement.content_part_ordinal
                for placement in input_plan.placements
                if placement.token_start < token_count
            }
        )
        for placement in input_plan.placements:
            if placement.token_start < token_count < placement.token_start + placement.token_count:
                raise ValueError("a prefix boundary cannot split one multimodal placement")
        part_payload = []
        jobs = {job.content_part_ordinal: job for job in input_plan.encoder_jobs}
        for ordinal in affected_parts:
            part = input_plan.content_parts[ordinal]
            media = part.media
            part_payload.append(
                {
                    "ordinal": ordinal,
                    "modality": part.modality,
                    "media_digest": None if media is None else media.content_digest,
                    "encoder": jobs[ordinal].encoder_id,
                    "encoder_processor": jobs[ordinal].processor_id,
                    "encoder_key": jobs[ordinal].cache_key,
                }
            )
        descriptor = self.loaded_adapter.descriptor
        payload = {
            "adapter": [descriptor.adapter_id, descriptor.adapter_revision],
            "adapter_source": self.loaded_adapter.source_digest,
            "model": [descriptor.model_spec.model_id, descriptor.model_spec.revision],
            "weights": {
                "descriptor_manifest": descriptor.model_spec.weights.manifest_digest,
                "actual_semantics": self._sealed_weights.evidence.semantic_digest,
            },
            "variant": {
                "base_model_id": input_plan.weight_variant.base_model_id,
                "base_revision": input_plan.weight_variant.base_revision,
                "adapter_id": input_plan.weight_variant.adapter_id,
                "adapter_revision": input_plan.weight_variant.adapter_revision,
                "adapter_scale": input_plan.weight_variant.adapter_scale,
            },
            "processor_recipe": input_plan.processor_recipe_digest,
            "tokens": input_plan.token_ids[:token_count],
            "positions": tuple(position.axes for position in input_plan.positions[:token_count]),
            "placements": tuple(
                (
                    placement.content_part_ordinal,
                    placement.token_start,
                    placement.token_count,
                    placement.feature_row_start,
                    placement.feature_row_count,
                    placement.consumer_layers,
                    placement.consumer_site,
                    placement.fusion.value,
                )
                for placement in input_plan.placements
                if placement.token_start < token_count
            ),
            "parts": part_payload,
            "encoded_features": (
                None
                if prepared_input.feature_bundle is None or not affected_parts
                else prepared_input.feature_bundle.semantic_digest_for_parts(tuple(affected_parts))
            ),
            "cache": {
                "layout": descriptor.model_spec.cache.layout,
                "page_size": descriptor.model_spec.cache.page_size_tokens,
                "dtype": descriptor.model_spec.cache.kv_dtype.value,
                "runtime_axis_order": self.store.layout.axis_order,
                "runtime_layer_ids": self.store.layout.layer_ids,
                "runtime_num_kv_heads": self.store.layout.num_kv_heads,
                "runtime_head_dim": self.store.layout.head_dim,
            },
            "states": tuple(
                (
                    state.state_id,
                    state.schema_version,
                    state.dtype.value,
                    state.layout,
                    state.commit_alignment_tokens,
                    state.prefix_replay.value,
                    state.sharding.value,
                )
                for state in descriptor.model_spec.states
            ),
            "topology": self._topology_id,
            "partition": self._partition_id,
            "decode_extension": (
                descriptor.decode_extension.provider.value,
                descriptor.decode_extension.provider_version,
                descriptor.decode_extension.max_proposal_tokens,
                descriptor.decode_extension.state_schema_id,
            ),
            "execution": {
                "plan": self.loaded_adapter.reference_plan.execution_plan_id,
                "program_recipe": self._program_recipe_digest,
                "implementation": self._implementation_digest,
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return f"b2s256:{hashlib.blake2s(encoded).hexdigest()}"

    def _validate_input_plan(self, input_plan: InputPlan) -> None:
        descriptor = self.loaded_adapter.descriptor
        if input_plan.model_id != descriptor.model_spec.model_id:
            raise ValueError("InputPlan model identity differs from the loaded adapter")
        if input_plan.weight_variant != descriptor.weight_variant:
            raise ValueError("InputPlan weight variant differs from the loaded adapter")
        modality_by_name = {
            modality.name: modality for modality in descriptor.model_spec.modalities
        }
        if any(part.modality not in modality_by_name for part in input_plan.content_parts):
            raise ValueError("InputPlan uses a modality absent from ModelSpec")
        jobs = {job.content_part_ordinal: job for job in input_plan.encoder_jobs}
        placements: dict[int, list[TokenPlacement]] = {}
        for placement in input_plan.placements:
            placements.setdefault(placement.content_part_ordinal, []).append(placement)
        for part in input_plan.content_parts:
            modality = modality_by_name[part.modality]
            if part.media is None:
                continue
            if not modality.mime_prefixes or not any(
                part.media.mime_type.startswith(prefix) for prefix in modality.mime_prefixes
            ):
                raise ValueError("InputPlan media MIME type differs from ModelSpec")
            job = jobs[part.ordinal]
            if modality.encoder_id is None or job.encoder_id != modality.encoder_id:
                raise ValueError("InputPlan encoder identity differs from ModelSpec")
            if descriptor.model_spec.layer_count is not None:
                for placement in placements[part.ordinal]:
                    if any(
                        layer_id >= descriptor.model_spec.layer_count
                        for layer_id in placement.consumer_layers
                    ):
                        raise ValueError("InputPlan placement consumes a layer outside ModelSpec")

    def _validate_prepared_input(self, prepared_input: PreparedInput) -> None:
        self._validate_input_plan(prepared_input.input_plan)
        if prepared_input.feature_bundle is not None:
            prepared_input.feature_bundle.verify_for_input_plan(prepared_input.input_plan)
            part_by_ordinal = {
                part.ordinal: part for part in prepared_input.input_plan.content_parts
            }
            modality_by_name = {
                modality.name: modality
                for modality in self.loaded_adapter.descriptor.model_spec.modalities
            }
            checked_jobs: set[str] = set()
            for binding in prepared_input.feature_bundle.bindings:
                feature = binding.feature
                if feature.job_id in checked_jobs:
                    continue
                checked_jobs.add(feature.job_id)
                part = part_by_ordinal[feature.content_part_ordinal]
                expected = modality_by_name[part.modality].feature_hidden_size
                if expected is None or feature.hidden_size != expected:
                    raise ValueError("encoded feature hidden size differs from ModelSpec")

    def _prepare_request_state(self) -> tuple[RecurrentStateProvision, ...]:
        with (
            ReferenceExecutionContext(),
            operator_strategy_scope(self._operator_config, device=str(self.device), model=""),
        ):
            require_reference_execution()
            initializations = self._program.initialize_request()
            require_reference_execution()
        if tuple(sorted(item.state_id for item in initializations)) != self._state_ids:
            raise ValueError("model program initial state ids differ from ModelSpec")
        spec_by_id = {
            state.state_id: state for state in self.loaded_adapter.descriptor.model_spec.states
        }
        provisions: list[RecurrentStateProvision] = []
        for initialization in initializations:
            spec = spec_by_id[initialization.state_id]
            expected_dtype = _torch_dtype(spec.dtype)
            if initialization.value.device != self.device:
                raise ValueError("initial recurrent state device differs from the engine")
            if initialization.value.dtype is not expected_dtype:
                raise ValueError("initial recurrent state dtype differs from StateSpec")
            provisions.append(
                RecurrentStateProvision(
                    state_id=initialization.state_id,
                    value=initialization.value,
                    position=initialization.position,
                    valid=initialization.valid,
                )
            )
        return tuple(provisions)

    def _find_longest_prefix(
        self, prepared_input: PreparedInput
    ) -> tuple[PrefixAdmission, PrefixSnapshot | None]:
        input_plan = prepared_input.input_plan
        block = self.allocator.block_size_tokens
        largest = (len(input_plan.token_ids) // block) * block
        candidate_keys: list[tuple[int, str]] = []
        for token_count in range(largest, 0, -block):
            try:
                prefix_key = self.prefix_key(prepared_input, token_count)
            except ValueError:
                continue
            candidate_keys.append((token_count, prefix_key))
            lease = self.prefix_cache.acquire(prefix_key, source_rank=0)
            if lease is None:
                continue
            with lease:
                snapshot = lease.snapshot
            if (
                token_count == len(input_plan.token_ids)
                and snapshot.arithmetic_state_commit_boundary == token_count
            ):
                # A fully consumed prompt needs at least one replay row to
                # regenerate its logits; an equal-boundary snapshot has none.
                continue
            if (
                snapshot.physical_block_boundary != token_count
                or snapshot.state_commit_alignment != self._state_alignment
                or tuple(state.state_id for state in snapshot.recurrent_states) != self._state_ids
            ):
                raise RuntimeError("prefix state schema differs from the active engine")
            replay_start = snapshot.arithmetic_state_commit_boundary
            if snapshot.replay_tokens != input_plan.token_ids[replay_start:token_count]:
                raise RuntimeError("prefix replay tokens differ from the admitted InputPlan")
            if replay_start < token_count and any(
                state.prefix_replay is PrefixReplayPolicy.NONE
                for state in self.loaded_adapter.descriptor.model_spec.states
            ):
                raise RuntimeError("model state schema forbids prefix arithmetic replay")
            return (
                PrefixAdmission(
                    True,
                    token_count,
                    snapshot.arithmetic_state_commit_boundary,
                    prefix_key,
                    tuple(candidate_keys),
                    _prefix_snapshot_identity(snapshot),
                ),
                snapshot,
            )
        return PrefixAdmission(
            False,
            0,
            0,
            None,
            tuple(candidate_keys),
        ), None

    def _begin_transaction(self, segment: PlanSegment, contract_txn: StateTxn) -> StateTransaction:
        if contract_txn.request_id != segment.request_id:
            raise ValueError("segment and transaction request identities differ")
        if (
            segment.input_plan_digest
            != self._prepared_inputs[segment.request_id].input_plan.semantic_digest
        ):
            raise ValueError("segment input digest differs from the admitted InputPlan")
        if (
            segment.execution_input_digest
            != self._prepared_inputs[segment.request_id].semantic_digest
        ):
            raise ValueError(
                "segment execution input digest differs from the admitted PreparedInput"
            )
        if contract_txn.expected_owner_ranks != (0,):
            raise ValueError("single-rank reference engine requires owner rank zero only")
        table = self.allocator.page_table(segment.request_id)
        if table.released or segment.cache_pages != table.logical_to_physical:
            raise ValueError("BatchPlan cache context differs from the active page table")
        block = self.allocator.block_size_tokens
        token_end = segment.token_start + segment.token_count
        write_ordinals = tuple(range(segment.token_start, token_end))
        write_by_page = _ordinal_offsets(write_ordinals, table.logical_to_physical, block)
        context_ordinals = tuple(range(token_end))
        read_by_page = _ordinal_offsets(context_ordinals, table.logical_to_physical, block)
        expected_segment_slots = tuple(
            table.logical_to_physical[ordinal // block] * block + ordinal % block
            for ordinal in write_ordinals
        )
        if segment.cache_slots != expected_segment_slots:
            raise ValueError(
                "BatchPlan cache slots differ from the exact page-table ordinal mapping"
            )
        expected_access_slots = tuple(sorted(expected_segment_slots))
        expected_read_pages = tuple(sorted(page for page, _ in read_by_page))
        expected_write_pages = tuple(sorted(page for page, _ in write_by_page))
        read_accesses = {access.resource_id: access for access in contract_txn.reads}
        write_accesses = {access.resource_id: access for access in contract_txn.writes}
        kv_resource = f"kv:{segment.request_id}"
        if set(read_accesses) != {kv_resource, *self._state_resource_ids(segment.request_id)}:
            raise ValueError("transaction read resources differ from the model state schema")
        if set(write_accesses) != {kv_resource, *self._state_resource_ids(segment.request_id)}:
            raise ValueError("transaction write resources differ from the model state schema")
        if read_accesses[kv_resource].pages != expected_read_pages:
            raise ValueError("transaction KV read pages differ from the exact context")
        if (
            write_accesses[kv_resource].pages != expected_write_pages
            or write_accesses[kv_resource].slots != expected_access_slots
        ):
            raise ValueError("transaction KV write pages/slots differ from the segment")

        reads = tuple(
            self.store.kv_region(segment.request_id, page, layer, offsets)
            for page, offsets in read_by_page
            for layer in self.store.layout.layer_ids
        ) + tuple(
            self.store.recurrent_region(segment.request_id, state_id)
            for state_id in self._state_ids
        )
        writes = tuple(
            self.store.kv_region(segment.request_id, page, layer, offsets)
            for page, offsets in write_by_page
            for layer in self.store.layout.layer_ids
        ) + tuple(
            self.store.recurrent_region(segment.request_id, state_id)
            for state_id in self._state_ids
        )
        runtime_plan = TxnPlan(
            transaction_id=contract_txn.transaction_id,
            owner_id=segment.request_id,
            base_revision=contract_txn.base_revision,
            reads=reads,
            writes=writes,
            target_position=token_end if self._state_ids else None,
        )
        if contract_txn.target_revision != runtime_plan.base_revision + 1:
            raise ValueError("contract transaction target revision is not executable")
        return self.store.begin(runtime_plan)

    def _state_resource_ids(self, request_id: str) -> tuple[str, ...]:
        return tuple(f"state:{state_id}:{request_id}" for state_id in self._state_ids)

    @staticmethod
    def _validate_result(segment: PlanSegment, result: ModelStepResult) -> None:
        if not isinstance(result, ModelStepResult):
            raise TypeError("model program must return ModelStepResult")
        if not bool(torch.isfinite(result.logits).all().item()):
            raise ValueError("model returned non-finite logits")

    def _sample_if_ready(self, segment: PlanSegment, result: ModelStepResult) -> int | None:
        if not segment.produces_output:
            return None
        return self._sampler.sample(result.logits, self._constraints.get(segment.request_id))

    def _finish_terminal_constraints(self, completions: tuple[SegmentCompletion, ...]) -> None:
        for completion in completions:
            constraint = self._constraints.get(completion.request_id)
            if constraint is None:
                continue
            state = self.scheduler.request_state(completion.request_id)
            reached_limit = (
                completion.generated_token_id is not None
                and len(state.generated_token_ids) + 1 >= state.max_new_tokens
            )
            if completion.finished or reached_limit:
                constraint.finish(truncated=reached_limit and not completion.finished)

    def _event(
        self,
        request_id: str,
        kind: TokenEventKind,
        token_id: int | None = None,
    ) -> TokenEvent:
        sequence = self._event_sequence[request_id]
        self._event_sequence[request_id] += 1
        return TokenEvent(
            request_id=request_id,
            sequence_no=sequence,
            kind=kind,
            token_id=token_id,
        )

    def _publish_prefix(
        self,
        prepared_input: PreparedInput,
        *,
        plan_id: str,
        segment_index: int,
        token_count: int,
        state_commit_position: int,
        commit_alignment: int,
        recurrent_checkpoints: tuple[RecurrentStateSnapshot, ...],
    ) -> PrefixPublication | None:
        input_plan = prepared_input.input_plan
        try:
            prefix_key = self.prefix_key(prepared_input, token_count)
        except ValueError:
            return None
        table = self.allocator.page_table(input_plan.request_id)
        page_count = token_count // self.allocator.block_size_tokens
        snapshot: PrefixSnapshot = self.store.capture_prefix(
            snapshot_id=f"prefix:{prefix_key[7:23]}:{input_plan.request_id}",
            prefix_key=prefix_key,
            owner_id=input_plan.request_id,
            page_ids=table.logical_to_physical[:page_count],
            state_ids=self._state_ids,
            physical_block_boundary=token_count,
            arithmetic_state_commit_boundary=state_commit_position,
            state_commit_alignment=commit_alignment,
            replay_tokens=input_plan.token_ids[state_commit_position:token_count],
            recurrent_checkpoints=recurrent_checkpoints,
        )
        inserted = self.prefix_cache.publish(snapshot)
        return PrefixPublication(
            request_id=input_plan.request_id,
            plan_id=plan_id,
            segment_index=segment_index,
            snapshot=_prefix_snapshot_identity(snapshot),
            inserted=inserted,
        )


def _ordinal_offsets(
    ordinals: tuple[int, ...],
    logical_to_physical: tuple[int, ...],
    block_size: int,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    groups: list[tuple[int, list[int]]] = []
    for ordinal in ordinals:
        logical_page, offset = divmod(ordinal, block_size)
        try:
            physical_page = logical_to_physical[logical_page]
        except IndexError as error:
            raise ValueError("BatchPlan token ordinal has no physical cache page") from error
        if not groups or groups[-1][0] != physical_page:
            groups.append((physical_page, []))
        groups[-1][1].append(offset)
    return tuple((page, tuple(offsets)) for page, offsets in groups)


def _torch_dtype(dtype: ScalarDType) -> torch.dtype:
    mapping = {
        ScalarDType.BF16: torch.bfloat16,
        ScalarDType.FP16: torch.float16,
        ScalarDType.FP32: torch.float32,
        ScalarDType.INT64: torch.int64,
    }
    try:
        return mapping[dtype]
    except KeyError as error:
        raise ValueError(f"state dtype {dtype.value!r} has no v0.1 eager tensor mapping") from error


class _Digest(Protocol):
    def update(self, data: bytes, /) -> None: ...


def _hash_tensor(digest: _Digest, tensor: torch.Tensor) -> None:
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    payload = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    digest.update(payload)


def _digest_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"b2s256:{hashlib.blake2s(encoded).hexdigest()}"


def _program_recipe_digest(
    program: ModelProgram,
    recipe: ProgramExecutionRecipe,
    *,
    adapter_source_digest: str,
) -> str:
    return _digest_payload(
        {
            "schema": "cubic.model-program.recipe.v1",
            "adapter_source": adapter_source_digest,
            "program_class": (
                type(program).__module__,
                type(program).__qualname__,
                _callable_digest(type(program)),
            ),
            "methods": tuple(
                (name, _callable_digest(getattr(program, name)))
                for name in (
                    "execution_recipe",
                    "execute_segment",
                    "initialize_request",
                    "named_weight_tensors",
                )
            ),
            "recipe": {
                "id": recipe.recipe_id,
                "revision": recipe.revision,
                "settings": recipe.settings,
                "artifact_digests": recipe.artifact_digests,
            },
        }
    )


def _reference_implementation_digest(loaded_adapter: LoadedModelAdapter) -> str:
    implementations = []
    for binding in loaded_adapter.reference_plan.bindings:
        implementation = loaded_adapter.reference_catalog.resolve(binding.signature.semantic_op)
        implementations.append(
            (
                repr(binding.signature),
                implementation.semantic_op,
                implementation.implementation_id,
                _callable_digest(implementation.function),
            )
        )
    return _digest_payload(
        {
            "schema": "cubic.reference-implementations.v1",
            "execution_plan": repr(loaded_adapter.reference_plan),
            "implementations": tuple(implementations),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
    )


def _callable_digest(function: Callable[..., object]) -> str:
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        source = getattr(function, "__qualname__", type(function).__qualname__)
    return f"b2s256:{hashlib.blake2s(source.encode()).hexdigest()}"
