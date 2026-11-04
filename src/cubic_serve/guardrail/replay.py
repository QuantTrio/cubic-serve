"""Canonical execution traces and deterministic correctness replay.

The trace is an evidence artifact, not an executable snapshot.  Recording and
replay both create an executor through a caller-owned zero-argument factory.
The executor receives only reconstructed input and batch plans; replay never
passes the recorded events or terminal digests to it.
"""

from __future__ import annotations

import hashlib
import json
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Protocol, get_args, get_origin, get_type_hints
from typing_extensions import TypeAliasType

import torch
from torch import Tensor

from cubic_serve.contracts.execution import BatchPhase, BatchPlan, ExecutionMode, StateTxn
from cubic_serve.contracts.request import InputPlan, TokenEvent
from cubic_serve.guardrail.compare import tensor_digest

TRACE_SCHEMA = "cubic.guardrail.replay"
TRACE_SCHEMA_VERSION = 5

_LOGITS_EVIDENCE_TOKEN = object()
_LOGITS_DTYPES = frozenset(
    {
        "torch.bfloat16",
        "torch.float16",
        "torch.float32",
        "torch.float64",
    }
)


def _require_content_digest(value: str, field: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{field} must be a string")
    algorithm, separator, digest = value.partition(":")
    if algorithm != "b2s256" or separator != ":" or len(digest) != 64:
        raise ValueError(f"{field} must use b2s256:<64 hex chars>")
    try:
        int(digest, 16)
    except ValueError as error:
        raise ValueError(f"{field} contains non-hexadecimal characters") from error


@dataclass(frozen=True, slots=True)
class ReplayExecutionEvidence:
    """Identity of the exact execution whose observation is being bound."""

    executor_id: str
    mode: ExecutionMode
    contract_digest: str
    model_digest: str
    implementation_digest: str
    topology_digest: str
    evidence_digest: str

    def __post_init__(self) -> None:
        if type(self.executor_id) is not str or not self.executor_id.strip():
            raise ValueError("executor_id must be a non-empty string")
        if not isinstance(self.mode, ExecutionMode):
            raise TypeError("mode must be an ExecutionMode")
        for field_name in (
            "contract_digest",
            "model_digest",
            "implementation_digest",
            "topology_digest",
            "evidence_digest",
        ):
            _require_content_digest(getattr(self, field_name), field_name)

    @property
    def digest(self) -> str:
        return _digest_json(_encode(self))


@dataclass(frozen=True, slots=True)
class PrefixCandidateEvidence:
    token_count: int
    prefix_key: str

    def __post_init__(self) -> None:
        if type(self.token_count) is not int or self.token_count <= 0:
            raise ValueError("prefix candidate token count must be positive")
        _require_content_digest(self.prefix_key, "prefix candidate key")


@dataclass(frozen=True, slots=True)
class PrefixAdmissionEvidence:
    """Exact request admission decision at one committed-plan boundary."""

    request_id: str
    after_plan_count: int
    use_prefix_cache: bool
    execution_fingerprint_digest: str
    candidates: tuple[PrefixCandidateEvidence, ...]
    hit: bool
    prefix_key: str | None
    physical_block_boundary: int
    arithmetic_state_commit_boundary: int
    state_commit_alignment: int
    base_revision: int
    source_request_id: str | None
    source_rank: int | None
    source_revision: int | None
    snapshot_semantic_digest: str | None

    def __post_init__(self) -> None:
        if type(self.request_id) is not str or not self.request_id:
            raise ValueError("prefix admission requires a request id")
        if type(self.after_plan_count) is not int or self.after_plan_count < 0:
            raise ValueError("prefix admission plan boundary must be non-negative")
        if type(self.use_prefix_cache) is not bool or type(self.hit) is not bool:
            raise TypeError("prefix admission cache controls must be bools")
        _require_content_digest(
            self.execution_fingerprint_digest,
            "prefix admission execution fingerprint",
        )
        if type(self.candidates) is not tuple or any(
            not isinstance(candidate, PrefixCandidateEvidence) for candidate in self.candidates
        ):
            raise TypeError("prefix admission candidates must be sealed evidence")
        boundaries = tuple(candidate.token_count for candidate in self.candidates)
        if boundaries != tuple(sorted(set(boundaries), reverse=True)):
            raise ValueError("prefix admission candidate boundaries must be unique and descending")
        if not self.use_prefix_cache and self.candidates:
            raise ValueError("disabled prefix cache cannot carry lookup candidates")
        if (
            min(
                self.physical_block_boundary,
                self.arithmetic_state_commit_boundary,
                self.state_commit_alignment,
                self.base_revision,
            )
            < 0
        ):
            raise ValueError("prefix admission boundaries/revision must be non-negative")

        selected = (
            self.prefix_key,
            self.source_request_id,
            self.source_rank,
            self.source_revision,
            self.snapshot_semantic_digest,
        )
        if not self.hit:
            if any(value is not None for value in selected):
                raise ValueError("prefix miss cannot claim selected snapshot identity")
            if any(
                (
                    self.physical_block_boundary,
                    self.arithmetic_state_commit_boundary,
                    self.state_commit_alignment,
                    self.base_revision,
                )
            ):
                raise ValueError("prefix miss cannot claim restored boundaries/revision")
            return

        if not self.use_prefix_cache or any(value is None for value in selected):
            raise ValueError("prefix hit requires cache use and complete snapshot identity")
        assert self.prefix_key is not None
        assert self.source_rank is not None
        assert self.source_revision is not None
        assert self.snapshot_semantic_digest is not None
        _require_content_digest(self.prefix_key, "admitted prefix key")
        _require_content_digest(self.snapshot_semantic_digest, "admitted snapshot semantic digest")
        if self.source_rank < 0 or self.source_revision < 0:
            raise ValueError("prefix snapshot rank/revision must be non-negative")
        if self.physical_block_boundary <= 0 or self.state_commit_alignment <= 0:
            raise ValueError("prefix hit boundaries/alignment must be positive")
        if not (0 <= self.arithmetic_state_commit_boundary <= self.physical_block_boundary):
            raise ValueError("prefix arithmetic boundary must lie within physical KV")
        if self.arithmetic_state_commit_boundary % self.state_commit_alignment:
            raise ValueError("prefix arithmetic boundary violates state alignment")
        if self.base_revision != self.source_revision:
            raise ValueError("prefix admission base revision must equal snapshot source revision")
        if not any(
            candidate.token_count == self.physical_block_boundary
            and candidate.prefix_key == self.prefix_key
            for candidate in self.candidates
        ):
            raise ValueError("selected prefix is absent from admission candidates")


@dataclass(frozen=True, slots=True)
class PrefixPublicationEvidence:
    """Stable identity of one prefix snapshot published after a committed segment."""

    request_id: str
    plan_id: str
    segment_index: int
    after_plan_count: int
    execution_fingerprint_digest: str
    prefix_key: str
    source_rank: int
    source_revision: int
    physical_block_boundary: int
    arithmetic_state_commit_boundary: int
    state_commit_alignment: int
    snapshot_semantic_digest: str
    inserted: bool

    def __post_init__(self) -> None:
        if not self.request_id or not self.plan_id or self.segment_index < 0:
            raise ValueError("prefix publication requires request/plan/segment identity")
        if self.after_plan_count <= 0:
            raise ValueError("prefix publication requires a positive committed-plan boundary")
        if min(self.source_rank, self.source_revision) < 0:
            raise ValueError("prefix publication rank/revision must be non-negative")
        if self.physical_block_boundary <= 0 or self.state_commit_alignment <= 0:
            raise ValueError("prefix publication boundaries/alignment must be positive")
        if not (0 <= self.arithmetic_state_commit_boundary <= self.physical_block_boundary):
            raise ValueError("prefix publication arithmetic boundary lies outside physical KV")
        if self.arithmetic_state_commit_boundary % self.state_commit_alignment:
            raise ValueError("prefix publication arithmetic boundary violates alignment")
        if type(self.inserted) is not bool:
            raise TypeError("prefix publication inserted must be a bool")
        _require_content_digest(
            self.execution_fingerprint_digest,
            "prefix publication execution fingerprint",
        )
        _require_content_digest(self.prefix_key, "published prefix key")
        _require_content_digest(
            self.snapshot_semantic_digest,
            "published snapshot semantic digest",
        )


@dataclass(frozen=True, slots=True)
class PrefixLifecycleEvidence:
    admissions: tuple[PrefixAdmissionEvidence, ...]
    publications: tuple[PrefixPublicationEvidence, ...] = ()

    def __post_init__(self) -> None:
        if type(self.admissions) is not tuple or any(
            not isinstance(item, PrefixAdmissionEvidence) for item in self.admissions
        ):
            raise TypeError("prefix lifecycle admissions must be evidence values")
        if type(self.publications) is not tuple or any(
            not isinstance(item, PrefixPublicationEvidence) for item in self.publications
        ):
            raise TypeError("prefix lifecycle publications must be evidence values")
        admission_ids = tuple(item.request_id for item in self.admissions)
        if len(set(admission_ids)) != len(admission_ids):
            raise ValueError("prefix lifecycle permits one admission per request")
        publication_ids = tuple((item.plan_id, item.segment_index) for item in self.publications)
        if len(set(publication_ids)) != len(publication_ids):
            raise ValueError("prefix lifecycle permits one publication per segment")


@dataclass(frozen=True, slots=True)
class TerminalDigest:
    """Committed cache and non-KV state evidence for one request/rank."""

    request_id: str
    owner_rank: int
    state_revision: int
    cache_digest: str
    state_digest: str

    def __post_init__(self) -> None:
        if type(self.request_id) is not str or not self.request_id:
            raise ValueError("terminal digest requires a request_id")
        if type(self.owner_rank) is not int or self.owner_rank < 0:
            raise ValueError("terminal digest owner_rank must be a non-negative integer")
        if type(self.state_revision) is not int or self.state_revision < 0:
            raise ValueError("terminal digest revision must be a non-negative integer")
        _require_content_digest(self.cache_digest, "cache_digest")
        _require_content_digest(self.state_digest, "state_digest")

    @property
    def key(self) -> tuple[str, int]:
        return (self.request_id, self.owner_rank)


@dataclass(frozen=True, slots=True, init=False)
class LogitsEvidence:
    """Exact tensor evidence for one logical logits row.

    Construction is deliberately sealed. Callers must pass the real row tensor
    to :func:`capture_logits_evidence`; there is no public API accepting a
    claimed digest. Deserialization is the only other issuer and is protected by
    the enclosing trace's canonical payload digest.
    """

    request_id: str
    plan_id: str
    segment_index: int
    row_index: int
    dtype: str
    shape: tuple[int, ...]
    bit_digest: str

    def __init__(
        self,
        *,
        request_id: str,
        plan_id: str,
        segment_index: int,
        row_index: int,
        dtype: str,
        shape: tuple[int, ...],
        bit_digest: str,
        _issuer: object | None = None,
    ) -> None:
        if _issuer is not _LOGITS_EVIDENCE_TOKEN:
            raise TypeError("LogitsEvidence must be issued from a real tensor")
        if type(request_id) is not str or not request_id:
            raise ValueError("logits evidence requires a request_id")
        if type(plan_id) is not str or not plan_id:
            raise ValueError("logits evidence requires a plan_id")
        if type(segment_index) is not int or segment_index < 0:
            raise ValueError("logits evidence segment_index must be non-negative")
        if type(row_index) is not int or row_index < 0:
            raise ValueError("logits evidence row_index must be non-negative")
        if type(dtype) is not str or dtype not in _LOGITS_DTYPES:
            raise ValueError(f"unsupported logits evidence dtype: {dtype!r}")
        if (
            type(shape) is not tuple
            or len(shape) != 1
            or type(shape[0]) is not int
            or shape[0] <= 0
        ):
            raise ValueError("logits evidence must describe one non-empty rank-1 row")
        _require_content_digest(bit_digest, "bit_digest")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "plan_id", plan_id)
        object.__setattr__(self, "segment_index", segment_index)
        object.__setattr__(self, "row_index", row_index)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "bit_digest", bit_digest)

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (self.request_id, self.plan_id, self.segment_index, self.row_index)


def capture_logits_evidence(
    *,
    request_id: str,
    plan_id: str,
    segment_index: int,
    row_index: int,
    logits: Tensor,
) -> LogitsEvidence:
    """Seal one real rank-1 logits tensor using its canonical exact bits."""

    if not isinstance(logits, Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if logits.layout is not torch.strided:
        raise ValueError("logits evidence requires a strided tensor")
    if logits.ndim != 1 or logits.numel() == 0:
        raise ValueError("logits evidence requires one non-empty rank-1 row")
    dtype = str(logits.dtype)
    if dtype not in _LOGITS_DTYPES:
        raise ValueError(f"unsupported logits evidence dtype: {dtype!r}")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("logits evidence rejects non-finite tensors")
    return _issue_logits_evidence(
        request_id=request_id,
        plan_id=plan_id,
        segment_index=segment_index,
        row_index=row_index,
        dtype=dtype,
        shape=tuple(logits.shape),
        bit_digest=tensor_digest(logits),
    )


def _issue_logits_evidence(
    *,
    request_id: str,
    plan_id: str,
    segment_index: int,
    row_index: int,
    dtype: str,
    shape: tuple[int, ...],
    bit_digest: str,
) -> LogitsEvidence:
    return LogitsEvidence(
        request_id=request_id,
        plan_id=plan_id,
        segment_index=segment_index,
        row_index=row_index,
        dtype=dtype,
        shape=shape,
        bit_digest=bit_digest,
        _issuer=_LOGITS_EVIDENCE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class ReplayObservation:
    """Outputs read back from one completed executor run."""

    events: tuple[TokenEvent, ...]
    terminal_digests: tuple[TerminalDigest, ...]
    logits_evidence: tuple[LogitsEvidence, ...] = ()
    prefix_lifecycle: PrefixLifecycleEvidence = PrefixLifecycleEvidence(())

    def __post_init__(self) -> None:
        if type(self.events) is not tuple or any(
            not isinstance(event, TokenEvent) for event in self.events
        ):
            raise TypeError("events must be a tuple of TokenEvent values")
        if type(self.terminal_digests) is not tuple or any(
            not isinstance(digest, TerminalDigest) for digest in self.terminal_digests
        ):
            raise TypeError("terminal_digests must be a tuple of TerminalDigest values")
        if type(self.logits_evidence) is not tuple or any(
            not isinstance(evidence, LogitsEvidence) for evidence in self.logits_evidence
        ):
            raise TypeError("logits_evidence must be a tuple of sealed LogitsEvidence values")
        logits_keys = tuple(evidence.key for evidence in self.logits_evidence)
        if len(set(logits_keys)) != len(logits_keys):
            raise ValueError("logits evidence keys must be unique")
        if not isinstance(self.prefix_lifecycle, PrefixLifecycleEvidence):
            raise TypeError("prefix_lifecycle must be PrefixLifecycleEvidence")


class ReplayExecutor(Protocol):
    """Fresh, synchronous executor used to record or replay one trace."""

    @property
    def execution_evidence(self) -> ReplayExecutionEvidence: ...

    def execute(
        self,
        input_plans: tuple[InputPlan, ...],
        batch_plans: tuple[BatchPlan, ...],
        expected_prefix_lifecycle: PrefixLifecycleEvidence | None = None,
    ) -> ReplayObservation: ...


ExecutorFactory = Callable[[], ReplayExecutor]


@dataclass(frozen=True, slots=True)
class ReplayTrace:
    """Immutable baseline observation bound to exact inputs and batch plans.

    Schema v5 carries allocator-independent prefix admission, restore, and
    publication identity. Physical page ids remain execution-local BatchPlan
    details and are never used as prefix snapshot identity.
    """

    execution: ReplayExecutionEvidence
    input_plans: tuple[InputPlan, ...]
    batch_plans: tuple[BatchPlan, ...]
    recorded_events: tuple[TokenEvent, ...]
    recorded_terminal_digests: tuple[TerminalDigest, ...]
    recorded_logits_evidence: tuple[LogitsEvidence, ...] = ()
    recorded_prefix_lifecycle: PrefixLifecycleEvidence = PrefixLifecycleEvidence(())

    def __post_init__(self) -> None:
        if not isinstance(self.execution, ReplayExecutionEvidence):
            raise TypeError("execution must be ReplayExecutionEvidence")
        if type(self.input_plans) is not tuple or any(
            not isinstance(plan, InputPlan) for plan in self.input_plans
        ):
            raise TypeError("input_plans must be a tuple of InputPlan values")
        if type(self.batch_plans) is not tuple or any(
            not isinstance(plan, BatchPlan) for plan in self.batch_plans
        ):
            raise TypeError("batch_plans must be a tuple of BatchPlan values")
        _validate_contract_binding(
            self.input_plans,
            self.batch_plans,
            ReplayObservation(
                self.recorded_events,
                self.recorded_terminal_digests,
                self.recorded_logits_evidence,
                self.recorded_prefix_lifecycle,
            ),
            self.execution,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": TRACE_SCHEMA,
            "schema_version": TRACE_SCHEMA_VERSION,
            "execution": _encode(self.execution),
            "input_plans": _encode(self.input_plans),
            "batch_plans": _encode(self.batch_plans),
            "recorded_events": _encode(self.recorded_events),
            "recorded_terminal_digests": _encode(self.recorded_terminal_digests),
            "recorded_logits_evidence": _encode(self.recorded_logits_evidence),
            "recorded_prefix_lifecycle": _encode(self.recorded_prefix_lifecycle),
        }

    @property
    def trace_id(self) -> str:
        return _digest_json(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "trace_id": self.trace_id}

    def to_json(self) -> str:
        """Return the one accepted canonical JSON representation."""

        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, encoded: str) -> ReplayTrace:
        """Decode only canonical, exact-schema, digest-valid trace JSON."""

        if type(encoded) is not str:
            raise TypeError("trace JSON must be a string")
        raw = _strict_json_loads(encoded)
        if type(raw) is not dict:
            raise ValueError("trace root must be a JSON object")
        if encoded != _canonical_json(raw):
            raise ValueError("trace JSON is not in canonical form")
        _require_exact_keys(
            raw,
            {
                "schema",
                "schema_version",
                "trace_id",
                "execution",
                "input_plans",
                "batch_plans",
                "recorded_events",
                "recorded_terminal_digests",
                "recorded_logits_evidence",
                "recorded_prefix_lifecycle",
            },
            "trace",
        )
        if raw["schema"] != TRACE_SCHEMA:
            raise ValueError("unsupported trace schema")
        if type(raw["schema_version"]) is not int or (
            raw["schema_version"] != TRACE_SCHEMA_VERSION
        ):
            raise ValueError("unsupported trace schema version")
        _require_content_digest(raw["trace_id"], "trace_id")

        trace = cls(
            execution=_decode(ReplayExecutionEvidence, raw["execution"]),
            input_plans=tuple(
                _decode(InputPlan, item)
                for item in _require_list(raw["input_plans"], "input_plans")
            ),
            batch_plans=tuple(
                _decode(BatchPlan, item)
                for item in _require_list(raw["batch_plans"], "batch_plans")
            ),
            recorded_events=tuple(
                _decode(TokenEvent, item)
                for item in _require_list(raw["recorded_events"], "recorded_events")
            ),
            recorded_terminal_digests=tuple(
                _decode(TerminalDigest, item)
                for item in _require_list(
                    raw["recorded_terminal_digests"],
                    "recorded_terminal_digests",
                )
            ),
            recorded_logits_evidence=tuple(
                _decode(LogitsEvidence, item)
                for item in _require_list(
                    raw["recorded_logits_evidence"],
                    "recorded_logits_evidence",
                )
            ),
            recorded_prefix_lifecycle=_decode(
                PrefixLifecycleEvidence,
                raw["recorded_prefix_lifecycle"],
            ),
        )
        if raw["trace_id"] != trace.trace_id:
            raise ValueError("trace_id does not match the canonical trace payload")
        return trace


@dataclass(frozen=True, slots=True)
class ReplayComparison:
    """Exact replay verdict, including both execution identities."""

    trace_id: str
    recorded_execution: ReplayExecutionEvidence
    replay_execution: ReplayExecutionEvidence
    execution_evidence_match: bool
    event_sequence_match: bool
    terminal_digest_match: bool
    prefix_lifecycle_match: bool
    logits_evidence_complete: bool
    logits_evidence_match: bool
    first_event_mismatch_index: int | None
    terminal_mismatch_keys: tuple[tuple[str, int], ...]
    first_logits_mismatch_key: tuple[str, str, int, int] | None
    replay_observation_digest: str

    def __post_init__(self) -> None:
        _require_content_digest(self.trace_id, "trace_id")
        _require_content_digest(self.replay_observation_digest, "replay_observation_digest")
        if not isinstance(self.recorded_execution, ReplayExecutionEvidence) or not isinstance(
            self.replay_execution, ReplayExecutionEvidence
        ):
            raise TypeError("recorded and replay execution evidence must be sealed")
        for field_name in (
            "execution_evidence_match",
            "event_sequence_match",
            "terminal_digest_match",
            "prefix_lifecycle_match",
            "logits_evidence_complete",
            "logits_evidence_match",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be a bool")
        if self.execution_evidence_match != (self.recorded_execution == self.replay_execution):
            raise ValueError("execution evidence match disagrees with exact evidence equality")
        if self.first_event_mismatch_index is not None and (
            type(self.first_event_mismatch_index) is not int or self.first_event_mismatch_index < 0
        ):
            raise ValueError("first event mismatch index must be non-negative")
        if self.event_sequence_match != (self.first_event_mismatch_index is None):
            raise ValueError("event match and first mismatch index disagree")
        if type(self.terminal_mismatch_keys) is not tuple or any(
            type(key) is not tuple
            or len(key) != 2
            or type(key[0]) is not str
            or not key[0]
            or type(key[1]) is not int
            or key[1] < 0
            for key in self.terminal_mismatch_keys
        ):
            raise ValueError("terminal mismatch keys must contain request/rank identities")
        if self.terminal_digest_match != (not self.terminal_mismatch_keys):
            raise ValueError("terminal match and mismatch keys disagree")
        if self.logits_evidence_match and not self.logits_evidence_complete:
            raise ValueError("incomplete logits evidence cannot match")
        if self.first_logits_mismatch_key is not None:
            key = self.first_logits_mismatch_key
            if (
                type(key) is not tuple
                or len(key) != 4
                or type(key[0]) is not str
                or not key[0]
                or type(key[1]) is not str
                or not key[1]
                or type(key[2]) is not int
                or key[2] < 0
                or type(key[3]) is not int
                or key[3] < 0
            ):
                raise ValueError("logits mismatch key must identify request/plan/segment/row")
        if (self.logits_evidence_complete and self.logits_evidence_match) != (
            self.first_logits_mismatch_key is None
        ):
            raise ValueError("logits verdict and first mismatch key disagree")

    @property
    def passed(self) -> bool:
        return (
            self.execution_evidence_match
            and self.event_sequence_match
            and self.terminal_digest_match
            and self.prefix_lifecycle_match
            and self.logits_evidence_complete
            and self.logits_evidence_match
        )


def record_trace(
    *,
    input_plans: tuple[InputPlan, ...],
    batch_plans: tuple[BatchPlan, ...],
    executor_factory: ExecutorFactory,
) -> ReplayTrace:
    """Execute once and record the resulting observation as baseline evidence.

    There is deliberately no ``expected`` argument.  The recorded events and
    terminal digests and logits evidence must be read back from the executor
    created by the factory.
    """

    cloned_inputs, cloned_batches = _clone_plans(input_plans, batch_plans)
    # A malformed claimed execution must fail before model code can observe it.
    # Observation-dependent checks run when ReplayTrace is constructed below.
    _validate_plan_binding(cloned_inputs, cloned_batches)
    execution, observation = _run_fresh_executor(
        executor_factory,
        cloned_inputs,
        cloned_batches,
    )
    return ReplayTrace(
        execution=execution,
        input_plans=input_plans,
        batch_plans=batch_plans,
        recorded_events=observation.events,
        recorded_terminal_digests=observation.terminal_digests,
        recorded_logits_evidence=observation.logits_evidence,
        recorded_prefix_lifecycle=observation.prefix_lifecycle,
    )


def replay_trace(
    trace: ReplayTrace,
    *,
    executor_factory: ExecutorFactory,
) -> ReplayComparison:
    """Replay plans and exactly compare execution, logits, events, and terminal state."""

    if not isinstance(trace, ReplayTrace):
        raise TypeError("trace must be a ReplayTrace")

    # The strict round trip prevents accidental object or mutable executor-state
    # sharing between the recorded observation and the replay invocation.
    sealed_trace = ReplayTrace.from_json(trace.to_json())
    execution, observation = _run_fresh_executor(
        executor_factory,
        sealed_trace.input_plans,
        sealed_trace.batch_plans,
        expected_prefix_lifecycle=sealed_trace.recorded_prefix_lifecycle,
    )
    _validate_contract_binding(
        sealed_trace.input_plans,
        sealed_trace.batch_plans,
        observation,
        execution,
    )

    event_mismatch = _first_mismatch(
        sealed_trace.recorded_events,
        observation.events,
    )
    expected_terminals = {digest.key: digest for digest in sealed_trace.recorded_terminal_digests}
    actual_terminals = {digest.key: digest for digest in observation.terminal_digests}
    terminal_mismatches = tuple(
        sorted(
            key
            for key in expected_terminals.keys() | actual_terminals.keys()
            if expected_terminals.get(key) != actual_terminals.get(key)
        )
    )
    required_logits_keys = _expected_logits_keys(
        sealed_trace.input_plans,
        sealed_trace.batch_plans,
    )
    recorded_logits = {evidence.key: evidence for evidence in sealed_trace.recorded_logits_evidence}
    replay_logits = {evidence.key: evidence for evidence in observation.logits_evidence}
    logits_complete = (
        tuple(recorded_logits) == required_logits_keys
        and tuple(replay_logits) == required_logits_keys
    )
    first_logits_mismatch = _first_logits_mismatch(
        required_logits_keys,
        recorded_logits,
        replay_logits,
    )
    logits_match = logits_complete and first_logits_mismatch is None
    prefix_lifecycle_match = sealed_trace.recorded_prefix_lifecycle == observation.prefix_lifecycle
    observation_digest = _digest_json(
        {
            "execution": _encode(execution),
            "events": _encode(observation.events),
            "terminal_digests": _encode(observation.terminal_digests),
            "logits_evidence": _encode(observation.logits_evidence),
            "prefix_lifecycle": _encode(observation.prefix_lifecycle),
        }
    )
    return ReplayComparison(
        trace_id=sealed_trace.trace_id,
        recorded_execution=sealed_trace.execution,
        replay_execution=execution,
        execution_evidence_match=sealed_trace.execution == execution,
        event_sequence_match=event_mismatch is None,
        terminal_digest_match=not terminal_mismatches,
        prefix_lifecycle_match=prefix_lifecycle_match,
        logits_evidence_complete=logits_complete,
        logits_evidence_match=logits_match,
        first_event_mismatch_index=event_mismatch,
        terminal_mismatch_keys=terminal_mismatches,
        first_logits_mismatch_key=first_logits_mismatch,
        replay_observation_digest=observation_digest,
    )


def _run_fresh_executor(
    executor_factory: ExecutorFactory,
    input_plans: tuple[InputPlan, ...],
    batch_plans: tuple[BatchPlan, ...],
    *,
    expected_prefix_lifecycle: PrefixLifecycleEvidence | None = None,
) -> tuple[ReplayExecutionEvidence, ReplayObservation]:
    if not callable(executor_factory):
        raise TypeError("executor_factory must be callable")
    executor = executor_factory()
    if executor is None or not callable(getattr(executor, "execute", None)):
        raise TypeError("executor_factory must return a synchronous ReplayExecutor")
    observation = executor.execute(
        input_plans,
        batch_plans,
        expected_prefix_lifecycle,
    )
    if not isinstance(observation, ReplayObservation):
        raise TypeError("executor must return ReplayObservation")
    execution = getattr(executor, "execution_evidence", None)
    if not isinstance(execution, ReplayExecutionEvidence):
        raise TypeError("executor must expose ReplayExecutionEvidence after execution")
    return execution, observation


def _clone_plans(
    input_plans: tuple[InputPlan, ...],
    batch_plans: tuple[BatchPlan, ...],
) -> tuple[tuple[InputPlan, ...], tuple[BatchPlan, ...]]:
    if type(input_plans) is not tuple or type(batch_plans) is not tuple:
        raise TypeError("input_plans and batch_plans must be tuples")
    return (
        tuple(_decode(InputPlan, _encode(plan)) for plan in input_plans),
        tuple(_decode(BatchPlan, _encode(plan)) for plan in batch_plans),
    )


def _validate_contract_binding(
    input_plans: tuple[InputPlan, ...],
    batch_plans: tuple[BatchPlan, ...],
    observation: ReplayObservation,
    execution: ReplayExecutionEvidence,
) -> None:
    request_ids, known_requests, final_transactions = _validate_plan_binding(
        input_plans,
        batch_plans,
    )
    _validate_prefix_lifecycle(
        input_plans,
        batch_plans,
        observation.prefix_lifecycle,
        execution_fingerprint_digest=execution.digest,
    )

    event_requests = {event.request_id for event in observation.events}
    if not event_requests <= known_requests:
        raise ValueError("observation event references an unknown request")
    for request_id in event_requests:
        sequence = tuple(
            event.sequence_no for event in observation.events if event.request_id == request_id
        )
        if sequence != tuple(range(len(sequence))):
            raise ValueError("event sequence numbers must be contiguous per request")

    terminal_keys = tuple(digest.key for digest in observation.terminal_digests)
    if terminal_keys != tuple(sorted(terminal_keys)) or len(set(terminal_keys)) != len(
        terminal_keys
    ):
        raise ValueError("terminal digests must be unique and sorted by request/rank")
    terminal_requests = {digest.request_id for digest in observation.terminal_digests}
    if terminal_requests != known_requests:
        raise ValueError("terminal digests must cover every and only traced request")
    terminal_by_request: dict[str, list[TerminalDigest]] = {
        request_id: [] for request_id in request_ids
    }
    for terminal in observation.terminal_digests:
        terminal_by_request[terminal.request_id].append(terminal)
    for request_id, transaction in final_transactions.items():
        terminals = terminal_by_request[request_id]
        if tuple(item.owner_rank for item in terminals) != transaction.expected_owner_ranks:
            raise ValueError("terminal digest owner ranks differ from the final transaction owners")
        if any(item.state_revision != transaction.target_revision for item in terminals):
            raise ValueError("terminal state revision differs from the final transaction revision")

    expected_logits_keys = _expected_logits_keys(input_plans, batch_plans)
    if not expected_logits_keys:
        raise ValueError("a completed replay trace requires at least one output logits row")
    logits_keys = tuple(evidence.key for evidence in observation.logits_evidence)
    if logits_keys and logits_keys != expected_logits_keys:
        raise ValueError(
            "logits evidence must cover every output-producing plan/segment row in "
            "canonical plan order, "
            "or be empty while the runtime evidence bridge is unavailable"
        )


def _validate_plan_binding(
    input_plans: tuple[InputPlan, ...],
    batch_plans: tuple[BatchPlan, ...],
) -> tuple[tuple[str, ...], set[str], dict[str, StateTxn]]:
    if not input_plans:
        raise ValueError("a replay trace requires at least one input plan")
    if not batch_plans:
        raise ValueError("a replay trace requires at least one batch plan")
    request_ids = tuple(plan.request_id for plan in input_plans)
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("input plan request ids must be unique within a trace")
    known_requests = set(request_ids)
    plan_ids = tuple(plan.plan_id for plan in batch_plans)
    if len(set(plan_ids)) != len(plan_ids):
        raise ValueError("batch plan ids must be unique within a trace")
    referenced_requests = {request_id for plan in batch_plans for request_id in plan.input_plan_ids}
    if referenced_requests != known_requests:
        raise ValueError("batch plans must reference every and only traced input request")
    input_by_request = {plan.request_id: plan for plan in input_plans}
    final_transactions = _validate_plan_lifecycle(input_by_request, batch_plans)
    return request_ids, known_requests, final_transactions


def _publication_cannot_supply_full_prompt_logits(
    candidate: PrefixCandidateEvidence,
    publication: PrefixPublicationEvidence,
    *,
    prompt_length: int,
) -> bool:
    """Return whether this snapshot has no row from which to produce prompt logits.

    A snapshot ending at the complete prompt is not resumable when its arithmetic
    state is committed at that same boundary: there are zero token rows to replay,
    so the final prompt logits cannot be regenerated.  The runtime deliberately
    skips exactly this candidate and may therefore report a miss even though the
    cache contains the publication.
    """

    return (
        candidate.token_count == prompt_length
        and publication.physical_block_boundary == candidate.token_count
        and publication.arithmetic_state_commit_boundary == publication.physical_block_boundary
    )


def _validate_prefix_lifecycle(
    input_plans: tuple[InputPlan, ...],
    batch_plans: tuple[BatchPlan, ...],
    lifecycle: PrefixLifecycleEvidence,
    *,
    execution_fingerprint_digest: str,
) -> None:
    """Close prefix admission/publication identity without stable physical page ids."""

    input_by_request = {plan.request_id: plan for plan in input_plans}
    first_occurrences: dict[str, tuple[int, Any, StateTxn]] = {}
    segment_locations: dict[tuple[str, int], tuple[Any, StateTxn]] = {}
    request_order: list[str] = []
    plan_indices = {plan.plan_id: index for index, plan in enumerate(batch_plans)}
    for plan_index, batch_plan in enumerate(batch_plans):
        for segment_index, (segment, transaction) in enumerate(
            zip(batch_plan.segments, batch_plan.state_transactions, strict=True)
        ):
            segment_locations[(batch_plan.plan_id, segment_index)] = (
                segment,
                transaction,
            )
            if segment.request_id not in first_occurrences:
                first_occurrences[segment.request_id] = (
                    plan_index,
                    segment,
                    transaction,
                )
                request_order.append(segment.request_id)

    if tuple(item.request_id for item in lifecycle.admissions) != tuple(request_order):
        raise ValueError("prefix lifecycle admissions must cover requests in first-plan order")
    admission_by_request = {item.request_id: item for item in lifecycle.admissions}
    for admission in lifecycle.admissions:
        if admission.execution_fingerprint_digest != execution_fingerprint_digest:
            raise ValueError("prefix admission differs from trace execution fingerprint")
        plan_index, first_segment, first_transaction = first_occurrences[admission.request_id]
        if admission.after_plan_count != plan_index:
            raise ValueError("request admission must occur immediately before its first plan")
        prompt_length = len(input_by_request[admission.request_id].token_ids)
        if any(candidate.token_count > prompt_length for candidate in admission.candidates):
            raise ValueError("prefix admission candidate exceeds the realized prompt")
        expected_start = admission.arithmetic_state_commit_boundary if admission.hit else 0
        if first_segment.token_start != expected_start:
            raise ValueError("first segment start differs from prefix resume boundary")
        if first_transaction.base_revision != admission.base_revision:
            raise ValueError("first transaction base revision differs from prefix admission")
        if admission.hit and admission.physical_block_boundary > prompt_length:
            raise ValueError("restored physical prefix exceeds the realized prompt")

    if any(item.plan_id not in plan_indices for item in lifecycle.publications):
        raise ValueError("prefix publication references an unknown plan")
    publication_order = tuple(
        (plan_indices[item.plan_id], item.segment_index) for item in lifecycle.publications
    )
    if publication_order != tuple(sorted(publication_order)):
        raise ValueError("prefix publications must follow committed plan/segment order")
    for publication in lifecycle.publications:
        if publication.execution_fingerprint_digest != execution_fingerprint_digest:
            raise ValueError("prefix publication differs from trace execution fingerprint")
        try:
            plan_index = plan_indices[publication.plan_id]
            segment, transaction = segment_locations[
                (publication.plan_id, publication.segment_index)
            ]
        except KeyError as error:
            raise ValueError("prefix publication references an unknown plan segment") from error
        planned = transaction.prefix_publish
        if planned is None or segment.request_id != publication.request_id:
            raise ValueError("prefix publication is absent from its segment transaction")
        if publication.after_plan_count != plan_index + 1:
            raise ValueError("prefix publication must follow its committed plan exactly")
        if (
            publication.source_revision != transaction.target_revision
            or publication.source_rank not in transaction.expected_owner_ranks
            or publication.physical_block_boundary != planned.prefix_token_count
            or publication.arithmetic_state_commit_boundary != planned.state_commit_position
            or publication.state_commit_alignment != planned.commit_alignment_tokens
        ):
            raise ValueError("prefix publication boundaries/revision differ from its plan")
        if publication.physical_block_boundary > len(
            input_by_request[publication.request_id].token_ids
        ):
            raise ValueError("published prefix exceeds the realized prompt")

    expected_publication_locations: list[tuple[str, int]] = []
    for batch_plan in batch_plans:
        for segment_index, (segment, transaction) in enumerate(
            zip(batch_plan.segments, batch_plan.state_transactions, strict=True)
        ):
            planned = transaction.prefix_publish
            admission = admission_by_request[segment.request_id]
            if planned is None or not admission.use_prefix_cache:
                continue
            split_placement = any(
                placement.token_start
                < planned.prefix_token_count
                < placement.token_start + placement.token_count
                for placement in input_by_request[segment.request_id].placements
            )
            if not split_placement:
                expected_publication_locations.append((batch_plan.plan_id, segment_index))
    if tuple((item.plan_id, item.segment_index) for item in lifecycle.publications) != tuple(
        expected_publication_locations
    ):
        raise ValueError("prefix lifecycle must cover every cache-enabled committed publication")

    # Schema v5 has no eviction event, so its trace language treats every
    # earlier eligible publication as resident.  Relaxing that rule here would
    # make an unproven eviction indistinguishable from a forged miss.
    for admission in lifecycle.admissions:
        if not admission.hit:
            prompt_length = len(input_by_request[admission.request_id].token_ids)
            for candidate in admission.candidates:
                for publication in lifecycle.publications:
                    if (
                        publication.after_plan_count <= admission.after_plan_count
                        and publication.prefix_key == candidate.prefix_key
                        and publication.physical_block_boundary == candidate.token_count
                        and not _publication_cannot_supply_full_prompt_logits(
                            candidate,
                            publication,
                            prompt_length=prompt_length,
                        )
                    ):
                        raise ValueError(
                            "prefix miss conflicts with an earlier eligible publication"
                        )
            continue
        matching_sources = tuple(
            publication
            for publication in lifecycle.publications
            if publication.after_plan_count <= admission.after_plan_count
            and publication.request_id == admission.source_request_id
            and publication.prefix_key == admission.prefix_key
            and publication.source_rank == admission.source_rank
            and publication.source_revision == admission.source_revision
            and publication.physical_block_boundary == admission.physical_block_boundary
            and publication.arithmetic_state_commit_boundary
            == admission.arithmetic_state_commit_boundary
            and publication.state_commit_alignment == admission.state_commit_alignment
            and publication.snapshot_semantic_digest == admission.snapshot_semantic_digest
        )
        if not matching_sources:
            raise ValueError("prefix hit is not backed by an earlier recorded publication identity")


def _validate_plan_lifecycle(
    input_by_request: dict[str, InputPlan],
    batch_plans: tuple[BatchPlan, ...],
) -> dict[str, StateTxn]:
    """Validate per-request plan span, phase, and revision continuity.

    Initial token offsets and base revisions are intentionally not constrained
    here.  The subsequent prefix-lifecycle validation binds those initial values
    to the schema-v5 admission/restore evidence.
    """

    previous_token_end: dict[str, int] = {}
    previous_target_revision: dict[str, int] = {}
    decode_started: set[str] = set()
    final_transactions: dict[str, StateTxn] = {}

    for batch_plan in batch_plans:
        for segment, transaction in zip(
            batch_plan.segments,
            batch_plan.state_transactions,
            strict=True,
        ):
            input_plan = input_by_request[segment.request_id]
            if segment.input_plan_digest != input_plan.semantic_digest:
                raise ValueError(
                    "segment input_plan_digest differs from the corresponding "
                    "InputPlan semantic digest"
                )

            token_end = segment.token_start + segment.token_count
            previous_end = previous_token_end.get(segment.request_id)
            if previous_end is not None and segment.token_start != previous_end:
                raise ValueError("segment token spans must be contiguous per request")

            prompt_length = len(input_plan.token_ids)
            if segment.phase is BatchPhase.PREFILL:
                if segment.request_id in decode_started:
                    raise ValueError("a request cannot return to prefill after decode")
                if token_end > prompt_length:
                    raise ValueError("prefill segment exceeds the realized prompt span")
                expected_ids = input_plan.token_ids[segment.token_start : token_end]
                expected_positions = input_plan.positions[segment.token_start : token_end]
                if segment.token_ids != expected_ids or segment.positions != expected_positions:
                    raise ValueError(
                        "prefill segment tokens/positions differ from the InputPlan span"
                    )
            elif segment.phase is BatchPhase.DECODE:
                if segment.token_start < prompt_length:
                    raise ValueError("decode segment starts inside the realized prompt span")
                decode_started.add(segment.request_id)
                decode_start = segment.token_start - prompt_length
                expected_positions = tuple(
                    input_plan.decode_positions.at(decode_start + offset)
                    for offset in range(segment.token_count)
                )
                if segment.positions != expected_positions:
                    raise ValueError(
                        "decode segment positions differ from the InputPlan continuation"
                    )
            else:
                raise ValueError("individual plan segments cannot use the mixed phase")

            previous_revision = previous_target_revision.get(segment.request_id)
            if previous_revision is not None and transaction.base_revision != previous_revision:
                raise ValueError("transaction state revisions must be contiguous per request")
            previous_token_end[segment.request_id] = token_end
            previous_target_revision[segment.request_id] = transaction.target_revision
            final_transactions[segment.request_id] = transaction

    return final_transactions


def _expected_logits_keys(
    input_plans: tuple[InputPlan, ...],
    batch_plans: tuple[BatchPlan, ...],
) -> tuple[tuple[str, str, int, int], ...]:
    prompt_lengths = {plan.request_id: len(plan.token_ids) for plan in input_plans}
    return tuple(
        (
            segment.request_id,
            plan.plan_id,
            segment_index,
            segment.token_count - 1,
        )
        for plan in batch_plans
        for segment_index, segment in enumerate(plan.segments)
        if segment.phase is BatchPhase.DECODE
        or segment.token_start + segment.token_count == prompt_lengths[segment.request_id]
    )


def _first_logits_mismatch(
    required_keys: tuple[tuple[str, str, int, int], ...],
    recorded: dict[tuple[str, str, int, int], LogitsEvidence],
    replayed: dict[tuple[str, str, int, int], LogitsEvidence],
) -> tuple[str, str, int, int] | None:
    for key in required_keys:
        if key not in recorded or key not in replayed or recorded[key] != replayed[key]:
            return key
    return None


def _first_mismatch(expected: tuple[Any, ...], actual: tuple[Any, ...]) -> int | None:
    for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=False)):
        if expected_item != actual_item:
            return index
    if len(expected) != len(actual):
        return min(len(expected), len(actual))
    return None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest_json(value: Any) -> str:
    return "b2s256:" + hashlib.blake2s(_canonical_json(value).encode("utf-8")).hexdigest()


def _strict_json_loads(encoded: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number is forbidden: {value}")

    try:
        return json.loads(
            encoded,
            object_pairs_hook=object_pairs,
            parse_constant=invalid_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError("invalid trace JSON") from error


def _require_exact_keys(value: dict[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"{path} keys differ: missing={missing}, unknown={unknown}")


def _require_list(value: Any, path: str) -> list[Any]:
    if type(value) is not list:
        raise TypeError(f"{path} must be a JSON array")
    return value


def _encode(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _encode(getattr(value, field.name)) for field in fields(value)}
    if type(value) is tuple:
        return [_encode(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"unsupported replay value type: {type(value).__name__}")


def _decode(annotation: Any, value: Any) -> Any:
    if isinstance(annotation, TypeAliasType):
        return _decode(annotation.__value__, value)
    origin = get_origin(annotation)
    if origin in (types.UnionType, typing.Union):
        alternatives = get_args(annotation)
        if value is None and type(None) in alternatives:
            return None
        non_none = tuple(item for item in alternatives if item is not type(None))
        if len(non_none) != 1:
            raise TypeError(f"unsupported replay union annotation: {annotation!r}")
        return _decode(non_none[0], value)
    if origin is tuple:
        if type(value) is not list:
            raise TypeError("tuple field must be encoded as a JSON array")
        arguments = get_args(annotation)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(_decode(arguments[0], item) for item in value)
        if len(arguments) != len(value):
            raise ValueError("fixed tuple field has the wrong length")
        return tuple(
            _decode(item_type, item) for item_type, item in zip(arguments, value, strict=True)
        )
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if type(value) is not str:
            raise TypeError(f"{annotation.__name__} must be encoded as a string")
        try:
            return annotation(value)
        except ValueError as error:
            raise ValueError(f"unknown {annotation.__name__} value: {value!r}") from error
    if annotation is LogitsEvidence:
        if type(value) is not dict:
            raise TypeError("LogitsEvidence must be encoded as a JSON object")
        _require_exact_keys(
            value,
            {
                "request_id",
                "plan_id",
                "segment_index",
                "row_index",
                "dtype",
                "shape",
                "bit_digest",
            },
            "LogitsEvidence",
        )
        return _issue_logits_evidence(
            request_id=_decode(str, value["request_id"]),
            plan_id=_decode(str, value["plan_id"]),
            segment_index=_decode(int, value["segment_index"]),
            row_index=_decode(int, value["row_index"]),
            dtype=_decode(str, value["dtype"]),
            shape=_decode(tuple[int, ...], value["shape"]),
            bit_digest=_decode(str, value["bit_digest"]),
        )
    if isinstance(annotation, type) and is_dataclass(annotation):
        if type(value) is not dict:
            raise TypeError(f"{annotation.__name__} must be encoded as a JSON object")
        declared_fields = fields(annotation)
        _require_exact_keys(
            value,
            {field.name for field in declared_fields},
            annotation.__name__,
        )
        hints = get_type_hints(annotation)
        return annotation(
            **{
                field.name: _decode(hints[field.name], value[field.name])
                for field in declared_fields
            }
        )
    if annotation is type(None):
        if value is not None:
            raise TypeError("expected null")
        return None
    if annotation in (str, int, float, bool):
        if type(value) is not annotation:
            raise TypeError(f"expected {annotation.__name__}, got {type(value).__name__}")
        return value
    raise TypeError(f"unsupported replay annotation: {annotation!r}")
