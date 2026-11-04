"""Adapter from production ReferenceEngine execution to guardrail replay."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping

from cubic_serve.contracts.execution import BatchPlan, ExecutionMode
from cubic_serve.contracts.request import InputPlan, TokenEvent
from cubic_serve.guardrail.replay import (
    LogitsEvidence,
    PrefixAdmissionEvidence,
    PrefixCandidateEvidence,
    PrefixLifecycleEvidence,
    PrefixPublicationEvidence,
    ReplayExecutionEvidence,
    ReplayObservation,
    TerminalDigest,
    capture_logits_evidence,
)
from cubic_serve.models import PreparedInput
from cubic_serve.runtime.engine import (
    PrefixAdmission,
    PrefixPublication,
    ReferenceEngine,
)


class ReferenceEngineReplayExecutor:
    """Regenerate each supplied plan, execute it, and read back committed state."""

    def __init__(
        self,
        *,
        engine_factory: Callable[[], ReferenceEngine],
        max_new_tokens: Mapping[str, int],
        prepared_inputs: Mapping[str, PreparedInput] | None = None,
        use_prefix_cache: bool | Mapping[str, bool] = False,
    ) -> None:
        if not callable(engine_factory):
            raise TypeError("reference replay requires a callable engine factory")
        limits = dict(max_new_tokens)
        if not limits or any(not request_id or limit <= 0 for request_id, limit in limits.items()):
            raise ValueError("replay request generation limits must be positive")
        self._engine_factory = engine_factory
        self._limits = limits
        self._prepared_inputs = dict(prepared_inputs or {})
        if any(
            request_id != prepared.input_plan.request_id
            for request_id, prepared in self._prepared_inputs.items()
        ):
            raise ValueError("prepared input mapping keys must equal their request ids")
        self._execution_evidence: ReplayExecutionEvidence | None = None
        if not isinstance(use_prefix_cache, bool | Mapping):
            raise TypeError("prefix cache replay mode must be a bool or request mapping")
        if isinstance(use_prefix_cache, Mapping):
            prefix_modes = dict(use_prefix_cache)
            if any(
                type(request_id) is not str or not request_id or type(enabled) is not bool
                for request_id, enabled in prefix_modes.items()
            ):
                raise TypeError("prefix cache replay mapping must contain string/bool pairs")
            self._prefix_modes: bool | dict[str, bool] = prefix_modes
        else:
            self._prefix_modes = use_prefix_cache

    @property
    def execution_evidence(self) -> ReplayExecutionEvidence:
        if self._execution_evidence is None:
            raise RuntimeError("execution evidence is available only after engine execution")
        return self._execution_evidence

    def execute(
        self,
        input_plans: tuple[InputPlan, ...],
        batch_plans: tuple[BatchPlan, ...],
        expected_prefix_lifecycle: PrefixLifecycleEvidence | None = None,
    ) -> ReplayObservation:
        request_ids = tuple(plan.request_id for plan in input_plans)
        if set(request_ids) != set(self._limits):
            raise ValueError("replay generation limits differ from traced requests")
        if not set(self._prepared_inputs) <= set(request_ids):
            raise ValueError("prepared input resolver contains an untraced request")
        engine = self._engine_factory()
        if not isinstance(engine, ReferenceEngine):
            raise TypeError("engine factory must return a fresh ReferenceEngine")
        prepared: list[PreparedInput] = []
        for input_plan in input_plans:
            prepared_input = self._prepared_inputs.get(input_plan.request_id)
            if prepared_input is None:
                if input_plan.encoder_jobs:
                    raise ValueError(
                        "multimodal replay requires its exact external PreparedInput artifact"
                    )
                prepared_input = PreparedInput(input_plan)
            elif prepared_input.input_plan != input_plan:
                raise ValueError("prepared replay artifact differs from the traced InputPlan")
            prepared.append(prepared_input)
        prepared_by_request = {item.input_plan.request_id: item for item in prepared}
        if isinstance(self._prefix_modes, bool):
            prefix_modes = {request_id: self._prefix_modes for request_id in request_ids}
        else:
            if set(self._prefix_modes) != set(request_ids):
                raise ValueError("prefix cache replay mapping differs from traced requests")
            prefix_modes = dict(self._prefix_modes)
        for batch_plan in batch_plans:
            for segment in batch_plan.segments:
                if (
                    segment.execution_input_digest
                    != prepared_by_request[segment.request_id].semantic_digest
                ):
                    raise ValueError(
                        "traced BatchPlan execution input differs from the exact "
                        "PreparedInput artifact"
                    )
        fingerprint = engine.execution_fingerprint
        self._execution_evidence = ReplayExecutionEvidence(
            executor_id=fingerprint.executor_id,
            mode=ExecutionMode.REFERENCE,
            contract_digest=fingerprint.contract_digest,
            model_digest=fingerprint.model_digest,
            implementation_digest=fingerprint.implementation_digest,
            topology_digest=fingerprint.topology_digest,
            evidence_digest=_prepared_execution_digest(
                engine_evidence_digest=fingerprint.evidence_digest,
                prepared_inputs=tuple(prepared),
                prefix_cache_modes=tuple(
                    (request_id, prefix_modes[request_id]) for request_id in request_ids
                ),
            ),
        )

        admissions_at: dict[int, list[str]] = {}
        seen_requests: set[str] = set()
        for plan_index, batch_plan in enumerate(batch_plans):
            for segment in batch_plan.segments:
                if segment.request_id in seen_requests:
                    continue
                seen_requests.add(segment.request_id)
                admissions_at.setdefault(plan_index, []).append(segment.request_id)
        if seen_requests != set(request_ids):
            raise ValueError("batch plans do not schedule every replay request")
        expected_admissions = (
            {}
            if expected_prefix_lifecycle is None
            else {item.request_id: item for item in expected_prefix_lifecycle.admissions}
        )
        if expected_prefix_lifecycle is not None:
            for plan_index, expected_plan in enumerate(batch_plans):
                _preflight_publications(
                    engine,
                    expected_plan,
                    plan_index=plan_index,
                    prepared_inputs=prepared_by_request,
                    prefix_cache_modes=prefix_modes,
                    execution_fingerprint_digest=self._execution_evidence.digest,
                    expected=expected_prefix_lifecycle.publications,
                )

        events: list[TokenEvent] = []
        logits_evidence: list[LogitsEvidence] = []
        prefix_admissions: list[PrefixAdmissionEvidence] = []
        prefix_publications: list[PrefixPublicationEvidence] = []
        for plan_index, expected_plan in enumerate(batch_plans):
            for request_id in admissions_at.get(plan_index, []):
                prepared_input = prepared_by_request[request_id]
                preview = engine.preview_prefix_admission(
                    prepared_input,
                    use_prefix_cache=prefix_modes[request_id],
                )
                preview_evidence = _capture_prefix_admission(
                    preview,
                    request_id=request_id,
                    after_plan_count=plan_index,
                    use_prefix_cache=prefix_modes[request_id],
                    execution_fingerprint_digest=self._execution_evidence.digest,
                )
                expected_admission = expected_admissions.get(request_id)
                if expected_prefix_lifecycle is not None and preview_evidence != expected_admission:
                    raise ValueError(
                        f"prefix admission evidence differs before model execution "
                        f"for {request_id!r}"
                    )
                actual = engine.admit(
                    prepared_input,
                    max_new_tokens=self._limits[request_id],
                    use_prefix_cache=prefix_modes[request_id],
                    expected_prefix_admission=preview,
                )
                actual_evidence = _capture_prefix_admission(
                    actual,
                    request_id=request_id,
                    after_plan_count=plan_index,
                    use_prefix_cache=prefix_modes[request_id],
                    execution_fingerprint_digest=self._execution_evidence.digest,
                )
                if actual_evidence != preview_evidence:
                    raise RuntimeError("prefix admission changed after fail-closed preview")
                prefix_admissions.append(actual_evidence)

            step = engine.step()
            if step is None:
                raise RuntimeError("reference engine drained before the supplied trace")
            if step.batch_plan != expected_plan:
                raise RuntimeError(f"regenerated BatchPlan differs at {expected_plan.plan_id!r}")
            if not step.post_commit_ok:
                failures = ", ".join(
                    f"{failure.request_id}@{failure.prefix_token_count}:{failure.error_type}"
                    for failure in step.prefix_publication_failures
                )
                raise RuntimeError(
                    f"reference replay observed post-commit prefix failures: {failures}"
                )
            events.extend(step.events)
            logits_evidence.extend(
                capture_logits_evidence(
                    request_id=row.request_id,
                    plan_id=row.plan_id,
                    segment_index=row.segment_index,
                    row_index=row.row_index,
                    logits=row.tensor,
                )
                for row in step.logits_rows
            )
            actual_publications = tuple(
                _capture_prefix_publication(
                    publication,
                    after_plan_count=plan_index + 1,
                    execution_fingerprint_digest=self._execution_evidence.digest,
                )
                for publication in step.prefix_publications
            )
            if expected_prefix_lifecycle is not None:
                expected_for_plan = tuple(
                    item
                    for item in expected_prefix_lifecycle.publications
                    if item.plan_id == expected_plan.plan_id
                )
                if actual_publications != expected_for_plan:
                    raise RuntimeError(
                        f"prefix publication evidence differs after {expected_plan.plan_id!r}"
                    )
            prefix_publications.extend(actual_publications)
        if not engine.scheduler.drained:
            raise RuntimeError("supplied trace ended before the reference engine drained")

        terminals = tuple(
            TerminalDigest(
                request_id=digest.request_id,
                owner_rank=digest.owner_rank,
                state_revision=digest.state_revision,
                cache_digest=digest.cache_digest,
                state_digest=digest.state_digest,
            )
            for digest in sorted(
                (engine.terminal_digest(request_id) for request_id in request_ids),
                key=lambda item: (item.request_id, item.owner_rank),
            )
        )
        return ReplayObservation(
            events=tuple(events),
            terminal_digests=terminals,
            logits_evidence=tuple(logits_evidence),
            prefix_lifecycle=PrefixLifecycleEvidence(
                tuple(prefix_admissions),
                tuple(prefix_publications),
            ),
        )


def _prepared_execution_digest(
    *,
    engine_evidence_digest: str,
    prepared_inputs: tuple[PreparedInput, ...],
    prefix_cache_modes: tuple[tuple[str, bool], ...],
) -> str:
    payload = {
        "schema": "cubic.reference-replay.prepared-inputs.v2",
        "engine_evidence_digest": engine_evidence_digest,
        "prepared_inputs": tuple(
            (prepared.input_plan.request_id, prepared.semantic_digest)
            for prepared in prepared_inputs
        ),
        "prefix_cache_modes": prefix_cache_modes,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"b2s256:{hashlib.blake2s(encoded).hexdigest()}"


def _capture_prefix_admission(
    admission: PrefixAdmission,
    *,
    request_id: str,
    after_plan_count: int,
    use_prefix_cache: bool,
    execution_fingerprint_digest: str,
) -> PrefixAdmissionEvidence:
    snapshot = admission.snapshot
    return PrefixAdmissionEvidence(
        request_id=request_id,
        after_plan_count=after_plan_count,
        use_prefix_cache=use_prefix_cache,
        execution_fingerprint_digest=execution_fingerprint_digest,
        candidates=tuple(
            PrefixCandidateEvidence(token_count, prefix_key)
            for token_count, prefix_key in admission.candidate_keys
        ),
        hit=admission.hit,
        prefix_key=admission.prefix_key,
        physical_block_boundary=admission.token_count,
        arithmetic_state_commit_boundary=admission.resume_token_count,
        state_commit_alignment=(0 if snapshot is None else snapshot.state_commit_alignment),
        base_revision=admission.base_revision,
        source_request_id=(None if snapshot is None else snapshot.source_owner_id),
        source_rank=(None if snapshot is None else snapshot.source_rank),
        source_revision=(None if snapshot is None else snapshot.source_revision),
        snapshot_semantic_digest=(None if snapshot is None else snapshot.semantic_digest),
    )


def _capture_prefix_publication(
    publication: PrefixPublication,
    *,
    after_plan_count: int,
    execution_fingerprint_digest: str,
) -> PrefixPublicationEvidence:
    snapshot = publication.snapshot
    return PrefixPublicationEvidence(
        request_id=publication.request_id,
        plan_id=publication.plan_id,
        segment_index=publication.segment_index,
        after_plan_count=after_plan_count,
        execution_fingerprint_digest=execution_fingerprint_digest,
        prefix_key=snapshot.prefix_key,
        source_rank=snapshot.source_rank,
        source_revision=snapshot.source_revision,
        physical_block_boundary=snapshot.physical_block_boundary,
        arithmetic_state_commit_boundary=snapshot.arithmetic_state_commit_boundary,
        state_commit_alignment=snapshot.state_commit_alignment,
        snapshot_semantic_digest=snapshot.semantic_digest,
        inserted=publication.inserted,
    )


def _preflight_publications(
    engine: ReferenceEngine,
    batch_plan: BatchPlan,
    *,
    plan_index: int,
    prepared_inputs: Mapping[str, PreparedInput],
    prefix_cache_modes: Mapping[str, bool],
    execution_fingerprint_digest: str,
    expected: tuple[PrefixPublicationEvidence, ...],
) -> None:
    expected_for_plan = tuple(item for item in expected if item.plan_id == batch_plan.plan_id)
    expected_by_segment = {item.segment_index: item for item in expected_for_plan}
    actual_segments: list[int] = []
    for segment_index, (segment, transaction) in enumerate(
        zip(batch_plan.segments, batch_plan.state_transactions, strict=True)
    ):
        planned = transaction.prefix_publish
        if planned is None or not prefix_cache_modes[segment.request_id]:
            continue
        try:
            prefix_key = engine.prefix_key(
                prepared_inputs[segment.request_id],
                planned.prefix_token_count,
            )
        except ValueError:
            continue
        actual_segments.append(segment_index)
        evidence = expected_by_segment.get(segment_index)
        if evidence is None:
            raise ValueError("trace omitted a planned prefix publication")
        expected_fields = (
            segment.request_id,
            batch_plan.plan_id,
            segment_index,
            plan_index + 1,
            execution_fingerprint_digest,
            prefix_key,
            engine.store.rank_id,
            transaction.target_revision,
            planned.prefix_token_count,
            planned.state_commit_position,
            planned.commit_alignment_tokens,
        )
        actual_fields = (
            evidence.request_id,
            evidence.plan_id,
            evidence.segment_index,
            evidence.after_plan_count,
            evidence.execution_fingerprint_digest,
            evidence.prefix_key,
            evidence.source_rank,
            evidence.source_revision,
            evidence.physical_block_boundary,
            evidence.arithmetic_state_commit_boundary,
            evidence.state_commit_alignment,
        )
        if actual_fields != expected_fields:
            raise ValueError("prefix publication key/boundary/revision failed preflight")
    if tuple(expected_by_segment) != tuple(actual_segments):
        raise ValueError("trace carries an unplanned prefix publication")
