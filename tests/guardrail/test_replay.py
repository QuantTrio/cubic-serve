from __future__ import annotations

import hashlib
import inspect
import json
import unittest
from collections.abc import Callable
from dataclasses import replace

import torch

from cubic_serve.contracts.execution import (
    BatchPhase,
    BatchPlan,
    BatchShape,
    ExecutionMode,
    PlanSegment,
    PrefixPublishPlan,
    StateAccess,
    StateTxn,
)
from cubic_serve.contracts.model import WeightVariantMeta
from cubic_serve.contracts.request import (
    ContentPart,
    DecodePositionPlan,
    InputPlan,
    TokenEvent,
    TokenEventKind,
    TokenPosition,
)
from cubic_serve.guardrail.replay import (
    LogitsEvidence,
    PrefixAdmissionEvidence,
    PrefixCandidateEvidence,
    PrefixLifecycleEvidence,
    PrefixPublicationEvidence,
    ReplayExecutionEvidence,
    ReplayObservation,
    ReplayTrace,
    TerminalDigest,
    capture_logits_evidence,
    record_trace,
    replay_trace,
)


def digest(label: str) -> str:
    return "b2s256:" + hashlib.blake2s(label.encode("utf-8")).hexdigest()


def input_plan() -> InputPlan:
    return InputPlan(
        request_id="request:a",
        model_id="toy",
        processor_recipe_digest=f"b2s256:{'a' * 64}",
        weight_variant=WeightVariantMeta(
            base_model_id="toy",
            base_revision="weights:1",
        ),
        content_parts=(ContentPart(ordinal=0, modality="text", text="fixture"),),
        token_ids=(11, 12),
        positions=(TokenPosition((0, 0, 0)), TokenPosition((1, 1, 1))),
        decode_positions=DecodePositionPlan(
            first=TokenPosition((2, 2, 2)),
            step_axes=(1, 1, 1),
        ),
    )


def batch_plan() -> BatchPlan:
    return BatchPlan(
        plan_id="batch:0",
        phase=BatchPhase.PREFILL,
        input_plan_ids=("request:a",),
        segments=(
            PlanSegment(
                request_id="request:a",
                input_plan_digest=input_plan().semantic_digest,
                execution_input_digest=input_plan().semantic_digest,
                phase=BatchPhase.PREFILL,
                produces_output=True,
                token_start=0,
                token_count=2,
                token_ids=(11, 12),
                positions=(TokenPosition((0, 0, 0)), TokenPosition((1, 1, 1))),
                cache_slots=(0, 1),
                cache_pages=(0,),
            ),
        ),
        shape=BatchShape(logical_m=2),
        topology_id="single",
        partition_id="toy",
        state_transactions=(
            StateTxn(
                transaction_id="txn:0",
                request_id="request:a",
                base_revision=0,
                target_revision=1,
                reads=(StateAccess(resource_id="kv:request:a", owner_rank=0, pages=(0,)),),
                writes=(
                    StateAccess(
                        resource_id="kv:request:a",
                        owner_rank=0,
                        slots=(0, 1),
                        pages=(0,),
                    ),
                ),
                expected_owner_ranks=(0,),
            ),
        ),
    )


def events(*, token_id: int = 42) -> tuple[TokenEvent, ...]:
    return (
        TokenEvent(
            request_id="request:a",
            sequence_no=0,
            kind=TokenEventKind.TOKEN,
            token_id=token_id,
        ),
        TokenEvent(
            request_id="request:a",
            sequence_no=1,
            kind=TokenEventKind.FINISH,
            payload_items=(("reason", "length"),),
        ),
    )


def terminal(*, state_label: str = "state:1") -> tuple[TerminalDigest, ...]:
    return (
        TerminalDigest(
            request_id="request:a",
            owner_rank=0,
            state_revision=1,
            cache_digest=digest("cache:1"),
            state_digest=digest(state_label),
        ),
    )


def logits_evidence(
    *,
    second_row: tuple[float, ...] = (4.0, 5.0, 6.0),
    second_dtype: torch.dtype = torch.bfloat16,
) -> tuple[LogitsEvidence, ...]:
    return (
        capture_logits_evidence(
            request_id="request:a",
            plan_id="batch:0",
            segment_index=0,
            row_index=1,
            logits=torch.tensor(second_row, dtype=second_dtype),
        ),
    )


def evidence(executor_id: str) -> ReplayExecutionEvidence:
    return ReplayExecutionEvidence(
        executor_id=executor_id,
        mode=ExecutionMode.REFERENCE,
        contract_digest=digest("contracts"),
        model_digest=digest("model"),
        implementation_digest=digest(executor_id),
        topology_digest=digest("single"),
        evidence_digest=digest(f"evidence:{executor_id}"),
    )


class FixtureExecutor:
    def __init__(
        self,
        *,
        identity: ReplayExecutionEvidence,
        observation: ReplayObservation,
        calls: list[tuple[tuple[InputPlan, ...], tuple[BatchPlan, ...]]],
    ) -> None:
        self._identity = identity
        self._observation = observation
        self._calls = calls

    @property
    def execution_evidence(self) -> ReplayExecutionEvidence:
        return self._identity

    def execute(
        self,
        input_plans: tuple[InputPlan, ...],
        batch_plans: tuple[BatchPlan, ...],
        expected_prefix_lifecycle: PrefixLifecycleEvidence | None = None,
    ) -> ReplayObservation:
        self._calls.append((input_plans, batch_plans))
        if self._observation.prefix_lifecycle.admissions:
            return self._observation
        return replace(
            self._observation,
            prefix_lifecycle=miss_lifecycle(
                self._identity,
                input_plans,
                batch_plans,
            ),
        )


def miss_lifecycle(
    identity: ReplayExecutionEvidence,
    input_plans: tuple[InputPlan, ...],
    batch_plans: tuple[BatchPlan, ...],
) -> PrefixLifecycleEvidence:
    first_plan: dict[str, int] = {}
    order: list[str] = []
    for plan_index, plan in enumerate(batch_plans):
        for segment in plan.segments:
            if segment.request_id not in first_plan:
                first_plan[segment.request_id] = plan_index
                order.append(segment.request_id)
    if set(first_plan) != {plan.request_id for plan in input_plans}:
        raise ValueError("fixture plans must reference every input")
    return PrefixLifecycleEvidence(
        tuple(
            PrefixAdmissionEvidence(
                request_id=request_id,
                after_plan_count=first_plan[request_id],
                use_prefix_cache=False,
                execution_fingerprint_digest=identity.digest,
                candidates=(),
                hit=False,
                prefix_key=None,
                physical_block_boundary=0,
                arithmetic_state_commit_boundary=0,
                state_commit_alignment=0,
                base_revision=0,
                source_request_id=None,
                source_rank=None,
                source_revision=None,
                snapshot_semantic_digest=None,
            )
            for request_id in order
        )
    )


def factory(
    *,
    identity: ReplayExecutionEvidence,
    observation: ReplayObservation,
    calls: list[tuple[tuple[InputPlan, ...], tuple[BatchPlan, ...]]],
    instances: list[FixtureExecutor] | None = None,
) -> Callable[[], FixtureExecutor]:
    def create() -> FixtureExecutor:
        executor = FixtureExecutor(
            identity=identity,
            observation=observation,
            calls=calls,
        )
        if instances is not None:
            instances.append(executor)
        return executor

    return create


class ReplayTests(unittest.TestCase):
    def make_trace(self) -> ReplayTrace:
        calls: list[tuple[tuple[InputPlan, ...], tuple[BatchPlan, ...]]] = []
        return record_trace(
            input_plans=(input_plan(),),
            batch_plans=(batch_plan(),),
            executor_factory=factory(
                identity=evidence("reference:eager"),
                observation=ReplayObservation(events(), terminal(), logits_evidence()),
                calls=calls,
            ),
        )

    def test_record_is_canonical_and_binds_decode_rows_and_terminal_state(self) -> None:
        source_input = input_plan()
        source_batch = batch_plan()
        calls: list[tuple[tuple[InputPlan, ...], tuple[BatchPlan, ...]]] = []
        trace = record_trace(
            input_plans=(source_input,),
            batch_plans=(source_batch,),
            executor_factory=factory(
                identity=evidence("reference:eager"),
                observation=ReplayObservation(events(), terminal(), logits_evidence()),
                calls=calls,
            ),
        )

        self.assertEqual(len(calls), 1)
        self.assertIsNot(calls[0][0][0], source_input)
        self.assertIsNot(calls[0][1][0], source_batch)
        encoded = trace.to_json()
        self.assertEqual(
            encoded,
            json.dumps(
                json.loads(encoded),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )
        payload = json.loads(encoded)
        self.assertEqual(payload["schema"], "cubic.guardrail.replay")
        self.assertEqual(payload["schema_version"], 5)
        self.assertEqual(payload["trace_id"], trace.trace_id)
        self.assertEqual(
            payload["input_plans"][0]["decode_positions"]["step_axes"],
            [1, 1, 1],
        )
        self.assertEqual(payload["batch_plans"][0]["segments"][0]["token_ids"], [11, 12])
        self.assertEqual(
            payload["batch_plans"][0]["segments"][0]["positions"][1]["axes"],
            [1, 1, 1],
        )
        self.assertEqual(
            payload["recorded_logits_evidence"][0]["dtype"],
            "torch.bfloat16",
        )
        self.assertEqual(payload["recorded_logits_evidence"][0]["shape"], [3])
        self.assertEqual(
            payload["recorded_logits_evidence"][0]["bit_digest"],
            logits_evidence()[0].bit_digest,
        )
        self.assertEqual(ReplayTrace.from_json(encoded), trace)

    def test_replay_uses_a_fresh_executor_and_compares_exact_observation(self) -> None:
        trace = self.make_trace()
        calls: list[tuple[tuple[InputPlan, ...], tuple[BatchPlan, ...]]] = []
        instances: list[FixtureExecutor] = []
        result = replay_trace(
            trace,
            executor_factory=factory(
                identity=evidence("reference:eager"),
                observation=ReplayObservation(events(), terminal(), logits_evidence()),
                calls=calls,
                instances=instances,
            ),
        )

        self.assertTrue(result.passed)
        self.assertEqual(len(instances), 1)
        self.assertEqual(len(calls), 1)
        self.assertIsNot(calls[0][0], trace.input_plans)
        self.assertIsNot(calls[0][1], trace.batch_plans)
        self.assertEqual(result.recorded_execution.executor_id, "reference:eager")
        self.assertEqual(result.replay_execution.executor_id, "reference:eager")
        self.assertTrue(result.execution_evidence_match)
        self.assertTrue(result.logits_evidence_complete)
        self.assertTrue(result.logits_evidence_match)
        self.assertIsNone(result.first_logits_mismatch_key)
        self.assertEqual(
            set(inspect.signature(replay_trace).parameters),
            {"trace", "executor_factory"},
        )

    def test_same_observation_from_different_execution_evidence_fails(self) -> None:
        trace = self.make_trace()
        result = replay_trace(
            trace,
            executor_factory=factory(
                identity=evidence("candidate:eager"),
                observation=ReplayObservation(events(), terminal(), logits_evidence()),
                calls=[],
            ),
        )

        self.assertFalse(result.passed)
        self.assertFalse(result.execution_evidence_match)
        self.assertTrue(result.event_sequence_match)
        self.assertTrue(result.terminal_digest_match)
        self.assertEqual(result.recorded_execution.executor_id, "reference:eager")
        self.assertEqual(result.replay_execution.executor_id, "candidate:eager")
        with self.assertRaisesRegex(ValueError, "exact evidence equality"):
            replace(result, execution_evidence_match=True)

    def test_replay_reports_event_and_terminal_mismatch(self) -> None:
        trace = self.make_trace()
        result = replay_trace(
            trace,
            executor_factory=factory(
                identity=evidence("reference:eager"),
                observation=ReplayObservation(
                    events(token_id=43),
                    terminal(state_label="state:corrupt"),
                    logits_evidence(),
                ),
                calls=[],
            ),
        )

        self.assertFalse(result.passed)
        self.assertTrue(result.execution_evidence_match)
        self.assertFalse(result.event_sequence_match)
        self.assertEqual(result.first_event_mismatch_index, 0)
        self.assertFalse(result.terminal_digest_match)
        self.assertEqual(result.terminal_mismatch_keys, (("request:a", 0),))

    def test_replay_reports_first_exact_logits_bit_mismatch(self) -> None:
        trace = record_trace(
            input_plans=(input_plan(),),
            batch_plans=(batch_plan(),),
            executor_factory=factory(
                identity=evidence("reference:eager"),
                observation=ReplayObservation(
                    events(),
                    terminal(),
                    logits_evidence(second_row=(4.0, 5.0, 0.0)),
                ),
                calls=[],
            ),
        )
        result = replay_trace(
            trace,
            executor_factory=factory(
                identity=evidence("reference:eager"),
                observation=ReplayObservation(
                    events(),
                    terminal(),
                    logits_evidence(second_row=(4.0, 5.0, -0.0)),
                ),
                calls=[],
            ),
        )

        self.assertFalse(result.passed)
        self.assertTrue(result.logits_evidence_complete)
        self.assertFalse(result.logits_evidence_match)
        self.assertEqual(
            result.first_logits_mismatch_key,
            ("request:a", "batch:0", 0, 1),
        )
        self.assertTrue(result.event_sequence_match)
        self.assertTrue(result.terminal_digest_match)

    def test_replay_includes_logits_dtype_and_shape_in_exact_verdict(self) -> None:
        trace = self.make_trace()
        variants = (
            logits_evidence(second_dtype=torch.float32),
            logits_evidence(second_row=(4.0, 5.0, 6.0, 7.0)),
        )
        for replay_logits in variants:
            with self.subTest(dtype=replay_logits[0].dtype, shape=replay_logits[0].shape):
                result = replay_trace(
                    trace,
                    executor_factory=factory(
                        identity=evidence("reference:eager"),
                        observation=ReplayObservation(events(), terminal(), replay_logits),
                        calls=[],
                    ),
                )
                self.assertFalse(result.passed)
                self.assertFalse(result.logits_evidence_match)
                self.assertEqual(
                    result.first_logits_mismatch_key,
                    ("request:a", "batch:0", 0, 1),
                )

    def test_logits_evidence_can_only_be_issued_from_a_real_finite_row(self) -> None:
        with self.assertRaisesRegex(TypeError, "real tensor"):
            LogitsEvidence(
                request_id="request:a",
                plan_id="batch:0",
                segment_index=0,
                row_index=0,
                dtype="torch.bfloat16",
                shape=(3,),
                bit_digest=digest("caller-claimed"),
            )
        with self.assertRaisesRegex(TypeError, "real tensor"):
            replace(logits_evidence()[0], bit_digest=digest("caller-claimed"))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            capture_logits_evidence(
                request_id="request:a",
                plan_id="batch:0",
                segment_index=0,
                row_index=0,
                logits=torch.tensor((1.0, float("nan")), dtype=torch.bfloat16),
            )
        with self.assertRaisesRegex(ValueError, "rank-1"):
            capture_logits_evidence(
                request_id="request:a",
                plan_id="batch:0",
                segment_index=0,
                row_index=0,
                logits=torch.ones((1, 3), dtype=torch.bfloat16),
            )

    def test_misbound_logits_evidence_fails_closed_but_empty_marks_bridge_incomplete(
        self,
    ) -> None:
        wrong_row = capture_logits_evidence(
            request_id="request:a",
            plan_id="batch:0",
            segment_index=0,
            row_index=0,
            logits=torch.tensor((1.0, 2.0, 3.0), dtype=torch.bfloat16),
        )
        with self.assertRaisesRegex(ValueError, "every output-producing"):
            record_trace(
                input_plans=(input_plan(),),
                batch_plans=(batch_plan(),),
                executor_factory=factory(
                    identity=evidence("reference:eager"),
                    observation=ReplayObservation(
                        events(),
                        terminal(),
                        (wrong_row,),
                    ),
                    calls=[],
                ),
            )

        trace = record_trace(
            input_plans=(input_plan(),),
            batch_plans=(batch_plan(),),
            executor_factory=factory(
                identity=evidence("reference:eager"),
                observation=ReplayObservation(events(), terminal()),
                calls=[],
            ),
        )
        result = replay_trace(
            trace,
            executor_factory=factory(
                identity=evidence("reference:eager"),
                observation=ReplayObservation(events(), terminal()),
                calls=[],
            ),
        )
        self.assertFalse(result.passed)
        self.assertFalse(result.logits_evidence_complete)
        self.assertFalse(result.logits_evidence_match)
        self.assertEqual(
            result.first_logits_mismatch_key,
            ("request:a", "batch:0", 0, 1),
        )

    def test_deserialization_rejects_noncanonical_unknown_tampered_and_invalid_data(self) -> None:
        encoded = self.make_trace().to_json()
        with self.assertRaisesRegex(ValueError, "canonical"):
            ReplayTrace.from_json(" " + encoded)

        unknown = json.loads(encoded)
        unknown["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "unknown"):
            ReplayTrace.from_json(self.canonical(unknown))

        tampered = json.loads(encoded)
        tampered["recorded_events"][0]["token_id"] = 43
        with self.assertRaisesRegex(ValueError, "trace_id"):
            ReplayTrace.from_json(self.canonical(tampered))

        invalid_contract = json.loads(encoded)
        invalid_contract["batch_plans"][0]["segments"][0]["token_count"] = 3
        self.reseal(invalid_contract)
        with self.assertRaisesRegex(ValueError, "every physical token"):
            ReplayTrace.from_json(self.canonical(invalid_contract))

        unknown_enum = json.loads(encoded)
        unknown_enum["execution"]["mode"] = "compiled"
        self.reseal(unknown_enum)
        with self.assertRaisesRegex(ValueError, "unknown ExecutionMode"):
            ReplayTrace.from_json(self.canonical(unknown_enum))

        old_schema = json.loads(encoded)
        old_schema["schema_version"] = 4
        self.reseal(old_schema)
        with self.assertRaisesRegex(ValueError, "schema version"):
            ReplayTrace.from_json(self.canonical(old_schema))

        invalid_logits = json.loads(encoded)
        invalid_logits["recorded_logits_evidence"][0]["dtype"] = "torch.int64"
        self.reseal(invalid_logits)
        with self.assertRaisesRegex(ValueError, "unsupported logits"):
            ReplayTrace.from_json(self.canonical(invalid_logits))

        duplicate_key = encoded.replace(
            '"schema":"cubic.guardrail.replay",',
            '"schema":"cubic.guardrail.replay","schema":"cubic.guardrail.replay",',
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ReplayTrace.from_json(duplicate_key)

    def test_observation_must_cover_terminal_state_and_contiguous_events(self) -> None:
        with self.assertRaisesRegex(ValueError, "cover every"):
            record_trace(
                input_plans=(input_plan(),),
                batch_plans=(batch_plan(),),
                executor_factory=factory(
                    identity=evidence("reference:eager"),
                    observation=ReplayObservation(events(), (), logits_evidence()),
                    calls=[],
                ),
            )

        broken_events = (
            TokenEvent(
                request_id="request:a",
                sequence_no=1,
                kind=TokenEventKind.TOKEN,
                token_id=42,
            ),
        )
        with self.assertRaisesRegex(ValueError, "contiguous"):
            record_trace(
                input_plans=(input_plan(),),
                batch_plans=(batch_plan(),),
                executor_factory=factory(
                    identity=evidence("reference:eager"),
                    observation=ReplayObservation(
                        broken_events,
                        terminal(),
                        logits_evidence(),
                    ),
                    calls=[],
                ),
            )

    def test_trace_binds_input_digest_and_observable_plan_lifecycle(self) -> None:
        source = input_plan()
        first = batch_plan()
        second = BatchPlan(
            plan_id="batch:1",
            phase=BatchPhase.DECODE,
            input_plan_ids=(source.request_id,),
            segments=(
                PlanSegment(
                    request_id=source.request_id,
                    input_plan_digest=source.semantic_digest,
                    execution_input_digest=source.semantic_digest,
                    phase=BatchPhase.DECODE,
                    produces_output=True,
                    token_start=2,
                    token_count=1,
                    token_ids=(42,),
                    positions=(source.decode_positions.at(0),),
                    cache_slots=(2,),
                    cache_pages=(0,),
                ),
            ),
            shape=BatchShape(logical_m=1),
            topology_id="single",
            partition_id="toy",
            state_transactions=(
                StateTxn(
                    transaction_id="txn:1",
                    request_id=source.request_id,
                    base_revision=1,
                    target_revision=2,
                    reads=(
                        StateAccess(
                            resource_id="kv:request:a",
                            owner_rank=0,
                            pages=(0,),
                        ),
                    ),
                    writes=(
                        StateAccess(
                            resource_id="kv:request:a",
                            owner_rank=0,
                            slots=(2,),
                            pages=(0,),
                        ),
                    ),
                    expected_owner_ranks=(0,),
                ),
            ),
        )
        second_logits = capture_logits_evidence(
            request_id=source.request_id,
            plan_id=second.plan_id,
            segment_index=0,
            row_index=0,
            logits=torch.tensor((7.0, 8.0, 9.0), dtype=torch.bfloat16),
        )
        completed_terminal = (replace(terminal()[0], state_revision=2),)
        trace_identity = evidence("reference:eager")
        trace = ReplayTrace(
            execution=trace_identity,
            input_plans=(source,),
            batch_plans=(first, second),
            recorded_events=events(),
            recorded_terminal_digests=completed_terminal,
            recorded_logits_evidence=(*logits_evidence(), second_logits),
            recorded_prefix_lifecycle=miss_lifecycle(
                trace_identity,
                (source,),
                (first, second),
            ),
        )

        misbound_segment = replace(
            second.segments[0],
            input_plan_digest=digest("different-input"),
        )
        with self.assertRaisesRegex(ValueError, "InputPlan semantic digest"):
            replace(
                trace,
                batch_plans=(
                    first,
                    replace(second, segments=(misbound_segment,)),
                ),
            )
        executor_calls: list[tuple[tuple[InputPlan, ...], tuple[BatchPlan, ...]]] = []
        with self.assertRaisesRegex(ValueError, "InputPlan semantic digest"):
            record_trace(
                input_plans=(source,),
                batch_plans=(
                    first,
                    replace(second, segments=(misbound_segment,)),
                ),
                executor_factory=factory(
                    identity=evidence("must-not-run"),
                    observation=ReplayObservation(
                        events(),
                        completed_terminal,
                        (*logits_evidence(), second_logits),
                    ),
                    calls=executor_calls,
                ),
            )
        self.assertEqual(executor_calls, [])

        gap_segment = replace(
            second.segments[0],
            token_start=3,
            positions=(source.decode_positions.at(1),),
            cache_slots=(3,),
        )
        with self.assertRaisesRegex(ValueError, "token spans.*contiguous"):
            replace(
                trace,
                batch_plans=(first, replace(second, segments=(gap_segment,))),
            )

        wrong_phase_segment = replace(second.segments[0], phase=BatchPhase.PREFILL)
        with self.assertRaisesRegex(ValueError, "prefill.*prompt span"):
            replace(
                trace,
                batch_plans=(
                    first,
                    replace(
                        second,
                        phase=BatchPhase.PREFILL,
                        segments=(wrong_phase_segment,),
                    ),
                ),
            )

        stale_revision = replace(
            second.state_transactions[0],
            base_revision=0,
            target_revision=1,
        )
        with self.assertRaisesRegex(ValueError, "state revisions.*contiguous"):
            replace(
                trace,
                batch_plans=(
                    first,
                    replace(second, state_transactions=(stale_revision,)),
                ),
            )

        with self.assertRaisesRegex(ValueError, "terminal state revision"):
            replace(
                trace,
                recorded_terminal_digests=(replace(completed_terminal[0], state_revision=1),),
            )

        tampered_json = json.loads(trace.to_json())
        tampered_json["batch_plans"][1]["segments"][0]["input_plan_digest"] = digest(
            "resealed-different-input"
        )
        self.reseal(tampered_json)
        with self.assertRaisesRegex(ValueError, "InputPlan semantic digest"):
            ReplayTrace.from_json(self.canonical(tampered_json))

    def test_nonzero_initial_resume_requires_recorded_prefix_hit(self) -> None:
        source = input_plan()
        resumed_segment = replace(
            batch_plan().segments[0],
            token_start=1,
            token_count=1,
            token_ids=(source.token_ids[1],),
            positions=(source.positions[1],),
            cache_slots=(1,),
        )
        resumed_txn = replace(
            batch_plan().state_transactions[0],
            base_revision=7,
            target_revision=8,
            writes=(
                StateAccess(
                    resource_id="kv:request:a",
                    owner_rank=0,
                    slots=(1,),
                    pages=(0,),
                ),
            ),
        )
        resumed_batch = replace(
            batch_plan(),
            segments=(resumed_segment,),
            shape=BatchShape(logical_m=1),
            state_transactions=(resumed_txn,),
        )
        resumed_logits = capture_logits_evidence(
            request_id=source.request_id,
            plan_id=resumed_batch.plan_id,
            segment_index=0,
            row_index=0,
            logits=torch.tensor((1.0, 2.0, 3.0), dtype=torch.bfloat16),
        )

        trace_identity = evidence("reference:eager")
        with self.assertRaisesRegex(ValueError, "resume boundary"):
            ReplayTrace(
                execution=trace_identity,
                input_plans=(source,),
                batch_plans=(resumed_batch,),
                recorded_events=events(),
                recorded_terminal_digests=(replace(terminal()[0], state_revision=8),),
                recorded_logits_evidence=(resumed_logits,),
                recorded_prefix_lifecycle=miss_lifecycle(
                    trace_identity,
                    (source,),
                    (resumed_batch,),
                ),
            )

    def test_full_prompt_publication_without_a_replay_row_permits_a_miss(self) -> None:
        execution = evidence("reference:eager")
        cold_input = replace(input_plan(), request_id="request:cold")
        warm_input = replace(input_plan(), request_id="request:warm")
        prefix_key = digest("shared-full-prompt-prefix")

        def request_plan(
            source: InputPlan,
            *,
            plan_id: str,
            state_commit_position: int | None,
        ) -> BatchPlan:
            template = batch_plan()
            segment = replace(
                template.segments[0],
                request_id=source.request_id,
                input_plan_digest=source.semantic_digest,
                execution_input_digest=source.semantic_digest,
            )
            transaction = template.state_transactions[0]
            transaction = replace(
                transaction,
                transaction_id=f"txn:{source.request_id}",
                request_id=source.request_id,
                reads=(
                    replace(
                        transaction.reads[0],
                        resource_id=f"kv:{source.request_id}",
                    ),
                ),
                writes=(
                    replace(
                        transaction.writes[0],
                        resource_id=f"kv:{source.request_id}",
                    ),
                ),
                prefix_publish=(
                    PrefixPublishPlan(
                        input_plan_id=source.request_id,
                        prefix_token_count=len(source.token_ids),
                        state_commit_position=state_commit_position,
                        commit_alignment_tokens=1,
                    )
                    if state_commit_position is not None
                    else None
                ),
            )
            return replace(
                template,
                plan_id=plan_id,
                input_plan_ids=(source.request_id,),
                segments=(segment,),
                state_transactions=(transaction,),
            )

        def make_trace(state_commit_position: int) -> ReplayTrace:
            cold_plan = request_plan(
                cold_input,
                plan_id="batch:cold",
                state_commit_position=state_commit_position,
            )
            warm_plan = request_plan(
                warm_input,
                plan_id="batch:warm",
                state_commit_position=None,
            )
            candidates = (PrefixCandidateEvidence(len(cold_input.token_ids), prefix_key),)
            admissions = tuple(
                PrefixAdmissionEvidence(
                    request_id=source.request_id,
                    after_plan_count=after_plan_count,
                    use_prefix_cache=True,
                    execution_fingerprint_digest=execution.digest,
                    candidates=candidates,
                    hit=False,
                    prefix_key=None,
                    physical_block_boundary=0,
                    arithmetic_state_commit_boundary=0,
                    state_commit_alignment=0,
                    base_revision=0,
                    source_request_id=None,
                    source_rank=None,
                    source_revision=None,
                    snapshot_semantic_digest=None,
                )
                for source, after_plan_count in ((cold_input, 0), (warm_input, 1))
            )
            publication = PrefixPublicationEvidence(
                request_id=cold_input.request_id,
                plan_id=cold_plan.plan_id,
                segment_index=0,
                after_plan_count=1,
                execution_fingerprint_digest=execution.digest,
                prefix_key=prefix_key,
                source_rank=0,
                source_revision=1,
                physical_block_boundary=len(cold_input.token_ids),
                arithmetic_state_commit_boundary=state_commit_position,
                state_commit_alignment=1,
                snapshot_semantic_digest=digest("shared-full-prompt-snapshot"),
                inserted=True,
            )
            return ReplayTrace(
                execution=execution,
                input_plans=(cold_input, warm_input),
                batch_plans=(cold_plan, warm_plan),
                recorded_events=(),
                recorded_terminal_digests=tuple(
                    TerminalDigest(
                        request_id=source.request_id,
                        owner_rank=0,
                        state_revision=1,
                        cache_digest=digest(f"cache:{source.request_id}"),
                        state_digest=digest(f"state:{source.request_id}"),
                    )
                    for source in (cold_input, warm_input)
                ),
                recorded_logits_evidence=tuple(
                    capture_logits_evidence(
                        request_id=source.request_id,
                        plan_id=plan.plan_id,
                        segment_index=0,
                        row_index=1,
                        logits=torch.tensor((1.0, 2.0), dtype=torch.bfloat16),
                    )
                    for source, plan in (
                        (cold_input, cold_plan),
                        (warm_input, warm_plan),
                    )
                ),
                recorded_prefix_lifecycle=PrefixLifecycleEvidence(
                    admissions,
                    (publication,),
                ),
            )

        trace = make_trace(state_commit_position=len(cold_input.token_ids))
        self.assertFalse(trace.recorded_prefix_lifecycle.admissions[1].hit)

        with self.assertRaisesRegex(ValueError, "earlier eligible publication"):
            make_trace(state_commit_position=len(cold_input.token_ids) - 1)

    @staticmethod
    def canonical(payload: object) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def reseal(cls, payload: dict[str, object]) -> None:
        body = {key: value for key, value in payload.items() if key != "trace_id"}
        payload["trace_id"] = (
            "b2s256:" + hashlib.blake2s(cls.canonical(body).encode("utf-8")).hexdigest()
        )


if __name__ == "__main__":
    unittest.main()
