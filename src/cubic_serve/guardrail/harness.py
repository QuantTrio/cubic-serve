"""One qualification harness shared by development and startup admission."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from torch import Tensor

from cubic_serve.guardrail.compare import (
    ComparisonContext,
    ComparisonPolicy,
    ComparisonResult,
    DifferenceClass,
    VerifiedWitness,
    compare_tensors,
    comparison_failure,
    stable_digest,
)
from cubic_serve.guardrail.evidence import (
    ComparisonRecord,
    EvidenceProfile,
    GuardrailEvidence,
    QualificationFingerprint,
)
from cubic_serve.guardrail.oracle import (
    OracleCallable,
    OracleSnapshot,
    ReductionWitnessMaterial,
    ScaleWitnessMaterial,
    clone_tensor_map,
    run_oracle,
    verify_reduction_witness,
    verify_scale_witness,
)


@dataclass(frozen=True, slots=True)
class TensorRule:
    """A per-output/state rule; exact is the only implicit/default policy."""

    policy: ComparisonPolicy = field(
        default_factory=lambda: ComparisonPolicy(DifferenceClass.EXACT)
    )
    material: ReductionWitnessMaterial | ScaleWitnessMaterial | None = None

    def __post_init__(self) -> None:
        if self.policy.difference_class is DifferenceClass.EXACT:
            if self.material is not None:
                raise ValueError("exact tensor rules cannot carry witness material")
        elif self.policy.difference_class is DifferenceClass.REDUCTION_ORDER:
            if not isinstance(self.material, ReductionWitnessMaterial):
                raise ValueError("reduction tensor rules require reduction material")
        elif not isinstance(self.material, ScaleWitnessMaterial):
            raise ValueError("scale tensor rules require scale material")


@dataclass(frozen=True, slots=True)
class QualificationCase:
    case_id: str
    semantic_op: str
    inputs: Mapping[str, Tensor]
    initial_state: Mapping[str, Tensor]
    rules: Mapping[str, TensorRule] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id or not self.semantic_op:
            raise ValueError("qualification case identity and semantic op are required")
        if any(not path.startswith(("output:", "state:")) for path in self.rules):
            raise ValueError("tensor rule paths must start with output: or state:")
        # Validate and snapshot names/types now. Runtime clones again to prevent a
        # candidate from sharing storage with either the oracle or case corpus.
        clone_tensor_map(self.inputs, "case input")
        clone_tensor_map(self.initial_state, "case state")


class QualificationHarness:
    """Run the independent oracle and candidate, then emit atomic-ready evidence."""

    def __init__(self, oracle: OracleCallable) -> None:
        self._oracle = oracle

    def qualify(
        self,
        *,
        profile: EvidenceProfile,
        fingerprint: QualificationFingerprint,
        case: QualificationCase,
        candidate: OracleCallable,
        candidate_id: str,
        launch_signature: str,
        stage: int = 0,
        rank: int = 0,
    ) -> GuardrailEvidence:
        if not candidate_id or not launch_signature:
            raise ValueError("candidate identity and launch signature are required")
        oracle_snapshot = run_oracle(self._oracle, case.inputs, case.initial_state)
        try:
            candidate_snapshot = _run_candidate(candidate, case.inputs, case.initial_state)
        except Exception as error:  # a crashing candidate is evidence of rejection
            result = comparison_failure(
                DifferenceClass.EXACT,
                0,
                f"candidate execution failed: {type(error).__name__}: {error}",
            )
            return GuardrailEvidence.now(
                profile=profile,
                fingerprint=fingerprint,
                records=(
                    _record(
                        case=case,
                        path="runtime",
                        candidate_id=candidate_id,
                        launch_signature=launch_signature,
                        stage=stage,
                        rank=rank,
                        result=result,
                    ),
                ),
            )

        context = ComparisonContext(
            op_signature_digest=fingerprint.op_signature_digest,
            candidate_launch_digest=stable_digest(
                {
                    "candidate_id": candidate_id,
                    "launch_signature": launch_signature,
                }
            ),
        )
        records = self._compare_snapshots(
            case=case,
            oracle=oracle_snapshot,
            candidate=candidate_snapshot,
            context=context,
            candidate_id=candidate_id,
            launch_signature=launch_signature,
            stage=stage,
            rank=rank,
        )
        return GuardrailEvidence.now(
            profile=profile,
            fingerprint=fingerprint,
            records=tuple(records),
        )

    def _compare_snapshots(
        self,
        *,
        case: QualificationCase,
        oracle: OracleSnapshot,
        candidate: OracleSnapshot,
        context: ComparisonContext,
        candidate_id: str,
        launch_signature: str,
        stage: int,
        rank: int,
    ) -> list[ComparisonRecord]:
        records: list[ComparisonRecord] = []
        observed_paths: set[str] = set()
        for prefix, expected_map, actual_map in (
            ("output", oracle.outputs, candidate.outputs),
            ("state", oracle.mutated_state, candidate.mutated_state),
        ):
            for name in sorted(set(expected_map) | set(actual_map)):
                path = f"{prefix}:{name}"
                observed_paths.add(path)
                rule = case.rules.get(path, TensorRule())
                expected = expected_map.get(name)
                actual = actual_map.get(name)
                if expected is None or actual is None:
                    present = "candidate" if expected is None else "oracle"
                    existing = actual if expected is None else expected
                    assert existing is not None
                    element_count = existing.numel()
                    result = comparison_failure(
                        rule.policy.difference_class,
                        element_count,
                        f"tensor exists only in {present} result",
                    )
                else:
                    result = _compare_one(expected, actual, rule, context)
                records.append(
                    _record(
                        case=case,
                        path=path,
                        candidate_id=candidate_id,
                        launch_signature=launch_signature,
                        stage=stage,
                        rank=rank,
                        result=result,
                    )
                )

        for path in sorted(set(case.rules) - observed_paths):
            records.append(
                _record(
                    case=case,
                    path=path,
                    candidate_id=candidate_id,
                    launch_signature=launch_signature,
                    stage=stage,
                    rank=rank,
                    result=comparison_failure(
                        case.rules[path].policy.difference_class,
                        0,
                        "tensor rule targets an output/state absent from both runs",
                    ),
                )
            )
        if not records:
            records.append(
                _record(
                    case=case,
                    path="empty",
                    candidate_id=candidate_id,
                    launch_signature=launch_signature,
                    stage=stage,
                    rank=rank,
                    result=comparison_failure(
                        DifferenceClass.EXACT, 0, "qualification case produced no tensors"
                    ),
                )
            )
        return records


def _run_candidate(
    candidate: OracleCallable,
    inputs: Mapping[str, Tensor],
    initial_state: Mapping[str, Tensor],
) -> OracleSnapshot:
    isolated_inputs = clone_tensor_map(inputs, "candidate input")
    isolated_state = clone_tensor_map(initial_state, "candidate state")
    outputs = candidate(isolated_inputs, isolated_state)
    return OracleSnapshot(
        outputs=clone_tensor_map(outputs, "candidate output"),
        mutated_state=clone_tensor_map(isolated_state, "candidate state"),
    )


def _compare_one(
    expected: Tensor,
    actual: Tensor,
    rule: TensorRule,
    context: ComparisonContext,
) -> ComparisonResult:
    try:
        witness: VerifiedWitness | None
        if rule.policy.difference_class is DifferenceClass.REDUCTION_ORDER:
            assert isinstance(rule.material, ReductionWitnessMaterial)
            witness = verify_reduction_witness(
                expected, actual, rule.policy, context, rule.material
            )
        elif rule.policy.difference_class is DifferenceClass.SCALE_DTYPE:
            assert isinstance(rule.material, ScaleWitnessMaterial)
            witness = verify_scale_witness(expected, actual, rule.policy, context, rule.material)
        else:
            witness = None
        return compare_tensors(
            expected,
            actual,
            rule.policy,
            context=context,
            witness=witness,
        )
    except Exception as error:
        return comparison_failure(
            rule.policy.difference_class,
            expected.numel(),
            f"comparison verification failed: {type(error).__name__}: {error}",
        )


def _record(
    *,
    case: QualificationCase,
    path: str,
    candidate_id: str,
    launch_signature: str,
    stage: int,
    rank: int,
    result: ComparisonResult,
) -> ComparisonRecord:
    return ComparisonRecord(
        case_id=f"{case.case_id}/{path}",
        semantic_op=case.semantic_op,
        stage=stage,
        rank=rank,
        candidate_id=candidate_id,
        launch_signature=launch_signature,
        result=result,
    )
