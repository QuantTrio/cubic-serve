"""Typed arithmetic admission.

At inspected pins llama.cpp c841aeeb8bb2fe417038dadfa9b007cf1a9ef950,
vLLM 73fac1e1597f8f7b4029956982e8653f0cf0f067, and SGLang
ba433bb4628f44d5031e910ad32194ad6c145ba8 bind graph dispatch to static
buffers/shapes. None attests the complete arithmetic leaf closure or its artifacts.
"""

from __future__ import annotations

from dataclasses import replace

from cubic_serve.contracts.execution import (
    ArithmeticEvidenceProjection,
    ArithmeticEvidenceSeal,
    BatchPhase,
    CandidateBinding,
    CompositeArithmeticEvidenceSeal,
    CompositeProgramManifest,
    GraphBucket,
    OpBinding,
    OperatorEvidenceSeal,
    OpSignature,
    canonical_op_signature_digest,
    canonical_phase_evidence_id,
    composite_evidence_material,
    operator_evidence_material,
    validate_composite_arithmetic_evidence,
    validate_operator_evidence,
)
from cubic_serve.guardrail.evidence import GuardrailEvidence
from cubic_serve.models.weights import SealedWeightSet


class ArithmeticEvidenceRejected(ValueError):
    """A candidate lacks exact, subject-bound arithmetic evidence."""


def op_signature_digest(signature: OpSignature) -> str:
    return canonical_op_signature_digest(signature)


def seal_arithmetic_evidence(
    *,
    signature: OpSignature,
    candidate: CandidateBinding,
    evidence: tuple[GuardrailEvidence, ...],
    sealed_weights: SealedWeightSet | None = None,
) -> ArithmeticEvidenceSeal:
    if type(signature) is not OpSignature or type(candidate) is not CandidateBinding:
        raise TypeError("arithmetic admission requires typed signature and candidate")
    if type(evidence) is not tuple or not evidence:
        raise ArithmeticEvidenceRejected("arithmetic admission requires immutable evidence")
    if any(type(item) is not GuardrailEvidence for item in evidence):
        raise ArithmeticEvidenceRejected(
            "dedicated and graph-padding witnesses are not eligible arithmetic evidence"
        )
    del sealed_weights
    raise ArithmeticEvidenceRejected(
        "GuardrailEvidence v1 omits case material, tensor bindings, and a required corpus"
    )


def phase_arithmetic_evidence_id(
    phase: BatchPhase, arithmetic: tuple[ArithmeticEvidenceProjection, ...]
) -> str:
    try:
        return canonical_phase_evidence_id(phase, arithmetic)
    except (TypeError, ValueError) as error:
        raise ArithmeticEvidenceRejected(str(error)) from error


def seal_composite_arithmetic_evidence(
    *,
    binding: OpBinding,
    children: tuple[OpBinding, ...],
    arithmetic: tuple[ArithmeticEvidenceSeal, ...],
    manifest: CompositeProgramManifest,
) -> CompositeArithmeticEvidenceSeal:
    """Project a composite candidate onto canonical, exact arithmetic leaves.

    This is an aggregator, not a production leaf issuer. Since production has no
    eligible leaf issuer yet, this cannot manufacture readiness by itself.
    """

    if type(binding) is not OpBinding:
        raise TypeError("composite arithmetic admission requires a typed top binding")
    if type(binding.candidate) is not CandidateBinding:
        raise ArithmeticEvidenceRejected("composite arithmetic requires a top candidate")
    if type(manifest) is not CompositeProgramManifest or any(
        type(items) is not tuple for items in (children, arithmetic)
    ):
        raise TypeError("composite arithmetic inputs must be immutable tuples")
    material = composite_evidence_material(binding, children, arithmetic, manifest)
    if material is None:
        raise ArithmeticEvidenceRejected("composite arithmetic requires a canonical closure")
    candidate = binding.candidate
    evidence_id = material
    sealed_binding = replace(binding, candidate=replace(candidate, evidence_id=evidence_id))

    seal = object.__new__(CompositeArithmeticEvidenceSeal)
    for name, value in (
        ("binding", sealed_binding),
        ("children", children),
        ("arithmetic", arithmetic),
        ("manifest", manifest),
        ("evidence_id", evidence_id),
    ):
        object.__setattr__(seal, name, value)
    if not validate_composite_arithmetic_evidence(seal, sealed_binding, children, manifest):
        raise ArithmeticEvidenceRejected("composite arithmetic factory validation failed")
    return seal


def seal_operator_evidence(
    *,
    execution_plan_id: str,
    bindings: tuple[OpBinding, ...],
    graph_buckets: tuple[GraphBucket, ...],
    arithmetic: tuple[ArithmeticEvidenceProjection, ...],
) -> OperatorEvidenceSeal:
    """Aggregate exact per-binding coverage and project it into every graph bucket."""

    if not execution_plan_id or not bindings or not graph_buckets:
        raise ArithmeticEvidenceRejected("operator evidence requires a complete graph plan")
    if any(type(items) is not tuple for items in (bindings, graph_buckets, arithmetic)):
        raise TypeError("operator evidence inputs must be immutable tuples")
    material = operator_evidence_material(execution_plan_id, bindings, graph_buckets, arithmetic)
    if material is None:
        raise ArithmeticEvidenceRejected("performance bindings and buckets require exact seals")
    phase_ids, evidence_id = material

    seal = object.__new__(OperatorEvidenceSeal)
    for name, value in (
        ("execution_plan_id", execution_plan_id),
        ("arithmetic", arithmetic),
        ("phase_evidence_ids", phase_ids),
        ("evidence_id", evidence_id),
    ):
        object.__setattr__(seal, name, value)
    if not validate_operator_evidence(seal, execution_plan_id, bindings, graph_buckets):
        raise ArithmeticEvidenceRejected("operator evidence factory validation failed")
    return seal


__all__ = (
    "ArithmeticEvidenceRejected",
    "op_signature_digest",
    "phase_arithmetic_evidence_id",
    "seal_arithmetic_evidence",
    "seal_composite_arithmetic_evidence",
    "seal_operator_evidence",
)
