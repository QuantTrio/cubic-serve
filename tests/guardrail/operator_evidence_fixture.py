from __future__ import annotations

from dataclasses import replace

from cubic_serve.contracts import execution as ec
from cubic_serve.guardrail.compare import stable_digest
from cubic_serve.guardrail.operator_evidence import (
    op_signature_digest,
    phase_arithmetic_evidence_id,
    seal_composite_arithmetic_evidence,
    seal_operator_evidence,
)


def _sealed_test_leaf(
    binding: ec.OpBinding,
    namespace: str,
    *,
    topology_digest: str | None = None,
    weight_semantics_digest: str | None = None,
    source_digest: str | None = None,
) -> tuple[ec.OpBinding, ec.ArithmeticEvidenceSeal]:
    """Private tests-only issuer for downstream composition mechanics."""

    candidate = binding.candidate
    assert candidate is not None
    signature_id = op_signature_digest(binding.signature)
    required_id = stable_digest(("trusted-test-fixture", namespace, signature_id))
    topology = topology_digest or stable_digest(binding.signature.topology_id)
    weights = weight_semantics_digest or stable_digest(("test-weights", signature_id))
    source = source_digest or stable_digest(candidate.implementation_id)
    item = object.__new__(ec.ArithmeticEvidenceSeal)
    values: tuple[tuple[str, object], ...] = (
        ("binding", binding),
        ("fingerprint_key", stable_digest(("fingerprint", namespace, signature_id))),
        ("source_digest", source),
        ("binary_digest", stable_digest((candidate.backend.value, candidate.implementation_id))),
        ("device_capability", "test"),
        ("driver_version", "test"),
        ("runtime_version", "test"),
        ("topology_digest", topology),
        ("weight_semantics_digest", weights),
        ("required_evidence_ids", (required_id,)),
        ("evidence_id", stable_digest("pending-tests-only-leaf")),
    )
    for name, value in values:
        object.__setattr__(item, name, value)
    object.__setattr__(item, "evidence_id", ec.canonical_arithmetic_evidence_id(item))
    sealed_binding = replace(binding, candidate=replace(candidate, evidence_id=item.evidence_id))
    object.__setattr__(item, "binding", sealed_binding)
    assert ec.validate_arithmetic_evidence(item, sealed_binding)
    return sealed_binding, item


def _test_program_manifest(
    binding: ec.OpBinding,
    children: tuple[ec.OpBinding, ...],
    arithmetic: tuple[ec.ArithmeticEvidenceSeal, ...],
    *,
    occurrences: tuple[ec.ProgramOccurrence, ...] | None = None,
    external_input_buffer_ids: tuple[str, ...] | None = None,
    external_output_buffer_ids: tuple[str, ...] | None = None,
    program_source_digest: str | None = None,
    execution_recipe_digest: str | None = None,
    config_digest: str | None = None,
    topology_digest: str | None = None,
    partition_digest: str | None = None,
) -> ec.CompositeProgramManifest:
    inputs = (
        tuple(f"buffer:external:input:{tensor.name}" for tensor in binding.signature.inputs)
        if external_input_buffer_ids is None
        else external_input_buffer_ids
    )
    generated: list[ec.ProgramOccurrence] = []
    prior_outputs = inputs
    for index, child in enumerate(children):
        child_inputs = (
            inputs
            if index == 0
            else (prior_outputs + inputs)[: len(child.signature.inputs)]
        )
        outputs = tuple(
            f"buffer:{index}:output:{tensor.name}" for tensor in child.signature.outputs
        )
        generated.append(
            ec.ProgramOccurrence(
                occurrence_id=f"occurrence:{index}",
                model_node_id=f"node:{index}",
                layer_id=f"layer:{index}",
                specialization_ordinal=index,
                input_buffer_ids=child_inputs,
                output_buffer_ids=outputs,
                weight_live_binding_digests=(stable_digest(("live-weight", index)),),
                state_read_effect_ids=(f"state-read:{index}",),
                state_write_effect_ids=(f"state-write:{index}",),
                dependency_ids=()
                if index == 0 or not set(child_inputs).intersection(prior_outputs)
                else (f"occurrence:{index - 1}",),
            )
        )
        prior_outputs = outputs
    selected = tuple(generated) if occurrences is None else occurrences
    outputs = (
        selected[-1].output_buffer_ids
        if external_output_buffer_ids is None
        else external_output_buffer_ids
    )
    return ec.CompositeProgramManifest(
        specializations=tuple(
            ec.ProgramSpecialization(index, item.required_evidence_ids[0])
            for index, item in enumerate(arithmetic)
        ),
        occurrences=selected,
        external_input_buffer_ids=inputs,
        external_output_buffer_ids=outputs,
        program_source_digest=program_source_digest or stable_digest("program-source"),
        execution_recipe_digest=execution_recipe_digest or stable_digest("execution-recipe"),
        config_digest=config_digest or stable_digest("config"),
        topology_digest=topology_digest or stable_digest(children[0].signature.topology_id),
        partition_digest=partition_digest or stable_digest("partition"),
    )


def sealed_test_performance_plan(
    execution_plan_id: str,
    bindings: tuple[ec.OpBinding, ...],
    buckets: tuple[ec.GraphBucket, ...],
) -> ec.ExecutionPlan:
    """Use the private issuer solely to exercise downstream typed-plan mechanics."""

    sealed_bindings: list[ec.OpBinding] = []
    arithmetic = []
    for index, binding in enumerate(bindings):
        sealed_binding, item = _sealed_test_leaf(binding, f"plan:{index}")
        arithmetic.append(item)
        sealed_bindings.append(sealed_binding)
    sealed_arithmetic = tuple(arithmetic)
    sealed_buckets = tuple(
        replace(
            bucket,
            evidence_id=phase_arithmetic_evidence_id(bucket.phase, sealed_arithmetic),
        )
        for bucket in buckets
    )
    frozen_bindings = tuple(sealed_bindings)
    operator = seal_operator_evidence(
        execution_plan_id=execution_plan_id,
        bindings=frozen_bindings,
        graph_buckets=sealed_buckets,
        arithmetic=sealed_arithmetic,
    )
    return ec.ExecutionPlan(
        execution_plan_id=execution_plan_id,
        mode=ec.ExecutionMode.PERFORMANCE,
        bindings=frozen_bindings,
        graph_buckets=sealed_buckets,
        operator_evidence=operator,
    )


def sealed_test_composite_performance_plan(
    execution_plan_id: str,
    binding: ec.OpBinding,
    children: tuple[ec.OpBinding, ...],
    buckets: tuple[ec.GraphBucket, ...],
) -> ec.ExecutionPlan:
    """Tests-only full composite path; it cannot issue production leaf evidence."""

    sealed = tuple(
        _sealed_test_leaf(child, f"composite:{index}")
        for index, child in enumerate(children)
    )
    sealed_children = tuple(item[0] for item in sealed)
    leaf_arithmetic = tuple(item[1] for item in sealed)
    manifest = _test_program_manifest(binding, sealed_children, leaf_arithmetic)
    composite = seal_composite_arithmetic_evidence(
        binding=binding,
        children=sealed_children,
        arithmetic=leaf_arithmetic,
        manifest=manifest,
    )
    arithmetic: tuple[ec.ArithmeticEvidenceProjection, ...] = (composite,)
    sealed_buckets = tuple(
        replace(bucket, evidence_id=phase_arithmetic_evidence_id(bucket.phase, arithmetic))
        for bucket in buckets
    )
    bindings = (composite.binding,)
    operator = seal_operator_evidence(
        execution_plan_id=execution_plan_id,
        bindings=bindings,
        graph_buckets=sealed_buckets,
        arithmetic=arithmetic,
    )
    return ec.ExecutionPlan(
        execution_plan_id=execution_plan_id,
        mode=ec.ExecutionMode.PERFORMANCE,
        bindings=bindings,
        graph_buckets=sealed_buckets,
        operator_evidence=operator,
    )
