from __future__ import annotations

import unittest
from dataclasses import replace
from typing import cast

from cubic_serve.contracts import execution as ec
from cubic_serve.contracts.execution import (
    ArithmeticEvidenceSeal,
    BatchPhase,
    CandidateBackend,
    CandidateBinding,
    CompositeArithmeticEvidenceSeal,
    ExecutionMode,
    ExecutionPlan,
    GraphBucket,
    GraphSafetyContract,
    OpBinding,
    OperatorEvidenceSeal,
    OpSignature,
    PaddingIsolation,
    TensorSpec,
)
from cubic_serve.contracts.model import ScalarDType
from cubic_serve.guardrail.compare import ComparisonResult, DifferenceClass, stable_digest
from cubic_serve.guardrail.evidence import (
    ComparisonRecord,
    EvidenceProfile,
    GuardrailEvidence,
    QualificationFingerprint,
)
from cubic_serve.guardrail.operator_evidence import (
    ArithmeticEvidenceRejected,
    op_signature_digest,
    seal_arithmetic_evidence,
    seal_composite_arithmetic_evidence,
    seal_operator_evidence,
)
from tests.guardrail.operator_evidence_fixture import (
    _sealed_test_leaf,
    _test_program_manifest,
    sealed_test_composite_performance_plan,
    sealed_test_performance_plan,
)


def _signature(semantic_op: str = "pointwise") -> OpSignature:
    return OpSignature(
        semantic_op=semantic_op,
        phase=BatchPhase.DECODE,
        inputs=(TensorSpec("x", ScalarDType.BF16, ("m", 4)),),
        outputs=(TensorSpec("y", ScalarDType.BF16, ("m", 4)),),
        accumulator_dtype=ScalarDType.FP32,
        layout="row",
        shape_items=(),
        cache_form=None,
        state_form=None,
        topology_id="single",
    )


def _binding(signature: OpSignature | None = None) -> OpBinding:
    selected = signature or _signature()
    safety = GraphSafetyContract(
        contract_id=f"graph-safe:{selected.semantic_op}",
        row_map_schema_id="rows.v1",
        isolation=PaddingIsolation.PRIVATE_SCRATCH,
        evidence_id="graph-padding-is-not-arithmetic",
    )
    return OpBinding(
        signature=selected,
        candidate=CandidateBinding(
            candidate_id=f"candidate:{selected.semantic_op}",
            implementation_id=f"implementation:{selected.semantic_op}",
            backend=CandidateBackend.TRITON,
            launch_signature="grid=m/block",
            workspace_bytes=0,
            evidence_id="caller-string-is-not-trusted",
            graph_safety=safety,
        ),
    )


def _bucket(*contracts: str) -> GraphBucket:
    return GraphBucket(
        bucket_id="decode:4",
        template_id="graph:decode:4",
        scope_id="model.forward",
        replay_abi_id="decode.v1",
        topology_id="single",
        partition_id="single",
        phase=BatchPhase.DECODE,
        capacity_items=(("request_slots", 4),),
        logical_m_values=(1, 2, 4),
        captured_m=4,
        row_map_schema_id="rows.v1",
        scratch_schema_id="scratch.v1",
        graph_safety_contract_ids=tuple(sorted(contracts or ("graph-safe:pointwise",))),
        evidence_id="caller-bucket-string-is-not-trusted",
    )


def _legacy_evidence(binding: OpBinding) -> GuardrailEvidence:
    candidate = binding.candidate
    assert candidate is not None
    fingerprint = QualificationFingerprint(
        contract_digest=stable_digest("contract"),
        source_digest=stable_digest("source"),
        binary_digest=stable_digest("binary"),
        op_signature_digest=op_signature_digest(binding.signature),
        implementation_id=candidate.implementation_id,
        device_capability="sm120",
        driver_version="driver:test",
        runtime_version="runtime:test",
        topology_digest=stable_digest("single"),
    )
    return GuardrailEvidence.now(
        profile=EvidenceProfile.STARTUP,
        fingerprint=fingerprint,
        records=(
            ComparisonRecord(
                case_id="all-zero/output:y",
                semantic_op=binding.signature.semantic_op,
                stage=0,
                rank=0,
                candidate_id=candidate.candidate_id,
                launch_signature=candidate.launch_signature,
                result=ComparisonResult(
                    passed=True,
                    difference_class=DifferenceClass.EXACT,
                    element_count=4,
                    first_mismatch_index=None,
                    max_absolute_error=0.0,
                    max_ulp_error=0,
                    max_allowed_error=0.0,
                    reason="exact for this unbound case",
                ),
            ),
        ),
    )


class OperatorEvidenceTests(unittest.TestCase):
    def test_evidence_types_cannot_be_manually_constructed(self) -> None:
        with self.assertRaisesRegex(TypeError, "guardrail"):
            ArithmeticEvidenceSeal()
        with self.assertRaisesRegex(TypeError, "guardrail"):
            CompositeArithmeticEvidenceSeal()
        with self.assertRaisesRegex(TypeError, "guardrail"):
            OperatorEvidenceSeal()

    def test_contract_exposes_no_blind_issuer_or_importable_token(self) -> None:
        for name in (
            "_OPERATOR_EVIDENCE_TOKEN",
            "_issue_arithmetic_evidence",
            "_issue_composite_arithmetic_evidence",
            "_issue_operator_evidence",
        ):
            self.assertFalse(hasattr(ec, name), name)

    def test_fake_subclasses_cannot_override_admission(self) -> None:
        plan = sealed_test_composite_performance_plan(
            "performance:composite",
            _binding(_signature("qwen.decode")),
            (_binding(_signature("attention")), _binding(_signature("linear"))),
            (_bucket("graph-safe:qwen.decode"),),
        )
        operator = plan.operator_evidence
        assert operator is not None
        composite = operator.arithmetic[0]
        assert type(composite) is CompositeArithmeticEvidenceSeal
        leaf = composite.arithmetic[0]

        class FakeLeaf(ArithmeticEvidenceSeal):
            def covers(self, _binding: object) -> bool:
                return True

        class FakeComposite(CompositeArithmeticEvidenceSeal):
            def covers(self, _binding: object) -> bool:
                return True

        class FakeOperator(OperatorEvidenceSeal):
            def covers(self, *_values: object) -> bool:
                return True

        fake_leaf = object.__new__(FakeLeaf)
        for name in ArithmeticEvidenceSeal.__annotations__:
            object.__setattr__(fake_leaf, name, getattr(leaf, name))
        fake_composite = object.__new__(FakeComposite)
        for name in CompositeArithmeticEvidenceSeal.__annotations__:
            object.__setattr__(fake_composite, name, getattr(composite, name))
        fake_operator = object.__new__(FakeOperator)
        for name in OperatorEvidenceSeal.__annotations__:
            object.__setattr__(fake_operator, name, getattr(operator, name))
        self.assertFalse(ec.validate_arithmetic_evidence(fake_leaf, composite.children[0]))
        self.assertFalse(
            ec.validate_composite_arithmetic_evidence(fake_composite, plan.bindings[0])
        )
        self.assertFalse(
            ec.validate_operator_evidence(
                fake_operator, plan.execution_plan_id, plan.bindings, plan.graph_buckets
            )
        )
        object.__setattr__(composite, "evidence_id", stable_digest("mutated-composite"))
        self.assertFalse(
            ec.validate_composite_arithmetic_evidence(composite, plan.bindings[0])
        )

    def test_scalar_subclasses_cannot_reuse_typed_evidence(self) -> None:
        class EvilStr(str):
            __hash__ = str.__hash__

            def __eq__(self, _other: object) -> bool:
                return True

            def __ne__(self, _other: object) -> bool:
                return False

        class EvilInt(int):
            def __eq__(self, _other: object) -> bool:
                return True

        plan = sealed_test_performance_plan(
            "performance:exact-scalars", (_binding(),), (_bucket(),)
        )
        operator = plan.operator_evidence
        candidate = plan.bindings[0].candidate
        assert operator is not None and candidate is not None
        leaf = operator.arithmetic[0]
        with self.assertRaises(ValueError):
            replace(plan.bindings[0].signature, semantic_op=EvilStr("foreign.semantic"))
        with self.assertRaises(ValueError):
            replace(candidate, workspace_bytes=EvilInt(0))
        foreign_signature, foreign_candidate = replace(plan.bindings[0].signature), replace(
            candidate
        )
        foreign = replace(
            plan.bindings[0], signature=foreign_signature, candidate=foreign_candidate
        )
        object.__setattr__(foreign_signature, "semantic_op", EvilStr("foreign.semantic"))
        object.__setattr__(foreign_candidate, "candidate_id", EvilStr("foreign.candidate"))
        wrong_workspace_candidate = replace(candidate)
        wrong_workspace = replace(plan.bindings[0], candidate=wrong_workspace_candidate)
        object.__setattr__(wrong_workspace_candidate, "workspace_bytes", EvilInt(0))
        for binding in (foreign, wrong_workspace):
            with self.subTest(binding=binding):
                self.assertFalse(ec.validate_arithmetic_evidence(leaf, binding))
                with self.assertRaisesRegex(ArithmeticEvidenceRejected, "exact seals"):
                    seal_operator_evidence(
                        execution_plan_id="performance:foreign",
                        bindings=(binding,),
                        graph_buckets=plan.graph_buckets,
                        arithmetic=(leaf,),
                    )

    def test_mutated_leaf_phase_and_operator_id_fail_recomputed_validation(self) -> None:
        leaf_plan = sealed_test_performance_plan(
            "performance:leaf", (_binding(),), (_bucket(),)
        )
        leaf_operator = leaf_plan.operator_evidence
        assert leaf_operator is not None
        leaf = leaf_operator.arithmetic[0]
        assert type(leaf) is ArithmeticEvidenceSeal
        object.__setattr__(leaf, "source_digest", stable_digest("mutated-source"))
        self.assertFalse(ec.validate_arithmetic_evidence(leaf, leaf_plan.bindings[0]))
        self.assertFalse(
            ec.validate_operator_evidence(
                leaf_operator,
                leaf_plan.execution_plan_id,
                leaf_plan.bindings,
                leaf_plan.graph_buckets,
            )
        )

        for field, value in (
            ("phase_evidence_ids", ((BatchPhase.DECODE, stable_digest("fake-phase")),)),
            ("evidence_id", stable_digest("fake-operator")),
        ):
            plan = sealed_test_performance_plan(
                f"performance:{field}", (_binding(),), (_bucket(),)
            )
            seal = plan.operator_evidence
            assert seal is not None
            object.__setattr__(seal, field, value)
            self.assertFalse(
                ec.validate_operator_evidence(
                    seal, plan.execution_plan_id, plan.bindings, plan.graph_buckets
                )
            )

    def test_legacy_exact_evidence_is_not_a_seal(self) -> None:
        binding = _binding()
        candidate = binding.candidate
        assert candidate is not None
        with self.assertRaisesRegex(ArithmeticEvidenceRejected, "case material"):
            seal_arithmetic_evidence(
                signature=binding.signature,
                candidate=candidate,
                evidence=(_legacy_evidence(binding),),
            )

    def test_graph_or_dedicated_witness_cannot_masquerade_as_arithmetic(self) -> None:
        binding = _binding()
        candidate = binding.candidate
        assert candidate is not None
        foreign = cast(tuple[GuardrailEvidence, ...], (object(),))
        with self.assertRaisesRegex(ArithmeticEvidenceRejected, "graph-padding"):
            seal_arithmetic_evidence(
                signature=binding.signature,
                candidate=candidate,
                evidence=foreign,
            )

    def test_caller_strings_cannot_construct_a_performance_plan(self) -> None:
        with self.assertRaisesRegex(ValueError, "typed arithmetic evidence"):
            ExecutionPlan(
                execution_plan_id="performance:unsealed",
                mode=ExecutionMode.PERFORMANCE,
                bindings=(_binding(),),
                graph_buckets=(_bucket(),),
            )

    def test_typed_aggregate_exactly_covers_binding_and_bucket(self) -> None:
        plan = sealed_test_performance_plan("performance:sealed", (_binding(),), (_bucket(),))
        seal = plan.operator_evidence
        assert seal is not None
        self.assertTrue(
            ec.validate_operator_evidence(
                seal, plan.execution_plan_id, plan.bindings, plan.graph_buckets
            )
        )
        self.assertEqual(plan.bindings[0].candidate.evidence_id, seal.arithmetic[0].evidence_id)  # type: ignore[union-attr]
        self.assertEqual(plan.graph_buckets[0].evidence_id, seal.phase_evidence_ids[0][1])

    def test_candidate_or_bucket_mutation_invalidates_typed_coverage(self) -> None:
        plan = sealed_test_performance_plan("performance:sealed", (_binding(),), (_bucket(),))
        candidate = plan.bindings[0].candidate
        assert candidate is not None
        changed_binding = replace(
            plan.bindings[0], candidate=replace(candidate, launch_signature="other-grid")
        )
        with self.assertRaisesRegex(ValueError, "typed arithmetic evidence"):
            replace(plan, bindings=(changed_binding,))
        changed_bucket = replace(plan.graph_buckets[0], evidence_id=stable_digest("other"))
        with self.assertRaisesRegex(ValueError, "typed arithmetic evidence"):
            replace(plan, graph_buckets=(changed_bucket,))

    def test_post_init_graph_bucket_mutations_are_rejected(self) -> None:
        for field, value in (
            ("captured_m", 0),
            ("logical_m_values", (4,)),
            ("logical_m_values", (2, 1, 4)),
            ("capacity_items", (("request_slots", 0),)),
        ):
            with self.subTest(field=field, value=value):
                plan = sealed_test_performance_plan(
                    f"performance:bucket:{field}", (_binding(),), (_bucket(),)
                )
                operator = plan.operator_evidence
                assert operator is not None
                bucket = plan.graph_buckets[0]
                object.__setattr__(bucket, field, value)
                self.assertFalse(
                    ec.validate_operator_evidence(
                        operator,
                        plan.execution_plan_id,
                        plan.bindings,
                        plan.graph_buckets,
                    )
                )
                with self.assertRaisesRegex(ArithmeticEvidenceRejected, "exact seals"):
                    seal_operator_evidence(
                        execution_plan_id=plan.execution_plan_id,
                        bindings=plan.bindings,
                        graph_buckets=plan.graph_buckets,
                        arithmetic=operator.arithmetic,
                    )
                with self.assertRaisesRegex(ValueError, "exact canonical graph buckets"):
                    ExecutionPlan(
                        plan.execution_plan_id,
                        ExecutionMode.PERFORMANCE,
                        plan.bindings,
                        plan.graph_buckets,
                        operator,
                    )

    def test_aggregate_rejects_missing_binding_or_bucket_phase_coverage(self) -> None:
        plan = sealed_test_performance_plan("performance:sealed", (_binding(),), (_bucket(),))
        seal = plan.operator_evidence
        assert seal is not None
        second = _binding(_signature("pointwise.second"))
        with self.assertRaisesRegex(ArithmeticEvidenceRejected, "exact seals"):
            seal_operator_evidence(
                execution_plan_id=plan.execution_plan_id,
                bindings=(plan.bindings[0], second),
                graph_buckets=plan.graph_buckets,
                arithmetic=seal.arithmetic,
            )
        wrong_bucket = replace(plan.graph_buckets[0], evidence_id=stable_digest("not-phase"))
        with self.assertRaisesRegex(ArithmeticEvidenceRejected, "exact seals"):
            seal_operator_evidence(
                execution_plan_id=plan.execution_plan_id,
                bindings=plan.bindings,
                graph_buckets=(wrong_bucket,),
                arithmetic=seal.arithmetic,
            )

    def test_composite_projection_covers_top_binding_and_complete_bucket_phase(self) -> None:
        top = _binding(_signature("qwen.decode"))
        children = (_binding(_signature("attention")), _binding(_signature("linear")))
        plan = sealed_test_composite_performance_plan(
            "performance:composite",
            top,
            children,
            (_bucket("graph-safe:qwen.decode"),),
        )
        operator = plan.operator_evidence
        assert operator is not None
        projection = operator.arithmetic[0]
        self.assertIs(type(projection), CompositeArithmeticEvidenceSeal)
        assert type(projection) is CompositeArithmeticEvidenceSeal
        self.assertTrue(
            ec.validate_composite_arithmetic_evidence(
                projection,
                plan.bindings[0],
                projection.children,
                projection.manifest,
            )
        )
        self.assertEqual(plan.bindings[0], projection.binding)
        self.assertEqual(plan.bindings[0].candidate.evidence_id, projection.evidence_id)  # type: ignore[union-attr]
        self.assertEqual(
            plan.graph_buckets[0].evidence_id,
            operator.phase_evidence_ids[0][1],
        )

    def test_composite_rejects_noncanonical_or_inexact_child_mapping(self) -> None:
        top = _binding(_signature("qwen.decode"))
        first, first_evidence = _sealed_test_leaf(
            _binding(_signature("attention")), "first"
        )
        second, second_evidence = _sealed_test_leaf(_binding(_signature("linear")), "second")
        second_candidate = second.candidate
        assert second_candidate is not None
        with self.assertRaisesRegex(ArithmeticEvidenceRejected, "canonical closure"):
            seal_composite_arithmetic_evidence(
                binding=top,
                children=(first,),
                arithmetic=(first_evidence,),
                manifest=_test_program_manifest(top, (first,), (first_evidence,)),
            )
        for children, arithmetic in (
            ((second, first), (second_evidence, first_evidence)),
            ((first, first), (first_evidence, first_evidence)),
            ((first, second, _binding(_signature("norm"))), (first_evidence, second_evidence)),
            (
                (
                    first,
                    replace(
                        second,
                        candidate=replace(second_candidate, launch_signature="wrong-launch"),
                    ),
                ),
                (first_evidence, second_evidence),
            ),
        ):
            with self.subTest(children=children), self.assertRaisesRegex(
                ArithmeticEvidenceRejected, "canonical closure"
            ):
                seal_composite_arithmetic_evidence(
                    binding=top,
                    children=children,
                    arithmetic=arithmetic,
                    manifest=_test_program_manifest(top, children, arithmetic),
                )

    def test_composite_expected_closure_compares_every_full_abi_field(self) -> None:
        plan = sealed_test_composite_performance_plan(
            "performance:composite",
            _binding(_signature("qwen.decode")),
            (_binding(_signature("attention")), _binding(_signature("linear"))),
            (_bucket("graph-safe:qwen.decode"),),
        )
        operator = plan.operator_evidence
        assert operator is not None
        projection = operator.arithmetic[0]
        assert type(projection) is CompositeArithmeticEvidenceSeal
        original, second = projection.children
        input_spec, output_spec = original.signature.inputs[0], original.signature.outputs[0]
        mutations = (
            replace(original.signature, semantic_op="attention.changed"),
            replace(original.signature, phase=BatchPhase.PREFILL),
            replace(
                original.signature,
                inputs=(replace(input_spec, dtype=ScalarDType.FP16),),
            ),
            replace(original.signature, inputs=(replace(input_spec, shape=("m", 8)),)),
            replace(
                original.signature,
                outputs=(replace(output_spec, dtype=ScalarDType.FP16),),
            ),
            replace(original.signature, outputs=(replace(output_spec, shape=("m", 8)),)),
            replace(original.signature, accumulator_dtype=ScalarDType.BF16),
            replace(original.signature, layout="column"),
            replace(original.signature, cache_form="paged-bf16"),
            replace(original.signature, state_form="recurrent-bf16"),
            replace(original.signature, topology_id="other-topology"),
            replace(original.signature, shape_items=(("head_dim", 4),)),
        )
        for signature in mutations:
            with self.subTest(field=signature):
                self.assertFalse(
                    ec.validate_composite_arithmetic_evidence(
                        projection,
                        plan.bindings[0],
                        (replace(original, signature=signature), second),
                        projection.manifest,
                    )
                )
        candidate = original.candidate
        assert candidate is not None
        self.assertFalse(
            ec.validate_composite_arithmetic_evidence(
                projection,
                plan.bindings[0],
                (replace(original, candidate=replace(candidate, candidate_id="wrong")), second),
                projection.manifest,
            )
        )
        self.assertFalse(
            ec.validate_composite_arithmetic_evidence(
                projection, plan.bindings[0], (original,), projection.manifest
            )
        )
        self.assertFalse(
            ec.validate_composite_arithmetic_evidence(
                projection, plan.bindings[0], (second, original), projection.manifest
            )
        )
        self.assertFalse(
            ec.validate_composite_arithmetic_evidence(
                projection,
                plan.bindings[0],
                (*projection.children, original),
                projection.manifest,
            )
        )

    def test_composite_identity_derives_topology_weights_and_artifacts_from_leaves(self) -> None:
        top = _binding(_signature("qwen.decode"))
        first, first_evidence = _sealed_test_leaf(
            _binding(_signature("attention")), "first"
        )
        second_binding = _binding(_signature("linear"))
        second, second_evidence = _sealed_test_leaf(second_binding, "second")
        baseline = seal_composite_arithmetic_evidence(
            binding=top,
            children=(first, second),
            arithmetic=(first_evidence, second_evidence),
            manifest=_test_program_manifest(
                top, (first, second), (first_evidence, second_evidence)
            ),
        )
        changed_weight, changed_weight_evidence = _sealed_test_leaf(
            second_binding,
            "second-weight",
            weight_semantics_digest=stable_digest("other-weights"),
        )
        weight_projection = seal_composite_arithmetic_evidence(
            binding=top,
            children=(first, changed_weight),
            arithmetic=(first_evidence, changed_weight_evidence),
            manifest=_test_program_manifest(
                top, (first, changed_weight), (first_evidence, changed_weight_evidence)
            ),
        )
        changed_source, changed_source_evidence = _sealed_test_leaf(
            second_binding,
            "second-source",
            source_digest=stable_digest("other-source"),
        )
        source_projection = seal_composite_arithmetic_evidence(
            binding=top,
            children=(first, changed_source),
            arithmetic=(first_evidence, changed_source_evidence),
            manifest=_test_program_manifest(
                top, (first, changed_source), (first_evidence, changed_source_evidence)
            ),
        )
        self.assertNotEqual(baseline.evidence_id, weight_projection.evidence_id)
        self.assertNotEqual(baseline.evidence_id, source_projection.evidence_id)
        foreign, foreign_evidence = _sealed_test_leaf(
            second_binding,
            "second-topology",
            topology_digest=stable_digest("other-topology"),
        )
        with self.assertRaisesRegex(ArithmeticEvidenceRejected, "canonical closure"):
            seal_composite_arithmetic_evidence(
                binding=top,
                children=(first, foreign),
                arithmetic=(first_evidence, foreign_evidence),
                manifest=_test_program_manifest(
                    top, (first, foreign), (first_evidence, foreign_evidence)
                ),
            )

    def test_occurrence_and_top_manifest_drift_never_matches_the_sealed_program(self) -> None:
        plan = sealed_test_composite_performance_plan(
            "performance:manifest",
            _binding(_signature("qwen.decode")),
            (_binding(_signature("attention")), _binding(_signature("linear"))),
            (_bucket("graph-safe:qwen.decode"),),
        )
        operator = plan.operator_evidence
        assert operator is not None
        baseline = operator.arithmetic[0]
        assert type(baseline) is CompositeArithmeticEvidenceSeal
        occurrences = baseline.manifest.occurrences

        def make_manifest(
            *, occurrences: tuple[ec.ProgramOccurrence, ...] | None = None, **kwargs: object
        ) -> ec.CompositeProgramManifest:
            return _test_program_manifest(
                baseline.binding,
                baseline.children,
                baseline.arithmetic,
                occurrences=occurrences,
                **kwargs,  # type: ignore[arg-type]
            )

        first, second = occurrences
        specializations = baseline.manifest.specializations
        invalid = (
            make_manifest(occurrences=occurrences[:-1]),
            make_manifest(occurrences=tuple(reversed(occurrences))),
            make_manifest(topology_digest=stable_digest("other-topology")),
            make_manifest(
                occurrences=(replace(first, input_buffer_ids=("buffer:unproduced",)), second)
            ),
            make_manifest(occurrences=(first, replace(second, dependency_ids=()))),
            make_manifest(
                occurrences=(
                    first,
                    replace(second, output_buffer_ids=first.output_buffer_ids),
                )
            ),
            make_manifest(external_output_buffer_ids=("buffer:unproduced",)),
            replace(
                baseline.manifest,
                specializations=(
                    *specializations,
                    ec.ProgramSpecialization(1, stable_digest("unreferenced-specialization")),
                ),
            ),
            replace(
                baseline.manifest,
                specializations=(
                    replace(specializations[0], leaf_ordinal=len(baseline.children)),
                    specializations[1],
                ),
            ),
            replace(
                baseline.manifest,
                specializations=(
                    replace(
                        specializations[0], artifact_launch_digest=stable_digest("other-artifact")
                    ),
                    specializations[1],
                ),
            ),
        )
        for manifest in invalid:
            with self.subTest(invalid=manifest), self.assertRaises(
                ArithmeticEvidenceRejected
            ):
                seal_composite_arithmetic_evidence(
                    binding=baseline.binding,
                    children=baseline.children,
                    arithmetic=baseline.arithmetic,
                    manifest=manifest,
                )
        occurrence_variants = (
            (
                replace(
                    first,
                    weight_live_binding_digests=(stable_digest("other-live-weight"),),
                ),
                second,
            ),
            (replace(first, state_write_effect_ids=("state-write:changed",)), second),
        )
        manifests = (
            *(make_manifest(occurrences=items) for items in occurrence_variants),
            make_manifest(program_source_digest=stable_digest("other-source")),
            make_manifest(execution_recipe_digest=stable_digest("other-recipe")),
            make_manifest(config_digest=stable_digest("other-config")),
            make_manifest(partition_digest=stable_digest("other-partition")),
        )
        for manifest in manifests:
            projection = seal_composite_arithmetic_evidence(
                binding=baseline.binding,
                children=baseline.children,
                arithmetic=baseline.arithmetic,
                manifest=manifest,
            )
            self.assertNotEqual(baseline.evidence_id, projection.evidence_id)
            self.assertFalse(
                ec.validate_composite_arithmetic_evidence(
                    projection,
                    projection.binding,
                    baseline.children,
                    baseline.manifest,
                )
            )
