from __future__ import annotations

import ast
import inspect
import unittest

import torch

import cubic_serve.guardrail.oracle as oracle_module
from cubic_serve.contracts.model import ScalarDType
from cubic_serve.guardrail.compare import (
    ComparisonPolicy,
    DifferenceClass,
    ReductionOrderProof,
    stable_digest,
)
from cubic_serve.guardrail.evidence import EvidenceProfile, QualificationFingerprint
from cubic_serve.guardrail.harness import (
    QualificationCase,
    QualificationHarness,
    TensorRule,
)
from cubic_serve.guardrail.oracle import (
    ReductionWitnessMaterial,
    reduction_material_identity,
)


def _fingerprint() -> QualificationFingerprint:
    return QualificationFingerprint(
        contract_digest=stable_digest("contract"),
        source_digest=stable_digest("source"),
        binary_digest=stable_digest("binary"),
        op_signature_digest=stable_digest("signature"),
        implementation_id="candidate.linear:v1",
        device_capability="cpu",
        driver_version="none",
        runtime_version="torch:test",
        topology_digest=stable_digest("topology"),
    )


def _oracle(inputs, state):
    state["counter"].add_(1.0)
    return {"y": inputs["x"] * 2.0}


class HarnessTests(unittest.TestCase):
    def test_development_and_startup_share_exact_harness(self) -> None:
        def candidate(inputs, state):
            state["counter"].add_(1.0)
            return {"y": inputs["x"] * 2.0}

        case = QualificationCase(
            case_id="linear/m1",
            semantic_op="linear",
            inputs={"x": torch.tensor([2.0])},
            initial_state={"counter": torch.tensor([0.0])},
        )
        harness = QualificationHarness(_oracle)

        for profile in (EvidenceProfile.FULL, EvidenceProfile.STARTUP):
            with self.subTest(profile=profile):
                evidence = harness.qualify(
                    profile=profile,
                    fingerprint=_fingerprint(),
                    case=case,
                    candidate=candidate,
                    candidate_id="candidate.linear",
                    launch_signature="block=1",
                )
                self.assertTrue(evidence.passed)
                self.assertEqual(
                    {record.case_id for record in evidence.records},
                    {"linear/m1/output:y", "linear/m1/state:counter"},
                )

    def test_output_match_does_not_hide_mutated_state_error(self) -> None:
        def bad_candidate(inputs, state):
            state["counter"].add_(2.0)
            return {"y": inputs["x"] * 2.0}

        evidence = QualificationHarness(_oracle).qualify(
            profile=EvidenceProfile.STARTUP,
            fingerprint=_fingerprint(),
            case=QualificationCase(
                case_id="stateful/decode",
                semantic_op="stateful_decode",
                inputs={"x": torch.tensor([2.0])},
                initial_state={"counter": torch.tensor([0.0])},
            ),
            candidate=bad_candidate,
            candidate_id="candidate.stateful",
            launch_signature="m=1",
        )

        self.assertFalse(evidence.passed)
        results = {record.case_id: record.result for record in evidence.records}
        self.assertTrue(results["stateful/decode/output:y"].passed)
        self.assertFalse(results["stateful/decode/state:counter"].passed)

    def test_reduction_rule_is_issued_inside_harness(self) -> None:
        operands = torch.tensor([[1e8, 1.0, -1e8]], dtype=torch.float32)
        indices = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        material = ReductionWitnessMaterial(
            operands,
            indices,
            operands[:, [0, 2, 1]],
            indices[:, [0, 2, 1]],
        )
        identity = reduction_material_identity(material)
        rule = TensorRule(
            policy=ComparisonPolicy(
                DifferenceClass.REDUCTION_ORDER,
                reduction=ReductionOrderProof(
                    operands_digest=identity.operands_digest,
                    indices_digest=identity.indices_digest,
                    reduction_length=identity.reduction_length,
                    accumulator_dtype=ScalarDType.FP32,
                ),
            ),
            material=material,
        )

        def sum_oracle(inputs, state):
            x = inputs["x"]
            return {"sum": (x[:, 0] + x[:, 1]) + x[:, 2]}

        def reordered_candidate(inputs, state):
            x = inputs["x"]
            return {"sum": (x[:, 0] + x[:, 2]) + x[:, 1]}

        evidence = QualificationHarness(sum_oracle).qualify(
            profile=EvidenceProfile.FULL,
            fingerprint=_fingerprint(),
            case=QualificationCase(
                case_id="reduce/order",
                semantic_op="reduce_sum",
                inputs={"x": operands},
                initial_state={},
                rules={"output:sum": rule},
            ),
            candidate=reordered_candidate,
            candidate_id="candidate.reduce",
            launch_signature="reverse",
        )

        self.assertTrue(evidence.passed)
        self.assertIsNotNone(evidence.records[0].result.witness_digest)

    def test_candidate_exception_is_a_rejection_record(self) -> None:
        def candidate(inputs, state):
            raise RuntimeError("kernel launch failed")

        evidence = QualificationHarness(_oracle).qualify(
            profile=EvidenceProfile.STARTUP,
            fingerprint=_fingerprint(),
            case=QualificationCase(
                case_id="crash",
                semantic_op="linear",
                inputs={"x": torch.tensor([1.0])},
                initial_state={"counter": torch.tensor([0.0])},
            ),
            candidate=candidate,
            candidate_id="candidate.crash",
            launch_signature="bad",
        )
        self.assertFalse(evidence.passed)
        self.assertIn("execution failed", evidence.records[0].result.reason)

    def test_oracle_has_no_production_operator_import(self) -> None:
        tree = ast.parse(inspect.getsource(oracle_module))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        self.assertFalse(
            any(module.startswith("cubic_serve.ops") for module in imported_modules)
        )


if __name__ == "__main__":
    unittest.main()
