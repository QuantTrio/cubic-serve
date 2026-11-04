from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cubic_serve.guardrail.compare import ComparisonResult, DifferenceClass
from cubic_serve.guardrail.evidence import (
    ComparisonRecord,
    EvidenceProfile,
    GuardrailEvidence,
    QualificationFingerprint,
)


class EvidenceTests(unittest.TestCase):
    def test_artifact_is_canonical_and_keyed_by_environment(self) -> None:
        fingerprint = QualificationFingerprint(
            contract_digest="b2s256:contract",
            source_digest="b2s256:source",
            binary_digest="b2s256:binary",
            op_signature_digest="b2s256:signature",
            implementation_id="reference.linear:v1",
            device_capability="sm_120",
            driver_version="595.71.05",
            runtime_version="torch:2.13.0+cu130",
            topology_digest="b2s256:topology",
        )
        evidence = GuardrailEvidence.now(
            profile=EvidenceProfile.FULL,
            fingerprint=fingerprint,
            records=(
                ComparisonRecord(
                    case_id="linear:m1",
                    semantic_op="linear",
                    stage=0,
                    rank=0,
                    candidate_id="reference.linear",
                    launch_signature="eager",
                    result=ComparisonResult(
                        passed=True,
                        difference_class=DifferenceClass.EXACT,
                        element_count=1,
                        first_mismatch_index=None,
                        max_absolute_error=0.0,
                        max_ulp_error=0,
                        max_allowed_error=0.0,
                        reason="admitted",
                    ),
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            evidence.write_atomic(path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertTrue(payload["passed"])
        self.assertEqual(payload["artifact_id"], evidence.artifact_id)
        self.assertEqual(payload["qualification_key"], fingerprint.key)
        self.assertEqual(payload["records"][0]["result"]["difference_class"], "exact")


if __name__ == "__main__":
    unittest.main()
