from __future__ import annotations

import unittest

from cubic_serve.models.weight_schema import (
    HeaderEvidenceKind,
    TensorHeader,
    WeightArchitectureIdentity,
    WeightNodeBinding,
    WeightSchemaCoverage,
    WeightSchemaError,
    _issue_weight_plan,
    tensor_headers_from_inventory,
)


class _FakeInventory:
    tensors: tuple[object, ...] = ()
    expected_names: tuple[str, ...] = ()
    inventory_digest = "b2s256:" + "0" * 64


class WeightSchemaInvariantTests(unittest.TestCase):
    _IDENTITY = WeightArchitectureIdentity(
        release_family="fixture",
        release_profile="fixture",
        model_type="fixture",
        graph_family="fixture",
        architecture="FixtureArchitecture",
        plan_status="architecture-only",
    )

    def test_model_without_mtp_has_a_closed_empty_extension(self) -> None:
        headers = (
            TensorHeader("head", "BF16", (2, 2)),
            TensorHeader("language", "BF16", (2, 2)),
            TensorHeader("vision", "BF16", (2, 2)),
        )
        plan = _issue_weight_plan(
            model_type="fixture",
            architecture_identity=self._IDENTITY,
            coverage=WeightSchemaCoverage.EXACT_CHECKPOINT_HEADERS,
            headers=headers,
            inventory=None,
            base_language_names=("language",),
            vision_names=("vision",),
            lm_head_names=("head",),
            mtp_extension_names=(),
            fp32_stability_names=(),
            int64_metadata_names=(),
            allowed_node_ids=("head", "language", "vision"),
            node_bindings=(
                WeightNodeBinding("head", ("head",)),
                WeightNodeBinding("language", ("language",)),
                WeightNodeBinding("vision", ("vision",)),
            ),
        )
        self.assertFalse(plan.mtp_extension_names)
        self.assertEqual(plan.header_evidence, HeaderEvidenceKind.HEADER_RECORDS_ONLY)
        self.assertIsNone(plan.inventory_digest)

    def test_dtype_exception_lists_are_exact_not_advisory(self) -> None:
        common = {
            "model_type": "fixture",
            "architecture_identity": self._IDENTITY,
            "coverage": WeightSchemaCoverage.EXACT_CHECKPOINT_HEADERS,
            "inventory": None,
            "base_language_names": ("language",),
            "vision_names": ("vision",),
            "lm_head_names": ("head",),
            "mtp_extension_names": (),
            "int64_metadata_names": (),
            "allowed_node_ids": ("head", "language", "vision"),
            "node_bindings": (
                WeightNodeBinding("head", ("head",)),
                WeightNodeBinding("language", ("language",)),
                WeightNodeBinding("vision", ("vision",)),
            ),
        }
        headers = (
            TensorHeader("head", "BF16", (2, 2)),
            TensorHeader("language", "F32", (2, 2)),
            TensorHeader("vision", "BF16", (2, 2)),
        )
        with self.assertRaisesRegex(WeightSchemaError, "expected BF16"):
            _issue_weight_plan(headers=headers, fp32_stability_names=(), **common)  # type: ignore[arg-type]
        bf16_headers = tuple(TensorHeader(header.name, "BF16", header.shape) for header in headers)
        with self.assertRaisesRegex(WeightSchemaError, "expected F32"):
            _issue_weight_plan(  # type: ignore[arg-type]
                headers=bf16_headers,
                fp32_stability_names=("language",),
                **common,
            )

    def test_semantic_digest_binds_partition_and_node_owner(self) -> None:
        headers = (
            TensorHeader("head", "BF16", (2, 2)),
            TensorHeader("language", "BF16", (2, 2)),
            TensorHeader("vision", "BF16", (2, 2)),
        )
        common = {
            "model_type": "fixture",
            "architecture_identity": self._IDENTITY,
            "coverage": WeightSchemaCoverage.EXACT_CHECKPOINT_HEADERS,
            "headers": headers,
            "inventory": None,
            "lm_head_names": ("head",),
            "mtp_extension_names": (),
            "fp32_stability_names": (),
            "int64_metadata_names": (),
            "allowed_node_ids": ("head", "language", "vision"),
        }
        bindings = (
            WeightNodeBinding("head", ("head",)),
            WeightNodeBinding("language", ("language",)),
            WeightNodeBinding("vision", ("vision",)),
        )
        baseline = _issue_weight_plan(  # type: ignore[arg-type]
            base_language_names=("language",),
            vision_names=("vision",),
            node_bindings=bindings,
            **common,
        )
        partition_swap = _issue_weight_plan(  # type: ignore[arg-type]
            base_language_names=("vision",),
            vision_names=("language",),
            node_bindings=bindings,
            **common,
        )
        node_swap = _issue_weight_plan(  # type: ignore[arg-type]
            base_language_names=("language",),
            vision_names=("vision",),
            node_bindings=(
                WeightNodeBinding("head", ("head",)),
                WeightNodeBinding("language", ("vision",)),
                WeightNodeBinding("vision", ("language",)),
            ),
            **common,
        )
        self.assertEqual(baseline.checkpoint_header_digest, partition_swap.checkpoint_header_digest)
        self.assertEqual(baseline.base_executable_digest, partition_swap.base_executable_digest)
        self.assertNotEqual(baseline.semantic_plan_digest, partition_swap.semantic_plan_digest)
        self.assertNotEqual(baseline.semantic_plan_digest, node_swap.semantic_plan_digest)

    def test_duck_typed_inventory_cannot_claim_attestation(self) -> None:
        fake = _FakeInventory()
        with self.assertRaisesRegex(WeightSchemaError, "core-issued CheckpointInventory"):
            tensor_headers_from_inventory(fake)  # type: ignore[arg-type]

        headers = (
            TensorHeader("head", "BF16", (2, 2)),
            TensorHeader("language", "BF16", (2, 2)),
            TensorHeader("vision", "BF16", (2, 2)),
        )
        with self.assertRaisesRegex(WeightSchemaError, "core-issued CheckpointInventory"):
            _issue_weight_plan(  # type: ignore[arg-type]
                model_type="fixture",
                architecture_identity=self._IDENTITY,
                coverage=WeightSchemaCoverage.EXACT_CHECKPOINT_HEADERS,
                headers=headers,
                inventory=fake,
                base_language_names=("language",),
                vision_names=("vision",),
                lm_head_names=("head",),
                mtp_extension_names=(),
                fp32_stability_names=(),
                int64_metadata_names=(),
                allowed_node_ids=("head", "language", "vision"),
                node_bindings=(
                    WeightNodeBinding("head", ("head",)),
                    WeightNodeBinding("language", ("language",)),
                    WeightNodeBinding("vision", ("vision",)),
                ),
            )


if __name__ == "__main__":
    unittest.main()
