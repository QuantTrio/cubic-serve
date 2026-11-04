from __future__ import annotations

import unittest

from cubic_serve.contracts.execution import BatchPhase, OpSignature, TensorSpec
from cubic_serve.contracts.model import ScalarDType
from cubic_serve.ops.catalog import (
    ReferenceCatalog,
    ReferenceImplementation,
    builtin_reference_catalog,
    resolve_reference_binding,
    seal_reference_plan,
)


def _signature(semantic_op: str) -> OpSignature:
    return OpSignature(
        semantic_op=semantic_op,
        phase=BatchPhase.DECODE,
        inputs=(
            TensorSpec("inputs", ScalarDType.BF16, ("m", "hidden")),
            TensorSpec("weight", ScalarDType.BF16, ("hidden", "hidden")),
        ),
        outputs=(TensorSpec("result", ScalarDType.BF16, ("m", "hidden")),),
        accumulator_dtype=ScalarDType.FP32,
        layout="fixture",
        shape_items=(("hidden", 1), ("m", 1)),
        cache_form=None,
        state_form=None,
        topology_id="single",
    )


class ReferenceCatalogTests(unittest.TestCase):
    def test_missing_reference_semantics_reject_readiness(self) -> None:
        with self.assertRaisesRegex(LookupError, "missing reference implementation"):
            seal_reference_plan(
                "reference:missing",
                (_signature("model.unknown"),),
                builtin_reference_catalog(),
            )

    def test_reference_plan_resolves_back_to_same_callable(self) -> None:
        catalog = builtin_reference_catalog()
        plan = seal_reference_plan("reference:linear", (_signature("linear"),), catalog)

        resolved = resolve_reference_binding(plan.bindings[0], catalog)

        self.assertIs(resolved, catalog.resolve("linear").function)

    def test_duplicate_semantic_reference_is_rejected(self) -> None:
        def function() -> None:
            return None

        with self.assertRaisesRegex(ValueError, "multiple reference implementations"):
            ReferenceCatalog(
                (
                    ReferenceImplementation("fixture", "fixture.v1", function),
                    ReferenceImplementation("fixture", "fixture.v2", function),
                )
            )

    def test_tampered_binding_identity_is_rejected(self) -> None:
        catalog = builtin_reference_catalog()
        plan = seal_reference_plan("reference:linear", (_signature("linear"),), catalog)
        binding_type = type(plan.bindings[0])
        tampered = binding_type(
            signature=plan.bindings[0].signature,
            reference_implementation_id="external.fake.v1",
        )

        with self.assertRaisesRegex(RuntimeError, "does not match"):
            resolve_reference_binding(tampered, catalog)


if __name__ == "__main__":
    unittest.main()
