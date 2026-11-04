from __future__ import annotations

import hashlib
import os
import unittest
from dataclasses import replace
from unittest import mock

import torch

from cubic_serve.guardrail import linear_artifact as artifact_guardrail
from cubic_serve.guardrail.linear_artifact import (
    LINEAR_ARTIFACT_SEMANTICS,
    BF16LinearArtifactManifest,
    LinearArtifactRejected,
    ReleaseBF16LinearArtifact,
    StructuralLinearArtifactWitness,
    verify_bf16_linear_artifact_launch,
)
from cubic_serve.ops.cuda.compiled_artifact import COMPILED_ARTIFACT_SCHEMA_ID
from cubic_serve.ops.cuda.linear import (
    BF16_LINEAR_OPERATOR_ID,
    BF16LinearArtifactIdentity,
    BF16LinearLaunchReceipt,
    BF16LinearPrimitive,
    BF16LinearTensorBinding,
)
from cubic_serve.ops.cuda.linear import kernel as linear_kernel

RUN_REAL = os.environ.get("CUBIC_RUN_CUDA_LINEAR_ARTIFACT_TESTS") == "1"


def _digest(label: str) -> str:
    return "b2s256:" + hashlib.blake2s(label.encode()).hexdigest()


def _identity() -> BF16LinearArtifactIdentity:
    return BF16LinearArtifactIdentity(
        schema_id=COMPILED_ARTIFACT_SCHEMA_ID,
        operator_id=BF16_LINEAR_OPERATOR_ID,
        compiled_hash=_digest("torch-cublas-closure"),
        source_hash=_digest("torch-mm-out"),
        stage_hashes=(),
        compiler_metadata_json=(
            '[["dispatch","aten.mm.out"],'
            '["selected_cubin","not exposed by vendor ABI"]]'
        ),
        specialization_json='[["K",32],["BLOCK_K",32]]',
        target=("cuda", 120, 32),
        driver_version=13020,
        sm=(12, 0),
        compiler_flags=(
            ("allow_bf16_reduced_precision_reduction", "false"),
            ("allow_tf32", "false"),
            ("cublas_workspace_bytes", "33554432"),
            ("cublaslt_workspace_bytes", "1048576"),
            ("deterministic_algorithms", "false"),
            ("preferred_blas_library", '"_BlasBackend.Cublas"'),
            ("workspace_config", '""'),
        ),
        runtime=(("torch", "test"), ("cuda", "test"), ("cublas", "test")),
        launch_signature="m=2,n=32,k=32,backend=torch-cublas,acc=fp32,out=bf16",
        grid=(1, 1, 1),
        backend="torch-cublas",
        backend_binary_hashes=(
            ("libcublas", _digest("libcublas")),
            ("libcublasLt", _digest("libcublasLt")),
        ),
    )


def _approval(identity: BF16LinearArtifactIdentity) -> ReleaseBF16LinearArtifact:
    return ReleaseBF16LinearArtifact(
        "release:test",
        BF16_LINEAR_OPERATOR_ID,
        identity.artifact_digest,
        identity.launch_specialization_digest,
        LINEAR_ARTIFACT_SEMANTICS,
    )


def _manifest(identity: BF16LinearArtifactIdentity) -> BF16LinearArtifactManifest:
    return BF16LinearArtifactManifest("release:test", (_approval(identity),))


class LinearArtifactManifestTests(unittest.TestCase):
    def test_content_address_covers_binary_source_and_full_launch(self) -> None:
        identity = _identity()
        manifest = _manifest(identity)
        self.assertEqual(
            artifact_guardrail._resolve_release_artifact(manifest, identity),
            _approval(identity),
        )

        binaries = tuple(
            (name, _digest("altered-library") if name == "libcublasLt" else digest)
            for name, digest in identity.backend_binary_hashes
        )
        faults = (
            replace(identity, compiled_hash=_digest("unknown-dispatch-closure")),
            replace(identity, backend_binary_hashes=binaries),
            replace(identity, source_hash=_digest("source-omits-final-k-tile")),
            replace(identity, specialization_json='[["K",31],["BLOCK_K",32]]'),
            replace(identity, grid=(2, 1, 1)),
        )
        for fault in faults:
            with self.subTest(fault=fault.artifact_digest), self.assertRaisesRegex(
                LinearArtifactRejected, "not release-approved"
            ):
                artifact_guardrail._resolve_release_artifact(manifest, fault)

    def test_semantics_and_numeric_policy_fail_closed(self) -> None:
        identity = _identity()
        incomplete = ReleaseBF16LinearArtifact(
            "release:test",
            BF16_LINEAR_OPERATOR_ID,
            identity.artifact_digest,
            identity.launch_specialization_digest,
            LINEAR_ARTIFACT_SEMANTICS[:-1],
        )
        with self.assertRaisesRegex(LinearArtifactRejected, "operator semantics"):
            artifact_guardrail._resolve_release_artifact(
                BF16LinearArtifactManifest("release:test", (incomplete,)), identity
            )
        reduced_bf16 = replace(
            identity,
            compiler_flags=tuple(
                (name, "true" if name == "allow_bf16_reduced_precision_reduction" else value)
                for name, value in identity.compiler_flags
            ),
        )
        with self.assertRaisesRegex(LinearArtifactRejected, "flags"):
            artifact_guardrail._resolve_release_artifact(
                _manifest(reduced_bf16), reduced_bf16
            )

    def test_manifest_entry_and_artifact_subclasses_cannot_override_admission(self) -> None:
        with self.assertRaisesRegex(TypeError, "cannot be subclassed"):

            class FakeRelease(ReleaseBF16LinearArtifact):
                pass

        with self.assertRaisesRegex(TypeError, "cannot be subclassed"):

            class FakeIdentity(BF16LinearArtifactIdentity):
                pass

    def test_manifest_revalidates_mutated_release_semantics_on_consume(self) -> None:
        identity = _identity()
        manifest = _manifest(identity)
        object.__setattr__(manifest.entries[0], "semantic_facts", ())
        with self.assertRaisesRegex(LinearArtifactRejected, "semantic facts"):
            artifact_guardrail._resolve_release_artifact(manifest, identity)

    def test_identity_revalidates_strict_runtime_and_operator_on_consume(self) -> None:
        identity = _identity()
        manifest = _manifest(identity)
        object.__setattr__(identity, "runtime", (("torch", "test"),))
        with self.assertRaisesRegex(LinearArtifactRejected, "runtime"):
            artifact_guardrail._resolve_release_artifact(manifest, identity)
        wrong_operator = replace(_identity(), operator_id="cubic.cuda.paged-gqa.triton.v1")
        with self.assertRaisesRegex(LinearArtifactRejected, "operator identity"):
            artifact_guardrail._resolve_release_artifact(_manifest(wrong_operator), wrong_operator)

    def test_receipt_requires_exact_type_and_binds_stream(self) -> None:
        primitive = object.__new__(BF16LinearPrimitive)
        primitive._artifact = object()
        primitive._last_bindings = ()
        primitive._signature = "signature"
        primitive._grid = (1, 1, 1)
        primitive._ordinal = 1
        primitive._last_stream = 17
        primitive._receipt_token = object()
        binding = BF16LinearTensorBinding(
            1, 1, 1, 0, (1,), (1,), "torch.bfloat16", "cuda:0"
        )
        primitive._last_bindings = (binding,)
        receipt = BF16LinearLaunchReceipt(
            id(primitive._artifact),
            primitive._last_bindings,
            primitive._signature,
            primitive._grid,
            primitive._last_stream,
            primitive._ordinal,
            primitive._receipt_token,
        )

        with self.assertRaisesRegex(TypeError, "cannot be subclassed"):

            class FakeReceipt(BF16LinearLaunchReceipt):
                pass
        with self.assertRaisesRegex(ValueError, "receipt"):
            primitive.validate_receipt(replace(receipt, stream=19))

    def test_primitive_has_no_instance_method_shadowing_surface(self) -> None:
        primitive = object.__new__(BF16LinearPrimitive)
        self.assertFalse(hasattr(primitive, "__dict__"))
        with self.assertRaises(AttributeError):
            primitive.validate_operands = lambda *args: None  # type: ignore[method-assign]

    def test_fake_artifact_and_witness_cannot_issue_production_evidence(self) -> None:
        identity = _identity()

        class FakeCompiledPrimitive:
            artifact = identity

        cpu = torch.zeros((1, 1), dtype=torch.bfloat16)
        with self.assertRaisesRegex(LinearArtifactRejected, "production"):
            verify_bf16_linear_artifact_launch(
                primitive=FakeCompiledPrimitive(),  # type: ignore[arg-type]
                x=cpu,
                weight=cpu,
                output=cpu,
                manifest=_manifest(identity),
            )
        with self.assertRaisesRegex(TypeError, "verifier-issued"):
            StructuralLinearArtifactWitness(
                _digest("artifact"),
                _digest("approval"),
                _digest("operands"),
                _digest("launch"),
                _digest("output"),
                object(),
            )


@unittest.skipUnless(RUN_REAL, "set CUBIC_RUN_CUDA_LINEAR_ARTIFACT_TESTS=1")
class RealLinearArtifactLaunchTests(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        self.device = torch.device("cuda:0")
        if torch.cuda.get_device_capability(self.device) != (12, 0):
            self.skipTest("real artifact qualification is pinned to SM120")

    def _case(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, BF16LinearPrimitive]:
        torch.manual_seed(2017)
        x = torch.randn((2, 32), device=self.device, dtype=torch.bfloat16)
        weight = torch.randn((32, 32), device=self.device, dtype=torch.bfloat16)
        output = torch.zeros((2, 32), device=self.device, dtype=torch.bfloat16)
        return x, weight, output, BF16LinearPrimitive(x, weight, output)

    def test_verifier_owns_launch_sentinel_sync_and_collection(self) -> None:
        x, weight, output, primitive = self._case()
        identity = primitive.project_artifact()
        with mock.patch.object(
            torch, "compile", side_effect=AssertionError("torch.compile was entered")
        ):
            witness = verify_bf16_linear_artifact_launch(
                primitive=primitive,
                x=x,
                weight=weight,
                output=output,
                manifest=_manifest(identity),
            )
        self.assertIsInstance(witness, StructuralLinearArtifactWitness)
        self.assertEqual(witness.artifact_digest, identity.artifact_digest)
        self.assertTrue(witness.witness_digest.startswith("b2s256:"))
        self.assertFalse(bool(torch.isnan(output).any().item()))
        with self.assertRaises(TypeError):
            verify_bf16_linear_artifact_launch(
                primitive=primitive,
                x=x,
                weight=weight,
                output=output,
                manifest=_manifest(identity),
                actual=output,  # type: ignore[call-arg]
            )

        with self.assertRaisesRegex(TypeError, "cannot be subclassed"):

            class FakeManifest(BF16LinearArtifactManifest):
                pass

    def test_wrong_objects_alias_grid_and_stale_output_reject(self) -> None:
        x, weight, output, primitive = self._case()
        manifest = _manifest(primitive.project_artifact())
        with self.assertRaisesRegex(LinearArtifactRejected, "exact bound"):
            verify_bf16_linear_artifact_launch(
                primitive=primitive,
                x=x.clone(),
                weight=weight,
                output=output,
                manifest=manifest,
            )
        with self.assertRaisesRegex(ValueError, "non-aliasing"):
            BF16LinearPrimitive(x, weight, x)

        original_grid = primitive._grid
        primitive._grid = (original_grid[0] + 1, *original_grid[1:])
        with self.assertRaisesRegex(LinearArtifactRejected, "not release-approved"):
            verify_bf16_linear_artifact_launch(
                primitive=primitive, x=x, weight=weight, output=output, manifest=manifest
            )
        primitive._grid = original_grid

        with mock.patch.object(
            linear_kernel.torch, "mm", return_value=output
        ), self.assertRaisesRegex(LinearArtifactRejected, "stale"):
            verify_bf16_linear_artifact_launch(
                primitive=primitive, x=x, weight=weight, output=output, manifest=manifest
            )

    def test_altered_vendor_binary_and_dispatch_source_reject_before_launch(self) -> None:
        x, weight, output, primitive = self._case()
        manifest = _manifest(primitive.project_artifact())
        artifact = primitive._artifact
        old_binaries = artifact.backend_binary_hashes
        object.__setattr__(
            artifact,
            "backend_binary_hashes",
            (("libcublas", _digest("changed")), old_binaries[1]),
        )
        with self.assertRaisesRegex(LinearArtifactRejected, "runtime differs"):
            verify_bf16_linear_artifact_launch(
                primitive=primitive, x=x, weight=weight, output=output, manifest=manifest
            )
        object.__setattr__(artifact, "backend_binary_hashes", old_binaries)

        source = artifact.source_hash
        object.__setattr__(artifact, "source_hash", _digest("changed-dispatch"))
        with self.assertRaisesRegex(LinearArtifactRejected, "not release-approved"):
            verify_bf16_linear_artifact_launch(
                primitive=primitive, x=x, weight=weight, output=output, manifest=manifest
            )
        object.__setattr__(artifact, "source_hash", source)

    def test_private_observation_rejects_stale_replay_and_changed_operands(self) -> None:
        x, weight, output, primitive = self._case()
        manifest = _manifest(primitive.project_artifact())
        stale = artifact_guardrail._observe_bf16_linear_artifact_launch(
            primitive=primitive, x=x, weight=weight, output=output, manifest=manifest
        )
        primitive.launch(x, weight, output)
        torch.cuda.synchronize(self.device)
        with self.assertRaisesRegex(LinearArtifactRejected, "receipt"):
            stale.consume()

        consumed = artifact_guardrail._observe_bf16_linear_artifact_launch(
            primitive=primitive, x=x, weight=weight, output=output, manifest=manifest
        )
        _witness, actual = consumed.consume()
        self.assertEqual(actual.device.type, "cpu")
        with self.assertRaisesRegex(LinearArtifactRejected, "already consumed"):
            consumed.consume()

        for tensor in (x, weight, output):
            changed = artifact_guardrail._observe_bf16_linear_artifact_launch(
                primitive=primitive, x=x, weight=weight, output=output, manifest=manifest
            )
            tensor.add_(1.0)
            with self.assertRaisesRegex(LinearArtifactRejected, "operands changed"):
                changed.consume()


if __name__ == "__main__":
    unittest.main()
