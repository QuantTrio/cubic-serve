from __future__ import annotations

import os
import unittest

import torch

from cubic_serve.guardrail.linear import (
    LinearQualificationError,
    VerifiedBF16LinearCandidateWitness,
    plan_bf16_linear_workspace,
    qualify_bf16_linear_candidate,
)
from cubic_serve.guardrail.linear_artifact import (
    LINEAR_ARTIFACT_SEMANTICS,
    BF16LinearArtifactManifest,
    ReleaseBF16LinearArtifact,
)
from cubic_serve.models import FullBF16WeightPolicy, SealedWeightSet, seal_full_bf16_weight_set
from cubic_serve.ops.cuda.linear import BF16_LINEAR_OPERATOR_ID, BF16LinearPrimitive

RUN_REAL = os.environ.get("CUBIC_RUN_CUDA_LINEAR_TESTS") == "1"


def _manifest(primitive: BF16LinearPrimitive) -> BF16LinearArtifactManifest:
    identity = primitive.project_artifact()
    entry = ReleaseBF16LinearArtifact(
        "release:test",
        BF16_LINEAR_OPERATOR_ID,
        identity.artifact_digest,
        identity.launch_specialization_digest,
        LINEAR_ARTIFACT_SEMANTICS,
    )
    return BF16LinearArtifactManifest("release:test", (entry,))


def _sealed(weight: torch.Tensor, name: str = "projection.weight"):
    inventory: list[tuple[str, object]] = [(name, weight)]
    return seal_full_bf16_weight_set(
        lambda: inventory,
        policy=FullBF16WeightPolicy(expected_names=(name,)),
    )


class BF16LinearWorkspaceTests(unittest.TestCase):
    def test_lm_head_shape_has_a_strict_full_workspace_plan(self) -> None:
        budget = 64 << 20
        plan = plan_bf16_linear_workspace(1, 248320, 5120, budget)
        self.assertGreater(plan.chunk_columns, 0)
        self.assertLessEqual(plan.peak_bytes, budget)
        self.assertEqual(plan.fixed_bytes, 2 * 248320 + 16 * 5120)
        self.assertEqual(plan.bytes_per_output_column, 10 * 5120 + 24)
        with self.assertRaisesRegex(LinearQualificationError, "one oracle column"):
            plan_bf16_linear_workspace(4, 248320, 5120, 1 << 20)

    def test_candidate_witness_cannot_be_faked(self) -> None:
        with self.assertRaisesRegex(TypeError, "verifier-issued"):
            VerifiedBF16LinearCandidateWitness(
                "b2s256:" + "1" * 64,
                "b2s256:" + "2" * 64,
                "b2s256:" + "3" * 64,
                "b2s256:" + "4" * 64,
                32,
                32,
                32,
                1 << 20,
                0,
                0.0,
                0.0,
                object(),
            )

    def test_sealed_weight_subclass_cannot_override_binding_admission(self) -> None:
        class FakeSealedWeights(SealedWeightSet):
            def tensor_binding_digest(self, *args: object, **kwargs: object) -> str:
                return "b2s256:" + "0" * 64

        cpu = torch.zeros((1, 1), dtype=torch.bfloat16)
        fake = object.__new__(FakeSealedWeights)
        with self.assertRaisesRegex(LinearQualificationError, "named sealed weights"):
            qualify_bf16_linear_candidate(
                primitive=object(),  # type: ignore[arg-type]
                x=cpu,
                weight=cpu,
                output=cpu,
                weight_name="projection.weight",
                sealed_weights=fake,
                manifest=object(),  # type: ignore[arg-type]
            )


@unittest.skipUnless(RUN_REAL, "set CUBIC_RUN_CUDA_LINEAR_TESTS=1")
class BF16LinearCandidateQualificationTests(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        self.device = torch.device("cuda:0")
        if torch.cuda.get_device_capability(self.device) != (12, 0):
            self.skipTest("real qualification is pinned to SM120")

    def _case(
        self, *, m: int = 4, n: int = 384, k: int = 512
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, BF16LinearPrimitive]:
        torch.manual_seed(1511)
        x = torch.randn((m, k), device=self.device, dtype=torch.bfloat16)
        weight = torch.randn((n, k), device=self.device, dtype=torch.bfloat16)
        output = torch.empty((m, n), device=self.device, dtype=torch.bfloat16)
        return x, weight, output, BF16LinearPrimitive(x, weight, output)

    def test_verifier_relaunches_bound_operands_and_owns_actual(self) -> None:
        x, weight, output, primitive = self._case()
        wrong_x = torch.randn_like(x)
        primitive.launch(wrong_x, weight, output)
        torch.cuda.synchronize(self.device)
        wrong_result = output.detach().cpu().clone()
        sealed, manifest = _sealed(weight), _manifest(primitive)
        torch.cuda.reset_peak_memory_stats(self.device)
        baseline = torch.cuda.memory_allocated(self.device)

        witness = qualify_bf16_linear_candidate(
            primitive=primitive,
            x=x,
            weight=weight,
            output=output,
            weight_name="projection.weight",
            sealed_weights=sealed,
            manifest=manifest,
            max_workspace_bytes=2 << 20,
        )
        gpu_extra = torch.cuda.max_memory_allocated(self.device) - baseline
        self.assertIsInstance(witness, VerifiedBF16LinearCandidateWitness)
        self.assertLessEqual(witness.max_observed_error, witness.max_allowed_error)
        self.assertLessEqual(witness.explicit_host_tensor_peak_bytes, 2 << 20)
        self.assertEqual(witness.planned_extra_gpu_bytes, 0)
        self.assertLess(gpu_extra, 1 << 20)
        self.assertFalse(torch.equal(wrong_result, output.detach().cpu()))

        arguments = dict(
            primitive=primitive,
            x=x,
            weight=weight,
            output=output,
            weight_name="projection.weight",
            sealed_weights=sealed,
            manifest=manifest,
        )
        with self.assertRaises(TypeError):
            qualify_bf16_linear_candidate(
                **arguments, actual=output, expected=output  # type: ignore[call-arg]
            )

    def test_workspace_rejects_before_launch_and_manifest_blocks_omitted_k(self) -> None:
        x, weight, output, primitive = self._case(m=1, n=32, k=512)
        output.fill_(17.0)
        sealed, manifest = _sealed(weight), _manifest(primitive)
        with self.assertRaisesRegex(LinearQualificationError, "one oracle column"):
            qualify_bf16_linear_candidate(
                primitive=primitive,
                x=x,
                weight=weight,
                output=output,
                weight_name="projection.weight",
                sealed_weights=sealed,
                manifest=manifest,
                max_workspace_bytes=1024,
            )
        self.assertTrue(bool((output == 17.0).all().item()))

        tiny_x = x.clone()
        tiny_x[0, -1] = 2.0**-20
        contribution = abs(float(tiny_x[0, -1]) * float(weight[0, -1]))
        old_envelope = (512 * 2.0**-24) * float(
            (tiny_x[0].float().abs() * weight[0].float().abs()).sum().item()
        )
        self.assertLess(contribution, old_envelope)
        artifact = primitive._artifact
        source = artifact.source_hash
        object.__setattr__(artifact, "source_hash", "b2s256:" + "0" * 64)
        with self.assertRaisesRegex(LinearQualificationError, "not release-approved"):
            qualify_bf16_linear_candidate(
                primitive=primitive,
                x=x,
                weight=weight,
                output=output,
                weight_name="projection.weight",
                sealed_weights=sealed,
                manifest=manifest,
                max_workspace_bytes=2 << 20,
            )
        object.__setattr__(artifact, "source_hash", source)

    def test_subnormal_nonfinite_and_overflow_corpora_reject(self) -> None:
        cases = (
            (float(torch.finfo(torch.bfloat16).tiny) / 2.0, 1.0, "subnormal"),
            (float("inf"), 1.0, "nonfinite"),
            (float(torch.finfo(torch.bfloat16).max), 2.0, "overflow|nonfinite"),
        )
        for x_value, weight_value, reason in cases:
            with self.subTest(reason=reason):
                x = torch.full((1, 32), x_value, device=self.device, dtype=torch.bfloat16)
                weight = torch.full(
                    (32, 32), weight_value, device=self.device, dtype=torch.bfloat16
                )
                output = torch.empty((1, 32), device=self.device, dtype=torch.bfloat16)
                primitive = BF16LinearPrimitive(x, weight, output)
                with self.assertRaisesRegex(LinearQualificationError, reason):
                    qualify_bf16_linear_candidate(
                        primitive=primitive,
                        x=x,
                        weight=weight,
                        output=output,
                        weight_name="projection.weight",
                        sealed_weights=_sealed(weight),
                        manifest=_manifest(primitive),
                        max_workspace_bytes=1 << 20,
                    )


if __name__ == "__main__":
    unittest.main()
