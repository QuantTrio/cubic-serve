from __future__ import annotations

import math
import unittest

import torch

from cubic_serve.contracts.model import ScalarDType
from cubic_serve.guardrail.compare import (
    ComparisonContext,
    ComparisonPolicy,
    DifferenceClass,
    PartitionedReductionProof,
    ReductionOrderProof,
    ScaleCastABI,
    ScaleDTypeProof,
    VerifiedBF16LinearSampleWitness,
    VerifiedReductionWitness,
    compare_tensors,
    scale_abi_digest,
    stable_digest,
    tensor_digest,
)
from cubic_serve.guardrail.oracle import (
    PartitionedReductionWitnessMaterial,
    ReductionWitnessMaterial,
    ScaleWitnessMaterial,
    WitnessVerificationError,
    partitioned_reduction_material_identity,
    reduction_material_identity,
    verify_observed_bf16_linear_sample,
    verify_partitioned_reduction_witness,
    verify_reduction_witness,
    verify_scale_witness,
)


def _context() -> ComparisonContext:
    return ComparisonContext(
        op_signature_digest=stable_digest("linear/m=1"),
        candidate_launch_digest=stable_digest("candidate/block=128"),
    )


def _digest(label: str) -> str:
    return stable_digest(label)


def _reduction_material(length: int = 16) -> ReductionWitnessMaterial:
    reference_operands = torch.full((1, length), 0.25, dtype=torch.float32)
    reference_indices = torch.arange(length, dtype=torch.int64).reshape(1, length)
    return ReductionWitnessMaterial(
        reference_operands=reference_operands,
        reference_indices=reference_indices,
        candidate_operands=reference_operands.flip(-1),
        candidate_indices=reference_indices.flip(-1),
    )


def _cancellation_material() -> ReductionWitnessMaterial:
    operands = torch.tensor([[1e8, 1.0, -1e8]], dtype=torch.float32)
    indices = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    return ReductionWitnessMaterial(
        reference_operands=operands,
        reference_indices=indices,
        candidate_operands=operands[:, [0, 2, 1]],
        candidate_indices=indices[:, [0, 2, 1]],
    )


def _reduction_policy(material: ReductionWitnessMaterial) -> ComparisonPolicy:
    identity = reduction_material_identity(material)
    return ComparisonPolicy(
        DifferenceClass.REDUCTION_ORDER,
        reduction=ReductionOrderProof(
            operands_digest=identity.operands_digest,
            indices_digest=identity.indices_digest,
            reduction_length=identity.reduction_length,
            accumulator_dtype=ScalarDType.FP32,
            candidate_accumulator_dtype=ScalarDType.FP32,
        ),
    )


def _partitioned_material() -> PartitionedReductionWitnessMaterial:
    return PartitionedReductionWitnessMaterial(
        left_operands=torch.ones((1, 4), dtype=torch.bfloat16),
        right_operands=torch.tensor([[4096.0, 1.0, -4096.0, 1.0]], dtype=torch.bfloat16),
        logical_indices=torch.arange(4, dtype=torch.int64).reshape(1, 4),
        partition_ids=torch.tensor([[0, 0, 1, 1]], dtype=torch.int64),
    )


def _partitioned_policy(material: PartitionedReductionWitnessMaterial) -> ComparisonPolicy:
    identity = partitioned_reduction_material_identity(material)
    return ComparisonPolicy(
        DifferenceClass.PARTITIONED_REDUCTION,
        partitioned_reduction=PartitionedReductionProof(
            operands_digest=identity.operands_digest,
            indices_digest=identity.indices_digest,
            partitions_digest=identity.partitions_digest,
            reduction_length=identity.reduction_length,
        ),
    )


class ExactComparisonTests(unittest.TestCase):
    def test_exact_has_no_global_tolerance(self) -> None:
        expected = torch.tensor([1.0], dtype=torch.bfloat16)
        actual = torch.nextafter(expected, torch.full_like(expected, float("inf")))

        result = compare_tensors(expected, actual, ComparisonPolicy(DifferenceClass.EXACT))

        self.assertFalse(result.passed)
        self.assertEqual(result.max_ulp_error, 1)
        self.assertEqual(result.first_mismatch_index, (0,))

    def test_nonfinite_is_never_admitted(self) -> None:
        value = torch.tensor([float("nan")], dtype=torch.float32)
        result = compare_tensors(value, value, ComparisonPolicy(DifferenceClass.EXACT))
        self.assertFalse(result.passed)
        self.assertIn("non-finite", result.reason)

    def test_exact_uses_bits_not_lossy_float64_for_large_integers(self) -> None:
        expected = torch.tensor([2**62], dtype=torch.int64)
        actual = torch.tensor([2**62 + 1], dtype=torch.int64)

        result = compare_tensors(expected, actual, ComparisonPolicy(DifferenceClass.EXACT))

        self.assertFalse(result.passed)
        self.assertIsNone(result.max_absolute_error)

    def test_exact_distinguishes_signed_zero(self) -> None:
        expected = torch.tensor([0.0], dtype=torch.float32)
        actual = torch.tensor([-0.0], dtype=torch.float32)

        result = compare_tensors(expected, actual, ComparisonPolicy(DifferenceClass.EXACT))

        self.assertFalse(result.passed)

    def test_old_naked_error_inputs_are_not_an_api(self) -> None:
        value = torch.ones(1)
        with self.assertRaises(TypeError):
            compare_tensors(  # type: ignore[call-arg]
                value,
                value,
                ComparisonPolicy(DifferenceClass.EXACT),
                reduction_sum_abs=torch.tensor([1e30]),
            )


class ReductionWitnessTests(unittest.TestCase):
    def test_reordered_reduction_uses_verified_two_tree_bound(self) -> None:
        material = _cancellation_material()
        policy = _reduction_policy(material)
        expected = torch.tensor([0.0], dtype=torch.float32)
        actual = torch.tensor([1.0], dtype=torch.float32)
        witness = verify_reduction_witness(
            expected, actual, policy, _context(), material
        )

        result = compare_tensors(
            expected, actual, policy, context=_context(), witness=witness
        )

        self.assertTrue(result.passed)
        self.assertGreater(result.max_allowed_error, 0.0)
        self.assertIsNotNone(result.witness_digest)

    def test_huge_error_is_not_hidden_by_real_witness(self) -> None:
        material = _reduction_material()
        policy = _reduction_policy(material)
        expected = torch.tensor([4.0], dtype=torch.bfloat16)
        actual = torch.tensor([100.0], dtype=torch.bfloat16)
        with self.assertRaisesRegex(WitnessVerificationError, "candidate output"):
            verify_reduction_witness(expected, actual, policy, _context(), material)

    def test_wrong_digest_is_rejected_during_live_recomputation(self) -> None:
        material = _reduction_material()
        identity = reduction_material_identity(material)
        policy = ComparisonPolicy(
            DifferenceClass.REDUCTION_ORDER,
            reduction=ReductionOrderProof(
                operands_digest="b2s256:" + "0" * 64,
                indices_digest=identity.indices_digest,
                reduction_length=identity.reduction_length,
                accumulator_dtype=ScalarDType.FP32,
            ),
        )
        with self.assertRaisesRegex(WitnessVerificationError, "operand digest"):
            verify_reduction_witness(
                torch.tensor([4.0], dtype=torch.bfloat16),
                torch.tensor([4.0], dtype=torch.bfloat16),
                policy,
                _context(),
                material,
            )

    def test_duplicate_index_fails_coverage_validation(self) -> None:
        material = _reduction_material(4)
        bad = ReductionWitnessMaterial(
            reference_operands=material.reference_operands,
            reference_indices=torch.tensor([[0, 1, 1, 3]], dtype=torch.int64),
            candidate_operands=material.candidate_operands,
            candidate_indices=material.candidate_indices,
        )
        with self.assertRaisesRegex(WitnessVerificationError, "exactly once"):
            reduction_material_identity(bad)

    def test_witness_cannot_be_manually_constructed(self) -> None:
        digest = stable_digest("x")
        with self.assertRaisesRegex(TypeError, "only be issued"):
            VerifiedReductionWitness(
                expected_binding=digest,
                actual_binding=digest,
                policy_digest=digest,
                context_digest=digest,
                operands_digest=digest,
                indices_digest=digest,
                reference_bound=(0.0,),
                candidate_bound=(0.0,),
                shape=(1,),
            )

    def test_witness_is_bound_to_actual_content_and_launch(self) -> None:
        material = _reduction_material()
        policy = _reduction_policy(material)
        expected = torch.tensor([4.0], dtype=torch.bfloat16)
        actual = expected.clone()
        witness = verify_reduction_witness(
            expected, actual, policy, _context(), material
        )
        actual.fill_(100.0)

        result = compare_tensors(
            expected, actual, policy, context=_context(), witness=witness
        )
        self.assertFalse(result.passed)
        self.assertIn("different candidate tensor", result.reason)

        other_context = ComparisonContext(
            op_signature_digest=_context().op_signature_digest,
            candidate_launch_digest=stable_digest("another launch"),
        )
        result = compare_tensors(
            expected,
            torch.tensor([4.0], dtype=torch.bfloat16),
            policy,
            context=other_context,
            witness=witness,
        )
        self.assertFalse(result.passed)

    def test_max_finite_has_finite_bound(self) -> None:
        maximum = torch.finfo(torch.bfloat16).max
        operands = torch.tensor([[maximum]], dtype=torch.bfloat16)
        indices = torch.tensor([[0]], dtype=torch.int64)
        material = ReductionWitnessMaterial(operands, indices, operands.clone(), indices.clone())
        policy = _reduction_policy(material)
        expected = torch.tensor([maximum], dtype=torch.bfloat16)
        actual = expected.clone()
        witness = verify_reduction_witness(
            expected, actual, policy, _context(), material
        )

        result = compare_tensors(
            expected, actual, policy, context=_context(), witness=witness
        )

        self.assertTrue(result.passed)
        self.assertTrue(math.isfinite(result.max_allowed_error))


class PartitionedReductionWitnessTests(unittest.TestCase):
    def test_fp32_subnormal_products_have_an_operand_bound_ftz_envelope(self) -> None:
        tiny = torch.tensor(2.0**-70, dtype=torch.bfloat16)
        material = PartitionedReductionWitnessMaterial(
            left_operands=torch.tensor([[tiny, tiny]], dtype=torch.bfloat16),
            right_operands=torch.tensor([[tiny, tiny]], dtype=torch.bfloat16),
            logical_indices=torch.arange(2, dtype=torch.int64).reshape(1, 2),
            partition_ids=torch.tensor([[0, 1]], dtype=torch.int64),
        )
        zero = torch.zeros(1, dtype=torch.bfloat16)

        witness = verify_partitioned_reduction_witness(
            zero, zero, _partitioned_policy(material), _context(), material
        )

        self.assertTrue(
            compare_tensors(
                zero,
                zero,
                _partitioned_policy(material),
                context=_context(),
                witness=witness,
            ).passed
        )

    def test_subnormal_envelope_does_not_hide_a_normal_scale_error(self) -> None:
        tiny = torch.tensor(2.0**-70, dtype=torch.bfloat16)
        material = PartitionedReductionWitnessMaterial(
            left_operands=torch.tensor([[tiny, tiny]], dtype=torch.bfloat16),
            right_operands=torch.tensor([[tiny, tiny]], dtype=torch.bfloat16),
            logical_indices=torch.arange(2, dtype=torch.int64).reshape(1, 2),
            partition_ids=torch.tensor([[0, 1]], dtype=torch.int64),
        )
        with self.assertRaisesRegex(WitnessVerificationError, "candidate output"):
            verify_partitioned_reduction_witness(
                torch.zeros(1, dtype=torch.bfloat16),
                torch.ones(1, dtype=torch.bfloat16),
                _partitioned_policy(material),
                _context(),
                material,
            )

    def test_tp4_partitioned_reduction_uses_all_declared_shards(self) -> None:
        material = PartitionedReductionWitnessMaterial(
            left_operands=torch.ones((1, 8), dtype=torch.bfloat16),
            right_operands=torch.tensor(
                [[4096.0, 1.0, -4096.0, 1.0, 2048.0, 1.0, -2048.0, 1.0]],
                dtype=torch.bfloat16,
            ),
            logical_indices=torch.arange(8, dtype=torch.int64).reshape(1, 8),
            partition_ids=torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]], dtype=torch.int64),
        )
        identity = partitioned_reduction_material_identity(material)
        policy = ComparisonPolicy(
            DifferenceClass.PARTITIONED_REDUCTION,
            partitioned_reduction=PartitionedReductionProof(
                operands_digest=identity.operands_digest,
                indices_digest=identity.indices_digest,
                partitions_digest=identity.partitions_digest,
                reduction_length=identity.reduction_length,
                partition_count=4,
            ),
        )
        expected = torch.tensor([4.0], dtype=torch.bfloat16)
        actual = torch.tensor([0.0], dtype=torch.bfloat16)
        witness = verify_partitioned_reduction_witness(
            expected, actual, policy, _context(), material
        )
        self.assertTrue(
            compare_tensors(
                expected, actual, policy, context=_context(), witness=witness
            ).passed
        )

    def test_tp2_bf16_partial_rounding_has_a_verified_hierarchical_bound(self) -> None:
        material = _partitioned_material()
        policy = _partitioned_policy(material)
        expected = torch.tensor([2.0], dtype=torch.bfloat16)
        actual = torch.tensor([0.0], dtype=torch.bfloat16)

        witness = verify_partitioned_reduction_witness(
            expected, actual, policy, _context(), material
        )
        result = compare_tensors(
            expected, actual, policy, context=_context(), witness=witness
        )

        self.assertTrue(result.passed)
        self.assertGreaterEqual(result.max_allowed_error, 2.0)

    def test_huge_error_is_not_hidden(self) -> None:
        material = _partitioned_material()
        with self.assertRaisesRegex(WitnessVerificationError, "candidate output"):
            verify_partitioned_reduction_witness(
                torch.tensor([2.0], dtype=torch.bfloat16),
                torch.tensor([1024.0], dtype=torch.bfloat16),
                _partitioned_policy(material),
                _context(),
                material,
            )

    def test_wrong_partition_map_is_rejected(self) -> None:
        material = _partitioned_material()
        wrong = PartitionedReductionWitnessMaterial(
            material.left_operands,
            material.right_operands,
            material.logical_indices,
            torch.tensor([[0, 1, 0, 1]], dtype=torch.int64),
        )
        with self.assertRaisesRegex(WitnessVerificationError, "partition digest"):
            verify_partitioned_reduction_witness(
                torch.tensor([2.0], dtype=torch.bfloat16),
                torch.tensor([0.0], dtype=torch.bfloat16),
                _partitioned_policy(material),
                _context(),
                wrong,
            )

    def test_missing_product_operand_is_rejected(self) -> None:
        material = _partitioned_material()
        missing = PartitionedReductionWitnessMaterial(
            material.left_operands[..., :-1],
            material.right_operands[..., :-1],
            material.logical_indices[..., :-1],
            material.partition_ids[..., :-1],
        )
        with self.assertRaisesRegex(WitnessVerificationError, "declared reduction length"):
            verify_partitioned_reduction_witness(
                torch.tensor([2.0], dtype=torch.bfloat16),
                torch.tensor([0.0], dtype=torch.bfloat16),
                _partitioned_policy(material),
                _context(),
                missing,
            )

    def test_partitioned_witness_cannot_be_manually_constructed(self) -> None:
        digest = stable_digest("x")
        with self.assertRaisesRegex(TypeError, "only be issued"):
            VerifiedReductionWitness(
                expected_binding=digest,
                actual_binding=digest,
                policy_digest=digest,
                context_digest=digest,
                operands_digest=digest,
                indices_digest=digest,
                partitions_digest=digest,
                difference_class=DifferenceClass.PARTITIONED_REDUCTION,
                reference_bound=(0.0,),
                candidate_bound=(0.0,),
                shape=(1,),
            )


class ObservedBF16LinearSampleTests(unittest.TestCase):
    def _case(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, -1.0, 1.0, 3.0]], dtype=torch.bfloat16)
        weight = torch.tensor(
            [[1.0, 3.0, -2.0, 4.0], [2.0, 1.0, 3.0, -1.0], [4.0, -2.0, 1.0, 2.0]],
            dtype=torch.bfloat16,
        )
        output = torch.mm(x.to(torch.float64), weight.to(torch.float64).t()).to(torch.bfloat16)
        return x, weight, output

    def _verify(
        self, x: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, *, launch: str = "launch"
    ) -> VerifiedBF16LinearSampleWitness:
        return verify_observed_bf16_linear_sample(
            x=x,
            weight=weight,
            output=output,
            input_rows=(0, 1),
            output_columns=(0, 2),
            operand_binding_digest=_digest("bound operands"),
            artifact_digest=_digest("artifact"),
            launch_specialization_digest=_digest(launch),
        )

    def test_full_k_samples_receive_a_bound_artifact_witness(self) -> None:
        witness = self._verify(*self._case())

        self.assertEqual(witness.reduction_length, 4)
        self.assertEqual(witness.sampled_element_count, 4)
        self.assertLessEqual(witness.max_observed_error, witness.max_allowed_error)
        self.assertEqual(witness.artifact_digest, _digest("artifact"))

    def test_artifact_launch_is_part_of_sample_identity(self) -> None:
        case = self._case()
        first = self._verify(*case, launch="launch-a")
        second = self._verify(*case, launch="launch-b")

        self.assertNotEqual(first.sample_identity_digest, second.sample_identity_digest)
        self.assertNotEqual(first.witness_digest, second.witness_digest)

    def test_huge_observed_error_is_rejected(self) -> None:
        x, weight, output = self._case()
        output[0, 0] = 1024.0
        with self.assertRaisesRegex(WitnessVerificationError, "observed sample exceeds"):
            self._verify(x, weight, output)

    def test_ftz_sensitive_products_are_bound_to_the_sampled_operands(self) -> None:
        tiny = torch.tensor(2.0**-70, dtype=torch.bfloat16)
        x = torch.tensor([[tiny, 1.0]], dtype=torch.bfloat16)
        weight = torch.tensor([[tiny, 1.0]], dtype=torch.bfloat16)
        output = torch.ones((1, 1), dtype=torch.bfloat16)

        witness = verify_observed_bf16_linear_sample(
            x=x,
            weight=weight,
            output=output,
            input_rows=(0,),
            output_columns=(0,),
            operand_binding_digest=_digest("bound operands"),
            artifact_digest=_digest("artifact"),
            launch_specialization_digest=_digest("launch"),
        )

        self.assertEqual(witness.sampled_element_count, 1)

    def test_bf16_subnormal_input_flush_is_bound_to_the_actual_product(self) -> None:
        subnormal = torch.tensor(2.0**-130, dtype=torch.bfloat16)
        x = subnormal.reshape(1, 1)
        weight = torch.tensor([[2.0**120]], dtype=torch.bfloat16)
        flushed = torch.zeros((1, 1), dtype=torch.bfloat16)

        witness = verify_observed_bf16_linear_sample(
            x=x,
            weight=weight,
            output=flushed,
            input_rows=(0,),
            output_columns=(0,),
            operand_binding_digest=_digest("bound operands"),
            artifact_digest=_digest("artifact"),
            launch_specialization_digest=_digest("launch"),
        )

        self.assertGreater(witness.max_allowed_error, 0.0)
        with self.assertRaisesRegex(WitnessVerificationError, "observed sample exceeds"):
            verify_observed_bf16_linear_sample(
                x=x,
                weight=weight,
                output=torch.ones((1, 1), dtype=torch.bfloat16),
                input_rows=(0,),
                output_columns=(0,),
                operand_binding_digest=_digest("bound operands"),
                artifact_digest=_digest("artifact"),
                launch_specialization_digest=_digest("launch"),
            )

    def test_invalid_coordinates_and_incomplete_k_are_rejected(self) -> None:
        x, weight, output = self._case()
        with self.assertRaisesRegex(WitnessVerificationError, "strictly increasing"):
            verify_observed_bf16_linear_sample(
                x=x,
                weight=weight,
                output=output,
                input_rows=(1, 0),
                output_columns=(0,),
                operand_binding_digest=_digest("bound operands"),
                artifact_digest=_digest("artifact"),
                launch_specialization_digest=_digest("launch"),
            )
        with self.assertRaisesRegex(WitnessVerificationError, "complete GEMM"):
            self._verify(x, weight[:, :-1], output)

    def test_sample_witness_cannot_be_manually_constructed(self) -> None:
        with self.assertRaisesRegex(TypeError, "only be issued"):
            VerifiedBF16LinearSampleWitness(
                operand_binding_digest=_digest("binding"),
                artifact_digest=_digest("artifact"),
                launch_specialization_digest=_digest("launch"),
                material_digest=_digest("material"),
                sampled_output=torch.zeros((1, 1), dtype=torch.bfloat16),
                bounds=torch.zeros((1, 1), dtype=torch.float64),
                candidate_shape=(1, 1),
                reduction_length=1,
                input_rows=(0,),
                output_columns=(0,),
                max_observed_error=0.0,
            )


class ScaleWitnessTests(unittest.TestCase):
    def test_scale_dtype_is_replayed_from_operands_scale_and_abi(self) -> None:
        operands = torch.tensor([1.0, 2.0], dtype=torch.float32)
        scale = torch.tensor([0.3333], dtype=torch.float32)
        abi = ScaleCastABI(
            abi_id="elementwise-scale:v1",
            cast_location="linear.scale",
            compute_dtype=ScalarDType.FP32,
            output_dtype=ScalarDType.FP32,
        )
        material = ScaleWitnessMaterial(operands, scale, abi)
        policy = ComparisonPolicy(
            DifferenceClass.SCALE_DTYPE,
            scale=ScaleDTypeProof(
                scale_digest=tensor_digest(scale),
                abi_digest=scale_abi_digest(abi),
                reference_dtype=ScalarDType.FP32,
                candidate_dtype=ScalarDType.FP16,
                cast_location="linear.scale",
            ),
        )
        expected = operands * scale
        actual = operands * scale.to(torch.float16).to(torch.float32)
        witness = verify_scale_witness(expected, actual, policy, _context(), material)

        result = compare_tensors(
            expected, actual, policy, context=_context(), witness=witness
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.max_absolute_error, 0.0)

    def test_scale_witness_rejects_candidate_huge_error(self) -> None:
        operands = torch.ones(1)
        scale = torch.tensor([0.25])
        abi = ScaleCastABI(
            abi_id="scale:v1",
            cast_location="linear.scale",
            compute_dtype=ScalarDType.FP32,
            output_dtype=ScalarDType.FP32,
        )
        policy = ComparisonPolicy(
            DifferenceClass.SCALE_DTYPE,
            scale=ScaleDTypeProof(
                scale_digest=tensor_digest(scale),
                abi_digest=scale_abi_digest(abi),
                reference_dtype=ScalarDType.FP32,
                candidate_dtype=ScalarDType.FP16,
                cast_location="linear.scale",
            ),
        )
        with self.assertRaisesRegex(WitnessVerificationError, "candidate output"):
            verify_scale_witness(
                operands * scale,
                torch.tensor([1e20]),
                policy,
                _context(),
                ScaleWitnessMaterial(operands, scale, abi),
            )

    def test_wrong_scale_digest_is_rejected(self) -> None:
        operands = torch.ones(1)
        scale = torch.tensor([0.25])
        abi = ScaleCastABI(
            abi_id="scale:v1",
            cast_location="linear.scale",
            compute_dtype=ScalarDType.FP32,
            output_dtype=ScalarDType.FP32,
        )
        policy = ComparisonPolicy(
            DifferenceClass.SCALE_DTYPE,
            scale=ScaleDTypeProof(
                scale_digest="b2s256:" + "f" * 64,
                abi_digest=scale_abi_digest(abi),
                reference_dtype=ScalarDType.FP32,
                candidate_dtype=ScalarDType.FP16,
                cast_location="linear.scale",
            ),
        )
        with self.assertRaisesRegex(WitnessVerificationError, "scale digest"):
            verify_scale_witness(
                operands * scale,
                operands * scale,
                policy,
                _context(),
                ScaleWitnessMaterial(operands, scale, abi),
            )


if __name__ == "__main__":
    unittest.main()
