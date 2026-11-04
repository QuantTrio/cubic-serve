"""Independent oracle execution and verified witness construction.

This module owns the data checks behind every non-exact waiver. It intentionally
depends only on contracts, PyTorch tensor primitives, and the comparator; it never
delegates numerical truth to a production/reference operator implementation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from cubic_serve.contracts.model import ScalarDType
from cubic_serve.guardrail.compare import (
    ComparisonContext,
    ComparisonPolicy,
    DifferenceClass,
    PartitionedReductionProof,
    ScaleCastABI,
    VerifiedBF16LinearSampleWitness,
    VerifiedReductionWitness,
    VerifiedScaleWitness,
    _issue_bf16_linear_sample_witness,
    _issue_reduction_witness,
    _issue_scale_witness,
    finite_ulp_size,
    scale_abi_digest,
    stable_digest,
    tensor_digest,
)


class WitnessVerificationError(ValueError):
    """Raised when witness material cannot prove the declared difference."""


TensorMap = Mapping[str, Tensor]
MutableTensorMap = MutableMapping[str, Tensor]
OracleCallable = Callable[[TensorMap, MutableTensorMap], Mapping[str, Tensor]]


@dataclass(frozen=True, slots=True)
class OracleSnapshot:
    outputs: dict[str, Tensor]
    mutated_state: dict[str, Tensor]


@dataclass(frozen=True, slots=True)
class ReductionWitnessMaterial:
    """Complete reduction rows for the reference and candidate evaluation trees."""

    reference_operands: Tensor
    reference_indices: Tensor
    candidate_operands: Tensor
    candidate_indices: Tensor


@dataclass(frozen=True, slots=True)
class PartitionedReductionWitnessMaterial:
    """Complete BF16 product factors and logical TP2 ownership for row linear."""

    left_operands: Tensor
    right_operands: Tensor
    logical_indices: Tensor
    partition_ids: Tensor


@dataclass(frozen=True, slots=True)
class ScaleWitnessMaterial:
    """Inputs needed to replay the declared scale-cast ABI from first principles."""

    baseline_operands: Tensor
    scale: Tensor
    abi: ScaleCastABI


@dataclass(frozen=True, slots=True)
class ReductionMaterialIdentity:
    operands_digest: str
    indices_digest: str
    reduction_length: int


@dataclass(frozen=True, slots=True)
class PartitionedReductionMaterialIdentity:
    operands_digest: str
    indices_digest: str
    partitions_digest: str
    reduction_length: int


def run_oracle(
    oracle: OracleCallable,
    inputs: TensorMap,
    initial_state: TensorMap,
) -> OracleSnapshot:
    """Run an independent eager oracle on isolated inputs and mutable state."""

    isolated_inputs = clone_tensor_map(inputs, "input")
    isolated_state = clone_tensor_map(initial_state, "state")
    outputs = oracle(isolated_inputs, isolated_state)
    checked_outputs = clone_tensor_map(outputs, "oracle output")
    checked_state = clone_tensor_map(isolated_state, "oracle state")
    return OracleSnapshot(outputs=checked_outputs, mutated_state=checked_state)


def clone_tensor_map(values: Mapping[str, Tensor], kind: str) -> dict[str, Tensor]:
    if any(not isinstance(key, str) or not key for key in values):
        raise ValueError(f"{kind} names must be non-empty strings")
    if any(not isinstance(value, Tensor) for value in values.values()):
        raise TypeError(f"all {kind} values must be tensors")
    return {key: value.detach().clone() for key, value in values.items()}


def reduction_material_identity(
    material: ReductionWitnessMaterial,
) -> ReductionMaterialIdentity:
    """Validate full coverage and return identities of real, canonical operands."""

    reference, candidate = _canonical_reduction_operands(material)
    operands_digest = stable_digest(
        {
            "reference": tensor_digest(reference),
            "candidate": tensor_digest(candidate),
        }
    )
    indices_digest = stable_digest(
        {
            "reference": tensor_digest(material.reference_indices),
            "candidate": tensor_digest(material.candidate_indices),
        }
    )
    return ReductionMaterialIdentity(
        operands_digest=operands_digest,
        indices_digest=indices_digest,
        reduction_length=reference.shape[-1],
    )


def partitioned_reduction_material_identity(
    material: PartitionedReductionWitnessMaterial,
) -> PartitionedReductionMaterialIdentity:
    """Validate complete TP ownership and bind the real BF16 factors."""

    left, right, partitions = _canonical_partitioned_operands(material)
    return _partitioned_identity(material, left, right, partitions)


def verify_reduction_witness(
    expected: Tensor,
    actual: Tensor,
    policy: ComparisonPolicy,
    context: ComparisonContext,
    material: ReductionWitnessMaterial,
) -> VerifiedReductionWitness:
    """Recompute coverage, identities, and both reduction-tree error bounds."""

    if policy.difference_class is not DifferenceClass.REDUCTION_ORDER:
        raise WitnessVerificationError("reduction material requires a reduction policy")
    proof = policy.reduction
    assert proof is not None
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        raise WitnessVerificationError("oracle and candidate result structures differ")
    if expected.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise WitnessVerificationError("reduction witness requires a supported float output")
    if not _all_finite(expected) or not _all_finite(actual):
        raise WitnessVerificationError("non-finite result cannot receive a witness")

    reference, candidate = _canonical_reduction_operands(material)
    if reference.shape[:-1] != expected.shape:
        raise WitnessVerificationError("reduction rows do not cover every output element")
    identity = reduction_material_identity(material)
    if identity.reduction_length != proof.reduction_length:
        raise WitnessVerificationError("declared reduction length does not match material")
    if identity.operands_digest != proof.operands_digest:
        raise WitnessVerificationError("operand digest does not match reduction proof")
    if identity.indices_digest != proof.indices_digest:
        raise WitnessVerificationError("index digest does not match reduction proof")

    # FP32 operands convert exactly to FP64.  Use an independent CPU FP64
    # reduction as the anchor and explicitly include its Higham bound.  This
    # keeps complete large-state witnesses practical without weakening them to
    # samples or silently trusting a vendor FP32 reduction.
    reference_rows = reference.detach().to(device="cpu", dtype=torch.float64)
    candidate_rows = candidate.detach().to(device="cpu", dtype=torch.float64)
    anchor = reference_rows.sum(dim=-1).reshape(expected.shape)
    reference_sum_abs = reference_rows.abs().sum(dim=-1).reshape(expected.shape)
    candidate_sum_abs = candidate_rows.abs().sum(dim=-1).reshape(expected.shape)
    if not all(_all_finite(value) for value in (anchor, reference_sum_abs, candidate_sum_abs)):
        raise WitnessVerificationError("reduction material overflows the finite oracle domain")
    anchor_error = reference_sum_abs * _gamma_from_unit(
        max(proof.reduction_length - 1, 0), 2.0**-53, "independent FP64 anchor"
    )

    output_anchor = anchor.to(dtype=expected.dtype)
    if not _all_finite(output_anchor):
        raise WitnessVerificationError("exact reduction is not representable as a finite output")
    output_rounding = finite_ulp_size(output_anchor)
    steps = max(proof.reduction_length - 1, 0)
    reference_gamma = _gamma(steps, proof.accumulator_dtype, "reference")
    candidate_gamma = _gamma(steps, proof.effective_candidate_dtype, "candidate")
    reference_bound = reference_sum_abs * reference_gamma + anchor_error + output_rounding
    candidate_bound = candidate_sum_abs * candidate_gamma + anchor_error + output_rounding
    if not _valid_bound(reference_bound) or not _valid_bound(candidate_bound):
        raise WitnessVerificationError("reduction error envelope is non-finite")

    expected_cpu = expected.detach().to(device="cpu", dtype=torch.float64)
    actual_cpu = actual.detach().to(device="cpu", dtype=torch.float64)
    observed_reference_error = (expected_cpu - anchor).abs()
    if bool((observed_reference_error > reference_bound).any().item()):
        raise WitnessVerificationError(
            "oracle output is inconsistent with the supplied reduction operands"
        )
    observed_candidate_error = (actual_cpu - anchor).abs()
    if bool((observed_candidate_error > candidate_bound).any().item()):
        raise WitnessVerificationError(
            "candidate output is inconsistent with its supplied reduction tree"
        )

    return _issue_reduction_witness(
        expected=expected,
        actual=actual,
        policy=policy,
        context=context,
        operands_digest=identity.operands_digest,
        indices_digest=identity.indices_digest,
        reference_bound=reference_bound.to(expected.device),
        candidate_bound=candidate_bound.to(expected.device),
    )


def verify_partitioned_reduction_witness(
    expected: Tensor,
    actual: Tensor,
    policy: ComparisonPolicy,
    context: ComparisonContext,
    material: PartitionedReductionWitnessMaterial,
) -> VerifiedReductionWitness:
    """Prove the fixed BF16 row-linear -> BF16 TP reduction hierarchy."""

    if policy.difference_class is not DifferenceClass.PARTITIONED_REDUCTION:
        raise WitnessVerificationError("partitioned material requires a partitioned policy")
    proof = policy.partitioned_reduction
    assert isinstance(proof, PartitionedReductionProof)
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        raise WitnessVerificationError("oracle and candidate result structures differ")
    if expected.dtype is not torch.bfloat16:
        raise WitnessVerificationError("partitioned reduction requires BF16 output")
    if not _all_finite(expected) or not _all_finite(actual):
        raise WitnessVerificationError("non-finite result cannot receive a witness")

    left, right, partitions = _canonical_partitioned_operands(material)
    partition_count = int(partitions.max().item()) + 1
    if partition_count != proof.partition_count:
        raise WitnessVerificationError("declared partition count does not match material")
    if left.shape[:-1] != expected.shape:
        raise WitnessVerificationError("product operands do not cover every output element")
    identity = _partitioned_identity(material, left, right, partitions)
    if identity.reduction_length != proof.reduction_length:
        raise WitnessVerificationError("declared reduction length does not match material")
    if identity.operands_digest != proof.operands_digest:
        raise WitnessVerificationError("operand digest does not match partitioned proof")
    if identity.indices_digest != proof.indices_digest:
        raise WitnessVerificationError("index digest does not match partitioned proof")
    if identity.partitions_digest != proof.partitions_digest:
        raise WitnessVerificationError("partition digest does not match partitioned proof")

    products = left.detach().to(device="cpu", dtype=torch.float64) * right.detach().to(
        device="cpu", dtype=torch.float64
    )
    if not _all_finite(products):
        raise WitnessVerificationError("BF16 products must be finite")
    product_error = _fp32_product_error_bound(products)
    exact_sum, reference_bound, candidate_bound = _partitioned_reduction_bounds(
        products,
        product_error,
        partitions.detach().to(device="cpu", dtype=torch.int64),
    )
    output_anchor = exact_sum.to(torch.bfloat16)
    if not _all_finite(output_anchor):
        raise WitnessVerificationError("exact reduction is not representable as BF16 output")
    if not _valid_bound(reference_bound) or not _valid_bound(candidate_bound):
        raise WitnessVerificationError("partitioned reduction envelope is non-finite")

    exact_sum = exact_sum.to(expected.device)
    reference_bound = reference_bound.to(expected.device)
    candidate_bound = candidate_bound.to(expected.device)
    if bool(((expected.to(torch.float64) - exact_sum).abs() > reference_bound).any().item()):
        raise WitnessVerificationError(
            "oracle output is inconsistent with the complete product operands"
        )
    if bool(((actual.to(torch.float64) - exact_sum).abs() > candidate_bound).any().item()):
        raise WitnessVerificationError(
            "candidate output is inconsistent with the partitioned reduction tree"
        )

    return _issue_reduction_witness(
        expected=expected,
        actual=actual,
        policy=policy,
        context=context,
        operands_digest=identity.operands_digest,
        indices_digest=identity.indices_digest,
        partitions_digest=identity.partitions_digest,
        difference_class=DifferenceClass.PARTITIONED_REDUCTION,
        reference_bound=reference_bound,
        candidate_bound=candidate_bound,
    )


def verify_observed_bf16_linear_sample(
    *,
    x: Tensor,
    weight: Tensor,
    output: Tensor,
    input_rows: tuple[int, ...],
    output_columns: tuple[int, ...],
    operand_binding_digest: str,
    artifact_digest: str,
    launch_specialization_digest: str,
    progress: Callable[[str], None] | None = None,
) -> VerifiedBF16LinearSampleWitness:
    """Enclose deterministic observed coordinates without relaunching the GEMM."""

    for label, digest in (
        ("operand binding", operand_binding_digest),
        ("artifact", artifact_digest),
        ("launch specialization", launch_specialization_digest),
    ):
        _require_content_digest(digest, label)
    if any(tensor.layout is not torch.strided for tensor in (x, weight, output)):
        raise WitnessVerificationError("linear sample requires strided tensors")
    if x.ndim != 2 or weight.ndim != 2 or output.ndim != 2:
        raise WitnessVerificationError("linear sample requires rank-two tensors")
    m, k = x.shape
    n, weight_k = weight.shape
    if k <= 0 or weight_k != k or output.shape != (m, n):
        raise WitnessVerificationError("linear sample shapes do not form one complete GEMM")
    if any(tensor.dtype is not torch.bfloat16 for tensor in (x, weight, output)):
        raise WitnessVerificationError("linear sample requires BF16 tensors")
    if any(tensor.device != x.device for tensor in (weight, output)):
        raise WitnessVerificationError("linear sample tensors must share one device")
    _validate_sample_indices(input_rows, m, "input rows")
    _validate_sample_indices(output_columns, n, "output columns")
    if progress is not None:
        progress("validated")

    full_rows = input_rows == tuple(range(m))
    full_columns = output_columns == tuple(range(n))
    if full_rows and full_columns:
        sampled_x_device = x
        sampled_weight_device = weight
        sampled_output_device = output
    elif x.device.type == "cpu":
        sampled_x_device = _cpu_bf16_select_rows(x, input_rows)
        sampled_weight_device = _cpu_bf16_select_rows(weight, output_columns)
        sampled_output_device = _cpu_bf16_select_matrix(
            output, input_rows, output_columns
        )
    else:
        row_index = torch.tensor(input_rows, dtype=torch.int64, device=x.device)
        column_index = torch.tensor(output_columns, dtype=torch.int64, device=x.device)
        sampled_x_device = x.index_select(0, row_index).detach()
        sampled_weight_device = weight.index_select(0, column_index).detach()
        sampled_output_device = (
            output.index_select(1, column_index)
            .index_select(0, row_index)
            .detach()
        )
    if progress is not None:
        progress("selected")
    for label, values in (
        ("x", sampled_x_device),
        ("weight", sampled_weight_device),
        ("output", sampled_output_device),
    ):
        _check_linear_bf16_values(values, label)
        if progress is not None:
            progress(f"{label}_checked")
    if progress is not None:
        progress("values_checked")
    sampled_x = sampled_x_device.to(device="cpu", copy=True).contiguous()
    sampled_weight = sampled_weight_device.to(device="cpu", copy=True).contiguous()
    sampled_output = sampled_output_device.to(device="cpu", copy=True).contiguous()
    if progress is not None:
        progress("copied")

    x64, weight64 = sampled_x.to(torch.float64), sampled_weight.to(torch.float64)
    abs_x, abs_weight = x64.abs(), weight64.abs()
    bf16_tiny = float(torch.finfo(torch.bfloat16).tiny)
    x_subnormal = (abs_x != 0) & (abs_x < bf16_tiny)
    weight_subnormal = (abs_weight != 0) & (abs_weight < bf16_tiny)
    has_bf16_subnormal = bool(x_subnormal.any() or weight_subnormal.any())
    min_x, max_x = _absolute_stats(abs_x)
    min_weight, max_weight = _absolute_stats(abs_weight)
    fp32_max = float(torch.finfo(torch.float32).max)
    if max_x * max_weight > fp32_max:
        raise WitnessVerificationError("linear sample can overflow an FP32 product")
    if progress is not None:
        progress("stats")

    anchor = torch.mm(x64, weight64.t())
    sum_abs = torch.mm(abs_x, abs_weight.t())
    if (
        math.isinf(min_x)
        or math.isinf(min_weight)
        or min_x * min_weight >= torch.finfo(torch.float32).tiny
    ) and not has_bf16_subnormal:
        product_error = torch.zeros_like(anchor)
    else:
        exact_products = x64[:, None, :] * weight64[None, :, :]
        product_error_per_term = _fp32_product_error_bound(exact_products)
        if has_bf16_subnormal:
            input_flush = x_subnormal[:, None, :] | weight_subnormal[None, :, :]
            product_error_per_term = torch.maximum(
                product_error_per_term,
                torch.where(input_flush, exact_products.abs(), 0.0),
            )
        product_error = product_error_per_term.sum(dim=-1)
    if progress is not None:
        progress("anchor")
    gamma32 = _gamma(k, ScalarDType.FP32, "sample FP32")
    gamma64 = _gamma_from_unit(k, 2.0**-53, "sample FP64")
    if not all(_all_finite(value) for value in (anchor, sum_abs, product_error)):
        raise WitnessVerificationError("linear sample FP64 enclosure is non-finite")
    sum_abs.div_(1.0 - gamma64)
    if float(sum_abs.max().item()) > fp32_max:
        raise WitnessVerificationError("linear sample reduction can overflow FP32")
    bounds = sum_abs.mul(gamma64)
    bounds.add_(product_error)
    bounds.add_((sum_abs + product_error) * gamma32)
    bounds.add_((anchor.abs() + bounds) * (2.0**-8))
    bounds.add_(_half_subnormal(ScalarDType.BF16))
    bf16_subnormal = anchor.abs() < torch.finfo(torch.bfloat16).tiny
    bounds.add_(torch.where(bf16_subnormal, anchor.abs(), torch.zeros_like(anchor)))
    difference = sampled_output.to(torch.float64).sub(anchor).abs()
    if not _valid_bound(bounds) or not _all_finite(difference):
        raise WitnessVerificationError("linear sample comparison is non-finite")
    if bool((difference > bounds).any().item()):
        raise WitnessVerificationError(
            "observed sample exceeds FP32-reassociation plus BF16-RNE bound"
        )
    if progress is not None:
        progress("bounded")

    material_digest = stable_digest(
        {
            "operation": "bf16(sum[k](bf16(x[m,k])*bf16(w[n,k])))",
            "sampled_x": tensor_digest(sampled_x),
            "sampled_weight": tensor_digest(sampled_weight),
            "candidate_shape": (m, n),
        }
    )
    if progress is not None:
        progress("digested")
    return _issue_bf16_linear_sample_witness(
        operand_binding_digest=operand_binding_digest,
        artifact_digest=artifact_digest,
        launch_specialization_digest=launch_specialization_digest,
        material_digest=material_digest,
        sampled_output=sampled_output,
        bounds=bounds,
        candidate_shape=(m, n),
        reduction_length=k,
        input_rows=input_rows,
        output_columns=output_columns,
        max_observed_error=float(difference.max().item()),
    )


def verify_scale_witness(
    expected: Tensor,
    actual: Tensor,
    policy: ComparisonPolicy,
    context: ComparisonContext,
    material: ScaleWitnessMaterial,
) -> VerifiedScaleWitness:
    """Replay both sides of the scale-cast ABI and issue an exact simulation witness."""

    if policy.difference_class is not DifferenceClass.SCALE_DTYPE:
        raise WitnessVerificationError("scale material requires a scale policy")
    proof = policy.scale
    assert proof is not None
    abi = material.abi
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        raise WitnessVerificationError("oracle and candidate result structures differ")
    if proof.cast_location != abi.cast_location:
        raise WitnessVerificationError("scale proof and ABI cast locations differ")
    observed_abi_digest = scale_abi_digest(abi)
    if proof.abi_digest != observed_abi_digest:
        raise WitnessVerificationError("scale ABI digest does not match proof")
    observed_scale_digest = tensor_digest(material.scale)
    if proof.scale_digest != observed_scale_digest:
        raise WitnessVerificationError("scale digest does not match proof")
    if not _all_finite(material.baseline_operands) or not _all_finite(material.scale):
        raise WitnessVerificationError("scale witness inputs must be finite")

    reference_simulated = _simulate_scale(
        material.baseline_operands,
        material.scale,
        scale_dtype=proof.reference_dtype,
        abi=abi,
    )
    candidate_simulated = _simulate_scale(
        material.baseline_operands,
        material.scale,
        scale_dtype=proof.candidate_dtype,
        abi=abi,
    )
    if reference_simulated.shape != expected.shape or reference_simulated.dtype != expected.dtype:
        raise WitnessVerificationError("scale ABI output structure differs from oracle output")
    if not _all_finite(reference_simulated) or not _all_finite(candidate_simulated):
        raise WitnessVerificationError("scale ABI simulation produced non-finite output")
    if tensor_digest(reference_simulated) != tensor_digest(expected):
        raise WitnessVerificationError("oracle output is inconsistent with scale ABI replay")
    if tensor_digest(candidate_simulated) != tensor_digest(actual):
        raise WitnessVerificationError("candidate output is inconsistent with scale ABI replay")

    return _issue_scale_witness(
        expected=expected,
        actual=actual,
        policy=policy,
        context=context,
        operands_digest=tensor_digest(material.baseline_operands),
        scale_digest=observed_scale_digest,
        abi_digest=observed_abi_digest,
        simulated=candidate_simulated,
    )


def _canonical_reduction_operands(
    material: ReductionWitnessMaterial,
) -> tuple[Tensor, Tensor]:
    tensors = (
        material.reference_operands,
        material.reference_indices,
        material.candidate_operands,
        material.candidate_indices,
    )
    if any(tensor.layout is not torch.strided for tensor in tensors):
        raise WitnessVerificationError("reduction witness requires strided tensors")
    if material.reference_operands.shape != material.reference_indices.shape or (
        material.candidate_operands.shape != material.candidate_indices.shape
    ):
        raise WitnessVerificationError("every reduction operand requires one logical index")
    if material.reference_operands.shape != material.candidate_operands.shape:
        raise WitnessVerificationError("reference and candidate reduction shapes differ")
    if material.reference_operands.ndim < 1 or material.reference_operands.shape[-1] <= 0:
        raise WitnessVerificationError("reduction material requires non-empty rows")
    if material.reference_operands.dtype != material.candidate_operands.dtype:
        raise WitnessVerificationError("reference and candidate operand dtypes differ")
    if material.reference_operands.device != material.reference_indices.device or (
        material.candidate_operands.device != material.candidate_indices.device
    ):
        raise WitnessVerificationError("each operand tensor must share its index-map device")
    if not material.reference_operands.is_floating_point():
        raise WitnessVerificationError("reduction operands must be floating point")
    if any(
        indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        for indices in (material.reference_indices, material.candidate_indices)
    ):
        raise WitnessVerificationError("reduction indices must use an integer dtype")
    if not _all_finite(material.reference_operands) or not _all_finite(material.candidate_operands):
        raise WitnessVerificationError("reduction operands must be finite")

    reduction_length = material.reference_operands.shape[-1]
    expected_indices = torch.arange(
        reduction_length,
        dtype=torch.int64,
        device=material.reference_indices.device,
    ).expand(material.reference_indices.shape)
    reference_order = torch.argsort(material.reference_indices.to(torch.int64), dim=-1)
    candidate_order = torch.argsort(material.candidate_indices.to(torch.int64), dim=-1)
    sorted_reference_indices = torch.gather(
        material.reference_indices.to(torch.int64), -1, reference_order
    )
    sorted_candidate_indices = torch.gather(
        material.candidate_indices.to(torch.int64), -1, candidate_order
    )
    if not torch.equal(sorted_reference_indices, expected_indices) or not torch.equal(
        sorted_candidate_indices.to(expected_indices.device), expected_indices
    ):
        raise WitnessVerificationError(
            "each reduction row must cover every logical index exactly once"
        )

    reference = torch.gather(material.reference_operands, -1, reference_order)
    candidate_order = candidate_order.to(material.candidate_operands.device)
    candidate = torch.gather(material.candidate_operands, -1, candidate_order)
    if tensor_digest(reference) != tensor_digest(candidate):
        raise WitnessVerificationError(
            "reference and candidate trees do not reduce the same indexed operands"
        )
    return reference, candidate


def _canonical_partitioned_operands(
    material: PartitionedReductionWitnessMaterial,
) -> tuple[Tensor, Tensor, Tensor]:
    tensors = (
        material.left_operands,
        material.right_operands,
        material.logical_indices,
        material.partition_ids,
    )
    if any(tensor.layout is not torch.strided for tensor in tensors):
        raise WitnessVerificationError("partitioned witness requires strided tensors")
    if any(tensor.shape != material.left_operands.shape for tensor in tensors[1:]):
        raise WitnessVerificationError("every product factor requires an index and partition")
    if material.left_operands.ndim < 1 or material.left_operands.numel() == 0:
        raise WitnessVerificationError("partitioned material requires non-empty rows")
    if material.left_operands.dtype is not torch.bfloat16 or (
        material.right_operands.dtype is not torch.bfloat16
    ):
        raise WitnessVerificationError("partitioned product operands must be BF16")
    if any(tensor.device != material.left_operands.device for tensor in tensors[1:]):
        raise WitnessVerificationError("partitioned operands and maps must share one device")
    if any(
        values.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        for values in (material.logical_indices, material.partition_ids)
    ):
        raise WitnessVerificationError("partitioned index maps must use an integer dtype")
    if not _all_finite(material.left_operands) or not _all_finite(material.right_operands):
        raise WitnessVerificationError("partitioned product operands must be finite")

    length = material.left_operands.shape[-1]
    expected_indices = torch.arange(
        length, dtype=torch.int64, device=material.logical_indices.device
    ).expand(material.logical_indices.shape)
    order = torch.argsort(material.logical_indices.to(torch.int64), dim=-1)
    indices = torch.gather(material.logical_indices.to(torch.int64), -1, order)
    if not torch.equal(indices, expected_indices):
        raise WitnessVerificationError(
            "each partitioned row must cover every logical index exactly once"
        )
    left = torch.gather(material.left_operands, -1, order)
    right = torch.gather(material.right_operands, -1, order)
    partitions = torch.gather(material.partition_ids.to(torch.int64), -1, order)
    partition_count = int(partitions.max().item()) + 1
    if (
        partition_count < 2
        or bool((partitions < 0).any().item())
        or any(
            not bool((partitions == partition).any(dim=-1).all().item())
            for partition in range(partition_count)
        )
    ):
        raise WitnessVerificationError("every row must contain every declared TP partition")
    partition_rows = partitions.reshape(-1, length)
    if not torch.equal(partition_rows, partition_rows[:1].expand_as(partition_rows)):
        raise WitnessVerificationError("TP partition ownership must be identical for every row")
    return left, right, partitions


def _partitioned_identity(
    material: PartitionedReductionWitnessMaterial,
    left: Tensor,
    right: Tensor,
    partitions: Tensor,
) -> PartitionedReductionMaterialIdentity:
    return PartitionedReductionMaterialIdentity(
        operands_digest=stable_digest({"left": tensor_digest(left), "right": tensor_digest(right)}),
        indices_digest=tensor_digest(material.logical_indices),
        partitions_digest=tensor_digest(partitions),
        reduction_length=left.shape[-1],
    )


def _partitioned_reduction_bounds(
    products: Tensor,
    product_error: Tensor,
    partitions: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Vectorized complete-row enclosure for a local-linear plus TP reduction.

    The previous implementation converted every K-wide output row to Python
    lists and applied ``math.fsum`` element by element.  That was exact enough
    but made a real model audit take minutes per projection.  Here the anchor
    is an independent CPU FP64 reduction.  Its own Higham error is explicitly
    included, so vectorization changes cost, not the accepted arithmetic.
    """

    length = products.shape[-1]
    rows = products.reshape(-1, length)
    errors = product_error.reshape(-1, length)
    owners = partitions.reshape(-1, length)
    fp64_gamma = _gamma_from_unit(max(length - 1, 0), 2.0**-53, "FP64 anchor")

    anchor = rows.sum(dim=-1)
    sum_abs = rows.abs().sum(dim=-1).div_(1.0 - fp64_gamma)
    product_rounding = errors.sum(dim=-1).div_(1.0 - fp64_gamma)
    anchor_error = sum_abs * fp64_gamma
    reference_reduction = (sum_abs + product_rounding) * _gamma(
        max(length - 1, 0), ScalarDType.FP32, "reference partitioned"
    )
    reference_reduction.add_(
        max(length - 1, 0) * _half_subnormal(ScalarDType.FP32)
    )
    reference_cast_input = (
        anchor.abs() + anchor_error + product_rounding + reference_reduction
    )
    _check_cast_input(reference_cast_input, ScalarDType.BF16)
    reference_bound = (
        anchor_error
        + product_rounding
        + reference_reduction
        + reference_cast_input * _unit_roundoff(ScalarDType.BF16)
        + _half_subnormal(ScalarDType.BF16)
    )

    partition_count = int(owners.max().item()) + 1
    candidate_compute = product_rounding.clone()
    partial_abs_upper = torch.zeros_like(anchor)
    for partition in range(partition_count):
        indices = torch.nonzero(owners[0] == partition, as_tuple=False).flatten()
        shard = rows.index_select(1, indices)
        shard_errors = errors.index_select(1, indices)
        shard_length = indices.numel()
        shard_gamma64 = _gamma_from_unit(
            max(shard_length - 1, 0), 2.0**-53, "FP64 shard anchor"
        )
        shard_anchor = shard.sum(dim=-1)
        shard_abs = shard.abs().sum(dim=-1).div_(1.0 - shard_gamma64)
        shard_anchor_error = shard_abs * shard_gamma64
        shard_product_error = shard_errors.sum(dim=-1).div_(1.0 - shard_gamma64)
        local = (shard_abs + shard_product_error) * _gamma(
            max(shard_length - 1, 0), ScalarDType.FP32, "candidate partitioned"
        )
        local.add_(max(shard_length - 1, 0) * _half_subnormal(ScalarDType.FP32))
        partial_input = (
            shard_anchor.abs() + shard_anchor_error + shard_product_error + local
        )
        _check_cast_input(partial_input, ScalarDType.BF16)
        partial_cast = (
            partial_input * _unit_roundoff(ScalarDType.BF16)
            + _half_subnormal(ScalarDType.BF16)
        )
        candidate_compute.add_(local + partial_cast)
        partial_abs_upper.add_(partial_input + partial_cast)

    collective_compute = partial_abs_upper * _gamma(
        max(partition_count - 1, 0), ScalarDType.FP32, "collective partitioned"
    )
    collective_compute.add_(
        max(partition_count - 1, 0) * _half_subnormal(ScalarDType.FP32)
    )
    collective_input = partial_abs_upper + collective_compute
    _check_cast_input(collective_input, ScalarDType.BF16)
    collective_cast = (
        collective_input * _unit_roundoff(ScalarDType.BF16)
        + _half_subnormal(ScalarDType.BF16)
    )
    candidate_bound = anchor_error + candidate_compute + collective_compute + collective_cast
    if not all(
        _all_finite(value)
        for value in (anchor, reference_bound, candidate_bound)
    ):
        raise WitnessVerificationError(
            "partitioned reduction overflows the finite oracle domain"
        )

    shape = products.shape[:-1]
    return anchor.reshape(shape), reference_bound.reshape(shape), candidate_bound.reshape(shape)


def _check_cast_input(values: Tensor, dtype: ScalarDType) -> None:
    if not _all_finite(values) or bool((values < 0).any().item()):
        raise WitnessVerificationError("cast bound requires a finite magnitude")
    if float(values.max().item()) > float(torch.finfo(_torch_dtype(dtype)).max):
        raise WitnessVerificationError("partitioned reduction intermediate can overflow")


def _fp32_product_error_bound(exact_products: Tensor) -> Tensor:
    """Bound BF16 product materialization, including FP32 subnormal flushing.

    A BF16-by-BF16 product has at most 16 significant bits and is therefore
    exactly representable by FP32 whenever its exponent is normal.  Products
    in the FP32 subnormal range may be gradually rounded or flushed to zero by
    an accelerator execution path.  Bind that ambiguity to each real product:
    the larger of gradual-rounding error and flush-to-zero error.  This is not
    a global tolerance; zeroing any normal product remains outside the proof.
    """

    rounded = exact_products.to(torch.float32).to(torch.float64)
    if not _all_finite(rounded):
        raise WitnessVerificationError("BF16 products overflow FP32")
    gradual_error = (exact_products - rounded).abs()
    subnormal = exact_products.abs() < torch.finfo(torch.float32).tiny
    return torch.where(subnormal, torch.maximum(gradual_error, exact_products.abs()), gradual_error)


def _simulate_scale(
    operands: Tensor,
    scale: Tensor,
    *,
    scale_dtype: ScalarDType,
    abi: ScaleCastABI,
) -> Tensor:
    compute_dtype = _torch_dtype(abi.compute_dtype)
    output_dtype = _torch_dtype(abi.output_dtype)
    try:
        operand_value = operands.to(dtype=compute_dtype)
        scale_value = scale.to(device=operands.device, dtype=_torch_dtype(scale_dtype))
        scale_value = scale_value.to(dtype=compute_dtype)
        return torch.mul(operand_value, scale_value).to(dtype=output_dtype)
    except RuntimeError as error:
        raise WitnessVerificationError("scale ABI operands are not broadcast-compatible") from error


def _torch_dtype(dtype: ScalarDType) -> torch.dtype:
    try:
        return {
            ScalarDType.BF16: torch.bfloat16,
            ScalarDType.FP16: torch.float16,
            ScalarDType.FP32: torch.float32,
        }[dtype]
    except KeyError as error:
        raise WitnessVerificationError(f"unsupported scale ABI dtype: {dtype}") from error


def _gamma(steps: int, dtype: ScalarDType, tree_name: str) -> float:
    unit_roundoff = _unit_roundoff(dtype)
    product = steps * unit_roundoff
    if product >= 1.0:
        raise WitnessVerificationError(
            f"{tree_name} reduction bound is undefined for {steps} rounding steps"
        )
    gamma = product / (1.0 - product)
    if not math.isfinite(gamma):
        raise WitnessVerificationError(f"{tree_name} reduction gamma is non-finite")
    return gamma


def _roundoff_bound(sum_abs: float, steps: int, dtype: ScalarDType) -> float:
    return sum_abs * _gamma(steps, dtype, "partitioned") + steps * _half_subnormal(dtype)


def _cast_bound(abs_input_upper: float, dtype: ScalarDType) -> float:
    if not math.isfinite(abs_input_upper) or abs_input_upper < 0:
        raise WitnessVerificationError("cast bound requires a finite magnitude")
    cast = torch.tensor(abs_input_upper, dtype=torch.float64).to(_torch_dtype(dtype))
    if not _all_finite(cast):
        raise WitnessVerificationError("partitioned reduction intermediate can overflow")
    return abs_input_upper * _unit_roundoff(dtype) + _half_subnormal(dtype)


def _unit_roundoff(dtype: ScalarDType) -> float:
    return {
        ScalarDType.BF16: 2.0**-8,
        ScalarDType.FP16: 2.0**-11,
        ScalarDType.FP32: 2.0**-24,
    }[dtype]


def _half_subnormal(dtype: ScalarDType) -> float:
    return {
        ScalarDType.BF16: 2.0**-134,
        ScalarDType.FP16: 2.0**-25,
        ScalarDType.FP32: 2.0**-150,
    }[dtype]


def _gamma_from_unit(steps: int, unit_roundoff: float, label: str) -> float:
    product = steps * unit_roundoff
    if product >= 1.0:
        raise WitnessVerificationError(f"{label} reduction envelope is undefined")
    return product / (1.0 - product)


def _validate_sample_indices(indices: tuple[int, ...], length: int, label: str) -> None:
    if (
        type(indices) is not tuple
        or not indices
        or any(type(index) is not int or index < 0 or index >= length for index in indices)
        or tuple(sorted(set(indices))) != indices
    ):
        raise WitnessVerificationError(
            f"{label} must be a non-empty strictly increasing in-range tuple"
        )


def _check_linear_bf16_values(values: Tensor, label: str) -> None:
    if values.dtype is torch.bfloat16:
        # Qualification must not launch a device reduction while peer ranks are
        # waiting at a boundary.  Copy the logical view, then inspect its BF16
        # exponent bits independently on the host.
        owned = values.detach().to(device="cpu", copy=True).contiguous()
        raw = owned.view(torch.uint8).numpy().tobytes()
        bits = np.frombuffer(raw, dtype="<u2")
        exponent = np.bitwise_and(bits, np.uint16(0x7F80))
        if bool(np.any(exponent == np.uint16(0x7F80))):
            raise WitnessVerificationError(f"linear sample {label} is non-finite")
        return
    if not _all_finite(values):
        raise WitnessVerificationError(f"linear sample {label} is non-finite")


def _cpu_bf16_select_rows(values: Tensor, rows: tuple[int, ...]) -> Tensor:
    raw = (
        values.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    )
    bits = np.frombuffer(raw, dtype="<u2").reshape(values.shape)
    selected = np.ascontiguousarray(bits[np.asarray(rows, dtype=np.int64)])
    return torch.from_numpy(selected.view(np.int16)).view(torch.bfloat16)


def _cpu_bf16_select_matrix(
    values: Tensor,
    rows: tuple[int, ...],
    columns: tuple[int, ...],
) -> Tensor:
    raw = (
        values.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    )
    bits = np.frombuffer(raw, dtype="<u2").reshape(values.shape)
    selected = np.ascontiguousarray(
        bits[np.ix_(np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64))]
    )
    return torch.from_numpy(selected.view(np.int16)).view(torch.bfloat16)


def _absolute_stats(values: Tensor) -> tuple[float, float]:
    if values.device.type == "cpu":
        array = values.numpy()
        minimum = float(np.min(array, where=array != 0, initial=np.inf))
        return minimum, float(np.max(array, initial=0.0))
    nonzero = values[values != 0]
    minimum = float(nonzero.min().item()) if nonzero.numel() else math.inf
    return minimum, float(values.max().item())


def _require_content_digest(value: str, label: str) -> None:
    if not value.startswith("b2s256:") or len(value) != 71:
        raise WitnessVerificationError(f"{label} must be a content_digest digest")
    try:
        int(value[7:], 16)
    except ValueError as error:
        raise WitnessVerificationError(f"{label} must be a content_digest digest") from error


def _valid_bound(bound: Tensor) -> bool:
    return _all_finite(bound) and not bool((bound < 0).any().item())


def _all_finite(tensor: Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())
