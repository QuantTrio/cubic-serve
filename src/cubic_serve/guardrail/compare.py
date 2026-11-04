"""Fail-closed tensor comparison with narrowly typed error witnesses.

There is deliberately no process-wide tolerance. Non-exact comparisons require a
verified witness issued by :mod:`cubic_serve.guardrail.oracle`; callers cannot pass
an error magnitude or a precomputed "expected" tensor to this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from cubic_serve._compat import StrEnum
from typing import Any

import torch
from torch import Tensor

from cubic_serve.contracts.model import ScalarDType


class DifferenceClass(StrEnum):
    EXACT = "exact"
    REDUCTION_ORDER = "reduction_order"
    PARTITIONED_REDUCTION = "partitioned_reduction"
    SCALE_DTYPE = "scale_dtype"


@dataclass(frozen=True, slots=True)
class ReductionOrderProof:
    """Static declaration for a reduction-order waiver.

    The digests name the real logical operands and the two index maps. The
    accumulator fields model the reference and candidate trees independently.
    ``candidate_accumulator_dtype=None`` means both trees use the same dtype.
    """

    operands_digest: str
    indices_digest: str
    reduction_length: int
    accumulator_dtype: ScalarDType
    candidate_accumulator_dtype: ScalarDType | None = None

    def __post_init__(self) -> None:
        _require_digest(self.operands_digest, "operands_digest")
        _require_digest(self.indices_digest, "indices_digest")
        if self.reduction_length <= 0:
            raise ValueError("reduction length must be positive")
        for dtype in (self.accumulator_dtype, self.effective_candidate_dtype):
            if dtype not in (ScalarDType.BF16, ScalarDType.FP16, ScalarDType.FP32):
                raise ValueError("unsupported floating accumulator dtype")

    @property
    def effective_candidate_dtype(self) -> ScalarDType:
        return self.candidate_accumulator_dtype or self.accumulator_dtype


@dataclass(frozen=True, slots=True)
class PartitionedReductionProof:
    """Declaration for BF16 row-linear -> FP32 shard sums -> BF16 TP sum."""

    operands_digest: str
    indices_digest: str
    partitions_digest: str
    reduction_length: int
    partition_count: int = 2

    def __post_init__(self) -> None:
        _require_digest(self.operands_digest, "operands_digest")
        _require_digest(self.indices_digest, "indices_digest")
        _require_digest(self.partitions_digest, "partitions_digest")
        if self.reduction_length <= 0:
            raise ValueError("reduction length must be positive")
        if self.partition_count < 2:
            raise ValueError("partitioned reduction requires at least two TP shards")


@dataclass(frozen=True, slots=True)
class ScaleDTypeProof:
    """Static declaration for one explicitly modelled scale-cast ABI."""

    scale_digest: str
    abi_digest: str
    reference_dtype: ScalarDType
    candidate_dtype: ScalarDType
    cast_location: str

    def __post_init__(self) -> None:
        _require_digest(self.scale_digest, "scale_digest")
        _require_digest(self.abi_digest, "abi_digest")
        if not self.cast_location:
            raise ValueError("scale proof requires a cast location")
        if self.reference_dtype is self.candidate_dtype:
            raise ValueError("scale proof must describe an actual dtype change")


@dataclass(frozen=True, slots=True)
class ScaleCastABI:
    """The only scale waiver ABI in v0.1: cast scale, multiply, cast output.

    All casts are round-to-nearest-even as implemented by PyTorch for the named
    scalar dtype. Keeping the operation explicit avoids a generic tolerance from
    silently covering dequantization, activation, or reduction errors.
    """

    abi_id: str
    cast_location: str
    compute_dtype: ScalarDType
    output_dtype: ScalarDType
    operation: str = "multiply"
    rounding_mode: str = "nearest_even"

    def __post_init__(self) -> None:
        if not self.abi_id or not self.cast_location:
            raise ValueError("scale ABI identity and cast location are required")
        if self.operation != "multiply":
            raise ValueError("v0.1 only admits the explicit multiply scale ABI")
        if self.rounding_mode != "nearest_even":
            raise ValueError("v0.1 only admits nearest-even casts")
        for dtype in (self.compute_dtype, self.output_dtype):
            if dtype not in (ScalarDType.BF16, ScalarDType.FP16, ScalarDType.FP32):
                raise ValueError("scale ABI requires a supported floating dtype")


@dataclass(frozen=True, slots=True)
class ComparisonPolicy:
    difference_class: DifferenceClass
    reduction: ReductionOrderProof | None = None
    scale: ScaleDTypeProof | None = None
    partitioned_reduction: PartitionedReductionProof | None = None

    def __post_init__(self) -> None:
        if self.difference_class is DifferenceClass.EXACT:
            if any(
                proof is not None
                for proof in (self.reduction, self.scale, self.partitioned_reduction)
            ):
                raise ValueError("EXACT policy cannot carry an error proof")
        elif self.difference_class is DifferenceClass.REDUCTION_ORDER:
            if (
                self.reduction is None
                or self.scale is not None
                or self.partitioned_reduction is not None
            ):
                raise ValueError("REDUCTION_ORDER requires exactly one reduction proof")
        elif self.difference_class is DifferenceClass.PARTITIONED_REDUCTION:
            if (
                self.partitioned_reduction is None
                or self.reduction is not None
                or self.scale is not None
            ):
                raise ValueError(
                    "PARTITIONED_REDUCTION requires exactly one partitioned proof"
                )
        elif self.difference_class is DifferenceClass.SCALE_DTYPE and (
            self.scale is None
            or self.reduction is not None
            or self.partitioned_reduction is not None
        ):
            raise ValueError("SCALE_DTYPE requires exactly one scale proof")


@dataclass(frozen=True, slots=True)
class ComparisonContext:
    """Stable identities that bind evidence to one op and candidate launch."""

    op_signature_digest: str
    candidate_launch_digest: str

    def __post_init__(self) -> None:
        _require_digest(self.op_signature_digest, "op_signature_digest")
        _require_digest(self.candidate_launch_digest, "candidate_launch_digest")

    @property
    def digest(self) -> str:
        return stable_digest(
            {
                "op_signature_digest": self.op_signature_digest,
                "candidate_launch_digest": self.candidate_launch_digest,
            }
        )


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    passed: bool
    difference_class: DifferenceClass
    element_count: int
    first_mismatch_index: tuple[int, ...] | None
    max_absolute_error: float | None
    max_ulp_error: int | None
    max_allowed_error: float | None
    reason: str
    witness_digest: str | None = None
    context_digest: str | None = None


# This is an integrity boundary, not a Python security sandbox. The private token
# prevents accidental/manual witness construction; only the independently checking
# oracle factory below this package boundary calls the private issuer functions.
_WITNESS_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class VerifiedReductionWitness:
    expected_binding: str
    actual_binding: str
    policy_digest: str
    context_digest: str
    operands_digest: str
    indices_digest: str
    partitions_digest: str | None
    difference_class: DifferenceClass
    reference_bound_digest: str
    candidate_bound_digest: str
    _reference_bound: tuple[float, ...]
    _candidate_bound: tuple[float, ...]
    _shape: tuple[int, ...]
    witness_digest: str

    def __init__(
        self,
        *,
        expected_binding: str,
        actual_binding: str,
        policy_digest: str,
        context_digest: str,
        operands_digest: str,
        indices_digest: str,
        reference_bound: tuple[float, ...],
        candidate_bound: tuple[float, ...],
        shape: tuple[int, ...],
        partitions_digest: str | None = None,
        difference_class: DifferenceClass = DifferenceClass.REDUCTION_ORDER,
        _token: object | None = None,
    ) -> None:
        if _token is not _WITNESS_TOKEN:
            raise TypeError("verified witnesses may only be issued by the oracle")
        if difference_class not in (
            DifferenceClass.REDUCTION_ORDER,
            DifferenceClass.PARTITIONED_REDUCTION,
        ):
            raise TypeError("reduction witness requires a reduction difference class")
        if (partitions_digest is None) is (
            difference_class is DifferenceClass.PARTITIONED_REDUCTION
        ):
            raise TypeError("partitioned reduction witness requires a partition digest")
        # Force the equal-length check while both trees are still independently
        # represented. The comparator validates both immutable tuples again.
        tuple(zip(reference_bound, candidate_bound, strict=True))
        payload = {
            "kind": difference_class.value,
            "expected_binding": expected_binding,
            "actual_binding": actual_binding,
            "policy_digest": policy_digest,
            "context_digest": context_digest,
            "operands_digest": operands_digest,
            "indices_digest": indices_digest,
            "partitions_digest": partitions_digest,
            "reference_bound_digest": stable_digest(reference_bound),
            "candidate_bound_digest": stable_digest(candidate_bound),
            "shape": shape,
        }
        object.__setattr__(self, "expected_binding", expected_binding)
        object.__setattr__(self, "actual_binding", actual_binding)
        object.__setattr__(self, "policy_digest", policy_digest)
        object.__setattr__(self, "context_digest", context_digest)
        object.__setattr__(self, "operands_digest", operands_digest)
        object.__setattr__(self, "indices_digest", indices_digest)
        object.__setattr__(self, "partitions_digest", partitions_digest)
        object.__setattr__(self, "difference_class", difference_class)
        object.__setattr__(self, "reference_bound_digest", payload["reference_bound_digest"])
        object.__setattr__(self, "candidate_bound_digest", payload["candidate_bound_digest"])
        object.__setattr__(self, "_reference_bound", reference_bound)
        object.__setattr__(self, "_candidate_bound", candidate_bound)
        object.__setattr__(self, "_shape", shape)
        object.__setattr__(self, "witness_digest", stable_digest(payload))


@dataclass(frozen=True, slots=True, init=False)
class VerifiedScaleWitness:
    expected_binding: str
    actual_binding: str
    policy_digest: str
    context_digest: str
    operands_digest: str
    scale_digest: str
    abi_digest: str
    simulated_digest: str
    _simulated_bytes: bytes
    _shape: tuple[int, ...]
    _dtype: torch.dtype
    witness_digest: str

    def __init__(
        self,
        *,
        expected_binding: str,
        actual_binding: str,
        policy_digest: str,
        context_digest: str,
        operands_digest: str,
        scale_digest: str,
        abi_digest: str,
        simulated: Tensor,
        _token: object | None = None,
    ) -> None:
        if _token is not _WITNESS_TOKEN:
            raise TypeError("verified witnesses may only be issued by the oracle")
        simulated_bytes = _tensor_bytes(simulated)
        payload = {
            "kind": DifferenceClass.SCALE_DTYPE.value,
            "expected_binding": expected_binding,
            "actual_binding": actual_binding,
            "policy_digest": policy_digest,
            "context_digest": context_digest,
            "operands_digest": operands_digest,
            "scale_digest": scale_digest,
            "abi_digest": abi_digest,
            "simulated_digest": tensor_digest(simulated),
            "shape": tuple(simulated.shape),
            "dtype": str(simulated.dtype),
        }
        object.__setattr__(self, "expected_binding", expected_binding)
        object.__setattr__(self, "actual_binding", actual_binding)
        object.__setattr__(self, "policy_digest", policy_digest)
        object.__setattr__(self, "context_digest", context_digest)
        object.__setattr__(self, "operands_digest", operands_digest)
        object.__setattr__(self, "scale_digest", scale_digest)
        object.__setattr__(self, "abi_digest", abi_digest)
        object.__setattr__(self, "simulated_digest", payload["simulated_digest"])
        object.__setattr__(self, "_simulated_bytes", simulated_bytes)
        object.__setattr__(self, "_shape", tuple(simulated.shape))
        object.__setattr__(self, "_dtype", simulated.dtype)
        object.__setattr__(self, "witness_digest", stable_digest(payload))


@dataclass(frozen=True, slots=True, init=False)
class VerifiedBF16LinearSampleWitness:
    """Verifier-issued full-K enclosure for deterministic output coordinates."""

    operand_binding_digest: str
    artifact_digest: str
    launch_specialization_digest: str
    sample_identity_digest: str
    sampled_output_digest: str
    bounds_digest: str
    candidate_shape: tuple[int, int]
    reduction_length: int
    input_rows: tuple[int, ...]
    output_columns: tuple[int, ...]
    sampled_element_count: int
    max_allowed_error: float
    max_observed_error: float
    witness_digest: str

    def __init__(
        self,
        *,
        operand_binding_digest: str,
        artifact_digest: str,
        launch_specialization_digest: str,
        material_digest: str,
        sampled_output: Tensor,
        bounds: Tensor,
        candidate_shape: tuple[int, int],
        reduction_length: int,
        input_rows: tuple[int, ...],
        output_columns: tuple[int, ...],
        max_observed_error: float,
        _token: object | None = None,
    ) -> None:
        if _token is not _WITNESS_TOKEN:
            raise TypeError("verified witnesses may only be issued by the oracle")
        sample_identity = stable_digest(
            {
                "operand_binding_digest": operand_binding_digest,
                "artifact_digest": artifact_digest,
                "launch_specialization_digest": launch_specialization_digest,
                "material_digest": material_digest,
                "candidate_shape": candidate_shape,
                "input_rows": input_rows,
                "output_columns": output_columns,
                "reduction_length": reduction_length,
            }
        )
        payload = {
            "kind": "bf16_linear_sample",
            "operand_binding_digest": operand_binding_digest,
            "artifact_digest": artifact_digest,
            "launch_specialization_digest": launch_specialization_digest,
            "sample_identity_digest": sample_identity,
            "sampled_output_digest": tensor_digest(sampled_output),
            "bounds_digest": tensor_digest(bounds),
            "sampled_element_count": sampled_output.numel(),
            "max_allowed_error": float(bounds.max().item()),
            "max_observed_error": max_observed_error,
        }
        for name in (
            "operand_binding_digest",
            "artifact_digest",
            "launch_specialization_digest",
        ):
            object.__setattr__(self, name, payload[name])
        object.__setattr__(self, "sample_identity_digest", sample_identity)
        object.__setattr__(self, "sampled_output_digest", payload["sampled_output_digest"])
        object.__setattr__(self, "bounds_digest", payload["bounds_digest"])
        object.__setattr__(self, "candidate_shape", candidate_shape)
        object.__setattr__(self, "reduction_length", reduction_length)
        object.__setattr__(self, "input_rows", input_rows)
        object.__setattr__(self, "output_columns", output_columns)
        object.__setattr__(self, "sampled_element_count", sampled_output.numel())
        object.__setattr__(self, "max_allowed_error", payload["max_allowed_error"])
        object.__setattr__(self, "max_observed_error", max_observed_error)
        object.__setattr__(self, "witness_digest", stable_digest(payload))


VerifiedWitness = VerifiedReductionWitness | VerifiedScaleWitness


def compare_tensors(
    expected: Tensor,
    actual: Tensor,
    policy: ComparisonPolicy,
    *,
    context: ComparisonContext | None = None,
    witness: VerifiedWitness | None = None,
) -> ComparisonResult:
    """Compare candidate output against an oracle under one proved difference class."""

    structural_error = _structural_error(expected, actual)
    if structural_error is not None:
        return comparison_failure(policy.difference_class, expected.numel(), structural_error)
    if not _all_finite(expected) or not _all_finite(actual):
        return comparison_failure(
            policy.difference_class,
            expected.numel(),
            "non-finite values are never admissible",
        )

    comparison_expected = expected
    allowed = torch.zeros_like(expected, dtype=torch.float64)
    witness_digest: str | None = None
    context_digest = context.digest if context is not None else None

    if policy.difference_class is DifferenceClass.EXACT:
        if witness is not None:
            return comparison_failure(
                policy.difference_class, expected.numel(), "EXACT comparison rejects witnesses"
            )
    elif context is None:
        return comparison_failure(
            policy.difference_class, expected.numel(), "non-exact comparison requires context"
        )
    elif policy.difference_class in (
        DifferenceClass.REDUCTION_ORDER,
        DifferenceClass.PARTITIONED_REDUCTION,
    ):
        if not isinstance(witness, VerifiedReductionWitness):
            return comparison_failure(
                policy.difference_class,
                expected.numel(),
                "missing oracle-verified reduction witness",
            )
        reason = _validate_common_witness(witness, expected, actual, policy, context)
        if policy.difference_class is DifferenceClass.REDUCTION_ORDER:
            reduction = policy.reduction
            assert reduction is not None
            operands_digest = reduction.operands_digest
            indices_digest = reduction.indices_digest
            partitions_digest = None
        else:
            partitioned = policy.partitioned_reduction
            assert partitioned is not None
            operands_digest = partitioned.operands_digest
            indices_digest = partitioned.indices_digest
            partitions_digest = partitioned.partitions_digest
        if reason is None and (
            witness.operands_digest != operands_digest
            or witness.indices_digest != indices_digest
        ):
            reason = "reduction witness digest does not match policy"
        if (
            reason is None
            and policy.difference_class is DifferenceClass.PARTITIONED_REDUCTION
            and witness.partitions_digest != partitions_digest
        ):
            reason = "partitioned reduction witness digest does not match policy"
        if reason is None and witness.difference_class is not policy.difference_class:
            reason = "reduction witness has a different reduction class"
        if reason is not None:
            return comparison_failure(policy.difference_class, expected.numel(), reason)
        bound_lengths = (len(witness._reference_bound), len(witness._candidate_bound))
        if witness._shape != tuple(expected.shape) or bound_lengths != (
            expected.numel(),
            expected.numel(),
        ):
            return comparison_failure(
                policy.difference_class, expected.numel(), "reduction witness shape mismatch"
            )
        if (
            stable_digest(witness._reference_bound) != witness.reference_bound_digest
            or stable_digest(witness._candidate_bound) != witness.candidate_bound_digest
        ):
            return comparison_failure(
                policy.difference_class, expected.numel(), "reduction witness payload is corrupt"
            )
        combined_bound = tuple(
            reference + candidate
            for reference, candidate in zip(
                witness._reference_bound, witness._candidate_bound, strict=True
            )
        )
        allowed = torch.tensor(combined_bound, dtype=torch.float64, device=expected.device).reshape(
            expected.shape
        )
        if not _all_finite(allowed) or bool((allowed < 0).any().item()):
            return comparison_failure(
                policy.difference_class, expected.numel(), "non-finite reduction envelope"
            )
        witness_digest = witness.witness_digest
    else:
        if not isinstance(witness, VerifiedScaleWitness):
            return comparison_failure(
                policy.difference_class,
                expected.numel(),
                "missing oracle-verified scale witness",
            )
        reason = _validate_common_witness(witness, expected, actual, policy, context)
        scale = policy.scale
        assert scale is not None
        if reason is None and (
            witness.scale_digest != scale.scale_digest or witness.abi_digest != scale.abi_digest
        ):
            reason = "scale witness digest does not match policy"
        if reason is not None:
            return comparison_failure(policy.difference_class, expected.numel(), reason)
        comparison_expected = _tensor_from_bytes(
            witness._simulated_bytes,
            shape=witness._shape,
            dtype=witness._dtype,
            device=expected.device,
        )
        if tensor_digest(comparison_expected) != witness.simulated_digest:
            return comparison_failure(
                policy.difference_class, expected.numel(), "scale witness payload is corrupt"
            )
        witness_digest = witness.witness_digest

    absolute_error = (
        actual.to(dtype=torch.float64) - comparison_expected.to(dtype=torch.float64)
    ).abs()
    if policy.difference_class in (DifferenceClass.EXACT, DifferenceClass.SCALE_DTYPE):
        mismatch = _bitwise_mismatch(comparison_expected, actual)
    else:
        mismatch = absolute_error > allowed
    passed = not bool(mismatch.any().item())
    first_mismatch = _first_index(mismatch) if not passed else None
    if not expected.is_floating_point() and bool(mismatch.any().item()):
        max_absolute_error = None
    else:
        max_absolute_error = float(absolute_error.max().item()) if expected.numel() else 0.0
    max_allowed_error = float(allowed.max().item()) if expected.numel() else 0.0
    if (
        max_absolute_error is not None and not math.isfinite(max_absolute_error)
    ) or not math.isfinite(max_allowed_error):
        return comparison_failure(
            policy.difference_class, expected.numel(), "comparison metrics are non-finite"
        )
    max_ulp_error = _max_ulp_distance(comparison_expected, actual)
    reason = "admitted" if passed else "candidate exceeds the proved error envelope"
    return ComparisonResult(
        passed=passed,
        difference_class=policy.difference_class,
        element_count=expected.numel(),
        first_mismatch_index=first_mismatch,
        max_absolute_error=max_absolute_error,
        max_ulp_error=max_ulp_error,
        max_allowed_error=max_allowed_error,
        reason=reason,
        witness_digest=witness_digest,
        context_digest=context_digest,
    )


def comparison_failure(
    difference_class: DifferenceClass,
    element_count: int,
    reason: str,
) -> ComparisonResult:
    """Create an auditable fail-closed result for harness/runtime failures."""

    return ComparisonResult(
        passed=False,
        difference_class=difference_class,
        element_count=element_count,
        first_mismatch_index=None,
        max_absolute_error=None,
        max_ulp_error=None,
        max_allowed_error=None,
        reason=reason,
    )


def stable_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return "b2s256:" + hashlib.blake2s(encoded).hexdigest()


def tensor_digest(tensor: Tensor) -> str:
    """Digest tensor structure and exact bit pattern in canonical contiguous order."""

    if tensor.layout is not torch.strided:
        raise ValueError("guardrail witnesses currently require strided tensors")
    metadata = {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "layout": str(tensor.layout),
    }
    digest = hashlib.blake2s()
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(_tensor_bytes(tensor))
    return "b2s256:" + digest.hexdigest()


def tensor_binding_digest(tensor: Tensor) -> str:
    """Bind both canonical content and the observable tensor structure."""

    return stable_digest(
        {
            "content": tensor_digest(tensor),
            "shape": tuple(tensor.shape),
            "stride": tuple(tensor.stride()),
            "dtype": str(tensor.dtype),
            "layout": str(tensor.layout),
            "device_type": tensor.device.type,
            "device_index": tensor.device.index,
        }
    )


def scale_abi_digest(abi: ScaleCastABI) -> str:
    return stable_digest(
        {
            "abi_id": abi.abi_id,
            "cast_location": abi.cast_location,
            "compute_dtype": abi.compute_dtype.value,
            "output_dtype": abi.output_dtype.value,
            "operation": abi.operation,
            "rounding_mode": abi.rounding_mode,
        }
    )


def finite_ulp_size(values: Tensor) -> Tensor:
    """Return finite adjacent spacing, including at +/- max finite values."""

    if values.dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
        raise ValueError("ULP size requires a floating tensor")
    if not _all_finite(values):
        raise ValueError("ULP size is undefined for non-finite values")
    positive_infinity = torch.full_like(values, float("inf"))
    negative_infinity = torch.full_like(values, float("-inf"))
    upper = torch.nextafter(values, positive_infinity)
    lower = torch.nextafter(values, negative_infinity)
    upper_distance = (upper.to(torch.float64) - values.to(torch.float64)).abs()
    lower_distance = (values.to(torch.float64) - lower.to(torch.float64)).abs()
    upper_distance = torch.where(
        torch.isfinite(upper_distance),
        upper_distance,
        torch.full_like(upper_distance, float("inf")),
    )
    lower_distance = torch.where(
        torch.isfinite(lower_distance),
        lower_distance,
        torch.full_like(lower_distance, float("inf")),
    )
    spacing = torch.minimum(upper_distance, lower_distance)
    if not _all_finite(spacing):
        raise ValueError("could not derive a finite ULP spacing")
    return spacing


def _issue_reduction_witness(
    *,
    expected: Tensor,
    actual: Tensor,
    policy: ComparisonPolicy,
    context: ComparisonContext,
    operands_digest: str,
    indices_digest: str,
    reference_bound: Tensor,
    candidate_bound: Tensor,
    partitions_digest: str | None = None,
    difference_class: DifferenceClass = DifferenceClass.REDUCTION_ORDER,
) -> VerifiedReductionWitness:
    return VerifiedReductionWitness(
        expected_binding=tensor_binding_digest(expected),
        actual_binding=tensor_binding_digest(actual),
        policy_digest=_policy_digest(policy),
        context_digest=context.digest,
        operands_digest=operands_digest,
        indices_digest=indices_digest,
        reference_bound=tuple(float(value) for value in reference_bound.reshape(-1).tolist()),
        candidate_bound=tuple(float(value) for value in candidate_bound.reshape(-1).tolist()),
        shape=tuple(expected.shape),
        partitions_digest=partitions_digest,
        difference_class=difference_class,
        _token=_WITNESS_TOKEN,
    )


def _issue_scale_witness(
    *,
    expected: Tensor,
    actual: Tensor,
    policy: ComparisonPolicy,
    context: ComparisonContext,
    operands_digest: str,
    scale_digest: str,
    abi_digest: str,
    simulated: Tensor,
) -> VerifiedScaleWitness:
    return VerifiedScaleWitness(
        expected_binding=tensor_binding_digest(expected),
        actual_binding=tensor_binding_digest(actual),
        policy_digest=_policy_digest(policy),
        context_digest=context.digest,
        operands_digest=operands_digest,
        scale_digest=scale_digest,
        abi_digest=abi_digest,
        simulated=simulated,
        _token=_WITNESS_TOKEN,
    )


def _issue_bf16_linear_sample_witness(
    *,
    operand_binding_digest: str,
    artifact_digest: str,
    launch_specialization_digest: str,
    material_digest: str,
    sampled_output: Tensor,
    bounds: Tensor,
    candidate_shape: tuple[int, int],
    reduction_length: int,
    input_rows: tuple[int, ...],
    output_columns: tuple[int, ...],
    max_observed_error: float,
) -> VerifiedBF16LinearSampleWitness:
    return VerifiedBF16LinearSampleWitness(
        operand_binding_digest=operand_binding_digest,
        artifact_digest=artifact_digest,
        launch_specialization_digest=launch_specialization_digest,
        material_digest=material_digest,
        sampled_output=sampled_output,
        bounds=bounds,
        candidate_shape=candidate_shape,
        reduction_length=reduction_length,
        input_rows=input_rows,
        output_columns=output_columns,
        max_observed_error=max_observed_error,
        _token=_WITNESS_TOKEN,
    )


def _validate_common_witness(
    witness: VerifiedWitness,
    expected: Tensor,
    actual: Tensor,
    policy: ComparisonPolicy,
    context: ComparisonContext,
) -> str | None:
    if witness.policy_digest != _policy_digest(policy):
        return "witness is bound to a different policy"
    if witness.context_digest != context.digest:
        return "witness is bound to a different op signature or candidate launch"
    if witness.expected_binding != tensor_binding_digest(expected):
        return "witness is bound to a different oracle tensor"
    if witness.actual_binding != tensor_binding_digest(actual):
        return "witness is bound to a different candidate tensor"
    return None


def _policy_digest(policy: ComparisonPolicy) -> str:
    if policy.difference_class is DifferenceClass.EXACT:
        payload: dict[str, Any] = {"difference_class": DifferenceClass.EXACT.value}
    elif policy.difference_class is DifferenceClass.REDUCTION_ORDER:
        reduction_proof = policy.reduction
        assert reduction_proof is not None
        payload = {
            "difference_class": DifferenceClass.REDUCTION_ORDER.value,
            "operands_digest": reduction_proof.operands_digest,
            "indices_digest": reduction_proof.indices_digest,
            "reduction_length": reduction_proof.reduction_length,
            "reference_accumulator_dtype": reduction_proof.accumulator_dtype.value,
            "candidate_accumulator_dtype": reduction_proof.effective_candidate_dtype.value,
        }
    elif policy.difference_class is DifferenceClass.PARTITIONED_REDUCTION:
        partitioned_proof = policy.partitioned_reduction
        assert partitioned_proof is not None
        payload = {
            "difference_class": DifferenceClass.PARTITIONED_REDUCTION.value,
            "operands_digest": partitioned_proof.operands_digest,
            "indices_digest": partitioned_proof.indices_digest,
            "partitions_digest": partitioned_proof.partitions_digest,
            "reduction_length": partitioned_proof.reduction_length,
            "partition_count": partitioned_proof.partition_count,
            "abi": "bf16-products:fp32-shard:bf16-partial:bf16-tp2:v1",
        }
    else:
        scale_proof = policy.scale
        assert scale_proof is not None
        payload = {
            "difference_class": DifferenceClass.SCALE_DTYPE.value,
            "scale_digest": scale_proof.scale_digest,
            "abi_digest": scale_proof.abi_digest,
            "reference_dtype": scale_proof.reference_dtype.value,
            "candidate_dtype": scale_proof.candidate_dtype.value,
            "cast_location": scale_proof.cast_location,
        }
    return stable_digest(payload)


def _structural_error(expected: Tensor, actual: Tensor) -> str | None:
    if expected.shape != actual.shape:
        return f"shape mismatch: expected {tuple(expected.shape)}, got {tuple(actual.shape)}"
    if expected.dtype != actual.dtype:
        return f"dtype mismatch: expected {expected.dtype}, got {actual.dtype}"
    if expected.layout != actual.layout:
        return f"layout mismatch: expected {expected.layout}, got {actual.layout}"
    return None


def _require_digest(value: str, name: str) -> None:
    if not value.startswith("b2s256:") or len(value) != 71:
        raise ValueError(f"{name} must be a content_digest digest")
    try:
        int(value[7:], 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a content_digest digest") from error


def _tensor_bytes(tensor: Tensor) -> bytes:
    contiguous = tensor.detach().to(device="cpu").contiguous().reshape(-1)
    return contiguous.view(torch.uint8).numpy().tobytes()


def _tensor_from_bytes(
    payload: bytes,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    raw = torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone()
    return raw.view(dtype).reshape(shape).to(device=device)


def _all_finite(tensor: Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def _first_index(mask: Tensor) -> tuple[int, ...] | None:
    nonzero = torch.nonzero(mask, as_tuple=False)
    if nonzero.numel() == 0:
        return None
    return tuple(int(value) for value in nonzero[0].tolist())


def _bitwise_mismatch(expected: Tensor, actual: Tensor) -> Tensor:
    element_size = expected.element_size()
    if element_size in (1, 2, 4, 8):
        raw_dtype = {
            1: torch.uint8,
            2: torch.int16,
            4: torch.int32,
            8: torch.int64,
        }[element_size]
        return expected.contiguous().view(raw_dtype) != actual.contiguous().view(raw_dtype)
    return expected != actual


def _max_ulp_distance(expected: Tensor, actual: Tensor) -> int:
    if expected.numel() == 0:
        return 0
    if expected.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return 0 if torch.equal(expected, actual) else 2**63 - 1

    if expected.dtype in (torch.bfloat16, torch.float16):
        raw_dtype = torch.int16
        bit_count = 16
    else:
        raw_dtype = torch.int32
        bit_count = 32
    mask = (1 << bit_count) - 1
    sign = 1 << (bit_count - 1)

    def ordered_bits(tensor: Tensor) -> Tensor:
        raw = tensor.contiguous().view(raw_dtype).to(dtype=torch.int64) & mask
        return torch.where((raw & sign) != 0, (~raw) & mask, raw | sign)

    distance = (ordered_bits(expected) - ordered_bits(actual)).abs()
    return int(distance.max().item())
