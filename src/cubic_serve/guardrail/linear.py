"""Verifier-owned arithmetic observation for one approved BF16 linear launch.

At current pins, vLLM 73fac1e compares BF16 matmul to torch with 10% generic
tolerance while separately testing row invariance; SGLang ba433bb tests exact
batch invariance; llama.cpp c841aeeb launches its selected compiled CUDA path.
Cubic instead binds one real retained artifact launch to a finite CPU enclosure.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from cubic_serve.guardrail.compare import stable_digest, tensor_digest
from cubic_serve.guardrail.linear_artifact import (
    BF16LinearArtifactManifest,
    LinearArtifactRejected,
    _observe_bf16_linear_artifact_launch,
)
from cubic_serve.models.weights import SealedWeightSet
from cubic_serve.ops.cuda.linear.kernel import (
    BF16LinearPrimitive,
    validate_bf16_linear_bound_operands,
)

_FP32_U, _FP64_U, _BF16_U = 2.0**-24, 2.0**-53, 2.0**-8
_FP32_TINY = float(torch.finfo(torch.float32).tiny)
_FP32_MAX = float(torch.finfo(torch.float32).max)
_BF16_TINY = float(torch.finfo(torch.bfloat16).tiny)
_DEFAULT_WORKSPACE_BYTES = 64 << 20
_WITNESS_TOKEN = object()


class LinearQualificationError(ValueError):
    """The owned launch cannot be enclosed by the approved arithmetic ABI."""


@dataclass(frozen=True, slots=True)
class BF16LinearWorkspacePlan:
    m: int
    n: int
    k: int
    budget_bytes: int
    fixed_bytes: int
    bytes_per_output_column: int
    chunk_columns: int
    peak_bytes: int


@dataclass(frozen=True, slots=True)
class VerifiedBF16LinearCandidateWitness:
    """One structural launch plus its observed numeric enclosure; not an arithmetic seal."""

    structural_witness_digest: str
    operand_material_digest: str
    bounds_digest: str
    corpus_policy_digest: str
    reduction_length: int
    output_columns: int
    chunk_columns: int
    explicit_host_tensor_peak_bytes: int
    planned_extra_gpu_bytes: int
    max_allowed_error: float
    max_observed_error: float
    _token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _WITNESS_TOKEN:
            raise TypeError("BF16 linear candidate witnesses are verifier-issued only")

    @property
    def witness_digest(self) -> str:
        payload = asdict(self)
        payload.pop("_token")
        return stable_digest(payload)


def plan_bf16_linear_workspace(
    m: int, n: int, k: int, max_workspace_bytes: int = _DEFAULT_WORKSPACE_BYTES
) -> BF16LinearWorkspacePlan:
    """Bound every explicit durable/FP64/chunk tensor before issuing a launch."""

    if any(type(value) is not int or value <= 0 for value in (m, n, k, max_workspace_bytes)):
        raise LinearQualificationError("linear workspace dimensions and budget must be positive")
    fixed = 2 * m * n + 16 * m * k
    per_column = 10 * k + 24 * m
    if fixed + per_column > max_workspace_bytes:
        raise LinearQualificationError("workspace cannot hold durable output and one oracle column")
    columns = min(n, (max_workspace_bytes - fixed) // per_column)
    peak = fixed + columns * per_column
    return BF16LinearWorkspacePlan(m, n, k, max_workspace_bytes, fixed, per_column, columns, peak)


def qualify_bf16_linear_candidate(
    *,
    primitive: BF16LinearPrimitive,
    x: Tensor,
    weight: Tensor,
    output: Tensor,
    weight_name: str,
    sealed_weights: SealedWeightSet,
    manifest: BF16LinearArtifactManifest,
    max_workspace_bytes: int = _DEFAULT_WORKSPACE_BYTES,
) -> VerifiedBF16LinearCandidateWitness:
    """Launch, collect, and enclose without accepting caller expected/actual tensors."""

    if type(sealed_weights) is not SealedWeightSet or not weight_name:
        raise LinearQualificationError("linear qualification requires named sealed weights")
    try:
        validate_bf16_linear_bound_operands(primitive, x, weight, output)
    except Exception as error:
        raise LinearQualificationError(str(error)) from error
    m, k = x.shape
    n = weight.shape[0]
    plan = plan_bf16_linear_workspace(m, n, k, max_workspace_bytes)
    weight_binding = sealed_weights.tensor_binding_digest(
        weight_name, weight, boundary="BF16 linear candidate qualification"
    )
    material = stable_digest(
        {
            "operation": "bf16(sum[k](bf16(x[m,k])*bf16(w[n,k])))",
            "x": tensor_digest(x),
            "weight": weight_binding,
            "shape": ((m, k), (n, k)),
            "indices": (0, k, 1),
        }
    )
    try:
        observation = _observe_bf16_linear_artifact_launch(
            primitive=primitive, x=x, weight=weight, output=output, manifest=manifest
        )
        structural, actual = observation.consume()
    except LinearArtifactRejected as error:
        raise LinearQualificationError(f"structural linear admission failed: {error}") from error

    x_cpu = x.detach().to(device="cpu", copy=True).contiguous()
    _check_bf16_values(x_cpu, "x")
    x64 = x_cpu.to(torch.float64)
    del x_cpu
    if not bool(torch.isfinite(x64).all().item()):
        raise LinearQualificationError("qualification x is nonfinite")
    abs_x64 = x64.abs()
    min_x, max_x = _absolute_stats(abs_x64)
    gamma32, gamma64 = _gamma(k, _FP32_U, "FP32"), _gamma(k, _FP64_U, "FP64")
    chunks: list[tuple[int, int, str]] = []
    maximum_allowed = maximum_observed = 0.0

    for start in range(0, n, plan.chunk_columns):
        end = min(start + plan.chunk_columns, n)
        weight_cpu = weight[start:end].detach().to(device="cpu", copy=True).contiguous()
        _check_bf16_values(weight_cpu, "weight")
        weight64 = weight_cpu.to(torch.float64)
        del weight_cpu
        if not bool(torch.isfinite(weight64).all().item()):
            raise LinearQualificationError("qualification weight is nonfinite")
        anchor = torch.mm(x64, weight64.t())
        weight64.abs_()
        min_weight, max_weight = _absolute_stats(weight64)
        if min_x < math.inf and min_weight < math.inf and min_x * min_weight < _FP32_TINY:
            raise LinearQualificationError("qualification corpus contains an FTZ-sensitive product")
        if max_x * max_weight > _FP32_MAX:
            raise LinearQualificationError("qualification corpus can overflow an FP32 product")
        sum_abs = torch.mm(abs_x64, weight64.t())
        del weight64
        if not _finite(anchor) or not _finite(sum_abs):
            raise LinearQualificationError("FP64 oracle enclosure is nonfinite")
        sum_abs.div_(1.0 - gamma64)
        if float(sum_abs.max().item()) > _FP32_MAX:
            raise LinearQualificationError("qualification reduction can overflow FP32")

        candidate = actual[:, start:end]
        _check_bf16_values(candidate, "output")
        difference = candidate.to(torch.float64).sub_(anchor).abs_()
        anchor.abs_()
        minimum_anchor, _maximum_anchor = _absolute_stats(anchor)
        if minimum_anchor < _BF16_TINY:
            raise LinearQualificationError("qualification corpus has an FTZ-sensitive result")
        sum_abs.mul_(gamma32 + gamma64)
        sum_abs.mul_(1.0 + _BF16_U).add_(anchor, alpha=_BF16_U)
        if bool((difference > sum_abs).any().item()):
            raise LinearQualificationError(
                "candidate exceeds FP32-reassociation plus BF16-RNE bound"
            )
        chunks.append((start, end, tensor_digest(sum_abs.contiguous())))
        maximum_allowed = max(maximum_allowed, float(sum_abs.max().item()))
        maximum_observed = max(maximum_observed, float(difference.max().item()))

    policy = stable_digest(
        {
            "schema": "cubic.bf16-linear-corpus.v1",
            "reject": ("bf16-subnormal", "fp32-subnormal-product", "overflow", "nonfinite"),
            "scope": "this qualification corpus only; release approval owns full-domain semantics",
            "verifier": "b2s256:" + hashlib.blake2s(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    return VerifiedBF16LinearCandidateWitness(
        structural.witness_digest,
        material,
        stable_digest(chunks),
        policy,
        k,
        n,
        plan.chunk_columns,
        plan.peak_bytes,
        0,
        maximum_allowed,
        maximum_observed,
        _WITNESS_TOKEN,
    )


def _check_bf16_values(tensor: Tensor, label: str) -> None:
    if tensor.device.type != "cpu" or tensor.dtype is not torch.bfloat16:
        raise LinearQualificationError(f"{label} corpus must be CPU BF16")
    bits = tensor.contiguous().view(torch.int16).numpy().view(np.uint16)
    exponent = np.bitwise_and(bits, np.uint16(0x7F80))
    magnitude = np.bitwise_and(bits, np.uint16(0x7FFF))
    if bool(np.any(exponent == np.uint16(0x7F80))):
        raise LinearQualificationError(f"qualification {label} is nonfinite")
    if bool(np.any((exponent == 0) & (magnitude != 0))):
        raise LinearQualificationError(f"qualification {label} contains BF16 subnormal values")


def _absolute_stats(values: Tensor) -> tuple[float, float]:
    array = values.numpy()
    minimum = float(np.min(array, where=array != 0, initial=np.inf))
    maximum = float(np.max(array, initial=0.0))
    return minimum, maximum


def _gamma(steps: int, unit_roundoff: float, label: str) -> float:
    product = steps * unit_roundoff
    if product >= 1.0:
        raise LinearQualificationError(f"{label} reduction envelope is undefined")
    return product / (1.0 - product)


def _finite(tensor: Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


__all__ = (
    "BF16LinearWorkspacePlan",
    "LinearQualificationError",
    "VerifiedBF16LinearCandidateWitness",
    "plan_bf16_linear_workspace",
    "qualify_bf16_linear_candidate",
)
