"""Core-issued evidence derived from actual loaded weight tensors."""

from __future__ import annotations

from typing_extensions import TypeAliasType

import hashlib
import json
import math
import os
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cubic_serve.contracts.model import (
    ParameterDTypeOverride,
    ScalarDType,
    TensorRole,
    WeightEncoding,
    WeightManifest,
)
from cubic_serve.models.checkpoint import CheckpointInventory, CheckpointTensor

_SEMANTIC_SCHEMA = "cubic.weights.semantic"
_EVIDENCE_SCHEMA = "cubic.weights.load-evidence"
_CHECKPOINT_SEMANTIC_SCHEMA = "cubic.weights.checkpoint-attested"
_CHECKPOINT_EVIDENCE_SCHEMA = "cubic.weights.checkpoint-load-evidence"
_SCHEMA_VERSION = 1
_EVIDENCE_TOKEN = object()
_HANDLE_TOKEN = object()

# One host readback allocation is the complete explicit scratch budget for the
# executable-byte seal. Callers may lower it for constrained environments, but
# may not silently enlarge it.
EXECUTABLE_WEIGHT_READBACK_WINDOW_BYTES = 20 * 1024 * 1024
EXECUTABLE_WEIGHT_HEADER_MAX_BYTES = 64 * 1024 * 1024

_DTYPE_BYTES = {
    ScalarDType.BF16: 2,
    ScalarDType.FP16: 2,
    ScalarDType.FP32: 4,
    ScalarDType.FP8_E4M3FN: 1,
    ScalarDType.FP8_E5M2: 1,
    ScalarDType.INT8: 1,
    ScalarDType.INT64: 8,
}

_TORCH_DTYPES: dict[torch.dtype, ScalarDType] = {
    torch.bfloat16: ScalarDType.BF16,
    torch.float16: ScalarDType.FP16,
    torch.float32: ScalarDType.FP32,
    torch.int8: ScalarDType.INT8,
    torch.int64: ScalarDType.INT64,
}
_CHECKPOINT_DTYPES = {
    "BF16": ScalarDType.BF16,
    "F16": ScalarDType.FP16,
    "F32": ScalarDType.FP32,
    "I8": ScalarDType.INT8,
    "I64": ScalarDType.INT64,
}
for _torch_name, _scalar_dtype in (
    ("float8_e4m3fn", ScalarDType.FP8_E4M3FN),
    ("float8_e5m2", ScalarDType.FP8_E5M2),
):
    _torch_dtype = getattr(torch, _torch_name, None)
    if isinstance(_torch_dtype, torch.dtype):
        _TORCH_DTYPES[_torch_dtype] = _scalar_dtype


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return f"b2s256:{hashlib.blake2s(_canonical_json(value).encode()).hexdigest()}"


def _require_digest(value: str, field: str) -> None:
    algorithm, separator, digest = value.partition(":")
    if algorithm != "b2s256" or separator != ":" or len(digest) != 64:
        raise ValueError(f"{field} must use b2s256:<64 hex chars>")
    try:
        int(digest, 16)
    except ValueError as error:
        raise ValueError(f"{field} contains non-hexadecimal characters") from error


def _validate_names(names: tuple[str, ...], field: str, *, allow_empty: bool) -> None:
    if type(names) is not tuple:
        raise TypeError(f"{field} must be an immutable tuple")
    if not allow_empty and not names:
        raise ValueError(f"{field} cannot be empty")
    if any(type(name) is not str or not name or name != name.strip() for name in names):
        raise ValueError(f"{field} must contain canonical non-empty strings")
    if names != tuple(sorted(set(names))):
        raise ValueError(f"{field} must be unique and sorted")


@dataclass(frozen=True, slots=True)
class FullBF16WeightPolicy:
    """Core-owned exact inventory and the only BF16 baseline exceptions.

    For the current world-size-one reference engine this is the complete global
    model inventory. A future distributed loader must issue rank-local policies,
    then have its coordinator prove the exact union and issue the global model
    digest; a rank-local evidence digest must not be compared directly with a
    global ``ModelSpec`` manifest.
    """

    expected_names: tuple[str, ...]
    fp32_stability_names: tuple[str, ...] = ()
    int64_metadata_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_names(self.expected_names, "expected_names", allow_empty=False)
        _validate_names(self.fp32_stability_names, "fp32_stability_names", allow_empty=True)
        _validate_names(self.int64_metadata_names, "int64_metadata_names", allow_empty=True)
        expected = set(self.expected_names)
        stability = set(self.fp32_stability_names)
        metadata = set(self.int64_metadata_names)
        if not stability <= expected or not metadata <= expected:
            raise ValueError("BF16 exception names must belong to the complete inventory")
        if stability & metadata:
            raise ValueError("a tensor cannot be both stability state and metadata")
        if expected == stability | metadata:
            raise ValueError("a full BF16 model requires at least one BF16 parameter tensor")

    @property
    def digest(self) -> str:
        return _digest(
            {
                "schema": "cubic.weights.full-bf16-policy",
                "version": _SCHEMA_VERSION,
                "expected_names": self.expected_names,
                "fp32_stability_names": self.fp32_stability_names,
                "int64_metadata_names": self.int64_metadata_names,
            }
        )


@dataclass(frozen=True, slots=True)
class CheckpointWeightSource:
    """Core-attested checkpoint identity retained through model construction."""

    inventory: CheckpointInventory
    semantic_plan_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.inventory, CheckpointInventory):
            raise TypeError("checkpoint weight source requires a CheckpointInventory")
        _require_digest(self.semantic_plan_digest, "semantic plan digest")


@dataclass(frozen=True, slots=True, init=False)
class WeightTensorEvidence:
    """Immutable identity of one tensor snapshot; only this module can issue it.

    ``semantic_digest`` identifies logical checkpoint bytes independent of
    placement; ``evidence_digest`` additionally binds the observed device.
    """

    name: str
    dtype: ScalarDType
    shape: tuple[int, ...]
    device: str
    numel: int
    byte_count: int
    bytes_digest: str
    semantic_digest: str
    evidence_digest: str

    def __init__(
        self,
        *,
        name: str,
        dtype: ScalarDType,
        shape: tuple[int, ...],
        device: str,
        numel: int,
        byte_count: int,
        bytes_digest: str,
        semantic_digest: str,
        evidence_digest: str,
        _token: object | None = None,
    ) -> None:
        if _token is not _EVIDENCE_TOKEN:
            raise TypeError("weight tensor evidence may only be issued from an actual tensor")
        if type(name) is not str or not name or name != name.strip():
            raise ValueError("weight tensor evidence requires a canonical name")
        if not isinstance(dtype, ScalarDType) or dtype not in _DTYPE_BYTES:
            raise TypeError("weight tensor evidence has an unsupported dtype")
        if type(shape) is not tuple or any(
            type(dimension) is not int or dimension < 0 for dimension in shape
        ):
            raise TypeError("weight tensor evidence shape must contain non-negative integers")
        if type(device) is not str or not device:
            raise ValueError("weight tensor evidence requires a device")
        if type(numel) is not int or numel <= 0 or math.prod(shape) != numel:
            raise ValueError("weight tensor evidence numel differs from its shape")
        if type(byte_count) is not int or byte_count != numel * _DTYPE_BYTES[dtype]:
            raise ValueError("weight tensor evidence byte count differs from dtype and shape")
        _require_digest(bytes_digest, "bytes_digest")
        _require_digest(semantic_digest, "tensor semantic_digest")
        _require_digest(evidence_digest, "tensor evidence_digest")
        semantic_payload = {
            "name": name,
            "dtype": dtype.value,
            "shape": shape,
            "numel": numel,
            "byte_count": byte_count,
            "bytes_digest": bytes_digest,
        }
        if semantic_digest != _digest(semantic_payload):
            raise ValueError("tensor semantic digest differs from its evidence fields")
        if evidence_digest != _digest({**semantic_payload, "device": device}):
            raise ValueError("tensor evidence digest differs from its load fields")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "numel", numel)
        object.__setattr__(self, "byte_count", byte_count)
        object.__setattr__(self, "bytes_digest", bytes_digest)
        object.__setattr__(self, "semantic_digest", semantic_digest)
        object.__setattr__(self, "evidence_digest", evidence_digest)


@dataclass(frozen=True, slots=True, init=False)
class WeightLoadEvidence:
    """Sealed full-BF16 manifest for the exact inventory supplied to this issuer."""

    manifest: WeightManifest
    tensors: tuple[WeightTensorEvidence, ...]
    policy_digest: str
    semantic_digest: str
    evidence_digest: str

    def __init__(
        self,
        *,
        manifest: WeightManifest,
        tensors: tuple[WeightTensorEvidence, ...],
        policy_digest: str,
        semantic_digest: str,
        evidence_digest: str,
        _token: object | None = None,
    ) -> None:
        if _token is not _EVIDENCE_TOKEN:
            raise TypeError("weight load evidence may only be issued from actual tensors")
        _require_digest(policy_digest, "policy_digest")
        _require_digest(semantic_digest, "weight semantic_digest")
        _require_digest(evidence_digest, "weight evidence_digest")
        if manifest.manifest_digest != semantic_digest:
            raise ValueError("WeightManifest digest must equal actual weight semantics")
        if not manifest.is_full_bf16_weights:
            raise ValueError("weight evidence must satisfy the full BF16 policy")
        if manifest.tensor_count != len(tensors):
            raise ValueError("WeightManifest tensor count differs from issued evidence")
        if manifest.total_numel != sum(tensor.numel for tensor in tensors):
            raise ValueError("WeightManifest element count differs from issued evidence")
        tensor_names = tuple(tensor.name for tensor in tensors)
        if tensor_names != tuple(sorted(set(tensor_names))):
            raise ValueError("weight tensor evidence names must be unique and sorted")
        expected_semantic_digest = _digest(
            {
                "schema": _SEMANTIC_SCHEMA,
                "version": _SCHEMA_VERSION,
                "tensors": tuple(_tensor_semantic_payload(tensor) for tensor in tensors),
            }
        )
        if semantic_digest != expected_semantic_digest:
            raise ValueError("weight semantic digest differs from its tensor evidence")
        expected_evidence_digest = _digest(
            {
                "schema": _EVIDENCE_SCHEMA,
                "version": _SCHEMA_VERSION,
                "policy_digest": policy_digest,
                "semantic_digest": semantic_digest,
                "tensors": tuple(_tensor_evidence_payload(tensor) for tensor in tensors),
            }
        )
        if evidence_digest != expected_evidence_digest:
            raise ValueError("weight evidence digest differs from its load evidence")
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "tensors", tensors)
        object.__setattr__(self, "policy_digest", policy_digest)
        object.__setattr__(self, "semantic_digest", semantic_digest)
        object.__setattr__(self, "evidence_digest", evidence_digest)


@dataclass(frozen=True, slots=True, init=False)
class CheckpointWeightLoadEvidence:
    """Checkpoint/header identity and placement, not equality of live bytes."""

    manifest: WeightManifest
    tensors: tuple[CheckpointTensor, ...]
    inventory_digest: str
    semantic_plan_digest: str
    policy_digest: str
    placements: tuple[tuple[str, str], ...]
    semantic_digest: str
    evidence_digest: str

    def __init__(
        self,
        *,
        manifest: WeightManifest,
        tensors: tuple[CheckpointTensor, ...],
        inventory_digest: str,
        semantic_plan_digest: str,
        policy_digest: str,
        placements: tuple[tuple[str, str], ...],
        semantic_digest: str,
        evidence_digest: str,
        _token: object | None = None,
    ) -> None:
        if _token is not _EVIDENCE_TOKEN:
            raise TypeError("checkpoint weight evidence may only be issued from attested tensors")
        for value, label in (
            (inventory_digest, "inventory digest"),
            (semantic_plan_digest, "semantic plan digest"),
            (policy_digest, "policy digest"),
            (semantic_digest, "weight semantic digest"),
            (evidence_digest, "weight evidence digest"),
        ):
            _require_digest(value, label)
        names = tuple(tensor.name for tensor in tensors)
        if names != tuple(sorted(set(names))) or not names:
            raise ValueError("attested weight records must be non-empty, unique, and sorted")
        if tuple(name for name, _ in placements) != names or any(
            not device for _, device in placements
        ):
            raise ValueError("weight placements must cover the attested records in order")
        expected_semantic = _checkpoint_semantic_digest(
            inventory_digest,
            semantic_plan_digest,
            tensors,
        )
        if semantic_digest != expected_semantic or manifest.manifest_digest != semantic_digest:
            raise ValueError("checkpoint weight semantic identity is inconsistent")
        if manifest.tensor_count != len(tensors) or manifest.total_numel != sum(
            tensor.numel for tensor in tensors
        ):
            raise ValueError("checkpoint weight manifest capacity is inconsistent")
        expected_evidence = _digest(
            {
                "schema": _CHECKPOINT_EVIDENCE_SCHEMA,
                "version": _SCHEMA_VERSION,
                "policy_digest": policy_digest,
                "semantic_digest": semantic_digest,
                "placements": placements,
            }
        )
        if evidence_digest != expected_evidence:
            raise ValueError("checkpoint weight load evidence is inconsistent")
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "tensors", tensors)
        object.__setattr__(self, "inventory_digest", inventory_digest)
        object.__setattr__(self, "semantic_plan_digest", semantic_plan_digest)
        object.__setattr__(self, "policy_digest", policy_digest)
        object.__setattr__(self, "placements", placements)
        object.__setattr__(self, "semantic_digest", semantic_digest)
        object.__setattr__(self, "evidence_digest", evidence_digest)


WeightEvidence = TypeAliasType("WeightEvidence", WeightLoadEvidence | CheckpointWeightLoadEvidence)


class WeightIntegrityError(RuntimeError):
    """The executable weight set differs from its core-sealed startup snapshot."""


class ExecutableWeightSealRejected(WeightIntegrityError):
    """Terminal admission failure while binding executable bytes to a checkpoint."""


@dataclass(frozen=True, slots=True)
class _WeightTensorSeal:
    name: str
    object_id: int
    data_ptr: int
    storage_data_ptr: int
    storage_nbytes: int
    dtype: torch.dtype
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    device: torch.device
    layout: torch.layout
    requires_grad: bool
    version: int


class SealedWeightSet:
    """Core-owned startup evidence plus a cheap live-tensor immutability guard.

    ``WeightLoadEvidence`` proves actual live bytes; checkpoint evidence proves
    only hashed checkpoint identity, headers, and placement. Runtime checks are
    O(number of tensors): they require the same inventory, objects, storage,
    metadata, and PyTorch version counters. This is an integrity boundary for
    trusted adapter and kernel code, not protection against ``.data`` or raw
    pointer writes; structural operator admission must keep weight operands
    read-only after startup.
    """

    __slots__ = ("_evidence", "_provider", "_seals")

    def __init__(
        self,
        *,
        evidence: WeightEvidence,
        provider: Callable[[], Iterable[tuple[str, object]]],
        seals: tuple[_WeightTensorSeal, ...],
        _token: object | None = None,
    ) -> None:
        if _token is not _HANDLE_TOKEN:
            raise TypeError("sealed weight sets may only be issued from actual tensors")
        self._evidence = evidence
        self._provider = provider
        self._seals = seals

    @property
    def evidence(self) -> WeightEvidence:
        return self._evidence

    def verify_unchanged(self, *, boundary: str) -> None:
        """Reject replacement, relocation, metadata, or versioned write drift."""

        if type(boundary) is not str or not boundary:
            raise ValueError("weight integrity boundary must be a non-empty string")
        try:
            current = _materialize_named_tensors(self._provider())
        except (TypeError, ValueError) as error:
            raise WeightIntegrityError(f"weight inventory drift at {boundary}: {error}") from error
        if len(current) != len(self._seals):
            raise WeightIntegrityError(
                f"weight inventory drift at {boundary}: tensor count changed"
            )
        for (name, tensor), seal in zip(current, self._seals, strict=True):
            try:
                candidate = _seal_tensor(name, tensor)
            except (TypeError, ValueError, RuntimeError) as error:
                raise WeightIntegrityError(
                    f"weight tensor {name!r} cannot be verified at {boundary}: {error}"
                ) from error
            if candidate != seal:
                raise WeightIntegrityError(
                    f"weight tensor {seal.name!r} drifted after startup at {boundary}"
                )

    def tensor_binding_digest(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        boundary: str,
    ) -> str:
        """Bind one live tensor to core-issued bytes without hashing it again."""

        if type(name) is not str or not name or name != name.strip():
            raise ValueError("weight binding requires a canonical tensor name")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("weight binding requires a tensor")
        self.verify_unchanged(boundary=boundary)
        seal = next((item for item in self._seals if item.name == name), None)
        if seal is None:
            raise WeightIntegrityError(f"weight tensor {name!r} is absent at {boundary}")
        if id(tensor) != seal.object_id or tensor.data_ptr() != seal.data_ptr:
            raise WeightIntegrityError(
                f"weight tensor {name!r} is not the sealed live tensor at {boundary}"
            )
        evidence = self._evidence
        if isinstance(evidence, WeightLoadEvidence):
            record = next(item for item in evidence.tensors if item.name == name)
            record_identity = record.semantic_digest
        else:
            checkpoint_record = next(item for item in evidence.tensors if item.name == name)
            record_identity = _digest(
                {
                    "inventory_digest": evidence.inventory_digest,
                    "tensor": _checkpoint_tensor_payload(checkpoint_record),
                }
            )
        return _digest(
            {
                "schema": "cubic.weights.live-tensor-binding",
                "version": _SCHEMA_VERSION,
                "weight_set_semantics": self._evidence.semantic_digest,
                "weight_set_evidence": self._evidence.evidence_digest,
                "record": record_identity,
                "name": name,
                "device": str(tensor.device),
                "shape": tuple(tensor.shape),
                "stride": tuple(tensor.stride()),
            }
        )


def _tensor_semantic_payload(tensor: WeightTensorEvidence) -> dict[str, Any]:
    return {
        "name": tensor.name,
        "dtype": tensor.dtype.value,
        "shape": tensor.shape,
        "numel": tensor.numel,
        "byte_count": tensor.byte_count,
        "bytes_digest": tensor.bytes_digest,
    }


def _tensor_evidence_payload(tensor: WeightTensorEvidence) -> dict[str, Any]:
    return {
        **_tensor_semantic_payload(tensor),
        "device": tensor.device,
        "semantic_digest": tensor.semantic_digest,
        "evidence_digest": tensor.evidence_digest,
    }


def _materialize_named_tensors(
    named_tensors: Iterable[tuple[str, object]],
) -> tuple[tuple[str, torch.Tensor], ...]:
    try:
        items = tuple(named_tensors)
    except TypeError as error:
        raise TypeError("named weight provider must return an iterable") from error
    materialized: list[tuple[str, torch.Tensor]] = []
    observed: set[str] = set()
    for index, item in enumerate(items):
        if type(item) is not tuple or len(item) != 2:
            raise TypeError(f"named tensor entry {index} must be a (name, tensor) tuple")
        name, value = item
        if type(name) is not str or not name or name != name.strip():
            raise ValueError("weight names must be canonical non-empty strings")
        if name in observed:
            raise ValueError(f"duplicate weight tensor name: {name!r}")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"weight {name!r} is not a tensor")
        observed.add(name)
        materialized.append((name, value))
    if not materialized:
        raise ValueError("weight tensor set cannot be empty")
    return tuple(materialized)


def _tensor_version(tensor: torch.Tensor) -> int:
    try:
        # PyTorch's internal mutation counter is deliberately part of this guard ABI.
        version = tensor._version
    except RuntimeError as error:
        raise ValueError("weight tensor must expose a PyTorch version counter") from error
    if type(version) is not int or version < 0:
        raise RuntimeError("weight tensor exposed an invalid version counter")
    return version


def _seal_tensor(name: str, tensor: torch.Tensor) -> _WeightTensorSeal:
    if tensor.layout is not torch.strided or tensor.is_quantized:
        raise TypeError(f"weight {name!r} must be a non-quantized strided tensor")
    if tensor.device.type == "meta":
        raise ValueError(f"weight {name!r} has no materialized storage")
    if tensor.numel() <= 0:
        raise ValueError(f"weight {name!r} cannot be empty")
    storage = tensor.untyped_storage()
    return _WeightTensorSeal(
        name=name,
        object_id=id(tensor),
        data_ptr=tensor.data_ptr(),
        storage_data_ptr=storage.data_ptr(),
        storage_nbytes=storage.nbytes(),
        dtype=tensor.dtype,
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        storage_offset=int(tensor.storage_offset()),
        device=tensor.device,
        layout=tensor.layout,
        requires_grad=tensor.requires_grad,
        version=_tensor_version(tensor),
    )


def _issue_tensor_evidence(name: str, tensor: torch.Tensor) -> WeightTensorEvidence:
    if tensor.layout is not torch.strided or tensor.is_quantized:
        raise TypeError(f"weight {name!r} must be a non-quantized strided tensor")
    if tensor.device.type == "meta":
        raise ValueError(f"weight {name!r} has no materialized bytes on the meta device")
    if tensor.numel() <= 0:
        raise ValueError(f"weight {name!r} cannot be empty")
    try:
        _TORCH_DTYPES[tensor.dtype]
    except KeyError as error:
        raise TypeError(f"weight {name!r} has unsupported dtype {tensor.dtype}") from error
    if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
        torch.isfinite(tensor).all().item()
    ):
        raise ValueError(f"weight {name!r} must contain only finite values")

    snapshot = tensor.detach().to(device="cpu", copy=True).contiguous()
    raw_bytes = snapshot.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    numel = tensor.numel()
    byte_count = numel * tensor.element_size()
    if len(raw_bytes) != byte_count:
        raise RuntimeError(f"weight {name!r} byte snapshot has an inconsistent length")
    bytes_digest = f"b2s256:{hashlib.blake2s(raw_bytes).hexdigest()}"
    return _issue_tensor_digest_evidence(name, tensor, bytes_digest)


def _issue_tensor_digest_evidence(
    name: str,
    tensor: torch.Tensor,
    bytes_digest: str,
) -> WeightTensorEvidence:
    """Issue one record after a core path has proved the supplied byte digest."""

    try:
        dtype = _TORCH_DTYPES[tensor.dtype]
    except KeyError as error:
        raise TypeError(f"weight {name!r} has unsupported dtype {tensor.dtype}") from error
    _require_digest(bytes_digest, "bytes_digest")
    device = str(tensor.device)
    semantic_payload = {
        "name": name,
        "dtype": dtype.value,
        "shape": tuple(tensor.shape),
        "numel": tensor.numel(),
        "byte_count": tensor.numel() * tensor.element_size(),
        "bytes_digest": bytes_digest,
    }
    semantic_digest = _digest(semantic_payload)
    evidence_digest = _digest({**semantic_payload, "device": device})
    return WeightTensorEvidence(
        name=name,
        dtype=dtype,
        shape=tuple(tensor.shape),
        device=device,
        numel=tensor.numel(),
        byte_count=tensor.numel() * tensor.element_size(),
        bytes_digest=bytes_digest,
        semantic_digest=semantic_digest,
        evidence_digest=evidence_digest,
        _token=_EVIDENCE_TOKEN,
    )


def _names_digest(names: tuple[str, ...]) -> str:
    return _digest({"tensor_names": names})


def _checkpoint_tensor_payload(tensor: CheckpointTensor) -> dict[str, Any]:
    return {
        "name": tensor.name,
        "dtype": tensor.dtype,
        "shape": tensor.shape,
        "shard_path": tensor.shard_path,
        "offset_ordinal": tensor.offset_ordinal,
        "numel": tensor.numel,
        "byte_count": tensor.byte_count,
    }


def _checkpoint_semantic_digest(
    inventory_digest: str,
    semantic_plan_digest: str,
    tensors: tuple[CheckpointTensor, ...],
) -> str:
    return _digest(
        {
            "schema": _CHECKPOINT_SEMANTIC_SCHEMA,
            "version": _SCHEMA_VERSION,
            "inventory_digest": inventory_digest,
            "semantic_plan_digest": semantic_plan_digest,
            "tensors": tuple(_checkpoint_tensor_payload(tensor) for tensor in tensors),
        }
    )


def _attested_records(
    inventory: CheckpointInventory,
    policy: FullBF16WeightPolicy,
) -> tuple[CheckpointTensor, ...]:
    if not isinstance(inventory, CheckpointInventory):
        raise TypeError("inventory must be a CheckpointInventory")
    if not isinstance(policy, FullBF16WeightPolicy):
        raise TypeError("policy must be a FullBF16WeightPolicy")
    by_name = {tensor.name: tensor for tensor in inventory.tensors}
    missing = sorted(set(policy.expected_names) - set(by_name))
    if missing:
        raise ValueError(f"weight policy is absent from checkpoint inventory: {missing}")
    return tuple(by_name[name] for name in policy.expected_names)


def _policy_dtype(policy: FullBF16WeightPolicy, name: str) -> ScalarDType:
    if name in policy.fp32_stability_names:
        return ScalarDType.FP32
    if name in policy.int64_metadata_names:
        return ScalarDType.INT64
    return ScalarDType.BF16


def _attested_override(
    *,
    group_id: str,
    role: TensorRole,
    dtype: ScalarDType,
    names: tuple[str, ...],
    records: dict[str, CheckpointTensor],
    reason: str,
) -> ParameterDTypeOverride | None:
    if not names:
        return None
    return ParameterDTypeOverride(
        group_id=group_id,
        role=role,
        dtype=dtype,
        tensor_count=len(names),
        total_numel=sum(records[name].numel for name in names),
        tensor_names_digest=_names_digest(names),
        reason=reason,
    )


def checkpoint_weight_manifest(
    inventory: CheckpointInventory,
    *,
    policy: FullBF16WeightPolicy,
    semantic_plan_digest: str,
) -> WeightManifest:
    """Derive an executable manifest from hashed shards and validated headers."""

    _require_digest(semantic_plan_digest, "semantic plan digest")
    tensors = _attested_records(inventory, policy)
    for tensor in tensors:
        try:
            dtype = _CHECKPOINT_DTYPES[tensor.dtype]
        except KeyError as error:
            raise TypeError(
                f"attested weight {tensor.name!r} has unsupported dtype {tensor.dtype}"
            ) from error
        expected = _policy_dtype(policy, tensor.name)
        if dtype is not expected:
            raise ValueError(
                f"attested weight {tensor.name!r} violates full BF16 policy: "
                f"expected {expected.value}, found {dtype.value}"
            )
    semantic_digest = _checkpoint_semantic_digest(
        inventory.inventory_digest,
        semantic_plan_digest,
        tensors,
    )
    records = {tensor.name: tensor for tensor in tensors}
    overrides = tuple(
        override
        for override in (
            _attested_override(
                group_id="full_bf16.fp32_stability",
                role=TensorRole.STABILITY_PARAMETER,
                dtype=ScalarDType.FP32,
                names=policy.fp32_stability_names,
                records=records,
                reason="checkpoint-attested FP32 numerical-stability parameters",
            ),
            _attested_override(
                group_id="full_bf16.int64_metadata",
                role=TensorRole.METADATA,
                dtype=ScalarDType.INT64,
                names=policy.int64_metadata_names,
                records=records,
                reason="checkpoint-attested non-numerical index metadata",
            ),
        )
        if override is not None
    )
    return WeightManifest(
        manifest_digest=semantic_digest,
        encoding=WeightEncoding.DENSE,
        default_dtype=ScalarDType.BF16,
        tensor_count=len(tensors),
        total_numel=sum(tensor.numel for tensor in tensors),
        complete=True,
        pruned_layer_ids=(),
        dtype_overrides=overrides,
    )


def _override(
    *,
    group_id: str,
    role: TensorRole,
    dtype: ScalarDType,
    names: tuple[str, ...],
    tensors: dict[str, WeightTensorEvidence],
    reason: str,
) -> ParameterDTypeOverride | None:
    if not names:
        return None
    return ParameterDTypeOverride(
        group_id=group_id,
        role=role,
        dtype=dtype,
        tensor_count=len(names),
        total_numel=sum(tensors[name].numel for name in names),
        tensor_names_digest=_names_digest(names),
        reason=reason,
    )


def _assemble_weight_evidence(
    issued: dict[str, WeightTensorEvidence],
    policy: FullBF16WeightPolicy,
) -> WeightLoadEvidence:
    frozen_tensors = tuple(issued[name] for name in policy.expected_names)
    semantic_digest = _digest(
        {
            "schema": _SEMANTIC_SCHEMA,
            "version": _SCHEMA_VERSION,
            "tensors": tuple(_tensor_semantic_payload(tensor) for tensor in frozen_tensors),
        }
    )
    overrides = tuple(
        override
        for override in (
            _override(
                group_id="full_bf16.fp32_stability",
                role=TensorRole.STABILITY_PARAMETER,
                dtype=ScalarDType.FP32,
                names=policy.fp32_stability_names,
                tensors=issued,
                reason="core-verified FP32 numerical-stability parameters",
            ),
            _override(
                group_id="full_bf16.int64_metadata",
                role=TensorRole.METADATA,
                dtype=ScalarDType.INT64,
                names=policy.int64_metadata_names,
                tensors=issued,
                reason="core-verified non-numerical index metadata",
            ),
        )
        if override is not None
    )
    manifest = WeightManifest(
        manifest_digest=semantic_digest,
        encoding=WeightEncoding.DENSE,
        default_dtype=ScalarDType.BF16,
        tensor_count=len(frozen_tensors),
        total_numel=sum(tensor.numel for tensor in frozen_tensors),
        complete=True,
        pruned_layer_ids=(),
        dtype_overrides=overrides,
    )
    evidence_digest = _digest(
        {
            "schema": _EVIDENCE_SCHEMA,
            "version": _SCHEMA_VERSION,
            "policy_digest": policy.digest,
            "semantic_digest": semantic_digest,
            "tensors": tuple(_tensor_evidence_payload(tensor) for tensor in frozen_tensors),
        }
    )
    return WeightLoadEvidence(
        manifest=manifest,
        tensors=frozen_tensors,
        policy_digest=policy.digest,
        semantic_digest=semantic_digest,
        evidence_digest=evidence_digest,
        _token=_EVIDENCE_TOKEN,
    )


def collect_full_bf16_weight_evidence(
    named_tensors: Iterable[tuple[str, object]],
    *,
    policy: FullBF16WeightPolicy,
) -> WeightLoadEvidence:
    """Snapshot an independently enumerated complete weight set and issue evidence.

    The API accepts no claimed manifest or digest. ``policy.expected_names`` is
    the core loader's independent checkpoint inventory; observed tensors must
    match it exactly before any manifest is issued. Tensor references are not
    retained after their compact evidence record is created, so checkpoint
    readers may supply a streaming iterator.
    """

    if not isinstance(policy, FullBF16WeightPolicy):
        raise TypeError("policy must be a FullBF16WeightPolicy")
    expected_names = set(policy.expected_names)
    stability_names = set(policy.fp32_stability_names)
    metadata_names = set(policy.int64_metadata_names)
    observed_names: set[str] = set()
    issued: dict[str, WeightTensorEvidence] = {}
    for index, item in enumerate(named_tensors):
        if type(item) is not tuple or len(item) != 2:
            raise TypeError(f"named tensor entry {index} must be a (name, tensor) tuple")
        name, value = item
        if type(name) is not str or not name or name != name.strip():
            raise ValueError("weight names must be canonical non-empty strings")
        if name in observed_names:
            raise ValueError(f"duplicate weight tensor name: {name!r}")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"weight {name!r} is not a tensor")
        observed_names.add(name)
        if name not in expected_names:
            continue
        evidence = _issue_tensor_evidence(name, value)
        expected_dtype = (
            ScalarDType.FP32
            if name in stability_names
            else ScalarDType.INT64
            if name in metadata_names
            else ScalarDType.BF16
        )
        if evidence.dtype is not expected_dtype:
            raise ValueError(
                f"weight {name!r} violates full BF16 policy: "
                f"expected {expected_dtype.value}, found {evidence.dtype.value}"
            )
        issued[name] = evidence
    if not observed_names:
        raise ValueError("weight tensor set cannot be empty")

    if tuple(sorted(observed_names)) != policy.expected_names:
        raise ValueError(
            "loaded weight inventory differs from core policy: "
            f"missing={sorted(expected_names - observed_names)}, "
            f"unexpected={sorted(observed_names - expected_names)}"
        )

    return _assemble_weight_evidence(issued, policy)


def seal_full_bf16_weight_set(
    provider: Callable[[], Iterable[tuple[str, object]]],
    *,
    policy: FullBF16WeightPolicy,
) -> SealedWeightSet:
    """Issue full byte evidence once and retain a cheap live-storage guard."""

    if not callable(provider):
        raise TypeError("weight provider must be callable")
    if not isinstance(policy, FullBF16WeightPolicy):
        raise TypeError("policy must be a FullBF16WeightPolicy")
    try:
        materialized = _materialize_named_tensors(provider())
        observed_names = tuple(name for name, _ in materialized)
        if observed_names != policy.expected_names:
            raise ValueError("live weight provider must enumerate the exact canonical policy order")
        evidence = collect_full_bf16_weight_evidence(materialized, policy=policy)
        seals = tuple(_seal_tensor(name, tensor) for name, tensor in materialized)
    except (TypeError, ValueError, RuntimeError) as error:
        raise WeightIntegrityError(f"weight set failed startup sealing: {error}") from error
    return SealedWeightSet(
        evidence=evidence,
        provider=provider,
        seals=seals,
        _token=_HANDLE_TOKEN,
    )


def _executable_shard_path(
    inventory: CheckpointInventory,
    record: CheckpointTensor,
) -> Path:
    if record.shard_path not in {item.relative_path for item in inventory.shard_files}:
        raise ValueError(f"weight {record.name!r} names an unattested shard")
    root = inventory.model_root.resolve(strict=True)
    path = root / record.shard_path
    _reject_executable_symlink_chain(root, record.shard_path)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not path.is_file():
        raise ValueError(f"weight {record.name!r} shard escapes the checkpoint root")
    return path


def _reject_executable_symlink_chain(root: Path, relative_path: str) -> None:
    candidate = root
    for component in Path(relative_path).parts:
        candidate /= component
        if candidate.is_symlink():
            raise ValueError("executable checkpoint shard path cannot contain symlinks")


@dataclass(frozen=True, slots=True)
class _ExecutableFileStamp:
    device: int
    inode: int
    byte_count: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True, slots=True)
class _ExecutableShardHandle:
    model_root: Path
    relative_path: str
    path: Path
    file_descriptor: int
    stamp: _ExecutableFileStamp
    expected_content_digest: str
    tensor_offsets: dict[str, tuple[int, int]]


def _executable_file_stamp(file_descriptor: int) -> _ExecutableFileStamp:
    status = os.fstat(file_descriptor)
    return _ExecutableFileStamp(
        device=status.st_dev,
        inode=status.st_ino,
        byte_count=status.st_size,
        modified_ns=status.st_mtime_ns,
        changed_ns=status.st_ctime_ns,
    )


def _executable_path_stamp(path: Path) -> _ExecutableFileStamp:
    status = path.stat()
    return _ExecutableFileStamp(
        device=status.st_dev,
        inode=status.st_ino,
        byte_count=status.st_size,
        modified_ns=status.st_mtime_ns,
        changed_ns=status.st_ctime_ns,
    )


def _hash_executable_file(
    file_descriptor: int,
    *,
    byte_count: int,
    readback: torch.Tensor,
) -> str:
    digest = hashlib.blake2s()
    offset = 0
    buffer = memoryview(readback.numpy())
    while offset < byte_count:
        chunk_bytes = min(readback.numel(), byte_count - offset)
        observed = os.preadv(file_descriptor, (buffer[:chunk_bytes],), offset)
        if observed <= 0:
            raise ValueError("executable checkpoint shard ended while it was hashed")
        digest.update(buffer[:observed])
        offset += observed
    if offset != byte_count:
        raise ValueError("executable checkpoint shard hash traversal is incomplete")
    return f"b2s256:{digest.hexdigest()}"


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"safetensors header contains duplicate key {key!r}")
        result[key] = value
    return result


def _pread_exact(
    file_descriptor: int,
    buffer: memoryview,
    *,
    offset: int,
) -> None:
    consumed = 0
    while consumed < len(buffer):
        observed = os.preadv(file_descriptor, (buffer[consumed:],), offset + consumed)
        if observed <= 0:
            raise ValueError("executable checkpoint shard ended during bounded read")
        consumed += observed


def _allocate_executable_source_buffer(window_bytes: int) -> bytearray:
    return bytearray(window_bytes)


def _reject_nonfinite_raw_chunk(
    name: str,
    live_bytes: memoryview,
    scratch_bytes: memoryview,
    dtype: torch.dtype,
) -> None:
    word_dtype: np.dtype[Any]
    if dtype is torch.bfloat16:
        word_dtype, exponent_mask = np.dtype("<u2"), 0x7F80
    elif dtype is torch.float32:
        word_dtype, exponent_mask = np.dtype("<u4"), 0x7F800000
    else:
        raise TypeError(f"loaded floating weight {name!r} has no finite-bit policy")
    finite_bytes = 1024 * 1024
    for start in range(0, len(live_bytes), finite_bytes):
        stop = min(len(live_bytes), start + finite_bytes)
        words = np.frombuffer(live_bytes[start:stop], dtype=word_dtype)
        scratch = np.frombuffer(scratch_bytes[: stop - start], dtype=word_dtype)
        np.bitwise_and(words, exponent_mask, out=scratch)
        if bool(np.any(scratch == exponent_mask)):
            raise ValueError(f"loaded weight {name!r} must contain only finite values")


def _parse_executable_safetensors_header(
    file_descriptor: int,
    *,
    file_bytes: int,
    records: tuple[CheckpointTensor, ...],
) -> dict[str, tuple[int, int]]:
    if sys.byteorder != "little":
        raise ValueError("executable safetensors sealing requires a little-endian host")
    length_bytes = bytearray(8)
    _pread_exact(file_descriptor, memoryview(length_bytes), offset=0)
    header_bytes = int.from_bytes(length_bytes, byteorder="little", signed=False)
    if not 0 < header_bytes <= EXECUTABLE_WEIGHT_HEADER_MAX_BYTES:
        raise ValueError("safetensors header exceeds the executable header budget")
    data_start = 8 + header_bytes
    if data_start > file_bytes:
        raise ValueError("safetensors header extends beyond its shard")
    encoded_header = bytearray(header_bytes)
    _pread_exact(file_descriptor, memoryview(encoded_header), offset=8)
    try:
        header = json.loads(encoded_header, object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("executable safetensors header is not strict JSON") from error
    if type(header) is not dict:
        raise ValueError("executable safetensors header must be an object")
    metadata = header.get("__metadata__")
    if metadata is not None and (
        type(metadata) is not dict
        or any(type(key) is not str or type(value) is not str for key, value in metadata.items())
    ):
        raise ValueError("safetensors metadata must map strings to strings")

    offsets: dict[str, tuple[int, int]] = {}
    ranges: list[tuple[int, int, str]] = []
    for record in records:
        entry = header.get(record.name)
        if type(entry) is not dict:
            raise ValueError(f"safetensors header is missing tensor {record.name!r}")
        if set(entry) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"safetensors tensor {record.name!r} entry is not canonical")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        data_offsets = entry.get("data_offsets")
        if dtype != record.dtype:
            raise ValueError(f"safetensors tensor {record.name!r} dtype differs from inventory")
        if (
            type(shape) is not list
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
            or tuple(shape) != record.shape
        ):
            raise ValueError(f"safetensors tensor {record.name!r} shape differs from inventory")
        if (
            type(data_offsets) is not list
            or len(data_offsets) != 2
            or any(type(value) is not int for value in data_offsets)
        ):
            raise ValueError(f"safetensors tensor {record.name!r} has invalid data offsets")
        relative_start, relative_stop = data_offsets
        if relative_start < 0 or relative_stop < relative_start:
            raise ValueError(f"safetensors tensor {record.name!r} has invalid data range")
        if relative_stop - relative_start != record.byte_count:
            raise ValueError(
                f"safetensors tensor {record.name!r} byte count differs from inventory"
            )
        absolute_start = data_start + relative_start
        absolute_stop = data_start + relative_stop
        if absolute_stop > file_bytes:
            raise ValueError(f"safetensors tensor {record.name!r} extends beyond its shard")
        for other_start, other_stop, other_name in ranges:
            if absolute_start < other_stop and other_start < absolute_stop:
                raise ValueError(
                    f"safetensors tensors {other_name!r} and {record.name!r} overlap"
                )
        ranges.append((absolute_start, absolute_stop, record.name))
        offsets[record.name] = (absolute_start, absolute_stop)
    return offsets


def _open_executable_shards(
    inventory: CheckpointInventory,
    records: tuple[CheckpointTensor, ...],
    *,
    readback: torch.Tensor,
) -> dict[str, _ExecutableShardHandle]:
    evidence_by_path = {item.relative_path: item for item in inventory.shard_files}
    opened: dict[str, _ExecutableShardHandle] = {}
    try:
        records_by_shard: dict[str, list[CheckpointTensor]] = {}
        for record in records:
            records_by_shard.setdefault(record.shard_path, []).append(record)
        for shard_path, shard_records in records_by_shard.items():
            record = shard_records[0]
            path = _executable_shard_path(inventory, record)
            evidence = evidence_by_path[shard_path]
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            file_descriptor = os.open(path, flags)
            try:
                stamp = _executable_file_stamp(file_descriptor)
                if stamp != _executable_path_stamp(path):
                    raise ValueError("executable checkpoint shard changed while it was opened")
                if stamp.byte_count != evidence.byte_count:
                    raise ValueError("executable checkpoint shard size differs from inventory")
                observed_content_digest = _hash_executable_file(
                    file_descriptor,
                    byte_count=stamp.byte_count,
                    readback=readback,
                )
                if _executable_file_stamp(file_descriptor) != stamp:
                    raise ValueError("executable checkpoint shard changed while it was hashed")
                if observed_content_digest != evidence.content_digest:
                    raise ValueError("executable checkpoint shard digest differs from inventory")
                tensor_offsets = _parse_executable_safetensors_header(
                    file_descriptor,
                    file_bytes=stamp.byte_count,
                    records=tuple(shard_records),
                )
                opened[shard_path] = _ExecutableShardHandle(
                    model_root=inventory.model_root.resolve(strict=True),
                    relative_path=record.shard_path,
                    path=path,
                    file_descriptor=file_descriptor,
                    stamp=stamp,
                    expected_content_digest=evidence.content_digest,
                    tensor_offsets=tensor_offsets,
                )
            except BaseException as error:
                try:
                    os.close(file_descriptor)
                except OSError as inner_close_error:
                    if isinstance(error, Exception):
                        raise inner_close_error from error
                raise
    except BaseException as error:
        open_cleanup_error: OSError | None = None
        for handle in opened.values():
            try:
                os.close(handle.file_descriptor)
            except OSError as candidate:
                if open_cleanup_error is None:
                    open_cleanup_error = candidate
        if open_cleanup_error is not None and isinstance(error, Exception):
            raise open_cleanup_error from error
        raise
    return opened


def _verify_executable_shards(
    handles: dict[str, _ExecutableShardHandle],
    *,
    readback: torch.Tensor,
) -> None:
    for handle in handles.values():
        _reject_executable_symlink_chain(handle.model_root, handle.relative_path)
        if _executable_file_stamp(handle.file_descriptor) != handle.stamp:
            raise ValueError("executable checkpoint shard changed during byte sealing")
        observed_content_digest = _hash_executable_file(
            handle.file_descriptor,
            byte_count=handle.stamp.byte_count,
            readback=readback,
        )
        if _executable_file_stamp(handle.file_descriptor) != handle.stamp:
            raise ValueError("executable checkpoint shard changed during final hashing")
        if observed_content_digest != handle.expected_content_digest:
            raise ValueError("executable checkpoint shard digest drifted during byte sealing")
        if _executable_path_stamp(handle.path) != handle.stamp:
            raise ValueError("executable checkpoint shard path changed during byte sealing")
        _reject_executable_symlink_chain(handle.model_root, handle.relative_path)


def _reject_executable_storage_overlap(
    materialized: tuple[tuple[str, torch.Tensor], ...],
) -> None:
    ranges: dict[torch.device, list[tuple[int, int, str]]] = {}
    for name, tensor in materialized:
        start = tensor.data_ptr()
        stop = start + int(tensor.numel()) * tensor.element_size()
        for other_start, other_stop, other_name in ranges.setdefault(tensor.device, []):
            if start < other_stop and other_start < stop:
                raise ValueError(
                    f"loaded weights {other_name!r} and {name!r} overlap live storage"
                )
        ranges[tensor.device].append((start, stop, name))


def _validate_executable_tensor(
    name: str,
    tensor: torch.Tensor,
    record: CheckpointTensor,
    policy: FullBF16WeightPolicy,
) -> None:
    if tensor.layout is not torch.strided or tensor.is_quantized:
        raise TypeError(f"loaded weight {name!r} must be a non-quantized strided tensor")
    if tensor.device.type not in {"cpu", "cuda"}:
        raise ValueError(f"loaded weight {name!r} is on unsupported device {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"loaded weight {name!r} must use contiguous storage")
    try:
        actual_dtype = _TORCH_DTYPES[tensor.dtype]
        source_dtype = _CHECKPOINT_DTYPES[record.dtype]
    except KeyError as error:
        raise TypeError(f"loaded weight {name!r} has an unsupported dtype") from error
    expected_dtype = _policy_dtype(policy, name)
    if actual_dtype is not expected_dtype or source_dtype is not expected_dtype:
        raise ValueError(
            f"loaded weight {name!r} violates full BF16 policy: "
            f"expected {expected_dtype.value}, found live={actual_dtype.value}, "
            f"source={source_dtype.value}"
        )
    if tuple(tensor.shape) != record.shape or tensor.numel() != record.numel:
        raise ValueError(f"loaded weight {name!r} differs from its checkpoint shape")
    if tensor.numel() * tensor.element_size() != record.byte_count:
        raise ValueError(f"loaded weight {name!r} byte size differs from its checkpoint")


def _compare_executable_tensor_bytes(
    name: str,
    tensor: torch.Tensor,
    record: CheckpointTensor,
    shard: _ExecutableShardHandle,
    readback: torch.Tensor,
    source_buffer: bytearray,
    window_bytes: int,
) -> str:
    live_bytes = tensor.detach().reshape(-1).view(torch.uint8).reshape(-1)
    live_readback = memoryview(readback.numpy())
    source_readback = memoryview(source_buffer)
    source_start, source_stop = shard.tensor_offsets[name]
    chunk_capacity = (window_bytes // tensor.element_size()) * tensor.element_size()
    if chunk_capacity <= 0:
        raise ValueError("executable weight window cannot hold one tensor element")
    consumed = 0
    source_hash = hashlib.blake2s()
    while consumed < record.byte_count:
        chunk_bytes = min(chunk_capacity, record.byte_count - consumed)
        readback[:chunk_bytes].copy_(
            live_bytes[consumed : consumed + chunk_bytes], non_blocking=False
        )
        _pread_exact(
            shard.file_descriptor,
            source_readback[:chunk_bytes],
            offset=source_start + consumed,
        )
        if live_readback[:chunk_bytes] != source_readback[:chunk_bytes]:
            raise ValueError(f"loaded weight {name!r} bytes differ from safetensors source")
        source_hash.update(source_readback[:chunk_bytes])
        if tensor.is_floating_point():
            _reject_nonfinite_raw_chunk(
                name,
                live_readback[:chunk_bytes],
                source_readback[:chunk_bytes],
                tensor.dtype,
            )
        consumed += chunk_bytes
    if consumed != record.byte_count or source_start + consumed != source_stop:
        raise RuntimeError(f"weight {name!r} byte traversal is incomplete")
    return f"b2s256:{source_hash.hexdigest()}"


def seal_executable_full_bf16_weight_set(
    provider: Callable[[], Iterable[tuple[str, object]]],
    *,
    inventory: CheckpointInventory,
    policy: FullBF16WeightPolicy,
    window_bytes: int = EXECUTABLE_WEIGHT_READBACK_WINDOW_BYTES,
) -> SealedWeightSet:
    """Prove every live byte equals its safetensors source within a fixed budget.

    Host peak is bounded by the pinned live readback window plus one source byte
    window and a header of at most 64 MiB while it is parsed; only the small
    tensor-offset table survives parsing. Finite checks use small reductions. No
    device scratch or full-tensor clone is created. A stable read-only descriptor
    prevents source traversal from crossing an inode, while before and after
    hashes bind it to the inventory. Successful evidence is therefore
    ``WeightLoadEvidence`` rather than checkpoint-header evidence.
    """

    shard_handles: dict[str, _ExecutableShardHandle] = {}
    sealed: SealedWeightSet | None = None
    failure: BaseException | None = None
    try:
        if not callable(provider):
            raise TypeError("weight provider must be callable")
        if not isinstance(policy, FullBF16WeightPolicy):
            raise TypeError("policy must be a FullBF16WeightPolicy")
        if type(window_bytes) is not int or not 0 < window_bytes <= (
            EXECUTABLE_WEIGHT_READBACK_WINDOW_BYTES
        ):
            raise ValueError("executable weight window must be a positive value at most 20 MiB")
        materialized = _materialize_named_tensors(provider())
        if tuple(name for name, _ in materialized) != policy.expected_names:
            raise ValueError("live weight provider must enumerate the exact canonical policy order")
        records = _attested_records(inventory, policy)
        for (name, tensor), record in zip(materialized, records, strict=True):
            _validate_executable_tensor(name, tensor, record, policy)
        _reject_executable_storage_overlap(materialized)
        cuda_devices = tuple(
            dict.fromkeys(tensor.device for _, tensor in materialized if tensor.is_cuda)
        )
        for device in cuda_devices:
            torch.cuda.synchronize(device)
        initial_seals = tuple(_seal_tensor(name, tensor) for name, tensor in materialized)
        readback = torch.empty(
            window_bytes,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=bool(cuda_devices),
        )
        source_buffer = _allocate_executable_source_buffer(window_bytes)
        shard_handles = _open_executable_shards(
            inventory,
            records,
            readback=readback,
        )
        issued = {
            name: _issue_tensor_digest_evidence(
                name,
                tensor,
                _compare_executable_tensor_bytes(
                    name,
                    tensor,
                    record,
                    shard_handles[record.shard_path],
                    readback,
                    source_buffer,
                    window_bytes,
                ),
            )
            for (name, tensor), record in zip(materialized, records, strict=True)
        }
        for device in cuda_devices:
            torch.cuda.synchronize(device)
        seals = tuple(_seal_tensor(name, tensor) for name, tensor in materialized)
        if seals != initial_seals:
            raise ValueError("live weight storage changed while executable bytes were sealed")
        _verify_executable_shards(shard_handles, readback=readback)
        evidence = _assemble_weight_evidence(issued, policy)
        sealed = SealedWeightSet(
            evidence=evidence,
            provider=provider,
            seals=seals,
            _token=_HANDLE_TOKEN,
        )
    except BaseException as error:
        failure = error

    cleanup_failure: BaseException | None = None
    for handle in shard_handles.values():
        try:
            os.close(handle.file_descriptor)
        except BaseException as error:
            if cleanup_failure is None:
                cleanup_failure = error
    if failure is not None:
        if isinstance(failure, ExecutableWeightSealRejected):
            raise failure
        if isinstance(failure, Exception):
            raise ExecutableWeightSealRejected(
                f"executable weight set failed startup byte sealing: {failure}"
            ) from failure
        raise failure
    if cleanup_failure is not None:
        if isinstance(cleanup_failure, Exception):
            raise ExecutableWeightSealRejected(
                f"executable weight set failed startup cleanup: {cleanup_failure}"
            ) from cleanup_failure
        raise cleanup_failure
    if sealed is None:
        raise ExecutableWeightSealRejected("executable weight sealing produced no result")
    return sealed


def seal_attested_full_bf16_weight_set(
    provider: Callable[[], Iterable[tuple[str, object]]],
    *,
    inventory: CheckpointInventory,
    policy: FullBF16WeightPolicy,
    semantic_plan_digest: str,
) -> SealedWeightSet:
    """Seal loaded storage against already hashed safetensors evidence.

    ``CheckpointInventory`` attests shard files and headers. This path validates
    live name, dtype, shape, placement, and storage identity, but deliberately
    does not claim that live bytes equal checkpoint payload bytes. Use
    ``seal_executable_full_bf16_weight_set`` when arithmetic qualification needs
    that stronger fact.
    """

    if not callable(provider):
        raise TypeError("weight provider must be callable")
    try:
        materialized = _materialize_named_tensors(provider())
        observed_names = tuple(name for name, _ in materialized)
        if observed_names != policy.expected_names:
            raise ValueError("live weight provider must enumerate the exact canonical policy order")
        records = _attested_records(inventory, policy)
        for (name, tensor), record in zip(materialized, records, strict=True):
            try:
                actual_dtype = _TORCH_DTYPES[tensor.dtype]
            except KeyError as error:
                raise TypeError(
                    f"loaded weight {name!r} has unsupported dtype {tensor.dtype}"
                ) from error
            expected_dtype = _policy_dtype(policy, name)
            if actual_dtype is not expected_dtype:
                raise ValueError(
                    f"loaded weight {name!r} violates full BF16 policy: "
                    f"expected {expected_dtype.value}, found {actual_dtype.value}"
                )
            if tuple(tensor.shape) != record.shape or tensor.numel() != record.numel:
                raise ValueError(f"loaded weight {name!r} differs from its attested header")
            if tensor.numel() * tensor.element_size() != record.byte_count:
                raise ValueError(f"loaded weight {name!r} byte size differs from its checkpoint")
        manifest = checkpoint_weight_manifest(
            inventory,
            policy=policy,
            semantic_plan_digest=semantic_plan_digest,
        )
        semantic_digest = manifest.manifest_digest
        placements = tuple((name, str(tensor.device)) for name, tensor in materialized)
        evidence_digest = _digest(
            {
                "schema": _CHECKPOINT_EVIDENCE_SCHEMA,
                "version": _SCHEMA_VERSION,
                "policy_digest": policy.digest,
                "semantic_digest": semantic_digest,
                "placements": placements,
            }
        )
        evidence = CheckpointWeightLoadEvidence(
            manifest=manifest,
            tensors=records,
            inventory_digest=inventory.inventory_digest,
            semantic_plan_digest=semantic_plan_digest,
            policy_digest=policy.digest,
            placements=placements,
            semantic_digest=semantic_digest,
            evidence_digest=evidence_digest,
            _token=_EVIDENCE_TOKEN,
        )
        seals = tuple(_seal_tensor(name, tensor) for name, tensor in materialized)
    except (TypeError, ValueError, RuntimeError) as error:
        raise WeightIntegrityError(
            f"checkpoint-attested weight set failed startup sealing: {error}"
        ) from error
    return SealedWeightSet(
        evidence=evidence,
        provider=provider,
        seals=seals,
        _token=_HANDLE_TOKEN,
    )
