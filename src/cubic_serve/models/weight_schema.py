"""Immutable checkpoint-header partition plans shared by built-in model packages."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from cubic_serve._compat import StrEnum

from cubic_serve.models.checkpoint import CheckpointInventory
from cubic_serve.models.weights import FullBF16WeightPolicy

_PLAN_ISSUER_TOKEN = object()


class WeightSchemaError(ValueError):
    """Checkpoint headers do not close over a built-in model schema."""


class WeightSchemaCoverage(StrEnum):
    EXACT_CHECKPOINT_HEADERS = "exact-checkpoint-headers"


class HeaderEvidenceKind(StrEnum):
    ATTESTED_INVENTORY = "attested-inventory"
    HEADER_RECORDS_ONLY = "header-records-only"


class WeightPartition(StrEnum):
    BASE_LANGUAGE = "base-language"
    VISION = "vision"
    LM_HEAD = "lm-head"
    MTP_EXTENSION = "mtp-extension"


@dataclass(frozen=True, slots=True)
class WeightArchitectureIdentity:
    release_family: str
    release_profile: str
    model_type: str
    graph_family: str
    architecture: str
    plan_status: str

    def __post_init__(self) -> None:
        for field in (
            "release_family",
            "release_profile",
            "model_type",
            "graph_family",
            "architecture",
            "plan_status",
        ):
            value = getattr(self, field)
            if type(value) is not str or not value or value != value.strip():
                raise WeightSchemaError(f"architecture identity {field} must be canonical")


@dataclass(frozen=True, slots=True)
class TensorHeader:
    name: str
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise WeightSchemaError("tensor header name must be canonical and non-empty")
        if self.dtype not in {"BF16", "F32", "I64"}:
            raise WeightSchemaError(f"unsupported built-in tensor dtype {self.dtype!r}")
        if not self.shape or any(
            type(dimension) is not int or dimension <= 0 for dimension in self.shape
        ):
            raise WeightSchemaError("tensor header shape must contain positive dimensions")


@dataclass(frozen=True, slots=True)
class WeightNodeBinding:
    node_id: str
    tensor_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.node_id or self.node_id != self.node_id.strip():
            raise WeightSchemaError("weight node binding requires a canonical node ID")
        _require_names(self.tensor_names, field="node tensor names", allow_empty=False)


@dataclass(frozen=True, slots=True)
class WeightTensorSemantic:
    name: str
    dtype: str
    shape: tuple[int, ...]
    partition: WeightPartition
    node_id: str

    def __post_init__(self) -> None:
        TensorHeader(self.name, self.dtype, self.shape)
        if not isinstance(self.partition, WeightPartition):
            raise TypeError("weight tensor partition must be WeightPartition")
        if not self.node_id or self.node_id != self.node_id.strip():
            raise WeightSchemaError("weight tensor semantic requires a canonical node ID")


@dataclass(frozen=True, slots=True, init=False)
class CheckpointWeightPlan:
    """Closed logical partition of one complete checkpoint header inventory.

    This type deliberately carries no optimized-packing qualification.  Its
    FullBF16WeightPolicy values are downstream dtype/inventory contracts, not
    checkpoint attestation; a production loader must retain this plan and
    require ``ATTESTED_INVENTORY`` rather than extracting a policy and dropping
    provenance.
    """

    model_type: str
    architecture_identity: WeightArchitectureIdentity
    coverage: WeightSchemaCoverage
    header_evidence: HeaderEvidenceKind
    inventory_digest: str | None
    checkpoint_headers: tuple[TensorHeader, ...]
    checkpoint_names: tuple[str, ...]
    checkpoint_header_digest: str
    tensor_semantics: tuple[WeightTensorSemantic, ...]
    semantic_plan_digest: str
    base_language_names: tuple[str, ...]
    vision_names: tuple[str, ...]
    lm_head_names: tuple[str, ...]
    base_executable_names: tuple[str, ...]
    base_executable_digest: str
    mtp_extension_names: tuple[str, ...]
    mtp_extension_digest: str
    fp32_stability_names: tuple[str, ...]
    int64_metadata_names: tuple[str, ...]
    allowed_node_ids: tuple[str, ...]
    node_bindings: tuple[WeightNodeBinding, ...]
    full_checkpoint_policy: FullBF16WeightPolicy
    base_executable_policy: FullBF16WeightPolicy

    def __init__(
        self,
        *,
        model_type: str,
        architecture_identity: WeightArchitectureIdentity,
        coverage: WeightSchemaCoverage,
        header_evidence: HeaderEvidenceKind,
        inventory_digest: str | None,
        checkpoint_headers: tuple[TensorHeader, ...],
        checkpoint_names: tuple[str, ...],
        checkpoint_header_digest: str,
        tensor_semantics: tuple[WeightTensorSemantic, ...],
        semantic_plan_digest: str,
        base_language_names: tuple[str, ...],
        vision_names: tuple[str, ...],
        lm_head_names: tuple[str, ...],
        base_executable_names: tuple[str, ...],
        base_executable_digest: str,
        mtp_extension_names: tuple[str, ...],
        mtp_extension_digest: str,
        fp32_stability_names: tuple[str, ...],
        int64_metadata_names: tuple[str, ...],
        allowed_node_ids: tuple[str, ...],
        node_bindings: tuple[WeightNodeBinding, ...],
        full_checkpoint_policy: FullBF16WeightPolicy,
        base_executable_policy: FullBF16WeightPolicy,
        _token: object | None = None,
    ) -> None:
        if _token is not _PLAN_ISSUER_TOKEN:
            raise TypeError("checkpoint weight plans may only be issued from validated headers")
        values: dict[str, object] = {
            "model_type": model_type,
            "architecture_identity": architecture_identity,
            "coverage": coverage,
            "header_evidence": header_evidence,
            "inventory_digest": inventory_digest,
            "checkpoint_headers": checkpoint_headers,
            "checkpoint_names": checkpoint_names,
            "checkpoint_header_digest": checkpoint_header_digest,
            "tensor_semantics": tensor_semantics,
            "semantic_plan_digest": semantic_plan_digest,
            "base_language_names": base_language_names,
            "vision_names": vision_names,
            "lm_head_names": lm_head_names,
            "base_executable_names": base_executable_names,
            "base_executable_digest": base_executable_digest,
            "mtp_extension_names": mtp_extension_names,
            "mtp_extension_digest": mtp_extension_digest,
            "fp32_stability_names": fp32_stability_names,
            "int64_metadata_names": int64_metadata_names,
            "allowed_node_ids": allowed_node_ids,
            "node_bindings": node_bindings,
            "full_checkpoint_policy": full_checkpoint_policy,
            "base_executable_policy": base_executable_policy,
        }
        for field, value in values.items():
            object.__setattr__(self, field, value)
        self._validate()

    def _validate(self) -> None:
        if not self.model_type or self.model_type != self.model_type.strip():
            raise WeightSchemaError("weight plan requires a canonical model_type")
        if not isinstance(self.architecture_identity, WeightArchitectureIdentity):
            raise TypeError("architecture_identity must be WeightArchitectureIdentity")
        if self.architecture_identity.model_type != self.model_type:
            raise WeightSchemaError("architecture identity model_type differs from weight plan")
        if not isinstance(self.coverage, WeightSchemaCoverage):
            raise TypeError("coverage must be WeightSchemaCoverage")
        headers = materialize_headers(self.checkpoint_headers)
        if headers != self.checkpoint_headers:
            raise WeightSchemaError("checkpoint headers must be unique and sorted")
        for field, names, allow_empty in (
            ("checkpoint_names", self.checkpoint_names, False),
            ("base_language_names", self.base_language_names, False),
            ("vision_names", self.vision_names, False),
            ("lm_head_names", self.lm_head_names, False),
            ("base_executable_names", self.base_executable_names, False),
            ("mtp_extension_names", self.mtp_extension_names, True),
            ("fp32_stability_names", self.fp32_stability_names, True),
            ("int64_metadata_names", self.int64_metadata_names, True),
            ("allowed_node_ids", self.allowed_node_ids, False),
        ):
            _require_names(names, field=field, allow_empty=allow_empty)
        header_names = tuple(header.name for header in headers)
        if header_names != self.checkpoint_names:
            raise WeightSchemaError("checkpoint names differ from checkpoint headers")
        language = set(self.base_language_names)
        vision = set(self.vision_names)
        lm_head = set(self.lm_head_names)
        mtp = set(self.mtp_extension_names)
        parts = (language, vision, lm_head, mtp)
        if any(left & right for index, left in enumerate(parts) for right in parts[index + 1 :]):
            raise WeightSchemaError("weight partitions must be pairwise disjoint")
        base_parts = language | vision | lm_head
        if base_parts != set(self.base_executable_names):
            raise WeightSchemaError("base executable partition is not closed")
        if set(self.checkpoint_names) != base_parts | mtp:
            raise WeightSchemaError("checkpoint partition is not closed")
        node_ids = tuple(binding.node_id for binding in self.node_bindings)
        if node_ids != tuple(sorted(set(node_ids))):
            raise WeightSchemaError("graph node bindings must have unique sorted node IDs")
        if node_ids != self.allowed_node_ids:
            raise WeightSchemaError("weight bindings must use every architecture node exactly once")
        bound = tuple(name for binding in self.node_bindings for name in binding.tensor_names)
        if len(bound) != len(set(bound)) or set(bound) != set(self.checkpoint_names):
            raise WeightSchemaError("graph node bindings must cover each tensor exactly once")
        semantic_names = tuple(semantic.name for semantic in self.tensor_semantics)
        if semantic_names != self.checkpoint_names:
            raise WeightSchemaError("tensor semantics must be unique, sorted, and complete")
        semantic_by_name = {semantic.name: semantic for semantic in self.tensor_semantics}
        partition_by_name = {
            name: partition
            for names, partition in (
                (self.base_language_names, WeightPartition.BASE_LANGUAGE),
                (self.vision_names, WeightPartition.VISION),
                (self.lm_head_names, WeightPartition.LM_HEAD),
                (self.mtp_extension_names, WeightPartition.MTP_EXTENSION),
            )
            for name in names
        }
        node_by_name = {
            name: binding.node_id for binding in self.node_bindings for name in binding.tensor_names
        }
        for header in headers:
            semantic = semantic_by_name[header.name]
            if (semantic.dtype, semantic.shape) != (header.dtype, header.shape):
                raise WeightSchemaError("tensor semantics differ from checkpoint headers")
            if semantic.partition is not partition_by_name[header.name]:
                raise WeightSchemaError("tensor semantic partition differs from weight partitions")
            if semantic.node_id != node_by_name[header.name]:
                raise WeightSchemaError("tensor semantic node differs from graph node bindings")
        if not isinstance(self.header_evidence, HeaderEvidenceKind):
            raise TypeError("header_evidence must be HeaderEvidenceKind")
        if self.header_evidence is HeaderEvidenceKind.ATTESTED_INVENTORY:
            if self.inventory_digest is None:
                raise WeightSchemaError("attested inventory requires its digest")
        elif self.inventory_digest is not None:
            raise WeightSchemaError("header-only plans cannot claim an inventory digest")
        expected_checkpoint_digest = header_digest(headers, domain=f"{self.model_type}:checkpoint")
        if self.checkpoint_header_digest != expected_checkpoint_digest:
            raise WeightSchemaError("checkpoint header digest differs from its header records")
        by_name = {header.name: header for header in headers}
        base_headers = tuple(by_name[name] for name in self.base_executable_names)
        mtp_headers = tuple(by_name[name] for name in self.mtp_extension_names)
        if self.base_executable_digest != header_digest(
            base_headers, domain=f"{self.model_type}:base"
        ):
            raise WeightSchemaError("base executable digest differs from its header records")
        if self.mtp_extension_digest != header_digest(mtp_headers, domain=f"{self.model_type}:mtp"):
            raise WeightSchemaError("MTP extension digest differs from its header records")
        fp32 = set(self.fp32_stability_names)
        int64 = set(self.int64_metadata_names)
        if fp32 & int64 or not (fp32 | int64) <= set(self.checkpoint_names):
            raise WeightSchemaError("dtype exception names must be disjoint inventory members")
        for header in headers:
            expected_dtype = (
                "F32" if header.name in fp32 else "I64" if header.name in int64 else "BF16"
            )
            if header.dtype != expected_dtype:
                raise WeightSchemaError(
                    f"tensor {header.name!r} dtype is {header.dtype}, expected {expected_dtype}"
                )
        expected_semantic_digest = semantic_plan_digest(
            architecture_identity=self.architecture_identity,
            coverage=self.coverage,
            tensor_semantics=self.tensor_semantics,
            fp32_stability_names=self.fp32_stability_names,
            int64_metadata_names=self.int64_metadata_names,
        )
        if self.semantic_plan_digest != expected_semantic_digest:
            raise WeightSchemaError("semantic plan digest differs from its architecture bindings")
        expected_full_policy = FullBF16WeightPolicy(
            expected_names=self.checkpoint_names,
            fp32_stability_names=self.fp32_stability_names,
            int64_metadata_names=self.int64_metadata_names,
        )
        if self.full_checkpoint_policy != expected_full_policy:
            raise WeightSchemaError("full checkpoint policy must cover the complete inventory")
        expected_base_policy = FullBF16WeightPolicy(
            expected_names=self.base_executable_names,
            fp32_stability_names=tuple(
                name for name in self.fp32_stability_names if name in base_parts
            ),
            int64_metadata_names=tuple(
                name for name in self.int64_metadata_names if name in base_parts
            ),
        )
        if self.base_executable_policy != expected_base_policy:
            raise WeightSchemaError("base policy must cover the complete executable base")


def tensor_headers_from_inventory(inventory: CheckpointInventory) -> tuple[TensorHeader, ...]:
    """Project core-issued inventory evidence without reading tensor payloads."""

    if type(inventory) is not CheckpointInventory:
        raise WeightSchemaError(
            "attested weight planning requires a core-issued CheckpointInventory"
        )
    headers = tuple(
        TensorHeader(name=tensor.name, dtype=tensor.dtype, shape=tensor.shape)
        for tensor in inventory.tensors
    )
    if tuple(header.name for header in headers) != inventory.expected_names:
        raise WeightSchemaError("inventory tensor projection differs from expected_names")
    return headers


def materialize_headers(headers: Iterable[TensorHeader]) -> tuple[TensorHeader, ...]:
    values = tuple(headers)
    if not values:
        raise WeightSchemaError("checkpoint header records cannot be empty")
    names = tuple(header.name for header in values)
    if len(names) != len(set(names)):
        raise WeightSchemaError("checkpoint header records contain duplicate tensor names")
    return tuple(sorted(values, key=lambda header: header.name))


def header_digest(headers: Iterable[TensorHeader], *, domain: str) -> str:
    values = tuple(headers)
    payload = {
        "schema": "cubic.checkpoint.weight-headers",
        "version": 1,
        "domain": domain,
        "tensors": tuple(
            {"name": header.name, "dtype": header.dtype, "shape": header.shape} for header in values
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return f"b2s256:{hashlib.blake2s(encoded.encode()).hexdigest()}"


def semantic_plan_digest(
    *,
    architecture_identity: WeightArchitectureIdentity,
    coverage: WeightSchemaCoverage,
    tensor_semantics: tuple[WeightTensorSemantic, ...],
    fp32_stability_names: tuple[str, ...],
    int64_metadata_names: tuple[str, ...],
) -> str:
    fp32 = set(fp32_stability_names)
    int64 = set(int64_metadata_names)
    payload = {
        "schema": "cubic.checkpoint.weight-semantic-plan",
        "version": 1,
        "architecture": {
            "release_family": architecture_identity.release_family,
            "release_profile": architecture_identity.release_profile,
            "model_type": architecture_identity.model_type,
            "graph_family": architecture_identity.graph_family,
            "architecture": architecture_identity.architecture,
            "plan_status": architecture_identity.plan_status,
        },
        "coverage": coverage.value,
        "tensors": tuple(
            {
                "name": semantic.name,
                "dtype": semantic.dtype,
                "shape": semantic.shape,
                "partition": semantic.partition.value,
                "node_id": semantic.node_id,
                "dtype_role": (
                    "fp32-stability"
                    if semantic.name in fp32
                    else "int64-metadata"
                    if semantic.name in int64
                    else "bf16-parameter"
                ),
            }
            for semantic in tensor_semantics
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return f"b2s256:{hashlib.blake2s(encoded.encode()).hexdigest()}"


def _issue_weight_plan(
    *,
    model_type: str,
    architecture_identity: WeightArchitectureIdentity,
    coverage: WeightSchemaCoverage,
    headers: tuple[TensorHeader, ...],
    inventory: CheckpointInventory | None,
    base_language_names: tuple[str, ...],
    vision_names: tuple[str, ...],
    lm_head_names: tuple[str, ...],
    mtp_extension_names: tuple[str, ...],
    fp32_stability_names: tuple[str, ...],
    int64_metadata_names: tuple[str, ...],
    allowed_node_ids: tuple[str, ...],
    node_bindings: tuple[WeightNodeBinding, ...],
) -> CheckpointWeightPlan:
    if inventory is not None and type(inventory) is not CheckpointInventory:
        raise WeightSchemaError(
            "attested weight-plan issuance requires a core-issued CheckpointInventory"
        )
    headers = materialize_headers(headers)
    if inventory is not None and tensor_headers_from_inventory(inventory) != headers:
        raise WeightSchemaError("attested inventory differs from supplied checkpoint headers")
    by_name = {header.name: header for header in headers}
    checkpoint_names = tuple(sorted(by_name))
    base_names = tuple(sorted((*base_language_names, *vision_names, *lm_head_names)))
    mtp_names = tuple(sorted(mtp_extension_names))
    full_policy = FullBF16WeightPolicy(
        expected_names=checkpoint_names,
        fp32_stability_names=fp32_stability_names,
        int64_metadata_names=int64_metadata_names,
    )
    base_policy = FullBF16WeightPolicy(
        expected_names=base_names,
        fp32_stability_names=tuple(
            name for name in fp32_stability_names if name in set(base_names)
        ),
        int64_metadata_names=tuple(
            name for name in int64_metadata_names if name in set(base_names)
        ),
    )
    base_headers = tuple(by_name[name] for name in base_names)
    mtp_headers = tuple(by_name[name] for name in mtp_names)
    partition_by_name = {
        name: partition
        for names, partition in (
            (base_language_names, WeightPartition.BASE_LANGUAGE),
            (vision_names, WeightPartition.VISION),
            (lm_head_names, WeightPartition.LM_HEAD),
            (mtp_names, WeightPartition.MTP_EXTENSION),
        )
        for name in names
    }
    node_by_name = {
        name: binding.node_id for binding in node_bindings for name in binding.tensor_names
    }
    try:
        tensor_semantics = tuple(
            WeightTensorSemantic(
                name=header.name,
                dtype=header.dtype,
                shape=header.shape,
                partition=partition_by_name[header.name],
                node_id=node_by_name[header.name],
            )
            for header in headers
        )
    except KeyError as error:
        raise WeightSchemaError(
            "partitions and node bindings must classify every checkpoint header"
        ) from error
    plan_digest = semantic_plan_digest(
        architecture_identity=architecture_identity,
        coverage=coverage,
        tensor_semantics=tensor_semantics,
        fp32_stability_names=fp32_stability_names,
        int64_metadata_names=int64_metadata_names,
    )
    return CheckpointWeightPlan(
        model_type=model_type,
        architecture_identity=architecture_identity,
        coverage=coverage,
        header_evidence=(
            HeaderEvidenceKind.ATTESTED_INVENTORY
            if inventory is not None
            else HeaderEvidenceKind.HEADER_RECORDS_ONLY
        ),
        inventory_digest=inventory.inventory_digest if inventory is not None else None,
        checkpoint_headers=headers,
        checkpoint_names=checkpoint_names,
        checkpoint_header_digest=header_digest(headers, domain=f"{model_type}:checkpoint"),
        tensor_semantics=tensor_semantics,
        semantic_plan_digest=plan_digest,
        base_language_names=base_language_names,
        vision_names=vision_names,
        lm_head_names=lm_head_names,
        base_executable_names=base_names,
        base_executable_digest=header_digest(base_headers, domain=f"{model_type}:base"),
        mtp_extension_names=mtp_names,
        mtp_extension_digest=header_digest(mtp_headers, domain=f"{model_type}:mtp"),
        fp32_stability_names=fp32_stability_names,
        int64_metadata_names=int64_metadata_names,
        allowed_node_ids=allowed_node_ids,
        node_bindings=node_bindings,
        full_checkpoint_policy=full_policy,
        base_executable_policy=base_policy,
        _token=_PLAN_ISSUER_TOKEN,
    )


def _require_names(names: tuple[str, ...], *, field: str, allow_empty: bool) -> None:
    if not allow_empty and not names:
        raise WeightSchemaError(f"{field} cannot be empty")
    if names != tuple(sorted(set(names))):
        raise WeightSchemaError(f"{field} must be unique and sorted")
    if any(not name or name != name.strip() for name in names):
        raise WeightSchemaError(f"{field} contains a non-canonical name")
