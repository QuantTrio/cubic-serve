"""Model, weight, cache, and persistent-state contracts.

This module deliberately contains no tensor execution, device discovery, or file I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from cubic_serve._compat import StrEnum


class ScalarDType(StrEnum):
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"
    FP8_E4M3FN = "fp8_e4m3fn"
    FP8_E5M2 = "fp8_e5m2"
    INT8 = "int8"
    INT4 = "int4"
    INT64 = "int64"


class WeightEncoding(StrEnum):
    DENSE = "dense"
    FP8 = "fp8"
    CUBIC = "cubic"


class TensorRole(StrEnum):
    PARAMETER = "parameter"
    STABILITY_PARAMETER = "stability_parameter"
    METADATA = "metadata"


def _require_content_digest(value: str, field: str) -> None:
    algorithm, separator, digest = value.partition(":")
    if algorithm != "b2s256" or separator != ":" or len(digest) != 64:
        raise ValueError(f"{field} must use b2s256:<64 hex chars>")
    try:
        int(digest, 16)
    except ValueError as error:
        raise ValueError(f"{field} contains non-hexadecimal characters") from error


@dataclass(frozen=True, slots=True)
class ParameterDTypeOverride:
    """An exact, loader-verified non-default tensor group in a checkpoint manifest."""

    group_id: str
    role: TensorRole
    dtype: ScalarDType
    tensor_count: int
    total_numel: int
    tensor_names_digest: str
    reason: str

    def __post_init__(self) -> None:
        if not self.group_id or not self.reason:
            raise ValueError("parameter override identity and reason are required")
        if self.tensor_count <= 0 or self.total_numel <= 0:
            raise ValueError("parameter override counts must be positive")
        _require_content_digest(self.tensor_names_digest, "tensor names digest")


@dataclass(frozen=True, slots=True)
class WeightManifest:
    """Completeness, pruning, encoding, and storage dtypes of exact checkpoint tensors."""

    manifest_digest: str
    encoding: WeightEncoding
    default_dtype: ScalarDType
    tensor_count: int
    total_numel: int
    complete: bool
    pruned_layer_ids: tuple[int, ...] = ()
    dtype_overrides: tuple[ParameterDTypeOverride, ...] = ()

    def __post_init__(self) -> None:
        _require_content_digest(self.manifest_digest, "weight manifest digest")
        if self.tensor_count <= 0 or self.total_numel <= 0:
            raise ValueError("weight manifest counts must be positive")
        if tuple(sorted(set(self.pruned_layer_ids))) != self.pruned_layer_ids:
            raise ValueError("pruned layer ids must be non-negative, unique, and sorted")
        if self.pruned_layer_ids and self.pruned_layer_ids[0] < 0:
            raise ValueError("pruned layer ids must be non-negative")
        group_ids = tuple(override.group_id for override in self.dtype_overrides)
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("parameter dtype override groups must be unique")
        if sum(override.tensor_count for override in self.dtype_overrides) >= self.tensor_count:
            raise ValueError("default dtype must cover at least one checkpoint tensor")
        if sum(override.total_numel for override in self.dtype_overrides) >= self.total_numel:
            raise ValueError("default dtype must cover at least one checkpoint element")

    @property
    def is_full_bf16_weights(self) -> bool:
        allowed_overrides = {
            (TensorRole.STABILITY_PARAMETER, ScalarDType.FP32),
            (TensorRole.METADATA, ScalarDType.INT64),
        }
        return (
            self.encoding is WeightEncoding.DENSE
            and self.default_dtype is ScalarDType.BF16
            and self.complete
            and not self.pruned_layer_ids
            and all(
                (override.role, override.dtype) in allowed_overrides
                for override in self.dtype_overrides
            )
        )


class StateInitialization(StrEnum):
    ZERO = "zero"
    MODEL_DEFINED = "model_defined"


class StateValidity(StrEnum):
    FULL_TENSOR = "full_tensor"
    EXPLICIT_MASK = "explicit_mask"


class PrefixReplayPolicy(StrEnum):
    NONE = "none"
    FROM_LAST_ALIGNED_COMMIT = "from_last_aligned_commit"


class StateSharding(StrEnum):
    REPLICATED = "replicated"
    TENSOR = "tensor"
    PIPELINE = "pipeline"
    EXPERT = "expert"


@dataclass(frozen=True, slots=True)
class ModalitySpec:
    """One canonical input modality understood by a model adapter."""

    name: str
    mime_prefixes: tuple[str, ...] = ()
    encoder_id: str | None = None
    feature_hidden_size: int | None = None

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip().lower():
            raise ValueError("modality name must be a non-empty canonical lowercase token")
        if len(set(self.mime_prefixes)) != len(self.mime_prefixes):
            raise ValueError(f"duplicate MIME prefix in modality {self.name!r}")
        if any(not prefix or prefix != prefix.lower() for prefix in self.mime_prefixes):
            raise ValueError("MIME prefixes must be non-empty lowercase strings")
        if self.encoder_id is not None and (
            not self.encoder_id or self.encoder_id != self.encoder_id.strip()
        ):
            raise ValueError("encoder identity must be non-empty and canonical")
        if (self.encoder_id is None) != (self.feature_hidden_size is None):
            raise ValueError(
                "encoder identity and positive feature hidden size must be declared together"
            )
        if self.feature_hidden_size is not None and (
            type(self.feature_hidden_size) is not int or self.feature_hidden_size <= 0
        ):
            raise ValueError("feature hidden size must be a positive integer")


@dataclass(frozen=True, slots=True)
class StateSpec:
    """Persistent non-KV state required to resume a model graph exactly."""

    state_id: str
    dtype: ScalarDType
    layout: str
    shape_symbols: tuple[str, ...]
    schema_version: str = "1"
    initialization: StateInitialization = StateInitialization.ZERO
    validity: StateValidity = StateValidity.EXPLICIT_MASK
    commit_alignment_tokens: int = 1
    prefix_replay: PrefixReplayPolicy = PrefixReplayPolicy.NONE
    sharding: StateSharding = StateSharding.REPLICATED
    transactional: bool = True

    def __post_init__(self) -> None:
        if not self.state_id or not self.layout or not self.schema_version:
            raise ValueError("state_id, layout, and schema_version are required")
        if not self.shape_symbols or any(not symbol for symbol in self.shape_symbols):
            raise ValueError("state shape must be expressed with non-empty symbols")
        if len(set(self.shape_symbols)) != len(self.shape_symbols):
            raise ValueError("state shape symbols must be unique")
        if self.commit_alignment_tokens <= 0:
            raise ValueError("state commit alignment must be positive")
        if self.commit_alignment_tokens > 1 and self.prefix_replay is PrefixReplayPolicy.NONE:
            raise ValueError("unaligned prefix state requires an explicit replay policy")


@dataclass(frozen=True, slots=True)
class CacheSpec:
    """Attention KV cache ABI; recurrent state is described separately."""

    kv_dtype: ScalarDType
    page_size_tokens: int
    layout: str
    prefix_enabled: bool

    def __post_init__(self) -> None:
        if self.page_size_tokens <= 0:
            raise ValueError("page_size_tokens must be positive")
        if not self.layout:
            raise ValueError("cache layout is required")


@dataclass(frozen=True, slots=True)
class WeightVariantMeta:
    """Identity of the effective weights; v0.1 accepts only the base variant."""

    base_model_id: str
    base_revision: str
    adapter_id: str | None = None
    adapter_revision: str | None = None
    adapter_scale: float | None = None

    def __post_init__(self) -> None:
        if not self.base_model_id or not self.base_revision:
            raise ValueError("base model identity and revision are required")
        adapter_fields = (self.adapter_id, self.adapter_revision, self.adapter_scale)
        if any(value is not None for value in adapter_fields) and not all(
            value is not None for value in adapter_fields
        ):
            raise ValueError("adapter id, revision, and scale must be declared together")

    @property
    def is_base(self) -> bool:
        return self.adapter_id is None


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Frozen semantic identity of one loaded model graph."""

    model_id: str
    architecture: str
    revision: str
    weights: WeightManifest
    cache: CacheSpec
    modalities: tuple[ModalitySpec, ...]
    states: tuple[StateSpec, ...] = ()
    layer_count: int | None = None

    def __post_init__(self) -> None:
        if not self.model_id or not self.architecture or not self.revision:
            raise ValueError("model_id, architecture, and revision are required")
        if self.layer_count is not None and self.layer_count <= 0:
            raise ValueError("layer_count must be positive when declared")

        modality_names = tuple(spec.name for spec in self.modalities)
        if not modality_names or "text" not in modality_names:
            raise ValueError("every generative model must declare the text input modality")
        if len(set(modality_names)) != len(modality_names):
            raise ValueError("modality names must be unique")

        state_ids = tuple(spec.state_id for spec in self.states)
        if len(set(state_ids)) != len(state_ids):
            raise ValueError("state ids must be unique")

    @property
    def is_full_bf16(self) -> bool:
        """Whether the model satisfies Cubic Serve's full-BF16 baseline contract."""

        return self.weights.is_full_bf16_weights and self.cache.kv_dtype is ScalarDType.BF16

    def require_full_bf16(self) -> None:
        if not self.is_full_bf16:
            raise ValueError(
                "full BF16 requires a complete unpruned dense BF16 manifest, only declared "
                "FP32 stability/I64 metadata exceptions, and BF16 attention KV cache"
            )
