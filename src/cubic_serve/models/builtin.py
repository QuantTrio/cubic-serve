"""Side-effect-free dispatch for built-in checkpoint configuration parsers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

if TYPE_CHECKING:
    from cubic_serve.models.quantized_checkpoint import CheckpointWeights, LogicalWeight

from cubic_serve.models.adapter import ModelDefinition
from cubic_serve.models.loader import LoadedModelAdapter
from cubic_serve.models.qwen3_5 import MODEL_DEFINITION as DENSE_MODEL_DEFINITION
from cubic_serve.models.qwen3_5 import (
    Qwen3_5Config,
)
from cubic_serve.models.qwen3_5.model import Qwen3_5TextConfig
from cubic_serve.models.qwen3_5_moe import MODEL_DEFINITION as MOE_MODEL_DEFINITION
from cubic_serve.models.qwen3_5_moe import (
    Qwen3_5MoeConfig,
)
from cubic_serve.models.qwen4_exp import MODEL_DEFINITION as QWEN4_MODEL_DEFINITION
from cubic_serve.models.qwen4_exp import Qwen4ExpConfig
from cubic_serve.models.qwen4_exp.model import Qwen4ExpTextConfig
from cubic_serve.models.weights import (
    CheckpointWeightSource,
    FullBF16WeightPolicy,
)
from cubic_serve.runtime.cache import KVLayout

BuiltinModelConfig = Qwen3_5Config | Qwen3_5MoeConfig | Qwen4ExpConfig
BUILTIN_MODEL_DEFINITIONS: Final = (
    DENSE_MODEL_DEFINITION, MOE_MODEL_DEFINITION, QWEN4_MODEL_DEFINITION,
)
BUILTIN_MODEL_TYPES: Final = tuple(item.model_type for item in BUILTIN_MODEL_DEFINITIONS)

__all__ = (
    "BUILTIN_MODEL_TYPES",
    "BuiltinModelConfig",
    "BuiltinModelConfigError",
    "Qwen3_5Admission",
    "admit_model_checkpoint",
    "admit_qwen3_5_reference",
    "builtin_model_definition",
    "builtin_model_definition_for_type",
    "builtin_text_config",
    "model_draft_provider",
    "parse_builtin_model_config",
    "parse_builtin_model_config_json",
    "parse_model_config",
    "parse_model_config_json",
    "resolve_model_definition",
)


def builtin_text_config(config: BuiltinModelConfig) -> Qwen3_5TextConfig | Qwen4ExpTextConfig:
    """Read the declared text schema without pretending model graphs are identical."""
    return config.text_config


def builtin_draft_provider(config: BuiltinModelConfig, *, method: str, max_depth: int):
    """Resolve declared model semantics; draft policy and execution stay shared."""
    return model_draft_provider(config, method=method, max_depth=max_depth)


def model_draft_provider(
    config: Any, *, method: str, max_depth: int,
    definition: ModelDefinition[Any] | None = None,
):
    try:
        factory = resolve_model_definition(config, definition=definition).draft_provider
    except BuiltinModelConfigError:
        factory = None
    if factory is None:
        raise BuiltinModelConfigError(
            "model has no qualified shared-graph draft construction binding"
        )
    return factory(config, method=method, max_depth=max_depth)


def builtin_model_definition(config: BuiltinModelConfig) -> ModelDefinition[Any]:
    """Resolve built-in startup semantics once, outside runtime execution."""
    model_type = getattr(getattr(config, "identity", None), "model_type", None)
    try:
        return builtin_model_definition_for_type(model_type)
    except BuiltinModelConfigError as error:
        raise BuiltinModelConfigError("model has no logical weight plan") from error


def builtin_model_definition_for_type(model_type: object) -> ModelDefinition[Any]:
    """The immutable project registry contains declarations, not model strategies."""
    for definition in BUILTIN_MODEL_DEFINITIONS:
        if definition.model_type == model_type:
            return definition
    raise BuiltinModelConfigError(
        f"unsupported built-in config.model_type; expected one of {BUILTIN_MODEL_TYPES!r}"
    )


def resolve_model_definition(
    config: Any, *, definition: ModelDefinition[Any] | None = None,
) -> ModelDefinition[Any]:
    """Project support is the default; only an explicit declaration overrides it."""
    if definition is None:
        return builtin_model_definition(config)
    if not isinstance(definition, ModelDefinition):
        raise TypeError("explicit model binding must be a ModelDefinition")
    return definition


def builtin_logical_weights(config: BuiltinModelConfig) -> tuple[LogicalWeight, ...]:
    """One complete logical checkpoint plan shared by its input and text modules."""
    return builtin_model_definition(config).logical_weights(config)


def admit_builtin_checkpoint(
    model_root: str | Path, *, model_config: BuiltinModelConfig | None = None
) -> CheckpointWeights:
    """Admit all modules once before their worker topologies start."""
    return admit_model_checkpoint(model_root, model_config=model_config)


def admit_model_checkpoint(
    model_root: str | Path, *, model_config: Any = None,
    definition: ModelDefinition[Any] | None = None,
    require_quantization: bool = False,
) -> CheckpointWeights:
    """Use one declared logical weight plan for every checkpoint module."""
    from cubic_serve.models.quantized_checkpoint import CheckpointWeights
    root = Path(model_root)
    config = (parse_model_config_json((root / "config.json").read_bytes(), definition=definition)
              if model_config is None else model_config)
    definition = resolve_model_definition(config, definition=definition)
    logical = definition.logical_weights(config)
    if (root / "gguf.index.json").is_file() or any(root.glob("*.gguf")):
        from cubic_serve.models.gguf_weights import admit_gguf_checkpoint

        declaration = dict(definition.checkpoint_imports).get("gguf")
        if declaration is None:
            raise BuiltinModelConfigError("model package has no GGUF checkpoint declaration")
        return admit_gguf_checkpoint(root, config, logical, declaration)
    return CheckpointWeights.admit(root, logical,
                                   verify_payloads=False,
                                   require_quantization=require_quantization)


class BuiltinModelConfigError(ValueError):
    """Root model_type is not an exact built-in loader key."""


@dataclass(frozen=True, slots=True)
class Qwen3_5Admission:
    model_root: Path
    vocabulary_size: int
    adapter: LoadedModelAdapter
    weight_policy: FullBF16WeightPolicy
    checkpoint_weights: CheckpointWeightSource
    kv_layout: KVLayout
    owned_layer_ids: tuple[int, ...]
    max_sequence_tokens: int


def admit_qwen3_5_reference(
    model_root: str | Path, *, verify_payloads: bool = True,
) -> Qwen3_5Admission:
    """Compatibility entry; model configuration belongs to the model package."""
    from cubic_serve.models.qwen3_5.serving import admit_reference

    return admit_reference(model_root, verify_payloads=verify_payloads)


def parse_builtin_model_config(
    raw: Mapping[str, object], *, compute_dtype: str | None = None
) -> BuiltinModelConfig:
    return parse_model_config(raw, compute_dtype=compute_dtype)


def parse_builtin_model_config_json(
    document: str | bytes | bytearray,
    *,
    compute_dtype: str | None = None,
) -> BuiltinModelConfig:
    return parse_model_config_json(document, compute_dtype=compute_dtype)


def parse_model_config(
    raw: Mapping[str, object],
    *,
    compute_dtype: str | None = None,
    definition: ModelDefinition[Any] | None = None,
) -> Any:
    """Use the explicit declaration, otherwise project built-in support only."""

    if compute_dtype is not None:
        from cubic_serve.models.quantization import QuantizationConfig

        compute_dtype = {"fp16": "float16", "bf16": "bfloat16"}.get(compute_dtype, compute_dtype)
        if (
            compute_dtype not in ("bfloat16", "float16")
            or QuantizationConfig.from_model_config(raw) is None
        ):
            raise BuiltinModelConfigError(
                "compute_dtype override requires quantization and FP16/BF16"
            )
    if definition is None:
        definition = builtin_model_definition_for_type(raw.get("model_type"))
    if compute_dtype is not None:
        if definition.compute_config is None:
            raise BuiltinModelConfigError("model has no compute dtype configuration binding")
        raw = definition.compute_config(raw, compute_dtype)
    return definition.parse_config(raw)


def parse_model_config_json(
    document: str | bytes | bytearray,
    *,
    compute_dtype: str | None = None,
    definition: ModelDefinition[Any] | None = None,
) -> Any:
    try:
        value = json.loads(
            document,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise BuiltinModelConfigError(f"invalid built-in config JSON: {error}") from error
    if not isinstance(value, Mapping):
        raise BuiltinModelConfigError("built-in config JSON must contain one object")
    return parse_model_config(
        cast(Mapping[str, object], value), compute_dtype=compute_dtype, definition=definition
    )


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise BuiltinModelConfigError(f"built-in config JSON has duplicate field {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise BuiltinModelConfigError(f"built-in config JSON has non-finite constant {value!r}")
