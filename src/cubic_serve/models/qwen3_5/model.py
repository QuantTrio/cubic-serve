"""Qwen3.5 config, mathematical graph, state schema, and adapter ABI."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from cubic_serve._compat import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch
import torch.nn.functional as functional
from safetensors import safe_open
from torch import Tensor

from cubic_serve.contracts.execution import (
    BatchPhase,
    DecodeExtensionMeta,
    DecodeProvider,
    OpSignature,
    TensorSpec,
)
from cubic_serve.contracts.model import (
    CacheSpec,
    ModalitySpec,
    ModelSpec,
    ScalarDType,
    StateSharding,
    StateSpec,
    WeightVariantMeta,
)
from cubic_serve.inputs import FeatureReader, FeatureSpan
from cubic_serve.models.adapter import ModelAdapterDescriptor
from cubic_serve.models.checkpoint import CheckpointInventory, inspect_checkpoint
from cubic_serve.models.protocol import (
    ModelBuildContext,
    ModelSegment,
    ModelStateIO,
    ModelStepResult,
    ProgramExecutionRecipe,
    SemanticOpExecutor,
    StateInitialization,
)
from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.models.weights import (
    checkpoint_weight_manifest,
)
from cubic_serve.ops.eager.activation import silu_stored as activation_silu
from cubic_serve.ops.eager.attention import full_attention as reference_full_attention
from cubic_serve.ops.eager.hybrid import (
    apply_partial_rotary,
    depthwise_causal_conv1d,
    gated_delta_net_recurrent,
    gated_rms_norm,
    interleaved_mrope_cos_sin,
    sigmoid_output_gate,
    zero_centered_rms_norm,
)
from cubic_serve.ops.eager.linear import linear as reference_linear
from cubic_serve.ops.eager.quantization import embedding_matrix, linear_matrix

QWEN3_5_MTP_MAX_PROPOSAL_TOKENS = 8

if TYPE_CHECKING:
    from cubic_serve.models.qwen3_5.weights import Qwen3_5WeightPlan

CONV_STATE_ID = "gdn_conv"
RECURRENT_STATE_ID = "gdn_recurrent"


class Qwen3_5ConfigError(ValueError):
    """Configuration does not match the closed Qwen3.5 release schema."""


class Qwen3_5GraphVariant(StrEnum):
    DENSE = "dense"
    MOE = "moe"


class AttentionKind(StrEnum):
    GATED_DELTA_NET = "gated-delta-net"
    FULL_ATTENTION = "full-attention"


class FeedForwardKind(StrEnum):
    DENSE = "dense"
    MIXTURE_OF_EXPERTS = "mixture-of-experts"


class LayerStateKind(StrEnum):
    """Logical state classes; runtime owns dtype and physical placement."""

    GDN_CONVOLUTION = "gdn-convolution"
    GDN_RECURRENT_MATRIX = "gdn-recurrent-matrix"
    ATTENTION_KV = "attention-kv"


class ArchitecturePlanStatus(StrEnum):
    ARCHITECTURE_ONLY = "architecture-only"


class MTPStatus(StrEnum):
    METADATA_ONLY = "metadata-only"


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Release identity is deliberately separate from the executable graph ABI."""

    release_family: str
    release_profile: str
    model_type: str
    graph_family: str
    architecture: str


@dataclass(frozen=True, slots=True)
class MRoPEConfig:
    interleaved: bool
    section: tuple[int, int, int]
    partial_rotary_factor: float
    rope_theta: float
    rope_type: str


@dataclass(frozen=True, slots=True)
class GatedDeltaNetConfig:
    convolution_kernel_size: int
    key_head_dim: int
    value_head_dim: int
    key_heads: int
    value_heads: int
    recurrent_state_dtype: str
    output_gate_type: str


@dataclass(frozen=True, slots=True)
class FullAttentionConfig:
    query_heads: int
    key_value_heads: int
    head_dim: int
    output_gate: bool


@dataclass(frozen=True, slots=True)
class MoEConfig:
    expert_count: int
    experts_per_token: int
    expert_intermediate_size: int
    shared_expert_intermediate_size: int
    sparse_step: int
    dense_layer_indexes: tuple[int, ...]
    normalize_topk_probability: bool


@dataclass(frozen=True, slots=True)
class MTPMetadata:
    status: MTPStatus
    hidden_layer_count: int
    dedicated_embeddings: bool
    hybrid: bool | None = None
    layer_schedule: tuple[AttentionKind, ...] = ()
    hidden_state_source_layer: int | None = None
    rope_theta: float | None = None


@dataclass(frozen=True, slots=True)
class VisionConfig:
    model_type: str
    layer_count: int
    hidden_size: int
    intermediate_size: int
    output_hidden_size: int
    head_count: int
    input_channels: int
    patch_size: int
    spatial_merge_size: int
    temporal_patch_size: int
    position_count: int
    activation: str
    deepstack_layer_indexes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ModalityTokenConfig:
    image: int
    video: int
    vision_start: int
    vision_end: int


@dataclass(frozen=True, slots=True)
class Qwen3_5TextConfig:
    model_type: str
    dtype: str
    vocabulary_size: int
    hidden_size: int
    layer_count: int
    max_position_embeddings: int
    activation: str
    rms_norm_epsilon: float
    layer_schedule: tuple[AttentionKind, ...]
    full_attention_interval: int
    dense_intermediate_size: int | None
    full_attention: FullAttentionConfig
    gated_delta_net: GatedDeltaNetConfig
    moe: MoEConfig | None
    mrope: MRoPEConfig
    mtp: MTPMetadata


@dataclass(frozen=True, slots=True)
class Qwen3_5GraphConfig:
    """Architecture facts shared by dense and MoE qwen3_5 graph variants."""

    variant: Qwen3_5GraphVariant
    root_model_type: str
    architecture: str
    text: Qwen3_5TextConfig
    vision: VisionConfig
    modality_tokens: ModalityTokenConfig

    def architecture_plan(self) -> Qwen3_5ArchitecturePlan:
        dense_indexes = frozenset(
            self.text.moe.dense_layer_indexes if self.text.moe is not None else ()
        )
        layers: list[Qwen3_5DecoderLayerPlan] = []
        for layer_index, attention in enumerate(self.text.layer_schedule):
            if self.text.moe is None or layer_index in dense_indexes:
                feed_forward = FeedForwardKind.DENSE
            elif (layer_index + 1) % self.text.moe.sparse_step == 0:
                feed_forward = FeedForwardKind.MIXTURE_OF_EXPERTS
            else:
                feed_forward = FeedForwardKind.DENSE
            layers.append(
                Qwen3_5DecoderLayerPlan(
                    layer_index=layer_index,
                    node_id=f"text.decoder.layer.{layer_index:04d}",
                    attention=attention,
                    state_requirements=(
                        (
                            LayerStateKind.GDN_CONVOLUTION,
                            LayerStateKind.GDN_RECURRENT_MATRIX,
                        )
                        if attention is AttentionKind.GATED_DELTA_NET
                        else (LayerStateKind.ATTENTION_KV,)
                    ),
                    feed_forward=feed_forward,
                )
            )
        return Qwen3_5ArchitecturePlan(
            status=ArchitecturePlanStatus.ARCHITECTURE_ONLY,
            graph_family="qwen3_5",
            vision_patch_embedding_node_id="vision.patch_embedding",
            vision_layers=tuple(
                VisionLayerPlan(index, f"vision.encoder.layer.{index:04d}")
                for index in range(self.vision.layer_count)
            ),
            vision_merger_node_id="vision.patch_merger",
            token_embedding_node_id="text.token_embedding",
            decoder_layers=tuple(layers),
            final_norm_node_id="text.final_norm",
            lm_head_node_id="text.lm_head",
            mtp_root_node_id="extension.mtp",
            mtp_layers=tuple(
                MTPLayerPlan(
                    index,
                    f"extension.mtp.layer.{index:04d}",
                    AttentionKind.FULL_ATTENTION,
                    (LayerStateKind.ATTENTION_KV,),
                )
                for index in range(self.text.mtp.hidden_layer_count)
            ),
        )


@dataclass(frozen=True, slots=True)
class VisionLayerPlan:
    layer_index: int
    node_id: str


@dataclass(frozen=True, slots=True)
class Qwen3_5DecoderLayerPlan:
    layer_index: int
    node_id: str
    attention: AttentionKind
    state_requirements: tuple[LayerStateKind, ...]
    feed_forward: FeedForwardKind

    @property
    def state_node_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{self.node_id}.state.{requirement.value}" for requirement in self.state_requirements
        )


@dataclass(frozen=True, slots=True)
class MTPLayerPlan:
    layer_index: int
    node_id: str
    attention: AttentionKind
    state_requirements: tuple[LayerStateKind, ...]

    @property
    def state_node_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{self.node_id}.state.{requirement.value}" for requirement in self.state_requirements
        )


@dataclass(frozen=True, slots=True)
class Qwen3_5ArchitecturePlan:
    status: ArchitecturePlanStatus
    graph_family: str
    vision_patch_embedding_node_id: str
    vision_layers: tuple[VisionLayerPlan, ...]
    vision_merger_node_id: str
    token_embedding_node_id: str
    decoder_layers: tuple[Qwen3_5DecoderLayerPlan, ...]
    final_norm_node_id: str
    lm_head_node_id: str
    mtp_root_node_id: str
    mtp_layers: tuple[MTPLayerPlan, ...]


@dataclass(frozen=True, slots=True)
class Qwen3_5Config:
    """Configuration for the qwen3_5 model architecture."""

    identity: ModelIdentity
    graph: Qwen3_5GraphConfig

    @property
    def text_config(self) -> Qwen3_5TextConfig:
        """Expose the original text schema without copying or changing policy."""
        return self.graph.text

    def architecture_plan(self) -> Qwen3_5ArchitecturePlan:
        return self.graph.architecture_plan()


_TOP_KEYS = frozenset(
    {
        "architectures",
        "image_token_id",
        "language_model_only",
        "name_or_path",
        "model_type",
        "quantization_config",
        "text_config",
        "tie_word_embeddings",
        "transformers_version",
        "video_token_id",
        "vision_config",
        "vision_end_token_id",
        "vision_start_token_id",
    }
)
_TOP_REQUIRED = frozenset(
    {
        "architectures",
        "image_token_id",
        "model_type",
        "text_config",
        "tie_word_embeddings",
        "video_token_id",
        "vision_config",
        "vision_end_token_id",
        "vision_start_token_id",
    }
)
_VISION_KEYS = frozenset(
    {
        "deepstack_visual_indexes",
        "depth",
        "hidden_act",
        "hidden_size",
        "in_channels",
        "initializer_range",
        "intermediate_size",
        "model_type",
        "num_heads",
        "num_position_embeddings",
        "out_hidden_size",
        "patch_size",
        "spatial_merge_size",
        "temporal_patch_size",
    }
)
_VISION_REQUIRED = _VISION_KEYS - {"initializer_range"}
_VISION_KEYS = _VISION_KEYS | {"torch_dtype"}
_COMMON_TEXT_KEYS = frozenset(
    {
        "attention_bias",
        "attention_dropout",
        "attn_output_gate",
        "bos_token_id",
        "dtype",
        "eos_token_id",
        "full_attention_interval",
        "head_dim",
        "hidden_act",
        "hidden_size",
        "initializer_range",
        "layer_types",
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_value_head_dim",
        "mamba_ssm_dtype",
        "max_position_embeddings",
        "model_type",
        "mtp_num_hidden_layers",
        "mtp_use_dedicated_embeddings",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "output_gate_type",
        "pad_token_id",
        "partial_rotary_factor",
        "quantization_config",
        "rms_norm_eps",
        "rope_parameters",
        "tie_word_embeddings",
        "use_cache",
        "vocab_size",
    }
)
_COMMON_TEXT_REQUIRED = frozenset(
    {
        "attention_bias",
        "attention_dropout",
        "dtype",
        "full_attention_interval",
        "head_dim",
        "hidden_act",
        "hidden_size",
        "layer_types",
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_value_head_dim",
        "mamba_ssm_dtype",
        "max_position_embeddings",
        "model_type",
        "mtp_num_hidden_layers",
        "mtp_use_dedicated_embeddings",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_parameters",
        "use_cache",
        "vocab_size",
    }
)
_MOE_KEYS = _COMMON_TEXT_KEYS | {
    "decoder_sparse_step",
    "intermediate_size",
    "mlp_only_layers",
    "moe_intermediate_size",
    "norm_topk_prob",
    "num_experts",
    "num_experts_per_tok",
    "output_router_logits",
    "router_aux_loss_coef",
    "shared_expert_intermediate_size",
}
_MOE_REQUIRED = _COMMON_TEXT_REQUIRED | {
    "attn_output_gate",
    "mlp_only_layers",
    "moe_intermediate_size",
    "num_experts",
    "num_experts_per_tok",
    "shared_expert_intermediate_size",
}
_DENSE_KEYS = _COMMON_TEXT_KEYS | {"intermediate_size"}
_DENSE_REQUIRED = _COMMON_TEXT_REQUIRED | {"attn_output_gate", "intermediate_size"}
_ROPE_KEYS = frozenset(
    {
        "mrope_interleaved",
        "mrope_section",
        "partial_rotary_factor",
        "rope_theta",
        "rope_type",
    }
)


def parse_qwen3_5_config(raw: Mapping[str, object]) -> Qwen3_5Config:
    """Parse the supported dense checkpoint with model_type=qwen3_5."""

    graph = _parse_qwen3_5_graph(raw, expected_variant=Qwen3_5GraphVariant.DENSE)
    return Qwen3_5Config(
        identity=ModelIdentity(
            release_family="qwen3_5",
            release_profile="qwen3_5",
            model_type="qwen3_5",
            graph_family="qwen3_5",
            architecture=graph.architecture,
        ),
        graph=graph,
    )


def parse_qwen3_5_config_json(document: str | bytes | bytearray) -> Qwen3_5Config:
    return parse_qwen3_5_config(_parse_json_object(document))


def _parse_qwen3_5_graph(
    raw: Mapping[str, object], *, expected_variant: Qwen3_5GraphVariant
) -> Qwen3_5GraphConfig:
    root = _mapping(raw, scope="config")
    _keys(root, allowed=_TOP_KEYS, required=_TOP_REQUIRED, scope="config")
    if "name_or_path" in root:
        _string(root, "name_or_path", scope="config")
    # Encoding admission is separate from architecture parsing. Validate its
    # syntax here so a malformed declaration is never silently discarded.
    text_declaration = root.get("text_config")
    if "quantization_config" in root or (
        isinstance(text_declaration, Mapping) and "quantization_config" in text_declaration
    ):
        from cubic_serve.models.quantization import QuantizationConfig

        QuantizationConfig.from_model_config(root)
    contracts = {
        Qwen3_5GraphVariant.DENSE: (
            "qwen3_5",
            "Qwen3_5ForConditionalGeneration",
            "qwen3_5_text",
            _DENSE_KEYS,
            _DENSE_REQUIRED,
        ),
        Qwen3_5GraphVariant.MOE: (
            "qwen3_5_moe",
            "Qwen3_5MoeForConditionalGeneration",
            "qwen3_5_moe_text",
            _MOE_KEYS,
            _MOE_REQUIRED,
        ),
    }
    root_type, architecture, text_type, allowed_text, required_text = contracts[expected_variant]
    if _string(root, "model_type", scope="config") != root_type:
        raise Qwen3_5ConfigError(
            f"config.model_type must be {root_type!r} for {expected_variant.value} qwen3_5"
        )
    if _string_tuple(root, "architectures", scope="config") != (architecture,):
        raise Qwen3_5ConfigError(f"config.architectures must be exactly ({architecture!r},)")
    if _boolean(root, "tie_word_embeddings", scope="config"):
        raise Qwen3_5ConfigError("config.tie_word_embeddings=true is unsupported")
    if "language_model_only" in root and _boolean(root, "language_model_only", scope="config"):
        raise Qwen3_5ConfigError("multimodal release cannot be language_model_only")

    text_raw = _child_mapping(root, "text_config", scope="config")
    _keys(text_raw, allowed=allowed_text, required=required_text, scope="config.text_config")
    if _string(text_raw, "model_type", scope="config.text_config") != text_type:
        raise Qwen3_5ConfigError(f"config.text_config.model_type must be {text_type!r}")
    if "tie_word_embeddings" in text_raw and _boolean(
        text_raw, "tie_word_embeddings", scope="config.text_config"
    ):
        raise Qwen3_5ConfigError(
            "config.text_config.tie_word_embeddings contradicts the untied root"
        )
    text = _parse_text(text_raw, variant=expected_variant, model_type=text_type)

    vision_raw = _child_mapping(root, "vision_config", scope="config")
    _keys(
        vision_raw,
        allowed=_VISION_KEYS,
        required=_VISION_REQUIRED,
        scope="config.vision_config",
    )
    if _string(vision_raw, "model_type", scope="config.vision_config") != root_type:
        raise Qwen3_5ConfigError(f"config.vision_config.model_type must be {root_type!r}")
    vision = _parse_vision(vision_raw, model_type=root_type)
    if vision.output_hidden_size != text.hidden_size:
        raise Qwen3_5ConfigError("vision output hidden size must equal text hidden size")

    tokens = ModalityTokenConfig(
        image=_integer(root, "image_token_id", scope="config", minimum=0),
        video=_integer(root, "video_token_id", scope="config", minimum=0),
        vision_start=_integer(root, "vision_start_token_id", scope="config", minimum=0),
        vision_end=_integer(root, "vision_end_token_id", scope="config", minimum=0),
    )
    sentinel_ids = (tokens.image, tokens.video, tokens.vision_start, tokens.vision_end)
    if len(set(sentinel_ids)) != 4 or any(value >= text.vocabulary_size for value in sentinel_ids):
        raise Qwen3_5ConfigError(
            "multimodal sentinel IDs must be distinct and inside the vocabulary"
        )
    return Qwen3_5GraphConfig(
        variant=expected_variant,
        root_model_type=root_type,
        architecture=architecture,
        text=text,
        vision=vision,
        modality_tokens=tokens,
    )


def _parse_text(
    raw: Mapping[str, object], *, variant: Qwen3_5GraphVariant, model_type: str
) -> Qwen3_5TextConfig:
    scope = "config.text_config"
    dtype = _string(raw, "dtype", scope=scope)
    if dtype not in ("bfloat16", "float16"):
        raise Qwen3_5ConfigError(f"{scope}.dtype must be 'bfloat16' or 'float16'")
    if _boolean(raw, "attention_bias", scope=scope):
        raise Qwen3_5ConfigError(f"{scope}.attention_bias=true is unsupported")
    if _number(raw, "attention_dropout", scope=scope, minimum=0.0) != 0.0:
        raise Qwen3_5ConfigError(f"{scope}.attention_dropout must be zero")
    if not _optional_boolean(raw, "attn_output_gate", default=True, scope=scope):
        raise Qwen3_5ConfigError("full attention operators require an output gate")
    if _string(raw, "hidden_act", scope=scope) not in ("silu", "swish"):
        raise Qwen3_5ConfigError("text operators require SiLU activation")
    if _boolean(raw, "mtp_use_dedicated_embeddings", scope=scope):
        raise Qwen3_5ConfigError("dedicated MTP embeddings are not implemented")
    if not _boolean(raw, "use_cache", scope=scope):
        raise Qwen3_5ConfigError(f"{scope}.use_cache must be true")
    hidden_size = _integer(raw, "hidden_size", scope=scope, minimum=1)
    layer_count = _integer(raw, "num_hidden_layers", scope=scope, minimum=1)
    interval = _integer(raw, "full_attention_interval", scope=scope, minimum=2)
    schedule = _attention_schedule(raw, layer_count=layer_count, interval=interval)
    query_heads = _integer(raw, "num_attention_heads", scope=scope, minimum=1)
    kv_heads = _integer(raw, "num_key_value_heads", scope=scope, minimum=1)
    if query_heads % kv_heads:
        raise Qwen3_5ConfigError("full-attention query heads must divide into KV groups")
    head_dim = _integer(raw, "head_dim", scope=scope, minimum=1)
    mrope = _parse_mrope(raw, head_dim=head_dim)
    gdn = _parse_gdn(raw)
    moe = _parse_moe(raw, layer_count=layer_count) if variant is Qwen3_5GraphVariant.MOE else None
    dense_intermediate = (
        _integer(raw, "intermediate_size", scope=scope, minimum=1)
        if "intermediate_size" in raw
        else None
    )
    if variant is Qwen3_5GraphVariant.DENSE and dense_intermediate is None:
        raise Qwen3_5ConfigError(f"{scope}.intermediate_size is required for dense graph")
    return Qwen3_5TextConfig(
        model_type=model_type,
        dtype=dtype,
        vocabulary_size=_integer(raw, "vocab_size", scope=scope, minimum=1),
        hidden_size=hidden_size,
        layer_count=layer_count,
        max_position_embeddings=_integer(raw, "max_position_embeddings", scope=scope, minimum=1),
        activation=_string(raw, "hidden_act", scope=scope),
        rms_norm_epsilon=_number(raw, "rms_norm_eps", scope=scope, exclusive_minimum=0.0),
        layer_schedule=schedule,
        full_attention_interval=interval,
        dense_intermediate_size=dense_intermediate,
        full_attention=FullAttentionConfig(
            query_heads=query_heads,
            key_value_heads=kv_heads,
            head_dim=head_dim,
            output_gate=_optional_boolean(raw, "attn_output_gate", default=True, scope=scope),
        ),
        gated_delta_net=gdn,
        moe=moe,
        mrope=mrope,
        mtp=MTPMetadata(
            status=MTPStatus.METADATA_ONLY,
            hidden_layer_count=_integer(raw, "mtp_num_hidden_layers", scope=scope, minimum=0),
            dedicated_embeddings=_boolean(raw, "mtp_use_dedicated_embeddings", scope=scope),
        ),
    )


def _parse_mrope(raw: Mapping[str, object], *, head_dim: int) -> MRoPEConfig:
    scope = "config.text_config"
    rope = _child_mapping(raw, "rope_parameters", scope=scope)
    _keys(rope, allowed=_ROPE_KEYS, required=_ROPE_KEYS, scope=f"{scope}.rope_parameters")
    if not _boolean(rope, "mrope_interleaved", scope=f"{scope}.rope_parameters"):
        raise Qwen3_5ConfigError("M-RoPE must be interleaved")
    raw_section = _integer_tuple(rope, "mrope_section", scope=f"{scope}.rope_parameters", minimum=1)
    if len(raw_section) != 3:
        raise Qwen3_5ConfigError("M-RoPE section must contain three axes")
    section = (raw_section[0], raw_section[1], raw_section[2])
    partial = _number(
        rope,
        "partial_rotary_factor",
        scope=f"{scope}.rope_parameters",
        exclusive_minimum=0.0,
        maximum=1.0,
    )
    if (
        "partial_rotary_factor" in raw
        and _number(
            raw,
            "partial_rotary_factor",
            scope=scope,
            exclusive_minimum=0.0,
            maximum=1.0,
        )
        != partial
    ):
        raise Qwen3_5ConfigError("partial_rotary_factor contradicts rope_parameters")
    rotary_dim = head_dim * partial
    if not rotary_dim.is_integer() or int(rotary_dim) % 2 or sum(section) * 2 != int(rotary_dim):
        raise Qwen3_5ConfigError("M-RoPE section does not partition the rotary dimension")
    rope_type = _string(rope, "rope_type", scope=f"{scope}.rope_parameters")
    if rope_type != "default":
        raise Qwen3_5ConfigError(f"unsupported rope type {rope_type!r}")
    return MRoPEConfig(
        interleaved=True,
        section=section,
        partial_rotary_factor=partial,
        rope_theta=_number(
            rope, "rope_theta", scope=f"{scope}.rope_parameters", exclusive_minimum=0.0
        ),
        rope_type=rope_type,
    )


def _parse_gdn(raw: Mapping[str, object]) -> GatedDeltaNetConfig:
    scope = "config.text_config"
    state_dtype = _string(raw, "mamba_ssm_dtype", scope=scope)
    if state_dtype != "float32":
        raise Qwen3_5ConfigError("GDN recurrent state dtype must be float32")
    key_heads = _integer(raw, "linear_num_key_heads", scope=scope, minimum=1)
    value_heads = _integer(raw, "linear_num_value_heads", scope=scope, minimum=1)
    if value_heads % key_heads:
        raise Qwen3_5ConfigError("GDN value heads must be divisible by key heads")
    gate = _optional_string(raw, "output_gate_type", default="silu", scope=scope)
    if gate not in {"sigmoid", "silu", "swish"}:
        raise Qwen3_5ConfigError(f"unsupported GDN output gate {gate!r}")
    return GatedDeltaNetConfig(
        convolution_kernel_size=_integer(raw, "linear_conv_kernel_dim", scope=scope, minimum=1),
        key_head_dim=_integer(raw, "linear_key_head_dim", scope=scope, minimum=1),
        value_head_dim=_integer(raw, "linear_value_head_dim", scope=scope, minimum=1),
        key_heads=key_heads,
        value_heads=value_heads,
        recurrent_state_dtype=state_dtype,
        output_gate_type=gate,
    )


def _parse_moe(raw: Mapping[str, object], *, layer_count: int) -> MoEConfig:
    scope = "config.text_config"
    expert_count = _integer(raw, "num_experts", scope=scope, minimum=1)
    active = _integer(raw, "num_experts_per_tok", scope=scope, minimum=1)
    if active > expert_count:
        raise Qwen3_5ConfigError("active experts cannot exceed expert count")
    dense_layers = _integer_tuple(raw, "mlp_only_layers", scope=scope, minimum=0)
    if len(set(dense_layers)) != len(dense_layers) or any(
        index >= layer_count for index in dense_layers
    ):
        raise Qwen3_5ConfigError("mlp_only_layers must be unique valid layer indexes")
    sparse_step = _optional_integer(raw, "decoder_sparse_step", default=1, scope=scope, minimum=1)
    if (dense_layers or sparse_step != 1) and "intermediate_size" not in raw:
        raise Qwen3_5ConfigError("dense MoE layers require intermediate_size")
    return MoEConfig(
        expert_count=expert_count,
        experts_per_token=active,
        expert_intermediate_size=_integer(raw, "moe_intermediate_size", scope=scope, minimum=1),
        shared_expert_intermediate_size=_integer(
            raw, "shared_expert_intermediate_size", scope=scope, minimum=0
        ),
        sparse_step=sparse_step,
        dense_layer_indexes=dense_layers,
        normalize_topk_probability=_optional_boolean(
            raw, "norm_topk_prob", default=True, scope=scope
        ),
    )


def _parse_vision(raw: Mapping[str, object], *, model_type: str) -> VisionConfig:
    scope = "config.vision_config"
    layers = _integer(raw, "depth", scope=scope, minimum=1)
    deepstack = _integer_tuple(raw, "deepstack_visual_indexes", scope=scope, minimum=0)
    if len(set(deepstack)) != len(deepstack) or any(index >= layers for index in deepstack):
        raise Qwen3_5ConfigError("deepstack indexes must be unique valid vision layers")
    return VisionConfig(
        model_type=model_type,
        layer_count=layers,
        hidden_size=_integer(raw, "hidden_size", scope=scope, minimum=1),
        intermediate_size=_integer(raw, "intermediate_size", scope=scope, minimum=1),
        output_hidden_size=_integer(raw, "out_hidden_size", scope=scope, minimum=1),
        head_count=_integer(raw, "num_heads", scope=scope, minimum=1),
        input_channels=_integer(raw, "in_channels", scope=scope, minimum=1),
        patch_size=_integer(raw, "patch_size", scope=scope, minimum=1),
        spatial_merge_size=_integer(raw, "spatial_merge_size", scope=scope, minimum=1),
        temporal_patch_size=_integer(raw, "temporal_patch_size", scope=scope, minimum=1),
        position_count=_integer(raw, "num_position_embeddings", scope=scope, minimum=1),
        activation=_string(raw, "hidden_act", scope=scope),
        deepstack_layer_indexes=deepstack,
    )






def _attention_schedule(
    raw: Mapping[str, object], *, layer_count: int, interval: int
) -> tuple[AttentionKind, ...]:
    names = _string_tuple(raw, "layer_types", scope="config.text_config")
    if len(names) != layer_count:
        raise Qwen3_5ConfigError("layer_types length must match num_hidden_layers")
    schedule = tuple(_attention_kind(name) for name in names)
    expected = tuple(
        AttentionKind.FULL_ATTENTION
        if (index + 1) % interval == 0
        else AttentionKind.GATED_DELTA_NET
        for index in range(layer_count)
    )
    if schedule != expected:
        raise Qwen3_5ConfigError("layer_types contradicts full_attention_interval")
    return schedule


def _attention_kind(value: str) -> AttentionKind:
    if value == "linear_attention":
        return AttentionKind.GATED_DELTA_NET
    if value == "full_attention":
        return AttentionKind.FULL_ATTENTION
    raise Qwen3_5ConfigError(f"unsupported attention kind {value!r}")


def _parse_json_object(document: str | bytes | bytearray) -> dict[str, object]:
    try:
        value = json.loads(
            document,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise Qwen3_5ConfigError(f"invalid config JSON: {error}") from error
    if not isinstance(value, Mapping):
        raise Qwen3_5ConfigError("config JSON must contain one object")
    return _mapping(cast(Mapping[str, object], value), scope="config")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise Qwen3_5ConfigError(f"config JSON contains duplicate field {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise Qwen3_5ConfigError(f"config JSON contains non-finite constant {value!r}")


def _mapping(raw: Mapping[str, object], *, scope: str) -> dict[str, object]:
    if any(not isinstance(key, str) for key in raw):
        raise Qwen3_5ConfigError(f"{scope} keys must be strings")
    return dict(raw)


def _child_mapping(raw: Mapping[str, object], key: str, *, scope: str) -> dict[str, object]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise Qwen3_5ConfigError(f"{scope}.{key} must be an object")
    return _mapping(cast(Mapping[str, object], value), scope=f"{scope}.{key}")


def _keys(
    raw: Mapping[str, object], *, allowed: frozenset[str], required: frozenset[str], scope: str
) -> None:
    present = frozenset(raw)
    missing = sorted(required - present)
    unknown = sorted(present - allowed)
    if missing or unknown:
        raise Qwen3_5ConfigError(f"{scope} schema mismatch: missing={missing}, unknown={unknown}")


def _string(raw: Mapping[str, object], key: str, *, scope: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise Qwen3_5ConfigError(f"{scope}.{key} must be a non-empty string")
    return value


def _optional_string(raw: Mapping[str, object], key: str, *, default: str, scope: str) -> str:
    return _string(raw, key, scope=scope) if key in raw else default


def _boolean(raw: Mapping[str, object], key: str, *, scope: str) -> bool:
    value = raw.get(key)
    if type(value) is not bool:
        raise Qwen3_5ConfigError(f"{scope}.{key} must be a boolean")
    return value


def _optional_boolean(raw: Mapping[str, object], key: str, *, default: bool, scope: str) -> bool:
    return _boolean(raw, key, scope=scope) if key in raw else default


def _integer(raw: Mapping[str, object], key: str, *, scope: str, minimum: int | None = None) -> int:
    value = raw.get(key)
    if type(value) is not int:
        raise Qwen3_5ConfigError(f"{scope}.{key} must be an integer")
    if minimum is not None and value < minimum:
        raise Qwen3_5ConfigError(f"{scope}.{key} must be >= {minimum}")
    return value


def _optional_integer(
    raw: Mapping[str, object], key: str, *, default: int, scope: str, minimum: int
) -> int:
    return _integer(raw, key, scope=scope, minimum=minimum) if key in raw else default


def _number(
    raw: Mapping[str, object],
    key: str,
    *,
    scope: str,
    minimum: float | None = None,
    exclusive_minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise Qwen3_5ConfigError(f"{scope}.{key} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise Qwen3_5ConfigError(f"{scope}.{key} must be finite")
    if minimum is not None and result < minimum:
        raise Qwen3_5ConfigError(f"{scope}.{key} must be >= {minimum}")
    if exclusive_minimum is not None and result <= exclusive_minimum:
        raise Qwen3_5ConfigError(f"{scope}.{key} must be > {exclusive_minimum}")
    if maximum is not None and result > maximum:
        raise Qwen3_5ConfigError(f"{scope}.{key} must be <= {maximum}")
    return result


def _sequence(raw: Mapping[str, object], key: str, *, scope: str) -> Sequence[object]:
    value = raw.get(key)
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise Qwen3_5ConfigError(f"{scope}.{key} must be an array")
    return cast(Sequence[object], value)


def _string_tuple(raw: Mapping[str, object], key: str, *, scope: str) -> tuple[str, ...]:
    values = _sequence(raw, key, scope=scope)
    if any(not isinstance(value, str) or not value for value in values):
        raise Qwen3_5ConfigError(f"{scope}.{key} must contain non-empty strings")
    return tuple(cast(str, value) for value in values)


def _integer_tuple(
    raw: Mapping[str, object], key: str, *, scope: str, minimum: int
) -> tuple[int, ...]:
    values = _sequence(raw, key, scope=scope)
    if any(type(value) is not int or value < minimum for value in values):
        raise Qwen3_5ConfigError(f"{scope}.{key} must contain integers >= {minimum}")
    return tuple(cast(int, value) for value in values)


class Qwen3_5ReferenceLoadError(RuntimeError):
    """The checkpoint cannot become the closed eager text program."""


class Qwen3_5FeatureFusionError(ValueError):
    """Authorized vision rows do not exactly close one decoder input span."""


class Qwen3_5OperatorCalls(Protocol):
    """Expensive semantic operations supplied by the active execution mode."""

    def linear(self, inputs: Tensor, weight: Any) -> Tensor: ...

    def full_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        query_ordinals: Tensor,
        key_ordinals: Tensor,
    ) -> Tensor: ...


class Qwen3_5ModelOperators(Qwen3_5OperatorCalls, Protocol):
    """Model-level lookup in addition to segment-local linear/attention calls."""

    def embedding(self, input_ids: Tensor, weight: Any, *, dtype: torch.dtype) -> Tensor: ...


class Qwen3_5WeightAccess(Protocol):
    """Least interface required by the canonical decoder math."""

    device: torch.device
    inventory_digest: str
    semantic_plan_digest: str

    @property
    def tensor_count(self) -> int: ...

    # Plain operands and bound matrices share the closed checkpoint namespace.
    # The logical consumer plan, not a synthetic Tensor subclass, fixes the type.
    def __getitem__(self, name: str) -> Any: ...

    def named_tensors(self) -> tuple[tuple[str, Tensor], ...]: ...


class _DirectReferenceOperators:
    def embedding(
        self, input_ids: Tensor, weight: Tensor | QuantizedMatrix, *, dtype: torch.dtype
    ) -> Tensor:
        if isinstance(weight, QuantizedMatrix):
            return embedding_matrix(input_ids, weight, dtype=dtype)
        return functional.embedding(input_ids, weight)

    def linear(self, inputs: Tensor, weight: Tensor | QuantizedMatrix) -> Tensor:
        if isinstance(weight, QuantizedMatrix):
            return linear_matrix(inputs, weight)
        return reference_linear(inputs, weight)

    def full_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        query_ordinals: Tensor,
        key_ordinals: Tensor,
    ) -> Tensor:
        return reference_full_attention(
            query,
            key,
            value,
            query_ordinals=query_ordinals,
            key_ordinals=key_ordinals,
            output_dtype=query.dtype,
        )


_DIRECT_REFERENCE_OPERATORS = _DirectReferenceOperators()


def fuse_qwen3_5_decoder_input(
    input_ids: Tensor,
    text_embeddings: Tensor,
    feature_spans: tuple[FeatureSpan, ...],
    *,
    image_token_id: int,
    video_token_id: int,
) -> Tensor:
    """Replace exactly the authorized Qwen media placeholder rows.

    ``FeatureSpan.token_offset`` is segment-relative. Request, job, feature-row,
    consumer-site, and consumer-layer authority has already been narrowed by the
    core ``FeatureReader``; this model boundary proves that the remaining rows
    cover every and only Qwen image/video placeholder in the current segment.
    """

    if (
        input_ids.dtype is not torch.int64
        or input_ids.ndim != 2
        or text_embeddings.dtype is not torch.bfloat16
        or text_embeddings.ndim != 3
        or tuple(text_embeddings.shape[:2]) != tuple(input_ids.shape)
        or input_ids.device != text_embeddings.device
    ):
        raise Qwen3_5FeatureFusionError(
            "decoder fusion requires int64 [batch, tokens] ids and BF16 "
            "[batch, tokens, hidden] embeddings on one device"
        )
    if type(feature_spans) is not tuple or any(
        not isinstance(span, FeatureSpan) for span in feature_spans
    ):
        raise TypeError("decoder features must be a tuple of authorized FeatureSpan values")
    if (
        type(image_token_id) is not int
        or type(video_token_id) is not int
        or min(image_token_id, video_token_id) < 0
        or image_token_id == video_token_id
    ):
        raise ValueError("Qwen image and video token identities must be distinct non-negative ints")

    media_mask = (input_ids == image_token_id) | (input_ids == video_token_id)
    if not feature_spans and not bool(media_mask.any().item()):
        return text_embeddings
    if input_ids.shape[0] != 1:
        raise Qwen3_5FeatureFusionError(
            "segment feature spans require one request row at the model boundary"
        )

    token_count = int(input_ids.shape[1])
    hidden_size = int(text_embeddings.shape[2])
    covered = torch.zeros(token_count, dtype=torch.bool, device=input_ids.device)
    validated: list[tuple[int, int, Tensor]] = []
    for span_index, span in enumerate(feature_spans):
        rows = span.rows
        start = span.token_offset
        end = start + span.token_count
        if end > token_count:
            raise Qwen3_5FeatureFusionError(
                f"decoder feature span {span_index} exceeds the current segment"
            )
        if (
            rows.dtype is not torch.bfloat16
            or rows.device != text_embeddings.device
            or rows.layout is not torch.strided
            or not rows.is_contiguous()
            or rows.ndim != 2
            or int(rows.shape[1]) != hidden_size
            or rows.requires_grad
        ):
            raise Qwen3_5FeatureFusionError(
                f"decoder feature span {span_index} differs from the BF16 row ABI"
            )
        if not bool(torch.isfinite(rows).all().item()):
            raise Qwen3_5FeatureFusionError(
                f"decoder feature span {span_index} contains a non-finite value"
            )
        if bool(covered[start:end].any().item()):
            raise Qwen3_5FeatureFusionError("decoder feature spans overlap")
        covered[start:end] = True
        validated.append((start, end, rows))

    if not torch.equal(covered, media_mask[0]):
        raise Qwen3_5FeatureFusionError(
            "authorized feature rows must cover every and only Qwen media placeholder"
        )

    fused = text_embeddings.clone(memory_format=torch.contiguous_format)
    with torch.no_grad():
        for start, end, rows in validated:
            fused[0, start:end].copy_(rows)
    return fused


@dataclass(frozen=True, slots=True)
class Qwen3_5LayerCache:
    """State owned by exactly one decoder layer."""

    convolution: Tensor | None = None
    recurrent: Tensor | None = None
    key: Tensor | None = None
    value: Tensor | None = None


@dataclass(frozen=True, slots=True)
class Qwen3_5ReferenceCache:
    """Immutable cache handle; a step returns new tensors instead of mutating it."""

    sequence_length: int
    batch_size: int
    layers: tuple[Qwen3_5LayerCache, ...]

    @property
    def kv_dtypes(self) -> tuple[torch.dtype, ...]:
        return tuple(
            sorted(
                {
                    tensor.dtype
                    for layer in self.layers
                    for tensor in (layer.key, layer.value)
                    if tensor is not None
                },
                key=str,
            )
        )


@dataclass(frozen=True, slots=True)
class Qwen3_5ReferenceStep:
    logits: Tensor
    hidden_states: Tensor
    cache: Qwen3_5ReferenceCache


@dataclass(frozen=True, slots=True)
class Qwen3_5MTPReferenceCache:
    """Private full-attention state for the single checkpoint MTP layer."""

    sequence_length: int
    batch_size: int
    layer: Qwen3_5LayerCache


@dataclass(frozen=True, slots=True)
class Qwen3_5MTPReferenceStep:
    logits: Tensor
    hidden_states: Tensor
    cache: Qwen3_5MTPReferenceCache


class Qwen3_5ReferenceWeights:
    """Read-only CUDA tensors retaining their checkpoint provenance."""

    __slots__ = ("_tensors", "device", "inventory_digest", "semantic_plan_digest")

    def __init__(
        self,
        *,
        tensors: dict[str, Tensor],
        device: torch.device,
        inventory: CheckpointInventory,
        plan: Qwen3_5WeightPlan,
    ) -> None:
        self._tensors = MappingProxyType(tensors)
        self.device = device
        self.inventory_digest = inventory.inventory_digest
        self.semantic_plan_digest = plan.semantic_plan_digest

    def __getitem__(self, name: str) -> Tensor:
        try:
            return self._tensors[name]
        except KeyError as error:
            raise Qwen3_5ReferenceLoadError(
                f"eager graph requested unloaded weight {name!r}"
            ) from error

    @property
    def tensor_count(self) -> int:
        return len(self._tensors)

    def named_tensors(self) -> tuple[tuple[str, Tensor], ...]:
        return tuple((name, self._tensors[name]) for name in sorted(self._tensors))


def load_qwen3_5_reference(
    model_root: str | Path,
    *,
    device: str | torch.device = "cuda:0",
    num_speculative_tokens: int = 0,
) -> Qwen3_5ReferenceModel:
    """Attest and stream the dense language graph to one device."""

    root = Path(model_root).resolve(strict=True)
    config_path = root / "config.json"
    try:
        config_document = config_path.read_bytes()
    except OSError as error:
        raise Qwen3_5ReferenceLoadError("qwen3_5 config.json is unavailable") from error
    config = parse_qwen3_5_config_json(config_document)
    inventory = inspect_checkpoint(root)
    config_digest = f"b2s256:{hashlib.blake2s(config_document).hexdigest()}"
    if config_digest != inventory.config_file.content_digest:
        raise Qwen3_5ReferenceLoadError("config.json changed during checkpoint admission")
    from cubic_serve.models.qwen3_5.weights import plan_qwen3_5_checkpoint_weights

    plan = plan_qwen3_5_checkpoint_weights(config, inventory)
    return materialize_qwen3_5_reference(
        config=config,
        inventory=inventory,
        plan=plan,
        device=device,
        num_speculative_tokens=num_speculative_tokens,
    )


def materialize_qwen3_5_reference(
    *,
    config: Qwen3_5Config,
    inventory: CheckpointInventory,
    plan: Qwen3_5WeightPlan,
    device: str | torch.device,
    num_speculative_tokens: int = 0,
) -> Qwen3_5ReferenceModel:
    """Load a graph from checkpoint evidence already issued by core admission."""

    if config.identity.model_type != plan.model_type:
        raise Qwen3_5ReferenceLoadError("weight plan differs from the qwen3_5 graph")
    if plan.inventory_digest != inventory.inventory_digest:
        raise Qwen3_5ReferenceLoadError("weight plan differs from the attested checkpoint")
    if type(num_speculative_tokens) is not int or not (
        0 <= num_speculative_tokens <= QWEN3_5_MTP_MAX_PROPOSAL_TOKENS
    ):
        raise Qwen3_5ReferenceLoadError(
            "num_speculative_tokens exceeds the Qwen3.5 MTP provider maximum"
        )
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise Qwen3_5ReferenceLoadError("CUDA was requested but is unavailable")
    weights = _load_text_tensors(
        inventory.model_root,
        target,
        inventory,
        plan,
        include_mtp=num_speculative_tokens > 0,
    )
    return Qwen3_5ReferenceModel(config=config, weights=weights)


def _load_text_tensors(
    root: Path,
    device: torch.device,
    inventory: CheckpointInventory,
    plan: Qwen3_5WeightPlan,
    *,
    include_mtp: bool,
) -> Qwen3_5ReferenceWeights:
    from cubic_serve.models.qwen3_5.weights import (
        qwen3_5_mtp_weight_policy,
        qwen3_5_text_weight_policy,
    )

    policy = qwen3_5_mtp_weight_policy(plan) if include_mtp else qwen3_5_text_weight_policy(plan)
    required = policy.expected_names
    required_set = set(required)
    records = {tensor.name: tensor for tensor in inventory.tensors if tensor.name in required_set}
    if set(records) != required_set:
        raise Qwen3_5ReferenceLoadError("attested inventory does not close the text weight set")
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name in required:
        by_shard[records[name].shard_path].append(name)

    loaded: dict[str, Tensor] = {}
    for shard_path, names in sorted(by_shard.items()):
        with safe_open(root / shard_path, framework="pt", device="cpu") as reader:
            for name in names:
                source = reader.get_tensor(name)
                record = records[name]
                if source.dtype is not torch.bfloat16 or tuple(source.shape) != record.shape:
                    raise Qwen3_5ReferenceLoadError(
                        f"loaded tensor {name!r} differs from its BF16 header evidence"
                    )
                loaded[name] = source.to(device=device, non_blocking=False)
    if tuple(sorted(loaded)) != required:
        raise Qwen3_5ReferenceLoadError("text weight loading ended with an open inventory")
    return Qwen3_5ReferenceWeights(
        tensors=loaded,
        device=device,
        inventory=inventory,
        plan=plan,
    )


class Qwen3_5ReferenceModel:
    """Dense Qwen3.8-27B text graph evaluated only through eager PyTorch ops."""

    __slots__ = ("compute_dtype", "config", "default_operators", "recurrent_state_dtype", "weights")
    _model_type = "qwen3_5"

    def __init__(
        self,
        *,
        config: Qwen3_5Config,
        weights: Qwen3_5WeightAccess,
        recurrent_state_dtype: torch.dtype | None = None,
        operators: Qwen3_5ModelOperators | None = None,
    ) -> None:
        if config.identity.model_type != self._model_type:
            raise ValueError(f"{type(self).__name__} requires model_type {self._model_type}")
        self.compute_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[
            config.graph.text.dtype
        ]
        state_dtype = recurrent_state_dtype or torch.float32
        if state_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("recurrent state dtype must be torch.bfloat16 or torch.float32")
        self.config = config
        self.recurrent_state_dtype = state_dtype
        self.weights = weights
        self.default_operators = operators or _DIRECT_REFERENCE_OPERATORS

    def empty_cache(self, batch_size: int = 1) -> Qwen3_5ReferenceCache:
        if batch_size <= 0:
            raise ValueError("cache batch size must be positive")
        return self._empty_cache(batch_size)

    def empty_mtp_cache(self, batch_size: int = 1) -> Qwen3_5MTPReferenceCache:
        if batch_size <= 0:
            raise ValueError("MTP cache batch size must be positive")
        return Qwen3_5MTPReferenceCache(
            sequence_length=0,
            batch_size=batch_size,
            layer=Qwen3_5LayerCache(),
        )

    def embed_input_ids(
        self,
        input_ids: Tensor,
        *,
        decoder_input_features: tuple[FeatureSpan, ...] | None = None,
    ) -> Tensor:
        """Apply the canonical decoder embedding boundary.

        Keeping this boundary reusable lets the distributed eager reference
        reconstruct one full operand at a time without duplicating model math.
        """

        embedding = self.weights["model.language_model.embed_tokens.weight"]
        hidden = self.default_operators.embedding(input_ids, embedding, dtype=self.compute_dtype)
        if decoder_input_features is not None:
            modality_tokens = self.config.graph.modality_tokens
            hidden = fuse_qwen3_5_decoder_input(
                input_ids,
                hidden,
                decoder_input_features,
                image_token_id=modality_tokens.image,
                video_token_id=modality_tokens.video,
            )
        return hidden

    def forward_layer(
        self,
        hidden: Tensor,
        *,
        layer_index: int,
        previous: Qwen3_5LayerCache,
        cosine: Tensor,
        sine: Tensor,
        previous_length: int,
        operators: Qwen3_5OperatorCalls | None = None,
        observer: Callable[[str, Tensor], None] | None = None,
    ) -> tuple[Tensor, Qwen3_5LayerCache]:
        """Execute exactly one canonical decoder layer."""

        text = self.config.graph.text
        if not 0 <= layer_index < text.layer_count:
            raise ValueError("decoder layer index lies outside qwen3_5")
        active_operators = operators or self.default_operators
        prefix = f"model.language_model.layers.{layer_index}"
        residual = hidden
        if observer is not None:
            observer("layer_input", hidden)
        hidden = zero_centered_rms_norm(
            hidden,
            self.weights[f"{prefix}.input_layernorm.weight"],
            epsilon=text.rms_norm_epsilon,
            output_dtype=self.compute_dtype,
        )
        if observer is not None:
            observer("input_norm", hidden)
        if text.layer_schedule[layer_index] is AttentionKind.GATED_DELTA_NET:
            mixed, layer_cache = self._gated_delta(
                hidden,
                prefix=prefix,
                previous=previous,
                operators=active_operators,
            )
        else:
            mixed, layer_cache = self._full_attention(
                hidden,
                prefix=prefix,
                previous=previous,
                cosine=cosine,
                sine=sine,
                previous_length=previous_length,
                operators=active_operators,
            )
        if observer is not None:
            observer("attention_mixed", mixed)
        hidden = residual + mixed
        if observer is not None:
            observer("attention", hidden)
        residual = hidden
        hidden = zero_centered_rms_norm(
            hidden,
            self.weights[f"{prefix}.post_attention_layernorm.weight"],
            epsilon=text.rms_norm_epsilon,
            output_dtype=self.compute_dtype,
        )
        if observer is not None:
            observer("mlp_input", hidden)
        mlp = self._mlp(
            hidden,
            prefix=f"{prefix}.mlp",
            operators=active_operators,
        )
        if observer is not None:
            observer("mlp_output", mlp)
        hidden = residual + mlp
        if observer is not None:
            observer("mlp", hidden)
        return hidden, layer_cache

    def compute_logits(
        self,
        hidden: Tensor,
        *,
        operators: Qwen3_5OperatorCalls | None = None,
    ) -> Tensor:
        """Apply the canonical final norm and full vocabulary projection."""

        active_operators = operators or self.default_operators
        hidden = self.normalize_hidden(hidden)
        return active_operators.linear(hidden[:, -1:], self.weights["lm_head.weight"])

    def normalize_hidden(self, hidden: Tensor) -> Tensor:
        """Expose the target-normalized state consumed by the MTP extension."""

        return zero_centered_rms_norm(
            hidden,
            self.weights["model.language_model.norm.weight"],
            epsilon=self.config.graph.text.rms_norm_epsilon,
            output_dtype=self.compute_dtype,
        )

    def project_normalized_logits(
        self,
        normalized_hidden: Tensor,
        *,
        operators: Qwen3_5OperatorCalls | None = None,
    ) -> Tensor:
        """Project every target-normalized row for speculative verification."""

        if (
            normalized_hidden.dtype is not self.compute_dtype
            or normalized_hidden.ndim != 3
            or normalized_hidden.shape[-1] != self.config.graph.text.hidden_size
            or normalized_hidden.device != self.weights.device
        ):
            raise ValueError("normalized hidden states must be BF16 [batch, tokens, hidden]")
        active_operators = operators or self.default_operators
        return active_operators.linear(normalized_hidden, self.weights["lm_head.weight"])

    def mtp_forward(
        self,
        input_ids: Tensor,
        target_hidden_states: Tensor,
        *,
        position_ids: Tensor,
        cache: Qwen3_5MTPReferenceCache | None = None,
        operators: Qwen3_5OperatorCalls | None = None,
        observer: Callable[[str, Tensor], None] | None = None,
    ) -> Qwen3_5MTPReferenceStep:
        """Run the checkpoint's one MTP layer in eager reference mode.

        The caller owns iteration semantics and may reuse this same layer for
        proposal depths one through three.  ``target_hidden_states`` are already
        normalized outputs of the target model for the state preceding each
        corresponding ``input_ids`` token.
        """

        text = self.config.graph.text
        if input_ids.dtype is not torch.int64 or input_ids.ndim != 2 or input_ids.numel() == 0:
            raise TypeError("MTP input_ids must be a non-empty rank-two torch.int64 tensor")
        if input_ids.device != self.weights.device:
            raise ValueError("MTP input_ids and model weights must be on the same device")
        if bool(((input_ids < 0) | (input_ids >= text.vocabulary_size)).any().item()):
            raise ValueError("MTP input_ids contain an out-of-vocabulary token")
        batch_size, token_count = input_ids.shape
        expected_hidden_shape = (batch_size, token_count, text.hidden_size)
        if (
            target_hidden_states.dtype is not self.compute_dtype
            or tuple(target_hidden_states.shape) != expected_hidden_shape
            or target_hidden_states.device != input_ids.device
        ):
            raise ValueError("MTP target hidden states must be BF16 [batch, tokens, hidden]")
        if cache is None:
            cache = self.empty_mtp_cache(batch_size)
        self._validate_mtp_cache(cache, batch_size)
        active_operators = operators or self.default_operators

        embeddings = self.embed_input_ids(input_ids)
        embeddings = zero_centered_rms_norm(
            embeddings,
            self.weights["mtp.pre_fc_norm_embedding.weight"],
            epsilon=text.rms_norm_epsilon,
            output_dtype=self.compute_dtype,
        )
        target_hidden_states = zero_centered_rms_norm(
            target_hidden_states,
            self.weights["mtp.pre_fc_norm_hidden.weight"],
            epsilon=text.rms_norm_epsilon,
            output_dtype=self.compute_dtype,
        )
        fc_input = torch.cat((embeddings, target_hidden_states), dim=-1)
        if observer is not None:
            observer("fc_input", fc_input)
        hidden = active_operators.linear(
            fc_input,
            self.weights["mtp.fc.weight"],
        )
        if observer is not None:
            observer("fc", hidden)

        if (
            position_ids.dtype is not torch.int64
            or tuple(position_ids.shape) != (3, batch_size, token_count)
            or position_ids.device != input_ids.device
        ):
            raise ValueError(
                "MTP position_ids must be int64 [3, batch, tokens] on the model device"
            )
        if bool(((position_ids < 0) | (position_ids >= text.max_position_embeddings)).any().item()):
            raise ValueError("MTP position_ids exceed the configured position domain")
        rotary_dim = int(text.full_attention.head_dim * text.mrope.partial_rotary_factor)
        cosine, sine = interleaved_mrope_cos_sin(
            position_ids,
            rotary_dim=rotary_dim,
            mrope_section=text.mrope.section,
            rope_theta=text.mrope.rope_theta,
            output_dtype=self.compute_dtype,
        )

        prefix = "mtp.layers.0"
        residual = hidden
        hidden = zero_centered_rms_norm(
            hidden,
            self.weights[f"{prefix}.input_layernorm.weight"],
            epsilon=text.rms_norm_epsilon,
            output_dtype=self.compute_dtype,
        )
        if observer is not None:
            observer("attention_input", hidden)
        mixed, layer_cache = self._full_attention(
            hidden,
            prefix=prefix,
            previous=cache.layer,
            cosine=cosine,
            sine=sine,
            previous_length=cache.sequence_length,
            operators=active_operators,
            observer=observer,
        )
        hidden = residual + mixed
        residual = hidden
        hidden = zero_centered_rms_norm(
            hidden,
            self.weights[f"{prefix}.post_attention_layernorm.weight"],
            epsilon=text.rms_norm_epsilon,
            output_dtype=self.compute_dtype,
        )
        if observer is not None:
            observer("mlp_input", hidden)
        hidden = residual + self._mlp(
            hidden,
            prefix=f"{prefix}.mlp",
            operators=active_operators,
        )
        hidden = zero_centered_rms_norm(
            hidden,
            self.weights["mtp.norm.weight"],
            epsilon=text.rms_norm_epsilon,
            output_dtype=self.compute_dtype,
        )
        if observer is not None:
            observer("mtp_output", hidden)
        logits = active_operators.linear(hidden[:, -1:], self.weights["lm_head.weight"])
        return Qwen3_5MTPReferenceStep(
            logits=logits,
            hidden_states=hidden,
            cache=Qwen3_5MTPReferenceCache(
                sequence_length=cache.sequence_length + token_count,
                batch_size=batch_size,
                layer=layer_cache,
            ),
        )

    def forward(
        self,
        input_ids: Tensor,
        *,
        cache: Qwen3_5ReferenceCache | None = None,
        position_ids: Tensor | None = None,
        operators: Qwen3_5OperatorCalls | None = None,
        decoder_input_features: tuple[FeatureSpan, ...] | None = None,
    ) -> Qwen3_5ReferenceStep:
        text = self.config.graph.text
        if input_ids.dtype is not torch.int64 or input_ids.ndim != 2 or input_ids.numel() == 0:
            raise TypeError("input_ids must be a non-empty rank-two torch.int64 tensor")
        if input_ids.device != self.weights.device:
            raise ValueError("input_ids and model weights must be on the same device")
        if bool(((input_ids < 0) | (input_ids >= text.vocabulary_size)).any().item()):
            raise ValueError("input_ids contain an out-of-vocabulary token")
        batch_size, token_count = input_ids.shape
        if cache is None:
            cache = self._empty_cache(batch_size)
        self._validate_cache(cache, batch_size)
        active_operators = operators or self.default_operators

        hidden = self.embed_input_ids(
            input_ids,
            decoder_input_features=decoder_input_features,
        )
        if position_ids is None:
            positions = torch.arange(
                cache.sequence_length,
                cache.sequence_length + token_count,
                dtype=torch.int64,
                device=input_ids.device,
            )
            position_ids = positions.view(1, 1, token_count).expand(3, batch_size, token_count)
        elif (
            position_ids.dtype is not torch.int64
            or tuple(position_ids.shape) != (3, batch_size, token_count)
            or position_ids.device != input_ids.device
        ):
            raise ValueError("position_ids must be int64 [3, batch, tokens] on the model device")
        rotary_dim = int(text.full_attention.head_dim * text.mrope.partial_rotary_factor)
        cosine, sine = interleaved_mrope_cos_sin(
            position_ids,
            rotary_dim=rotary_dim,
            mrope_section=text.mrope.section,
            rope_theta=text.mrope.rope_theta,
            output_dtype=self.compute_dtype,
        )

        next_layers: list[Qwen3_5LayerCache] = []
        for layer_index in range(text.layer_count):
            hidden, layer_cache = self.forward_layer(
                hidden,
                layer_index=layer_index,
                previous=cache.layers[layer_index],
                cosine=cosine,
                sine=sine,
                previous_length=cache.sequence_length,
                operators=active_operators,
            )
            next_layers.append(layer_cache)

        normalized_hidden = self.normalize_hidden(hidden)
        logits = self.project_normalized_logits(
            normalized_hidden[:, -1:],
            operators=active_operators,
        )
        return Qwen3_5ReferenceStep(
            logits=logits,
            hidden_states=normalized_hidden,
            cache=Qwen3_5ReferenceCache(
                sequence_length=cache.sequence_length + token_count,
                batch_size=batch_size,
                layers=tuple(next_layers),
            ),
        )

    def _empty_cache(self, batch_size: int) -> Qwen3_5ReferenceCache:
        text = self.config.graph.text
        gdn = text.gated_delta_net
        layers: list[Qwen3_5LayerCache] = []
        for attention_kind in text.layer_schedule:
            if attention_kind is AttentionKind.GATED_DELTA_NET:
                convolution = torch.zeros(
                    batch_size,
                    gdn.key_heads * gdn.key_head_dim * 2 + gdn.value_heads * gdn.value_head_dim,
                    gdn.convolution_kernel_size - 1,
                    dtype=self.compute_dtype,
                    device=self.weights.device,
                )
                recurrent = torch.zeros(
                    batch_size,
                    gdn.value_heads,
                    gdn.value_head_dim,
                    gdn.key_head_dim,
                    dtype=self.recurrent_state_dtype,
                    device=self.weights.device,
                )
                layers.append(Qwen3_5LayerCache(convolution=convolution, recurrent=recurrent))
            else:
                layers.append(Qwen3_5LayerCache())
        return Qwen3_5ReferenceCache(0, batch_size, tuple(layers))

    def _validate_cache(self, cache: Qwen3_5ReferenceCache, batch_size: int) -> None:
        text = self.config.graph.text
        if not isinstance(cache, Qwen3_5ReferenceCache):
            raise TypeError("cache must be a Qwen3_5ReferenceCache")
        if cache.sequence_length < 0 or cache.batch_size != batch_size:
            raise ValueError("cache sequence or batch identity differs from this step")
        if len(cache.layers) != text.layer_count:
            raise ValueError("cache layer count differs from the model")
        gdn = text.gated_delta_net
        attention = text.full_attention
        for layer_index, (kind, layer) in enumerate(
            zip(text.layer_schedule, cache.layers, strict=True)
        ):
            if not isinstance(layer, Qwen3_5LayerCache):
                raise TypeError(f"cache layer {layer_index} has an invalid state image")
            if kind is AttentionKind.GATED_DELTA_NET:
                if layer.key is not None or layer.value is not None:
                    raise ValueError(f"linear-attention layer {layer_index} contains KV state")
                _require_cache_tensor(
                    layer.convolution,
                    label=f"layer {layer_index} convolution",
                    dtype=self.compute_dtype,
                    shape=(
                        batch_size,
                        gdn.key_heads * gdn.key_head_dim * 2 + gdn.value_heads * gdn.value_head_dim,
                        gdn.convolution_kernel_size - 1,
                    ),
                    device=self.weights.device,
                )
                _require_cache_tensor(
                    layer.recurrent,
                    label=f"layer {layer_index} recurrent",
                    dtype=self.recurrent_state_dtype,
                    shape=(
                        batch_size,
                        gdn.value_heads,
                        gdn.value_head_dim,
                        gdn.key_head_dim,
                    ),
                    device=self.weights.device,
                )
                continue
            if layer.convolution is not None or layer.recurrent is not None:
                raise ValueError(f"full-attention layer {layer_index} contains recurrent state")
            if cache.sequence_length == 0:
                if layer.key is not None or layer.value is not None:
                    raise ValueError(f"empty attention layer {layer_index} contains KV state")
                continue
            expected_shape = (
                batch_size,
                attention.key_value_heads,
                cache.sequence_length,
                attention.head_dim,
            )
            _require_cache_tensor(
                layer.key,
                label=f"layer {layer_index} key",
                dtype=self.compute_dtype,
                shape=expected_shape,
                device=self.weights.device,
            )
            _require_cache_tensor(
                layer.value,
                label=f"layer {layer_index} value",
                dtype=self.compute_dtype,
                shape=expected_shape,
                device=self.weights.device,
            )

    def _validate_mtp_cache(self, cache: Qwen3_5MTPReferenceCache, batch_size: int) -> None:
        if not isinstance(cache, Qwen3_5MTPReferenceCache):
            raise TypeError("MTP cache must be a Qwen3_5MTPReferenceCache")
        if cache.sequence_length < 0 or cache.batch_size != batch_size:
            raise ValueError("MTP cache sequence or batch identity differs from this step")
        if cache.layer.convolution is not None or cache.layer.recurrent is not None:
            raise ValueError("MTP full-attention cache contains recurrent state")
        if cache.sequence_length == 0:
            if cache.layer.key is not None or cache.layer.value is not None:
                raise ValueError("empty MTP cache contains KV state")
            return
        attention = self.config.graph.text.full_attention
        expected_shape = (
            batch_size,
            attention.key_value_heads,
            cache.sequence_length,
            attention.head_dim,
        )
        _require_cache_tensor(
            cache.layer.key,
            label="MTP key",
            dtype=self.compute_dtype,
            shape=expected_shape,
            device=self.weights.device,
        )
        _require_cache_tensor(
            cache.layer.value,
            label="MTP value",
            dtype=self.compute_dtype,
            shape=expected_shape,
            device=self.weights.device,
        )

    def _gated_delta(
        self,
        hidden: Tensor,
        *,
        prefix: str,
        previous: Qwen3_5LayerCache,
        operators: Qwen3_5OperatorCalls,
    ) -> tuple[Tensor, Qwen3_5LayerCache]:
        text = self.config.graph.text
        gdn = text.gated_delta_net
        if previous.convolution is None or previous.recurrent is None:
            raise ValueError("gated-delta layer is missing its explicit state")
        op = f"{prefix}.linear_attn"
        mixed = operators.linear(hidden, self.weights[f"{op}.in_proj_qkv.weight"])
        mixed, convolution = depthwise_causal_conv1d(
            mixed,
            self.weights[f"{op}.conv1d.weight"].squeeze(1),
            previous.convolution,
            activation="silu",
            output_dtype=self.compute_dtype,
        )
        key_width = gdn.key_heads * gdn.key_head_dim
        value_width = gdn.value_heads * gdn.value_head_dim
        query, key, value = torch.split(mixed, [key_width, key_width, value_width], dim=-1)
        batch, tokens, _ = hidden.shape
        query = query.view(batch, tokens, gdn.key_heads, gdn.key_head_dim)
        key = key.view(batch, tokens, gdn.key_heads, gdn.key_head_dim)
        value = value.view(batch, tokens, gdn.value_heads, gdn.value_head_dim)
        decay_input = operators.linear(hidden, self.weights[f"{op}.in_proj_a.weight"])
        beta_input = operators.linear(hidden, self.weights[f"{op}.in_proj_b.weight"])
        core, recurrent = gated_delta_net_recurrent(
            query,
            key,
            value,
            decay_input,
            beta_input,
            self.weights[f"{op}.A_log"],
            self.weights[f"{op}.dt_bias"],
            previous.recurrent,
            l2_epsilon=1e-6,
            output_dtype=self.compute_dtype,
        )
        gate = operators.linear(hidden, self.weights[f"{op}.in_proj_z.weight"])
        gate = gate.view(batch, tokens, gdn.value_heads, gdn.value_head_dim)
        core = gated_rms_norm(
            core,
            gate,
            self.weights[f"{op}.norm.weight"],
            epsilon=text.rms_norm_epsilon,
            output_dtype=self.compute_dtype,
        )
        output = operators.linear(core.flatten(2), self.weights[f"{op}.out_proj.weight"])
        return output, Qwen3_5LayerCache(convolution=convolution, recurrent=recurrent)

    def _full_attention(
        self,
        hidden: Tensor,
        *,
        prefix: str,
        previous: Qwen3_5LayerCache,
        cosine: Tensor,
        sine: Tensor,
        previous_length: int,
        operators: Qwen3_5OperatorCalls,
        observer: Callable[[str, Tensor], None] | None = None,
    ) -> tuple[Tensor, Qwen3_5LayerCache]:
        text = self.config.graph.text
        attention = text.full_attention
        op = f"{prefix}.self_attn"
        batch, tokens, _ = hidden.shape
        projected = operators.linear(hidden, self.weights[f"{op}.q_proj.weight"])
        projected = projected.view(
            batch,
            tokens,
            attention.query_heads,
            attention.head_dim * 2,
        )
        query, gate = projected.chunk(2, dim=-1)
        key = operators.linear(hidden, self.weights[f"{op}.k_proj.weight"])
        value = operators.linear(hidden, self.weights[f"{op}.v_proj.weight"])
        if observer is not None:
            observer("q_projection", projected)
            observer("k_projection", key)
            observer("v_projection", value)
        key = key.view(batch, tokens, attention.key_value_heads, attention.head_dim)
        value = value.view(batch, tokens, attention.key_value_heads, attention.head_dim)
        query = zero_centered_rms_norm(
            query,
            self.weights[f"{op}.q_norm.weight"],
            epsilon=text.rms_norm_epsilon,
            output_dtype=self.compute_dtype,
        ).transpose(1, 2)
        key = zero_centered_rms_norm(
            key,
            self.weights[f"{op}.k_norm.weight"],
            epsilon=text.rms_norm_epsilon,
            output_dtype=self.compute_dtype,
        ).transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = apply_partial_rotary(
            query,
            key,
            cosine,
            sine,
            output_dtype=self.compute_dtype,
        )
        if previous.key is not None or previous.value is not None:
            if previous.key is None or previous.value is None:
                raise ValueError("attention cache must contain both key and value")
            key = torch.cat((previous.key, key), dim=2)
            value = torch.cat((previous.value, value), dim=2)

        query_ordinals = (
            (previous_length + torch.arange(tokens, dtype=torch.int64, device=hidden.device))
            .view(1, tokens)
            .expand(batch, tokens)
        )
        key_ordinals = (
            torch.arange(key.shape[2], dtype=torch.int64, device=hidden.device)
            .view(1, key.shape[2])
            .expand(batch, key.shape[2])
        )
        output = (
            operators.full_attention(
                query,
                key,
                value,
                query_ordinals=query_ordinals,
                key_ordinals=key_ordinals,
            )
            .transpose(1, 2)
            .contiguous()
        )
        output = output.view(batch, tokens, attention.query_heads * attention.head_dim)
        gate = gate.reshape_as(output)
        output = sigmoid_output_gate(output, gate, output_dtype=self.compute_dtype)
        output = operators.linear(output, self.weights[f"{op}.o_proj.weight"])
        return output, Qwen3_5LayerCache(key=key, value=value)

    def _mlp(
        self,
        hidden: Tensor,
        *,
        prefix: str,
        operators: Qwen3_5OperatorCalls,
    ) -> Tensor:
        gate = activation_silu(operators.linear(hidden, self.weights[f"{prefix}.gate_proj.weight"]))
        up = operators.linear(hidden, self.weights[f"{prefix}.up_proj.weight"])
        return operators.linear(
            gate * up,
            self.weights[f"{prefix}.down_proj.weight"],
        )


def _require_cache_tensor(
    tensor: Tensor | None,
    *,
    label: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if tensor is None:
        raise ValueError(f"{label} is missing")
    if tensor.layout is not torch.strided:
        raise TypeError(f"{label} must use strided storage")
    if tensor.dtype is not dtype or tuple(tensor.shape) != shape or tensor.device != device:
        raise ValueError(f"{label} dtype, shape, or device differs from its state contract")


__all__ = (
    "Qwen3_5FeatureFusionError",
    "Qwen3_5LayerCache",
    "Qwen3_5MTPReferenceCache",
    "Qwen3_5MTPReferenceStep",
    "Qwen3_5ModelOperators",
    "Qwen3_5OperatorCalls",
    "Qwen3_5ReferenceCache",
    "Qwen3_5ReferenceLoadError",
    "Qwen3_5ReferenceModel",
    "Qwen3_5ReferenceStep",
    "Qwen3_5ReferenceWeights",
    "Qwen3_5WeightAccess",
    "fuse_qwen3_5_decoder_input",
    "load_qwen3_5_reference",
    "materialize_qwen3_5_reference",
)


_BLOCK_SIZE = 16


def _state_id(layer_index: int, kind: str) -> str:
    return f"qwen3_5.layer.{layer_index:04d}.gdn.{kind}"


def _linear_signature(phase: BatchPhase) -> OpSignature:
    return OpSignature(
        semantic_op="linear",
        phase=phase,
        inputs=(
            TensorSpec("inputs", ScalarDType.BF16, ("batch", "m", "k")),
            TensorSpec("weight", ScalarDType.BF16, ("n", "k")),
        ),
        outputs=(TensorSpec("result", ScalarDType.BF16, ("batch", "m", "n")),),
        accumulator_dtype=ScalarDType.FP32,
        layout="bmk_nk",
        shape_items=(),
        cache_form=None,
        state_form=None,
        topology_id="single",
    )


def _attention_signature(phase: BatchPhase) -> OpSignature:
    return OpSignature(
        semantic_op="attention.full",
        phase=phase,
        inputs=(
            TensorSpec(
                "query",
                ScalarDType.BF16,
                ("batch", "query_heads", "m", "head_dim"),
            ),
            TensorSpec(
                "key",
                ScalarDType.BF16,
                ("batch", "kv_heads", "context", "head_dim"),
            ),
            TensorSpec(
                "value",
                ScalarDType.BF16,
                ("batch", "kv_heads", "context", "head_dim"),
            ),
            TensorSpec("query_ordinals", ScalarDType.INT64, ("batch", "m")),
            TensorSpec("key_ordinals", ScalarDType.INT64, ("batch", "context")),
        ),
        outputs=(
            TensorSpec(
                "output",
                ScalarDType.BF16,
                ("batch", "query_heads", "m", "head_dim"),
            ),
        ),
        accumulator_dtype=ScalarDType.FP32,
        layout="bhsd",
        shape_items=(),
        cache_form="paged_bf16_view",
        state_form=None,
        topology_id="single",
    )


def _required_signatures() -> tuple[OpSignature, ...]:
    return tuple(
        signature
        for phase in (BatchPhase.PREFILL, BatchPhase.DECODE)
        for signature in (_linear_signature(phase), _attention_signature(phase))
    )


class _SegmentOperators(Qwen3_5OperatorCalls):
    def __init__(self, ops: SemanticOpExecutor, phase: BatchPhase) -> None:
        self._ops = ops
        self._linear = _linear_signature(phase)
        self._attention = _attention_signature(phase)

    def linear(self, inputs: Tensor, weight: Tensor) -> Tensor:
        return cast(Tensor, self._ops.call(self._linear, inputs, weight))

    def full_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        query_ordinals: Tensor,
        key_ordinals: Tensor,
    ) -> Tensor:
        return cast(
            Tensor,
            self._ops.call(
                self._attention,
                query,
                key,
                value,
                query_ordinals=query_ordinals,
                key_ordinals=key_ordinals,
            ),
        )


def _decoder_input_features(
    segment: ModelSegment,
    features: FeatureReader,
    config: Qwen3_5Config,
) -> tuple[FeatureSpan, ...] | None:
    """Narrow one segment capability to Qwen's sole supported fusion site."""

    spans = features.spans("decoder.input", 0)
    for layer_index in range(1, config.graph.text.layer_count):
        if features.spans("decoder.input", layer_index):
            raise Qwen3_5FeatureFusionError(
                "qwen3_5 base vision rows may only be consumed at decoder input layer 0"
            )
    if segment.phase is BatchPhase.DECODE:
        if spans:
            raise Qwen3_5FeatureFusionError("decode cannot consume prompt media features")
        return None

    tokens = config.graph.modality_tokens
    contains_media_placeholder = any(
        token_id in (tokens.image, tokens.video) for token_id in segment.token_ids
    )
    return spans if spans or contains_media_placeholder else None


class Qwen3_5Program:
    """Bridge the real eager graph to logical paged state owned by runtime."""

    def __init__(self, model: Qwen3_5ReferenceModel, owned_layer_ids: tuple[int, ...]) -> None:
        expected = tuple(range(model.config.graph.text.layer_count))
        if owned_layer_ids != expected:
            raise ValueError("single-rank qwen3_5 must own the complete decoder layer range")
        self.model = model
        self._owned_layer_ids = owned_layer_ids

    def named_weight_tensors(self) -> tuple[tuple[str, Tensor], ...]:
        return self.model.weights.named_tensors()

    def execution_recipe(self) -> ProgramExecutionRecipe:
        text = self.model.config.graph.text
        model_source = Path(__file__).with_name("model.py").read_bytes()
        return ProgramExecutionRecipe(
            recipe_id="cubic.qwen3_5.dense.reference",
            revision="2",
            settings=(
                ("decoder_layers", str(text.layer_count)),
                ("feature_fusion", "decoder.input@layer:0;identity_rows"),
                ("full_attention_interval", str(text.full_attention_interval)),
                ("owned_layer_ids", f"0-{text.layer_count - 1}"),
            ),
            artifact_digests=(
                ("checkpoint_inventory", self.model.weights.inventory_digest),
                ("model_source", f"b2s256:{hashlib.blake2s(model_source).hexdigest()}"),
                ("semantic_plan", self.model.weights.semantic_plan_digest),
            ),
        )

    def initialize_request(self) -> tuple[StateInitialization, ...]:
        initial = self.model.empty_cache()
        states: list[StateInitialization] = []
        for layer_index, layer in enumerate(initial.layers):
            if layer.convolution is None:
                continue
            assert layer.recurrent is not None
            states.extend(
                (
                    StateInitialization(
                        _state_id(layer_index, "convolution"),
                        layer.convolution,
                    ),
                    StateInitialization(
                        _state_id(layer_index, "recurrent"),
                        layer.recurrent,
                    ),
                )
            )
        return tuple(states)

    def execute_segment(
        self,
        segment: ModelSegment,
        ops: SemanticOpExecutor,
        state: ModelStateIO,
        features: FeatureReader,
    ) -> ModelStepResult:
        decoder_input_features = _decoder_input_features(
            segment,
            features,
            self.model.config,
        )
        previous_length = segment.token_start
        previous_ordinals = tuple(range(previous_length))
        layers: list[Qwen3_5LayerCache] = []
        for layer_index, kind in enumerate(self.model.config.graph.text.layer_schedule):
            if kind is AttentionKind.GATED_DELTA_NET:
                layers.append(
                    Qwen3_5LayerCache(
                        convolution=state.read_recurrent_state(
                            _state_id(layer_index, "convolution")
                        ),
                        recurrent=state.read_recurrent_state(_state_id(layer_index, "recurrent")),
                    )
                )
            elif previous_ordinals:
                key, value = state.read_kv(layer_index, previous_ordinals)
                layers.append(
                    Qwen3_5LayerCache(
                        key=key.transpose(0, 1).unsqueeze(0),
                        value=value.transpose(0, 1).unsqueeze(0),
                    )
                )
            else:
                layers.append(Qwen3_5LayerCache())

        input_ids = torch.tensor(
            (segment.token_ids,),
            dtype=torch.int64,
            device=self.model.weights.device,
        )
        raw_positions = torch.tensor(
            tuple(position.axes for position in segment.positions),
            dtype=torch.int64,
            device=self.model.weights.device,
        )
        if raw_positions.shape[1] == 1:
            raw_positions = raw_positions.expand(-1, 3)
        elif raw_positions.shape[1] != 3:
            raise ValueError("qwen3_5 positions require one or three realized axes")
        result = self.model.forward(
            input_ids,
            cache=Qwen3_5ReferenceCache(previous_length, 1, tuple(layers)),
            position_ids=raw_positions.transpose(0, 1).unsqueeze(1),
            operators=_SegmentOperators(ops, segment.phase),
            decoder_input_features=decoder_input_features,
        )

        target_position = previous_length + segment.token_count
        if result.cache.sequence_length != target_position:
            raise RuntimeError("qwen3_5 returned the wrong state position")
        ordinals = tuple(range(previous_length, target_position))
        for layer_index, (kind, layer) in enumerate(
            zip(self.model.config.graph.text.layer_schedule, result.cache.layers, strict=True)
        ):
            if kind is AttentionKind.GATED_DELTA_NET:
                assert layer.convolution is not None and layer.recurrent is not None
                state.write_recurrent_state(
                    _state_id(layer_index, "convolution"),
                    layer.convolution,
                    target_position=target_position,
                )
                state.write_recurrent_state(
                    _state_id(layer_index, "recurrent"),
                    layer.recurrent,
                    target_position=target_position,
                )
                continue
            assert layer.key is not None and layer.value is not None
            state.write_kv(
                layer_index,
                ordinals,
                layer.key[0, :, previous_length:target_position].transpose(0, 1),
                layer.value[0, :, previous_length:target_position].transpose(0, 1),
            )

        return ModelStepResult(logits=result.logits[0, -1])


class Qwen3_5Adapter:
    def __init__(
        self,
        *,
        config: Qwen3_5Config,
        inventory: CheckpointInventory,
        plan: Qwen3_5WeightPlan,
    ) -> None:
        self.config = config
        self.inventory = inventory
        self.plan = plan
        text = config.graph.text
        states = tuple(
            state
            for layer_index, kind in enumerate(text.layer_schedule)
            if kind is AttentionKind.GATED_DELTA_NET
            for state in (
                StateSpec(
                    state_id=_state_id(layer_index, "convolution"),
                    dtype=ScalarDType.BF16,
                    layout="batch_channel_history",
                    shape_symbols=("batch", "channel", "history"),
                    sharding=StateSharding.TENSOR,
                ),
                StateSpec(
                    state_id=_state_id(layer_index, "recurrent"),
                    dtype=ScalarDType.FP32,
                    layout="batch_value_head_value_dim_key_dim",
                    shape_symbols=("batch", "value_head", "value_dim", "key_dim"),
                    sharding=StateSharding.TENSOR,
                ),
            )
        )
        from cubic_serve.models.qwen3_5.weights import qwen3_5_text_weight_policy

        self.weight_policy = qwen3_5_text_weight_policy(plan)
        weights = checkpoint_weight_manifest(
            inventory,
            policy=self.weight_policy,
            semantic_plan_digest=plan.semantic_plan_digest,
        )
        model_id = "qwen3_5:text"
        self._descriptor = ModelAdapterDescriptor(
            adapter_id="quanttrio.qwen3_5.dense",
            adapter_revision="1",
            model_spec=ModelSpec(
                model_id=model_id,
                architecture=config.identity.architecture,
                revision=inventory.inventory_digest,
                weights=weights,
                cache=CacheSpec(
                    kv_dtype=ScalarDType.BF16,
                    page_size_tokens=_BLOCK_SIZE,
                    layout="layer_token_kv_head_dim",
                    prefix_enabled=True,
                ),
                modalities=(
                    ModalitySpec(name="text"),
                    ModalitySpec(
                        name="image",
                        mime_prefixes=("image/",),
                        encoder_id=(f"qwen3_5.vision.reference@{plan.semantic_plan_digest}"),
                        feature_hidden_size=text.hidden_size,
                    ),
                    ModalitySpec(
                        name="video",
                        mime_prefixes=("video/",),
                        encoder_id=(f"qwen3_5.vision.reference@{plan.semantic_plan_digest}"),
                        feature_hidden_size=text.hidden_size,
                    ),
                ),
                states=states,
                layer_count=text.layer_count,
            ),
            weight_variant=WeightVariantMeta(
                base_model_id=model_id,
                base_revision=inventory.inventory_digest,
            ),
            decode_extension=(
                DecodeExtensionMeta(
                    provider=DecodeProvider.MTP,
                    provider_version="qwen3_5.mtp.v1",
                    max_proposal_tokens=QWEN3_5_MTP_MAX_PROPOSAL_TOKENS,
                    state_schema_id="qwen3_5.mtp.full_attention.bf16_kv.v1",
                )
                if plan.mtp_extension_names
                else DecodeExtensionMeta()
            ),
            required_ops=_required_signatures(),
        )

    def describe(self) -> ModelAdapterDescriptor:
        return self._descriptor

    def build(self, context: ModelBuildContext) -> Qwen3_5Program:
        if context.model_root != self.inventory.model_root:
            raise ValueError("qwen3_5 build root differs from checkpoint admission")
        if context.rank_id != 0 or context.world_size != 1:
            raise ValueError("qwen3_5 reference program is currently single-rank")
        model = materialize_qwen3_5_reference(
            config=self.config,
            inventory=self.inventory,
            plan=self.plan,
            device=context.device,
        )
        return Qwen3_5Program(model, context.owned_layer_ids)


def build_qwen3_5_draft_provider(config, *, method: str, max_depth: int):
    """Bind the existing model draft construction; engine policy stays shared."""
    from functools import partial

    from cubic_serve.executor.draft import DraftProvider, DraftSpec
    from cubic_serve.models.builtin import BuiltinModelConfigError

    if method != "mtp":
        raise BuiltinModelConfigError(
            "model has no qualified shared-graph draft construction binding"
        )
    from cubic_serve.executor.draft import HiddenConditionedReference
    from cubic_serve.executor.mtp_runner import PackedMTPStage

    t = config.graph.text
    if t.mtp.hidden_layer_count != 1 or t.mtp.dedicated_embeddings or t.moe is not None:
        raise BuiltinModelConfigError("dense MTP requires one declared shared-embedding block")
    spec = DraftSpec(method, t.vocabulary_size, t.hidden_size, max_depth)
    return DraftProvider(
        spec,
        partial(PackedMTPStage, config, max_depth=max_depth),
        partial(HiddenConditionedReference, spec=spec),
        ("mtp.fc.weight",),
    )


def with_qwen_text_compute_dtype(raw, compute_dtype):
    """Preserve the model document while declaring quantized execution dtype."""
    from cubic_serve.models.builtin import BuiltinModelConfigError

    text = raw.get("text_config")
    if not isinstance(text, Mapping):
        raise BuiltinModelConfigError("quantized compute policy requires text_config")
    # The original document remains attested by the checkpoint inventory.
    # Exporter dtype is a storage hint; this graph declares execution math.
    return {**raw, "text_config": {**text, "dtype": compute_dtype}}
