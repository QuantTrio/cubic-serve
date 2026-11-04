"""Configuration for checkpoints whose ``model_type`` is ``qwen4_exp``.

The initial profile is marketed as Qwen3.8-Flash-Next.  This package is named
from its checkpoint graph key and does not masquerade as qwen3_5.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from cubic_serve._compat import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch
from torch import Tensor

from cubic_serve.contracts.execution import ParallelTopology
from cubic_serve.ops.eager import hyperconnection as hc
from cubic_serve.ops.eager.activation import sigmoid_stored, silu_stored
from cubic_serve.ops.eager.hybrid import (
    apply_partial_rotary,
    depthwise_causal_conv1d,
    gated_delta_net_recurrent,
    gated_rms_norm,
    interleaved_mrope_cos_sin,
    sigmoid_output_gate,
    zero_centered_rms_norm,
)
from cubic_serve.ops.eager.moe import topk_routes
from cubic_serve.ops.eager.ngram import dilated_depthwise_conv, hash_indices
from cubic_serve.ops.eager.sparse_indexer import attend_selected, select_blocks
from cubic_serve.runtime.cache import KVLayout, StateSlotSpec
from cubic_serve.runtime.cache.pages import PageFieldSpec

if TYPE_CHECKING:
    from cubic_serve.executor.text_graph import TextGraphBindings
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.ops.cuda.position.qwen_mrope_table import MRoPETableEvidence


class Qwen4ExpConfigError(ValueError):
    """Configuration does not match the closed qwen4_exp schema."""


class AttentionKind(StrEnum):
    GATED_DELTA_NET = "gated-delta-net"
    FULL_ATTENTION = "full-attention"


class LayerStateKind(StrEnum):
    GDN_CONVOLUTION = "gdn-convolution"
    GDN_RECURRENT_MATRIX = "gdn-recurrent-matrix"
    ATTENTION_KV = "attention-kv"
    QSA_RAW_KEY_RING = "qsa-raw-key-ring"
    QSA_POSITION_RING = "qsa-position-ring"
    QSA_COMPRESSED_KEY_CACHE = "qsa-compressed-key-cache"
    PLE_CONVOLUTION = "ple-convolution"
    PLE_NGRAM_TOKEN_CONTEXT = "ple-ngram-token-context"


class FullAttentionKind(StrEnum):
    QSA = "qsa"


class StateReplication(StrEnum):
    TP_REPLICATED = "tp-replicated"
    TP_PARTITIONED = "tp-partitioned"


class PrefixCacheBoundary(StrEnum):
    COMPRESSION_GROUP_ALIGNED_ONLY = "compression-group-aligned-only"


class ArchitecturePlanStatus(StrEnum):
    ARCHITECTURE_ONLY = "architecture-only"


class MTPStatus(StrEnum):
    METADATA_ONLY = "metadata-only"


@dataclass(frozen=True, slots=True)
class Qwen4ExpIdentity:
    release_family: str
    release_profile: str
    model_type: str
    graph_family: str
    architecture: str


@dataclass(frozen=True, slots=True)
class MRoPEConfig:
    section: tuple[int, int, int]
    partial_rotary_factor: float
    rope_theta: float


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
    normalize_topk_probability: bool


@dataclass(frozen=True, slots=True)
class QSAConfig:
    query_heads: int
    key_value_heads: int
    head_dim: int
    token_budget: int
    compression_ratio: int


@dataclass(frozen=True, slots=True)
class PLEConfig:
    layer_ids: tuple[int, ...]
    embedding_dim: int
    convolution_kernel_size: int
    ngram_size: int
    heads_per_ngram: int
    vocabulary_base: int
    vocabulary_multiple: int
    split_parts: int


@dataclass(frozen=True, slots=True)
class HyperconnectionConfig:
    stream_count: int
    low_rank: int


@dataclass(frozen=True, slots=True)
class QSAStateMetadata:
    """Logical persistent-state ABI; batch/block extents remain runtime-owned."""

    main_kv_dtype: str
    raw_key_dtype: str
    compressed_key_dtype: str
    side_cache_replication: StateReplication
    raw_key_width: int
    raw_mrope_position_components: int
    raw_storage_width: int
    raw_base_capacity: int
    raw_slot_rule: str
    compressed_key_heads: int
    compressed_key_head_dim: int
    compressed_rows_divisor: int
    prefix_cache_boundary: PrefixCacheBoundary


@dataclass(frozen=True, slots=True)
class PLEStateMetadata:
    convolution_dtype: str
    token_context_dtype: str
    replication: StateReplication
    convolution_channels: int
    convolution_context_width: int
    token_context_length: int


@dataclass(frozen=True, slots=True)
class MTPMetadata:
    status: MTPStatus
    hidden_layer_count: int
    dedicated_embeddings: bool
    hybrid: bool
    layer_schedule: tuple[AttentionKind, ...]
    hidden_state_source_layer: int | None
    rope_theta: float


@dataclass(frozen=True, slots=True)
class VisionConfig:
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
class Qwen4ExpTextConfig:
    dtype: str
    vocabulary_size: int
    hidden_size: int
    layer_count: int
    max_position_embeddings: int
    activation: str
    rms_norm_epsilon: float
    layer_schedule: tuple[AttentionKind, ...]
    full_attention_interval: int
    full_attention: FullAttentionConfig
    gated_delta_net: GatedDeltaNetConfig
    moe: MoEConfig
    mrope: MRoPEConfig
    hyperconnection: HyperconnectionConfig
    ple: PLEConfig
    qsa: QSAConfig
    mtp: MTPMetadata
    ple_embedding_dtype: str = "bfloat16"
    eos_token_id: int | None = None


@dataclass(frozen=True, slots=True)
class VisionLayerPlan:
    layer_index: int
    node_id: str


@dataclass(frozen=True, slots=True)
class DecoderLayerPlan:
    layer_index: int
    node_id: str
    attention: AttentionKind
    full_attention_kind: FullAttentionKind | None
    state_requirements: tuple[LayerStateKind, ...]
    uses_ple: bool

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
    full_attention_kind: FullAttentionKind | None
    state_requirements: tuple[LayerStateKind, ...]

    @property
    def state_node_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{self.node_id}.state.{requirement.value}" for requirement in self.state_requirements
        )


@dataclass(frozen=True, slots=True)
class Qwen4ExpArchitecturePlan:
    status: ArchitecturePlanStatus
    graph_family: str
    vision_patch_embedding_node_id: str
    vision_layers: tuple[VisionLayerPlan, ...]
    vision_merger_node_id: str
    token_embedding_node_id: str
    decoder_layers: tuple[DecoderLayerPlan, ...]
    final_mixer_node_id: str
    lm_head_node_id: str
    mtp_root_node_id: str
    mtp_layers: tuple[MTPLayerPlan, ...]
    qsa_state: QSAStateMetadata
    ple_state: PLEStateMetadata


@dataclass(frozen=True, slots=True)
class Qwen4ExpConfig:
    identity: Qwen4ExpIdentity
    text: Qwen4ExpTextConfig
    vision: VisionConfig
    modality_tokens: ModalityTokenConfig

    @property
    def text_config(self) -> Qwen4ExpTextConfig:
        """Expose the original text schema without copying or changing policy."""
        return self.text

    def architecture_plan(self) -> Qwen4ExpArchitecturePlan:
        ple_ids = frozenset(self.text.ple.layer_ids)

        def state_requirements(
            attention: AttentionKind, *, uses_ple: bool
        ) -> tuple[LayerStateKind, ...]:
            requirements = (
                (
                    LayerStateKind.GDN_CONVOLUTION,
                    LayerStateKind.GDN_RECURRENT_MATRIX,
                )
                if attention is AttentionKind.GATED_DELTA_NET
                else (
                    LayerStateKind.ATTENTION_KV,
                    LayerStateKind.QSA_RAW_KEY_RING,
                    LayerStateKind.QSA_POSITION_RING,
                    LayerStateKind.QSA_COMPRESSED_KEY_CACHE,
                )
            )
            return (
                (
                    *requirements,
                    LayerStateKind.PLE_CONVOLUTION,
                    LayerStateKind.PLE_NGRAM_TOKEN_CONTEXT,
                )
                if uses_ple
                else requirements
            )

        return Qwen4ExpArchitecturePlan(
            status=ArchitecturePlanStatus.ARCHITECTURE_ONLY,
            graph_family="qwen4_exp",
            vision_patch_embedding_node_id="vision.patch_embedding",
            vision_layers=tuple(
                VisionLayerPlan(index, f"vision.encoder.layer.{index:04d}")
                for index in range(self.vision.layer_count)
            ),
            vision_merger_node_id="vision.patch_merger",
            token_embedding_node_id="text.token_embedding",
            decoder_layers=tuple(
                DecoderLayerPlan(
                    layer_index=index,
                    node_id=f"text.decoder.layer.{index:04d}",
                    attention=attention,
                    full_attention_kind=(
                        None
                        if attention is AttentionKind.GATED_DELTA_NET
                        else FullAttentionKind.QSA
                    ),
                    state_requirements=state_requirements(
                        attention, uses_ple=(index + 1) in ple_ids
                    ),
                    uses_ple=(index + 1) in ple_ids,
                )
                for index, attention in enumerate(self.text.layer_schedule)
            ),
            final_mixer_node_id="text.final_hyperconnection_mixer",
            lm_head_node_id="text.lm_head",
            mtp_root_node_id="extension.mtp",
            mtp_layers=tuple(
                MTPLayerPlan(
                    index,
                    f"extension.mtp.layer.{index:04d}",
                    attention,
                    (None if attention is AttentionKind.GATED_DELTA_NET else FullAttentionKind.QSA),
                    state_requirements(attention, uses_ple=False),
                )
                for index, attention in enumerate(self.text.mtp.layer_schedule)
            ),
            qsa_state=QSAStateMetadata(
                main_kv_dtype="bfloat16",
                raw_key_dtype="bfloat16",
                compressed_key_dtype="bfloat16",
                side_cache_replication=StateReplication.TP_PARTITIONED,
                raw_key_width=self.text.qsa.head_dim,
                raw_mrope_position_components=3,
                raw_storage_width=self.text.qsa.head_dim + 3 * 4,
                raw_base_capacity=self.text.qsa.compression_ratio,
                raw_slot_rule="absolute-position-mod-capacity",
                compressed_key_heads=self.text.qsa.key_value_heads,
                compressed_key_head_dim=self.text.qsa.head_dim,
                compressed_rows_divisor=self.text.qsa.compression_ratio,
                prefix_cache_boundary=PrefixCacheBoundary.COMPRESSION_GROUP_ALIGNED_ONLY,
            ),
            ple_state=PLEStateMetadata(
                convolution_dtype="bfloat16",
                token_context_dtype="int64",
                replication=StateReplication.TP_REPLICATED,
                convolution_channels=(
                    self.text.hidden_size * self.text.hyperconnection.stream_count
                ),
                convolution_context_width=(
                    (self.text.ple.convolution_kernel_size - 1) * self.text.ple.ngram_size
                ),
                token_context_length=self.text.ple.ngram_size - 1,
            ),
        )


_TOP_KEYS = frozenset(
    {
        "quantization_config",
        "architectures",
        "image_token_id",
        "language_model_only",
        "model_type",
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
_TEXT_KEYS = frozenset(
    {
        "ple_embedding_dtype",
        "attention_bias",
        "attention_dropout",
        "attn_output_gate",
        "bos_token_id",
        "decoder_sparse_step",
        "dtype",
        "eos_token_id",
        "full_attention_interval",
        "hc_count",
        "hc_lowrank",
        "head_dim",
        "heads_per_ngram",
        "hidden_act",
        "hidden_size",
        "indexer_budget",
        "indexer_compress_ratio",
        "indexer_head_dim",
        "indexer_kv_heads",
        "indexer_n_heads",
        "initializer_range",
        "intermediate_size",
        "layer_types",
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_value_head_dim",
        "make_ngram_vocab_size_divisible_by",
        "mamba_ssm_dtype",
        "max_position_embeddings",
        "mlp_only_layers",
        "model_type",
        "moe_intermediate_size",
        "mtp",
        "mtp_num_hidden_layers",
        "mtp_use_dedicated_embeddings",
        "ngram_size",
        "ngram_vocab_size_base",
        "norm_topk_prob",
        "num_attention_heads",
        "num_experts",
        "num_experts_per_tok",
        "num_hidden_layers",
        "num_key_value_heads",
        "output_gate_type",
        "output_router_logits",
        "pad_token_id",
        "partial_rotary_factor",
        "ple_conv_kernel_size",
        "ple_embed_dim",
        "ple_layer_ids",
        "rms_norm_eps",
        "rope_parameters",
        "router_aux_loss_coef",
        "shared_expert_intermediate_size",
        "split_ngram_parts",
        "tie_word_embeddings",
        "use_cache",
        "vocab_size",
    }
)
_TEXT_REQUIRED = frozenset(
    {
        "attention_bias",
        "attention_dropout",
        "dtype",
        "full_attention_interval",
        "hc_count",
        "hc_lowrank",
        "head_dim",
        "heads_per_ngram",
        "hidden_act",
        "hidden_size",
        "indexer_budget",
        "indexer_compress_ratio",
        "indexer_head_dim",
        "indexer_kv_heads",
        "indexer_n_heads",
        "layer_types",
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_value_head_dim",
        "make_ngram_vocab_size_divisible_by",
        "mamba_ssm_dtype",
        "max_position_embeddings",
        "model_type",
        "moe_intermediate_size",
        "mtp",
        "mtp_num_hidden_layers",
        "mtp_use_dedicated_embeddings",
        "ngram_size",
        "ngram_vocab_size_base",
        "num_attention_heads",
        "num_experts",
        "num_experts_per_tok",
        "num_hidden_layers",
        "num_key_value_heads",
        "ple_conv_kernel_size",
        "ple_embed_dim",
        "ple_layer_ids",
        "rms_norm_eps",
        "rope_parameters",
        "shared_expert_intermediate_size",
        "split_ngram_parts",
        "use_cache",
        "vocab_size",
    }
)
_ROPE_KEYS = frozenset(
    {
        "mrope_interleaved",
        "mrope_section",
        "partial_rotary_factor",
        "rope_theta",
        "rope_type",
    }
)
_MTP_KEYS = frozenset(
    {
        "hybrid",
        "layer_types",
        "mtp_use_hidden_state_from_layer",
        "num_hidden_layers",
        "rope_theta",
    }
)


def parse_qwen4_exp_config(raw: Mapping[str, object]) -> Qwen4ExpConfig:
    root = _mapping(raw, scope="config")
    _keys(root, allowed=_TOP_KEYS, required=_TOP_REQUIRED, scope="config")
    if _string(root, "model_type", scope="config") != "qwen4_exp":
        raise Qwen4ExpConfigError("config.model_type must be 'qwen4_exp'")
    if _string_tuple(root, "architectures", scope="config") != (
        "Qwen4ExpForConditionalGeneration",
    ):
        raise Qwen4ExpConfigError("config.architectures does not match qwen4_exp")
    if _boolean(root, "tie_word_embeddings", scope="config"):
        raise Qwen4ExpConfigError("tied word embeddings are unsupported")
    if "language_model_only" in root and _boolean(root, "language_model_only", scope="config"):
        raise Qwen4ExpConfigError("multimodal profile cannot be language_model_only")

    text_raw = _child_mapping(root, "text_config", scope="config")
    _keys(text_raw, allowed=_TEXT_KEYS, required=_TEXT_REQUIRED, scope="config.text_config")
    if _string(text_raw, "model_type", scope="config.text_config") != "qwen4_exp_text":
        raise Qwen4ExpConfigError("text_config.model_type must be 'qwen4_exp_text'")
    if "tie_word_embeddings" in text_raw and _boolean(
        text_raw, "tie_word_embeddings", scope="config.text_config"
    ):
        raise Qwen4ExpConfigError(
            "config.text_config.tie_word_embeddings contradicts the untied root"
        )
    text = _parse_text(text_raw)

    vision_raw = _child_mapping(root, "vision_config", scope="config")
    _keys(
        vision_raw,
        allowed=_VISION_KEYS,
        required=_VISION_REQUIRED,
        scope="config.vision_config",
    )
    if _string(vision_raw, "model_type", scope="config.vision_config") != "qwen4_exp":
        raise Qwen4ExpConfigError("vision_config.model_type must be 'qwen4_exp'")
    vision = _parse_vision(vision_raw)
    if vision.output_hidden_size != text.hidden_size:
        raise Qwen4ExpConfigError("vision output hidden size must equal text hidden size")

    tokens = ModalityTokenConfig(
        image=_integer(root, "image_token_id", scope="config", minimum=0),
        video=_integer(root, "video_token_id", scope="config", minimum=0),
        vision_start=_integer(root, "vision_start_token_id", scope="config", minimum=0),
        vision_end=_integer(root, "vision_end_token_id", scope="config", minimum=0),
    )
    ids = (tokens.image, tokens.video, tokens.vision_start, tokens.vision_end)
    if len(set(ids)) != 4 or any(value >= text.vocabulary_size for value in ids):
        raise Qwen4ExpConfigError(
            "multimodal sentinel IDs must be distinct and inside the vocabulary"
        )
    config = Qwen4ExpConfig(
        identity=Qwen4ExpIdentity(
            release_family="qwen4_exp",
            release_profile="qwen4_exp",
            model_type="qwen4_exp",
            graph_family="qwen4_exp",
            architecture="Qwen4ExpForConditionalGeneration",
        ),
        text=text,
        vision=vision,
        modality_tokens=tokens,
    )
    return config


def parse_qwen4_exp_config_json(document: str | bytes | bytearray) -> Qwen4ExpConfig:
    try:
        value = json.loads(
            document,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise Qwen4ExpConfigError(f"invalid config JSON: {error}") from error
    if not isinstance(value, Mapping):
        raise Qwen4ExpConfigError("config JSON must contain one object")
    return parse_qwen4_exp_config(cast(Mapping[str, object], value))


def _parse_text(raw: Mapping[str, object]) -> Qwen4ExpTextConfig:
    scope = "config.text_config"
    ple_dtype = _optional_string(raw, "ple_embedding_dtype", default="bfloat16", scope=scope)
    if ple_dtype not in ("bfloat16", "float8_e4m3fn"):
        raise Qwen4ExpConfigError("unsupported PLE embedding storage dtype")
    if _string(raw, "dtype", scope=scope) != "bfloat16":
        raise Qwen4ExpConfigError("text dtype must be bfloat16")
    if _boolean(raw, "attention_bias", scope=scope):
        raise Qwen4ExpConfigError("attention bias is unsupported")
    if _number(raw, "attention_dropout", scope=scope, minimum=0.0) != 0.0:
        raise Qwen4ExpConfigError("attention dropout must be zero")
    if not _optional_boolean(raw, "attn_output_gate", default=True, scope=scope):
        raise Qwen4ExpConfigError("full attention operators require an output gate")
    if _string(raw, "hidden_act", scope=scope) not in ("silu", "swish"):
        raise Qwen4ExpConfigError("text operators require SiLU activation")
    if _boolean(raw, "mtp_use_dedicated_embeddings", scope=scope):
        raise Qwen4ExpConfigError("dedicated MTP embeddings are not implemented")
    if not _boolean(raw, "use_cache", scope=scope):
        raise Qwen4ExpConfigError("use_cache must be true")
    layers = _integer(raw, "num_hidden_layers", scope=scope, minimum=1)
    hidden = _integer(raw, "hidden_size", scope=scope, minimum=1)
    interval = _integer(raw, "full_attention_interval", scope=scope, minimum=2)
    schedule = _attention_schedule(raw, layer_count=layers, interval=interval)
    head_dim = _integer(raw, "head_dim", scope=scope, minimum=1)
    mrope = _parse_mrope(raw, head_dim=head_dim)
    gdn = _parse_gdn(raw)
    query_heads = _integer(raw, "num_attention_heads", scope=scope, minimum=1)
    kv_heads = _integer(raw, "num_key_value_heads", scope=scope, minimum=1)
    if query_heads % kv_heads:
        raise Qwen4ExpConfigError("query heads must divide into KV groups")

    experts = _integer(raw, "num_experts", scope=scope, minimum=1)
    active = _integer(raw, "num_experts_per_tok", scope=scope, minimum=1)
    if active > experts:
        raise Qwen4ExpConfigError("active experts cannot exceed expert count")
    if "mlp_only_layers" in raw and _integer_tuple(raw, "mlp_only_layers", scope=scope, minimum=0):
        raise Qwen4ExpConfigError("qwen4_exp operators do not support dense MLP layers")
    if (
        "decoder_sparse_step" in raw
        and _integer(raw, "decoder_sparse_step", scope=scope, minimum=1) != 1
    ):
        raise Qwen4ExpConfigError("qwen4_exp operators require MoE in every layer")

    streams = _integer(raw, "hc_count", scope=scope, minimum=2)
    ple_ids = _integer_tuple(raw, "ple_layer_ids", scope=scope, minimum=1)
    if len(set(ple_ids)) != len(ple_ids) or any(layer_id > layers for layer_id in ple_ids):
        raise Qwen4ExpConfigError("PLE uses unique 1-based decoder layer IDs")
    ple_dim = _integer(raw, "ple_embed_dim", scope=scope, minimum=1)
    ngram = _integer(raw, "ngram_size", scope=scope, minimum=2)
    heads_per_ngram = _integer(raw, "heads_per_ngram", scope=scope, minimum=1)
    if ple_dim % ((ngram - 1) * heads_per_ngram):
        raise Qwen4ExpConfigError("PLE embedding dim does not divide into ngram heads")

    qsa = QSAConfig(
        query_heads=_integer(raw, "indexer_n_heads", scope=scope, minimum=1),
        key_value_heads=_integer(raw, "indexer_kv_heads", scope=scope, minimum=1),
        head_dim=_integer(raw, "indexer_head_dim", scope=scope, minimum=1),
        token_budget=_integer(raw, "indexer_budget", scope=scope, minimum=1),
        compression_ratio=_integer(raw, "indexer_compress_ratio", scope=scope, minimum=1),
    )
    if qsa.key_value_heads != 1:
        raise Qwen4ExpConfigError("QSA requires indexer_kv_heads=1")
    if qsa.token_budget % qsa.compression_ratio:
        raise Qwen4ExpConfigError("QSA token budget must be divisible by compression ratio")
    if qsa.head_dim < int(head_dim * mrope.partial_rotary_factor):
        raise Qwen4ExpConfigError("QSA head dim cannot cover the rotary dimension")
    mtp = _parse_mtp(raw, mrope=mrope, text_layer_count=layers)

    return Qwen4ExpTextConfig(
        dtype="bfloat16",
        vocabulary_size=_integer(raw, "vocab_size", scope=scope, minimum=1),
        hidden_size=hidden,
        layer_count=layers,
        max_position_embeddings=_integer(raw, "max_position_embeddings", scope=scope, minimum=1),
        activation=_string(raw, "hidden_act", scope=scope),
        rms_norm_epsilon=_number(raw, "rms_norm_eps", scope=scope, exclusive_minimum=0.0),
        layer_schedule=schedule,
        full_attention_interval=interval,
        full_attention=FullAttentionConfig(
            query_heads=query_heads,
            key_value_heads=kv_heads,
            head_dim=head_dim,
            output_gate=_optional_boolean(raw, "attn_output_gate", default=True, scope=scope),
        ),
        gated_delta_net=gdn,
        moe=MoEConfig(
            expert_count=experts,
            experts_per_token=active,
            expert_intermediate_size=_integer(raw, "moe_intermediate_size", scope=scope, minimum=1),
            shared_expert_intermediate_size=_integer(
                raw, "shared_expert_intermediate_size", scope=scope, minimum=0
            ),
            normalize_topk_probability=_optional_boolean(
                raw, "norm_topk_prob", default=True, scope=scope
            ),
        ),
        mrope=mrope,
        hyperconnection=HyperconnectionConfig(
            stream_count=streams,
            low_rank=_integer(raw, "hc_lowrank", scope=scope, minimum=1),
        ),
        ple=PLEConfig(
            layer_ids=ple_ids,
            embedding_dim=ple_dim,
            convolution_kernel_size=_integer(raw, "ple_conv_kernel_size", scope=scope, minimum=1),
            ngram_size=ngram,
            heads_per_ngram=heads_per_ngram,
            vocabulary_base=_integer(raw, "ngram_vocab_size_base", scope=scope, minimum=1),
            vocabulary_multiple=_integer(
                raw, "make_ngram_vocab_size_divisible_by", scope=scope, minimum=1
            ),
            split_parts=_integer(raw, "split_ngram_parts", scope=scope, minimum=1),
        ),
        qsa=qsa,
        mtp=mtp,
        ple_embedding_dtype=ple_dtype,
        eos_token_id=_integer(raw, "eos_token_id", scope=scope, minimum=0)
        if "eos_token_id" in raw
        else None,
    )


def _parse_mrope(raw: Mapping[str, object], *, head_dim: int) -> MRoPEConfig:
    scope = "config.text_config"
    rope = _child_mapping(raw, "rope_parameters", scope=scope)
    _keys(rope, allowed=_ROPE_KEYS, required=_ROPE_KEYS, scope=f"{scope}.rope_parameters")
    if not _boolean(rope, "mrope_interleaved", scope=f"{scope}.rope_parameters"):
        raise Qwen4ExpConfigError("M-RoPE must be interleaved")
    section_raw = _integer_tuple(rope, "mrope_section", scope=f"{scope}.rope_parameters", minimum=1)
    if len(section_raw) != 3:
        raise Qwen4ExpConfigError("M-RoPE section must contain three axes")
    section = (section_raw[0], section_raw[1], section_raw[2])
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
        raise Qwen4ExpConfigError("partial rotary factor contradicts rope parameters")
    rotary_dim = head_dim * partial
    if not rotary_dim.is_integer() or int(rotary_dim) % 2 or sum(section) * 2 != int(rotary_dim):
        raise Qwen4ExpConfigError("M-RoPE section does not partition rotary dimension")
    if _string(rope, "rope_type", scope=f"{scope}.rope_parameters") != "default":
        raise Qwen4ExpConfigError("only default M-RoPE is supported")
    return MRoPEConfig(
        section=section,
        partial_rotary_factor=partial,
        rope_theta=_number(
            rope, "rope_theta", scope=f"{scope}.rope_parameters", exclusive_minimum=0.0
        ),
    )


def _parse_gdn(raw: Mapping[str, object]) -> GatedDeltaNetConfig:
    scope = "config.text_config"
    state_dtype = _string(raw, "mamba_ssm_dtype", scope=scope)
    if state_dtype != "float32":
        raise Qwen4ExpConfigError("GDN recurrent state dtype must be float32")
    key_heads = _integer(raw, "linear_num_key_heads", scope=scope, minimum=1)
    value_heads = _integer(raw, "linear_num_value_heads", scope=scope, minimum=1)
    if value_heads % key_heads:
        raise Qwen4ExpConfigError("GDN value heads must be divisible by key heads")
    gate = _optional_string(raw, "output_gate_type", default="silu", scope=scope)
    if gate not in {"sigmoid", "silu", "swish"}:
        raise Qwen4ExpConfigError(f"unsupported GDN output gate {gate!r}")
    return GatedDeltaNetConfig(
        convolution_kernel_size=_integer(raw, "linear_conv_kernel_dim", scope=scope, minimum=1),
        key_head_dim=_integer(raw, "linear_key_head_dim", scope=scope, minimum=1),
        value_head_dim=_integer(raw, "linear_value_head_dim", scope=scope, minimum=1),
        key_heads=key_heads,
        value_heads=value_heads,
        recurrent_state_dtype=state_dtype,
        output_gate_type=gate,
    )


def _parse_mtp(
    raw: Mapping[str, object], *, mrope: MRoPEConfig, text_layer_count: int
) -> MTPMetadata:
    scope = "config.text_config"
    count = _integer(raw, "mtp_num_hidden_layers", scope=scope, minimum=0)
    nested = _child_mapping(raw, "mtp", scope=scope)
    _keys(nested, allowed=_MTP_KEYS, required=_MTP_KEYS, scope=f"{scope}.mtp")
    if _integer(nested, "num_hidden_layers", scope=f"{scope}.mtp", minimum=0) != count:
        raise Qwen4ExpConfigError("nested MTP layer count contradicts outer metadata")
    names = _string_tuple(nested, "layer_types", scope=f"{scope}.mtp")
    if len(names) != count:
        raise Qwen4ExpConfigError("MTP layer schedule length is inconsistent")
    schedule = tuple(_attention_kind(name) for name in names)
    theta = _number(nested, "rope_theta", scope=f"{scope}.mtp", exclusive_minimum=0.0)
    if theta != mrope.rope_theta:
        raise Qwen4ExpConfigError("MTP rope theta contradicts base M-RoPE")
    source = _nullable_integer(
        nested, "mtp_use_hidden_state_from_layer", scope=f"{scope}.mtp", minimum=0
    )
    if source is not None and source >= text_layer_count:
        raise Qwen4ExpConfigError("MTP hidden-state source must refer to a text layer")
    return MTPMetadata(
        status=MTPStatus.METADATA_ONLY,
        hidden_layer_count=count,
        dedicated_embeddings=_boolean(raw, "mtp_use_dedicated_embeddings", scope=scope),
        hybrid=_boolean(nested, "hybrid", scope=f"{scope}.mtp"),
        layer_schedule=schedule,
        hidden_state_source_layer=source,
        rope_theta=theta,
    )


def _parse_vision(raw: Mapping[str, object]) -> VisionConfig:
    scope = "config.vision_config"
    layers = _integer(raw, "depth", scope=scope, minimum=1)
    deepstack = _integer_tuple(raw, "deepstack_visual_indexes", scope=scope, minimum=0)
    if len(set(deepstack)) != len(deepstack) or any(index >= layers for index in deepstack):
        raise Qwen4ExpConfigError("deepstack indexes must be unique valid vision layers")
    return VisionConfig(
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
        raise Qwen4ExpConfigError("layer_types length must match num_hidden_layers")
    schedule = tuple(_attention_kind(name) for name in names)
    expected = tuple(
        AttentionKind.FULL_ATTENTION
        if (index + 1) % interval == 0
        else AttentionKind.GATED_DELTA_NET
        for index in range(layer_count)
    )
    if schedule != expected:
        raise Qwen4ExpConfigError("layer_types contradicts full_attention_interval")
    return schedule


def _attention_kind(value: str) -> AttentionKind:
    if value == "linear_attention":
        return AttentionKind.GATED_DELTA_NET
    if value == "full_attention":
        return AttentionKind.FULL_ATTENTION
    raise Qwen4ExpConfigError(f"unsupported attention kind {value!r}")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise Qwen4ExpConfigError(f"config JSON contains duplicate field {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise Qwen4ExpConfigError(f"config JSON contains non-finite constant {value!r}")


def _mapping(raw: Mapping[str, object], *, scope: str) -> dict[str, object]:
    if any(not isinstance(key, str) for key in raw):
        raise Qwen4ExpConfigError(f"{scope} keys must be strings")
    return dict(raw)


def _child_mapping(raw: Mapping[str, object], key: str, *, scope: str) -> dict[str, object]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise Qwen4ExpConfigError(f"{scope}.{key} must be an object")
    return _mapping(cast(Mapping[str, object], value), scope=f"{scope}.{key}")


def _keys(
    raw: Mapping[str, object], *, allowed: frozenset[str], required: frozenset[str], scope: str
) -> None:
    present = frozenset(raw)
    missing = sorted(required - present)
    unknown = sorted(present - allowed)
    if missing or unknown:
        raise Qwen4ExpConfigError(f"{scope} schema mismatch: missing={missing}, unknown={unknown}")


def _string(raw: Mapping[str, object], key: str, *, scope: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise Qwen4ExpConfigError(f"{scope}.{key} must be a non-empty string")
    return value


def _optional_string(raw: Mapping[str, object], key: str, *, default: str, scope: str) -> str:
    return _string(raw, key, scope=scope) if key in raw else default


def _boolean(raw: Mapping[str, object], key: str, *, scope: str) -> bool:
    value = raw.get(key)
    if type(value) is not bool:
        raise Qwen4ExpConfigError(f"{scope}.{key} must be a boolean")
    return value


def _optional_boolean(raw: Mapping[str, object], key: str, *, default: bool, scope: str) -> bool:
    return _boolean(raw, key, scope=scope) if key in raw else default


def _integer(raw: Mapping[str, object], key: str, *, scope: str, minimum: int | None = None) -> int:
    value = raw.get(key)
    if type(value) is not int:
        raise Qwen4ExpConfigError(f"{scope}.{key} must be an integer")
    if minimum is not None and value < minimum:
        raise Qwen4ExpConfigError(f"{scope}.{key} must be >= {minimum}")
    return value


def _nullable_integer(
    raw: Mapping[str, object], key: str, *, scope: str, minimum: int
) -> int | None:
    return None if raw.get(key) is None else _integer(raw, key, scope=scope, minimum=minimum)


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
        raise Qwen4ExpConfigError(f"{scope}.{key} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise Qwen4ExpConfigError(f"{scope}.{key} must be finite")
    if minimum is not None and result < minimum:
        raise Qwen4ExpConfigError(f"{scope}.{key} must be >= {minimum}")
    if exclusive_minimum is not None and result <= exclusive_minimum:
        raise Qwen4ExpConfigError(f"{scope}.{key} must be > {exclusive_minimum}")
    if maximum is not None and result > maximum:
        raise Qwen4ExpConfigError(f"{scope}.{key} must be <= {maximum}")
    return result


def _sequence(raw: Mapping[str, object], key: str, *, scope: str) -> Sequence[object]:
    value = raw.get(key)
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise Qwen4ExpConfigError(f"{scope}.{key} must be an array")
    return cast(Sequence[object], value)


def _string_tuple(raw: Mapping[str, object], key: str, *, scope: str) -> tuple[str, ...]:
    values = _sequence(raw, key, scope=scope)
    if any(not isinstance(value, str) or not value for value in values):
        raise Qwen4ExpConfigError(f"{scope}.{key} must contain non-empty strings")
    return tuple(cast(str, value) for value in values)


def _integer_tuple(
    raw: Mapping[str, object], key: str, *, scope: str, minimum: int
) -> tuple[int, ...]:
    values = _sequence(raw, key, scope=scope)
    if any(type(value) is not int or value < minimum for value in values):
        raise Qwen4ExpConfigError(f"{scope}.{key} must contain integers >= {minimum}")
    return tuple(cast(int, value) for value in values)


# The model composes mathematical operators only. Weight placement, collectives,
# graph capture, admission and the public serving loop remain engine authority.


@dataclass(frozen=True)
class Qwen4ExpLayerState:
    convolution: Tensor | None = None
    recurrent: Tensor | None = None
    key: Tensor | None = None
    value: Tensor | None = None
    index_keys: Tensor | None = None
    compressed_index_keys: Tensor | None = None
    ple_history: Tensor | None = None
    ple_convolution: Tensor | None = None
    index_positions: Tensor | None = None


@dataclass(frozen=True)
class Qwen4ExpReferenceState:
    length: int
    layers: tuple[Qwen4ExpLayerState, ...]


class Qwen4ExpWeights(Protocol):
    """Logical tensor namespace; serialized matrix formats remain opaque."""

    def __getitem__(self, name: str) -> Any: ...


class Qwen4ExpOperators(Protocol):
    def linear(self, inputs: Tensor, weight: Any) -> Tensor: ...

    def embedding(self, ids: Tensor, weight: Any, *, dtype: torch.dtype) -> Tensor: ...

    def embedding_shards(
        self,
        ids: Tensor,
        base: str,
        weights: Qwen4ExpWeights,
        *,
        parts: int,
        dtype: torch.dtype,
    ) -> Tensor: ...


class Qwen4ExpReferenceModel:
    """Independent MTP0 composition with explicit feature, position and state inputs.

    ``operators`` supplies linear, embedding and sharded-table lookup; those
    calls own storage/format handling. This class never names NVFP4 or Cubic.
    The reference retains complete raw index keys for clarity; a compressed
    candidate cache must independently reproduce these pooled keys.
    ``expert_range`` preserves checkpoint expert IDs; ``reduce_experts``
    completes the owner-local route fold before the shared expert is added.
    Matrix-shard completion remains the linear operator's responsibility.
    """

    def __init__(
        self, config: Qwen4ExpConfig, weights: Qwen4ExpWeights, operators: Qwen4ExpOperators,
        *, recurrent_state_dtype: torch.dtype | None = None,
        expert_range: tuple[int, int] | None = None,
        reduce_experts: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        self.config, self.weights, self.operators = config, weights, operators
        self.expert_range = (
            (0, config.text.moe.expert_count) if expert_range is None else expert_range
        )
        first, end = self.expert_range
        if not 0 <= first < end <= config.text.moe.expert_count:
            raise ValueError("reference expert ownership is outside the checkpoint")
        if self.expert_range != (0, config.text.moe.expert_count) and reduce_experts is None:
            raise ValueError("partitioned reference experts require an owner reduction")
        self.reduce_experts = reduce_experts
        self.recurrent_state_dtype = recurrent_state_dtype or getattr(
            torch, config.text.gated_delta_net.recurrent_state_dtype
        )
        if self.recurrent_state_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("GDN recurrent state requires BF16 or FP32 storage")

    def _linear(self, x: Tensor, name: str) -> Tensor:
        return self.operators.linear(x, self.weights[name + ".weight"])

    def _norm(self, x: Tensor, name: str) -> Tensor:
        return zero_centered_rms_norm(
            x,
            self.weights[name + ".weight"],
            epsilon=self.config.text.rms_norm_epsilon,
            output_dtype=x.dtype,
        )

    def _mix(self, x: Tensor, name: str) -> tuple[Tensor, Tensor]:
        t = self.config.text
        return hc.mix(
            x,
            self.weights[name + ".hc_norm.weight"],
            self.weights[name + ".input_mix_weight_down.weight"],
            self.weights[name + ".input_mix_weight_up.weight"],
            streams=t.hyperconnection.stream_count,
            eps=t.rms_norm_epsilon,
            project=self.operators.linear,
        )

    def _inject(self, residual: Tensor, block: Tensor, normalized: Tensor, name: str) -> Tensor:
        gates = hc.injection(
            normalized,
            self.weights[name + ".block_inject_weight.weight"],
            streams=self.config.text.hyperconnection.stream_count,
            project=self.operators.linear,
        )
        return hc.combine(residual, block, gates)

    def _ple(
        self, hidden: Tensor, tokens: Tensor, state: Qwen4ExpLayerState, prefix: str
    ) -> tuple[Tensor, Qwen4ExpLayerState]:
        t, p = self.config.text, self.config.text.ple
        if t.eos_token_id is None:
            raise ValueError("PLE execution requires the checkpoint EOS token ID")
        b, _, _ = hidden.shape
        history = state.ple_history
        if history is None:
            history = torch.full(
                (b, p.ngram_size - 1), t.eos_token_id, dtype=torch.int64, device=hidden.device
            )
        base = prefix + ".ple_embedding"
        ids, history = hash_indices(
            tokens,
            history,
            self.weights[base + ".layer_multipliers"],
            self.weights[base + ".ngram_heads_vocab_sizes"],
            self.weights[base + ".ngram_heads_offsets"],
            eos_token_id=t.eos_token_id,
        )
        embeddings = self.operators.embedding_shards(
            ids, base + ".ngram_embedding", self.weights, parts=p.split_parts, dtype=hidden.dtype
        ).flatten(-2)
        key = hc.grouped_rms_norm(
            self._linear(embeddings, prefix + ".key_proj"),
            self.weights[prefix + ".norm_key.weight"],
            group_size=t.hidden_size,
            eps=t.rms_norm_epsilon,
        ).reshape(b, tokens.shape[1], t.hyperconnection.stream_count, t.hidden_size)
        query = hc.grouped_rms_norm(
            hidden,
            self.weights[prefix + ".norm_query.weight"],
            group_size=t.hidden_size,
            eps=t.rms_norm_epsilon,
        ).reshape_as(key)
        value = self._linear(embeddings, prefix + ".value_proj")
        gated = hc.signed_root_gate(key, query, value)
        conv_input = hc.grouped_rms_norm(
            gated,
            self.weights[prefix + ".norm_conv.weight"],
            group_size=t.hidden_size,
            eps=t.rms_norm_epsilon,
        )
        previous = state.ple_convolution
        if previous is None:
            previous = torch.zeros(
                b,
                (p.convolution_kernel_size - 1) * p.ngram_size,
                hidden.shape[-1],
                dtype=hidden.dtype,
                device=hidden.device,
            )
        conv, next_conv = dilated_depthwise_conv(
            conv_input, previous, self.weights[prefix + ".conv1d.weight"], dilation=p.ngram_size
        )
        return hidden + (gated + conv), replace(
            state, ple_history=history, ple_convolution=next_conv
        )

    def _gdn(
        self, hidden: Tensor, state: Qwen4ExpLayerState, prefix: str
    ) -> tuple[Tensor, Qwen4ExpLayerState]:
        t, g = self.config.text, self.config.text.gated_delta_net
        b, tokens, _ = hidden.shape
        kw, vw = g.key_heads * g.key_head_dim, g.value_heads * g.value_head_dim
        convolution = state.convolution
        recurrent = state.recurrent
        if convolution is None:
            convolution = torch.zeros(
                b,
                2 * kw + vw,
                g.convolution_kernel_size - 1,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            recurrent = torch.zeros(
                b,
                g.value_heads,
                g.value_head_dim,
                g.key_head_dim,
                device=hidden.device,
                dtype=self.recurrent_state_dtype,
            )
        if recurrent is None or recurrent.dtype != self.recurrent_state_dtype:
            raise ValueError("GDN recurrent state differs from the selected storage policy")
        mixed, convolution = depthwise_causal_conv1d(
            self._linear(hidden, prefix + ".in_proj_qkv"),
            self.weights[prefix + ".conv1d.weight"].squeeze(1),
            convolution,
            activation="silu",
            output_dtype=hidden.dtype,
        )
        q, k, v = torch.split(mixed, [kw, kw, vw], dim=-1)
        assert recurrent is not None
        core, recurrent = gated_delta_net_recurrent(
            q.reshape(b, tokens, g.key_heads, g.key_head_dim),
            k.reshape(b, tokens, g.key_heads, g.key_head_dim),
            v.reshape(b, tokens, g.value_heads, g.value_head_dim),
            self._linear(hidden, prefix + ".in_proj_a"),
            self._linear(hidden, prefix + ".in_proj_b"),
            self.weights[prefix + ".A_log"],
            self.weights[prefix + ".dt_bias"],
            recurrent,
            l2_epsilon=1e-6,
            output_dtype=hidden.dtype,
        )
        gate = self._linear(hidden, prefix + ".in_proj_z").reshape_as(core)
        core = gated_rms_norm(
            core,
            gate,
            self.weights[prefix + ".norm.weight"],
            epsilon=t.rms_norm_epsilon,
            output_dtype=hidden.dtype,
            activation=g.output_gate_type,
        )
        return self._linear(core.flatten(2), prefix + ".out_proj"), replace(
            state, convolution=convolution, recurrent=recurrent
        )

    def _qsa(
        self, hidden: Tensor, state: Qwen4ExpLayerState, prefix: str, length: int,
        position_ids: Tensor | None = None,
    ) -> tuple[Tensor, Qwen4ExpLayerState]:
        t, a, index = self.config.text, self.config.text.full_attention, self.config.text.qsa
        b, tokens, _ = hidden.shape
        rotary = int(a.head_dim * t.mrope.partial_rotary_factor)

        def rope(positions: Tensor) -> tuple[Tensor, Tensor]:
            return interleaved_mrope_cos_sin(
                positions[None, None].expand(3, b, -1) if positions.ndim == 1 else positions,
                rotary_dim=rotary,
                mrope_section=t.mrope.section,
                rope_theta=t.mrope.rope_theta,
                output_dtype=hidden.dtype,
            )

        cos, sin = rope(torch.arange(length, length + tokens, device=hidden.device)
                        if position_ids is None else position_ids)
        position_history = None
        if position_ids is not None:
            previous_positions = state.index_positions
            if previous_positions is None:
                previous_positions = torch.arange(length, device=hidden.device)[
                    None, None
                ].expand(3, b, -1)
            position_history = torch.cat((previous_positions, position_ids), dim=-1)
        projected = self._linear(hidden, prefix + ".q_proj").reshape(
            b, tokens, a.query_heads, 2 * a.head_dim
        )
        query, gate = projected.chunk(2, -1)
        query = self._norm(query, prefix + ".q_norm").transpose(1, 2)
        key = self._norm(
            self._linear(hidden, prefix + ".k_proj").reshape(
                b, tokens, a.key_value_heads, a.head_dim
            ),
            prefix + ".k_norm",
        ).transpose(1, 2)
        value = (
            self._linear(hidden, prefix + ".v_proj")
            .reshape(b, tokens, a.key_value_heads, a.head_dim)
            .transpose(1, 2)
        )
        query, key = apply_partial_rotary(query, key, cos, sin, output_dtype=hidden.dtype)
        if state.key is not None:
            assert state.value is not None
            key, value = torch.cat((state.key, key), 2), torch.cat((state.value, value), 2)
        iq, ik = torch.split(
            self._linear(hidden, prefix + ".indexer.index_qk_proj"),
            [index.query_heads * index.head_dim, index.head_dim],
            -1,
        )
        iq = self._norm(
            iq.reshape(b, tokens, index.query_heads, index.head_dim),
            prefix + ".indexer.q_layernorm",
        ).transpose(1, 2)
        iq, _ = apply_partial_rotary(iq, iq, cos, sin, output_dtype=hidden.dtype)
        raw = ik if state.index_keys is None else torch.cat((state.index_keys, ik), 1)
        complete = raw.shape[1] // index.compression_ratio
        # Completed groups are stored, rounded state. Recomputing their norm
        # with a later batch shape would replace that state boundary with a new
        # floating-point evaluation.
        pooled = state.compressed_index_keys
        if pooled is None:
            pooled = raw[:, :0]
        previous_complete = pooled.shape[1]
        if complete > previous_complete:
            begin = previous_complete * index.compression_ratio
            end = complete * index.compression_ratio
            new_groups = (
                raw[:, begin:end]
                .reshape(b, complete - previous_complete, index.compression_ratio, index.head_dim)
                .float()
                .mean(2)
                .to(hidden.dtype)
            )
            new_groups = self._norm(new_groups, prefix + ".indexer.k_layernorm").unsqueeze(1)
            starts = (torch.arange(previous_complete, complete, device=hidden.device)
                      * index.compression_ratio)
            bc, bs = rope(starts if position_history is None else position_history[:, :, starts])
            new_groups, _ = apply_partial_rotary(
                new_groups, new_groups, bc, bs, output_dtype=hidden.dtype
            )
            pooled = torch.cat((pooled, new_groups.squeeze(1)), 1)
        visible = torch.arange(length + 1, length + tokens + 1, device=hidden.device)[None].expand(
            b, -1
        )
        selected = select_blocks(
            iq.transpose(1, 2),
            pooled,
            visible,
            compression=index.compression_ratio,
            token_budget=index.token_budget,
        )
        attention = getattr(self.operators, "selected_attention", attend_selected)
        output = attention(query, key, value, selected).transpose(1, 2)
        output = sigmoid_output_gate(output, gate, output_dtype=output.dtype).flatten(2)
        return self._linear(output, prefix + ".o_proj"), replace(
            state, key=key, value=value, index_keys=raw, compressed_index_keys=pooled,
            index_positions=position_history,
        )

    def _mlp(self, hidden: Tensor, prefix: str) -> Tensor:
        m = self.config.text.moe
        flat = hidden.reshape(-1, hidden.shape[-1])
        ids, weights = topk_routes(
            self._linear(flat, prefix + ".gate"),
            top_k=m.experts_per_token,
            normalize=m.normalize_topk_probability,
        )
        result = torch.zeros_like(flat)

        def expert(x: Tensor, name: str) -> Tensor:
            middle = silu_stored(self._linear(x, name + ".gate_proj")) * self._linear(
                x, name + ".up_proj"
            )
            return self._linear(middle, name + ".down_proj")

        for row in range(flat.shape[0]):
            for slot, eid in enumerate(ids[row].tolist()):
                if not self.expert_range[0] <= eid < self.expert_range[1]:
                    continue
                output = expert(flat[row : row + 1], prefix + f".experts.{eid}")
                result[row] = result[row] + output[0] * weights[row, slot]
        if self.reduce_experts is not None:
            result = self.reduce_experts(result)
        if m.shared_expert_intermediate_size:
            shared = expert(flat, prefix + ".shared_expert")
            result = result + shared * sigmoid_stored(
                self._linear(flat, prefix + ".shared_expert_gate")
            )
        return result.reshape_as(hidden)

    def _validate_state(self, state: Qwen4ExpReferenceState, tokens: Tensor) -> None:
        t, batch = self.config.text, tokens.shape[0]
        if type(state.length) is not int or state.length < 0 or len(state.layers) != t.layer_count:
            raise ValueError("reference state length/layer schema differs")
        for index, (kind, layer) in enumerate(zip(t.layer_schedule, state.layers, strict=True)):
            expected: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
            if state.length:
                if kind is AttentionKind.GATED_DELTA_NET:
                    g = t.gated_delta_net
                    channels = 2 * g.key_heads * g.key_head_dim + g.value_heads * g.value_head_dim
                    expected["convolution"] = (
                        (batch, channels, g.convolution_kernel_size - 1),
                        torch.bfloat16,
                    )
                    expected["recurrent"] = (
                        (batch, g.value_heads, g.value_head_dim, g.key_head_dim),
                        self.recurrent_state_dtype,
                    )
                else:
                    a = t.full_attention
                    expected["key"] = expected["value"] = (
                        (batch, a.key_value_heads, state.length, a.head_dim),
                        torch.bfloat16,
                    )
                    expected["index_keys"] = ((batch, state.length, t.qsa.head_dim), torch.bfloat16)
                    expected["compressed_index_keys"] = (
                        (batch, state.length // t.qsa.compression_ratio, t.qsa.head_dim),
                        torch.bfloat16,
                    )
                    if layer.index_positions is not None:
                        expected["index_positions"] = ((3, batch, state.length), torch.int64)
                if index + 1 in t.ple.layer_ids:
                    p = t.ple
                    expected["ple_history"] = ((batch, p.ngram_size - 1), torch.int64)
                    expected["ple_convolution"] = (
                        (
                            batch,
                            (p.convolution_kernel_size - 1) * p.ngram_size,
                            t.hidden_size * t.hyperconnection.stream_count,
                        ),
                        torch.bfloat16,
                    )
            for name in Qwen4ExpLayerState.__dataclass_fields__:
                value = getattr(layer, name)
                if name not in expected:
                    if value is not None:
                        raise ValueError(f"layer {index} contains undeclared {name} state")
                elif not isinstance(value, Tensor) or (
                    (tuple(value.shape), value.dtype) != expected[name]
                    or value.device != tokens.device
                ):
                    raise ValueError(f"layer {index} has missing or invalid {name} state")

    def forward(
        self, tokens: Tensor, state: Qwen4ExpReferenceState | None = None, *,
        input_embeddings: Tensor | None = None, position_ids: Tensor | None = None,
    ) -> tuple[Tensor, Qwen4ExpReferenceState]:
        logits, state, _ = self.forward_with_feedback(
            tokens, state, input_embeddings=input_embeddings, position_ids=position_ids,
        )
        return logits, state

    def forward_with_feedback(
        self, tokens: Tensor, state: Qwen4ExpReferenceState | None = None, *,
        input_embeddings: Tensor | None = None, position_ids: Tensor | None = None,
    ) -> tuple[Tensor, Qwen4ExpReferenceState, Tensor]:
        """Keep the pre-final-mixer residual separate from vocabulary logits."""
        if tokens.ndim != 2 or tokens.dtype != torch.int64 or not tokens.numel():
            raise ValueError("reference tokens must be nonempty INT64 [batch,time]")
        t = self.config.text
        if bool(((tokens < 0) | (tokens >= t.vocabulary_size)).any()):
            raise ValueError("token ID lies outside the checkpoint vocabulary")
        if state is None:
            state = Qwen4ExpReferenceState(
                0, tuple(Qwen4ExpLayerState() for _ in range(t.layer_count))
            )
        if (
            len(state.layers) != t.layer_count
            or state.length + tokens.shape[1] > t.max_position_embeddings
        ):
            raise ValueError("reference state layer/context extent differs")
        self._validate_state(state, tokens)
        if position_ids is None and any(
            layer.index_positions is not None for layer in state.layers
        ):
            raise ValueError("multimodal reference continuation requires explicit positions")
        if position_ids is not None and (
            position_ids.shape != (3, *tokens.shape) or position_ids.dtype != torch.int64
            or position_ids.device != tokens.device
            or bool(((position_ids < 0) | (position_ids >= t.max_position_embeddings)).any())
        ):
            raise ValueError("reference position axes differ from the declared context")
        hidden = (self.operators.embedding(
            tokens, self.weights["model.language_model.embed_tokens.weight"], dtype=torch.bfloat16
        ) if input_embeddings is None else input_embeddings)
        if (hidden.shape != (*tokens.shape, t.hidden_size) or hidden.dtype != torch.bfloat16
                or hidden.device != tokens.device or not bool(torch.isfinite(hidden).all())):
            raise ValueError("reference input embeddings differ from the declared feature boundary")
        hidden = (
            hidden.unsqueeze(-2)
            .expand(*hidden.shape[:-1], t.hyperconnection.stream_count, t.hidden_size)
            .flatten(-2)
            .contiguous()
        )
        layers = []
        for layer, (kind, previous) in enumerate(zip(t.layer_schedule, state.layers, strict=True)):
            prefix = f"model.language_model.layers.{layer}"
            if layer + 1 in t.ple.layer_ids:
                hidden, previous = self._ple(hidden, tokens, previous, prefix + ".ple")
            mixed, normalized = self._mix(hidden, prefix + ".attn_hyper_connection")
            if kind is AttentionKind.GATED_DELTA_NET:
                block, next_state = self._gdn(mixed, previous, prefix + ".linear_attn")
            else:
                block, next_state = self._qsa(mixed, previous, prefix + ".self_attn", state.length,
                                            position_ids)
            hidden = self._inject(hidden, block, normalized, prefix + ".attn_hyper_connection")
            mixed, normalized = self._mix(hidden, prefix + ".mlp_hyper_connection")
            hidden = self._inject(
                hidden,
                self._mlp(mixed, prefix + ".mlp"),
                normalized,
                prefix + ".mlp_hyper_connection",
            )
            layers.append(next_state)
        mixed, _ = self._mix(hidden, "model.language_model.hyper_connection_mixer")
        return self._linear(mixed, "lm_head"), Qwen4ExpReferenceState(
            state.length + tokens.shape[1], tuple(layers)
        ), hidden


class Qwen4ExpTextGraphBinder(Protocol):
    """Runtime-provided fixed-address bindings, without model traversal authority.

    Prefixes are logical checkpoint module names, never physical rank/device
    assignments. A returned step must be completely bound before capture;
    collective/cache ownership and numerical admission remain runtime duties.
    """

    def bind(
        self, operation: str, weight_prefix: str, *, layer_id: int | None
    ) -> Callable[[], None]: ...


@dataclass(frozen=True, slots=True)
class Qwen4ExpBoundTextGraph:
    """Model-owned MTP0 traversal over already-bound shared operations.

    Construction or launch does not grant performance admission. The executor
    must capture and qualify this whole sequence under one topology-wide lease.
    """

    steps: tuple[tuple[str, Callable[[], None]], ...]
    logits_step: Callable[[], None] | None = None

    def launch(self) -> None:
        for _, launch in self.steps:
            launch()

    def launch_feedback(self) -> None:
        """Advance the same state and feedback while leaving logits untouched."""
        for _, launch in self.steps:
            if launch is not self.logits_step:
                launch()


def bind_qwen4_exp_text_graph(
    config: Qwen4ExpConfig,
    binder: Qwen4ExpTextGraphBinder,
    *,
    owned_layer_ids: tuple[int, ...] | None = None,
    first_stage: bool = True,
    last_stage: bool = True,
) -> Qwen4ExpBoundTextGraph:
    """Bind checkpoint semantic order; never choose a CUDA kernel or topology.

    Incoming/outgoing PP buffers already contain the expanded residual streams.
    Runtime owns transfers. PLE precedes the layer's attention mixer; each mixer
    saves its normalized residual until the corresponding injection. There is
    no extra final RMSNorm: the checkpoint's final mixer feeds lm_head directly.
    """
    layers = tuple(range(config.text.layer_count)) if owned_layer_ids is None else owned_layer_ids
    if (
        type(layers) is not tuple
        or not layers
        or layers != tuple(range(layers[0], layers[-1] + 1))
        or layers[0] < 0
        or layers[-1] >= config.text.layer_count
        or type(first_stage) is not bool
        or type(last_stage) is not bool
        or first_stage != (layers[0] == 0)
        or last_stage != (layers[-1] == config.text.layer_count - 1)
    ):
        raise ValueError("text graph layers must be one contiguous declared pipeline stage")
    steps: list[tuple[str, Callable[[], None]]] = []

    def add(operation: str, prefix: str, layer: int | None) -> None:
        launch = binder.bind(operation, prefix, layer_id=layer)
        if not callable(launch):
            raise TypeError(f"text graph binding {operation!r} did not return a fixed launch")
        steps.append((f"{prefix}:{operation}", launch))

    if first_stage:
        add("embedding", "model.language_model.embed_tokens", None)
    for layer in layers:
        prefix = f"model.language_model.layers.{layer}"
        if layer + 1 in config.text.ple.layer_ids:
            add("ple", prefix + ".ple", layer)
        add("hc_mix", prefix + ".attn_hyper_connection", layer)
        if config.text.layer_schedule[layer] is AttentionKind.GATED_DELTA_NET:
            add("gdn", prefix + ".linear_attn", layer)
        else:
            add("qsa", prefix + ".self_attn", layer)
        add("hc_inject", prefix + ".attn_hyper_connection", layer)
        add("hc_mix", prefix + ".mlp_hyper_connection", layer)
        add("moe", prefix + ".mlp", layer)
        add("hc_inject", prefix + ".mlp_hyper_connection", layer)
    if last_stage:
        add("hc_mix", "model.language_model.hyper_connection_mixer", None)
        add("lm_head", "lm_head", None)
    return Qwen4ExpBoundTextGraph(tuple(steps), steps[-1][1] if last_stage else None)


def qwen4_exp_cache_layout(
    config: Qwen4ExpConfig,
    topology: ParallelTopology,
    rank: int,
    *,
    page_size: int,
    dtype: torch.dtype = torch.bfloat16,
    recurrent_state_dtype: torch.dtype | None = None,
) -> tuple[KVLayout, tuple[StateSlotSpec, ...]]:
    """Declare rank-local state using the checkpoint's original layer identities.

    QSA compressed keys share main KV page ownership. Raw keys and their exact
    rotary positions share request checkpoint slots with GDN and lexical history.
    Each layer is a separate contiguous field for fixed-address block bindings.
    This schema grants neither graph qualification nor runtime admission.
    """
    if not 0 <= rank < len(topology.placements):
        raise ValueError("cache rank is outside the declared topology")
    text = config.text
    tp, pp = topology.tp_size, topology.pp_size
    gdn, attention, index = text.gated_delta_net, text.full_attention, text.qsa
    from cubic_serve.models.parallel_weights import IndexerTP
    indexer_plan = IndexerTP(index.query_heads, index.head_dim, index.compression_ratio,
                             tp, topology.placements[rank].tensor_rank)
    local_groups = indexer_plan.cache_page_groups(page_size)
    state_dtype = recurrent_state_dtype or getattr(torch, gdn.recurrent_state_dtype)
    if state_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("GDN recurrent state requires BF16 or FP32 storage")
    if (
        text.layer_count % pp
        or gdn.key_heads % tp
        or gdn.value_heads % tp
        or attention.query_heads % tp
        or (
            attention.key_value_heads % tp
            if tp <= attention.key_value_heads
            else tp % attention.key_value_heads
        )
        or type(page_size) is not int
        or page_size <= 0
        or page_size % index.compression_ratio
        or dtype not in (torch.bfloat16, torch.float16)
    ):
        raise ValueError("cache topology, page geometry or activation dtype is incompatible")
    stage = topology.placements[rank].pipeline_stage
    count = text.layer_count // pp
    layers = range(stage * count, (stage + 1) * count)
    specs: list[StateSlotSpec] = []
    page_fields: list[PageFieldSpec] = []
    full: list[int] = []
    for layer in layers:
        prefix = f"model.language_model.layers.{layer}.state."
        if text.layer_schedule[layer] is AttentionKind.GATED_DELTA_NET:
            keys, values = gdn.key_heads // tp, gdn.value_heads // tp
            specs.extend(
                (
                    StateSlotSpec(
                        prefix + LayerStateKind.GDN_CONVOLUTION.value,
                        dtype,
                        (
                            2 * keys * gdn.key_head_dim + values * gdn.value_head_dim,
                            gdn.convolution_kernel_size - 1,
                        ),
                    ),
                    StateSlotSpec(
                        prefix + LayerStateKind.GDN_RECURRENT_MATRIX.value,
                        state_dtype,
                        (values, gdn.value_head_dim, gdn.key_head_dim),
                    ),
                )
            )
        else:
            full.append(layer)
            specs.extend(
                (
                    StateSlotSpec(
                        prefix + LayerStateKind.QSA_RAW_KEY_RING.value,
                        dtype,
                        (index.compression_ratio, indexer_plan.key_width),
                    ),
                    StateSlotSpec(
                        prefix + LayerStateKind.QSA_POSITION_RING.value,
                        torch.int64,
                        (index.compression_ratio, 3),
                    ),
                )
            )
            page_fields.append(
                PageFieldSpec(
                    prefix + LayerStateKind.QSA_COMPRESSED_KEY_CACHE.value,
                    dtype,
                    (local_groups, index.head_dim),
                )
            )
        if layer + 1 in text.ple.layer_ids:
            ple = text.ple
            if text.eos_token_id is None:
                raise ValueError("lexical cache initialization requires a declared EOS token")
            specs.extend(
                (
                    StateSlotSpec(
                        prefix + LayerStateKind.PLE_NGRAM_TOKEN_CONTEXT.value,
                        torch.int64,
                        (ple.ngram_size - 1,),
                        initial_value=text.eos_token_id,
                    ),
                    StateSlotSpec(
                        prefix + LayerStateKind.PLE_CONVOLUTION.value,
                        dtype,
                        (
                            (ple.convolution_kernel_size - 1) * ple.ngram_size,
                            text.hidden_size * text.hyperconnection.stream_count,
                        ),
                    ),
                )
            )
    layout = KVLayout(
        tuple(full),
        page_size,
        max(1, attention.key_value_heads // tp),
        attention.head_dim,
        dtype=dtype,
        page_fields=tuple(sorted(page_fields, key=lambda field: field.field_id)),
    )
    return layout, tuple(sorted(specs, key=lambda spec: spec.state_id))


@dataclass(frozen=True, slots=True)
class Qwen4ExpGraphResources:
    """Rank executor's fixed operands; all allocation and leases remain external.

    Source indices retain each operator's contract. GDN reads a valid, cleared
    slot for a fresh sequence; it does not accept a negative source sentinel.
    PLE/QSA support negative sources for empty history. Separate per-field maps
    prevent silently applying one initialization convention to every operator.
    """

    hidden: Tensor
    tokens: Tensor
    segment_lengths: Tensor
    row_activity: Tensor
    active_rows: Tensor
    positions: Tensor
    logical_positions: Tensor
    states: Mapping[str, Tensor]
    state_sources: Mapping[str, Tensor]
    state_destinations: Mapping[str, Tensor]
    key: Tensor
    value: Tensor
    written: Tensor
    kv_layer_ids: tuple[int, ...]
    page_fields: Mapping[str, Tensor]
    page_table: Tensor
    page_destinations: Tensor
    token_offsets: Tensor
    visible_lengths: Tensor
    cosine_table: Tensor
    sine_table: Tensor
    rope_evidence: MRoPETableEvidence
    ple_shards: Mapping[int, tuple[tuple[int, Tensor | QuantizedMatrix], ...]]
    ple_scales: Mapping[int, Tensor]
    embedding: Callable[[], None]
    first_expert: int
    local_expert_count: int
    indexer_plan: object | None = None
    indexer_group: object | None = None
    reduce_attention: Callable[[Tensor], object] | None = None
    reduce_lexical: Callable[[Tensor], object] | None = None
    reduce_routes: Callable[[Tensor], None] | None = None
    reduce_shared: Callable[[Tensor], object] | None = None
    reduce_experts: Callable[[Tensor], object] | None = None


class Qwen4ExpOperationBinder:
    """Map checkpoint semantic names to shared, format-opaque text blocks.

    Input weights already have their declared rank ownership. This binder does
    not partition or migrate experts, allocate persistent state, or communicate
    implicitly. It must be called in model traversal order at startup.
    """

    def __init__(
        self,
        config: Qwen4ExpConfig,
        weights: Qwen4ExpWeights,
        helper: TextGraphBindings,
        resources: Qwen4ExpGraphResources,
    ) -> None:
        self.config, self.weights, self.helper, self.resources = config, weights, helper, resources
        text = config.text
        if (
            resources.hidden.ndim != 2
            or resources.hidden.shape[1] != text.hidden_size * text.hyperconnection.stream_count
            or resources.first_expert < 0
            or resources.local_expert_count <= 0
            or resources.first_expert + resources.local_expert_count > text.moe.expert_count
        ):
            raise ValueError("bound hidden or original expert ownership differs from the model")
        self.mixed: Tensor | None = None
        self.normalized: Tensor | None = None
        self.injection_logits: Tensor | None = None
        self._pending_injection: tuple[list[Callable[[], object]], Tensor, Tensor] | None = None
        self.update: Tensor | None = None
        self.logits: Tensor | None = None

    def _tensor(self, name: str) -> Tensor:
        tensor = self.weights[name]
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be an ordinary semantic tensor")
        return tensor

    def _state(self, layer: int, kind: LayerStateKind) -> tuple[Tensor, Tensor, Tensor]:
        name = f"model.language_model.layers.{layer}.state.{kind.value}"
        r = self.resources
        return r.states[name], r.state_sources[name], r.state_destinations[name]

    def bind(
        self, operation: str, weight_prefix: str, *, layer_id: int | None
    ) -> Callable[[], None]:
        r, h, t, p = self.resources, self.helper, self.config.text, weight_prefix
        pending, self._pending_injection = self._pending_injection, None
        streams = t.hyperconnection.stream_count
        if operation == "embedding":
            return r.embedding
        if operation == "hc_mix":
            self.injection_logits = None if layer_id is None else h.buffer(
                "hc.injection_logits", r.hidden.shape[0], streams
            )
            block, self.normalized = h.mix(
                r.hidden,
                self._tensor(p + ".hc_norm.weight"),
                self.weights[p + ".input_mix_weight_down.weight"],
                self.weights[p + ".input_mix_weight_up.weight"],
                streams=streams,
                epsilon=t.rms_norm_epsilon,
                injection=None if self.injection_logits is None else (
                    self.weights[p + ".block_inject_weight.weight"], self.injection_logits
                ),
                residual_update=None if pending is None else (pending[1], pending[2]),
            )
            if pending is not None:
                # Bind-time adjacency only. Any intervening operation (including
                # PLE or a pipeline boundary) retains the standalone injection.
                pending[0].clear()
            self.mixed = block.output
            self.update = None
            return block.launch
        if operation == "hc_inject":
            if self.update is None or self.normalized is None or self.injection_logits is None:
                raise ValueError("HC injection requires a preceding mixed block")
            block = h.inject(
                r.hidden,
                self.update,
                self.normalized,
                self.weights[p + ".block_inject_weight.weight"],
                streams=streams,
                logits=self.injection_logits,
            )
            calls: list[Callable[[], object]] = [block.launch]
            self._pending_injection = calls, self.update, self.injection_logits

            def inject() -> None:
                for call in calls:
                    call()

            return inject
        if operation == "lm_head":
            if self.mixed is None:
                raise ValueError("lm_head requires the final HC mixer")
            weight = self.weights[p + ".weight"]
            self.logits = h.buffer("model.logits", r.hidden.shape[0], weight.shape[0])
            call = h.linear(self.mixed, weight, self.logits)

            def launch() -> None:
                call()

            return launch
        if layer_id is None:
            raise ValueError("layer operation requires its original layer id")
        if operation == "ple":
            lexical, source, destination = self._state(
                layer_id, LayerStateKind.PLE_NGRAM_TOKEN_CONTEXT
            )
            conv, conv_source, conv_destination = self._state(
                layer_id, LayerStateKind.PLE_CONVOLUTION
            )
            if t.eos_token_id is None:
                raise ValueError("lexical execution requires a declared EOS token")
            base = p + ".ple_embedding"
            return h.ple(
                r.hidden,
                r.tokens,
                shards=r.ple_shards[layer_id],
                table_scale=r.ple_scales[layer_id],
                multipliers=self._tensor(base + ".layer_multipliers"),
                vocab_sizes=self._tensor(base + ".ngram_heads_vocab_sizes"),
                offsets=self._tensor(base + ".ngram_heads_offsets"),
                eos_token_id=t.eos_token_id,
                hash_state=lexical,
                hash_sources=source,
                hash_destinations=destination,
                conv_state=conv,
                conv_sources=conv_source,
                conv_destinations=conv_destination,
                segment_lengths=r.segment_lengths,
                row_activity=r.row_activity,
                key_projection=self.weights[p + ".key_proj.weight"],
                value_projection=self.weights[p + ".value_proj.weight"],
                key_norm=self._tensor(p + ".norm_key.weight"),
                query_norm=self._tensor(p + ".norm_query.weight"),
                conv_norm=self._tensor(p + ".norm_conv.weight"),
                conv_weight=self._tensor(p + ".conv1d.weight"),
                streams=streams,
                dilation=t.ple.ngram_size,
                epsilon=t.rms_norm_epsilon,
                reduce_embedding=r.reduce_lexical,
            ).launch
        if self.mixed is None:
            raise ValueError("attention and feed-forward require a preceding HC mixer")
        if operation == "gdn":
            conv, cs, cd = self._state(layer_id, LayerStateKind.GDN_CONVOLUTION)
            recurrence, rs, rd = self._state(layer_id, LayerStateKind.GDN_RECURRENT_MATRIX)
            g = t.gated_delta_net
            if recurrence.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError("GDN recurrent state requires BF16 or FP32 storage")
            heads = recurrence.shape[1]
            if heads < g.value_heads and r.reduce_attention is None:
                raise ValueError("partitioned GDN output requires its declared TP reduction")
            packed = conv.shape[1]
            key_width = packed - heads * g.value_head_dim
            if key_width <= 0 or key_width % (2 * g.key_head_dim):
                raise ValueError("local GDN state does not partition into original heads")
            block = h.gdn(
                self.mixed,
                {
                    name: self.weights[p + "." + leaf + ".weight"]
                    for name, leaf in (
                        ("qkv", "in_proj_qkv"),
                        ("a", "in_proj_a"),
                        ("b", "in_proj_b"),
                        ("z", "in_proj_z"),
                        ("out", "out_proj"),
                    )
                },
                conv_weight=self._tensor(p + ".conv1d.weight"),
                decay=self._tensor(p + ".A_log"),
                bias=self._tensor(p + ".dt_bias"),
                norm=self._tensor(p + ".norm.weight"),
                conv_state=conv,
                recurrent_state=recurrence,
                conv_sources=cs,
                conv_destinations=cd,
                recurrent_sources=rs,
                recurrent_destinations=rd,
                segment_lengths=r.segment_lengths,
                row_activity=r.row_activity,
                key_heads=key_width // (2 * g.key_head_dim),
                key_dim=g.key_head_dim,
                value_heads=heads,
                value_dim=g.value_head_dim,
                epsilon=t.rms_norm_epsilon,
                output_gate=g.output_gate_type,
                reduce_output=r.reduce_attention,
            )
        elif operation == "qsa":
            raw, source, destination = self._state(layer_id, LayerStateKind.QSA_RAW_KEY_RING)
            positions, ps, pd = self._state(layer_id, LayerStateKind.QSA_POSITION_RING)
            # Paired raw-key and position histories must use identical slot maps.
            # Pointer identity is checked before capture; no device readback here.
            if ps.data_ptr() != source.data_ptr() or pd.data_ptr() != destination.data_ptr():
                raise ValueError("QSA key and exact-position histories require shared slot maps")
            a, index = t.full_attention, t.qsa
            query = self.weights[p + ".q_proj.weight"]
            if query.shape[0] % (2 * a.head_dim):
                raise ValueError("local query/gate projection splits a declared head")
            if query.shape[0] < 2 * a.query_heads * a.head_dim and r.reduce_attention is None:
                raise ValueError("partitioned attention output requires its declared TP reduction")
            block = h.qsa(
                self.mixed,
                {
                    name: self.weights[p + "." + leaf + ".weight"]
                    for name, leaf in (
                        ("q", "q_proj"),
                        ("k", "k_proj"),
                        ("v", "v_proj"),
                        ("out", "o_proj"),
                        ("index_qk", "indexer.index_qk_proj"),
                    )
                },
                {
                    name: self._tensor(p + "." + leaf + ".weight")
                    for name, leaf in (
                        ("q", "q_norm"),
                        ("k", "k_norm"),
                        ("index_q", "indexer.q_layernorm"),
                        ("index_k", "indexer.k_layernorm"),
                    )
                },
                query_heads=query.shape[0] // (2 * a.head_dim),
                head_dim=a.head_dim,
                index_heads=index.query_heads,
                indexer_plan=r.indexer_plan,
                indexer_group=r.indexer_group,
                compression=index.compression_ratio,
                token_budget=index.token_budget,
                epsilon=t.rms_norm_epsilon,
                key_cache=r.key,
                value_cache=r.value,
                written=r.written,
                cache_layer=r.kv_layer_ids.index(layer_id),
                compressed_cache=r.page_fields[
                    f"model.language_model.layers.{layer_id}.state."
                    + LayerStateKind.QSA_COMPRESSED_KEY_CACHE.value
                ],
                raw_history=raw,
                position_history=positions,
                sources=source,
                destinations=destination,
                segment_lengths=r.segment_lengths,
                row_activity=r.row_activity,
                logical_positions=r.logical_positions,
                positions=r.positions,
                page_table=r.page_table,
                page_destinations=r.page_destinations,
                token_offsets=r.token_offsets,
                visible_lengths=r.visible_lengths,
                cosine_table=r.cosine_table,
                sine_table=r.sine_table,
                rope_evidence=r.rope_evidence,
                reduce_output=r.reduce_attention,
            )
        elif operation == "moe":
            if not t.moe.normalize_topk_probability:
                raise ValueError("shared routed block requires normalized selected probabilities")
            if r.local_expert_count < t.moe.expert_count and r.reduce_experts is None:
                raise ValueError("partitioned expert ownership requires its declared EP reduction")
            expert_down = self.weights[f"{p}.experts.{r.first_expert}.down_proj.weight"]
            if expert_down.shape[1] < t.moe.expert_intermediate_size and r.reduce_routes is None:
                raise ValueError(
                    "partitioned expert intermediate requires its declared TP reduction"
                )
            shared_down = self.weights[p + ".shared_expert.down_proj.weight"]
            if (
                shared_down.shape[1] < t.moe.shared_expert_intermediate_size
                and r.reduce_shared is None
            ):
                raise ValueError("partitioned shared expert requires its declared TP reduction")
            experts = range(r.first_expert, r.first_expert + r.local_expert_count)
            block = h.moe(
                self.mixed,
                router=self.weights[p + ".gate.weight"],
                gate=tuple(self.weights[f"{p}.experts.{e}.gate_proj.weight"] for e in experts),
                up=tuple(self.weights[f"{p}.experts.{e}.up_proj.weight"] for e in experts),
                down=tuple(self.weights[f"{p}.experts.{e}.down_proj.weight"] for e in experts),
                shared_gate=self.weights[p + ".shared_expert.gate_proj.weight"],
                shared_up=self.weights[p + ".shared_expert.up_proj.weight"],
                shared_down=self.weights[p + ".shared_expert.down_proj.weight"],
                shared_router=self.weights[p + ".shared_expert_gate.weight"],
                activity=r.row_activity,
                active_rows=r.active_rows,
                first_expert=r.first_expert,
                top_k=t.moe.experts_per_token,
                reduce_routes=r.reduce_routes,
                reduce_shared=r.reduce_shared,
                reduce_experts=r.reduce_experts,
            )
        else:
            raise ValueError(f"unknown Flash-Next operation {operation!r}")
        self.update = block.output
        return block.launch


def validate_mtp_metadata(config: Qwen4ExpConfig) -> None:
    mtp = config.text.mtp
    if (
        mtp.hidden_layer_count != 1
        or mtp.layer_schedule != (AttentionKind.FULL_ATTENTION,)
        or mtp.dedicated_embeddings
        or mtp.hidden_state_source_layer not in (None, config.text.layer_count - 1)
    ):
        raise ValueError("unsupported checkpoint MTP feedback or layer contract")


@dataclass(frozen=True, slots=True)
class Qwen4ExpMTPReferenceStep:
    logits: Tensor
    feedback: Tensor
    state: Qwen4ExpReferenceState


class Qwen4ExpMTPReference:
    """Independent original operators; no packed format or serving authority.

    Normalize hidden across the complete HC width, then apply the same HxH
    projection to each branch. The folded final mixer feeds the shared LM head;
    the materialized pre-mixer residual feeds the next proposal. PLE is absent.
    """

    def __init__(self, target: Qwen4ExpReferenceModel, *, spec: Any) -> None:
        validate_mtp_metadata(target.config)
        self.target = target
        text = target.config.text
        self.spec = spec
        if (spec.method != "mtp" or spec.vocabulary_size != text.vocabulary_size
                or spec.feedback_width != text.hidden_size * text.hyperconnection.stream_count):
            raise ValueError("draft contract differs from checkpoint feedback semantics")
        # State validation uses the declared draft traversal, without mutating
        # the target config or loading a second checkpoint.
        draft_config = replace(target.config, text=replace(
            text, layer_count=1, layer_schedule=text.mtp.layer_schedule,
            ple=replace(text.ple, layer_ids=()),
        ))
        self._state_model = Qwen4ExpReferenceModel(
            draft_config, target.weights, target.operators,
            recurrent_state_dtype=target.recurrent_state_dtype,
            expert_range=target.expert_range, reduce_experts=target.reduce_experts,
        )

    def forward(
        self, tokens: Tensor, feedback: Tensor | None,
        state: Qwen4ExpReferenceState | None = None,
        *, position_ids: Tensor | None = None, input_embeddings: Tensor | None = None,
    ) -> Qwen4ExpMTPReferenceStep:
        t, m = self.target.config.text, self.target
        if tokens.ndim != 2 or tokens.dtype is not torch.int64 or not tokens.numel():
            raise ValueError("MTP tokens require nonempty INT64 [batch,time]")
        if bool(((tokens < 0) | (tokens >= t.vocabulary_size)).any()):
            raise ValueError("MTP token lies outside the checkpoint vocabulary")
        if (
            feedback is None or feedback.shape != (*tokens.shape, self.spec.feedback_width)
            or feedback.dtype is not torch.bfloat16 or feedback.device != tokens.device
            or not bool(torch.isfinite(feedback).all())
        ):
            raise ValueError("MTP input requires the declared multi-stream feedback")
        state = state or Qwen4ExpReferenceState(0, (Qwen4ExpLayerState(),))
        self._state_model._validate_state(state, tokens)
        if state.length + tokens.shape[1] > t.max_position_embeddings:
            raise ValueError("MTP context exceeds the checkpoint limit")
        if position_ids is not None and (
            position_ids.shape != (3, *tokens.shape) or position_ids.dtype is not torch.int64
            or position_ids.device != tokens.device
            or bool(((position_ids < 0) | (position_ids >= t.max_position_embeddings)).any())
        ):
            raise ValueError("MTP position axes differ from the checkpoint")
        if position_ids is None and state.layers[0].index_positions is not None:
            raise ValueError("MTP positioned continuation requires explicit axes")
        if input_embeddings is None:
            embedding = m.operators.embedding(
                tokens, m.weights["model.language_model.embed_tokens.weight"], dtype=torch.bfloat16
            )
        else:
            if (
                input_embeddings.shape != (*tokens.shape, t.hidden_size)
                or input_embeddings.dtype is not torch.bfloat16
                or input_embeddings.device != tokens.device
                or not bool(torch.isfinite(input_embeddings).all())
            ):
                raise ValueError("MTP embedding differs from the shared input feature boundary")
            embedding = input_embeddings
        projected_embedding = m._linear(m._norm(embedding, "mtp.pre_fc_norm_embedding"),
                                        "mtp.fc_embedding")
        normalized = m._norm(feedback, "mtp.pre_fc_norm_hidden")
        projected = m._linear(normalized.reshape(
            *tokens.shape, t.hyperconnection.stream_count, t.hidden_size), "mtp.fc_hidden")
        hidden = (projected + projected_embedding.unsqueeze(-2)).flatten(-2)
        prefix = "mtp.layers.0"
        mixed, norm = m._mix(hidden, prefix + ".attn_hyper_connection")
        block, next_layer = m._qsa(mixed, state.layers[0], prefix + ".self_attn",
                                  state.length, position_ids)
        hidden = m._inject(hidden, block, norm, prefix + ".attn_hyper_connection")
        mixed, norm = m._mix(hidden, prefix + ".mlp_hyper_connection")
        hidden = m._inject(hidden, m._mlp(mixed, prefix + ".mlp"), norm,
                           prefix + ".mlp_hyper_connection")
        sample_hidden, _ = m._mix(hidden, "mtp.hyper_connection_mixer")
        logits = m._linear(sample_hidden, "lm_head")
        return Qwen4ExpMTPReferenceStep(logits, hidden, Qwen4ExpReferenceState(
            state.length + tokens.shape[1], (next_layer,)))


def build_qwen4_exp_draft_provider(config, *, method: str, max_depth: int):
    """Bind the existing model draft construction; engine policy stays shared."""
    from functools import partial

    from cubic_serve.executor.draft import DraftProvider, DraftSpec
    from cubic_serve.models.builtin import BuiltinModelConfigError

    if method != "mtp":
        raise BuiltinModelConfigError(
            "model has no qualified shared-graph draft construction binding"
        )
    from cubic_serve.executor.mtp_runner import build_flash_mtp_stage
    from cubic_serve.models.qwen4_exp.model import (
        Qwen4ExpMTPReference,
        validate_mtp_metadata,
    )

    validate_mtp_metadata(config)
    t = config.text
    spec = DraftSpec(
        method, t.vocabulary_size, t.hidden_size * t.hyperconnection.stream_count, max_depth
    )
    return DraftProvider(
        spec,
        partial(build_flash_mtp_stage, config, max_depth=max_depth),
        partial(Qwen4ExpMTPReference, spec=spec),
        ("mtp.fc_embedding.weight", "mtp.fc_hidden.weight"),
    )
