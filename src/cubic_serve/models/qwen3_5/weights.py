"""Qwen3.5 checkpoint schema, executable slice, and TP/PP weight semantics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from cubic_serve._compat import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import torch
from safetensors import safe_open
from torch import Tensor

from cubic_serve.contracts.execution import (
    GraphAssignment,
    ParallelTopology,
    PartitionPlan,
    PipelineEdge,
)
from cubic_serve.contracts.model import ScalarDType
from cubic_serve.models.checkpoint import CheckpointInventory
from cubic_serve.models.parallel_weights import AxisSpan as AxisSpan
from cubic_serve.models.parallel_weights import (
    LogicalWeightSlice,
    RankLocalTensors,
    SafeTensorReader,
    confined_checkpoint_path,
    equal_axis_span,
    materialize_axis_spans,
)
from cubic_serve.models.quantized_checkpoint import LogicalWeight
from cubic_serve.models.qwen3_5.model import (
    AttentionKind,
    Qwen3_5ArchitecturePlan,
    Qwen3_5Config,
    Qwen3_5GraphConfig,
)
from cubic_serve.models.weight_schema import (
    CheckpointWeightPlan,
    TensorHeader,
    WeightArchitectureIdentity,
    WeightNodeBinding,
    WeightSchemaCoverage,
    WeightSchemaError,
    _issue_weight_plan,
    materialize_headers,
    tensor_headers_from_inventory,
)
from cubic_serve.models.weights import FullBF16WeightPolicy

if TYPE_CHECKING:
    from cubic_serve.models.qwen3_5_moe.config import Qwen3_5MoeConfig

Qwen3_5WeightPlan = CheckpointWeightPlan

__all__ = (
    "Qwen3_5WeightPlan",
    "TensorHeader",
    "WeightSchemaError",
    "plan_qwen3_5_checkpoint_weights",
    "plan_qwen3_5_weight_headers",
    "qwen3_5_mtp_weight_policy",
    "qwen3_5_text_weight_policy",
)


@dataclass(frozen=True, slots=True)
class _ExpectedHeader:
    dtype: str
    shape: tuple[int, ...]
    node_id: str
    partition: str


class _AddHeader(Protocol):
    def __call__(
        self,
        name: str,
        shape: tuple[int, ...],
        node_id: str,
        partition: str,
        dtype: str = "BF16",
    ) -> None: ...


def plan_qwen3_5_checkpoint_weights(
    config: Qwen3_5Config, inventory: CheckpointInventory
) -> Qwen3_5WeightPlan:
    """Plan an already-attested inventory without reading tensor payloads."""

    return _plan(
        config,
        tensor_headers_from_inventory(inventory),
        inventory=inventory,
    )


def plan_qwen3_5_weight_headers(
    config: Qwen3_5Config, headers: Iterable[TensorHeader]
) -> Qwen3_5WeightPlan:
    """Pure header-record API for CI fixtures and pre-attestation validation."""

    return _plan(config, materialize_headers(headers), inventory=None)


def qwen3_5_text_weight_policy(plan: Qwen3_5WeightPlan) -> FullBF16WeightPolicy:
    """Close the executable text slice without claiming the unloaded vision tower."""

    if not isinstance(plan, Qwen3_5WeightPlan) or plan.model_type != "qwen3_5":
        raise TypeError("qwen3_5 text policy requires a validated qwen3_5 weight plan")
    names = tuple(sorted((*plan.base_language_names, *plan.lm_head_names)))
    selected = set(names)
    return FullBF16WeightPolicy(
        expected_names=names,
        fp32_stability_names=tuple(name for name in plan.fp32_stability_names if name in selected),
        int64_metadata_names=tuple(name for name in plan.int64_metadata_names if name in selected),
    )


def qwen3_5_mtp_weight_policy(plan: Qwen3_5WeightPlan) -> FullBF16WeightPolicy:
    """Close the text graph plus its explicitly enabled MTP extension."""

    if not isinstance(plan, Qwen3_5WeightPlan) or plan.model_type != "qwen3_5":
        raise TypeError("qwen3_5 MTP policy requires a validated qwen3_5 weight plan")
    names = tuple(
        sorted((*plan.base_language_names, *plan.lm_head_names, *plan.mtp_extension_names))
    )
    selected = set(names)
    return FullBF16WeightPolicy(
        expected_names=names,
        fp32_stability_names=tuple(name for name in plan.fp32_stability_names if name in selected),
        int64_metadata_names=tuple(name for name in plan.int64_metadata_names if name in selected),
    )


def _plan(
    config: Qwen3_5Config,
    headers: tuple[TensorHeader, ...],
    *,
    inventory: CheckpointInventory | None,
) -> Qwen3_5WeightPlan:
    if config.identity.model_type != "qwen3_5":
        raise WeightSchemaError("qwen3_5 weight schema received another model_type")
    expected = _expected_headers(config)
    actual = {header.name: header for header in headers}
    missing = sorted(set(expected) - set(actual))
    unknown = sorted(set(actual) - set(expected))
    if missing or unknown:
        raise WeightSchemaError(
            f"qwen3_5 checkpoint tensor set is not closed: missing={missing}, unknown={unknown}"
        )
    for name, contract in expected.items():
        header = actual[name]
        if header.dtype != contract.dtype:
            raise WeightSchemaError(
                f"qwen3_5 tensor {name!r} dtype mismatch: {header.dtype} != {contract.dtype}"
            )
        if header.shape != contract.shape:
            raise WeightSchemaError(
                f"qwen3_5 tensor {name!r} shape mismatch: {header.shape} != {contract.shape}"
            )

    partition_names = {
        partition: tuple(
            sorted(name for name, item in expected.items() if item.partition == partition)
        )
        for partition in ("language", "vision", "lm_head", "mtp")
    }
    by_node: dict[str, list[str]] = defaultdict(list)
    for name, contract in expected.items():
        by_node[contract.node_id].append(name)
    bindings = tuple(
        WeightNodeBinding(node_id=node, tensor_names=tuple(sorted(names)))
        for node, names in sorted(by_node.items())
    )
    plan = _issue_weight_plan(
        model_type="qwen3_5",
        architecture_identity=WeightArchitectureIdentity(
            release_family=config.identity.release_family,
            release_profile=config.identity.release_profile,
            model_type=config.identity.model_type,
            graph_family=config.identity.graph_family,
            architecture=config.identity.architecture,
            plan_status=config.architecture_plan().status.value,
        ),
        coverage=WeightSchemaCoverage.EXACT_CHECKPOINT_HEADERS,
        headers=headers,
        inventory=inventory,
        base_language_names=partition_names["language"],
        vision_names=partition_names["vision"],
        lm_head_names=partition_names["lm_head"],
        mtp_extension_names=partition_names["mtp"],
        fp32_stability_names=(),
        int64_metadata_names=(),
        allowed_node_ids=_architecture_node_ids(config.architecture_plan()),
        node_bindings=bindings,
    )
    return plan


def _architecture_node_ids(plan: Qwen3_5ArchitecturePlan) -> tuple[str, ...]:
    return tuple(
        sorted(
            (
                plan.vision_patch_embedding_node_id,
                *(layer.node_id for layer in plan.vision_layers),
                plan.vision_merger_node_id,
                plan.token_embedding_node_id,
                *(layer.node_id for layer in plan.decoder_layers),
                plan.final_norm_node_id,
                plan.lm_head_node_id,
                *((plan.mtp_root_node_id,) if plan.mtp_layers else ()),
                *(layer.node_id for layer in plan.mtp_layers),
            )
        )
    )


def _expected_headers(config: Qwen3_5Config) -> dict[str, _ExpectedHeader]:
    text = config.graph.text
    plan = config.architecture_plan()
    expected: dict[str, _ExpectedHeader] = {}

    def add(
        name: str,
        shape: tuple[int, ...],
        node_id: str,
        partition: str,
        dtype: str = "BF16",
    ) -> None:
        if name in expected:
            raise WeightSchemaError(f"internal qwen3_5 schema duplicates {name!r}")
        expected[name] = _ExpectedHeader(dtype, shape, node_id, partition)

    hidden = text.hidden_size
    vocabulary = text.vocabulary_size
    add(
        "model.language_model.embed_tokens.weight",
        (vocabulary, hidden),
        plan.token_embedding_node_id,
        "language",
    )
    add(
        "model.language_model.norm.weight",
        (hidden,),
        plan.final_norm_node_id,
        "language",
    )
    add("lm_head.weight", (vocabulary, hidden), plan.lm_head_node_id, "lm_head")

    intermediate = text.dense_intermediate_size
    if intermediate is None:
        raise WeightSchemaError("qwen3_5 dense profile requires intermediate_size")
    for layer in plan.decoder_layers:
        prefix = f"model.language_model.layers.{layer.layer_index}"
        add(f"{prefix}.input_layernorm.weight", (hidden,), layer.node_id, "language")
        add(
            f"{prefix}.post_attention_layernorm.weight",
            (hidden,),
            layer.node_id,
            "language",
        )
        _add_dense_mlp(
            add,
            prefix=f"{prefix}.mlp",
            hidden=hidden,
            intermediate=intermediate,
            node=layer.node_id,
            partition="language",
        )
        if layer.attention is AttentionKind.GATED_DELTA_NET:
            _add_gdn(
                add,
                config.graph,
                prefix=f"{prefix}.linear_attn",
                node=layer.node_id,
                partition="language",
            )
        else:
            _add_full_attention(
                add,
                config.graph,
                prefix=f"{prefix}.self_attn",
                node=layer.node_id,
                partition="language",
            )

    _add_vision(add, config.graph, plan)
    if plan.mtp_layers:
        add("mtp.fc.weight", (hidden, hidden * 2), plan.mtp_root_node_id, "mtp")
        for suffix in ("norm.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight"):
            add(f"mtp.{suffix}", (hidden,), plan.mtp_root_node_id, "mtp")
    for mtp_layer in plan.mtp_layers:
        prefix = f"mtp.layers.{mtp_layer.layer_index}"
        add(f"{prefix}.input_layernorm.weight", (hidden,), mtp_layer.node_id, "mtp")
        add(
            f"{prefix}.post_attention_layernorm.weight",
            (hidden,),
            mtp_layer.node_id,
            "mtp",
        )
        _add_dense_mlp(
            add,
            prefix=f"{prefix}.mlp",
            hidden=hidden,
            intermediate=intermediate,
            node=mtp_layer.node_id,
            partition="mtp",
        )
        _add_full_attention(
            add,
            config.graph,
            prefix=f"{prefix}.self_attn",
            node=mtp_layer.node_id,
            partition="mtp",
        )
    return expected


def _add_dense_mlp(
    add: _AddHeader,
    *,
    prefix: str,
    hidden: int,
    intermediate: int,
    node: str,
    partition: str,
) -> None:
    add(f"{prefix}.down_proj.weight", (hidden, intermediate), node, partition)
    add(f"{prefix}.gate_proj.weight", (intermediate, hidden), node, partition)
    add(f"{prefix}.up_proj.weight", (intermediate, hidden), node, partition)


def _add_gdn(
    add: _AddHeader,
    graph: Qwen3_5GraphConfig,
    *,
    prefix: str,
    node: str,
    partition: str,
    stability_dtype: str = "BF16",
) -> None:
    text = graph.text
    gdn = text.gated_delta_net
    key_dim = gdn.key_heads * gdn.key_head_dim
    value_dim = gdn.value_heads * gdn.value_head_dim
    convolution_dim = key_dim * 2 + value_dim
    hidden = text.hidden_size
    add(f"{prefix}.A_log", (gdn.value_heads,), node, partition, stability_dtype)
    add(
        f"{prefix}.conv1d.weight",
        (convolution_dim, 1, gdn.convolution_kernel_size),
        node,
        partition,
    )
    add(f"{prefix}.dt_bias", (gdn.value_heads,), node, partition)
    add(f"{prefix}.in_proj_a.weight", (gdn.value_heads, hidden), node, partition)
    add(f"{prefix}.in_proj_b.weight", (gdn.value_heads, hidden), node, partition)
    add(f"{prefix}.in_proj_qkv.weight", (convolution_dim, hidden), node, partition)
    add(f"{prefix}.in_proj_z.weight", (value_dim, hidden), node, partition)
    add(
        f"{prefix}.norm.weight",
        (gdn.value_head_dim,),
        node,
        partition,
        stability_dtype,
    )
    add(f"{prefix}.out_proj.weight", (hidden, value_dim), node, partition)


def _add_full_attention(
    add: _AddHeader,
    graph: Qwen3_5GraphConfig,
    *,
    prefix: str,
    node: str,
    partition: str,
) -> None:
    attention = graph.text.full_attention
    hidden = graph.text.hidden_size
    q_dim = attention.query_heads * attention.head_dim
    q_projection_dim = q_dim * (2 if attention.output_gate else 1)
    kv_dim = attention.key_value_heads * attention.head_dim
    add(f"{prefix}.q_proj.weight", (q_projection_dim, hidden), node, partition)
    add(f"{prefix}.k_proj.weight", (kv_dim, hidden), node, partition)
    add(f"{prefix}.v_proj.weight", (kv_dim, hidden), node, partition)
    add(f"{prefix}.o_proj.weight", (hidden, q_dim), node, partition)
    add(f"{prefix}.q_norm.weight", (attention.head_dim,), node, partition)
    add(f"{prefix}.k_norm.weight", (attention.head_dim,), node, partition)


def _add_vision(add: _AddHeader, graph: Qwen3_5GraphConfig, plan: Qwen3_5ArchitecturePlan) -> None:
    vision = graph.vision
    hidden = vision.hidden_size
    intermediate = vision.intermediate_size
    patch_node = plan.vision_patch_embedding_node_id
    add(
        "model.visual.patch_embed.proj.weight",
        (
            hidden,
            vision.input_channels,
            vision.temporal_patch_size,
            vision.patch_size,
            vision.patch_size,
        ),
        patch_node,
        "vision",
    )
    add("model.visual.patch_embed.proj.bias", (hidden,), patch_node, "vision")
    add(
        "model.visual.pos_embed.weight",
        (vision.position_count, hidden),
        patch_node,
        "vision",
    )
    for layer in plan.vision_layers:
        prefix = f"model.visual.blocks.{layer.layer_index}"
        for norm in ("norm1", "norm2"):
            add(f"{prefix}.{norm}.weight", (hidden,), layer.node_id, "vision")
            add(f"{prefix}.{norm}.bias", (hidden,), layer.node_id, "vision")
        add(f"{prefix}.attn.qkv.weight", (hidden * 3, hidden), layer.node_id, "vision")
        add(f"{prefix}.attn.qkv.bias", (hidden * 3,), layer.node_id, "vision")
        add(f"{prefix}.attn.proj.weight", (hidden, hidden), layer.node_id, "vision")
        add(f"{prefix}.attn.proj.bias", (hidden,), layer.node_id, "vision")
        add(
            f"{prefix}.mlp.linear_fc1.weight",
            (intermediate, hidden),
            layer.node_id,
            "vision",
        )
        add(f"{prefix}.mlp.linear_fc1.bias", (intermediate,), layer.node_id, "vision")
        add(
            f"{prefix}.mlp.linear_fc2.weight",
            (hidden, intermediate),
            layer.node_id,
            "vision",
        )
        add(f"{prefix}.mlp.linear_fc2.bias", (hidden,), layer.node_id, "vision")
    merge_hidden = hidden * vision.spatial_merge_size**2
    merger = plan.vision_merger_node_id
    add("model.visual.merger.norm.weight", (hidden,), merger, "vision")
    add("model.visual.merger.norm.bias", (hidden,), merger, "vision")
    add(
        "model.visual.merger.linear_fc1.weight",
        (merge_hidden, merge_hidden),
        merger,
        "vision",
    )
    add("model.visual.merger.linear_fc1.bias", (merge_hidden,), merger, "vision")
    add(
        "model.visual.merger.linear_fc2.weight",
        (vision.output_hidden_size, merge_hidden),
        merger,
        "vision",
    )
    add(
        "model.visual.merger.linear_fc2.bias",
        (vision.output_hidden_size,),
        merger,
        "vision",
    )


class Qwen3_5ParallelWeightError(ValueError):
    """The distributed weight contract or its materialization is not closed."""


class WeightShardLayout(StrEnum):
    REPLICATED = "replicated"
    VOCAB = "vocab"
    COLUMN = "column"
    ROW = "row"
    GDN_QKV = "gdn-qkv"


@dataclass(frozen=True, slots=True)
class RankWeightBinding:
    tensor_name: str
    node_id: str
    rank: int
    pipeline_stage: int
    tensor_rank: int
    layout: WeightShardLayout
    shard_axis: int
    spans: tuple[AxisSpan, ...]
    global_shape: tuple[int, ...]
    local_shape: tuple[int, ...]

    @property
    def dtype(self) -> torch.dtype:
        """Dense checkpoint slices retain the loader's BF16 weight contract."""

        return torch.bfloat16

    def __post_init__(self) -> None:
        if not self.tensor_name or not self.node_id:
            raise Qwen3_5ParallelWeightError("weight binding identity is required")
        if min(self.rank, self.pipeline_stage, self.tensor_rank, self.shard_axis) < 0:
            raise Qwen3_5ParallelWeightError("weight placement values must be non-negative")
        if not isinstance(self.layout, WeightShardLayout):
            raise TypeError("weight binding layout must be a WeightShardLayout")
        if not self.global_shape or any(size <= 0 for size in self.global_shape):
            raise Qwen3_5ParallelWeightError("global weight shape must be non-empty")
        if len(self.local_shape) != len(self.global_shape) or any(
            size <= 0 for size in self.local_shape
        ):
            raise Qwen3_5ParallelWeightError("local weight rank must match its global shape")
        if self.shard_axis >= len(self.global_shape) or not self.spans:
            raise Qwen3_5ParallelWeightError("weight shard axis/spans are invalid")
        if any(span.stop > self.global_shape[self.shard_axis] for span in self.spans):
            raise Qwen3_5ParallelWeightError("weight slice exceeds its global dimension")
        expected = list(self.global_shape)
        expected[self.shard_axis] = sum(span.size for span in self.spans)
        if tuple(expected) != self.local_shape:
            raise Qwen3_5ParallelWeightError("local shape differs from its source slices")
        if self.layout is WeightShardLayout.REPLICATED:
            if self.spans != (AxisSpan(0, self.global_shape[self.shard_axis]),):
                raise Qwen3_5ParallelWeightError("replicated weight must retain the full tensor")
        elif len(self.spans) != (3 if self.layout is WeightShardLayout.GDN_QKV else 1):
            raise Qwen3_5ParallelWeightError("weight layout has the wrong slice count")


@dataclass(frozen=True, slots=True)
class KVHeadOwnership:
    layer_index: int
    rank: int
    head_span: AxisSpan
    replica_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if min(self.layer_index, self.rank) < 0:
            raise Qwen3_5ParallelWeightError("KV-head owner values must be non-negative")
        if (
            not self.replica_ranks
            or tuple(sorted(set(self.replica_ranks))) != self.replica_ranks
            or self.rank not in self.replica_ranks
        ):
            raise Qwen3_5ParallelWeightError("KV-head replica group must be explicit and closed")


@dataclass(frozen=True, slots=True)
class GDNStateOwnership:
    layer_index: int
    rank: int
    convolution_channel_spans: tuple[AxisSpan, AxisSpan, AxisSpan]
    key_head_span: AxisSpan
    value_head_span: AxisSpan
    convolution_shape: tuple[int, int]
    recurrent_shape: tuple[int, int, int]
    convolution_dtype: torch.dtype = torch.bfloat16
    recurrent_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if min(self.layer_index, self.rank) < 0:
            raise Qwen3_5ParallelWeightError("GDN state owner values must be non-negative")
        if self.convolution_shape[0] != sum(span.size for span in self.convolution_channel_spans):
            raise Qwen3_5ParallelWeightError("GDN convolution state differs from Q/K/V slices")
        if self.recurrent_shape[0] != self.value_head_span.size:
            raise Qwen3_5ParallelWeightError("GDN recurrent state differs from value-head shard")
        if (
            self.convolution_dtype is not torch.bfloat16
            or self.recurrent_dtype is not torch.float32
        ):
            raise Qwen3_5ParallelWeightError("GDN state dtypes must remain BF16/FP32")


@dataclass(frozen=True, slots=True)
class Qwen3_5ParallelWeightPlan:
    config: Qwen3_5Config
    topology: ParallelTopology
    partition: PartitionPlan
    semantic_plan_digest: str
    global_names: tuple[str, ...]
    bindings: tuple[RankWeightBinding, ...]
    kv_heads: tuple[KVHeadOwnership, ...]
    gdn_states: tuple[GDNStateOwnership, ...]
    mtp_kv_heads: tuple[KVHeadOwnership, ...] = ()

    def __post_init__(self) -> None:
        if self.topology.ep_size != 1:
            raise Qwen3_5ParallelWeightError("the dense qwen3_5 graph requires EP1")
        if self.partition.topology_id != self.topology.topology_id:
            raise Qwen3_5ParallelWeightError("partition and topology identities differ")
        expected = _expected_headers(self.config)
        enabled = any(name.startswith("mtp.") for name in self.global_names)
        names = tuple(
            sorted(
                name
                for name, header in expected.items()
                if header.partition
                in (("language", "lm_head", "mtp") if enabled else ("language", "lm_head"))
            )
        )
        if self.global_names != names:
            raise Qwen3_5ParallelWeightError("parallel inventory differs from configured tensors")
        if bool(self.mtp_kv_heads) != enabled:
            raise Qwen3_5ParallelWeightError(
                "MTP weights and KV ownership must be enabled together"
            )

        placement_by_rank = {placement.rank: placement for placement in self.topology.placements}
        assignment_by_node = {item.node_id: item for item in self.partition.assignments}
        grouped: dict[str, list[RankWeightBinding]] = defaultdict(list)
        for binding in self.bindings:
            placement = placement_by_rank.get(binding.rank)
            assignment = assignment_by_node.get(binding.node_id)
            if placement is None or assignment is None:
                raise Qwen3_5ParallelWeightError("weight binding references an unknown owner")
            if (
                binding.pipeline_stage != placement.pipeline_stage
                or binding.tensor_rank != placement.tensor_rank
                or binding.pipeline_stage != assignment.pipeline_stage
                or binding.rank not in assignment.owner_ranks
            ):
                raise Qwen3_5ParallelWeightError("weight binding differs from graph placement")
            grouped[binding.tensor_name].append(binding)
        if tuple(sorted(grouped)) != self.global_names:
            raise Qwen3_5ParallelWeightError("rank bindings do not close the text inventory")

        for name, owners in grouped.items():
            exemplar = owners[0]
            assignment = assignment_by_node[exemplar.node_id]
            if tuple(sorted(binding.rank for binding in owners)) != assignment.owner_ranks:
                raise Qwen3_5ParallelWeightError(f"rank-local contract is open for {name!r}")
            if any(
                binding.node_id != exemplar.node_id
                or binding.layout is not exemplar.layout
                or binding.shard_axis != exemplar.shard_axis
                or binding.global_shape != exemplar.global_shape
                for binding in owners
            ):
                raise Qwen3_5ParallelWeightError(f"rank bindings disagree for {name!r}")
            dimension = exemplar.global_shape[exemplar.shard_axis]
            if exemplar.layout is WeightShardLayout.REPLICATED:
                if any(binding.spans != (AxisSpan(0, dimension),) for binding in owners):
                    raise Qwen3_5ParallelWeightError(f"replica set is incomplete for {name!r}")
            else:
                intervals = sorted(span for binding in owners for span in binding.spans)
                cursor = 0
                for span in intervals:
                    if span.start != cursor:
                        raise Qwen3_5ParallelWeightError(
                            f"semantic slices overlap or leave a gap for {name!r}"
                        )
                    cursor = span.stop
                if cursor != dimension:
                    raise Qwen3_5ParallelWeightError(f"semantic slices are open for {name!r}")

        for rank in range(self.topology.world_size):
            local = tuple(
                sorted(binding.tensor_name for binding in self.bindings if binding.rank == rank)
            )
            expected = tuple(
                sorted(
                    name
                    for name, owners in grouped.items()
                    if rank in assignment_by_node[owners[0].node_id].owner_ranks
                )
            )
            if local != expected or len(local) != len(set(local)):
                raise Qwen3_5ParallelWeightError(f"rank {rank} local inventory is not closed")

        self._validate_state_ownership(assignment_by_node, grouped)
        self._validate_mtp_state_ownership(assignment_by_node, grouped)

    @property
    def mtp_enabled(self) -> bool:
        return bool(self.mtp_kv_heads)

    def _validate_mtp_state_ownership(
        self,
        assignments: dict[str, GraphAssignment],
        weights: dict[str, list[RankWeightBinding]],
    ) -> None:
        if not self.mtp_kv_heads:
            return
        if self.topology.pp_size != 1:
            raise Qwen3_5ParallelWeightError("MTP parallel ownership is currently TP-only")
        assignment = assignments.get("extension.mtp.layer.0000")
        if assignment is None:
            raise Qwen3_5ParallelWeightError("MTP layer assignment is missing")
        owners = tuple(sorted(self.mtp_kv_heads, key=lambda owner: owner.rank))
        if tuple(owner.rank for owner in owners) != assignment.owner_ranks:
            raise Qwen3_5ParallelWeightError("MTP KV owners do not close the TP stage")
        spans = sorted(owner.head_span for owner in owners)
        cursor = 0
        for span in spans:
            if span.start != cursor:
                raise Qwen3_5ParallelWeightError("MTP KV-head ownership overlaps or has a gap")
            cursor = span.stop
        if cursor != self.config.graph.text.full_attention.key_value_heads:
            raise Qwen3_5ParallelWeightError("MTP KV-head ownership must cover configured heads")
        prefix = "mtp.layers.0.self_attn"
        for owner in owners:
            expected = AxisSpan(
                owner.head_span.start * self.config.graph.text.full_attention.head_dim,
                owner.head_span.stop * self.config.graph.text.full_attention.head_dim,
            )
            for projection in ("k_proj", "v_proj"):
                binding = next(
                    item
                    for item in weights[f"{prefix}.{projection}.weight"]
                    if item.rank == owner.rank
                )
                if binding.spans != (expected,):
                    raise Qwen3_5ParallelWeightError("MTP KV weights and state ownership differ")

    def _validate_state_ownership(
        self,
        assignments: dict[str, GraphAssignment],
        weights: dict[str, list[RankWeightBinding]],
    ) -> None:
        kv_by_layer: dict[int, list[KVHeadOwnership]] = defaultdict(list)
        for owner in self.kv_heads:
            kv_by_layer[owner.layer_index].append(owner)
        text = self.config.graph.text
        gdn = text.gated_delta_net
        expected_full_attention = {
            i for i, kind in enumerate(text.layer_schedule) if kind is AttentionKind.FULL_ATTENTION
        }
        if set(kv_by_layer) != expected_full_attention:
            raise Qwen3_5ParallelWeightError(
                "KV-head owners differ from the configured attention schedule"
            )
        for layer_index, kv_owners in kv_by_layer.items():
            assignment = assignments[f"text.decoder.layer.{layer_index:04d}"]
            if tuple(sorted(owner.rank for owner in kv_owners)) != assignment.owner_ranks:
                raise Qwen3_5ParallelWeightError("full-attention KV owners do not close a stage")
            spans = sorted({owner.head_span for owner in kv_owners})
            cursor = 0
            for span in spans:
                if span.start != cursor:
                    raise Qwen3_5ParallelWeightError("KV-head ownership overlaps or has a gap")
                cursor = span.stop
            if cursor != self.config.graph.text.full_attention.key_value_heads:
                raise Qwen3_5ParallelWeightError("KV-head ownership must cover configured heads")
            for owner in kv_owners:
                peers = tuple(item.rank for item in kv_owners if item.head_span == owner.head_span)
                if peers != owner.replica_ranks:
                    raise Qwen3_5ParallelWeightError("KV-head replica group is implicit or wrong")
                expected = AxisSpan(
                    owner.head_span.start * self.config.graph.text.full_attention.head_dim,
                    owner.head_span.stop * self.config.graph.text.full_attention.head_dim,
                )
                prefix = f"model.language_model.layers.{layer_index}.self_attn"
                for projection in ("k_proj", "v_proj"):
                    binding = next(
                        item
                        for item in weights[f"{prefix}.{projection}.weight"]
                        if item.rank == owner.rank
                    )
                    if binding.spans != (expected,):
                        raise Qwen3_5ParallelWeightError(
                            "KV weight and state head ownership differ"
                        )

        gdn_by_layer: dict[int, list[GDNStateOwnership]] = defaultdict(list)
        for gdn_owner in self.gdn_states:
            gdn_by_layer[gdn_owner.layer_index].append(gdn_owner)
        if set(gdn_by_layer) != set(range(text.layer_count)) - expected_full_attention:
            raise Qwen3_5ParallelWeightError(
                "GDN state owners differ from the configured recurrent schedule"
            )
        for layer_index, gdn_owners in gdn_by_layer.items():
            assignment = assignments[f"text.decoder.layer.{layer_index:04d}"]
            if tuple(sorted(owner.rank for owner in gdn_owners)) != assignment.owner_ranks:
                raise Qwen3_5ParallelWeightError("GDN state owners do not close a stage")
            for attribute, dimension in (
                ("key_head_span", gdn.key_heads),
                ("value_head_span", gdn.value_heads),
            ):
                spans = sorted(getattr(owner, attribute) for owner in gdn_owners)
                cursor = 0
                for span in spans:
                    if span.start != cursor:
                        raise Qwen3_5ParallelWeightError("GDN state heads overlap or have a gap")
                    cursor = span.stop
                if cursor != dimension:
                    raise Qwen3_5ParallelWeightError("GDN state heads do not close the model")
            channel_spans = sorted(
                span for owner in gdn_owners for span in owner.convolution_channel_spans
            )
            cursor = 0
            for span in channel_spans:
                if span.start != cursor:
                    raise Qwen3_5ParallelWeightError(
                        "GDN convolution channels overlap or have a gap"
                    )
                cursor = span.stop
            if (
                cursor
                != 2 * gdn.key_heads * gdn.key_head_dim + gdn.value_heads * gdn.value_head_dim
            ):
                raise Qwen3_5ParallelWeightError("GDN convolution channels do not close the model")
            for state_owner in gdn_owners:
                q_span, k_span, value_span = state_owner.convolution_channel_spans
                if (
                    q_span.size != state_owner.key_head_span.size * gdn.key_head_dim
                    or k_span.size != state_owner.key_head_span.size * gdn.key_head_dim
                    or value_span.size != state_owner.value_head_span.size * gdn.value_head_dim
                    or state_owner.convolution_shape[1] != gdn.convolution_kernel_size - 1
                    or state_owner.recurrent_shape
                    != (state_owner.value_head_span.size, gdn.value_head_dim, gdn.key_head_dim)
                ):
                    raise Qwen3_5ParallelWeightError("GDN state shape differs from its head shard")
                prefix = f"model.language_model.layers.{layer_index}.linear_attn"
                for suffix in ("in_proj_qkv.weight", "conv1d.weight"):
                    binding = next(
                        item
                        for item in weights[f"{prefix}.{suffix}"]
                        if item.rank == state_owner.rank
                    )
                    if binding.spans != state_owner.convolution_channel_spans:
                        raise Qwen3_5ParallelWeightError(
                            "GDN weight and convolution-state ownership differ"
                        )
                for suffix in ("A_log", "dt_bias", "in_proj_a.weight", "in_proj_b.weight"):
                    binding = next(
                        item
                        for item in weights[f"{prefix}.{suffix}"]
                        if item.rank == state_owner.rank
                    )
                    if binding.spans != (state_owner.value_head_span,):
                        raise Qwen3_5ParallelWeightError(
                            "GDN parameter and recurrent-state head ownership differ"
                        )
                value_width = AxisSpan(
                    state_owner.value_head_span.start * gdn.value_head_dim,
                    state_owner.value_head_span.stop * gdn.value_head_dim,
                )
                for suffix in ("in_proj_z.weight", "out_proj.weight"):
                    binding = next(
                        item
                        for item in weights[f"{prefix}.{suffix}"]
                        if item.rank == state_owner.rank
                    )
                    if binding.spans != (value_width,):
                        raise Qwen3_5ParallelWeightError(
                            "GDN value projection and state ownership differ"
                        )

    def bindings_for_rank(self, rank: int) -> tuple[RankWeightBinding, ...]:
        if rank < 0 or rank >= self.topology.world_size:
            raise Qwen3_5ParallelWeightError("unknown rank")
        return tuple(binding for binding in self.bindings if binding.rank == rank)


class Qwen3_5RankWeights(RankLocalTensors[RankWeightBinding]):
    """One rank's exact local tensors, without a distributed-forward claim."""

    __slots__ = ()

    def __init__(
        self,
        *,
        rank: int,
        device: torch.device,
        inventory_digest: str,
        bindings: tuple[RankWeightBinding, ...],
        tensors: dict[str, Tensor],
    ) -> None:
        try:
            super().__init__(
                rank=rank,
                device=device,
                inventory_digest=inventory_digest,
                bindings=bindings,
                tensors=tensors,
                dtypes={binding.tensor_name: torch.bfloat16 for binding in bindings},
            )
        except ValueError as error:
            raise Qwen3_5ParallelWeightError(str(error)) from error


def plan_qwen3_5_dense_weights(
    config: Qwen3_5Config,
    weight_plan: Qwen3_5WeightPlan,
    topology: ParallelTopology,
    *,
    include_mtp: bool = False,
) -> Qwen3_5ParallelWeightPlan:
    """Assign dense text, and optionally MTP, to a contiguous TP x PP grid."""

    if config.identity.model_type != "qwen3_5" or weight_plan.model_type != "qwen3_5":
        raise Qwen3_5ParallelWeightError("parallel plan requires the dense qwen3_5 graph")
    if weight_plan.architecture_identity.release_profile != config.identity.release_profile:
        raise Qwen3_5ParallelWeightError("configuration and weight-plan identities differ")
    if topology.ep_size != 1:
        raise Qwen3_5ParallelWeightError("the dense qwen3_5 graph requires EP1")
    if type(include_mtp) is not bool:
        raise TypeError("include_mtp must be an explicit boolean")
    if include_mtp and topology.pp_size != 1:
        raise Qwen3_5ParallelWeightError("MTP parallel planning is currently TP-only")

    if include_mtp and config.graph.text.mtp.hidden_layer_count != 1:
        raise Qwen3_5ParallelWeightError("MTP execution requires one declared prediction layer")
    plan_qwen3_5_weight_headers(config, weight_plan.checkpoint_headers)
    text = config.graph.text
    intermediate = text.dense_intermediate_size
    if intermediate is None:
        raise Qwen3_5ParallelWeightError("parallel plan requires the dense qwen3_5 graph")
    tp_dimensions = (
        text.vocabulary_size,
        intermediate,
        text.full_attention.query_heads,
        text.full_attention.key_value_heads,
        text.gated_delta_net.key_heads,
        text.gated_delta_net.value_heads,
    )
    if any(dimension % topology.tp_size for dimension in tp_dimensions):
        raise Qwen3_5ParallelWeightError(
            "qwen3_5 dense heads, vocabulary, and intermediate width must divide TP"
        )

    architecture = config.architecture_plan()
    layers_per_stage, remainder = divmod(len(architecture.decoder_layers), topology.pp_size)
    if remainder:
        raise Qwen3_5ParallelWeightError("pipeline stages require contiguous equal layer ranges")
    stage_ranks = {
        stage: tuple(
            sorted(
                placement.rank
                for placement in topology.placements
                if placement.pipeline_stage == stage
            )
        )
        for stage in range(topology.pp_size)
    }
    node_stage = {architecture.token_embedding_node_id: 0}
    for layer in architecture.decoder_layers:
        node_stage[layer.node_id] = layer.layer_index // layers_per_stage
    node_stage[architecture.final_norm_node_id] = topology.pp_size - 1
    node_stage[architecture.lm_head_node_id] = topology.pp_size - 1
    if include_mtp:
        node_stage[architecture.mtp_root_node_id] = 0
        for mtp_layer in architecture.mtp_layers:
            node_stage[mtp_layer.node_id] = 0
    assignments = tuple(
        GraphAssignment(node_id, stage, stage_ranks[stage]) for node_id, stage in node_stage.items()
    )
    partition = PartitionPlan(
        partition_id=f"qwen3_5:text:{topology.topology_id}",
        topology_id=topology.topology_id,
        assignments=assignments,
        pipeline_edges=tuple(
            PipelineEdge(
                edge_id=(
                    f"qwen3_5.pp:lane:{tensor_rank}"
                    if topology.pp_size == 2
                    else f"qwen3_5.pp:{stage}->{stage + 1}:lane:{tensor_rank}"
                ),
                source_node_id=architecture.decoder_layers[
                    (stage + 1) * layers_per_stage - 1
                ].node_id,
                target_node_id=architecture.decoder_layers[(stage + 1) * layers_per_stage].node_id,
                source_rank=stage_ranks[stage][tensor_rank],
                target_rank=stage_ranks[stage + 1][tensor_rank],
                tensor_dtype=ScalarDType.BF16,
                tensor_layout="batch_token_hidden",
            )
            for stage in range(topology.pp_size - 1)
            for tensor_rank in range(topology.tp_size)
        ),
    )

    text_names = (
        qwen3_5_mtp_weight_policy(weight_plan).expected_names
        if include_mtp
        else qwen3_5_text_weight_policy(weight_plan).expected_names
    )
    semantic_by_name = {item.name: item for item in weight_plan.tensor_semantics}
    placement_by_rank = {placement.rank: placement for placement in topology.placements}
    bindings: list[RankWeightBinding] = []
    for name in text_names:
        semantic = semantic_by_name[name]
        try:
            stage = node_stage[semantic.node_id]
        except KeyError as error:
            raise Qwen3_5ParallelWeightError(
                f"text tensor {name!r} is bound outside the executable graph"
            ) from error
        layout, axis = _weight_layout(name)
        for rank in stage_ranks[stage]:
            placement = placement_by_rank[rank]
            spans = _weight_spans(
                config,
                name=name,
                layout=layout,
                dimension=semantic.shape[axis],
                tensor_rank=placement.tensor_rank,
                tp_size=topology.tp_size,
            )
            local_shape = list(semantic.shape)
            local_shape[axis] = sum(span.size for span in spans)
            bindings.append(
                RankWeightBinding(
                    tensor_name=name,
                    node_id=semantic.node_id,
                    rank=rank,
                    pipeline_stage=stage,
                    tensor_rank=placement.tensor_rank,
                    layout=layout,
                    shard_axis=axis,
                    spans=spans,
                    global_shape=semantic.shape,
                    local_shape=tuple(local_shape),
                )
            )

    kv_heads: list[KVHeadOwnership] = []
    gdn_states: list[GDNStateOwnership] = []
    mtp_kv_heads: list[KVHeadOwnership] = []
    gdn = config.graph.text.gated_delta_net
    key_width = gdn.key_heads * gdn.key_head_dim
    value_width = gdn.value_heads * gdn.value_head_dim
    for layer in architecture.decoder_layers:
        stage = node_stage[layer.node_id]
        if layer.attention is AttentionKind.FULL_ATTENTION:
            head_size = config.graph.text.full_attention.key_value_heads // topology.tp_size
            for rank in stage_ranks[stage]:
                tensor_rank = placement_by_rank[rank].tensor_rank
                span = AxisSpan(tensor_rank * head_size, (tensor_rank + 1) * head_size)
                kv_heads.append(KVHeadOwnership(layer.layer_index, rank, span, (rank,)))
            continue
        for rank in stage_ranks[stage]:
            tensor_rank = placement_by_rank[rank].tensor_rank
            channel_spans = _segmented_spans(
                (key_width, key_width, value_width), tensor_rank, topology.tp_size
            )
            key_heads = _equal_span(gdn.key_heads, tensor_rank, topology.tp_size)
            value_heads = _equal_span(gdn.value_heads, tensor_rank, topology.tp_size)
            gdn_states.append(
                GDNStateOwnership(
                    layer_index=layer.layer_index,
                    rank=rank,
                    convolution_channel_spans=cast(
                        tuple[AxisSpan, AxisSpan, AxisSpan], channel_spans
                    ),
                    key_head_span=key_heads,
                    value_head_span=value_heads,
                    convolution_shape=(
                        sum(span.size for span in channel_spans),
                        gdn.convolution_kernel_size - 1,
                    ),
                    recurrent_shape=(
                        value_heads.size,
                        gdn.value_head_dim,
                        gdn.key_head_dim,
                    ),
                )
            )

    if include_mtp:
        head_size = config.graph.text.full_attention.key_value_heads // topology.tp_size
        for rank in stage_ranks[0]:
            tensor_rank = placement_by_rank[rank].tensor_rank
            span = AxisSpan(tensor_rank * head_size, (tensor_rank + 1) * head_size)
            mtp_kv_heads.append(KVHeadOwnership(0, rank, span, (rank,)))

    return Qwen3_5ParallelWeightPlan(
        config=config,
        topology=topology,
        partition=partition,
        semantic_plan_digest=weight_plan.semantic_plan_digest,
        global_names=text_names,
        bindings=tuple(bindings),
        kv_heads=tuple(kv_heads),
        gdn_states=tuple(gdn_states),
        mtp_kv_heads=tuple(mtp_kv_heads),
    )


def plan_qwen3_5_tp2_pp2_weights(
    config: Qwen3_5Config,
    weight_plan: Qwen3_5WeightPlan,
    topology: ParallelTopology,
) -> Qwen3_5ParallelWeightPlan:
    """Compatibility wrapper for the original closed TP2 x PP2 placement."""

    if (topology.tp_size, topology.pp_size, topology.ep_size) != (2, 2, 1):
        raise Qwen3_5ParallelWeightError("TP2 x PP2 planner requires TP2 x PP2 with EP1")
    return plan_qwen3_5_dense_weights(config, weight_plan, topology)


def load_qwen3_5_rank_weights(
    *,
    inventory: CheckpointInventory,
    weight_plan: Qwen3_5WeightPlan,
    parallel_plan: Qwen3_5ParallelWeightPlan,
    rank: int,
    device: str | torch.device,
    progress: Callable[[str, int, int], None] | None = None,
) -> Qwen3_5RankWeights:
    """Read only rank-owned safetensors ranges; never materialize then slice a full weight."""

    if weight_plan.inventory_digest != inventory.inventory_digest:
        raise Qwen3_5ParallelWeightError("weight plan differs from the attested checkpoint")
    if parallel_plan.semantic_plan_digest != weight_plan.semantic_plan_digest:
        raise Qwen3_5ParallelWeightError("parallel plan differs from the semantic weight plan")
    expected_policy = (
        qwen3_5_mtp_weight_policy(weight_plan)
        if parallel_plan.mtp_enabled
        else qwen3_5_text_weight_policy(weight_plan)
    )
    if parallel_plan.global_names != expected_policy.expected_names:
        raise Qwen3_5ParallelWeightError("parallel plan differs from its executable inventory")
    bindings = parallel_plan.bindings_for_rank(rank)
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise Qwen3_5ParallelWeightError("CUDA was requested but is unavailable")
    placement = next(item for item in parallel_plan.topology.placements if item.rank == rank)
    if target.type == "cuda" and target.index != placement.device_index:
        raise Qwen3_5ParallelWeightError("rank was assigned to a different CUDA device")
    records = {record.name: record for record in inventory.tensors}
    by_shard: dict[str, list[RankWeightBinding]] = defaultdict(list)
    for binding in bindings:
        try:
            record = records[binding.tensor_name]
        except KeyError as error:
            raise Qwen3_5ParallelWeightError(
                f"attested inventory is missing {binding.tensor_name!r}"
            ) from error
        if record.dtype != "BF16" or record.shape != binding.global_shape:
            raise Qwen3_5ParallelWeightError(f"attested header differs for {binding.tensor_name!r}")
        by_shard[record.shard_path].append(binding)

    mtp_names = set(weight_plan.mtp_extension_names)
    sections = {
        binding.tensor_name: "mtp" if binding.tensor_name in mtp_names else "base"
        for binding in bindings
    }
    totals: dict[str, int] = defaultdict(int)
    completed: dict[str, int] = defaultdict(int)
    for binding in bindings:
        totals[sections[binding.tensor_name]] += 2 * torch.Size(binding.local_shape).numel()
    if progress is not None:
        for section in ("base", "mtp"):
            if totals[section]:
                progress(section, 0, totals[section])

    loaded: dict[str, Tensor] = {}
    try:
        for shard_path, shard_bindings in sorted(by_shard.items()):
            path = _confined_shard(inventory.model_root, shard_path)
            with safe_open(path, framework="pt", device="cpu") as raw_reader:
                reader = cast(SafeTensorReader, raw_reader)
                available = set(reader.keys())
                for binding in sorted(
                    shard_bindings,
                    key=lambda item: records[item.tensor_name].offset_ordinal,
                ):
                    if binding.tensor_name not in available:
                        raise Qwen3_5ParallelWeightError(
                            f"checkpoint shard is missing {binding.tensor_name!r}"
                        )
                    loaded[binding.tensor_name] = _copy_binding(reader, binding, target)
                    section = sections[binding.tensor_name]
                    completed[section] += 2 * torch.Size(binding.local_shape).numel()
                    if progress is not None:
                        progress(section, completed[section], totals[section])
    except Exception:
        loaded.clear()
        raise
    return Qwen3_5RankWeights(
        rank=rank,
        device=target,
        inventory_digest=inventory.inventory_digest,
        bindings=bindings,
        tensors=loaded,
    )


def _copy_binding(
    reader: SafeTensorReader,
    binding: RankWeightBinding,
    device: torch.device,
) -> Tensor:
    try:
        return materialize_axis_spans(
            reader,
            binding,
            dtype=torch.bfloat16,
            device=device,
        )
    except ValueError as error:
        raise Qwen3_5ParallelWeightError(str(error)) from error


def _confined_shard(root: Path, relative: str) -> Path:
    try:
        return confined_checkpoint_path(root, relative)
    except ValueError as error:
        raise Qwen3_5ParallelWeightError(str(error)) from error


def _weight_layout(name: str) -> tuple[WeightShardLayout, int]:
    if name in {"model.language_model.embed_tokens.weight", "lm_head.weight"}:
        return WeightShardLayout.VOCAB, 0
    replicated_suffixes = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "q_norm.weight",
        "k_norm.weight",
    )
    if name == "mtp.fc.weight":
        return WeightShardLayout.COLUMN, 0
    if name in {
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    }:
        return WeightShardLayout.REPLICATED, 0
    if (
        name == "model.language_model.norm.weight"
        or name.endswith(replicated_suffixes)
        or name.endswith("linear_attn.norm.weight")
    ):
        return WeightShardLayout.REPLICATED, 0
    if name.endswith(("mlp.gate_proj.weight", "mlp.up_proj.weight")):
        return WeightShardLayout.COLUMN, 0
    if name.endswith("mlp.down_proj.weight"):
        return WeightShardLayout.ROW, 1
    if ".self_attn." in name:
        if name.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight")):
            return WeightShardLayout.COLUMN, 0
        if name.endswith("o_proj.weight"):
            return WeightShardLayout.ROW, 1
    if ".linear_attn." in name:
        if name.endswith(("in_proj_qkv.weight", "conv1d.weight")):
            return WeightShardLayout.GDN_QKV, 0
        if name.endswith(
            (
                "A_log",
                "dt_bias",
                "in_proj_a.weight",
                "in_proj_b.weight",
                "in_proj_z.weight",
            )
        ):
            return WeightShardLayout.COLUMN, 0
        if name.endswith("out_proj.weight"):
            return WeightShardLayout.ROW, 1
    raise Qwen3_5ParallelWeightError(f"no audited TP layout exists for {name!r}")


def _weight_spans(
    config: Qwen3_5Config | Qwen3_5MoeConfig,
    *,
    name: str,
    layout: WeightShardLayout,
    dimension: int,
    tensor_rank: int,
    tp_size: int,
) -> tuple[AxisSpan, ...]:
    if layout is WeightShardLayout.REPLICATED:
        return (AxisSpan(0, dimension),)
    if layout is WeightShardLayout.GDN_QKV:
        gdn = config.graph.text.gated_delta_net
        key_width = gdn.key_heads * gdn.key_head_dim
        value_width = gdn.value_heads * gdn.value_head_dim
        if dimension != key_width * 2 + value_width:
            raise Qwen3_5ParallelWeightError(f"GDN Q/K/V dimension differs for {name!r}")
        return _segmented_spans((key_width, key_width, value_width), tensor_rank, tp_size)
    return (_equal_span(dimension, tensor_rank, tp_size),)


def _segmented_spans(
    dimensions: tuple[int, int, int], tensor_rank: int, tp_size: int
) -> tuple[AxisSpan, ...]:
    spans: list[AxisSpan] = []
    offset = 0
    for dimension in dimensions:
        local = _equal_span(dimension, tensor_rank, tp_size)
        spans.append(AxisSpan(offset + local.start, offset + local.stop))
        offset += dimension
    return tuple(spans)


def _equal_span(dimension: int, tensor_rank: int, tp_size: int) -> AxisSpan:
    try:
        return equal_axis_span(dimension, tensor_rank, tp_size)
    except ValueError as error:
        raise Qwen3_5ParallelWeightError(str(error)) from error


def logical_quantized_weights(
    config: Qwen3_5Config,
) -> tuple[LogicalWeight, ...]:
    """Bind dense logical weights without adding execution policy to the model."""
    return _logical_quantized_headers(config.graph, _expected_headers(config))


def quantized_owner_slices(
    config: Qwen3_5Config,
    topology: ParallelTopology,
    rank: int,
    *,
    include_mtp: bool = False,
    logical_weights: tuple[LogicalWeight, ...] | None = None,
) -> tuple[LogicalWeightSlice, ...]:
    """Place logical Dense leaves independently of their checkpoint encodings."""
    if topology.ep_size != 1:
        raise Qwen3_5ParallelWeightError("Dense logical weights require EP1")
    stage_weights, tensor_rank = _quantized_stage_weights(
        config,
        logical_weights if logical_weights is not None else logical_quantized_weights(config),
        topology,
        rank,
        include_mtp=include_mtp,
    )
    return tuple(
        _quantized_tensor_slice(config, weight, tensor_rank, topology.tp_size)
        for weight in stage_weights
    )


def _quantized_stage_weights(
    config: Qwen3_5Config | Qwen3_5MoeConfig,
    logical: tuple[LogicalWeight, ...],
    topology: ParallelTopology,
    rank: int,
    *,
    include_mtp: bool,
) -> tuple[tuple[LogicalWeight, ...], int]:
    placement = next((p for p in topology.placements if p.rank == rank), None)
    layers = len(config.graph.text.layer_schedule)
    if placement is None or layers % topology.pp_size:
        raise Qwen3_5ParallelWeightError(
            "logical ownership requires valid rank and equal PP stages"
        )
    if include_mtp and topology.pp_size != 1:
        raise Qwen3_5ParallelWeightError("MTP proposal placement across PP is not yet closed")
    per_stage = layers // topology.pp_size
    selected = []
    for weight in logical:
        name = weight.name
        if weight.partition == "vision" or (weight.partition == "mtp" and not include_mtp):
            continue
        if name.startswith("model.language_model.layers."):
            stage = int(name.split(".layers.", 1)[1].split(".", 1)[0]) // per_stage
        elif name in ("lm_head.weight", "model.language_model.norm.weight"):
            stage = topology.pp_size - 1
        elif name == "model.language_model.embed_tokens.weight" or weight.partition == "mtp":
            stage = 0
        else:
            raise Qwen3_5ParallelWeightError(f"unowned logical text weight {name!r}")
        if stage == placement.pipeline_stage:
            selected.append(weight)
    return tuple(selected), placement.tensor_rank


def _quantized_tensor_slice(
    config: Qwen3_5Config | Qwen3_5MoeConfig,
    weight: LogicalWeight,
    tensor_rank: int,
    tp_size: int,
) -> LogicalWeightSlice:
    layout, axis = _weight_layout(weight.name)
    dimension = weight.shape[axis]
    attention = config.graph.text.full_attention
    spans: tuple[AxisSpan, ...]
    if (
        weight.name.endswith(("self_attn.k_proj.weight", "self_attn.v_proj.weight"))
        and tp_size > attention.key_value_heads
    ):
        if tp_size % attention.key_value_heads:
            raise Qwen3_5ParallelWeightError("KV head replication must divide TP")
        head = tensor_rank // (tp_size // attention.key_value_heads)
        spans = (_equal_span(dimension, head, attention.key_value_heads),)
    else:
        spans = _weight_spans(
            config,
            name=weight.name,
            layout=layout,
            dimension=dimension,
            tensor_rank=tensor_rank,
            tp_size=tp_size,
        )
    return LogicalWeightSlice(weight.name, axis, spans, weight.shape)


def _logical_quantized_headers(
    graph: Qwen3_5GraphConfig, headers: dict[str, _ExpectedHeader]
) -> tuple[LogicalWeight, ...]:
    result: list[LogicalWeight] = []
    gdn = graph.text.gated_delta_net
    for name, header in headers.items():
        operation = "tensor"
        parts: tuple[int, ...] = ()
        if len(header.shape) == 2 and name.endswith(".weight"):
            operation = "embedding" if name.endswith("embed_tokens.weight") else "linear"
            if name.endswith("linear_attn.in_proj_qkv.weight"):
                parts = (gdn.key_heads * gdn.key_head_dim,) * 2 + (
                    gdn.value_heads * gdn.value_head_dim,
                )
            elif name.endswith(".attn.qkv.weight"):
                parts = (header.shape[0] // 3,) * 3
        result.append(
            LogicalWeight(
                name,
                header.shape,
                header.dtype,
                header.partition,
                operation,
                _aliases(name.removesuffix(".weight")),
                parts,
            )
        )
    return tuple(sorted(result, key=lambda item: item.name))


def _aliases(module: str) -> tuple[str, ...]:
    result = {module.replace("model.language_model.", "model.")}
    if module.startswith("model.visual."):
        result.add(module.removeprefix("model."))
    for source, target in (
        ("in_proj_qkv", "in_proj_qkvz"),
        ("in_proj_z", "in_proj_qkvz"),
        ("self_attn.q_proj", "self_attn.qkv_proj"),
        ("self_attn.k_proj", "self_attn.qkv_proj"),
        ("self_attn.v_proj", "self_attn.qkv_proj"),
        ("mlp.gate_proj", "mlp.gate_up_proj"),
        ("mlp.up_proj", "mlp.gate_up_proj"),
    ):
        if module.endswith(source):
            result.add(module[: -len(source)] + target)
            result.add((module[: -len(source)] + target).replace("model.language_model.", "model."))
    return tuple(sorted(result - {module}))
