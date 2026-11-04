"""Checkpoint expert ownership with independent tensor and expert shard axes.

EP partitions expert IDs; ETP partitions each owned expert's intermediate
dimension. Shared projections retain conventional stage TP.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from cubic_serve._compat import StrEnum
from typing import cast

import torch
from safetensors import safe_open
from torch import Tensor

from cubic_serve.contracts.execution import (
    ExpertAssignment,
    GraphAssignment,
    ParallelTopology,
    PartitionPlan,
    PipelineEdge,
)
from cubic_serve.contracts.model import ScalarDType
from cubic_serve.models.checkpoint import CheckpointInventory
from cubic_serve.models.parallel_weights import (
    AxisSpan,
    RankLocalTensors,
    SafeTensorReader,
    confined_checkpoint_path,
    equal_axis_span,
    materialize_axis_spans,
)
from cubic_serve.models.qwen3_5.model import AttentionKind
from cubic_serve.models.qwen3_5.weights import GDNStateOwnership, KVHeadOwnership
from cubic_serve.models.qwen3_5_moe.config import Qwen3_5MoeConfig
from cubic_serve.models.qwen3_5_moe.weights import (
    Qwen3_5MoeWeightPlan,
    _expected_headers,
    plan_qwen3_5_moe_weight_headers,
    qwen3_5_moe_mtp_weight_policy,
    qwen3_5_moe_text_weight_policy,
)


class Qwen3_5MoeParallelError(ValueError):
    """The MoE rank-local ownership or payload is not closed."""


class MoeWeightLayout(StrEnum):
    REPLICATED = "replicated"
    VOCAB = "vocab"
    COLUMN = "column"
    ROW = "row"
    GDN_QKV = "gdn-qkv"
    KV_REPLICATED = "kv-replicated"
    EXPERT = "expert"


@dataclass(frozen=True, slots=True)
class MoeRankWeightBinding:
    tensor_name: str
    node_id: str
    rank: int
    pipeline_stage: int
    tensor_rank: int
    layout: MoeWeightLayout
    dtype: torch.dtype
    shard_axis: int
    spans: tuple[AxisSpan, ...]
    global_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    secondary_axis: int | None = None
    secondary_spans: tuple[AxisSpan, ...] = ()

    def __post_init__(self) -> None:
        if not self.tensor_name or not self.node_id:
            raise Qwen3_5MoeParallelError("weight binding identity is required")
        if min(self.rank, self.pipeline_stage, self.tensor_rank, self.shard_axis) < 0:
            raise Qwen3_5MoeParallelError("weight placement values must be non-negative")
        if self.dtype not in (torch.bfloat16, torch.float32):
            raise Qwen3_5MoeParallelError("MoE baseline weights must be BF16 or declared FP32")
        if not self.global_shape or len(self.local_shape) != len(self.global_shape):
            raise Qwen3_5MoeParallelError("weight shapes must have one stable rank")
        if self.shard_axis >= len(self.global_shape) or not self.spans:
            raise Qwen3_5MoeParallelError("weight spans do not identify a source axis")
        if any(span.stop > self.global_shape[self.shard_axis] for span in self.spans):
            raise Qwen3_5MoeParallelError("weight span exceeds its source dimension")
        if self.secondary_axis is None:
            if self.secondary_spans:
                raise Qwen3_5MoeParallelError("secondary spans require a source axis")
        elif (
            self.secondary_axis == self.shard_axis
            or not 0 <= self.secondary_axis < len(self.global_shape)
            or not self.secondary_spans
            or any(
                span.stop > self.global_shape[self.secondary_axis]
                for span in self.secondary_spans
            )
        ):
            raise Qwen3_5MoeParallelError("secondary weight spans are invalid")
        expected = list(self.global_shape)
        expected[self.shard_axis] = sum(span.size for span in self.spans)
        if self.secondary_axis is not None:
            expected[self.secondary_axis] = sum(span.size for span in self.secondary_spans)
        if tuple(expected) != self.local_shape:
            raise Qwen3_5MoeParallelError("local shape differs from its exact source spans")


@dataclass(frozen=True, slots=True)
class Qwen3_5MoeParallelPlan:
    config: Qwen3_5MoeConfig
    topology: ParallelTopology
    partition: PartitionPlan
    semantic_plan_digest: str
    global_names: tuple[str, ...]
    bindings: tuple[MoeRankWeightBinding, ...]
    kv_heads: tuple[KVHeadOwnership, ...]
    gdn_states: tuple[GDNStateOwnership, ...]
    mtp_kv_heads: tuple[KVHeadOwnership, ...] = ()

    def __post_init__(self) -> None:
        if self.partition.topology_id != self.topology.topology_id:
            raise Qwen3_5MoeParallelError("partition and topology identities differ")
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
            raise Qwen3_5MoeParallelError("parallel inventory differs from configured tensors")
        if bool(self.mtp_kv_heads) != enabled:
            raise Qwen3_5MoeParallelError("MTP weights and KV ownership must be enabled together")

        assignments = {item.node_id: item for item in self.partition.assignments}
        placements = {item.rank: item for item in self.topology.placements}
        by_name: dict[str, list[MoeRankWeightBinding]] = defaultdict(list)
        for binding in self.bindings:
            assignment = assignments.get(binding.node_id)
            placement = placements.get(binding.rank)
            if assignment is None or placement is None:
                raise Qwen3_5MoeParallelError("weight binding references an unknown owner")
            if (
                binding.rank not in assignment.owner_ranks
                or binding.pipeline_stage != assignment.pipeline_stage
                or binding.pipeline_stage != placement.pipeline_stage
                or binding.tensor_rank != placement.tensor_rank
            ):
                raise Qwen3_5MoeParallelError("weight binding differs from graph placement")
            by_name[binding.tensor_name].append(binding)
        if tuple(sorted(by_name)) != self.global_names:
            raise Qwen3_5MoeParallelError("rank bindings leave the executable inventory open")
        for name, owners in by_name.items():
            exemplar = owners[0]
            expected_owners = (
                self.topology.expert_tensor_size
                if _is_unpacked_mtp_expert(name) and self.topology.ep_size > 1
                else self.topology.tp_size
            )
            if (
                len(owners) != expected_owners
                or len({item.rank for item in owners}) != expected_owners
            ):
                raise Qwen3_5MoeParallelError(f"{name!r} must bind {expected_owners} stage rank(s)")
            if any(
                item.node_id != exemplar.node_id
                or item.layout is not exemplar.layout
                or item.dtype is not exemplar.dtype
                or item.shard_axis != exemplar.shard_axis
                or item.global_shape != exemplar.global_shape
                or item.secondary_axis != exemplar.secondary_axis
                for item in owners
            ):
                raise Qwen3_5MoeParallelError(f"rank bindings disagree for {name!r}")
            if exemplar.layout is MoeWeightLayout.EXPERT:
                self._validate_expert_spans(name, owners)
            else:
                self._validate_spans(name, owners)

        text = self.config.graph.text
        assert text.moe is not None
        experts = text.moe.expert_count
        expected_experts = (text.layer_count + int(self.mtp_enabled)) * experts
        if len(self.partition.expert_assignments) != expected_experts:
            raise Qwen3_5MoeParallelError("every routed expert in every layer needs one owner")
        layers_per_stage = text.layer_count // self.topology.pp_size
        for layer in range(text.layer_count):
            node = f"text.decoder.layer.{layer:04d}"
            stage = layer // layers_per_stage
            stage_base = stage * self.topology.tp_size
            layer_owners = tuple(
                item for item in self.partition.expert_assignments if item.node_id == node
            )
            if tuple(item.expert_id for item in layer_owners) != tuple(range(experts)):
                raise Qwen3_5MoeParallelError("expert ids must be canonical and contiguous")
            if any(
                item.owner_rank
                != stage_base
                + (
                    (item.expert_id // (experts // self.topology.ep_size))
                    * self.topology.expert_tensor_size
                    if self.topology.ep_size > 1
                    else 0
                )
                for item in layer_owners
            ):
                raise Qwen3_5MoeParallelError("expert assignment differs from packed weight slices")

        for item in self.partition.expert_assignments:
            expected = tuple(range(
                item.owner_rank, item.owner_rank + self.topology.expert_tensor_size
            ))
            if item.expert_tensor_ranks != (
                expected
                if self.topology.expert_tensor_size > 1 or self.topology.ep_size == 1
                else ()
            ):
                raise Qwen3_5MoeParallelError("expert tensor shards must close the owner group")

        if self.mtp_enabled:
            mtp = tuple(
                item
                for item in self.partition.expert_assignments
                if item.node_id == "extension.mtp.layer.0000"
            )
            if tuple(item.expert_id for item in mtp) != tuple(range(experts)) or any(
                item.owner_rank
                != (
                    (item.expert_id // (experts // self.topology.ep_size))
                    * self.topology.expert_tensor_size
                    if self.topology.ep_size > 1
                    else 0
                )
                for item in mtp
            ):
                raise Qwen3_5MoeParallelError(
                    "MTP expert ownership differs from the target topology"
                )

        self._validate_state()

    @property
    def mtp_enabled(self) -> bool:
        return bool(self.mtp_kv_heads)

    def _validate_spans(self, name: str, owners: list[MoeRankWeightBinding]) -> None:
        exemplar = owners[0]
        if any(item.secondary_axis is not None for item in owners):
            raise Qwen3_5MoeParallelError("ordinary TP weights have one declared shard axis")
        dimension = exemplar.global_shape[exemplar.shard_axis]
        ordered = sorted(owners, key=lambda item: item.tensor_rank)
        if exemplar.layout is MoeWeightLayout.REPLICATED:
            full = (AxisSpan(0, dimension),)
            if any(item.spans != full for item in owners):
                raise Qwen3_5MoeParallelError(f"replica set is incomplete for {name!r}")
            return
        if exemplar.layout is MoeWeightLayout.KV_REPLICATED:
            ordered = sorted(owners, key=lambda item: item.tensor_rank)
            for item in ordered:
                heads = _kv_span(self.config, item.tensor_rank, self.topology.tp_size)
                dim = self.config.graph.text.full_attention.head_dim
                if item.spans != (AxisSpan(heads.start * dim, heads.stop * dim),):
                    raise Qwen3_5MoeParallelError(f"KV ownership differs from config for {name!r}")
            return
        intervals = sorted(span for item in owners for span in item.spans)
        cursor = 0
        for span in intervals:
            if span.start != cursor:
                raise Qwen3_5MoeParallelError(f"semantic spans overlap or leave a gap for {name!r}")
            cursor = span.stop
        if cursor != dimension:
            raise Qwen3_5MoeParallelError(f"semantic spans do not close {name!r}")

    def _validate_expert_spans(self, name: str, owners: list[MoeRankWeightBinding]) -> None:
        etp = self.topology.expert_tensor_size
        for item in owners:
            owner, lane = self.topology.expert_coordinates(item.rank)
            intermediate_axis = _expert_intermediate_axis(name, item.global_shape)
            dimension = item.global_shape[intermediate_axis]
            gate_up = name.endswith("gate_up_proj")
            width = dimension // (2 if gate_up else 1)
            part = _equal(width, lane, etp)
            intermediate_spans = (
                (part, AxisSpan(width + part.start, width + part.stop))
                if gate_up else (part,)
            )
            if _is_unpacked_mtp_expert(name):
                expected_axis, expected_spans = intermediate_axis, intermediate_spans
                other_axis, other_spans = None, ()
                expected_rank = owner * etp + lane
                expert_id = _mtp_expert_id(name)
                experts_per_owner = self.config.graph.text.moe.expert_count // self.topology.ep_size
                if owner != expert_id // experts_per_owner:
                    raise Qwen3_5MoeParallelError(
                        "MTP expert is bound outside its checkpoint owner"
                    )
            elif self.topology.ep_size == 1:
                expected_axis, expected_spans = intermediate_axis, intermediate_spans
                other_axis, other_spans = None, ()
                expected_rank = lane
            else:
                expected_axis = 0
                expected_spans = (
                    _equal(item.global_shape[0], owner, self.topology.ep_size),
                )
                other_axis = intermediate_axis if etp > 1 else None
                other_spans = intermediate_spans if etp > 1 else ()
                expected_rank = owner * etp + lane
            if (
                item.tensor_rank != expected_rank
                or item.shard_axis != expected_axis
                or item.spans != expected_spans
                or item.secondary_axis != other_axis
                or item.secondary_spans != other_spans
            ):
                raise Qwen3_5MoeParallelError(
                    "expert checkpoint spans differ from EP/ETP ownership"
                )

    def _validate_state(self) -> None:
        text, tp = self.config.graph.text, self.topology.tp_size
        gdn = text.gated_delta_net
        kv_by_layer: dict[int, list[KVHeadOwnership]] = defaultdict(list)
        for owner in self.kv_heads:
            kv_by_layer[owner.layer_index].append(owner)
        attention_layers = {
            i for i, kind in enumerate(text.layer_schedule) if kind is AttentionKind.FULL_ATTENTION
        }
        if set(kv_by_layer) != attention_layers:
            raise Qwen3_5MoeParallelError("KV owners differ from configured attention layers")
        for owners in (*kv_by_layer.values(), *((self.mtp_kv_heads,) if self.mtp_enabled else ())):
            if len(owners) != tp or len({owner.rank for owner in owners}) != tp:
                raise Qwen3_5MoeParallelError("KV owners do not close the stage")
            for owner in owners:
                expected = _kv_span(self.config, owner.rank % tp, tp)
                peers = tuple(item.rank for item in owners if item.head_span == expected)
                if owner.head_span != expected or owner.replica_ranks != peers:
                    raise Qwen3_5MoeParallelError(
                        "KV head or replica ownership differs from config"
                    )
        gdn_by_layer: dict[int, list[GDNStateOwnership]] = defaultdict(list)
        for owner in self.gdn_states:
            gdn_by_layer[owner.layer_index].append(owner)
        if set(gdn_by_layer) != set(range(text.layer_count)) - attention_layers:
            raise Qwen3_5MoeParallelError("GDN owners differ from configured recurrent layers")
        widths = (gdn.key_heads * gdn.key_head_dim,) * 2 + (gdn.value_heads * gdn.value_head_dim,)
        for owners in gdn_by_layer.values():
            if len(owners) != tp or len({owner.rank for owner in owners}) != tp:
                raise Qwen3_5MoeParallelError("GDN owners do not close the stage")
            for owner in owners:
                lane = owner.rank % tp
                channels = _segmented(widths, lane, tp)
                values = _equal(gdn.value_heads, lane, tp)
                if (
                    owner.key_head_span != _equal(gdn.key_heads, lane, tp)
                    or owner.value_head_span != values
                    or owner.convolution_channel_spans != channels
                    or owner.convolution_shape
                    != (sum(x.size for x in channels), gdn.convolution_kernel_size - 1)
                    or owner.recurrent_shape != (values.size, gdn.value_head_dim, gdn.key_head_dim)
                ):
                    raise Qwen3_5MoeParallelError("GDN ownership differs from configured geometry")

    def bindings_for_rank(self, rank: int) -> tuple[MoeRankWeightBinding, ...]:
        if not 0 <= rank < self.topology.world_size:
            raise Qwen3_5MoeParallelError("unknown rank")
        return tuple(item for item in self.bindings if item.rank == rank)


class Qwen3_5MoeRankWeights(RankLocalTensors[MoeRankWeightBinding]):
    """Read-only rank-local slices; routed expert payloads never reconstruct globally."""

    __slots__ = ()

    def __init__(
        self,
        *,
        rank: int,
        device: torch.device,
        inventory_digest: str,
        bindings: tuple[MoeRankWeightBinding, ...],
        tensors: dict[str, Tensor],
    ) -> None:
        try:
            super().__init__(
                rank=rank,
                device=device,
                inventory_digest=inventory_digest,
                bindings=bindings,
                tensors=tensors,
                dtypes={item.tensor_name: item.dtype for item in bindings},
            )
        except ValueError as error:
            raise Qwen3_5MoeParallelError(str(error)) from error


def plan_qwen3_5_moe_weights(
    config: Qwen3_5MoeConfig,
    weight_plan: Qwen3_5MoeWeightPlan,
    topology: ParallelTopology,
    *,
    include_mtp: bool = False,
) -> Qwen3_5MoeParallelPlan:
    """Place checkpoint experts by expert ID or paired intermediate TP slices."""

    if config.identity.model_type != "qwen3_5_moe" or weight_plan.model_type != "qwen3_5_moe":
        raise Qwen3_5MoeParallelError("parallel plan requires qwen3_5_moe")
    if weight_plan.architecture_identity.release_profile != config.identity.release_profile:
        raise Qwen3_5MoeParallelError("configuration and weight-plan identities differ")
    if type(include_mtp) is not bool or (include_mtp and topology.pp_size != 1):
        raise Qwen3_5MoeParallelError("MoE MTP planning requires PP1")

    plan_qwen3_5_moe_weight_headers(config, weight_plan.checkpoint_headers)
    text = config.graph.text
    assert text.moe is not None
    if include_mtp and text.mtp.hidden_layer_count != 1:
        raise Qwen3_5MoeParallelError("MTP execution requires one declared prediction layer")
    if text.layer_count % topology.pp_size or text.moe.expert_count % topology.ep_size:
        raise Qwen3_5MoeParallelError("layers and experts must divide their owner groups")
    architecture = config.architecture_plan()
    pp_size = topology.pp_size
    etp = topology.expert_tensor_size
    layers_per_stage = len(architecture.decoder_layers) // pp_size
    stage_ranks = {
        stage: tuple(item.rank for item in topology.placements if item.pipeline_stage == stage)
        for stage in range(pp_size)
    }
    node_stage = {architecture.token_embedding_node_id: 0}
    node_stage.update(
        {item.node_id: item.layer_index // layers_per_stage for item in architecture.decoder_layers}
    )
    node_stage[architecture.final_norm_node_id] = pp_size - 1
    node_stage[architecture.lm_head_node_id] = pp_size - 1
    if include_mtp:
        node_stage[architecture.mtp_root_node_id] = 0
        node_stage.update((layer.node_id, 0) for layer in architecture.mtp_layers)
    assignments = tuple(
        GraphAssignment(node, stage, stage_ranks[stage]) for node, stage in node_stage.items()
    )
    expert_assignments = tuple(
        ExpertAssignment(
            node_id=layer.node_id,
            expert_id=expert,
            owner_rank=(layer.layer_index // layers_per_stage) * topology.tp_size
            + expert // (text.moe.expert_count // topology.ep_size) * etp,
            expert_tensor_ranks=(
                tuple(range(
                    (layer.layer_index // layers_per_stage) * topology.tp_size
                    + expert // (text.moe.expert_count // topology.ep_size) * etp,
                    (layer.layer_index // layers_per_stage) * topology.tp_size
                    + (expert // (text.moe.expert_count // topology.ep_size) + 1) * etp,
                )) if etp > 1 or topology.ep_size == 1 else ()
            ),
        )
        for layer in architecture.decoder_layers
        for expert in range(text.moe.expert_count)
    )
    if include_mtp:
        expert_assignments += tuple(
            ExpertAssignment(
                node_id=architecture.mtp_layers[0].node_id,
                expert_id=expert,
                owner_rank=expert // (text.moe.expert_count // topology.ep_size) * etp,
                expert_tensor_ranks=tuple(range(
                    expert // (text.moe.expert_count // topology.ep_size) * etp,
                    (expert // (text.moe.expert_count // topology.ep_size) + 1) * etp,
                )) if etp > 1 or topology.ep_size == 1 else (),
            )
            for expert in range(text.moe.expert_count)
        )
    partition = PartitionPlan(
        partition_id=f"qwen3_5_moe:text:{topology.topology_id}",
        topology_id=topology.topology_id,
        assignments=assignments,
        expert_assignments=expert_assignments,
        pipeline_edges=tuple(
            PipelineEdge(
                edge_id=f"qwen3_5_moe.pp:{stage}->{stage + 1}:lane:{lane}",
                source_node_id=architecture.decoder_layers[
                    (stage + 1) * layers_per_stage - 1
                ].node_id,
                target_node_id=architecture.decoder_layers[(stage + 1) * layers_per_stage].node_id,
                source_rank=stage_ranks[stage][lane],
                target_rank=stage_ranks[stage + 1][lane],
                tensor_dtype=ScalarDType.BF16,
                tensor_layout="batch_token_hidden",
            )
            for stage in range(pp_size - 1)
            for lane in range(topology.tp_size)
        ),
    )

    names = (
        qwen3_5_moe_mtp_weight_policy(weight_plan).expected_names
        if include_mtp
        else qwen3_5_moe_text_weight_policy(weight_plan).expected_names
    )
    semantics = {item.name: item for item in weight_plan.tensor_semantics}
    placement = {item.rank: item for item in topology.placements}
    bindings: list[MoeRankWeightBinding] = []
    for name in names:
        semantic = semantics[name]
        stage = node_stage[semantic.node_id]
        layout, axis = _layout(name)
        tensor_expert = layout is MoeWeightLayout.EXPERT and topology.ep_size == 1
        if tensor_expert:
            axis = _expert_intermediate_axis(name, semantic.shape)
        dtype = torch.float32 if semantic.dtype == "F32" else torch.bfloat16
        ranks = stage_ranks[stage]
        if _is_unpacked_mtp_expert(name) and topology.ep_size > 1:
            owner = _mtp_expert_id(name) // (text.moe.expert_count // topology.ep_size)
            ranks = tuple(range(owner * etp, (owner + 1) * etp))
        for rank in ranks:
            tensor_rank = placement[rank].tensor_rank
            owner, lane = topology.expert_coordinates(rank)
            unpacked_expert = _is_unpacked_mtp_expert(name)
            spans = (
                _spans(
                    config,
                    layout=layout,
                    dimension=semantic.shape[axis],
                    tensor_rank=tensor_rank,
                    tp_size=topology.tp_size,
                )
            )
            secondary_axis = None
            secondary_spans: tuple[AxisSpan, ...] = ()
            if layout is MoeWeightLayout.EXPERT:
                intermediate_axis = _expert_intermediate_axis(name, semantic.shape)
                dimension = semantic.shape[intermediate_axis]
                width = dimension // (2 if name.endswith("gate_up_proj") else 1)
                part = _equal(width, lane, etp)
                intermediate_spans = (
                    (part, AxisSpan(width + part.start, width + part.stop))
                    if name.endswith("gate_up_proj") else (part,)
                )
                if tensor_expert or unpacked_expert:
                    axis, spans = intermediate_axis, intermediate_spans
                else:
                    spans = (_equal(semantic.shape[0], owner, topology.ep_size),)
                    if etp > 1:
                        secondary_axis, secondary_spans = intermediate_axis, intermediate_spans
            local_shape = list(semantic.shape)
            local_shape[axis] = sum(item.size for item in spans)
            if secondary_axis is not None:
                local_shape[secondary_axis] = sum(item.size for item in secondary_spans)
            bindings.append(
                MoeRankWeightBinding(
                    name,
                    semantic.node_id,
                    rank,
                    stage,
                    tensor_rank,
                    layout,
                    dtype,
                    axis,
                    spans,
                    semantic.shape,
                    tuple(local_shape),
                    secondary_axis,
                    secondary_spans,
                )
            )

    kv_heads: list[KVHeadOwnership] = []
    gdn_states: list[GDNStateOwnership] = []
    mtp_kv_heads: list[KVHeadOwnership] = []
    gdn = config.graph.text.gated_delta_net
    key_width = gdn.key_heads * gdn.key_head_dim
    value_width = gdn.value_heads * gdn.value_head_dim
    for layer in architecture.decoder_layers:
        stage = layer.layer_index // layers_per_stage
        if layer.attention is AttentionKind.FULL_ATTENTION:
            for rank in stage_ranks[stage]:
                tensor_rank = placement[rank].tensor_rank
                span = _kv_span(config, tensor_rank, topology.tp_size)
                replicas = tuple(
                    peer
                    for peer in stage_ranks[stage]
                    if _kv_span(config, placement[peer].tensor_rank, topology.tp_size) == span
                )
                kv_heads.append(KVHeadOwnership(layer.layer_index, rank, span, replicas))
            continue
        for rank in stage_ranks[stage]:
            tensor_rank = placement[rank].tensor_rank
            channels = _segmented(
                (key_width, key_width, value_width), tensor_rank, topology.tp_size
            )
            key_heads = _equal(gdn.key_heads, tensor_rank, topology.tp_size)
            value_heads = _equal(gdn.value_heads, tensor_rank, topology.tp_size)
            gdn_states.append(
                GDNStateOwnership(
                    layer.layer_index,
                    rank,
                    cast(tuple[AxisSpan, AxisSpan, AxisSpan], channels),
                    key_heads,
                    value_heads,
                    (sum(item.size for item in channels), gdn.convolution_kernel_size - 1),
                    (value_heads.size, gdn.value_head_dim, gdn.key_head_dim),
                )
            )

    if include_mtp:
        for rank in stage_ranks[0]:
            tensor_rank = placement[rank].tensor_rank
            span = _kv_span(config, tensor_rank, topology.tp_size)
            replicas = tuple(
                peer
                for peer in stage_ranks[0]
                if _kv_span(config, placement[peer].tensor_rank, topology.tp_size) == span
            )
            mtp_kv_heads.append(KVHeadOwnership(0, rank, span, replicas))

    return Qwen3_5MoeParallelPlan(
        config,
        topology,
        partition,
        weight_plan.semantic_plan_digest,
        names,
        tuple(bindings),
        tuple(kv_heads),
        tuple(gdn_states),
        tuple(mtp_kv_heads),
    )


def load_qwen3_5_moe_rank_weights(
    *,
    inventory: CheckpointInventory,
    weight_plan: Qwen3_5MoeWeightPlan,
    parallel_plan: Qwen3_5MoeParallelPlan,
    rank: int,
    device: str | torch.device,
    progress: Callable[[str, int, int], None] | None = None,
) -> Qwen3_5MoeRankWeights:
    """Stream only exact source ranges assigned to one rank."""

    if weight_plan.inventory_digest != inventory.inventory_digest:
        raise Qwen3_5MoeParallelError("weight plan differs from checkpoint evidence")
    if parallel_plan.semantic_plan_digest != weight_plan.semantic_plan_digest:
        raise Qwen3_5MoeParallelError("parallel plan differs from checkpoint semantics")
    bindings = parallel_plan.bindings_for_rank(rank)
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise Qwen3_5MoeParallelError("CUDA was requested but is unavailable")
    placement = next(item for item in parallel_plan.topology.placements if item.rank == rank)
    if target.type == "cuda" and target.index != placement.device_index:
        raise Qwen3_5MoeParallelError("rank was assigned to another CUDA device")
    records = {item.name: item for item in inventory.tensors}
    by_shard: dict[str, list[MoeRankWeightBinding]] = defaultdict(list)
    for binding in bindings:
        record = records.get(binding.tensor_name)
        expected_dtype = "F32" if binding.dtype is torch.float32 else "BF16"
        if record is None or record.dtype != expected_dtype or record.shape != binding.global_shape:
            raise Qwen3_5MoeParallelError(f"attested header differs for {binding.tensor_name!r}")
        by_shard[record.shard_path].append(binding)

    mtp_names = set(weight_plan.mtp_extension_names)
    sections = {
        binding.tensor_name: "mtp" if binding.tensor_name in mtp_names else "base"
        for binding in bindings
    }
    totals: dict[str, int] = defaultdict(int)
    completed_by_section: dict[str, int] = defaultdict(int)
    for binding in bindings:
        totals[sections[binding.tensor_name]] += torch.Size(binding.local_shape).numel() * (
            4 if binding.dtype is torch.float32 else 2
        )
    if progress is not None:
        for section in ("base", "mtp"):
            if totals[section]:
                progress(section, 0, totals[section])

    loaded: dict[str, Tensor] = {}
    for shard, shard_bindings in sorted(by_shard.items()):
        try:
            path = confined_checkpoint_path(inventory.model_root, shard)
        except ValueError as error:
            raise Qwen3_5MoeParallelError(str(error)) from error
        with safe_open(path, framework="pt", device="cpu") as raw:
            reader = cast(SafeTensorReader, raw)
            available = set(reader.keys())
            for binding in sorted(
                shard_bindings, key=lambda item: records[item.tensor_name].offset_ordinal
            ):
                if binding.tensor_name not in available:
                    raise Qwen3_5MoeParallelError(f"checkpoint shard omits {binding.tensor_name!r}")
                try:
                    loaded[binding.tensor_name] = materialize_axis_spans(
                        reader,
                        binding,
                        dtype=binding.dtype,
                        device=target,
                        secondary_axis=binding.secondary_axis,
                        secondary_spans=binding.secondary_spans,
                    )
                except ValueError as error:
                    raise Qwen3_5MoeParallelError(str(error)) from error
                section = sections[binding.tensor_name]
                completed_by_section[section] += torch.Size(binding.local_shape).numel() * (
                    4 if binding.dtype is torch.float32 else 2
                )
                if progress is not None:
                    progress(section, completed_by_section[section], totals[section])
    return Qwen3_5MoeRankWeights(
        rank=rank,
        device=target,
        inventory_digest=inventory.inventory_digest,
        bindings=bindings,
        tensors=loaded,
    )


def _layout(name: str) -> tuple[MoeWeightLayout, int]:
    if name in {"model.language_model.embed_tokens.weight", "lm_head.weight"}:
        return MoeWeightLayout.VOCAB, 0
    if name.endswith(("mlp.experts.gate_up_proj", "mlp.experts.down_proj")):
        return MoeWeightLayout.EXPERT, 0
    if _is_unpacked_mtp_expert(name):
        return MoeWeightLayout.EXPERT, 0
    if name == "mtp.fc.weight":
        return MoeWeightLayout.COLUMN, 0
    if name.endswith("mlp.gate.weight"):
        return MoeWeightLayout.REPLICATED, 0
    if name.endswith("mlp.shared_expert_gate.weight"):
        return MoeWeightLayout.REPLICATED, 0
    if name.endswith(("mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight")):
        return MoeWeightLayout.COLUMN, 0
    if name.endswith("mlp.shared_expert.down_proj.weight"):
        return MoeWeightLayout.ROW, 1
    if name in {
        "model.language_model.norm.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    } or name.endswith(
        (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "q_norm.weight",
            "k_norm.weight",
            "linear_attn.norm.weight",
        )
    ):
        return MoeWeightLayout.REPLICATED, 0
    if ".self_attn." in name:
        if name.endswith("q_proj.weight"):
            return MoeWeightLayout.COLUMN, 0
        if name.endswith(("k_proj.weight", "v_proj.weight")):
            return MoeWeightLayout.KV_REPLICATED, 0
        if name.endswith("o_proj.weight"):
            return MoeWeightLayout.ROW, 1
    if ".linear_attn." in name:
        if name.endswith(("in_proj_qkv.weight", "conv1d.weight")):
            return MoeWeightLayout.GDN_QKV, 0
        if name.endswith(
            ("A_log", "dt_bias", "in_proj_a.weight", "in_proj_b.weight", "in_proj_z.weight")
        ):
            return MoeWeightLayout.COLUMN, 0
        if name.endswith("out_proj.weight"):
            return MoeWeightLayout.ROW, 1
    raise Qwen3_5MoeParallelError(f"no audited MoE layout exists for {name!r}")


def _expert_intermediate_axis(name: str, shape: tuple[int, ...]) -> int:
    """Packed [E, O, I] and unpacked [O, I] experts share the matrix axes."""
    return len(shape) - (1 if name.endswith(("down_proj", "down_proj.weight")) else 2)


def _is_unpacked_mtp_expert(name: str) -> bool:
    return name.startswith("mtp.layers.0.mlp.experts.") and name.endswith("_proj.weight")


def _mtp_expert_id(name: str) -> int:
    try:
        return int(name.split(".experts.", 1)[1].split(".", 1)[0])
    except (IndexError, ValueError) as error:
        raise Qwen3_5MoeParallelError(f"invalid MTP expert tensor name {name!r}") from error


def _kv_span(config: Qwen3_5MoeConfig, lane: int, tp: int) -> AxisSpan:
    heads = config.graph.text.full_attention.key_value_heads
    if tp <= heads:
        return _equal(heads, lane, tp)
    if tp % heads:
        raise Qwen3_5MoeParallelError("TP must divide KV heads or replicate each head equally")
    head = lane // (tp // heads)
    return AxisSpan(head, head + 1)


def _spans(
    config: Qwen3_5MoeConfig,
    *,
    layout: MoeWeightLayout,
    dimension: int,
    tensor_rank: int,
    tp_size: int,
) -> tuple[AxisSpan, ...]:
    if layout is MoeWeightLayout.REPLICATED:
        return (AxisSpan(0, dimension),)
    if layout is MoeWeightLayout.KV_REPLICATED:
        heads = _kv_span(config, tensor_rank, tp_size)
        width = config.graph.text.full_attention.head_dim
        return (AxisSpan(heads.start * width, heads.stop * width),)
    if layout is MoeWeightLayout.GDN_QKV:
        gdn = config.graph.text.gated_delta_net
        return _segmented(
            (
                gdn.key_heads * gdn.key_head_dim,
                gdn.key_heads * gdn.key_head_dim,
                gdn.value_heads * gdn.value_head_dim,
            ),
            tensor_rank,
            tp_size,
        )
    return (_equal(dimension, tensor_rank, tp_size),)


def _segmented(
    dimensions: tuple[int, int, int], tensor_rank: int, tp_size: int
) -> tuple[AxisSpan, ...]:
    result: list[AxisSpan] = []
    offset = 0
    for dimension in dimensions:
        span = _equal(dimension, tensor_rank, tp_size)
        result.append(AxisSpan(offset + span.start, offset + span.stop))
        offset += dimension
    return tuple(result)


def _equal(dimension: int, rank: int, size: int) -> AxisSpan:
    try:
        return equal_axis_span(dimension, rank, size)
    except ValueError as error:
        raise Qwen3_5MoeParallelError(str(error)) from error


__all__ = (
    "MoeRankWeightBinding",
    "MoeWeightLayout",
    "Qwen3_5MoeParallelError",
    "Qwen3_5MoeParallelPlan",
    "Qwen3_5MoeRankWeights",
    "load_qwen3_5_moe_rank_weights",
    "plan_qwen3_5_moe_tp4_pp2_ep4",
    "plan_qwen3_5_moe_weights",
)


def plan_qwen3_5_moe_tp4_pp2_ep4(
    config: Qwen3_5MoeConfig,
    weight_plan: Qwen3_5MoeWeightPlan,
    topology: ParallelTopology,
) -> Qwen3_5MoeParallelPlan:
    """Compatibility spelling retained for callers that explicitly request PP2."""

    if topology.pp_size != 2 or topology.ep_size != topology.tp_size:
        raise Qwen3_5MoeParallelError("the legacy planner spelling requires PP2")
    return plan_qwen3_5_moe_weights(config, weight_plan, topology)
