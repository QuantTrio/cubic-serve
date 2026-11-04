"""Exact eager TP x PP reference execution for dense ``qwen3_5``.

Weights and hybrid state remain rank-local.  Before each canonical model op,
the tensor ranks temporarily all-gather that op's complete operand.  This
preserves the single-rank reduction shape and therefore permits bit-exact
guardrail comparison.  It is deliberately not a performance TP candidate and
never retains a reconstructed full model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from cubic_serve.contracts.execution import CollectiveStep
from cubic_serve.executor.parallel import EagerNcclPartition, SealedPartition, seal_partition
from cubic_serve.executor.runner import require_reference_execution
from cubic_serve.models.parallel_weights import reconstruct_axis_spans
from cubic_serve.models.qwen3_5.model import (
    AttentionKind,
    Qwen3_5Config,
    Qwen3_5LayerCache,
    Qwen3_5MTPReferenceCache,
    Qwen3_5ReferenceCache,
    Qwen3_5ReferenceModel,
)
from cubic_serve.models.qwen3_5.weights import (
    AxisSpan,
    GDNStateOwnership,
    KVHeadOwnership,
    Qwen3_5ParallelWeightPlan,
    Qwen3_5RankWeights,
    WeightShardLayout,
)
from cubic_serve.ops.eager.hybrid import interleaved_mrope_cos_sin

_RECIPE_LANE = "qwen3_5-canonical-reconstruct-v1"
_MTP_RECIPE_LANE = "qwen3_5-mtp-canonical-reconstruct-v1"


class Qwen3_5DistributedReferenceError(RuntimeError):
    """The exact distributed-reference recipe is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class Qwen3_5DistributedCache:
    """One rank's immutable, stage-local hybrid state shard."""

    sequence_length: int
    batch_size: int
    rank: int
    layer_start: int
    layer_end: int
    layers: tuple[Qwen3_5LayerCache, ...]

    def __post_init__(self) -> None:
        if min(self.sequence_length, self.rank, self.layer_start) < 0 or self.batch_size <= 0:
            raise ValueError("distributed cache identity is invalid")
        if self.layer_end <= self.layer_start:
            raise ValueError("distributed cache requires a non-empty stage")
        if len(self.layers) != self.layer_end - self.layer_start:
            raise ValueError("distributed cache does not close its stage layer range")


@dataclass(frozen=True, slots=True)
class Qwen3_5DistributedStep:
    """One exact-M forward result; only the last PP stage owns logits."""

    logits: Tensor | None
    hidden_states: Tensor | None
    verification_logits: Tensor | None
    cache: Qwen3_5DistributedCache
    max_reconstructed_operand_bytes: int


@dataclass(frozen=True, slots=True)
class Qwen3_5DistributedMTPCache:
    """One rank's private shard of the MTP full-attention KV."""

    sequence_length: int
    batch_size: int
    rank: int
    layer: Qwen3_5LayerCache


@dataclass(frozen=True, slots=True)
class Qwen3_5DistributedMTPStep:
    logits: Tensor
    hidden_states: Tensor
    cache: Qwen3_5DistributedMTPCache
    max_reconstructed_operand_bytes: int


def seal_qwen3_5_distributed_reference(
    config: Qwen3_5Config,
    parallel_plan: Qwen3_5ParallelWeightPlan,
) -> SealedPartition:
    """Add the reusable, fully ordered reference collective recipe."""

    if config.identity.model_type != "qwen3_5":
        raise Qwen3_5DistributedReferenceError("distributed reference requires dense qwen3_5")
    if parallel_plan.topology.ep_size != 1:
        raise Qwen3_5DistributedReferenceError("dense reference requires EP1")

    assignments = {item.node_id: item for item in parallel_plan.partition.assignments}
    architecture = config.architecture_plan()
    if any(layer.node_id not in assignments for layer in architecture.decoder_layers):
        raise Qwen3_5DistributedReferenceError("parallel plan omits a decoder layer")

    sequence_by_stage = [0] * parallel_plan.topology.pp_size
    collectives: list[CollectiveStep] = []

    def add(stage: int, step_id: str) -> None:
        collectives.append(
            CollectiveStep(
                step_id=step_id,
                sequence_no=sequence_by_stage[stage],
                group_id=f"tp:{stage}",
                operation="all_gather",
                lane_id=_RECIPE_LANE,
            )
        )
        sequence_by_stage[stage] += 1

    add(0, _weight_step_id("model.language_model.embed_tokens.weight"))
    for layer in architecture.decoder_layers:
        stage = assignments[layer.node_id].pipeline_stage
        for state_name in _state_names(layer.layer_index, layer.attention):
            add(stage, _state_step_id(layer.layer_index, state_name))
        for weight_name in _layer_weight_sequence(layer.layer_index, layer.attention):
            add(stage, _weight_step_id(weight_name))
    add(parallel_plan.topology.pp_size - 1, _weight_step_id("lm_head.weight"))

    partition = replace(
        parallel_plan.partition,
        partition_id=parallel_plan.partition.partition_id + ":eager-reference-reconstruct-v1",
        collectives=tuple(collectives),
    )
    return seal_partition(parallel_plan.topology, partition)


def _mtp_collectives(
    config: Qwen3_5Config,
    parallel_plan: Qwen3_5ParallelWeightPlan,
) -> tuple[CollectiveStep, ...]:
    """Build the fixed TP-only MTP oracle lane."""

    if config.identity.model_type != "qwen3_5" or not parallel_plan.mtp_enabled:
        raise Qwen3_5DistributedReferenceError("MTP oracle requires an enabled qwen3_5 plan")
    topology = parallel_plan.topology
    if topology.pp_size != 1 or topology.ep_size != 1:
        raise Qwen3_5DistributedReferenceError("MTP oracle currently requires TP-only execution")
    assignments = {item.node_id: item for item in parallel_plan.partition.assignments}
    layer_node = config.architecture_plan().mtp_layers[0].node_id
    if layer_node not in assignments:
        raise Qwen3_5DistributedReferenceError("MTP parallel plan omits its decoder layer")

    collectives: list[CollectiveStep] = []
    for sequence_no, step_id in enumerate(
        (
            _mtp_state_step_id("key"),
            _mtp_state_step_id("value"),
            *(_mtp_weight_step_id(name) for name in _mtp_weight_sequence()),
        )
    ):
        collectives.append(
            CollectiveStep(
                step_id=step_id,
                sequence_no=sequence_no,
                group_id="tp:0",
                operation="all_gather",
                lane_id=_MTP_RECIPE_LANE,
            )
        )
    return tuple(collectives)


def seal_qwen3_5_target_mtp_distributed_reference(
    config: Qwen3_5Config,
    parallel_plan: Qwen3_5ParallelWeightPlan,
) -> SealedPartition:
    """Seal serialized target and MTP lanes over one TP process group."""

    target = seal_qwen3_5_distributed_reference(config, parallel_plan)
    partition = replace(
        parallel_plan.partition,
        partition_id=parallel_plan.partition.partition_id
        + ":target-mtp-eager-reference-reconstruct-v1",
        collectives=(*target.partition.collectives, *_mtp_collectives(config, parallel_plan)),
    )
    return seal_partition(parallel_plan.topology, partition)


class _CanonicalWeightWindow:
    """Reconstruct at most the currently requested full weight operand."""

    def __init__(
        self,
        rank_weights: Qwen3_5RankWeights,
        parallel_plan: Qwen3_5ParallelWeightPlan,
        runtime: EagerNcclPartition,
        *,
        weight_step_id: Callable[[str], str] | None = None,
    ) -> None:
        self.device = rank_weights.device
        self.inventory_digest = rank_weights.inventory_digest
        self.semantic_plan_digest = parallel_plan.semantic_plan_digest
        self._rank = rank_weights.rank
        self._local = dict(rank_weights.named_tensors())
        self._binding = {item.tensor_name: item for item in rank_weights.bindings}
        self._all_bindings = parallel_plan.bindings
        self._runtime = runtime
        self._weight_step_id = weight_step_id or _weight_step_id
        self._active = False
        self.peak_operand_bytes = 0

    @property
    def tensor_count(self) -> int:
        return len(self._local)

    def named_tensors(self) -> tuple[tuple[str, Tensor], ...]:
        """Expose only resident shards; full-policy validation therefore rejects this view."""

        return tuple((name, self._local[name]) for name in sorted(self._local))

    def open(self) -> None:
        if self._active:
            raise Qwen3_5DistributedReferenceError("weight reconstruction window is re-entered")
        self._active = True
        self.peak_operand_bytes = 0

    def close(self) -> None:
        self._active = False

    def __getitem__(self, name: str) -> Tensor:
        if not self._active:
            raise Qwen3_5DistributedReferenceError(
                "full operands may exist only inside one reference forward"
            )
        try:
            local = self._local[name]
            binding = self._binding[name]
        except KeyError as error:
            raise Qwen3_5DistributedReferenceError(
                f"rank {self._rank} requested an unowned weight {name!r}"
            ) from error
        if binding.layout is WeightShardLayout.REPLICATED:
            return local

        peers = tuple(
            sorted(
                (item for item in self._all_bindings if item.tensor_name == name),
                key=lambda item: item.tensor_rank,
            )
        )
        gathered = self._runtime.tp_all_gather(
            self._weight_step_id(name),
            local,
            dim=binding.shard_axis,
        )
        full = _reconstruct_equal_shards(
            gathered,
            axis=binding.shard_axis,
            local_size=binding.local_shape[binding.shard_axis],
            global_size=binding.global_shape[binding.shard_axis],
            spans_by_rank=tuple(item.spans for item in peers),
        )
        if tuple(full.shape) != binding.global_shape or full.dtype is not torch.bfloat16:
            raise Qwen3_5DistributedReferenceError(
                f"reconstructed weight {name!r} differs from its global binding"
            )
        self.peak_operand_bytes = max(
            self.peak_operand_bytes,
            full.numel() * full.element_size(),
        )
        return full


class Qwen3_5DistributedReferenceModel:
    """Stage-local exact reference over rank-local Qwen3.8-27B shards."""

    def __init__(
        self,
        *,
        config: Qwen3_5Config,
        weights: Qwen3_5RankWeights,
        parallel_plan: Qwen3_5ParallelWeightPlan,
        runtime: EagerNcclPartition,
    ) -> None:
        expected = (
            seal_qwen3_5_target_mtp_distributed_reference(config, parallel_plan)
            if parallel_plan.mtp_enabled
            else seal_qwen3_5_distributed_reference(config, parallel_plan)
        )
        if (
            runtime.sealed.topology_digest != expected.topology_digest
            or runtime.sealed.partition_digest != expected.partition_digest
        ):
            raise Qwen3_5DistributedReferenceError(
                "NCCL runtime was not initialized from the exact Qwen reference recipe"
            )
        if weights.rank != runtime.rank or weights.device != runtime.device:
            raise Qwen3_5DistributedReferenceError("rank weights and NCCL placement differ")

        placement = parallel_plan.topology.placements[runtime.rank]
        owned = tuple(
            layer.layer_index
            for layer in config.architecture_plan().decoder_layers
            if placement.rank
            in next(
                assignment.owner_ranks
                for assignment in parallel_plan.partition.assignments
                if assignment.node_id == layer.node_id
            )
        )
        if not owned or owned != tuple(range(owned[0], owned[-1] + 1)):
            raise Qwen3_5DistributedReferenceError("PP stage layers must be contiguous")

        self.config = config
        self.parallel_plan = parallel_plan
        self.runtime = runtime
        self.rank = runtime.rank
        self.pipeline_stage = placement.pipeline_stage
        self.tensor_rank = placement.tensor_rank
        self.is_first_stage = self.pipeline_stage == 0
        self.is_last_stage = self.pipeline_stage == parallel_plan.topology.pp_size - 1
        self.layer_start = owned[0]
        self.layer_end = owned[-1] + 1
        self._weights = _CanonicalWeightWindow(weights, parallel_plan, runtime)
        self._canonical = Qwen3_5ReferenceModel(config=config, weights=self._weights)

    def empty_cache(self, batch_size: int = 1) -> Qwen3_5DistributedCache:
        if batch_size <= 0:
            raise ValueError("cache batch size must be positive")
        text = self.config.graph.text
        layers: list[Qwen3_5LayerCache] = []
        for layer_index in range(self.layer_start, self.layer_end):
            if text.layer_schedule[layer_index] is AttentionKind.GATED_DELTA_NET:
                gdn_owner = self._gdn_owner(layer_index)
                layers.append(
                    Qwen3_5LayerCache(
                        convolution=torch.zeros(
                            batch_size,
                            gdn_owner.convolution_shape[0],
                            gdn_owner.convolution_shape[1],
                            dtype=torch.bfloat16,
                            device=self.runtime.device,
                        ),
                        recurrent=torch.zeros(
                            batch_size,
                            *gdn_owner.recurrent_shape,
                            dtype=self._canonical.recurrent_state_dtype,
                            device=self.runtime.device,
                        ),
                    )
                )
            else:
                kv_owner = self._kv_owner(layer_index)
                shape = (
                    batch_size,
                    kv_owner.head_span.size,
                    0,
                    text.full_attention.head_dim,
                )
                layers.append(
                    Qwen3_5LayerCache(
                        key=torch.empty(shape, dtype=torch.bfloat16, device=self.runtime.device),
                        value=torch.empty(shape, dtype=torch.bfloat16, device=self.runtime.device),
                    )
                )
        return Qwen3_5DistributedCache(
            sequence_length=0,
            batch_size=batch_size,
            rank=self.rank,
            layer_start=self.layer_start,
            layer_end=self.layer_end,
            layers=tuple(layers),
        )

    def forward(
        self,
        input_ids: Tensor,
        *,
        cache: Qwen3_5DistributedCache | None = None,
        position_ids: Tensor | None = None,
        epoch_id: str,
        return_all_logits: bool = False,
        layer_observer: Callable[[int, str, Tensor], None] | None = None,
    ) -> Qwen3_5DistributedStep:
        require_reference_execution()
        text = self.config.graph.text
        if input_ids.dtype is not torch.int64 or input_ids.ndim != 2 or input_ids.numel() == 0:
            raise TypeError("input_ids must be a non-empty rank-two torch.int64 tensor")
        if input_ids.device != self.runtime.device:
            raise ValueError("input_ids must reside on this rank device")
        if bool(((input_ids < 0) | (input_ids >= text.vocabulary_size)).any().item()):
            raise ValueError("input_ids contain an out-of-vocabulary token")
        batch_size, token_count = input_ids.shape
        active_cache = cache or self.empty_cache(batch_size)
        self._validate_cache(active_cache, batch_size)

        if position_ids is None:
            positions = torch.arange(
                active_cache.sequence_length,
                active_cache.sequence_length + token_count,
                dtype=torch.int64,
                device=input_ids.device,
            )
            position_ids = positions.view(1, 1, token_count).expand(3, batch_size, token_count)
        elif (
            position_ids.dtype is not torch.int64
            or tuple(position_ids.shape) != (3, batch_size, token_count)
            or position_ids.device != input_ids.device
        ):
            raise ValueError("position_ids must be int64 [3, batch, tokens] on the rank device")
        rotary_dim = int(text.full_attention.head_dim * text.mrope.partial_rotary_factor)
        cosine, sine = interleaved_mrope_cos_sin(
            position_ids,
            rotary_dim=rotary_dim,
            mrope_section=text.mrope.section,
            rope_theta=text.mrope.rope_theta,
            output_dtype=torch.bfloat16,
        )

        next_layers: list[Qwen3_5LayerCache] = []
        logits: Tensor | None = None
        hidden_states: Tensor | None = None
        verification_logits: Tensor | None = None
        self._weights.open()
        distributed_moe = getattr(self, "_moe", None)
        if distributed_moe is not None:
            distributed_moe.observer = layer_observer
        try:
            epoch = (
                self.runtime.collective_epoch(epoch_id, lane_id=_RECIPE_LANE)
                if self.parallel_plan.mtp_enabled
                else self.runtime.collective_epoch(epoch_id)
            )
            with epoch:
                if self.is_first_stage:
                    hidden = self._canonical.embed_input_ids(input_ids)
                else:
                    hidden = self.runtime.pipeline_recv(
                        self._pipeline_edge_id(self.pipeline_stage - 1),
                        (batch_size, token_count, text.hidden_size),
                    )

                for local_index, layer_index in enumerate(range(self.layer_start, self.layer_end)):
                    full_previous = self._restore_layer_cache(
                        layer_index,
                        active_cache.layers[local_index],
                    )
                    hidden, full_next = self._canonical.forward_layer(
                        hidden,
                        layer_index=layer_index,
                        previous=full_previous,
                        cosine=cosine,
                        sine=sine,
                        previous_length=active_cache.sequence_length,
                        observer=(
                            None
                            if layer_observer is None
                            else lambda phase, value, layer=layer_index: layer_observer(
                                layer, phase, value
                            )
                        ),
                    )
                    next_layers.append(self._slice_layer_cache(layer_index, full_next))
                    del full_previous, full_next

                if self.is_last_stage:
                    hidden_states = self._canonical.normalize_hidden(hidden)
                    projected = self._canonical.project_normalized_logits(
                        hidden_states if return_all_logits else hidden_states[:, -1:]
                    )
                    logits = projected[:, -1:]
                    verification_logits = projected if return_all_logits else None
                else:
                    self.runtime.pipeline_send(
                        self._pipeline_edge_id(self.pipeline_stage), hidden.contiguous()
                    )
        finally:
            if distributed_moe is not None:
                distributed_moe.observer = None
            self._weights.close()

        return Qwen3_5DistributedStep(
            logits=logits,
            hidden_states=hidden_states,
            verification_logits=verification_logits,
            cache=Qwen3_5DistributedCache(
                sequence_length=active_cache.sequence_length + token_count,
                batch_size=batch_size,
                rank=self.rank,
                layer_start=self.layer_start,
                layer_end=self.layer_end,
                layers=tuple(next_layers),
            ),
            max_reconstructed_operand_bytes=self._weights.peak_operand_bytes,
        )

    @staticmethod
    def _cache_collective_step(layer_index: int, state_name: str) -> str:
        return _state_step_id(layer_index, state_name)

    def _restore_layer_cache(
        self,
        layer_index: int,
        local: Qwen3_5LayerCache,
    ) -> Qwen3_5LayerCache:
        if self.config.graph.text.layer_schedule[layer_index] is AttentionKind.GATED_DELTA_NET:
            if local.convolution is None or local.recurrent is None:
                raise Qwen3_5DistributedReferenceError("GDN state shard is incomplete")
            gdn_owners = self._gdn_owners(layer_index)
            convolution = _reconstruct_equal_shards(
                self.runtime.tp_all_gather(
                    self._cache_collective_step(layer_index, "convolution"),
                    local.convolution,
                    dim=1,
                ),
                axis=1,
                local_size=local.convolution.shape[1],
                global_size=sum(
                    span.size for owner in gdn_owners for span in owner.convolution_channel_spans
                ),
                spans_by_rank=tuple(owner.convolution_channel_spans for owner in gdn_owners),
            )
            recurrent = _reconstruct_equal_shards(
                self.runtime.tp_all_gather(
                    self._cache_collective_step(layer_index, "recurrent"),
                    local.recurrent,
                    dim=1,
                ),
                axis=1,
                local_size=local.recurrent.shape[1],
                global_size=sum(owner.value_head_span.size for owner in gdn_owners),
                spans_by_rank=tuple((owner.value_head_span,) for owner in gdn_owners),
            )
            return Qwen3_5LayerCache(convolution=convolution, recurrent=recurrent)

        if local.key is None or local.value is None:
            raise Qwen3_5DistributedReferenceError("attention KV shard is incomplete")
        kv_owners = self._kv_owners(layer_index)
        key = _reconstruct_equal_shards(
            self.runtime.tp_all_gather(
                self._cache_collective_step(layer_index, "key"), local.key, dim=1
            ),
            axis=1,
            local_size=local.key.shape[1],
            global_size=sum(owner.head_span.size for owner in kv_owners),
            spans_by_rank=tuple((owner.head_span,) for owner in kv_owners),
        )
        value = _reconstruct_equal_shards(
            self.runtime.tp_all_gather(
                self._cache_collective_step(layer_index, "value"), local.value, dim=1
            ),
            axis=1,
            local_size=local.value.shape[1],
            global_size=sum(owner.head_span.size for owner in kv_owners),
            spans_by_rank=tuple((owner.head_span,) for owner in kv_owners),
        )
        if key.shape[2] == 0:
            return Qwen3_5LayerCache()
        return Qwen3_5LayerCache(key=key, value=value)

    def _slice_layer_cache(
        self,
        layer_index: int,
        full: Qwen3_5LayerCache,
    ) -> Qwen3_5LayerCache:
        if self.config.graph.text.layer_schedule[layer_index] is AttentionKind.GATED_DELTA_NET:
            gdn_owner = self._gdn_owner(layer_index)
            if full.convolution is None or full.recurrent is None:
                raise Qwen3_5DistributedReferenceError("canonical GDN state is incomplete")
            convolution = torch.cat(
                tuple(
                    full.convolution.narrow(1, span.start, span.size)
                    for span in gdn_owner.convolution_channel_spans
                ),
                dim=1,
            )
            recurrent = full.recurrent.narrow(
                1, gdn_owner.value_head_span.start, gdn_owner.value_head_span.size
            ).clone(memory_format=torch.contiguous_format)
            return Qwen3_5LayerCache(convolution=convolution, recurrent=recurrent)

        kv_owner = self._kv_owner(layer_index)
        if full.key is None or full.value is None:
            raise Qwen3_5DistributedReferenceError("canonical attention KV is incomplete")
        return Qwen3_5LayerCache(
            key=full.key.narrow(1, kv_owner.head_span.start, kv_owner.head_span.size).clone(
                memory_format=torch.contiguous_format
            ),
            value=full.value.narrow(1, kv_owner.head_span.start, kv_owner.head_span.size).clone(
                memory_format=torch.contiguous_format
            ),
        )

    def _validate_cache(self, cache: Qwen3_5DistributedCache, batch_size: int) -> None:
        if not isinstance(cache, Qwen3_5DistributedCache):
            raise TypeError("cache must be a Qwen3_5DistributedCache")
        if (
            cache.rank != self.rank
            or cache.batch_size != batch_size
            or cache.layer_start != self.layer_start
            or cache.layer_end != self.layer_end
        ):
            raise ValueError("distributed cache belongs to another rank, batch, or PP stage")
        for offset, layer in enumerate(cache.layers):
            layer_index = self.layer_start + offset
            if self.config.graph.text.layer_schedule[layer_index] is AttentionKind.GATED_DELTA_NET:
                gdn_owner = self._gdn_owner(layer_index)
                _require_tensor(
                    layer.convolution,
                    dtype=torch.bfloat16,
                    shape=(batch_size, *gdn_owner.convolution_shape),
                    device=self.runtime.device,
                )
                _require_tensor(
                    layer.recurrent,
                    dtype=self._canonical.recurrent_state_dtype,
                    shape=(batch_size, *gdn_owner.recurrent_shape),
                    device=self.runtime.device,
                )
                if layer.key is not None or layer.value is not None:
                    raise ValueError("GDN cache shard contains attention KV")
            else:
                kv_owner = self._kv_owner(layer_index)
                shape = (
                    batch_size,
                    kv_owner.head_span.size,
                    cache.sequence_length,
                    self.config.graph.text.full_attention.head_dim,
                )
                _require_tensor(layer.key, torch.bfloat16, shape, self.runtime.device)
                _require_tensor(layer.value, torch.bfloat16, shape, self.runtime.device)
                if layer.convolution is not None or layer.recurrent is not None:
                    raise ValueError("attention cache shard contains recurrent state")

    def _gdn_owner(self, layer_index: int) -> GDNStateOwnership:
        return next(
            owner
            for owner in self.parallel_plan.gdn_states
            if owner.layer_index == layer_index and owner.rank == self.rank
        )

    def _gdn_owners(self, layer_index: int) -> tuple[GDNStateOwnership, ...]:
        return tuple(
            sorted(
                (
                    owner
                    for owner in self.parallel_plan.gdn_states
                    if owner.layer_index == layer_index
                ),
                key=lambda owner: self.parallel_plan.topology.placements[owner.rank].tensor_rank,
            )
        )

    def _kv_owner(self, layer_index: int) -> KVHeadOwnership:
        return next(
            owner
            for owner in self.parallel_plan.kv_heads
            if owner.layer_index == layer_index and owner.rank == self.rank
        )

    def _kv_owners(self, layer_index: int) -> tuple[KVHeadOwnership, ...]:
        return tuple(
            sorted(
                (
                    owner
                    for owner in self.parallel_plan.kv_heads
                    if owner.layer_index == layer_index
                ),
                key=lambda owner: self.parallel_plan.topology.placements[owner.rank].tensor_rank,
            )
        )

    def _pipeline_edge_id(self, source_stage: int) -> str:
        placements = {item.rank: item for item in self.parallel_plan.topology.placements}
        matches = tuple(
            edge.edge_id
            for edge in self.parallel_plan.partition.pipeline_edges
            if placements[edge.source_rank].pipeline_stage == source_stage
            and placements[edge.source_rank].tensor_rank == self.tensor_rank
            and placements[edge.target_rank].pipeline_stage == source_stage + 1
            and placements[edge.target_rank].tensor_rank == self.tensor_rank
        )
        if len(matches) != 1:
            raise Qwen3_5DistributedReferenceError(
                "pipeline edge differs from the rank stage and tensor lane"
            )
        return matches[0]


class Qwen3_5DistributedMTPReferenceModel:
    """Exact TP oracle for the checkpoint's cyclic single MTP layer."""

    def __init__(
        self,
        *,
        config: Qwen3_5Config,
        weights: Qwen3_5RankWeights,
        parallel_plan: Qwen3_5ParallelWeightPlan,
        runtime: EagerNcclPartition,
    ) -> None:
        expected = seal_qwen3_5_target_mtp_distributed_reference(config, parallel_plan)
        if (
            runtime.sealed.topology_digest != expected.topology_digest
            or runtime.sealed.partition_digest != expected.partition_digest
        ):
            raise Qwen3_5DistributedReferenceError(
                "NCCL runtime was not initialized from the exact MTP oracle recipe"
            )
        if weights.rank != runtime.rank or weights.device != runtime.device:
            raise Qwen3_5DistributedReferenceError("MTP rank weights and NCCL placement differ")
        if parallel_plan.topology.pp_size != 1 or not parallel_plan.mtp_enabled:
            raise Qwen3_5DistributedReferenceError("MTP distributed reference is TP-only")
        self.config = config
        self.parallel_plan = parallel_plan
        self.runtime = runtime
        self.rank = runtime.rank
        self._weights = _CanonicalWeightWindow(
            weights,
            parallel_plan,
            runtime,
            weight_step_id=_mtp_weight_step_id,
        )
        self._canonical = Qwen3_5ReferenceModel(config=config, weights=self._weights)

    def empty_cache(self, batch_size: int = 1) -> Qwen3_5DistributedMTPCache:
        if batch_size <= 0:
            raise ValueError("MTP cache batch size must be positive")
        owner = self._owner()
        shape = (
            batch_size,
            owner.head_span.size,
            0,
            self.config.graph.text.full_attention.head_dim,
        )
        return Qwen3_5DistributedMTPCache(
            sequence_length=0,
            batch_size=batch_size,
            rank=self.rank,
            layer=Qwen3_5LayerCache(
                key=torch.empty(shape, dtype=torch.bfloat16, device=self.runtime.device),
                value=torch.empty(shape, dtype=torch.bfloat16, device=self.runtime.device),
            ),
        )

    def forward(
        self,
        input_ids: Tensor,
        target_hidden_states: Tensor,
        *,
        position_ids: Tensor,
        cache: Qwen3_5DistributedMTPCache | None = None,
        epoch_id: str,
        observer: Callable[[str, Tensor], None] | None = None,
    ) -> Qwen3_5DistributedMTPStep:
        require_reference_execution()
        if input_ids.ndim != 2:
            raise TypeError("MTP input_ids must be rank two")
        batch_size = input_ids.shape[0]
        active_cache = cache or self.empty_cache(batch_size)
        self._validate_cache(active_cache, batch_size)
        self._weights.open()
        try:
            with self.runtime.collective_epoch(epoch_id, lane_id=_MTP_RECIPE_LANE):
                full_cache = self._restore_cache(active_cache)
                step = self._canonical.mtp_forward(
                    input_ids,
                    target_hidden_states,
                    cache=full_cache,
                    position_ids=position_ids,
                    observer=observer,
                )
                local_cache = self._slice_cache(step.cache)
        finally:
            self._weights.close()
        return Qwen3_5DistributedMTPStep(
            logits=step.logits,
            hidden_states=step.hidden_states,
            cache=local_cache,
            max_reconstructed_operand_bytes=self._weights.peak_operand_bytes,
        )

    def _restore_cache(self, cache: Qwen3_5DistributedMTPCache) -> Qwen3_5MTPReferenceCache:
        local = cache.layer
        if local.key is None or local.value is None:
            raise Qwen3_5DistributedReferenceError("MTP KV shard is incomplete")
        owners = tuple(
            sorted(
                self.parallel_plan.mtp_kv_heads,
                key=lambda owner: self.parallel_plan.topology.placements[
                    owner.rank
                ].tensor_rank,
            )
        )
        key = _reconstruct_equal_shards(
            self.runtime.tp_all_gather(_mtp_state_step_id("key"), local.key, dim=1),
            axis=1,
            local_size=local.key.shape[1],
            global_size=sum(owner.head_span.size for owner in owners),
            spans_by_rank=tuple((owner.head_span,) for owner in owners),
        )
        value = _reconstruct_equal_shards(
            self.runtime.tp_all_gather(_mtp_state_step_id("value"), local.value, dim=1),
            axis=1,
            local_size=local.value.shape[1],
            global_size=sum(owner.head_span.size for owner in owners),
            spans_by_rank=tuple((owner.head_span,) for owner in owners),
        )
        layer = Qwen3_5LayerCache() if cache.sequence_length == 0 else Qwen3_5LayerCache(
            key=key,
            value=value,
        )
        return Qwen3_5MTPReferenceCache(
            sequence_length=cache.sequence_length,
            batch_size=cache.batch_size,
            layer=layer,
        )

    def _slice_cache(self, cache: Qwen3_5MTPReferenceCache) -> Qwen3_5DistributedMTPCache:
        owner = self._owner()
        if cache.layer.key is None or cache.layer.value is None:
            raise Qwen3_5DistributedReferenceError("canonical MTP KV state is incomplete")
        return Qwen3_5DistributedMTPCache(
            sequence_length=cache.sequence_length,
            batch_size=cache.batch_size,
            rank=self.rank,
            layer=Qwen3_5LayerCache(
                key=cache.layer.key.narrow(
                    1, owner.head_span.start, owner.head_span.size
                ).clone(memory_format=torch.contiguous_format),
                value=cache.layer.value.narrow(
                    1, owner.head_span.start, owner.head_span.size
                ).clone(memory_format=torch.contiguous_format),
            ),
        )

    def _validate_cache(self, cache: Qwen3_5DistributedMTPCache, batch_size: int) -> None:
        if not isinstance(cache, Qwen3_5DistributedMTPCache):
            raise TypeError("cache must be a Qwen3_5DistributedMTPCache")
        if cache.rank != self.rank or cache.batch_size != batch_size or cache.sequence_length < 0:
            raise ValueError("MTP cache belongs to another rank, batch, or sequence")
        owner = self._owner()
        shape = (
            batch_size,
            owner.head_span.size,
            cache.sequence_length,
            self.config.graph.text.full_attention.head_dim,
        )
        _require_tensor(cache.layer.key, torch.bfloat16, shape, self.runtime.device)
        _require_tensor(cache.layer.value, torch.bfloat16, shape, self.runtime.device)
        if cache.layer.convolution is not None or cache.layer.recurrent is not None:
            raise ValueError("MTP cache shard contains recurrent state")

    def _owner(self) -> KVHeadOwnership:
        return next(owner for owner in self.parallel_plan.mtp_kv_heads if owner.rank == self.rank)


def shard_qwen3_5_reference_cache(
    cache: Qwen3_5ReferenceCache,
    parallel_plan: Qwen3_5ParallelWeightPlan,
    *,
    rank: int,
) -> Qwen3_5DistributedCache:
    """Project one full oracle cache into the exact rank-local state ABI."""

    assignments = {item.node_id: item for item in parallel_plan.partition.assignments}
    layer_indices = tuple(
        layer_index
        for layer_index in range(len(cache.layers))
        if rank in assignments[f"text.decoder.layer.{layer_index:04d}"].owner_ranks
    )
    if not layer_indices:
        raise Qwen3_5DistributedReferenceError("rank owns no decoder layers")
    projected: list[Qwen3_5LayerCache] = []
    gdn_by_layer = {
        owner.layer_index: owner for owner in parallel_plan.gdn_states if owner.rank == rank
    }
    kv_by_layer = {
        owner.layer_index: owner for owner in parallel_plan.kv_heads if owner.rank == rank
    }
    for layer_index in layer_indices:
        full = cache.layers[layer_index]
        if layer_index in gdn_by_layer:
            gdn_owner = gdn_by_layer[layer_index]
            if full.convolution is None or full.recurrent is None:
                raise Qwen3_5DistributedReferenceError("oracle GDN state is incomplete")
            projected.append(
                Qwen3_5LayerCache(
                    convolution=torch.cat(
                        tuple(
                            full.convolution.narrow(1, span.start, span.size)
                            for span in gdn_owner.convolution_channel_spans
                        ),
                        dim=1,
                    ),
                    recurrent=full.recurrent.narrow(
                        1,
                        gdn_owner.value_head_span.start,
                        gdn_owner.value_head_span.size,
                    ).clone(memory_format=torch.contiguous_format),
                )
            )
        else:
            kv_owner = kv_by_layer[layer_index]
            if full.key is None or full.value is None:
                raise Qwen3_5DistributedReferenceError("oracle attention KV is incomplete")
            projected.append(
                Qwen3_5LayerCache(
                    key=full.key.narrow(1, kv_owner.head_span.start, kv_owner.head_span.size).clone(
                        memory_format=torch.contiguous_format
                    ),
                    value=full.value.narrow(
                        1, kv_owner.head_span.start, kv_owner.head_span.size
                    ).clone(memory_format=torch.contiguous_format),
                )
            )
    return Qwen3_5DistributedCache(
        sequence_length=cache.sequence_length,
        batch_size=cache.batch_size,
        rank=rank,
        layer_start=layer_indices[0],
        layer_end=layer_indices[-1] + 1,
        layers=tuple(projected),
    )


def _reconstruct_equal_shards(
    gathered: Tensor,
    *,
    axis: int,
    local_size: int,
    global_size: int,
    spans_by_rank: tuple[tuple[AxisSpan, ...], ...],
) -> Tensor:
    """Restore semantic span order from an equal-size rank-order gather."""

    try:
        return reconstruct_axis_spans(
            gathered,
            axis=axis,
            local_size=local_size,
            global_size=global_size,
            spans_by_rank=spans_by_rank,
        )
    except ValueError as error:
        raise Qwen3_5DistributedReferenceError(str(error)) from error


def _weight_step_id(name: str) -> str:
    return f"qwen3_5.reference.weight:{name}"


def _mtp_weight_step_id(name: str) -> str:
    return f"qwen3_5.mtp.reference.weight:{name}"


def _state_step_id(layer_index: int, state_name: str) -> str:
    return f"qwen3_5.reference.state:{layer_index}:{state_name}"


def _mtp_state_step_id(state_name: str) -> str:
    return f"qwen3_5.mtp.reference.state:0:{state_name}"


def _mtp_weight_sequence() -> tuple[str, ...]:
    prefix = "mtp.layers.0"
    attention = f"{prefix}.self_attn"
    mlp = f"{prefix}.mlp"
    return (
        "model.language_model.embed_tokens.weight",
        "mtp.fc.weight",
        f"{attention}.q_proj.weight",
        f"{attention}.k_proj.weight",
        f"{attention}.v_proj.weight",
        f"{attention}.o_proj.weight",
        f"{mlp}.gate_proj.weight",
        f"{mlp}.up_proj.weight",
        f"{mlp}.down_proj.weight",
        "lm_head.weight",
    )


def _state_names(
    layer_index: int,
    attention: AttentionKind,
) -> tuple[str, str]:
    del layer_index
    if attention is AttentionKind.GATED_DELTA_NET:
        return ("convolution", "recurrent")
    return ("key", "value")


def _layer_weight_sequence(
    layer_index: int,
    attention: AttentionKind,
) -> tuple[str, ...]:
    prefix = f"model.language_model.layers.{layer_index}"
    if attention is AttentionKind.GATED_DELTA_NET:
        op = f"{prefix}.linear_attn"
        attention_names: tuple[str, ...] = (
            f"{op}.in_proj_qkv.weight",
            f"{op}.conv1d.weight",
            f"{op}.in_proj_a.weight",
            f"{op}.in_proj_b.weight",
            f"{op}.A_log",
            f"{op}.dt_bias",
            f"{op}.in_proj_z.weight",
            f"{op}.out_proj.weight",
        )
    else:
        op = f"{prefix}.self_attn"
        attention_names = (
            f"{op}.q_proj.weight",
            f"{op}.k_proj.weight",
            f"{op}.v_proj.weight",
            f"{op}.o_proj.weight",
        )
    mlp = f"{prefix}.mlp"
    return (
        *attention_names,
        f"{mlp}.gate_proj.weight",
        f"{mlp}.up_proj.weight",
        f"{mlp}.down_proj.weight",
    )


def _require_tensor(
    tensor: Tensor | None,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if tensor is None:
        raise ValueError("distributed state tensor is missing")
    if (
        tensor.dtype is not dtype
        or tuple(tensor.shape) != shape
        or tensor.device != device
        or tensor.layout is not torch.strided
        or not tensor.is_contiguous()
    ):
        raise ValueError("distributed state tensor violates its rank-local ABI")


__all__ = (
    "Qwen3_5DistributedCache",
    "Qwen3_5DistributedMTPCache",
    "Qwen3_5DistributedMTPReferenceModel",
    "Qwen3_5DistributedMTPStep",
    "Qwen3_5DistributedReferenceError",
    "Qwen3_5DistributedReferenceModel",
    "Qwen3_5DistributedStep",
    "seal_qwen3_5_distributed_reference",
    "seal_qwen3_5_target_mtp_distributed_reference",
    "shard_qwen3_5_reference_cache",
)
