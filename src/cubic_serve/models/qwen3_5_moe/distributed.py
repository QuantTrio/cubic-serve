"""Independent eager MoE execution with explicit TP or EP expert ownership.

Experts remain rank-local tensor slices. EP dispatches rows to expert owners;
EP1 evaluates the same routes on all intermediate shards and reduces partials.
Canonical non-expert operands are reconstructed one at a time. This reference
is never a performance candidate or fallback.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import cast

import torch
from torch import Tensor

from cubic_serve.contracts.execution import CollectiveStep
from cubic_serve.executor.distributed_reference import (
    _MTP_RECIPE_LANE,
    Qwen3_5DistributedMTPCache,
    Qwen3_5DistributedMTPReferenceModel,
    Qwen3_5DistributedReferenceModel,
    _CanonicalWeightWindow,
    _mtp_state_step_id,
)
from cubic_serve.executor.parallel import EagerNcclPartition, SealedPartition, seal_partition
from cubic_serve.models.parallel_weights import reconstruct_axis_spans
from cubic_serve.models.qwen3_5.model import (
    AttentionKind,
    Qwen3_5Config,
    Qwen3_5LayerCache,
    Qwen3_5MTPReferenceCache,
    Qwen3_5OperatorCalls,
)
from cubic_serve.models.qwen3_5.weights import Qwen3_5ParallelWeightPlan
from cubic_serve.models.qwen3_5_moe.config import Qwen3_5MoeConfig
from cubic_serve.models.qwen3_5_moe.model import (
    Qwen3_5MoeStageModel,
    execute_local_packed_experts,
    route_qwen3_5_moe,
)
from cubic_serve.models.qwen3_5_moe.parallel import (
    MoeWeightLayout,
    Qwen3_5MoeParallelPlan,
    Qwen3_5MoeRankWeights,
)
from cubic_serve.ops.eager.activation import sigmoid_stored as activation_sigmoid
from cubic_serve.ops.eager.activation import silu_stored as activation_silu

_TP_LANE = "qwen3_5_moe.canonical-reconstruct-v1"
_EP_LANE = "qwen3_5_moe.expert-route-v1"
_ROW_LANE = "qwen3_5_moe.canonical-row-replicate-v1"
_MTP_STEP = "mtp:0"


class Qwen3_5MoeDistributedReferenceError(RuntimeError):
    """The sealed MoE reference ownership or collective recipe is open."""


def seal_qwen3_5_moe_distributed_reference(
    config: Qwen3_5MoeConfig,
    parallel_plan: Qwen3_5MoeParallelPlan,
) -> SealedPartition:
    """Seal every gather, dispatch, combine, and row-replication in call order."""

    if config.identity.model_type != "qwen3_5_moe":
        raise Qwen3_5MoeDistributedReferenceError("MoE reference requires qwen3_5_moe")
    topology = parallel_plan.topology
    sequence: dict[tuple[str, str], int] = {}
    collectives: list[CollectiveStep] = []

    def add(group: str, lane: str, step: str, operation: str) -> None:
        key = (group, lane)
        collectives.append(CollectiveStep(step, sequence.get(key, 0), group, operation, lane))
        sequence[key] = sequence.get(key, 0) + 1

    add("tp:0", _TP_LANE, _weight_step("model.language_model.embed_tokens.weight"), "all_gather")
    layers_per_stage = config.graph.text.layer_count // topology.pp_size
    for layer in config.architecture_plan().decoder_layers:
        stage = layer.layer_index // layers_per_stage
        tp_group, ep_group = f"tp:{stage}", f"ep:{stage}"
        for state in (
            ("convolution", "recurrent")
            if layer.attention is AttentionKind.GATED_DELTA_NET
            else ("key", "value")
        ):
            add(tp_group, _TP_LANE, _state_step(layer.layer_index, state), "all_gather")
        for name in _layer_gather_sequence(layer.layer_index, layer.attention):
            add(tp_group, _TP_LANE, _weight_step(name), "all_gather")
        if topology.expert_tensor_size == 1 and topology.ep_size > 1:
            add(ep_group, _EP_LANE, _dispatch_step(layer.layer_index), "all_to_all")
            add(ep_group, _EP_LANE, _combine_step(layer.layer_index), "all_to_all")
        add(tp_group, _ROW_LANE, _row_step(layer.layer_index), "all_reduce")
    add(
        f"tp:{topology.pp_size - 1}",
        _TP_LANE,
        _weight_step("lm_head.weight"),
        "all_gather",
    )
    partition = replace(
        parallel_plan.partition,
        partition_id=parallel_plan.partition.partition_id + ":eager-reference-ep-v1",
        collectives=tuple(collectives),
    )
    return seal_partition(topology, partition)


class _MoeWeightWindow:
    """Expose expert-local tensors and one ephemeral canonical non-expert operand."""

    def __init__(
        self,
        weights: Qwen3_5MoeRankWeights,
        plan: Qwen3_5MoeParallelPlan,
        runtime: EagerNcclPartition,
        *,
        weight_step: Callable[[str], str] | None = None,
    ) -> None:
        self.device = weights.device
        self.inventory_digest = weights.inventory_digest
        self.semantic_plan_digest = plan.semantic_plan_digest
        self._rank = weights.rank
        self._local = dict(weights.named_tensors())
        self._bindings = {item.tensor_name: item for item in weights.bindings}
        self._all_bindings = plan.bindings
        self._runtime = runtime
        self._weight_step = weight_step or _weight_step
        self._active = False
        self.peak_operand_bytes = 0

    @property
    def tensor_count(self) -> int:
        return len(self._local)

    def named_tensors(self) -> tuple[tuple[str, Tensor], ...]:
        return tuple((name, self._local[name]) for name in sorted(self._local))

    def open(self) -> None:
        if self._active:
            raise Qwen3_5MoeDistributedReferenceError("weight window was re-entered")
        self._active, self.peak_operand_bytes = True, 0

    def close(self) -> None:
        self._active = False

    def __getitem__(self, name: str) -> Tensor:
        if not self._active:
            raise Qwen3_5MoeDistributedReferenceError("canonical operands require an active epoch")
        try:
            local, binding = self._local[name], self._bindings[name]
        except KeyError as error:
            raise Qwen3_5MoeDistributedReferenceError(
                f"rank {self._rank} does not own {name!r}"
            ) from error
        if binding.layout in (MoeWeightLayout.REPLICATED, MoeWeightLayout.EXPERT):
            return local
        peers = tuple(
            sorted(
                (item for item in self._all_bindings if item.tensor_name == name),
                key=lambda item: item.tensor_rank,
            )
        )
        gathered = self._runtime.tp_all_gather(
            self._weight_step(name), local, dim=binding.shard_axis
        )
        if binding.layout is MoeWeightLayout.KV_REPLICATED:
            full = _deduplicate_kv_replicas(
                gathered,
                axis=binding.shard_axis,
                local_size=binding.local_shape[binding.shard_axis],
                global_size=binding.global_shape[binding.shard_axis],
            )
        else:
            try:
                full = reconstruct_axis_spans(
                    gathered,
                    axis=binding.shard_axis,
                    local_size=binding.local_shape[binding.shard_axis],
                    global_size=binding.global_shape[binding.shard_axis],
                    spans_by_rank=tuple(item.spans for item in peers),
                )
            except ValueError as error:
                raise Qwen3_5MoeDistributedReferenceError(str(error)) from error
        if tuple(full.shape) != binding.global_shape or full.dtype is not binding.dtype:
            raise Qwen3_5MoeDistributedReferenceError(f"canonical operand differs for {name!r}")
        self.peak_operand_bytes = max(self.peak_operand_bytes, full.numel() * full.element_size())
        return full


class _DistributedMoe:
    def __init__(
        self,
        config: Qwen3_5MoeConfig,
        weights: _MoeWeightWindow,
        plan: Qwen3_5MoeParallelPlan,
        runtime: EagerNcclPartition,
        *,
        step: int | str | None = None,
        packed_experts: tuple[Tensor, Tensor] | None = None,
    ) -> None:
        placement = next(item for item in plan.topology.placements if item.rank == runtime.rank)
        self.config, self.weights, self.runtime = config, weights, runtime
        self.tensor_rank = placement.tensor_rank
        self.ep_size = plan.topology.ep_size
        self.etp_size = plan.topology.expert_tensor_size
        self.expert_owner = plan.topology.expert_coordinates(runtime.rank)[0]
        self.stage_base = placement.pipeline_stage * plan.topology.tp_size
        self.step = step
        self.packed_experts = packed_experts
        self.observer: Callable[[int, str, Tensor], None] | None = None

    def __call__(self, hidden: Tensor, *, prefix: str, operators: Qwen3_5OperatorCalls) -> Tensor:
        moe = self.config.graph.text.moe
        if moe is None or not moe.normalize_topk_probability:
            raise Qwen3_5MoeDistributedReferenceError("the routed MoE contract is incomplete")
        flat = hidden.reshape(-1, hidden.shape[-1])
        gate_weight = self.weights[f"{prefix}.gate.weight"]
        routes = route_qwen3_5_moe(flat, gate_weight, top_k=moe.experts_per_token)
        layer = self.step if self.step is not None else int(prefix.split(".")[3])
        if self.observer is not None:
            self.observer(layer, "router_logits", operators.linear(flat, gate_weight))
            self.observer(layer, "route", routes.expert_ids)
        gate_up, down = (
            self.packed_experts
            if self.packed_experts is not None
            else (
                self.weights[f"{prefix}.experts.gate_up_proj"],
                self.weights[f"{prefix}.experts.down_proj"],
            )
        )
        rows = torch.arange(flat.shape[0], dtype=torch.int64, device=flat.device)
        if self.ep_size == 1:
            source_rows = rows
            source_hidden = flat
            returned = execute_local_packed_experts(
                flat.repeat_interleave(moe.experts_per_token, dim=0).contiguous(),
                routes.expert_ids.reshape(-1).contiguous(),
                gate_up,
                down,
                first_expert_id=0,
            )
        elif self.etp_size > 1:
            source_rows = rows
            source_hidden = flat
            expanded = flat.repeat_interleave(moe.experts_per_token, dim=0).contiguous()
            ids = routes.expert_ids.reshape(-1)
            first = self.expert_owner * (moe.expert_count // self.ep_size)
            owned = (ids >= first) & (ids < first + gate_up.shape[0])
            returned = torch.zeros_like(expanded)
            if bool(owned.any().item()):
                returned[owned] = execute_local_packed_experts(
                    expanded[owned].contiguous(), ids[owned].contiguous(), gate_up, down,
                    first_expert_id=first,
                )
        else:
            source_rows = rows[rows.remainder(self.ep_size) == self.tensor_rank]
            source_hidden = flat.index_select(0, source_rows)
            source_ids = routes.expert_ids.index_select(0, source_rows)
            expanded = source_hidden.repeat_interleave(moe.experts_per_token, dim=0).contiguous()
            destinations = (
                self.stage_base
                + source_ids.reshape(-1).div(
                    moe.expert_count // self.ep_size, rounding_mode="floor"
                )
            ).contiguous()
            dispatched = self.runtime.ep_dispatch(_dispatch_step(layer), expanded, destinations)
            received_ids = self._received_ids(routes.expert_ids, dispatched.receive_splits)
            first = self.tensor_rank * (moe.expert_count // self.ep_size)
            expert = execute_local_packed_experts(
                dispatched.values,
                received_ids,
                gate_up,
                down,
                first_expert_id=first,
            )
            returned = self.runtime.ep_combine(_combine_step(layer), dispatched, expert)
        weighted = returned.view(source_rows.numel(), moe.experts_per_token, flat.shape[1])
        weighted = (weighted * routes.weights.index_select(0, source_rows).unsqueeze(-1)).to(
            torch.bfloat16
        )
        source_output = torch.zeros_like(source_hidden)
        for route_index in range(moe.experts_per_token):
            source_output = (source_output + weighted[:, route_index]).to(torch.bfloat16)
        routed = torch.zeros_like(flat)
        routed.index_copy_(0, source_rows, source_output)
        routed = self.runtime.tp_all_reduce(_row_step(layer), routed.contiguous())

        gate = activation_silu(
            operators.linear(flat, self.weights[f"{prefix}.shared_expert.gate_proj.weight"])
        )
        up = operators.linear(flat, self.weights[f"{prefix}.shared_expert.up_proj.weight"])
        shared = operators.linear(
            gate * up, self.weights[f"{prefix}.shared_expert.down_proj.weight"]
        )
        shared_gate = activation_sigmoid(
            operators.linear(flat, self.weights[f"{prefix}.shared_expert_gate.weight"])
        )
        return (
            (routed + (shared * shared_gate).to(torch.bfloat16))
            .to(torch.bfloat16)
            .reshape_as(hidden)
        )

    def _received_ids(self, global_ids: Tensor, receive_splits: tuple[int, ...]) -> Tensor:
        owned: list[Tensor] = []
        moe = self.config.graph.text.moe
        if moe is None:
            raise Qwen3_5MoeDistributedReferenceError("the routed MoE contract is missing")
        width = moe.expert_count // self.ep_size
        for source_lane, expected in enumerate(receive_splits):
            source = global_ids[source_lane :: self.ep_size].reshape(-1)
            selected = source[
                (source >= self.tensor_rank * width) & (source < (self.tensor_rank + 1) * width)
            ]
            if selected.numel() != expected:
                raise Qwen3_5MoeDistributedReferenceError(
                    "dispatch metadata differs from global routes"
                )
            owned.append(selected)
        return torch.cat(owned).contiguous()


class Qwen3_5MoeDistributedReferenceModel(Qwen3_5DistributedReferenceModel):
    """Stage-local full decoder reusing the Qwen hybrid-state execution framework."""

    def __init__(
        self,
        *,
        config: Qwen3_5MoeConfig,
        weights: Qwen3_5MoeRankWeights,
        parallel_plan: Qwen3_5MoeParallelPlan,
        runtime: EagerNcclPartition,
        recurrent_state_dtype: torch.dtype | None = None,
    ) -> None:
        expected = seal_qwen3_5_moe_distributed_reference(config, parallel_plan)
        if (runtime.sealed.topology_digest, runtime.sealed.partition_digest) != (
            expected.topology_digest,
            expected.partition_digest,
        ):
            raise Qwen3_5MoeDistributedReferenceError("runtime does not use the exact MoE recipe")
        if weights.rank != runtime.rank or weights.device != runtime.device:
            raise Qwen3_5MoeDistributedReferenceError("rank weights differ from NCCL placement")
        placement = next(
            item for item in parallel_plan.topology.placements if item.rank == runtime.rank
        )
        assignments = {item.node_id: item for item in parallel_plan.partition.assignments}
        owned = tuple(
            layer.layer_index
            for layer in config.architecture_plan().decoder_layers
            if runtime.rank in assignments[layer.node_id].owner_ranks
        )
        if not owned or owned != tuple(range(owned[0], owned[-1] + 1)):
            raise Qwen3_5MoeDistributedReferenceError("pipeline stage layers are not contiguous")
        window = _MoeWeightWindow(weights, parallel_plan, runtime)
        self.config = cast(Qwen3_5Config, config)
        self.parallel_plan = cast(Qwen3_5ParallelWeightPlan, parallel_plan)
        self.runtime, self.rank = runtime, runtime.rank
        self.pipeline_stage, self.tensor_rank = placement.pipeline_stage, placement.tensor_rank
        self.is_first_stage = self.pipeline_stage == 0
        self.is_last_stage = self.pipeline_stage == parallel_plan.topology.pp_size - 1
        self.layer_start, self.layer_end = owned[0], owned[-1] + 1
        self._weights = cast(_CanonicalWeightWindow, window)
        self._moe = _DistributedMoe(config, window, parallel_plan, runtime)
        self._canonical = Qwen3_5MoeStageModel(
            config=config,
            weights=window,
            moe=self._moe,
            recurrent_state_dtype=recurrent_state_dtype,
        )

    def _restore_layer_cache(self, layer_index: int, local: Qwen3_5LayerCache) -> Qwen3_5LayerCache:
        if self.config.graph.text.layer_schedule[layer_index] is AttentionKind.GATED_DELTA_NET:
            return super()._restore_layer_cache(layer_index, local)
        if local.key is None or local.value is None:
            raise Qwen3_5MoeDistributedReferenceError("attention KV shard is incomplete")
        values = tuple(
            _deduplicate_kv_replicas(
                self.runtime.tp_all_gather(_state_step(layer_index, name), tensor, dim=1),
                axis=1,
                local_size=tensor.shape[1],
                global_size=self.config.graph.text.full_attention.key_value_heads,
            )
            for name, tensor in (("key", local.key), ("value", local.value))
        )
        return (
            Qwen3_5LayerCache()
            if values[0].shape[2] == 0
            else Qwen3_5LayerCache(key=values[0], value=values[1])
        )

    @staticmethod
    def _cache_collective_step(layer_index: int, state_name: str) -> str:
        return _state_step(layer_index, state_name)

    def _pipeline_edge_id(self, source_stage: int) -> str:
        if source_stage not in range(self.parallel_plan.topology.pp_size - 1):
            raise Qwen3_5MoeDistributedReferenceError("pipeline source stage is invalid")
        return f"qwen3_5_moe.pp:lane:{self.tensor_rank}"


def seal_qwen3_5_moe_mtp_distributed_reference(
    config: Qwen3_5MoeConfig,
    parallel_plan: Qwen3_5MoeParallelPlan,
) -> SealedPartition:
    """Seal the independent eager MTP oracle under the target expert topology."""

    topology = parallel_plan.topology
    if (
        config.identity.model_type != "qwen3_5_moe"
        or not parallel_plan.mtp_enabled
        or topology.pp_size != 1
    ):
        raise Qwen3_5MoeDistributedReferenceError(
            "MoE MTP reference requires TP x PP1 and MTP weights"
        )
    sequence: dict[str, int] = {}
    collectives: list[CollectiveStep] = []

    def add(group: str, step: str, operation: str) -> None:
        collectives.append(
            CollectiveStep(step, sequence.get(group, 0), group, operation, _MTP_RECIPE_LANE)
        )
        sequence[group] = sequence.get(group, 0) + 1

    for state in ("key", "value"):
        add("tp:0", _mtp_state_step_id(state), "all_gather")
    for name in (
        "model.language_model.embed_tokens.weight",
        "mtp.fc.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
    ):
        add("tp:0", _mtp_weight_step(name), "all_gather")
    if topology.expert_tensor_size == 1 and topology.ep_size > 1:
        add("ep:0", _dispatch_step(_MTP_STEP), "all_to_all")
        add("ep:0", _combine_step(_MTP_STEP), "all_to_all")
    add("tp:0", _row_step(_MTP_STEP), "all_reduce")
    for name in (
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
        "mtp.layers.0.mlp.shared_expert.up_proj.weight",
        "mtp.layers.0.mlp.shared_expert.down_proj.weight",
        "lm_head.weight",
    ):
        add("tp:0", _mtp_weight_step(name), "all_gather")
    partition = replace(
        parallel_plan.partition,
        partition_id=parallel_plan.partition.partition_id + ":mtp-eager-reference-ep-v1",
        collectives=tuple(collectives),
    )
    return seal_partition(topology, partition)


class Qwen3_5MoeDistributedMTPReferenceModel(Qwen3_5DistributedMTPReferenceModel):
    """Independent eager oracle for one Qwen3.5 MoE MTP proposal step."""

    def __init__(
        self,
        *,
        config: Qwen3_5MoeConfig,
        weights: Qwen3_5MoeRankWeights,
        parallel_plan: Qwen3_5MoeParallelPlan,
        runtime: EagerNcclPartition,
        packed_experts: tuple[Tensor, Tensor] | None = None,
    ) -> None:
        expected = seal_qwen3_5_moe_mtp_distributed_reference(config, parallel_plan)
        if (runtime.sealed.topology_digest, runtime.sealed.partition_digest) != (
            expected.topology_digest,
            expected.partition_digest,
        ):
            raise Qwen3_5MoeDistributedReferenceError(
                "runtime does not use the exact MoE MTP eager recipe"
            )
        if weights.rank != runtime.rank or weights.device != runtime.device:
            raise Qwen3_5MoeDistributedReferenceError("MTP weights differ from NCCL placement")
        moe = config.graph.text.moe
        if moe is None:
            raise Qwen3_5MoeDistributedReferenceError("MTP MoE contract is absent")
        first = parallel_plan.topology.expert_coordinates(runtime.rank)[0] * (
            moe.expert_count // parallel_plan.topology.ep_size
        )
        local_experts = moe.expert_count // parallel_plan.topology.ep_size
        by_name = dict(weights.named_tensors())
        prefix = "mtp.layers.0.mlp.experts"
        if packed_experts is None:
            gate_up = torch.stack(
                tuple(
                    torch.cat(
                        (
                            by_name[f"{prefix}.{expert}.gate_proj.weight"],
                            by_name[f"{prefix}.{expert}.up_proj.weight"],
                        ),
                        dim=0,
                    )
                    for expert in range(first, first + local_experts)
                )
            ).contiguous()
            down = torch.stack(
                tuple(
                    by_name[f"{prefix}.{expert}.down_proj.weight"]
                    for expert in range(first, first + local_experts)
                )
            ).contiguous()
        else:
            gate_up, down = packed_experts
        window = _MoeWeightWindow(
            weights,
            parallel_plan,
            runtime,
            weight_step=_mtp_weight_step,
        )
        self.config = cast(Qwen3_5Config, config)
        self.parallel_plan = cast(Qwen3_5ParallelWeightPlan, parallel_plan)
        self.runtime, self.rank = runtime, runtime.rank
        self._weights = cast(_CanonicalWeightWindow, window)
        self._canonical = Qwen3_5MoeStageModel(
            config=config,
            weights=window,
            moe=_DistributedMoe(
                config,
                window,
                parallel_plan,
                runtime,
                step=_MTP_STEP,
                packed_experts=(gate_up, down),
            ),
        )

    def _restore_cache(self, cache: Qwen3_5DistributedMTPCache) -> Qwen3_5MTPReferenceCache:
        local = cache.layer
        if local.key is None or local.value is None:
            raise Qwen3_5MoeDistributedReferenceError("MTP KV shard is incomplete")
        key, value = (
            _deduplicate_kv_replicas(
                self.runtime.tp_all_gather(_mtp_state_step_id(name), tensor, dim=1),
                axis=1,
                local_size=tensor.shape[1],
                global_size=self.config.graph.text.full_attention.key_value_heads,
            )
            for name, tensor in (("key", local.key), ("value", local.value))
        )
        layer = (
            Qwen3_5LayerCache()
            if cache.sequence_length == 0
            else Qwen3_5LayerCache(key=key, value=value)
        )
        return Qwen3_5MTPReferenceCache(
            sequence_length=cache.sequence_length,
            batch_size=cache.batch_size,
            layer=layer,
        )


def _deduplicate_kv_replicas(
    gathered: Tensor,
    *,
    axis: int,
    local_size: int,
    global_size: int,
) -> Tensor:
    leading = gathered.movedim(axis, 0)
    if (
        global_size <= 0
        or local_size <= 0
        or global_size % local_size
        or leading.shape[0] % global_size
    ):
        raise Qwen3_5MoeDistributedReferenceError("KV gather does not close configured head groups")
    copies = leading.shape[0] // global_size
    if copies < 1:
        raise Qwen3_5MoeDistributedReferenceError("KV gather is incomplete")
    unique = []
    for group in range(global_size // local_size):
        first = leading.narrow(0, group * copies * local_size, local_size)
        for replica in range(1, copies):
            other = leading.narrow(0, (group * copies + replica) * local_size, local_size)
            if not torch.equal(first, other):
                raise Qwen3_5MoeDistributedReferenceError("KV replicas differ")
        unique.append(first)
    return torch.cat(unique, dim=0).movedim(0, axis).contiguous()


def _weight_step(name: str) -> str:
    return f"qwen3_5_moe.reference.weight:{name}"


def _mtp_weight_step(name: str) -> str:
    return f"qwen3_5_moe.mtp.reference.weight:{name}"


def _state_step(layer: int, name: str) -> str:
    return f"qwen3_5_moe.reference.state:{layer}:{name}"


def _dispatch_step(layer: int | str) -> str:
    return f"qwen3_5_moe.reference.ep.dispatch:{layer}"


def _combine_step(layer: int | str) -> str:
    return f"qwen3_5_moe.reference.ep.combine:{layer}"


def _row_step(layer: int | str) -> str:
    return f"qwen3_5_moe.reference.row-replicate:{layer}"


def _layer_gather_sequence(layer: int, attention: AttentionKind) -> tuple[str, ...]:
    prefix = f"model.language_model.layers.{layer}"
    op = (
        f"{prefix}.linear_attn"
        if attention is AttentionKind.GATED_DELTA_NET
        else f"{prefix}.self_attn"
    )
    attention_names = (
        (
            f"{op}.in_proj_qkv.weight",
            f"{op}.conv1d.weight",
            f"{op}.in_proj_a.weight",
            f"{op}.in_proj_b.weight",
            f"{op}.A_log",
            f"{op}.dt_bias",
            f"{op}.in_proj_z.weight",
            f"{op}.out_proj.weight",
        )
        if attention is AttentionKind.GATED_DELTA_NET
        else (
            f"{op}.q_proj.weight",
            f"{op}.k_proj.weight",
            f"{op}.v_proj.weight",
            f"{op}.o_proj.weight",
        )
    )
    mlp = f"{prefix}.mlp"
    return (
        *attention_names,
        f"{mlp}.shared_expert.gate_proj.weight",
        f"{mlp}.shared_expert.up_proj.weight",
        f"{mlp}.shared_expert.down_proj.weight",
    )


__all__ = (
    "Qwen3_5MoeDistributedMTPReferenceModel",
    "Qwen3_5MoeDistributedReferenceError",
    "Qwen3_5MoeDistributedReferenceModel",
    "seal_qwen3_5_moe_distributed_reference",
    "seal_qwen3_5_moe_mtp_distributed_reference",
)
