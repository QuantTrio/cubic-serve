"""Packed Qwen rank stages sharing the canonical decode body and ownership plan.

This is a pre-readiness execution building block, not a performance admission
shortcut. Graph portfolios, topology-wide leases and numerical seals belong to
the controller. Original checkpoint experts never change physical owner.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import Any, cast

import torch
import torch.distributed as dist
import triton  # type: ignore[import-untyped]
from torch import Tensor
from torch.distributed import ProcessGroup

from cubic_serve.ops.cuda.collective import all_reduce, barrier
from cubic_serve.contracts.execution import GroupKind, ParallelTopology, RankReplayResources
from cubic_serve.executor.decode_overlap import (
    DecodeInterleave,
    DecodeOverlapPlan,
    common_overlap_thread_candidates,
    compare_decode_graphs,
    decode_calibration_prepare,
    format_decode_measurements,
    measure_decode_graph,
)
from cubic_serve.executor.model_runner import (
    _Buffers,
    _copy_decode_input_rows,
    _DecodeBody,
    _local_config,
    _MRoPETables,
)
from cubic_serve.models.builtin import builtin_model_definition
from cubic_serve.models.parallel_weights import AxisSpan
from cubic_serve.models.quantization import ActivationFormat
from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.models.qwen3_5.model import (
    CONV_STATE_ID,
    RECURRENT_STATE_ID,
    AttentionKind,
    Qwen3_5Config,
)
from cubic_serve.ops.cuda.collective import FixedRankBF16SumPrimitive
from cubic_serve.ops.cuda.compiled_artifact import CompiledArtifactIdentity, LaunchReceipt
from cubic_serve.ops.cuda.linear.quantized import (
    PackedLinearOperators,
    PackedOperatorBinding,
    PackedProjectionDispatch,
    _fp8_quant,
    _PackedLaunch,
)
from cubic_serve.ops.cuda.moe.route_completion import RouteCompletionCosts, RouteCompletionPlanner
from cubic_serve.ops.cuda.position import build_bf16_mrope_tables
from cubic_serve.ops.strategies import (
    CollectiveParameters,
    OperatorConfig,
    collective_parameters,
    collective_thread_config,
    operator_strategy_scope,
    overlap_collective_threads,
)
from cubic_serve.runtime.cache import CacheArena, CacheArenaSpec, KVLayout, StateSlotSpec
from cubic_serve.runtime.cache.quantization import KVCacheStorageSpec
from cubic_serve.runtime.quantized import QuantizedWeights


class _ScaleCollective(PackedOperatorBinding):
    """Static local scale producer -> NCCL MAX -> quantization/GEMM."""

    def __init__(
        self,
        producer: PackedOperatorBinding,
        consumer: PackedOperatorBinding,
        scales: Tensor,
        group: ProcessGroup,
    ) -> None:
        super().__init__(consumer.output, (), scratch=(*consumer.scratch, scales))
        self.producer, self.consumer, self.scales, self.group = producer, consumer, scales, group

    def validate_operands(self) -> None:
        self.producer.validate_operands()
        self.consumer.validate_operands()

    def launch(self) -> tuple[LaunchReceipt, ...]:
        self.validate_operands()
        before = self.producer.launch()
        all_reduce(self.scales, op=dist.ReduceOp.MAX, group=self.group)
        return (*before, *self.consumer.launch())

    def project_artifacts(self) -> tuple[CompiledArtifactIdentity, ...]:
        return (*self.producer.project_artifacts(), *self.consumer.project_artifacts())


class _RankOperators(PackedLinearOperators):
    def __init__(
        self,
        weights: QuantizedWeights,
        group: ProcessGroup | None,
        config: OperatorConfig | None,
        expert_group: ProcessGroup | None = None,
        *,
        device: str = "",
        model: str = "",
    ) -> None:
        super().__init__(
            config=config,
            device=device,
            model=model,
            prepared_weights=weights.prepared_weights,
            materialized_weights=weights.materialized_weights,
        )
        self.group, self.expert_group = group, expert_group
        self.row_weights = {
            id(weights[b.tensor_name])
            for b in weights.bindings
            if b.shard_axis == 1 and b.local_shape[1] < b.global_shape[1]
        }

    def bind_linear(
        self,
        x: Tensor,
        weight: Tensor | QuantizedMatrix,
        output: Tensor,
        *,
        fp8_token_scales: Tensor | None = None,
    ) -> PackedOperatorBinding:
        if (
            self.group is not None
            and id(weight) in self.row_weights
            and isinstance(weight, QuantizedMatrix)
            and weight.encoding.activation is ActivationFormat.FP8_TOKEN
        ):
            if fp8_token_scales is not None:
                raise ValueError("rank-local FP8 scale ownership cannot be overridden")
            scales = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
            producer = self.bind_fp8_token_scales(x, scales)
            consumer = super().bind_linear(x, weight, output, fp8_token_scales=scales)
            return _ScaleCollective(producer, consumer, scales, self.group)
        return super().bind_linear(x, weight, output, fp8_token_scales=fp8_token_scales)

    def bind_linear_group(
        self, x: Tensor, weights: tuple[Tensor | QuantizedMatrix, ...],
        outputs: tuple[Tensor, ...],
    ) -> PackedOperatorBinding | None:
        # A packed group must not bypass its independent full-K scale producer.
        # Row-parallel members keep their existing individually qualified path.
        if self.group is not None and any(
            isinstance(w, QuantizedMatrix) and id(w) in self.row_weights for w in weights
        ):
            return None
        return super().bind_linear_group(x, weights, outputs)

    def bind_routed_fp8_input(
        self,
        x: Tensor,
        *,
        row_parallel: bool = False,
    ) -> tuple[Tensor, Tensor, PackedOperatorBinding]:
        if not row_parallel or self.expert_group is None:
            return super().bind_routed_fp8_input(x)
        codes = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        scales = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
        producer = self.bind_fp8_token_scales(x, scales)
        consumer = PackedOperatorBinding(
            codes,
            (
                _PackedLaunch.compile(
                    _fp8_quant,
                    (x, codes.view(torch.uint8), scales),
                    (x.shape[1], triton.next_power_of_2(x.shape[1]), True),
                    (x.shape[0],),
                    {"enable_fp_fusion": False},
                ),
            ),
            scratch=(codes,),
        )
        return codes, scales, _ScaleCollective(producer, consumer, scales, self.expert_group)


class _StageWeights:
    """Translate local layer numbers only; never synthesize weights or change encoding."""

    def __init__(self, source: QuantizedWeights, layer_start: int) -> None:
        self.source, self.layer_start = source, layer_start
        self.device = source.device
        self.inventory_digest = source.inventory_digest
        self.semantic_plan_digest = source.semantic_plan_digest

    @property
    def tensor_count(self) -> int:
        return self.source.tensor_count

    def named_tensors(self) -> tuple[tuple[str, Tensor], ...]:
        # The reference weight protocol exposes physical payloads for residency
        # inspection; executable lookup above retains logical matrix objects.
        return self.source.named_tensors()

    def __getitem__(self, name: str) -> Any:
        prefix = "model.language_model.layers."
        if name.startswith(prefix):
            layer, tail = name[len(prefix) :].split(".", 1)
            name = f"{prefix}{int(layer) + self.layer_start}.{tail}"
        return self.source[name]


class QuantizedDecodeStage:
    """One rank's fixed-M stage, including TP reductions and native PP transfers."""

    def __init__(
        self,
        config: Qwen3_5Config,
        weights: QuantizedWeights,
        topology: ParallelTopology,
        *,
        tp_group: ProcessGroup | None,
        pp_group: ProcessGroup | None,
        ep_group: ProcessGroup | None = None,
        expert_tp_group: ProcessGroup | None = None,
        captured_m: int,
        table_pages: int = 1,
        page_size: int = 4,
        operator_config: OperatorConfig | None = None,
        arena: CacheArena | None = None,
        mrope_tables: _MRoPETables | None = None,
        state_slots: int | None = None,
        resource: RankReplayResources | None = None,
        route_completion_costs: RouteCompletionCosts | None = None,
        model_factory: Callable[..., Any] | None = None,
        cache_storage: KVCacheStorageSpec | None = None,
        shared_scratch: dict[tuple, Tensor] | None = None,
    ) -> None:
        private_start = torch.cuda.memory_allocated(weights.device)
        rank = weights.rank
        if resource is not None and resource.rank != rank:
            raise ValueError("packed stage resource belongs to a different rank")
        scratch_id = "lane" if resource is None else resource.scratch_arena_id
        placement = next(p for p in topology.placements if p.rank == rank)
        stage, tensor_rank = placement.pipeline_stage, placement.tensor_rank
        tp, pp = topology.tp_size, topology.pp_size
        if (
            not dist.is_initialized()
            or dist.get_rank() != rank
            or weights.device.index != placement.device_index
            or config.graph.text.dtype not in ("bfloat16", "float16")
            or captured_m < 1
            or captured_m & (captured_m - 1)
        ):
            raise ValueError("packed stage differs from the rank/FP16-BF16/power-of-two contract")
        stage_ranks = tuple(p.rank for p in topology.placements if p.pipeline_stage == stage)
        lane_ranks = tuple(p.rank for p in topology.placements if p.tensor_rank == tensor_rank)
        for group, size, expected in ((tp_group, tp, stage_ranks), (pp_group, pp, lane_ranks)):
            if (size == 1 and group is not None) or (
                size > 1
                and (group is None or tuple(dist.get_process_group_ranks(group)) != expected)
            ):
                raise ValueError("packed stage process group differs from checkpoint ownership")
        if topology.ep_size == tp:
            ep_group = tp_group if ep_group is None else ep_group
        if topology.ep_size == 1:
            expert_tp_group = tp_group if expert_tp_group is None else expert_tp_group
        for group, size, kind in (
            (ep_group, topology.ep_size, GroupKind.EXPERT),
            (expert_tp_group, topology.expert_tensor_size, GroupKind.EXPERT_TENSOR),
        ):
            expected = (
                stage_ranks
                if size == tp
                else next(
                    (g.ranks for g in topology.groups if g.kind is kind and rank in g.ranks), ()
                )
            )
            if (size == 1 and group is not None) or (
                size > 1
                and (group is None or tuple(dist.get_process_group_ranks(group)) != expected)
            ):
                raise ValueError("expert group differs from the declared EP/ETP ownership grid")
        self.rank, self.topology, self.tp_group, self.pp_group = rank, topology, tp_group, pp_group
        self.first, self.last = stage == 0, stage == pp - 1
        self.previous = lane_ranks[stage - 1] if stage else None
        self.next = lane_ranks[stage + 1] if stage + 1 < pp else None
        local_config = _local_config(config, stage=stage, tp_size=tp, pp_size=pp)
        text = local_config.graph.text
        compute_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[text.dtype]
        moe = config.graph.text.moe
        expert_span = None
        if moe is not None:
            expert_rank, _ = topology.expert_coordinates(rank)
            expert_span = AxisSpan(
                expert_rank * moe.expert_count // topology.ep_size,
                (expert_rank + 1) * moe.expert_count // topology.ep_size,
            )
            local_moe = replace(
                moe,
                shared_expert_intermediate_size=moe.shared_expert_intermediate_size // tp,
                expert_intermediate_size=moe.expert_intermediate_size
                // topology.expert_tensor_size,
            )
            text = replace(text, moe=local_moe)
            local_config = replace(local_config, graph=replace(local_config.graph, text=text))
        stage_weights = _StageWeights(weights, stage * text.layer_count)
        operators = _RankOperators(
            weights,
            tp_group,
            operator_config,
            expert_tp_group,
            device=torch.cuda.get_device_name(weights.device),
            model=weights.model_identity,
        )
        model_factory = (builtin_model_definition(local_config).text_model
                         if model_factory is None else model_factory)
        if model_factory is None:
            raise ValueError("text stage requires a declared native text constructor")
        model = model_factory(
            config=local_config, weights=stage_weights, operators=operators,
            recurrent_state_dtype=torch.bfloat16,
        )
        full = tuple(
            i for i, k in enumerate(text.layer_schedule) if k is AttentionKind.FULL_ATTENTION
        )
        g = text.gated_delta_net
        count = text.layer_count - len(full)
        states = tuple(
            sorted(
                (
                    StateSlotSpec(
                        CONV_STATE_ID,
                        compute_dtype,
                        (
                            count,
                            2 * g.key_heads * g.key_head_dim + g.value_heads * g.value_head_dim,
                            g.convolution_kernel_size - 1,
                        ),
                    ),
                    StateSlotSpec(
                        RECURRENT_STATE_ID,
                        torch.bfloat16,
                        (count, g.value_heads, g.value_head_dim, g.key_head_dim),
                    ),
                ),
                key=lambda s: s.state_id,
            )
        )
        # Finish transient table construction before allocating the durable
        # cache pool. Other buckets reuse these same immutable tables.
        rotary_dim = int(text.full_attention.head_dim * text.mrope.partial_rotary_factor)
        if mrope_tables is None:
            mrope_tables = build_bf16_mrope_tables(
                max_position=text.max_position_embeddings,
                device=weights.device,
                rotary_dim=rotary_dim,
                section=text.mrope.section,
                rope_theta=text.mrope.rope_theta,
                dtype=compute_dtype,
            )
        elif mrope_tables[2][:4] != (
            text.max_position_embeddings,
            rotary_dim,
            text.mrope.section,
            float(text.mrope.rope_theta).hex(),
        ):
            raise ValueError("shared M-RoPE tables differ from the model position contract")
        self.mrope_tables = mrope_tables
        layout = KVLayout(
            full, page_size, text.full_attention.key_value_heads, text.full_attention.head_dim,
            dtype=compute_dtype,
        )
        if cache_storage is not None:
            layout = replace(layout, storage=cache_storage)
            states = tuple(sorted((*states, *layout.quantization_state_specs),
                                  key=lambda spec: spec.state_id))
        self.cache_storage = layout.storage
        if arena is None:
            # Request ownership is independent of the graph's physical rows:
            # padding and multi-token prefill do not own recurrent state slots.
            owned_slots = captured_m if state_slots is None else state_slots
            spec = CacheArenaSpec(
                f"packed.rank{rank}.M{captured_m}",
                max(owned_slots * 2, table_pages + owned_slots),
                owned_slots,
                states,
                scratch_capacity=captured_m,
                scratch_arena_ids=(scratch_id,),
            )
            arena = CacheArena(spec, layout, weights.device, rank_id=rank)
        elif (
            arena.rank_id != rank
            or arena.device != weights.device
            or arena.layout != layout
            or arena.spec.state_specs != states
            or (state_slots is not None and arena.spec.state_slot_capacity != state_slots)
            or arena.spec.scratch_capacity < captured_m
            or scratch_id not in arena.spec.scratch_arena_ids
        ):
            raise ValueError("shared packed arena differs from rank, layout or bucket capacity")
        # Buckets share durable cache/state and one exclusively leased cache
        # scratch lane. Their activation, projection and collective scratch
        # remains separately bound; sharing never permits overlapping replay.
        self.arena = arena
        self.resource = resource or RankReplayResources(
            rank,
            f"graph{rank}.M{captured_m}",
            f"stream{rank}.M{captured_m}",
            f"buffer{rank}.M{captured_m}",
            f"workspace{rank}.M{captured_m}",
            scratch_id,
        )
        self.stage, self.tensor_rank, self.captured_m = stage, tensor_rank, captured_m
        self.embedding_mask = (
            torch.empty((captured_m, 1), dtype=compute_dtype, device=weights.device)
            if self.first
            else None
        )
        self.feature_rows = (
            torch.zeros((captured_m, text.hidden_size), dtype=compute_dtype, device=weights.device)
            if self.first else None
        )
        self.feature_width = text.hidden_size
        self.features_replicated = True
        self._vocab_span = AxisSpan(
            tensor_rank * text.vocabulary_size, (tensor_rank + 1) * text.vocabulary_size
        )

        sums: dict[tuple[int, tuple[int, ...]], FixedRankBF16SumPrimitive] = {}
        if route_completion_costs is None:
            route_completion_costs = {}

        def sum_binding(
            group: ProcessGroup | None,
            shape: tuple[int, ...],
        ) -> Callable[[Tensor], None] | None:
            if group is None:
                return None
            key = id(group), shape
            if key not in sums:
                template = torch.empty(shape, dtype=compute_dtype, device=weights.device)
                sums[key] = FixedRankBF16SumPrimitive(template, group)
            return sums[key].launch

        with operator_strategy_scope(
            operators.config,
            device=operators.device_identity,
            model=operators.model_identity,
        ):
            ordinary_sum = sum_binding(tp_group, (captured_m, text.hidden_size))
            route_completion = (
                RouteCompletionPlanner(
                    torch.empty((captured_m, moe.experts_per_token, text.hidden_size),
                                device=weights.device, dtype=compute_dtype),
                    expert_tp_group, collective_parameters(), route_completion_costs,
                ) if moe is not None and expert_tp_group is not None else None
            )
            route_sum = (
                sum_binding(expert_tp_group, (captured_m, moe.experts_per_token, text.hidden_size))
                if moe is not None and (route_completion is None or not route_completion.selected)
                else None
            )
            self.route_completion = route_completion
            expert_sum = (
                sum_binding(ep_group, (captured_m, text.hidden_size)) if moe is not None else None
            )
            self._sums = sums
            self.body = _DecodeBody(
                model,
                self.arena,
                captured_m=captured_m,
                table_pages=table_pages,
                mrope_tables=mrope_tables,
                packed_operators=operators,
                first_stage=self.first,
                last_stage=self.last,
                tp_reduce=ordinary_sum,
                embedding_mask=self.embedding_mask,
                feature_rows=self.feature_rows,
                expert_span=expert_span,
                expert_route_reduce=route_sum,
                route_completion=route_completion,
                expert_output_reduce=expert_sum,
                shared_scratch=shared_scratch,
            )
        self._embedding_table = None
        if self.first and operators.config.resolve(
            "collective", device=operators.device_identity, model=operators.model_identity
        ).embedding_host_mirror:
            from cubic_serve.ops.cuda.embedding.immutable import ImmutableEmbeddingTable
            weight = model.weights["model.language_model.embed_tokens.weight"]
            self._embedding_table = ImmutableEmbeddingTable.create(
                weight, self._vocab_span, tp_group, weights.device, compute_dtype,
                operators=operators,
            )
            if self._embedding_table is not None:
                self.body.embedding = self._embedding_table.bind(
                    self.body.buffers.token_ids, self.body.buffers.hidden)
                self.body._embedding_complete = True
        self._overlap_factory: dict[str, Any] = dict(
            config=config,
            weights=weights,
            topology=topology,
            tp_group=tp_group,
            pp_group=pp_group,
            ep_group=ep_group,
            expert_tp_group=expert_tp_group,
            table_pages=table_pages,
            page_size=page_size,
            cache_storage=cache_storage,
            operator_config=operator_config,
            arena=self.arena,
            mrope_tables=self.mrope_tables,
            resource=resource,
            route_completion_costs=route_completion_costs,
            model_factory=model_factory,
        )
        policy = (operator_config or OperatorConfig()).resolve(
            "collective", device=operators.device_identity, model=operators.model_identity
        )
        assert isinstance(policy, CollectiveParameters)
        self._overlap_policy = policy
        with operator_strategy_scope(
            operator_config or OperatorConfig(),
            device=operators.device_identity, model=operators.model_identity,
        ):
            self._overlap_thread_candidates = overlap_collective_threads()
        self.overlap_plan = DecodeOverlapPlan()
        self._private_bytes = max(0, torch.cuda.memory_allocated(weights.device) - private_start)
        self._overlap_resources: tuple[Any, ...] | None = None
        self._use_overlap = False
        self.overlap_measurements: list[tuple[tuple[int, ...], float]] = []
        self._captured_once = False
        self._graph: torch.cuda.CUDAGraph | None = None
        self.stream = torch.cuda.Stream(device=weights.device)  # type: ignore[no-untyped-call]

    @property
    def program(self) -> tuple[Any, ...]:
        """Retained execution edges, excluding startup construction recipes.

        Overlap children own their bound operators too. Their model factories
        are construction inputs, not executable edges of a captured graph.
        """
        return self.body, self._overlap_resources

    @property
    def buffers(self) -> _Buffers:
        return self.body.buffers

    @property
    def hidden(self) -> Tensor:
        return self.buffers.hidden

    @property
    def logits(self) -> Tensor | None:
        return self.buffers.logits if self.last else None

    @property
    def embedding_span(self) -> AxisSpan | None:
        return self._vocab_span if self.first else None

    @property
    def embedding_input_span(self) -> AxisSpan | None:
        if self._embedding_table is not None:
            return AxisSpan(0, self._embedding_table.vocabulary)
        return self.embedding_span

    @property
    def vocab_span(self) -> AxisSpan | None:
        return self._vocab_span if self.last else None

    @property
    def graph(self) -> torch.cuda.CUDAGraph:
        if self._graph is None:
            raise ValueError("packed stage has no startup-captured graph")
        if self._use_overlap and self._overlap_resources is not None:
            return cast(torch.cuda.CUDAGraph, self._overlap_resources[0])
        return self._graph

    def synchronize(self) -> None:
        self.stream.synchronize()

    def set_active_rows(self, logical_m: int) -> None:
        # GEMM capacity stays physical M. Batch fill supplies explicit row
        # activity and continuation metadata; padding never acquires identity.
        if type(logical_m) is not int or not 0 < logical_m <= self.captured_m:
            raise ValueError("logical rows lie outside the packed graph capacity")

    def prepare_scratch(self, resource: RankReplayResources | None = None) -> None:
        if resource is not None and resource != self.resource:
            raise ValueError("packed stage cannot retarget its captured resource domain")
        self.body.prepare_capture_scratch(self.resource)
        if self.embedding_mask is not None:
            self.embedding_mask.zero_()
        if self.feature_rows is not None:
            self.feature_rows.zero_()

    def bind_tokens(self, tokens: Tensor) -> None:
        """Input fill before the captured forward, under the controller's lease."""
        b = self.body.buffers
        if self.first:
            assert self.embedding_mask is not None
            span = getattr(self, "embedding_input_span", self.embedding_span)
            assert span is not None
            owns = (tokens >= span.start) & (tokens < span.stop)
            b.token_ids.copy_(torch.where(owns, tokens - span.start, 0))
            self.embedding_mask.copy_(owns[:, None])

    def bind_scratch(self, resource: RankReplayResources) -> None:
        """Restore private padding bindings without clearing live scratch state."""
        if resource != self.resource:
            raise ValueError("packed stage cannot retarget its captured resource domain")
        self.body.bind_capture_scratch(resource)
        if self.embedding_mask is not None:
            self.embedding_mask.zero_()

    def fill_token_ids(self, tokens: Tensor) -> None:
        """Shared serving-audit input ABI; preserve rank-local vocabulary ownership."""
        self.bind_tokens(tokens)

    def launch(self) -> None:
        # Rank IDs in P2POp are global, even with a pipeline subgroup.
        if self.previous is not None:
            assert self.pp_group is not None
            for request in dist.batch_isend_irecv(
                [
                    dist.P2POp(
                        dist.irecv, self.body.buffers.hidden, self.previous, group=self.pp_group
                    )
                ]
            ):
                request.wait()
        self.body.launch()
        if self.next is not None:
            assert self.pp_group is not None
            for request in dist.batch_isend_irecv(
                [dist.P2POp(dist.isend, self.body.buffers.hidden, self.next, group=self.pp_group)]
            ):
                request.wait()

    def capture(self) -> None:
        if self._captured_once:
            raise ValueError("packed stage already captured; runtime recapture is forbidden")
        self.prepare_scratch()
        self.stream.wait_stream(torch.cuda.current_stream(self.arena.device))
        for _ in range(2):
            with torch.cuda.stream(self.stream):
                self.launch()
            self.stream.synchronize()
            barrier()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=self.stream):
            self.launch()
        graph.instantiate()
        self.stream.synchronize()
        barrier()
        self._graph = graph
        self._captured_once = True

    def captured_body_graph(self, *, produce_logits: bool = True) -> torch.cuda.CUDAGraph:
        """Expose the sealed serial launch body to startup composition owners."""
        if not self._captured_once or self._graph is None:
            raise ValueError("body graph is not available for composition")
        if not produce_logits:
            raise ValueError("the requested body phase was not captured")
        # launch() is the serial body even when ordinary decode has selected
        # an overlap portfolio. Preserve that exact composition contract.
        return self._graph

    def select_decode_overlap(self, independent_rows: bool) -> None:
        """Select a startup-captured variant under the parent's replay lease."""
        self._use_overlap = independent_rows and self._overlap_resources is not None

    @property
    def owned_decode_graph(self) -> torch.cuda.CUDAGraph:
        """The sealed variant selected for independent one-token owners."""
        return self._overlap_resources[0] if self._overlap_resources is not None else self._graph

    def _capture_interleaved(self, children: list[QuantizedDecodeStage]) -> tuple[Any, ...]:
        schedule = DecodeInterleave(len(children), self.arena.device)

        def launch() -> None:
            start = 0
            for child in children:
                stop = start + child.captured_m
                _copy_decode_input_rows(self.buffers, child.buffers, start, stop)
                assert child.embedding_mask is not None and self.embedding_mask is not None
                child.embedding_mask.copy_(self.embedding_mask[start:stop])
                start = stop
            schedule.launch(
                [
                    (
                        partial(operation, operand)
                        for operation, operand in child.body.launch_steps()
                    )
                    for child in children
                ]
            )
            start = 0
            for child in children:
                stop = start + child.captured_m
                for name in ("hidden", "norm", "logits"):
                    getattr(self.buffers, name)[start:stop].copy_(getattr(child.buffers, name))
                start = stop

        self.prepare_scratch()
        self.stream.wait_stream(torch.cuda.current_stream())
        for _ in range(2):
            with torch.cuda.stream(self.stream):
                launch()
            self.synchronize()
            barrier()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=self.stream):
            launch()
        graph.instantiate()
        self.synchronize()
        barrier()
        # Retain the identical launch recipe for deferred cold observers. It is
        # never a serving fallback: runtime dispatch replays resources[0] only.
        return graph, children, schedule, launch

    def _overlap_child_factory(self, plan: DecodeOverlapPlan) -> dict[str, Any]:
        factory = dict(self._overlap_factory)
        factory["operator_config"] = collective_thread_config(
            factory.get("operator_config") or OperatorConfig(), plan.collective_threads,
        )
        return factory

    def _bind_overlap_plan(self, plan: DecodeOverlapPlan) -> None:
        children = [
            QuantizedDecodeStage(captured_m=m, **self._overlap_child_factory(plan))
            for m in plan.parts
        ]
        for child in children:
            child.prepare_scratch()
            child.launch()
            torch.cuda.synchronize(self.arena.device)
            barrier()
        self._overlap_resources = self._capture_interleaved(children)
        self.overlap_plan = plan

    def prepare_decode_overlap(self, *, retained_plan: DecodeOverlapPlan | None = None) -> None:
        """Bounded pre-readiness search; no runtime capture or eager fallback.

        This selects a candidate, never supplies a numerical/performance seal.
        A new binding must explicitly declare row independence and absence of
        hidden collectives. Routing and PP dependencies need separate support.
        """
        if self._graph is None or self._overlap_resources is not None or self.overlap_measurements:
            raise ValueError("overlap planning requires one fresh startup-captured stage")
        if retained_plan is not None:
            retained_parts = retained_plan.parts
            # Rebind the cold-profiled plan to final cache addresses, without
            # reopening selection after KV has consumed the available memory.
            if not retained_parts:
                return
            if (
                len(retained_parts) < 2
                or any(m <= 0 for m in retained_parts)
                or sum(retained_parts) != self.captured_m
                or self.topology.tp_size == 1
                or self.topology.pp_size != 1
                or self.body._packed_moe is not None
            ):
                raise ValueError("invalid retained decode overlap partition")
            self._bind_overlap_plan(retained_plan)
            return
        if (
            self.topology.tp_size == 1
            or self.topology.pp_size != 1
            or self.body._packed_moe is not None
        ):
            return
        projections = [
            value
            for value in vars(self.body).values()
            if isinstance(value, PackedProjectionDispatch) and value.operation == "linear"
        ]
        counts = [projection.row_tile_count for projection in projections]
        row_tiles = (
            0
            if not counts or any(c is None for c in counts)
            else min(c for c in counts if c is not None)
        )
        common = torch.tensor([row_tiles], dtype=torch.int64, device=self.arena.device)
        all_reduce(common, op=dist.ReduceOp.MIN)
        row_tiles = int(common.item())
        if not row_tiles:
            return
        policy = self._overlap_policy
        partitions = policy.overlap_partitions(self.captured_m, row_tiles)
        if not partitions:
            return

        prepare = decode_calibration_prepare(
            rows=self.captured_m,
            vocabulary_size=self._vocab_span.size * self.topology.tp_size,
            device=self.arena.device,
            reset=self.prepare_scratch,
            bind_tokens=self.bind_tokens,
        )

        def measure(graph: torch.cuda.CUDAGraph) -> float:
            return measure_decode_graph(graph, prepare, self.stream, policy.decode_overlap_samples)

        baseline = measure(self._graph)
        best = 1 + policy.decode_overlap_min_gain_percent / 100
        self.overlap_measurements.append(((self.captured_m,), baseline))
        allocation_start = torch.cuda.memory_allocated(self.arena.device)
        thread_candidates = common_overlap_thread_candidates(
            self._overlap_thread_candidates, self.arena.device,
        )
        candidates = (
            DecodeOverlapPlan(parts=parts, collective_threads=threads)
            for threads in thread_candidates for parts in partitions
        )
        for plan in candidates:
            parts = plan.parts
            children = []
            for m in parts:
                free, _ = torch.cuda.mem_get_info(self.arena.device)
                # Check each allocation against a full-body upper bound, then
                # charge its actual size. Charging every child as full-M at
                # once would reject useful multi-part plans unnecessarily.
                used = max(0, torch.cuda.memory_allocated(self.arena.device) - allocation_start)
                allowed = int(
                    used + self._private_bytes <= policy.decode_overlap_max_bytes
                    and self._private_bytes < free // 2
                )
                common.fill_(allowed)
                all_reduce(common, op=dist.ReduceOp.MIN)
                if not int(common.item()):
                    break
                children.append(
                    QuantizedDecodeStage(captured_m=m, **self._overlap_child_factory(plan))
                )
            if len(children) != len(parts):
                del children
                continue
            for child in children:
                child.prepare_scratch()
                child.launch()
                torch.cuda.synchronize(self.arena.device)
                barrier()
            resources = self._capture_interleaved(children)
            control, elapsed = compare_decode_graphs(
                self._graph, resources[0], prepare, self.stream, policy.decode_overlap_samples
            )
            self.overlap_measurements.append(((self.captured_m,), control))
            self.overlap_measurements.append((parts, elapsed))
            gain = control / elapsed
            if gain > best:
                self.overlap_plan = plan
                best = gain
            # Selection retains a recipe, not every winner's private arena.
            # Otherwise an early winner consumes the later candidate's budget.
            resources[0].reset()
            del resources, children, child
        if self.overlap_plan.parts:
            self._bind_overlap_plan(self.overlap_plan)
        self.prepare_scratch()
        self.synchronize()
        barrier()
        logging.getLogger(__name__).info(
            "decode overlap rank=%d M=%d calibrated_ms=%s selected=%s",
            self.rank,
            self.captured_m,
            format_decode_measurements(self.overlap_measurements),
            self.overlap_plan,
        )

    def replay(self) -> None:
        # The controller orders input fill and durable consumption around this
        # named stream; replay must not silently launch on its caller stream.
        with torch.cuda.stream(self.stream):
            self.graph.replay()

    def close(self) -> None:
        """Release graph-held NCCL resources before destroying process groups."""
        torch.cuda.synchronize(self.arena.device)
        if self._overlap_resources is not None:
            self._overlap_resources[0].reset()
            self._overlap_resources = None
        if self._graph is not None:
            self._graph.reset()
            self._graph = None
