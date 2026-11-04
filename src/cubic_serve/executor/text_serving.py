"""Fixed-address serving stage for the model-owned Flash-Next text traversal.

This adapter owns storage, communication and capture. Model arithmetic stays in
qwen4_exp; matrix formats stay in the shared linear operators. Constructing or
capturing a stage is not performance admission or numerical qualification.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from cubic_serve.contracts.execution import GroupKind, ParallelTopology, RankReplayResources
from cubic_serve.executor.text_graph import TextGraphBindings
from cubic_serve.models.parallel_weights import AxisSpan
from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.models.qwen4_exp.model import (
    Qwen4ExpConfig,
    Qwen4ExpGraphResources,
    Qwen4ExpOperationBinder,
    bind_qwen4_exp_text_graph,
    qwen4_exp_cache_layout,
)
from cubic_serve.models.qwen4_exp.weights import logical_quantized_weights, quantized_owner_slices
from cubic_serve.ops.cuda.attention.paged_decode import PageCOWCopyPrimitive
from cubic_serve.ops.cuda.collective import FixedRankBF16SumPrimitive
from cubic_serve.ops.cuda.embedding.token import bind_scaled_sharded_embedding
from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
from cubic_serve.ops.cuda.position.qwen_mrope_table import (
    MRoPETableEvidence,
    build_bf16_mrope_tables,
)
from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope
from cubic_serve.runtime.cache import CacheArena, CacheArenaSpec
from cubic_serve.runtime.cache.quantization import KVCacheStorageSpec
from cubic_serve.runtime.quantized import QuantizedWeights


@dataclass(slots=True)
class TextServingBuffers:
    token_ids: Tensor
    position_ids: Tensor
    page_source: Tensor
    page_destination: Tensor
    token_offsets: Tensor
    page_table: Tensor
    sequence_lengths: Tensor
    segment_lengths: Tensor
    row_activity: Tensor
    state_slot_maps: tuple[Tensor, ...]
    norm: Tensor
    hidden: Tensor
    target_hidden: Tensor | None = None


class FlashNextDecodeStage:
    """One startup-bound bucket with original rank ownership and explicit sums."""

    def __init__(
        self,
        config: Qwen4ExpConfig,
        weights: QuantizedWeights,
        topology: ParallelTopology,
        *,
        tp_group: ProcessGroup | None,
        pp_group: ProcessGroup | None,
        captured_m: int,
        ep_group: ProcessGroup | None = None,
        expert_tp_group: ProcessGroup | None = None,
        table_pages: int = 1,
        page_size: int = 16,
        arena: CacheArena | None = None,
        state_slots: int | None = None,
        operator_config: OperatorConfig | None = None,
        recurrent_state_dtype: torch.dtype | None = None,
        cache_storage: KVCacheStorageSpec | None = None,
        mrope_tables: tuple[Tensor, Tensor, MRoPETableEvidence] | None = None,
        resource: RankReplayResources | None = None,
        input_transform: Callable[[TextGraphBindings, Tensor], Callable[[], None]] | None = None,
        shared_scratch: dict[tuple, Tensor] | None = None,
    ) -> None:
        self.rank = weights.rank
        placement = topology.placements[self.rank]
        self.stage, self.tensor_rank = placement.pipeline_stage, placement.tensor_rank
        self.first, self.last = self.stage == 0, self.stage == topology.pp_size - 1
        self.captured_m = captured_m
        self.config, self.weights, self.topology = config, weights, topology
        from cubic_serve.models.parallel_weights import IndexerTP
        index = config.text.qsa
        self.indexer_plan = IndexerTP(index.query_heads, index.head_dim, index.compression_ratio,
                                      topology.tp_size, self.tensor_rank)
        self.pp_group = pp_group
        if (
            not dist.is_initialized()
            or dist.get_rank() != self.rank
            or weights.device.index != placement.device_index
            or captured_m < 1
            or captured_m & (captured_m - 1)
            or table_pages < 1
        ):
            raise ValueError("text stage differs from fixed rank/bucket ownership")
        stage_ranks = tuple(p.rank for p in topology.placements if p.pipeline_stage == self.stage)
        lane_ranks = tuple(p.rank for p in topology.placements if p.tensor_rank == self.tensor_rank)
        for group, ranks in ((tp_group, stage_ranks), (pp_group, lane_ranks)):
            self._validate_group(group, ranks)
        if topology.ep_size == 1:
            expert_tp_group = tp_group if expert_tp_group is None else expert_tp_group
        if topology.ep_size == topology.tp_size:
            ep_group = tp_group if ep_group is None else ep_group
        for group, size, kind in (
            (ep_group, topology.ep_size, GroupKind.EXPERT),
            (expert_tp_group, topology.expert_tensor_size, GroupKind.EXPERT_TENSOR),
        ):
            ranks = (
                stage_ranks
                if size == topology.tp_size
                else next(
                    (g.ranks for g in topology.groups if g.kind == kind and self.rank in g.ranks),
                    (self.rank,),
                )
            )
            self._validate_group(group, tuple(ranks))
        self.previous = None if self.first else lane_ranks[self.stage - 1]
        self.next = None if self.last else lane_ranks[self.stage + 1]
        self.stream = torch.cuda.Stream(device=weights.device)  # type: ignore[no-untyped-call]
        self._graph: torch.cuda.CUDAGraph | None = None
        self._state_graph: torch.cuda.CUDAGraph | None = None
        self._captured_once = False
        self._closed = False
        text = config.text
        if text.dtype not in ("bfloat16", "float16"):
            raise ValueError("text stage requires BF16/FP16 activation storage")
        dtype = torch.bfloat16 if text.dtype == "bfloat16" else torch.float16
        self.recurrent_state_dtype = recurrent_state_dtype or getattr(
            torch, text.gated_delta_net.recurrent_state_dtype
        )
        layout, specs = qwen4_exp_cache_layout(
            config, topology, self.rank, page_size=page_size, dtype=dtype,
            recurrent_state_dtype=self.recurrent_state_dtype,
        )
        if cache_storage is not None:
            layout = replace(layout, storage=cache_storage)
            specs = tuple(sorted((*specs, *layout.quantization_state_specs),
                                 key=lambda spec: spec.state_id))
        self.cache_storage = layout.storage
        scratch_id = (
            resource.scratch_arena_id if resource is not None else f"text.rank{self.rank}.scratch"
        )
        if arena is None:
            owners = captured_m if state_slots is None else state_slots
            arena = CacheArena(
                CacheArenaSpec(
                    f"text.rank{self.rank}",
                    max(table_pages + owners, 2 * owners),
                    owners,
                    specs,
                    captured_m,
                    (scratch_id,),
                ),
                layout,
                weights.device,
                rank_id=self.rank,
            )
        if (
            arena.rank_id != self.rank
            or arena.layout != layout
            or arena.spec.state_specs != specs
            or arena.device != weights.device
            or arena.spec.scratch_capacity < captured_m
            or scratch_id not in arena.spec.scratch_arena_ids
        ):
            raise ValueError("shared text arena differs from the model state/lease schema")
        self.arena = arena
        self.resource = resource or RankReplayResources(
            self.rank,
            f"text.graph{self.rank}.M{captured_m}",
            f"text.stream{self.rank}.M{captured_m}",
            f"text.buffer{self.rank}.M{captured_m}",
            f"text.workspace{self.rank}.M{captured_m}",
            scratch_id,
        )
        if self.resource.rank != self.rank:
            raise ValueError("text graph resource belongs to another rank")

        def allocate(*shape: int, storage: torch.dtype = dtype) -> Tensor:
            return torch.zeros(shape, dtype=storage, device=weights.device)

        hidden = allocate(captured_m, text.hidden_size * text.hyperconnection.stream_count)
        slot_map = allocate(2, 1, captured_m, storage=torch.int32)
        self.buffers = TextServingBuffers(
            allocate(captured_m, storage=torch.int64),
            allocate(3, captured_m, storage=torch.int64),
            allocate(captured_m, storage=torch.int32),
            allocate(captured_m, storage=torch.int32),
            allocate(captured_m, storage=torch.int32),
            allocate(captured_m, table_pages, storage=torch.int32),
            allocate(captured_m, storage=torch.int32),
            allocate(captured_m, storage=torch.int32),
            allocate(captured_m, storage=torch.int32),
            (slot_map,),
            allocate(captured_m, text.hidden_size),
            hidden,
        )
        self.embedding_mask = allocate(captured_m, 1) if self.first else None
        self.feature_rows = allocate(captured_m, text.hidden_size) if self.first else None
        self.feature_width = text.hidden_size
        self.features_replicated = True
        self._logical_positions = allocate(captured_m, storage=torch.int64)
        self._active_rows = allocate(1, storage=torch.int32)
        self._active_rows.fill_(captured_m)
        self._vocab_span = AxisSpan(
            self.tensor_rank * text.vocabulary_size // topology.tp_size,
            (self.tensor_rank + 1) * text.vocabulary_size // topology.tp_size,
        )
        if mrope_tables is None:
            mrope_tables = build_bf16_mrope_tables(
                max_position=text.max_position_embeddings,
                device=weights.device,
                rotary_dim=int(text.full_attention.head_dim * text.mrope.partial_rotary_factor),
                section=text.mrope.section,
                rope_theta=text.mrope.rope_theta,
                dtype=dtype,
            )
        if mrope_tables[2][:4] != (
            text.max_position_embeddings,
            int(text.full_attention.head_dim * text.mrope.partial_rotary_factor),
            text.mrope.section,
            float(text.mrope.rope_theta).hex(),
        ):
            raise ValueError("shared rotary tables differ from the declared model positions")
        self.mrope_tables = mrope_tables
        self.operators = PackedLinearOperators(
            config=operator_config or OperatorConfig(),
            prepared_weights=weights.prepared_weights,
            materialized_weights=weights.materialized_weights,
        )
        self._sums: dict[tuple[int, tuple[int, ...]], FixedRankBF16SumPrimitive] = {}

        def reduction(
            group: ProcessGroup | None, shape: tuple[int, ...]
        ) -> Callable[[Tensor], None] | None:
            if group is None:
                return None
            identity = id(group), shape
            if identity not in self._sums:
                self._sums[identity] = FixedRankBF16SumPrimitive(allocate(*shape), group)
            return self._sums[identity].launch

        logical = logical_quantized_weights(config)
        owner_bindings = weights.bindings or quantized_owner_slices(
            config, topology, self.rank, logical_weights=logical
        )
        owners_by_name = {o.tensor_name: o for o in owner_bindings}
        logical_shapes = {w.name: w.shape for w in logical}
        count = text.layer_count // topology.pp_size
        layer_ids = tuple(range(self.stage * count, (self.stage + 1) * count))
        tables, scales = {}, {}
        for layer in layer_ids:
            if layer + 1 not in text.ple.layer_ids:
                continue
            base = f"model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding"
            start, inventory = 0, []
            for part in range(text.ple.split_parts):
                name = f"{base}.shard_{part}.weight"
                if name in owners_by_name:
                    table = weights[name]
                    if isinstance(table, QuantizedMatrix) and not all(
                        t.device.type == "cpu" and t.is_pinned() for t in table.payload.values()
                    ):
                        table = self.operators.materialize_weight(table, weights.device)
                    inventory.append((start + owners_by_name[name].spans[0].start, table))
                start += logical_shapes[name][0]
            tables[layer] = tuple(inventory)
            scale = (
                weights[base + ".weight_scale"]
                if text.ple_embedding_dtype == "float8_e4m3fn"
                else allocate(1)
            )
            if not isinstance(scale, Tensor):
                raise ValueError("lexical scale must be an ordinary tensor")
            if text.ple_embedding_dtype != "float8_e4m3fn":
                scale.fill_(1)
            scales[layer] = scale
        self._scratch_keepalive: list[Tensor] = []
        with operator_strategy_scope(self.operators.config, device="", model=""):
            self.helper = TextGraphBindings(
                self.operators, weights.device, dtype, cache_arena=arena,
                cache_source_slots=slot_map[0, 0], cache_destination_slots=slot_map[1, 0],
                shared_scratch=shared_scratch,
            )
            b = self.buffers

            def embed() -> None:
                return None

            if self.first:
                embedding = allocate(captured_m, text.hidden_size)
                embedding_weight = weights["model.language_model.embed_tokens.weight"]
                scale = allocate(1)
                scale.fill_(1)
                lookup = self.operators.bind_scaled_sharded_embedding(
                    b.token_ids.view(captured_m, 1),
                    ((self._vocab_span.start, embedding_weight),),
                    scale,
                    b.row_activity,
                    embedding,
                )
                self._scratch_keepalive.extend((*lookup.scratch, embedding, scale))
                total = reduction(tp_group, tuple(embedding.shape))
                target = hidden.view(
                    captured_m, text.hyperconnection.stream_count, text.hidden_size
                )
                source = embedding[:, None]
                transform = (None if input_transform is None
                             else input_transform(self.helper, hidden))

                def embed() -> None:
                    lookup.launch()
                    if total is not None:
                        total(embedding)
                    target.copy_(torch.where(
                        self.embedding_mask[:, None].bool(), source, self.feature_rows[:, None]
                    ))
                    if transform is not None:
                        transform()

            expert_rank, _ = topology.expert_coordinates(self.rank)
            local_experts = text.moe.expert_count // topology.ep_size
            resources = Qwen4ExpGraphResources(
                hidden=hidden,
                tokens=b.token_ids,
                segment_lengths=b.segment_lengths,
                row_activity=b.row_activity,
                active_rows=self._active_rows,
                positions=b.position_ids,
                logical_positions=self._logical_positions,
                states=arena.states,
                state_sources={name: slot_map[0, 0] for name in arena.states},
                state_destinations={name: slot_map[1, 0] for name in arena.states},
                key=arena.key,
                value=arena.value,
                written=arena.written,
                kv_layer_ids=layout.layer_ids,
                page_fields=arena.page_fields,
                page_table=b.page_table,
                page_destinations=b.page_destination,
                token_offsets=b.token_offsets,
                visible_lengths=b.sequence_lengths,
                cosine_table=mrope_tables[0],
                sine_table=mrope_tables[1],
                rope_evidence=mrope_tables[2],
                ple_shards=tables,
                ple_scales=scales,
                embedding=embed,
                first_expert=expert_rank * local_experts,
                local_expert_count=local_experts,
                indexer_plan=self.indexer_plan,
                indexer_group=tp_group,
                reduce_attention=reduction(tp_group, (captured_m, text.hidden_size)),
                reduce_lexical=reduction(tp_group, (captured_m, text.ple.embedding_dim)),
                reduce_routes=reduction(
                    expert_tp_group, (captured_m, text.moe.experts_per_token, text.hidden_size)
                ),
                reduce_shared=reduction(tp_group, (captured_m, text.hidden_size)),
                reduce_experts=reduction(ep_group, (captured_m, text.hidden_size)),
            )
            binder = Qwen4ExpOperationBinder(config, weights, self.helper, resources)
            self.program = bind_qwen4_exp_text_graph(
                config,
                binder,
                owned_layer_ids=layer_ids,
                first_stage=self.first,
                last_stage=self.last,
            )
            self.logits = binder.logits
            if binder.mixed is not None:
                b.norm = binder.mixed
            self._page_copy = PageCOWCopyPrimitive(
                arena.key,
                arena.value,
                arena.written,
                b.page_source,
                b.page_destination,
                page_fields=tuple(arena.page_fields.values()),
                cache_encoding=self.helper.cache_encoding,
            )
        self._prepare_padding(scratch_id)
        self.prepare_scratch()

    def _prepare_padding(self, scratch_id: str) -> None:
        arena, captured_m, weights = self.arena, self.captured_m, self.weights
        page_slots = arena.padding_page_slots(captured_m, scratch_id)
        self._scratch_pages = torch.tensor(page_slots, dtype=torch.int64, device=weights.device)
        self._scratch_page_reset = (self._scratch_pages if len(set(page_slots)) == len(page_slots)
                                   else torch.tensor(tuple(sorted(set(page_slots))),
                                                     dtype=torch.int64, device=weights.device))
        state_slots = arena.padding_state_slots(captured_m, scratch_id)
        self._scratch_states = torch.tensor(state_slots, dtype=torch.int64, device=weights.device)
        self._scratch_state_reset = (self._scratch_states if len(set(state_slots)) == len(state_slots)
                                     else torch.tensor(tuple(sorted(set(state_slots))),
                                                       dtype=torch.int64, device=weights.device))

    @staticmethod
    def _validate_group(group: ProcessGroup | None, ranks: tuple[int, ...]) -> None:
        if (len(ranks) == 1 and group is not None) or (
            len(ranks) > 1
            and (group is None or tuple(dist.get_process_group_ranks(group)) != ranks)
        ):
            raise ValueError("process group differs from declared rank ownership")

    @property
    def hidden(self) -> Tensor:
        return self.buffers.hidden

    @property
    def draft_feedback(self) -> Tensor:
        """The model's materialized residual precedes its final HC mixer."""
        return self.buffers.hidden

    @property
    def embedding_span(self) -> AxisSpan | None:
        return self._vocab_span if self.first else None

    @property
    def embedding_input_span(self) -> AxisSpan | None:
        return AxisSpan(0, self.config.text.vocabulary_size) if self.first else None

    @property
    def vocab_span(self) -> AxisSpan | None:
        return self._vocab_span if self.last else None

    @property
    def graph(self) -> torch.cuda.CUDAGraph:
        if self._graph is None or self._closed:
            raise ValueError("text stage has no live startup-captured graph")
        return self._graph

    def set_active_rows(self, logical_m: int) -> None:
        if not 0 < logical_m <= self.captured_m:
            raise ValueError("logical rows exceed the captured text bucket")

    def bind_scratch(self, resource: RankReplayResources) -> None:
        if resource != self.resource:
            raise ValueError("captured text resources cannot be retargeted")
        b = self.buffers
        b.segment_lengths.zero_()
        b.row_activity.zero_()
        b.position_ids.zero_()
        b.token_offsets.zero_()
        b.sequence_lengths.fill_(1)
        b.page_source.copy_(self._scratch_pages)
        b.page_destination.copy_(self._scratch_pages)
        b.page_table.copy_(self._scratch_pages[:, None])
        b.state_slot_maps[0].copy_(self._scratch_states[None, None])
        if self.embedding_mask is not None:
            self.embedding_mask.zero_()
        if self.feature_rows is not None:
            self.feature_rows.zero_()

    def prepare_scratch(self, resource: RankReplayResources | None = None) -> None:
        self.bind_scratch(resource or self.resource)
        self.arena.reset_state_rows(self._scratch_state_reset)
        self.arena.reset_page_rows(self._scratch_page_reset)

    def fill_token_ids(self, tokens: Tensor) -> None:
        self.buffers.token_ids.copy_(tokens)
        if self.embedding_mask is not None:
            self.embedding_mask.fill_(1)

    def launch_projection(self) -> None:
        """Use the bound head input without replaying stateful producers."""
        if self._closed or not callable(self.program.logits_step):
            raise ValueError("text stage has no available bound output projection")
        self.program.logits_step()

    def launch(self, *, produce_logits: bool = True) -> None:
        if self._closed:
            raise ValueError("text stage is closed")
        if self.previous is not None:
            for work in dist.batch_isend_irecv(
                [dist.P2POp(dist.irecv, self.hidden, self.previous, group=self.pp_group)]
            ):
                work.wait()
        b = self.buffers
        self._logical_positions.copy_(b.sequence_lengths)
        self._logical_positions.sub_(1)
        self._page_copy.launch(
            self.arena.key, self.arena.value, self.arena.written, b.page_source, b.page_destination
        )
        if produce_logits:
            self.program.launch()
        else:
            self.program.launch_feedback()
        if self.next is not None:
            for work in dist.batch_isend_irecv(
                [dist.P2POp(dist.isend, self.hidden, self.next, group=self.pp_group)]
            ):
                work.wait()

    def launch_state(self) -> None:
        """Provider state/feedback phase, also used by its sealed state graph."""
        self.launch(produce_logits=False)

    def capture(self) -> None:
        if self._captured_once or self._closed:
            raise ValueError("text stage recapture or closed-stage capture is forbidden")
        self.prepare_scratch()
        self.stream.wait_stream(torch.cuda.current_stream(self.arena.device))
        for _ in range(2):
            with torch.cuda.stream(self.stream):
                self.launch()
            self.stream.synchronize()
            dist.barrier()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=self.stream):
            self.launch()
        graph.instantiate()
        self.stream.synchronize()
        dist.barrier()
        self._graph, self._captured_once = graph, True
        if hasattr(self, "draft_spec"):
            state_graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(state_graph, stream=self.stream):
                self.launch(produce_logits=False)
            state_graph.instantiate()
            self.stream.synchronize()
            dist.barrier()
            self._state_graph = state_graph

    def captured_body_graph(self, *, produce_logits: bool = True) -> torch.cuda.CUDAGraph:
        """The sealed equivalent of launch(), for startup composition owners."""
        if self._closed or not self._captured_once:
            raise ValueError("body graph is not available for composition")
        graph = self.graph if produce_logits else self._state_graph
        if graph is None:
            raise ValueError("the requested body phase was not captured")
        return graph

    def select_decode_overlap(self, independent_rows: bool) -> None:
        """This baseline has one serial graph and no qualified overlap portfolio."""
        if type(independent_rows) is not bool:
            raise TypeError("decode ownership classification must be boolean")

    @property
    def owned_decode_graph(self) -> torch.cuda.CUDAGraph:
        """The sealed serial graph also owns the independent decode case."""
        return self.graph

    def replay(self) -> None:
        with torch.cuda.stream(self.stream):
            self.graph.replay()

    def replay_state(self) -> None:
        """Replay the startup-captured draft state/feedback phase."""
        if self._state_graph is None or self._closed:
            raise ValueError("text draft has no sealed feedback graph")
        with torch.cuda.stream(self.stream):
            self._state_graph.replay()

    def synchronize(self) -> None:
        self.stream.synchronize()

    def close(self) -> None:
        self.synchronize()
        if self._graph is not None:
            self._graph.reset()
            self._graph = None
        if self._state_graph is not None:
            self._state_graph.reset()
            self._state_graph = None
        self._closed = True
