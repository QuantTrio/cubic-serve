"""Rank-local MTP stages over shared packed operators, cache and graph contracts."""

from __future__ import annotations

from typing_extensions import TypeAliasType

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.cuda.graphs import CUDAGraph
from torch.cuda.graphs import graph as capture_graph
from torch.cuda.streams import Stream
from torch.distributed import ProcessGroup

from cubic_serve.models.parallel_weights import AxisSpan
from cubic_serve.models.qwen3_5.model import Qwen3_5Config
from cubic_serve.models.qwen3_5.weights import Qwen3_5ParallelWeightPlan, Qwen3_5RankWeights
from cubic_serve.models.qwen3_5_moe.config import Qwen3_5MoeConfig
from cubic_serve.models.qwen3_5_moe.parallel import (
    Qwen3_5MoeParallelPlan,
    Qwen3_5MoeRankWeights,
)
from cubic_serve.ops.cuda.attention import (
    BF16PagedGQADecodePrimitive,
    PageTokenWritePrimitive,
)
from cubic_serve.ops.cuda.attention.paged_decode import paged_gqa_kv_splits
from cubic_serve.ops.cuda.collective import FixedRankBF16SumPrimitive, barrier
from cubic_serve.ops.cuda.embedding import BF16TokenEmbeddingPrimitive
from cubic_serve.ops.cuda.launch_cost import LinearLaunchCost, device_resources, linear_cost_scope
from cubic_serve.ops.cuda.linear import (
    BF16LinearKernel,
    bind_bf16_linear_kernel,
    launch_bf16_linear_kernel,
)
from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
from cubic_serve.ops.cuda.moe import (
    BF16MoeSharedCombinePrimitive,
    BF16RoutedExpertsPrimitive,
    MoeBranchStreams,
)
from cubic_serve.ops.cuda.normalization import ZeroCenteredRMSNormPrimitive
from cubic_serve.ops.cuda.pointwise import (
    BF16PackedSwiGLUPrimitive,
    BF16ResidualAddPrimitive,
    BF16SigmoidOutputGatePrimitive,
    QwenAttentionQGateSplitPrimitive,
)
from cubic_serve.ops.cuda.position import (
    MRoPECosSinGatherPrimitive,
    MRoPETableEvidence,
    QwenMRoPEApplyPrimitive,
    build_bf16_mrope_tables,
)

Qwen3_5MTPObserver = TypeAliasType("Qwen3_5MTPObserver", Callable[[str, Tensor], None])


def _mtp_graph_capacity_supported(captured_m: int, max_proposal_tokens: int) -> bool:
    return max_proposal_tokens > 0 and captured_m > 0 and not captured_m & (captured_m - 1)


class Qwen3_5MTPPerformanceError(RuntimeError):
    pass


class _PackedGateUp:
    """Preserve each encoded projection and the existing BF16 activation boundary."""

    def __init__(self, operators, value, weights, output):
        self.output = output
        self.parts = tuple(torch.empty((value.shape[0], weight.shape[0]),
                                      device=output.device, dtype=output.dtype)
                           for weight in weights)
        joint = operators.bind_linear_group(value, weights, self.parts)
        self.bindings = ((joint,) if joint is not None else tuple(
            operators.bind_linear(value, weight, part)
            for weight, part in zip(weights, self.parts, strict=True)))

    def launch(self):
        receipts = tuple(receipt for binding in self.bindings for receipt in binding.launch())
        offset = 0
        for part in self.parts:
            self.output[:, offset:offset + part.shape[1]].copy_(part)
            offset += part.shape[1]
        return receipts


@dataclass(slots=True)
class Qwen3_5MTPArena:
    key: Tensor
    value: Tensor
    written: Tensor

    @classmethod
    def allocate(
        cls, *, pages: int, block_size: int, heads: int, head_dim: int, device: torch.device
    ) -> Qwen3_5MTPArena:
        shape = (pages, 1, block_size, heads, head_dim)
        return cls(
            torch.zeros(shape, dtype=torch.bfloat16, device=device),
            torch.zeros(shape, dtype=torch.bfloat16, device=device),
            torch.zeros((pages, 1, block_size), dtype=torch.bool, device=device),
        )


@dataclass(slots=True)
class _MTPBuffers:
    token_ids: Tensor
    position_ids: Tensor
    page_destination: Tensor
    token_offsets: Tensor
    page_table: Tensor
    sequence_lengths: Tensor
    target_hidden: Tensor
    hidden: Tensor
    embedding: Tensor
    embedding_norm: Tensor
    target_norm: Tensor
    joined: Tensor
    fc_local: Tensor
    norm: Tensor
    update: Tensor
    after_attention: Tensor
    packed_mlp: Tensor
    mlp_active: Tensor
    attention_projected: Tensor
    query: Tensor
    attention_gate: Tensor
    key: Tensor
    value: Tensor
    normalized_query: Tensor
    normalized_key: Tensor
    rotated_query: Tensor
    rotated_key: Tensor
    attention_output: Tensor
    attention_partial: Tensor
    attention_statistics: Tensor
    attention_arrivals: Tensor
    cosine: Tensor
    sine: Tensor
    logits: Tensor
    router_logits: Tensor
    routed: Tensor
    shared_gate: Tensor


class Qwen3_5TPMTPBody:
    """One fixed-capacity MTP layer with TP collectives inside capture."""

    def __init__(
        self,
        *,
        config: Qwen3_5Config | Qwen3_5MoeConfig,
        parallel_plan: Qwen3_5ParallelWeightPlan | Qwen3_5MoeParallelPlan,
        weights: Qwen3_5RankWeights | Qwen3_5MoeRankWeights,
        tp_group: ProcessGroup,
        captured_m: int,
        table_pages: int,
        page_capacity: int,
        max_proposal_tokens: int,
        block_size: int = 16,
        arena: Qwen3_5MTPArena | None = None,
        packed_gate_up: Tensor | None = None,
        packed_experts: tuple[Tensor, Tensor] | None = None,
        mrope_tables: tuple[Tensor, Tensor, MRoPETableEvidence] | None = None,
        packed_operators: PackedLinearOperators | None = None,
        embedding_features: Tensor | None = None,
        cache_metadata: tuple[Tensor, Tensor, Tensor] | None = None,
        shared_scratch: dict[tuple, Tensor] | None = None,
    ) -> None:
        topology = parallel_plan.topology
        text = config.graph.text
        if packed_operators is not None and text.moe is not None:
            raise Qwen3_5MTPPerformanceError("packed routed MTP block has no qualified binding")
        self._packed_operators = packed_operators
        self._embedding_features = embedding_features
        if (
            not dist.is_initialized()
            or not parallel_plan.mtp_enabled
            or topology.pp_size != 1
            or topology.tp_size % topology.ep_size
            or topology.tp_size != topology.world_size
            or topology.tp_size < 1
            or weights.rank != dist.get_rank()
            or weights.device.type != "cuda"
            or type(max_proposal_tokens) is not int
            or max_proposal_tokens < 1
            or not _mtp_graph_capacity_supported(captured_m, max_proposal_tokens)
            or min(table_pages, page_capacity, block_size) <= 0
        ):
            raise Qwen3_5MTPPerformanceError("MTP body differs from the TP graph ABI")
        rank = weights.rank
        placement = topology.placements[rank]
        if tuple(dist.get_process_group_ranks(tp_group)) != tuple(range(topology.world_size)):
            raise Qwen3_5MTPPerformanceError("MTP TP group differs from rank placement")

        attention = text.full_attention
        local_q_heads = attention.query_heads // topology.tp_size
        local_kv_heads = max(1, attention.key_value_heads // topology.tp_size)
        moe = text.moe
        intermediate = (
            text.dense_intermediate_size
            if moe is None
            else moe.shared_expert_intermediate_size
        )
        if intermediate is None or intermediate % topology.tp_size:
            raise Qwen3_5MTPPerformanceError("MTP intermediate does not divide TP")
        local_intermediate = intermediate // topology.tp_size
        hidden, head_dim = text.hidden_size, attention.head_dim
        local_q_width = local_q_heads * head_dim
        rotary = int(head_dim * text.mrope.partial_rotary_factor)
        device = weights.device

        from cubic_serve.executor.text_graph import TextGraphBindings, full_attention_partitions

        attention_splits = full_attention_partitions(
            captured_m, max_tokens=text.max_position_embeddings,
            query_heads=local_q_heads, kv_heads=local_kv_heads, head_dim=head_dim,
            device=device, shared_scratch=shared_scratch,
        )

        self._activation_scratch = TextGraphBindings(
            packed_operators, device, torch.bfloat16, shared_scratch=shared_scratch,
        )

        def empty(*shape: int, role: str | None = None) -> Tensor:
            if role is not None:
                return self._activation_scratch.buffer("mtp." + role, *shape)
            return torch.empty(shape, dtype=torch.bfloat16, device=device)

        if arena is None:
            arena = Qwen3_5MTPArena.allocate(
                pages=page_capacity,
                block_size=block_size,
                heads=local_kv_heads,
                head_dim=head_dim,
                device=device,
            )
        elif cache_metadata is None and (
            arena.key.shape
            != (page_capacity, 1, block_size, local_kv_heads, head_dim)
            or arena.value.shape != arena.key.shape
            or arena.written.shape != (page_capacity, 1, block_size)
            or arena.key.dtype is not torch.bfloat16
            or arena.value.dtype is not torch.bfloat16
            or arena.written.dtype is not torch.bool
            or arena.key.device != device
        ):
            raise Qwen3_5MTPPerformanceError("shared MTP arena differs from the graph ABI")
        if mrope_tables is None:
            cosine_table, sine_table, table_evidence = build_bf16_mrope_tables(
                max_position=text.max_position_embeddings,
                device=device,
                rotary_dim=rotary,
                section=text.mrope.section,
                rope_theta=text.mrope.rope_theta,
            )
        else:
            cosine_table, sine_table, table_evidence = mrope_tables
        b = _MTPBuffers(
            token_ids=torch.zeros(captured_m, dtype=torch.int64, device=device),
            position_ids=torch.zeros((3, captured_m), dtype=torch.int64, device=device),
            page_destination=torch.zeros(captured_m, dtype=torch.int32, device=device),
            token_offsets=torch.zeros(captured_m, dtype=torch.int32, device=device),
            page_table=torch.zeros((captured_m, table_pages), dtype=torch.int32, device=device),
            sequence_lengths=torch.ones(captured_m, dtype=torch.int32, device=device),
            target_hidden=empty(captured_m, hidden),
            hidden=empty(captured_m, hidden),
            embedding=empty(captured_m, hidden, role="embedding"),
            embedding_norm=empty(captured_m, hidden, role="embedding_norm"),
            target_norm=empty(captured_m, hidden, role="target_norm"),
            joined=empty(captured_m, 2 * hidden, role="joined"),
            fc_local=empty(captured_m, hidden // topology.tp_size, role="fc_local"),
            norm=empty(captured_m, hidden),
            update=empty(captured_m, hidden, role="update"),
            after_attention=empty(captured_m, hidden, role="after_attention"),
            packed_mlp=empty(captured_m, 2 * local_intermediate, role="packed_mlp"),
            mlp_active=empty(captured_m, local_intermediate, role="mlp_active"),
            attention_projected=empty(captured_m, local_q_heads, 2, head_dim, role="attention_projected"),
            query=empty(captured_m, local_q_heads, head_dim, role="query"),
            attention_gate=empty(captured_m, local_q_heads, head_dim, role="attention_gate"),
            key=empty(captured_m, local_kv_heads, head_dim, role="key"),
            value=empty(captured_m, local_kv_heads, head_dim, role="value"),
            normalized_query=empty(captured_m, local_q_heads, head_dim, role="normalized_query"),
            normalized_key=empty(captured_m, local_kv_heads, head_dim, role="normalized_key"),
            rotated_query=empty(captured_m, local_q_heads, head_dim, role="rotated_query"),
            rotated_key=empty(captured_m, local_kv_heads, head_dim, role="rotated_key"),
            attention_output=empty(captured_m, local_q_heads, head_dim, role="attention_output"),
            attention_partial=self._activation_scratch.buffer(
                "attention.partial", captured_m, local_q_heads,
                attention_splits, head_dim, dtype=torch.float32, serial=True,
            ),
            attention_statistics=self._activation_scratch.buffer(
                "attention.statistics", captured_m, local_q_heads,
                attention_splits, 2, dtype=torch.float32, serial=True,
            ),
            attention_arrivals=torch.zeros(
                (captured_m, local_q_heads), dtype=torch.int32, device=device
            ),
            cosine=empty(captured_m, rotary, role="cosine"),
            sine=empty(captured_m, rotary, role="sine"),
            # Alignment executes the state-only phase. Terminal head logits
            # are consumed within the same exclusive topology lease.
            logits=empty(captured_m, text.vocabulary_size // topology.tp_size, role="sample_logits"),
            router_logits=empty(captured_m, 1 if moe is None else moe.expert_count, role="router_logits"),
            routed=empty(captured_m, hidden, role="routed"),
            shared_gate=empty(captured_m, 1, role="shared_gate"),
        )
        by_name = (dict(weights.named_tensors()) if packed_operators is None else
                   {binding.tensor_name: weights[binding.tensor_name]
                    for binding in weights.bindings})
        prefix = "mtp.layers.0"
        expected_gate_up = (2 * local_intermediate, hidden)
        shared_prefix = f"{prefix}.mlp" + ("" if moe is None else ".shared_expert")
        gate_up = packed_gate_up
        if gate_up is None and packed_operators is not None:
            gate_up = (by_name[f"{shared_prefix}.gate_proj.weight"],
                       by_name[f"{shared_prefix}.up_proj.weight"])
        elif gate_up is None:
            gate_up = torch.cat(
                (
                    by_name[f"{shared_prefix}.gate_proj.weight"],
                    by_name[f"{shared_prefix}.up_proj.weight"],
                ),
                dim=0,
            ).contiguous()
        if packed_operators is None and (
            gate_up.shape != expected_gate_up
            or gate_up.dtype is not torch.bfloat16
            or gate_up.device != device
            or not gate_up.is_contiguous()
        ):
            raise Qwen3_5MTPPerformanceError("shared MTP gate/up differs from the graph ABI")
        if packed_operators is not None and (
            not isinstance(gate_up, tuple) or len(gate_up) != 2
            or any(weight.shape != (local_intermediate, hidden) for weight in gate_up)
        ):
            raise Qwen3_5MTPPerformanceError("encoded MTP gate/up differs from declared geometry")
        embedding_span = self._span(parallel_plan, rank, "model.language_model.embed_tokens.weight")
        vocab_span = self._span(parallel_plan, rank, "lm_head.weight")
        self.rank = rank
        self.captured_m = captured_m
        self.vocab_span = vocab_span
        self.embedding_span = embedding_span
        self.arena = arena
        self.buffers = b
        self._weights = by_name
        self._tp_group = tp_group
        self._tp_size = topology.tp_size
        self._fc_span = AxisSpan(
            placement.tensor_rank * (hidden // topology.tp_size),
            (placement.tensor_rank + 1) * (hidden // topology.tp_size),
        )
        self._active_rows = torch.full(
            (1,), captured_m, dtype=torch.int32, device=device
        )
        # Proposals share the ordinary projection selector, while their serial
        # execution cost remains independent of the target's overlap scope.
        self._linear_cost = LinearLaunchCost(device_resources(device), 0.0, ())
        self._gate_up = gate_up
        self._moe = moe
        self.mrope_tables = (cosine_table, sine_table, table_evidence)
        self._embedding_mask = empty(captured_m, 1)
        self._cosine_table, self._sine_table = cosine_table, sine_table

        if packed_operators is None:
            self._embedding = BF16TokenEmbeddingPrimitive(
                b.token_ids, by_name["model.language_model.embed_tokens.weight"], b.embedding
            )
        else:
            from cubic_serve.ops.cuda.linear.quantized import PackedProjectionDispatch

            self._embedding = PackedProjectionDispatch(packed_operators, operation="embedding")
            self._embedding.prepare(
                b.token_ids, by_name["model.language_model.embed_tokens.weight"], b.embedding)
            self._embedding.seal()
        self._embedding_norm = ZeroCenteredRMSNormPrimitive(
            b.embedding,
            by_name["mtp.pre_fc_norm_embedding.weight"],
            b.embedding_norm,
            epsilon=text.rms_norm_epsilon,
        )
        self._target_norm = ZeroCenteredRMSNormPrimitive(
            b.target_hidden,
            by_name["mtp.pre_fc_norm_hidden.weight"],
            b.target_norm,
            epsilon=text.rms_norm_epsilon,
        )
        self._fc = self._linear(b.joined, by_name["mtp.fc.weight"], b.fc_local)
        self._layer_norm = ZeroCenteredRMSNormPrimitive(
            b.hidden,
            by_name[f"{prefix}.input_layernorm.weight"],
            b.norm,
            epsilon=text.rms_norm_epsilon,
        )
        self._q = self._linear(
            b.norm,
            by_name[f"{prefix}.self_attn.q_proj.weight"],
            b.attention_projected.reshape(captured_m, -1),
        )
        self._kv = self._linear(
            b.norm,
            by_name[f"{prefix}.self_attn.k_proj.weight"],
            b.key.reshape(captured_m, -1),
        )
        self._v = self._linear(b.norm, by_name[f"{prefix}.self_attn.v_proj.weight"],
                               b.value.reshape(captured_m, -1))
        self._q_norm = ZeroCenteredRMSNormPrimitive(
            b.query.reshape(-1, head_dim),
            by_name[f"{prefix}.self_attn.q_norm.weight"],
            b.normalized_query.reshape(-1, head_dim),
            epsilon=text.rms_norm_epsilon,
        )
        self._k_norm = ZeroCenteredRMSNormPrimitive(
            b.key.reshape(-1, head_dim),
            by_name[f"{prefix}.self_attn.k_norm.weight"],
            b.normalized_key.reshape(-1, head_dim),
            epsilon=text.rms_norm_epsilon,
        )
        self._mrope_gather = MRoPECosSinGatherPrimitive(
            b.position_ids,
            cosine_table,
            sine_table,
            b.cosine,
            b.sine,
            evidence=table_evidence,
        )
        self._mrope_apply = QwenMRoPEApplyPrimitive(
            b.normalized_query,
            b.normalized_key,
            b.cosine,
            b.sine,
            b.rotated_query,
            b.rotated_key,
        )
        self._q_gate = QwenAttentionQGateSplitPrimitive(
            b.attention_projected, b.query, b.attention_gate
        )
        self._cache_binding = None
        cache_encoding, cache_window = None, None
        if cache_metadata is not None:
            from cubic_serve.ops.cuda.cache_quantization import PagedKVGraphBinding
            segments, activity, owner_slots = cache_metadata
            self._cache_binding = PagedKVGraphBinding(
                arena, positions=b.sequence_lengths, positions_are_lengths=True,
                segments=segments, activity=activity, source_slots=owner_slots[0, 0],
                destination_slots=owner_slots[1, 0], owner_stride=owner_slots.shape[1],
            )
            cache_encoding = self._cache_binding.encoding
            cache_window, self._page_write, self._cache_checkpoint = (
                self._cache_binding.bind_layer(b.rotated_key, b.value,
                                              layer=0, pages=b.page_destination,
                                              query=b.rotated_query))
        else:
            self._page_write = PageTokenWritePrimitive(
                arena.key,
                arena.value,
                arena.written,
                b.page_destination,
                b.token_offsets,
                b.rotated_key,
                b.value,
                layer=0,
            )
        self._attention_query = (b.rotated_query if self._cache_binding is None else
                                 self._cache_binding.query_for(b.rotated_query))
        self._attention = BF16PagedGQADecodePrimitive(
            self._attention_query,
            arena.key,
            arena.value,
            b.page_table,
            b.sequence_lengths,
            b.attention_output,
            layer=0,
            partial=b.attention_partial,
            statistics=b.attention_statistics,
            arrivals=b.attention_arrivals,
            cache_encoding=cache_encoding,
            cache_window=cache_window,
        )
        self._attention_gate = BF16SigmoidOutputGatePrimitive(
            b.attention_output, b.attention_gate, b.rotated_query
        )
        self._o = self._linear(
            b.rotated_query.reshape(captured_m, local_q_width),
            by_name[f"{prefix}.self_attn.o_proj.weight"],
            b.update,
        )
        self._residual = BF16ResidualAddPrimitive(b.hidden, b.update, b.after_attention)
        self._post_norm = ZeroCenteredRMSNormPrimitive(
            b.after_attention,
            by_name[f"{prefix}.post_attention_layernorm.weight"],
            b.norm,
            epsilon=text.rms_norm_epsilon,
        )
        self._gate_up_linear = self._linear(b.norm, gate_up, b.packed_mlp)
        self._swiglu = BF16PackedSwiGLUPrimitive(b.packed_mlp, b.mlp_active)
        self._down = self._linear(
            b.mlp_active, by_name[f"{shared_prefix}.down_proj.weight"], b.update
        )
        if moe is not None:
            first_expert = topology.expert_coordinates(rank)[0] * (
                moe.expert_count // topology.ep_size
            )
            local_experts = moe.expert_count // topology.ep_size
            expert_intermediate = moe.expert_intermediate_size // (
                topology.tp_size // topology.ep_size
            )
            if packed_experts is None:
                expert_gate_up = torch.empty(
                    (local_experts, 2 * expert_intermediate, hidden),
                    dtype=torch.bfloat16,
                    device=device,
                )
                expert_down = torch.empty(
                    (local_experts, hidden, expert_intermediate),
                    dtype=torch.bfloat16,
                    device=device,
                )
                for local, expert in enumerate(
                    range(first_expert, first_expert + local_experts)
                ):
                    expert_gate_up[local, : expert_intermediate].copy_(
                        by_name[f"{prefix}.mlp.experts.{expert}.gate_proj.weight"]
                    )
                    expert_gate_up[local, expert_intermediate :].copy_(
                        by_name[f"{prefix}.mlp.experts.{expert}.up_proj.weight"]
                    )
                    expert_down[local].copy_(
                        by_name[f"{prefix}.mlp.experts.{expert}.down_proj.weight"]
                    )
            else:
                expert_gate_up, expert_down = packed_experts
            if (
                expert_gate_up.shape
                != (local_experts, 2 * expert_intermediate, hidden)
                or expert_down.shape
                != (local_experts, hidden, expert_intermediate)
                or expert_gate_up.dtype is not torch.bfloat16
                or expert_down.dtype is not torch.bfloat16
                or expert_gate_up.device != device
                or expert_down.device != device
                or not expert_gate_up.is_contiguous()
                or not expert_down.is_contiguous()
            ):
                raise Qwen3_5MTPPerformanceError(
                    "shared MTP expert pack differs from the graph ABI"
                )
            self._expert_gate_up, self._expert_down = expert_gate_up, expert_down
            self._router = self._linear(
                b.norm, by_name[f"{prefix}.mlp.gate.weight"], b.router_logits
            )
            self._shared_gate = self._linear(
                b.norm, by_name[f"{prefix}.mlp.shared_expert_gate.weight"], b.shared_gate
            )
            self._routed_experts = BF16RoutedExpertsPrimitive(
                b.norm,
                b.router_logits,
                b.sequence_lengths,
                self._active_rows,
                expert_gate_up,
                expert_down,
                b.routed,
                first_expert=first_expert,
                top_k=moe.experts_per_token,
            )
            self._moe_combine = BF16MoeSharedCombinePrimitive(
                b.update, b.shared_gate, b.routed, b.update
            )
            self._moe_branches = MoeBranchStreams(device, 1)
        self._final_residual = BF16ResidualAddPrimitive(
            b.after_attention, b.update, b.hidden
        )
        self._final_norm = ZeroCenteredRMSNormPrimitive(
            b.hidden,
            by_name["mtp.norm.weight"],
            b.norm,
            epsilon=text.rms_norm_epsilon,
        )
        self._lm_head = self._linear(b.norm, by_name["lm_head.weight"], b.logits)
        self._tp_fixed_sum = FixedRankBF16SumPrimitive(b.hidden, self._tp_group)

    @staticmethod
    def _span(
        plan: Qwen3_5ParallelWeightPlan | Qwen3_5MoeParallelPlan,
        rank: int,
        name: str,
    ) -> AxisSpan:
        matches = tuple(
            item for item in plan.bindings_for_rank(rank) if item.tensor_name == name
        )
        if len(matches) != 1 or len(matches[0].spans) != 1:
            raise Qwen3_5MTPPerformanceError(f"MTP rank has no single {name} span")
        return matches[0].spans[0]

    def _linear(self, value: Tensor, weight, output: Tensor):
        if self._packed_operators is not None:
            if isinstance(weight, tuple):
                return _PackedGateUp(self._packed_operators, value, weight, output)
            return self._packed_operators.bind_linear(value, weight, output)
        with linear_cost_scope(self._linear_cost):
            return bind_bf16_linear_kernel(
                value, weight, output, self._active_rows, reduction="fixed_tile"
            )

    def _launch_linear(
        self,
        primitive: BF16LinearKernel,
        value: Tensor,
        weight: Tensor,
        output: Tensor,
    ) -> None:
        if self._packed_operators is not None:
            primitive.launch()
            return
        launch_bf16_linear_kernel(primitive, value, weight, output, self._active_rows)

    def _tp_sum(self, value: Tensor) -> None:
        self._tp_fixed_sum.launch(value)

    def launch_state(self, observer: Qwen3_5MTPObserver | None = None) -> None:
        """Advance the MTP layer and its KV state without producing logits."""

        b, w = self.buffers, self._weights
        if self._moe is not None:
            self._moe_branches.begin()
        prefix = "mtp.layers.0"
        self._embedding.launch(
            b.token_ids, w["model.language_model.embed_tokens.weight"], b.embedding
        )
        torch.mul(b.embedding, self._embedding_mask, out=b.embedding)
        self._tp_sum(b.embedding)
        if self._embedding_features is not None:
            torch.add(b.embedding, self._embedding_features, out=b.embedding)
        self._embedding_norm.launch(
            b.embedding, w["mtp.pre_fc_norm_embedding.weight"], b.embedding_norm
        )
        self._target_norm.launch(
            b.target_hidden, w["mtp.pre_fc_norm_hidden.weight"], b.target_norm
        )
        b.joined[:, : b.embedding_norm.shape[1]].copy_(b.embedding_norm)
        b.joined[:, b.embedding_norm.shape[1] :].copy_(b.target_norm)
        if observer is not None:
            observer("fc_input", b.joined)
        self._launch_linear(self._fc, b.joined, w["mtp.fc.weight"], b.fc_local)
        b.hidden.zero_()
        b.hidden[:, self._fc_span.start : self._fc_span.stop].copy_(b.fc_local)
        self._tp_sum(b.hidden)
        if observer is not None:
            observer("fc", b.hidden)
        self._mrope_gather.launch(
            b.position_ids, self._cosine_table, self._sine_table, b.cosine, b.sine
        )
        self._layer_norm.launch(
            b.hidden, w[f"{prefix}.input_layernorm.weight"], b.norm
        )
        if observer is not None:
            observer("attention_input", b.norm)
        self._launch_linear(
            self._q,
            b.norm,
            w[f"{prefix}.self_attn.q_proj.weight"],
            b.attention_projected.reshape(self.captured_m, -1),
        )
        self._q_gate.launch(b.attention_projected, b.query, b.attention_gate)
        self._launch_linear(
            self._kv,
            b.norm,
            w[f"{prefix}.self_attn.k_proj.weight"],
            b.key.reshape(self.captured_m, -1),
        )
        self._launch_linear(
            self._v,
            b.norm,
            w[f"{prefix}.self_attn.v_proj.weight"],
            b.value.reshape(self.captured_m, -1),
        )
        if observer is not None:
            observer("q_projection", b.attention_projected)
            observer("k_projection", b.key)
            observer("v_projection", b.value)
        self._q_norm.launch(
            b.query.reshape(-1, b.query.shape[-1]),
            w[f"{prefix}.self_attn.q_norm.weight"],
            b.normalized_query.reshape(-1, b.query.shape[-1]),
        )
        self._k_norm.launch(
            b.key.reshape(-1, b.key.shape[-1]),
            w[f"{prefix}.self_attn.k_norm.weight"],
            b.normalized_key.reshape(-1, b.key.shape[-1]),
        )
        self._mrope_apply.launch(
            b.normalized_query,
            b.normalized_key,
            b.cosine,
            b.sine,
            b.rotated_query,
            b.rotated_key,
        )
        if self._cache_binding is not None:
            self._cache_binding.prepare_metadata()
            self._page_write.launch()
        else:
            self._page_write.launch(
                self.arena.key,
                self.arena.value,
                self.arena.written,
                b.page_destination,
                b.token_offsets,
                b.rotated_key,
                b.value,
            )
        self._attention.launch(
            self._attention_query,
            self.arena.key,
            self.arena.value,
            b.page_table,
            b.sequence_lengths,
            b.attention_output,
        )
        if self._cache_binding is not None:
            self._cache_checkpoint.launch()
        self._attention_gate.launch(b.attention_output, b.attention_gate, b.rotated_query)
        self._launch_linear(
            self._o,
            b.rotated_query.reshape(self.captured_m, -1),
            w[f"{prefix}.self_attn.o_proj.weight"],
            b.update,
        )
        self._tp_sum(b.update)
        self._residual.launch(b.hidden, b.update, b.after_attention)
        self._post_norm.launch(
            b.after_attention, w[f"{prefix}.post_attention_layernorm.weight"], b.norm
        )
        if observer is not None:
            observer("mlp_input", b.norm)
        if self._moe is not None:
            with self._moe_branches.fork_shared():
                self._launch_linear(
                    self._gate_up_linear, b.norm, self._gate_up, b.packed_mlp
                )
                self._swiglu.launch(b.packed_mlp, b.mlp_active)
                self._launch_linear(
                    self._down,
                    b.mlp_active,
                    w[f"{prefix}.mlp.shared_expert.down_proj.weight"],
                    b.update,
                )
                self._launch_linear(
                    self._shared_gate,
                    b.norm,
                    w[f"{prefix}.mlp.shared_expert_gate.weight"],
                    b.shared_gate,
                )
            self._launch_linear(
                self._router,
                b.norm,
                w[f"{prefix}.mlp.gate.weight"],
                b.router_logits,
            )
            self._routed_experts.launch(
                b.norm,
                b.router_logits,
                b.sequence_lengths,
                self._active_rows,
                self._expert_gate_up,
                self._expert_down,
                b.routed,
            )
            if observer is not None:
                observer("route", self._routed_experts.route_ids)
            self._moe_branches.join_shared()
            self._moe_combine.launch(b.update, b.shared_gate, b.routed, b.update)
        else:
            self._launch_linear(
                self._gate_up_linear, b.norm, self._gate_up, b.packed_mlp
            )
            self._swiglu.launch(b.packed_mlp, b.mlp_active)
            self._launch_linear(
                self._down,
                b.mlp_active,
                w[f"{prefix}.mlp.down_proj.weight"],
                b.update,
            )
        self._tp_sum(b.update)
        self._final_residual.launch(b.after_attention, b.update, b.hidden)
        if observer is not None:
            observer("mlp_output", b.hidden)

    def launch_logits(self, observer: Qwen3_5MTPObserver | None = None) -> None:
        """Project the already-advanced MTP hidden state to vocabulary logits."""

        self.normalize_feedback(observer)
        self.launch_projection()

    def launch_projection(self) -> None:
        """Consume the already-rounded head input without advancing producers."""
        b, w = self.buffers, self._weights
        self._launch_linear(self._lm_head, b.norm, w["lm_head.weight"], b.logits)

    def normalize_feedback(self, observer: Qwen3_5MTPObserver | None = None) -> None:
        """Preserve the provider's normalized feedback boundary without a head."""

        b, w = self.buffers, self._weights
        self._final_norm.launch(b.hidden, w["mtp.norm.weight"], b.norm)
        if observer is not None:
            observer("mtp_output", b.norm)

    def launch(self, observer: Qwen3_5MTPObserver | None = None) -> None:
        self.launch_state(observer)
        self.launch_logits(observer)

    def bind_token_ids(self, token_ids: Tensor) -> None:
        span = self.embedding_span
        owned = (token_ids >= span.start) & (token_ids < span.stop)
        self.buffers.token_ids.copy_(torch.where(owned, token_ids - span.start, 0))
        self._embedding_mask.copy_(owned[:, None])


@dataclass(slots=True)
class Qwen3_5TPMTPGraphSlot:
    body: Qwen3_5TPMTPBody
    stream: Stream
    graph: CUDAGraph
    state_graph: CUDAGraph

    @classmethod
    def capture(cls, body: Qwen3_5TPMTPBody) -> Qwen3_5TPMTPGraphSlot:
        device = body.buffers.hidden.device
        barrier()
        stream = Stream(device=device)  # type: ignore[no-untyped-call]
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            body.launch()
            body.launch()
        stream.synchronize()
        barrier()
        graph = CUDAGraph()
        with capture_graph(graph, stream=stream):
            body.launch()
        stream.synchronize()
        barrier()
        state_graph = CUDAGraph()
        with capture_graph(state_graph, stream=stream):
            body.launch_state()
        stream.synchronize()
        barrier()
        return cls(body, stream, graph, state_graph)

    @property
    def logits(self) -> Tensor:
        return self.body.buffers.logits

    @property
    def hidden(self) -> Tensor:
        return self.body.buffers.norm

    @property
    def vocab_span(self) -> AxisSpan:
        return self.body.vocab_span

    def replay(self) -> None:
        """Replay on the caller-selected current stream."""

        self.graph.replay()

    def replay_state(self) -> None:
        """Replay only the state/KV advance needed after an accepted draft."""

        self.state_graph.replay()

    def synchronize(self) -> None:
        self.stream.synchronize()


class PackedMTPStage:
    """Bind the existing dense MTP math to common packed operators and leases.

    The provider's only persistent state is its independently owned KV cache.
    Explicit row maps still name feedback/checkpoint ownership; an empty
    recurrent schema never makes a continuation or padding a request owner.
    """

    typed_state_boundary = True

    def __init__(self, config, weights, topology, *, max_depth, captured_m,
                 tp_group=None, pp_group=None, ep_group=None, expert_tp_group=None,
                 table_pages=1, page_size=16, arena=None, state_slots=None,
                 operator_config=None, mrope_tables=None, resource=None,
                 route_completion_costs=None, cache_storage=None, shared_scratch=None):
        from types import SimpleNamespace

        from cubic_serve.contracts.execution import RankReplayResources
        from cubic_serve.executor.quantized_model_runner import _RankOperators
        from cubic_serve.executor.text_serving import TextServingBuffers
        from cubic_serve.ops.cuda.attention.paged_decode import PageCOWCopyPrimitive
        from cubic_serve.runtime.cache import CacheArena, CacheArenaSpec, KVLayout

        text = config.graph.text
        if (topology.pp_size != 1 or text.moe is not None
                or text.mtp.hidden_layer_count != 1 or text.mtp.dedicated_embeddings
                or text.dtype != "bfloat16"):
            raise ValueError("packed MTP requires a declared single dense BF16 draft block")
        self.rank, self.captured_m = weights.rank, captured_m
        self.stage, self.tensor_rank = 0, topology.placements[self.rank].tensor_rank
        self.config, self.weights, self.topology = config, weights, topology
        self.first = self.last = True
        self.stream = Stream(device=weights.device)
        self._graph, self._closed = None, False
        self._captured_once = False
        self._state_graph = None
        scratch_id = resource.scratch_arena_id if resource is not None else "mtp.lane"
        layout = KVLayout((0,), page_size,
                          max(1, text.full_attention.key_value_heads // topology.tp_size),
                          text.full_attention.head_dim)
        if cache_storage is not None:
            layout = replace(layout, storage=cache_storage)
        states = layout.quantization_state_specs
        self.cache_storage = layout.storage
        if arena is None:
            owners = captured_m if state_slots is None else state_slots
            arena = CacheArena(CacheArenaSpec(
                f"mtp.rank{self.rank}", max(table_pages + owners, 2 * owners),
                owners, states, captured_m, (scratch_id,)), layout, weights.device,
                rank_id=self.rank)
        if (arena.layout != layout or arena.spec.state_specs != states or arena.rank_id != self.rank
                or arena.device != weights.device or arena.spec.scratch_capacity < captured_m
                or scratch_id not in arena.spec.scratch_arena_ids):
            raise ValueError("draft arena differs from declared KV ownership")
        self.arena = arena
        self.resource = resource or RankReplayResources(
            self.rank, f"mtp.graph{self.rank}.M{captured_m}",
            f"mtp.stream{self.rank}.M{captured_m}", f"mtp.buffer{self.rank}.M{captured_m}",
            f"mtp.workspace{self.rank}.M{captured_m}", scratch_id)
        self.feature_width, self.features_replicated = text.hidden_size, True
        self.feature_rows = torch.zeros((captured_m, text.hidden_size),
                                       device=weights.device, dtype=torch.bfloat16)
        plan = SimpleNamespace(topology=topology, mtp_enabled=True,
                               bindings_for_rank=lambda rank: weights.bindings)
        operators = _RankOperators(weights, tp_group, operator_config)
        def metadata(*shape):
            return torch.zeros(shape, device=weights.device, dtype=torch.int32)
        segments, activity, owner_slots = (metadata(captured_m), metadata(captured_m),
                                          metadata(2, 1, captured_m))
        self.body = Qwen3_5TPMTPBody(
            config=config, parallel_plan=plan, weights=weights,
            tp_group=tp_group if tp_group is not None else dist.group.WORLD,
            captured_m=captured_m, table_pages=table_pages, page_capacity=arena.key.shape[0],
            max_proposal_tokens=max_depth, block_size=page_size, arena=arena,
            packed_operators=operators, embedding_features=self.feature_rows,
            mrope_tables=mrope_tables, shared_scratch=shared_scratch,
            cache_metadata=(segments, activity, owner_slots) if layout.storage.quantized else None)
        b = self.body.buffers
        self.buffers = TextServingBuffers(
            b.token_ids, b.position_ids, metadata(captured_m), b.page_destination,
            b.token_offsets, b.page_table, b.sequence_lengths, segments,
            activity, (owner_slots,), b.norm, b.hidden,
            b.target_hidden)
        self.hidden, self.logits, self.draft_feedback = b.hidden, b.logits, b.norm
        self.embedding_span, self.vocab_span = self.body.embedding_span, self.body.vocab_span
        self.embedding_mask = self.body._embedding_mask
        self.mrope_tables = self.body.mrope_tables
        self._scratch_pages = torch.tensor(arena.padding_page_slots(captured_m, scratch_id),
                                          device=weights.device, dtype=torch.int32)
        self._scratch_states = torch.tensor(arena.padding_state_slots(captured_m, scratch_id),
                                            device=weights.device, dtype=torch.int32)
        from cubic_serve.ops.cuda.cache_quantization import paged_cache_encoding
        self._page_copy = PageCOWCopyPrimitive(
            arena.key, arena.value, arena.written, self.buffers.page_source, b.page_destination,
            page_fields=tuple(arena.page_fields.values()), cache_encoding=paged_cache_encoding(arena))
        self._overlap_resources = None

    def bind_scratch(self, resource):
        if resource != self.resource:
            raise ValueError("draft resources cannot be retargeted")
        b = self.buffers
        b.token_ids.zero_()
        b.position_ids.zero_()
        b.token_offsets.zero_()
        b.sequence_lengths.fill_(1)
        b.segment_lengths.zero_()
        b.row_activity.zero_()
        b.page_source.copy_(self._scratch_pages)
        b.page_destination.copy_(self._scratch_pages)
        b.page_table.copy_(self._scratch_pages[:, None])
        b.state_slot_maps[0].copy_(self._scratch_states[None, None])
        b.target_hidden.zero_()
        self.feature_rows.zero_()
        self.embedding_mask.zero_()

    def prepare_scratch(self, resource=None):
        self.bind_scratch(resource or self.resource)
        self.arena.reset_page_rows(self._scratch_pages.long())

    def fill_token_ids(self, tokens):
        self.body.bind_token_ids(tokens)

    def set_active_rows(self, logical_m):
        if not 0 < logical_m <= self.captured_m:
            raise ValueError("draft logical rows exceed captured capacity")
        # Native packed projections execute the physical bucket. Inactive rows
        # write only private pages; activity is independent of segmentation.
        self.body._active_rows.fill_(logical_m)

    def launch(self, *, produce_logits=True):
        if self._closed:
            raise ValueError("draft stage is closed")
        b = self.buffers
        self._page_copy.launch(self.arena.key, self.arena.value, self.arena.written,
                               b.page_source, b.page_destination)
        self.body.launch_state()
        if produce_logits:
            self.body.launch_logits()
        else:
            self.body.normalize_feedback()

    def launch_state(self):
        """Shared state-only phase without host input binding or sampling."""
        self.launch(produce_logits=False)

    def launch_projection(self):
        """Shared draft head capability; feedback normalization is already done."""
        if self._closed:
            raise ValueError("draft stage is closed")
        self.body.launch_projection()

    @property
    def graph(self):
        if self._graph is None or self._closed:
            raise ValueError("draft has no live startup-captured graph")
        return self._graph

    def capture(self):
        if self._graph is not None or self._closed:
            raise ValueError("draft recapture is forbidden")
        self.prepare_scratch()
        self.stream.wait_stream(torch.cuda.current_stream(self.arena.device))
        for _ in range(2):
            with torch.cuda.stream(self.stream):
                self.launch()
            self.stream.synchronize()
            barrier()
        graph = CUDAGraph()
        with capture_graph(graph, stream=self.stream):
            self.launch()
        self.stream.synchronize()
        barrier()
        self._graph = graph
        state_graph = CUDAGraph()
        with capture_graph(state_graph, stream=self.stream):
            self.launch(produce_logits=False)
        self.stream.synchronize()
        barrier()
        self._state_graph = state_graph
        self._captured_once = True

    def replay(self):
        if self._graph is None or self._closed:
            raise ValueError("draft has no live sealed graph")
        with torch.cuda.stream(self.stream):
            self._graph.replay()

    def replay_state(self):
        if self._state_graph is None or self._closed:
            raise ValueError("draft has no live sealed feedback graph")
        with torch.cuda.stream(self.stream):
            self._state_graph.replay()

    def select_decode_overlap(self, independent_rows):
        if type(independent_rows) is not bool:
            raise TypeError("row ownership classification must be boolean")

    def synchronize(self):
        self.stream.synchronize()

    def close(self):
        self.synchronize()
        if self._graph is not None:
            self._graph.reset()
            self._graph = None
        if self._state_graph is not None:
            self._state_graph.reset()
            self._state_graph = None
        self._closed = True


__all__ = (
    "Qwen3_5MTPArena",
    "Qwen3_5MTPObserver",
    "Qwen3_5MTPPerformanceError",
    "Qwen3_5TPMTPBody",
    "Qwen3_5TPMTPGraphSlot",
)


class _MTPWeightView:
    """Lossless semantic view over the already admitted checkpoint owners.

    Draft reuses the existing HC/QSA/MoE traversal and cache schema. This view
    translates only checkpoint names; encoded payload and rank ownership stay
    in the original QuantizedMatrix and LogicalWeightSlice objects.
    """

    def __init__(self, weights: Any) -> None:
        self.original = weights
        for name in ("rank", "device", "prepared_weights", "materialized_weights"):
            setattr(self, name, getattr(weights, name))
        self.bindings = tuple(
            replace(binding, tensor_name=self.to_traversal(binding.tensor_name))
            for binding in weights.bindings
            if binding.tensor_name.startswith("mtp.")
            or binding.tensor_name in ("model.language_model.embed_tokens.weight", "lm_head.weight")
        )

    @staticmethod
    def to_traversal(name: str) -> str:
        if name.startswith("mtp.layers."):
            return "model.language_model.layers." + name.removeprefix("mtp.layers.")
        if name.startswith("mtp.hyper_connection_mixer."):
            return "model.language_model." + name.removeprefix("mtp.")
        return name

    def __getitem__(self, name: str):
        if name.startswith("model.language_model.layers."):
            name = "mtp.layers." + name.removeprefix("model.language_model.layers.")
        elif name.startswith("model.language_model.hyper_connection_mixer."):
            name = "mtp." + name.removeprefix("model.language_model.")
        return self.original[name]


def build_flash_mtp_stage(config, weights, topology, *, max_depth: int, **options):
    """Bind model-owned input math to shared format/cache/graph operators."""
    from cubic_serve.executor.draft import DraftSpec
    from cubic_serve.executor.text_serving import FlashNextDecodeStage
    from cubic_serve.models.qwen4_exp.model import validate_mtp_metadata
    from cubic_serve.ops.cuda.collective import FixedRankBF16SumPrimitive, barrier
    from cubic_serve.ops.cuda.pointwise.qwen_decode import ResidualAddPrimitive

    validate_mtp_metadata(config)
    if topology.pp_size != 1:
        raise ValueError("MTP pipeline composition is not qualified")
    t, rows = config.text, options["captured_m"]
    if t.dtype != "bfloat16":
        raise ValueError("MTP original-reference activation storage requires BF16")
    streams, width = t.hyperconnection.stream_count, t.hidden_size
    feedback = torch.zeros((rows, streams * width), dtype=torch.bfloat16, device=weights.device)
    view = _MTPWeightView(weights)
    draft_config = replace(config, text=replace(
        t, layer_count=1, layer_schedule=t.mtp.layer_schedule,
        ple=replace(t.ple, layer_ids=()),
    ))
    tp_group = options.get("tp_group")
    spans = {binding.tensor_name: binding.spans[0] for binding in weights.bindings
             if binding.tensor_name in ("mtp.fc_embedding.weight", "mtp.fc_hidden.weight")}

    def bind_input(helper, hidden):
        embedding = helper.buffer("draft.embedding", rows, width)
        norm_embedding = helper.buffer("draft.norm_embedding", rows, width)
        norm_hidden = helper.buffer("draft.norm_hidden", rows, streams * width)
        emb_weight, hid_weight = weights["mtp.fc_embedding.weight"], weights["mtp.fc_hidden.weight"]
        emb_local = helper.buffer("draft.embedding_local", rows, emb_weight.shape[0])
        hid_local = helper.buffer("draft.hidden_local", rows * streams, hid_weight.shape[0])
        emb_full = helper.buffer("draft.embedding_projected", rows, width)
        hid_full = helper.buffer("draft.hidden_projected", rows * streams, width)
        repeated = helper.buffer("draft.embedding_repeated", rows, streams * width)
        norm_e = helper.norm(embedding, weights["mtp.pre_fc_norm_embedding.weight"],
                             norm_embedding, epsilon=t.rms_norm_epsilon)
        norm_h = helper.norm(feedback, weights["mtp.pre_fc_norm_hidden.weight"],
                             norm_hidden, epsilon=t.rms_norm_epsilon)
        fc_e = helper.linear(norm_embedding, emb_weight, emb_local)
        fc_h = helper.linear(norm_hidden.view(rows * streams, width), hid_weight, hid_local)
        sums = (() if tp_group is None else (
            FixedRankBF16SumPrimitive(emb_full, tp_group),
            FixedRankBF16SumPrimitive(hid_full, tp_group),
        ))
        add = ResidualAddPrimitive(hid_full.view(rows, streams * width), repeated, hidden)
        # Unique column ownership is declared by the checkpoint slicing layer,
        # rather than inferred from a model name or a device generation.
        es, hs = spans["mtp.fc_embedding.weight"], spans["mtp.fc_hidden.weight"]
        if es.size != emb_local.shape[1] or hs.size != hid_local.shape[1]:
            raise ValueError("MTP input projection differs from original column ownership")

        def launch():
            embedding.copy_(hidden.view(rows, streams, width)[:, 0])
            norm_e()
            norm_h()
            fc_e()
            fc_h()
            emb_full.zero_()
            hid_full.zero_()
            emb_full[:, es.start:es.stop].copy_(emb_local)
            hid_full[:, hs.start:hs.stop].copy_(hid_local)
            if sums:
                sums[0].launch(emb_full)
                sums[1].launch(hid_full)
            repeated.view(rows, streams, width).copy_(emb_full[:, None])
            add.launch(hid_full.view(rows, streams * width), repeated, hidden)

        return launch

    stage = FlashNextDecodeStage(draft_config, view, topology,
                                input_transform=bind_input, **options)
    stage.buffers.target_hidden = feedback
    stage.draft_spec = DraftSpec("mtp", t.vocabulary_size, streams * width, max_depth)
    stage.original_model_config = config
    return stage
