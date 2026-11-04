"""Captured BF16 decode graph runner for injected ``model_type=qwen3_5`` semantics.

The body is deliberately model-specific while every launch primitive remains
reusable.  Runtime state is supplied by the shared physical cache arena; this
module owns only one slot's fixed buffers, stream, and CUDA Graph.
Rank-local TP/PP execution never reconstructs a global weight and owns no fallback path.

Graph boundary sources: llama.cpp 9723942a (``ggml-cuda.cu``), vLLM
7fbfca26 (``cudagraph_dispatcher.py``/``cudagraph_utils.py``), and SGLang 6afb5e17
(``decode_cuda_graph_runner.py``/``full_cuda_graph_backend.py``).
"""

from __future__ import annotations

from typing_extensions import TypeAliasType

import logging
import math
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from functools import partial as _defer_collective
from typing import cast

import torch
import torch.distributed as dist
from torch import Tensor
from torch.cuda.graphs import CUDAGraph
from torch.cuda.graphs import graph as capture_graph
from torch.cuda.streams import Stream
from torch.distributed import ProcessGroup

from cubic_serve.contracts import execution as ec
from cubic_serve.contracts.execution import RankReplayResources
from cubic_serve.contracts.model import ScalarDType
from cubic_serve.executor.cuda_graph import (
    CudaGraphRejected,
    GraphProgramSession,
    GraphProgramSlot,
    GraphRuntimeBinding,
    ReplayIntent,
)
from cubic_serve.executor.decode_overlap import (
    DecodeInterleave,
    DecodeOverlapPlan,
    compare_decode_graphs,
    decode_calibration_prepare,
    format_decode_measurements,
    measure_decode_graph,
)
from cubic_serve.guardrail import graph_padding as gp
from cubic_serve.models.parallel_weights import AxisSpan
from cubic_serve.models.qwen3_5.model import (
    CONV_STATE_ID,
    RECURRENT_STATE_ID,
    AttentionKind,
    Qwen3_5Config,
    Qwen3_5GraphVariant,
    Qwen3_5ReferenceModel,
    Qwen3_5TextConfig,
)
from cubic_serve.models.qwen3_5.weights import (
    Qwen3_5ParallelWeightPlan,
    Qwen3_5RankWeights,
)
from cubic_serve.models.qwen3_5_moe.model import Qwen3_5MoeStageModel
from cubic_serve.ops.cuda.attention import (
    BF16PagedGQADecodePrimitive,
    PageCOWCopyPrimitive,
    PageTokenWritePrimitive,
)
from cubic_serve.ops.cuda.attention.paged_decode import paged_gqa_kv_splits
from cubic_serve.ops.cuda.collective import FixedRankBF16SumPrimitive
from cubic_serve.ops.cuda.embedding import BF16TokenEmbeddingPrimitive
from cubic_serve.ops.cuda.launch_cost import (
    LinearLaunchCost,
    current_linear_cost,
    device_resources,
    device_work_cost,
    linear_cost_scope,
)
from cubic_serve.ops.cuda.linear import (
    BF16LinearKernel,
    BF16LinearPrimitive,
    bind_bf16_linear_kernel,
    launch_bf16_linear_kernel,
)
from cubic_serve.ops.cuda.linear.grouped import BF16GroupedLinearPrimitive, bind_grouped_linear
from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators, PackedProjectionDispatch
from cubic_serve.ops.cuda.linear_attention import (
    CausalConv1DDecodePrimitive,
    GatedDeltaDecodePrimitive,
)
from cubic_serve.ops.cuda.moe import (
    BF16MoeSharedCombinePrimitive,
    BF16RoutedExpertsPrimitive,
    MoeBranchStreams,
)
from cubic_serve.ops.cuda.moe.quantized import PackedRoutedExpertsPrimitive
from cubic_serve.ops.cuda.moe.route_completion import RouteCompletionPlanner
from cubic_serve.ops.cuda.normalization import (
    BF16ResidualRMSNormPrimitive,
    ZeroCenteredRMSNormPrimitive,
)
from cubic_serve.ops.cuda.pointwise import (
    BF16PackedGDNProjectionSplitPrimitive,
    BF16PackedSwiGLUPrimitive,
    BF16ResidualAddPrimitive,
    BF16SigmoidOutputGatePrimitive,
    BF16SwiGLUPrimitive,
    GatedRMSNormPrimitive,
    QwenAttentionQGateSplitPrimitive,
)
from cubic_serve.ops.cuda.position import (
    MRoPECosSinGatherPrimitive,
    MRoPETableEvidence,
    QwenMRoPEApplyPrimitive,
    build_bf16_mrope_tables,
)
from cubic_serve.ops.strategies import collective_parameters, gdn_parameters, launch_parameters
from cubic_serve.runtime.cache import CacheArena, CacheArenaSpec, KVLayout, StateSlotSpec
from cubic_serve.runtime.model_state import (
    STATE_PATH_ID,
    DecodeSession,
    DecodeState,
    QualificationFactory,
    QualificationRun,
)

GDN_CONV_STATE_ID = CONV_STATE_ID
GDN_RECURRENT_STATE_ID = RECURRENT_STATE_ID
PROGRAM_ID = "cubic.qwen3_5.dense.decode.cuda_graph.v1"
REPLAY_ABI_ID = "cubic.qwen3_5.dense.decode.v1"
SEMANTIC_OP = "graph_program.composite.qwen3_5.dense.decode"
CANDIDATE_ID = "qwen3_5.dense.decode.cuda_graph"
COMPOSITE_CHILD_CLOSURE_ID = "cubic.qwen3_5.dense.decode.children.v1"
ROW_MAP_SCHEMA_ID = "cubic.rows.active-or-private-scratch.v1"
SCRATCH_SCHEMA_ID = "cubic.qwen3_5.decode.scratch.v1"
GRAPH_SAFETY_CONTRACT_ID = "cubic.qwen3_5.decode.padding-private.v1"
_LAYOUT = "logical_decode_rows_to_fixed_physical_rows"
_CACHE_FORM = "bf16_paged_kv"
_STATE_FORM = "bf16_conv_and_fp32_recurrent_cow"
_CANARY = (0x13579BDF, 0x2468ACE0)

_MRoPETables = TypeAliasType("_MRoPETables", tuple[Tensor, Tensor, MRoPETableEvidence])
_ChildOperatorClosure = TypeAliasType("_ChildOperatorClosure", tuple[tuple[str, tuple[tuple[str, int], ...]], ...])


class Qwen3_5PerformanceRejected(RuntimeError):
    """The model, arena, or capture differs from the fixed decode ABI."""


@dataclass(slots=True)
class _Buffers:
    token_ids: Tensor
    position_ids: Tensor
    page_source: Tensor
    page_destination: Tensor
    token_offsets: Tensor
    page_table: Tensor
    sequence_lengths: Tensor
    segment_lengths: Tensor
    row_activity: Tensor
    conv_slots: Tensor
    recurrent_slots: Tensor
    hidden: Tensor
    after_attention: Tensor
    norm: Tensor
    update: Tensor
    mlp_gate: Tensor
    mlp_up: Tensor
    mlp_active: Tensor
    gdn_packed: Tensor
    gdn_convolved: Tensor
    gdn_a: Tensor
    gdn_b: Tensor
    gdn_core: Tensor
    gdn_gate: Tensor
    gdn_normalized: Tensor
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
    guard: Tensor

    @property
    def state_slot_maps(self) -> tuple[Tensor, ...]:
        return self.conv_slots, self.recurrent_slots


def _copy_decode_input_rows(source: _Buffers, target: _Buffers, start: int, stop: int) -> None:
    """Copy explicit ownership and segment metadata into disjoint activation scratch."""
    for name in (
        "token_ids", "position_ids", "page_source", "page_destination", "token_offsets",
        "page_table", "sequence_lengths", "segment_lengths", "row_activity", "conv_slots",
        "recurrent_slots", "guard",
    ):
        value = getattr(source, name)
        if name == "guard":
            part = value
        elif name == "position_ids":
            part = value[:, start:stop]
        elif name in ("conv_slots", "recurrent_slots"):
            part = value[:, :, start:stop]
        else:
            part = value[start:stop]
        getattr(target, name).copy_(part)


@dataclass(slots=True)
class _HostBuffers:
    token_ids: Tensor
    position_ids: Tensor
    page_source: Tensor
    page_destination: Tensor
    token_offsets: Tensor
    page_table: Tensor
    sequence_lengths: Tensor
    segment_lengths: Tensor
    row_activity: Tensor
    conv_slots: Tensor
    recurrent_slots: Tensor
    guard: Tensor

    @classmethod
    def like(cls, buffers: _Buffers) -> _HostBuffers:
        def pinned(value: Tensor) -> Tensor:
            return torch.empty(value.shape, dtype=value.dtype, device="cpu", pin_memory=True)

        return cls(*(pinned(getattr(buffers, name)) for name in cls.__annotations__))

    def copy_to(self, buffers: _Buffers) -> None:
        for name in type(self).__annotations__:
            getattr(buffers, name).copy_(getattr(self, name), non_blocking=True)


class _DecodeBody:
    """All tensors and bound launches for one fixed captured-M lane."""

    def __init__(
        self,
        model: Qwen3_5ReferenceModel,
        arena: CacheArena,
        *,
        captured_m: int,
        table_pages: int,
        mrope_tables: _MRoPETables | None = None,
        packed_operators: PackedLinearOperators | None = None,
        first_stage: bool = True,
        last_stage: bool = True,
        tp_reduce: Callable[[Tensor], None] | None = None,
        embedding_mask: Tensor | None = None,
        feature_rows: Tensor | None = None,
        expert_span: AxisSpan | None = None,
        expert_route_reduce: Callable[[Tensor], None] | None = None,
        route_completion: RouteCompletionPlanner | None = None,
        expert_output_reduce: Callable[[Tensor], None] | None = None,
        shared_scratch: dict[tuple, Tensor] | None = None,
    ) -> None:
        if packed_operators is None and isinstance(model.default_operators, PackedLinearOperators):
            packed_operators = model.default_operators
        self._first_stage, self._last_stage = first_stage, last_stage
        self._tp_reduce, self._embedding_mask = tp_reduce, embedding_mask
        self._feature_rows = feature_rows
        self._expert_output_reduce = expert_output_reduce
        text = model.config.graph.text
        if text.dtype not in ("bfloat16", "float16"):
            raise Qwen3_5PerformanceRejected(
                "model graphs require FP16 or BF16 execution tensors"
            )
        gdn, attention = text.gated_delta_net, text.full_attention
        schedule = text.layer_schedule
        full_layers = tuple(
            i for i, kind in enumerate(schedule) if kind is AttentionKind.FULL_ATTENTION
        )
        gdn_layers = tuple(
            i for i, kind in enumerate(schedule) if kind is AttentionKind.GATED_DELTA_NET
        )
        expected_conv = (
            len(gdn_layers),
            2 * gdn.key_heads * gdn.key_head_dim + gdn.value_heads * gdn.value_head_dim,
            gdn.convolution_kernel_size - 1,
        )
        expected_recurrent = (
            len(gdn_layers),
            gdn.value_heads,
            gdn.value_head_dim,
            gdn.key_head_dim,
        )
        if (
            captured_m <= 0
            or table_pages <= 0
            or model.weights.device.type != "cuda"
            or arena.device != model.weights.device
            or arena.layout.layer_ids != full_layers
            or arena.layout.num_kv_heads != attention.key_value_heads
            or arena.layout.head_dim != attention.head_dim
            or GDN_CONV_STATE_ID not in arena.states
            or GDN_RECURRENT_STATE_ID not in arena.states
            or tuple(arena.states[GDN_CONV_STATE_ID].shape[1:]) != expected_conv
            or tuple(arena.states[GDN_RECURRENT_STATE_ID].shape[1:]) != expected_recurrent
        ):
            raise Qwen3_5PerformanceRejected("model and physical arena do not close decode ABI")

        device, hidden = model.weights.device, text.hidden_size
        compute_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[text.dtype]
        packed_moe = text.moe if packed_operators is not None else None
        self._packed_moe = packed_moe
        intermediate = (
            packed_moe.shared_expert_intermediate_size
            if packed_moe is not None else text.dense_intermediate_size
        )
        if intermediate is None:
            raise Qwen3_5PerformanceRejected("dense decode graph requires dense intermediate size")
        packed = expected_conv[1]
        kv_width = attention.key_value_heads * attention.head_dim
        rotary = int(attention.head_dim * text.mrope.partial_rotary_factor)
        from cubic_serve.executor.text_graph import TextGraphBindings, full_attention_partitions

        attention_splits = full_attention_partitions(
            captured_m, max_tokens=text.max_position_embeddings,
            query_heads=attention.query_heads, kv_heads=attention.key_value_heads,
            head_dim=attention.head_dim, device=device, shared_scratch=shared_scratch,
        )
        self._activation_scratch = TextGraphBindings(
            packed_operators, device, compute_dtype, shared_scratch=shared_scratch,
        )

        def empty(*shape: int, role: str | None = None) -> Tensor:
            if role is not None:
                return self._activation_scratch.buffer("decode." + role, *shape)
            return torch.empty(shape, dtype=compute_dtype, device=device)

        if mrope_tables is None:
            mrope_tables = build_bf16_mrope_tables(
                max_position=text.max_position_embeddings,
                device=device,
                rotary_dim=rotary,
                section=text.mrope.section,
                rope_theta=text.mrope.rope_theta,
                dtype=compute_dtype,
            )
        cosine_table, sine_table, table_evidence = mrope_tables
        self.model, self.arena = model, arena
        self.schedule, self.full_layers, self.gdn_layers = schedule, full_layers, gdn_layers
        self._full_local = {layer: local for local, layer in enumerate(full_layers)}
        self._gdn_local = {layer: local for local, layer in enumerate(gdn_layers)}
        self.conv_state = (arena.states[GDN_CONV_STATE_ID].reshape(-1, packed, expected_conv[2])
                           if arena.states[GDN_CONV_STATE_ID].is_contiguous() else arena.states[GDN_CONV_STATE_ID])
        self.recurrent_state = (arena.states[GDN_RECURRENT_STATE_ID].reshape(
            -1, gdn.value_heads, gdn.value_head_dim, gdn.key_head_dim
        ) if arena.states[GDN_RECURRENT_STATE_ID].is_contiguous() else arena.states[GDN_RECURRENT_STATE_ID])
        self.buffers = b = _Buffers(
            token_ids=torch.zeros(captured_m, dtype=torch.int64, device=device),
            position_ids=torch.zeros((3, captured_m), dtype=torch.int64, device=device),
            page_source=torch.zeros(captured_m, dtype=torch.int32, device=device),
            page_destination=torch.zeros(captured_m, dtype=torch.int32, device=device),
            token_offsets=torch.zeros(captured_m, dtype=torch.int32, device=device),
            page_table=torch.zeros((captured_m, table_pages), dtype=torch.int32, device=device),
            sequence_lengths=torch.ones(captured_m, dtype=torch.int32, device=device),
            segment_lengths=torch.ones(captured_m, dtype=torch.int32, device=device),
            row_activity=torch.ones(captured_m, dtype=torch.int32, device=device),
            conv_slots=torch.zeros(
                (2, len(gdn_layers), captured_m), dtype=torch.int32, device=device
            ),
            recurrent_slots=torch.zeros(
                (2, len(gdn_layers), captured_m), dtype=torch.int32, device=device
            ),
            hidden=empty(captured_m, hidden),
            after_attention=empty(captured_m, hidden),
            norm=empty(captured_m, hidden),
            update=empty(captured_m, hidden),
            mlp_gate=empty(captured_m, intermediate, role="mlp_gate"),
            mlp_up=empty(captured_m, intermediate, role="mlp_up"),
            mlp_active=empty(captured_m, intermediate, role="mlp_active"),
            gdn_packed=empty(captured_m, packed, role="gdn_packed"),
            gdn_convolved=empty(captured_m, packed, role="gdn_convolved"),
            gdn_a=empty(captured_m, gdn.value_heads, role="gdn_a"),
            gdn_b=empty(captured_m, gdn.value_heads, role="gdn_b"),
            gdn_core=empty(captured_m, gdn.value_heads, gdn.value_head_dim, role="gdn_core"),
            gdn_gate=empty(captured_m, gdn.value_heads, gdn.value_head_dim, role="gdn_gate"),
            gdn_normalized=empty(captured_m, gdn.value_heads, gdn.value_head_dim, role="gdn_normalized"),
            attention_projected=empty(captured_m, attention.query_heads, 2, attention.head_dim, role="attention_projected"),
            query=empty(captured_m, attention.query_heads, attention.head_dim, role="query"),
            attention_gate=empty(captured_m, attention.query_heads, attention.head_dim, role="attention_gate"),
            key=empty(captured_m, attention.key_value_heads, attention.head_dim, role="key"),
            value=empty(captured_m, attention.key_value_heads, attention.head_dim, role="value"),
            normalized_query=empty(captured_m, attention.query_heads, attention.head_dim, role="normalized_query"),
            normalized_key=empty(captured_m, attention.key_value_heads, attention.head_dim, role="normalized_key"),
            rotated_query=empty(captured_m, attention.query_heads, attention.head_dim, role="rotated_query"),
            rotated_key=empty(captured_m, attention.key_value_heads, attention.head_dim, role="rotated_key"),
            attention_output=empty(captured_m, attention.query_heads, attention.head_dim, role="attention_output"),
            attention_partial=self._activation_scratch.buffer(
                "attention.partial", captured_m, attention.query_heads,
                attention_splits, attention.head_dim, dtype=torch.float32, serial=True,
            ),
            attention_statistics=self._activation_scratch.buffer(
                "attention.statistics", captured_m, attention.query_heads,
                attention_splits, 2, dtype=torch.float32, serial=True,
            ),
            attention_arrivals=torch.zeros(
                (captured_m, attention.query_heads), dtype=torch.int32, device=device
            ),
            cosine=empty(captured_m, rotary, role="cosine"),
            sine=empty(captured_m, rotary, role="sine"),
            # Sampling consumes this terminal output before the topology lease
            # ends. Feedback/hidden boundaries remain separately owned.
            logits=empty(captured_m, text.vocabulary_size, role="sample_logits"),
            guard=torch.zeros(2, dtype=torch.int64, device=device),
        )
        weights = model.weights
        first_gdn = f"model.language_model.layers.{gdn_layers[0]}.linear_attn"
        first_attention = f"model.language_model.layers.{full_layers[0]}.self_attn"
        first_mlp = "model.language_model.layers.0.mlp"
        if packed_moe is not None:
            first_mlp += ".shared_expert"
        packed_linear = PackedProjectionDispatch(packed_operators) if packed_operators else None
        packed_embedding = (
            PackedProjectionDispatch(packed_operators, operation="embedding")
            if packed_operators else None
        )

        def linear(
            x: Tensor, weight: Tensor, output: Tensor,
        ) -> BF16LinearPrimitive | PackedProjectionDispatch:
            if packed_linear is None:
                return BF16LinearPrimitive(x, weight, output)
            packed_linear.prepare(x, weight, output)
            return packed_linear

        self.embedding = None
        if first_stage:
            self.embedding = packed_embedding or BF16TokenEmbeddingPrimitive(
                b.token_ids, weights["model.language_model.embed_tokens.weight"], b.hidden
            )
        if first_stage and packed_embedding is not None:
            packed_embedding.prepare(
                b.token_ids, weights["model.language_model.embed_tokens.weight"], b.hidden
            )
            packed_embedding.seal()
        self.mrope_gather = MRoPECosSinGatherPrimitive(
            b.position_ids,
            cosine_table,
            sine_table,
            b.cosine,
            b.sine,
            evidence=table_evidence,
        )
        self.cosine_table, self.sine_table = cosine_table, sine_table
        from cubic_serve.ops.cuda.cache_quantization import (
            PagedKVGraphBinding,
            paged_cache_encoding,
        )
        encoding = paged_cache_encoding(arena)
        self._cache_binding = None
        if encoding is not None:
            self._cache_binding = PagedKVGraphBinding(
                arena, positions=b.sequence_lengths, positions_are_lengths=True,
                segments=b.segment_lengths, activity=b.row_activity,
                source_slots=b.conv_slots[0, 0], destination_slots=b.conv_slots[1, 0],
                owner_stride=b.conv_slots.shape[1],
            )
        self.page_copy = PageCOWCopyPrimitive(
            arena.key, arena.value, arena.written, b.page_source, b.page_destination,
            page_fields=tuple(arena.page_fields.values()), cache_encoding=encoding,
        )
        self.hidden_norm = ZeroCenteredRMSNormPrimitive(
            b.hidden,
            weights["model.language_model.layers.0.input_layernorm.weight"],
            b.norm,
            epsilon=text.rms_norm_epsilon,
        )
        self.q_norm = ZeroCenteredRMSNormPrimitive(
            b.query.reshape(-1, attention.head_dim),
            weights[f"{first_attention}.q_norm.weight"],
            b.normalized_query.reshape(-1, attention.head_dim),
            epsilon=text.rms_norm_epsilon,
        )
        self.k_norm = ZeroCenteredRMSNormPrimitive(
            b.key.reshape(-1, attention.head_dim),
            weights[f"{first_attention}.k_norm.weight"],
            b.normalized_key.reshape(-1, attention.head_dim),
            epsilon=text.rms_norm_epsilon,
        )
        self.gdn_norm = GatedRMSNormPrimitive(
            b.gdn_core,
            b.gdn_gate,
            weights[f"{first_gdn}.norm.weight"],
            b.gdn_normalized,
            epsilon=text.rms_norm_epsilon,
        )
        self.linear_gdn_qkv = linear(
            b.norm, weights[f"{first_gdn}.in_proj_qkv.weight"], b.gdn_packed
        )
        self.linear_gdn_ab = linear(
            b.norm, weights[f"{first_gdn}.in_proj_a.weight"], b.gdn_a
        )
        self.linear_gdn_z = linear(
            b.norm, weights[f"{first_gdn}.in_proj_z.weight"], b.gdn_gate.reshape(captured_m, -1)
        )
        self.linear_attention_q = linear(
            b.norm,
            weights[f"{first_attention}.q_proj.weight"],
            b.attention_projected.reshape(captured_m, -1),
        )
        self.linear_attention_kv = linear(
            b.norm, weights[f"{first_attention}.k_proj.weight"], b.key.reshape(captured_m, kv_width)
        )
        self.linear_output = linear(
            b.gdn_normalized.reshape(captured_m, -1),
            weights[f"{first_gdn}.out_proj.weight"],
            b.update,
        )
        self.linear_mlp_wide = linear(
            b.norm, weights[f"{first_mlp}.gate_proj.weight"], b.mlp_gate
        )
        self.linear_mlp_down = linear(
            b.mlp_active, weights[f"{first_mlp}.down_proj.weight"], b.update
        )
        self.linear_logits = (
            linear(b.norm, weights["lm_head.weight"], b.logits) if last_stage else None
        )
        self.conv = CausalConv1DDecodePrimitive(
            b.gdn_packed,
            weights[f"{first_gdn}.conv1d.weight"].squeeze(1),
            self.conv_state,
            b.conv_slots[0, 0],
            b.conv_slots[1, 0],
            b.segment_lengths,
            b.gdn_convolved,
            checkpoint_destinations=True,
        )
        self.gdn = GatedDeltaDecodePrimitive(
            b.gdn_convolved,
            b.gdn_packed,
            b.gdn_a,
            b.gdn_b,
            weights[f"{first_gdn}.A_log"],
            weights[f"{first_gdn}.dt_bias"],
            self.recurrent_state,
            b.recurrent_slots[0, 0],
            b.recurrent_slots[1, 0],
            b.segment_lengths,
            b.gdn_core,
            q_heads=gdn.key_heads,
            scale=gdn.key_head_dim**-0.5,
            checkpoint_destinations=True,
            row_activity=b.row_activity,
        )
        self.q_gate_split = QwenAttentionQGateSplitPrimitive(
            b.attention_projected, b.query, b.attention_gate
        )
        self.mrope_apply = QwenMRoPEApplyPrimitive(
            b.normalized_query,
            b.normalized_key,
            b.cosine,
            b.sine,
            b.rotated_query,
            b.rotated_key,
        )
        cache_layers = (() if self._cache_binding is None else tuple(
            self._cache_binding.bind_layer(b.rotated_key, b.value, layer=local,
                                           pages=b.page_destination, query=b.rotated_query)
            for local in range(len(full_layers))))
        self._cache_checkpoints = tuple(layer[2] for layer in cache_layers)
        self._attention_queries = tuple(
            b.rotated_query if self._cache_binding is None else
            self._cache_binding.query_for(b.rotated_query) for _ in full_layers
        )
        self.page_writes = (tuple(layer[1] for layer in cache_layers)
                            if cache_layers else tuple(
            PageTokenWritePrimitive(
                arena.key,
                arena.value,
                arena.written,
                b.page_destination,
                b.token_offsets,
                b.rotated_key,
                b.value,
                layer=local,
            )
            for local in range(len(full_layers))
        ))
        self.attentions = tuple(
            BF16PagedGQADecodePrimitive(
                self._attention_queries[local],
                arena.key,
                arena.value,
                b.page_table,
                b.sequence_lengths,
                b.attention_output,
                layer=local,
                partial=b.attention_partial,
                statistics=b.attention_statistics,
                arrivals=b.attention_arrivals,
                cache_encoding=encoding,
                cache_window=cache_layers[local][0] if cache_layers else None,
            )
            for local in range(len(full_layers))
        )
        self.sigmoid_gate = BF16SigmoidOutputGatePrimitive(
            b.attention_output, b.attention_gate, b.rotated_query
        )
        self.residual = BF16ResidualAddPrimitive(b.hidden, b.update, b.after_attention)
        self.residual_norm = BF16ResidualRMSNormPrimitive(
            b.hidden,
            b.update,
            b.after_attention,
            weights["model.language_model.layers.0.input_layernorm.weight"],
            b.norm,
            epsilon=text.rms_norm_epsilon,
        )
        self.mlp_gate = BF16SwiGLUPrimitive(b.mlp_gate, b.mlp_up, b.mlp_active)
        cache_scratch = (() if self._cache_binding is None else self._cache_binding.scratch)
        self.packed_scratch: tuple[Tensor, ...] = cache_scratch
        self._packed_experts: list[PackedRoutedExpertsPrimitive] = []
        if packed_moe is not None:
            self._packed_router_logits = empty(captured_m, packed_moe.expert_count)
            self._packed_shared_gate = empty(captured_m, 1)
            self._packed_routed_output = empty(captured_m, hidden)
            # Activity, not segment starts, determines ownership of every row.
            self._packed_active_rows = torch.full(
                (1,), captured_m, dtype=torch.int32, device=device
            )
            self._packed_moe_combine = BF16MoeSharedCombinePrimitive(
                b.update, self._packed_shared_gate, self._packed_routed_output, b.update
            )
            self.packed_scratch = (
                self._packed_router_logits, self._packed_shared_gate,
                self._packed_routed_output, self._packed_active_rows,
                *cache_scratch,
            )
        if packed_linear is not None:
            for layer, kind in enumerate(schedule):
                prefix = f"model.language_model.layers.{layer}"
                if kind is AttentionKind.GATED_DELTA_NET:
                    op = f"{prefix}.linear_attn"
                    members = (
                        ("in_proj_qkv", b.gdn_packed), ("in_proj_a", b.gdn_a),
                        ("in_proj_b", b.gdn_b), ("in_proj_z", b.gdn_gate.reshape(captured_m, -1)),
                    )
                    packed_linear.prepare_group(
                        b.norm, tuple(weights[f"{op}.{suffix}.weight"] for suffix, _ in members),
                        tuple(output for _, output in members),
                    )
                    packed_linear.prepare(
                        b.gdn_normalized.reshape(captured_m, -1),
                        weights[f"{op}.out_proj.weight"], b.update,
                    )
                else:
                    op = f"{prefix}.self_attn"
                    members = (
                        ("q_proj", b.attention_projected.reshape(captured_m, -1)),
                        ("k_proj", b.key.reshape(captured_m, -1)),
                        ("v_proj", b.value.reshape(captured_m, -1)),
                    )
                    packed_linear.prepare_group(
                        b.norm, tuple(weights[f"{op}.{suffix}.weight"] for suffix, _ in members),
                        tuple(output for _, output in members),
                    )
                    packed_linear.prepare(
                        b.rotated_query.reshape(captured_m, -1), weights[f"{op}.o_proj.weight"],
                        b.update,
                    )
                mlp = f"{prefix}.mlp"
                mlp_inputs: list[Tensor] = []
                mlp_outputs: list[Tensor] = []
                if packed_moe is not None:
                    packed_linear.prepare(
                        b.norm, weights[f"{mlp}.gate.weight"], self._packed_router_logits
                    )
                    packed_linear.prepare(
                        b.norm, weights[f"{mlp}.shared_expert_gate.weight"],
                        self._packed_shared_gate,
                    )
                    mlp_inputs.extend((weights[f"{mlp}.gate.weight"],
                                       weights[f"{mlp}.shared_expert_gate.weight"]))
                    mlp_outputs.extend((self._packed_router_logits, self._packed_shared_gate))
                    assert packed_operators is not None
                    owned = expert_span or AxisSpan(0, packed_moe.expert_count)
                    gate_weights, up_weights, down_weights = (
                        tuple(weights[f"{mlp}.experts.{expert}.{role}_proj.weight"]
                              for expert in range(owned.start, owned.stop))
                        for role in ("gate", "up", "down")
                    )
                    experts = PackedRoutedExpertsPrimitive(
                        b.norm, self._packed_router_logits, b.row_activity,
                        self._packed_active_rows, gate_weights, up_weights, down_weights,
                        self._packed_routed_output, first_expert=owned.start,
                        top_k=packed_moe.experts_per_token, operators=packed_operators,
                        reduce_routes=expert_route_reduce,
                        route_completion=route_completion,
                    )
                    self._packed_experts.append(experts)
                    self.packed_scratch += experts.scratch
                    mlp += ".shared_expert"
                for suffix, output in (("gate_proj", b.mlp_gate), ("up_proj", b.mlp_up)):
                    mlp_inputs.append(weights[f"{mlp}.{suffix}.weight"])
                    mlp_outputs.append(output)
                packed_linear.prepare_group(b.norm, tuple(mlp_inputs), tuple(mlp_outputs))
                packed_linear.prepare(
                    b.mlp_active, weights[f"{mlp}.down_proj.weight"], b.update
                )
            packed_linear.seal()
            self.packed_scratch += packed_linear.scratch

        self._packed_branches: MoeBranchStreams | None = None
        self._packed_branch_costs: tuple[float, float] | None = None
        if (packed_linear is not None and self._packed_experts
                and launch_parameters().shared_expert_overlap
                and not any(op.has_collectives for op in self._packed_experts)):
            # Projection outputs may share a serial FP32 workspace. Routed
            # experts must own separate writable storage before fork/join is
            # allowed. Read-only norm/weights remain shared; no input copy.
            shared = tuple((t.data_ptr(), t.data_ptr() + t.numel() * t.element_size())
                           for t in packed_linear.scratch)
            independent = all(
                not any(t.data_ptr() < end and start < t.data_ptr() + t.numel() * t.element_size()
                        for start, end in shared)
                for op in self._packed_experts for t in op.scratch
            )
            if independent:
                assert packed_moe is not None
                rate = device_work_cost(b.norm.device, b.norm.dtype)

                def work(matrices, rows: float, active: float = 1.0):
                    operations = traffic = 0.0
                    for matrix in matrices:
                        n, k = matrix.shape
                        storage = (matrix.numel() * matrix.element_size()
                                   if isinstance(matrix, Tensor) else
                                   sum(t.numel() * t.element_size()
                                       for t in matrix.payload.values()))
                        operations += 2 * rows * n * k
                        traffic += active * storage + rows * (n + k) * b.norm.element_size()
                    return operations, traffic

                serial = parallel = 0.0
                owned = expert_span or AxisSpan(0, packed_moe.expert_count)
                population = captured_m * packed_moe.experts_per_token / packed_moe.expert_count
                # Uniform route population is a planning prior, never a remap
                # or a replacement for the actual device ownership metadata.
                active = -math.expm1(-population)
                for layer in range(len(schedule)):
                    prefix = f"model.language_model.layers.{layer}.mlp"
                    shared_work = work(
                        (weights[f"{prefix}.shared_expert.{role}_proj.weight"]
                         for role in ("gate", "up", "down")), captured_m,
                    )
                    routed_work = work(
                        (weights[f"{prefix}.experts.{expert}.{role}_proj.weight"]
                         for expert in range(owned.start, owned.stop)
                         for role in ("gate", "up", "down")), population, active,
                    )
                    a, c = rate.branch_costs(
                        (*shared_work, 4), (*routed_work, 6),
                        fork_join_launches=launch_parameters().branch_fork_join_launches,
                    )
                    serial += a
                    parallel += c
                self._packed_branch_costs = serial, parallel
                if parallel < serial:
                    self._packed_branches = MoeBranchStreams(
                        b.norm.device, len(self._packed_experts),
                    )

    def _launch_shared_mlp(self, prefix: str) -> None:
        b, weights = self.buffers, self.model.weights
        self.linear_mlp_wide.launch(b.norm, weights[f"{prefix}.gate_proj.weight"], b.mlp_gate)
        self.linear_mlp_wide.launch(b.norm, weights[f"{prefix}.up_proj.weight"], b.mlp_up)
        self.mlp_gate.launch(b.mlp_gate, b.mlp_up, b.mlp_active)
        self.linear_mlp_down.launch(b.mlp_active, weights[f"{prefix}.down_proj.weight"], b.update)

    def prepare_capture_scratch(self, resource: ec.RankReplayResources) -> None:
        """Warm and capture only against this resource's ownerless scratch."""

        self._bind_capture_scratch(resource, clear=True)

    def bind_capture_scratch(self, resource: ec.RankReplayResources) -> None:
        """Rebind padding rows without clearing the private scratch backing."""

        self._bind_capture_scratch(resource, clear=False)

    def _bind_capture_scratch(self, resource: ec.RankReplayResources, *, clear: bool) -> None:
        arena, b = self.arena, self.buffers
        spec, captured_m = arena.spec, b.token_ids.shape[0]
        try:
            scratch_index = spec.scratch_arena_ids.index(resource.scratch_arena_id)
        except ValueError as error:
            raise Qwen3_5PerformanceRejected(
                "capture resource names an unknown cache scratch arena"
            ) from error
        if spec.scratch_capacity < captured_m:
            raise Qwen3_5PerformanceRejected("capture scratch cannot cover captured M")
        page_slots = torch.tensor(
            arena.padding_page_slots(captured_m, resource.scratch_arena_id),
            dtype=torch.int32, device=arena.device,
        )
        padding_states = arena.padding_state_slots(captured_m, resource.scratch_arena_id)
        state_slots = torch.tensor(
            padding_states, dtype=torch.int32, device=arena.device
        )
        local_layers = torch.arange(len(self.gdn_layers), dtype=torch.int32, device=arena.device)
        aggregate = state_slots[:, None] * len(self.gdn_layers) + local_layers[None, :]

        if clear:
            arena.reset_page_rows(page_slots.to(torch.int64))
            arena.reset_state_rows(torch.tensor(
                tuple(dict.fromkeys(padding_states)), dtype=torch.int64, device=arena.device))
        b.token_ids.zero_()
        b.position_ids.zero_()
        b.page_source.copy_(page_slots)
        b.page_destination.copy_(page_slots)
        b.token_offsets.zero_()
        b.page_table.copy_(page_slots[:, None].expand_as(b.page_table))
        b.sequence_lengths.fill_(1)
        b.segment_lengths.fill_(1)
        b.row_activity.zero_()
        for indices in (b.conv_slots, b.recurrent_slots):
            indices[0].copy_(aggregate.t())
            indices[1].copy_(aggregate.t())
        b.guard.copy_(torch.tensor(_CANARY, dtype=torch.int64, device=arena.device))

    def launch(self) -> None:
        for collective, operand in self.launch_steps():
            collective(operand)

    def launch_steps(self) -> Iterator[tuple[Callable[[Tensor], None], Tensor]]:
        """Yield only explicit collective boundaries; arithmetic has one definition.

        Callers must finish the yielded operation before advancing this body's
        iterator. Independent row bodies may compute during that operation.
        Opaque operator-internal collectives are not overlap capabilities.
        """
        b, weights = self.buffers, self.model.weights
        if self._packed_branches is not None:
            self._packed_branches.begin()
        if self._first_stage:
            assert self.embedding is not None
            self.embedding.launch(
                b.token_ids, weights["model.language_model.embed_tokens.weight"], b.hidden
            )
            if self._embedding_mask is not None:
                torch.mul(b.hidden, self._embedding_mask, out=b.hidden)
            if self._tp_reduce is not None and not getattr(self, "_embedding_complete", False):
                yield self._tp_reduce, b.hidden
            if self._feature_rows is not None:
                torch.add(b.hidden, self._feature_rows, out=b.hidden)
        self.mrope_gather.launch(
            b.position_ids, self.cosine_table, self.sine_table, b.cosine, b.sine
        )
        self.page_copy.launch(
            self.arena.key,
            self.arena.value,
            self.arena.written,
            b.page_source,
            b.page_destination,
        )
        if self._cache_binding is not None:
            self._cache_binding.prepare_metadata()
        for layer, kind in enumerate(self.schedule):
            prefix = f"model.language_model.layers.{layer}"
            self.hidden_norm.launch(b.hidden, weights[f"{prefix}.input_layernorm.weight"], b.norm)
            if kind is AttentionKind.GATED_DELTA_NET:
                op, local = f"{prefix}.linear_attn", self._gdn_local[layer]
                self.linear_gdn_qkv.launch(
                    b.norm, weights[f"{op}.in_proj_qkv.weight"], b.gdn_packed
                )
                self.conv.launch(
                    b.gdn_packed,
                    weights[f"{op}.conv1d.weight"].squeeze(1),
                    self.conv_state,
                    b.conv_slots[0, local],
                    b.conv_slots[1, local],
                    b.segment_lengths,
                    b.gdn_convolved,
                )
                self.linear_gdn_ab.launch(b.norm, weights[f"{op}.in_proj_a.weight"], b.gdn_a)
                self.linear_gdn_ab.launch(b.norm, weights[f"{op}.in_proj_b.weight"], b.gdn_b)
                self.gdn.launch(
                    b.gdn_convolved,
                    b.gdn_packed,
                    b.gdn_a,
                    b.gdn_b,
                    weights[f"{op}.A_log"],
                    weights[f"{op}.dt_bias"],
                    self.recurrent_state,
                    b.recurrent_slots[0, local],
                    b.recurrent_slots[1, local],
                    b.segment_lengths,
                    b.gdn_core,
                )
                self.linear_gdn_z.launch(
                    b.norm,
                    weights[f"{op}.in_proj_z.weight"],
                    b.gdn_gate.reshape(b.hidden.shape[0], -1),
                )
                self.gdn_norm.launch(
                    b.gdn_core, b.gdn_gate, weights[f"{op}.norm.weight"], b.gdn_normalized
                )
                self.linear_output.launch(
                    b.gdn_normalized.reshape(b.hidden.shape[0], -1),
                    weights[f"{op}.out_proj.weight"],
                    b.update,
                )
            else:
                op, local = f"{prefix}.self_attn", self._full_local[layer]
                self.linear_attention_q.launch(
                    b.norm,
                    weights[f"{op}.q_proj.weight"],
                    b.attention_projected.reshape(b.hidden.shape[0], -1),
                )
                self.q_gate_split.launch(b.attention_projected, b.query, b.attention_gate)
                self.linear_attention_kv.launch(
                    b.norm, weights[f"{op}.k_proj.weight"], b.key.reshape(b.hidden.shape[0], -1)
                )
                self.linear_attention_kv.launch(
                    b.norm, weights[f"{op}.v_proj.weight"], b.value.reshape(b.hidden.shape[0], -1)
                )
                self.q_norm.launch(
                    b.query.reshape(-1, b.query.shape[-1]),
                    weights[f"{op}.q_norm.weight"],
                    b.normalized_query.reshape(-1, b.query.shape[-1]),
                )
                self.k_norm.launch(
                    b.key.reshape(-1, b.key.shape[-1]),
                    weights[f"{op}.k_norm.weight"],
                    b.normalized_key.reshape(-1, b.key.shape[-1]),
                )
                self.mrope_apply.launch(
                    b.normalized_query,
                    b.normalized_key,
                    b.cosine,
                    b.sine,
                    b.rotated_query,
                    b.rotated_key,
                )
                if self._cache_binding is not None:
                    self.page_writes[local].launch()
                else:
                    self.page_writes[local].launch(
                        self.arena.key,
                        self.arena.value,
                        self.arena.written,
                        b.page_destination,
                        b.token_offsets,
                        b.rotated_key,
                        b.value,
                    )
                self.attentions[local].launch(
                    self._attention_queries[local],
                    self.arena.key,
                    self.arena.value,
                    b.page_table,
                    b.sequence_lengths,
                    b.attention_output,
                )
                if self._cache_binding is not None:
                    self._cache_checkpoints[local].launch()
                self.sigmoid_gate.launch(b.attention_output, b.attention_gate, b.rotated_query)
                self.linear_output.launch(
                    b.rotated_query.reshape(b.hidden.shape[0], -1),
                    weights[f"{op}.o_proj.weight"],
                    b.update,
                )
            if self._tp_reduce is not None:
                yield self._tp_reduce, b.update
            self.residual.launch(b.hidden, b.update, b.after_attention)
            self.hidden_norm.launch(
                b.after_attention, weights[f"{prefix}.post_attention_layernorm.weight"], b.norm
            )
            mlp = f"{prefix}.mlp"
            if self._packed_moe is not None:
                self.linear_mlp_wide.launch(
                    b.norm, weights[f"{mlp}.gate.weight"], self._packed_router_logits
                )
                self.linear_mlp_wide.launch(
                    b.norm, weights[f"{mlp}.shared_expert_gate.weight"], self._packed_shared_gate
                )
                mlp += ".shared_expert"
            if self._packed_branches is not None:
                with self._packed_branches.fork_shared():
                    self._launch_shared_mlp(mlp)
                self._packed_experts[layer].launch()
                self._packed_branches.join_shared()
            else:
                if self._packed_moe is not None:
                    self._packed_experts[layer].launch()
                self._launch_shared_mlp(mlp)
            if self._tp_reduce is not None:
                yield self._tp_reduce, b.update
            if self._packed_moe is not None and self._expert_output_reduce is not None:
                yield self._expert_output_reduce, self._packed_routed_output
            if self._packed_moe is not None:
                self._packed_moe_combine.launch(
                    b.update, self._packed_shared_gate, self._packed_routed_output, b.update
                )
            self.residual.launch(b.after_attention, b.update, b.hidden)
        if self._last_stage:
            assert self.linear_logits is not None
            self.hidden_norm.launch(b.hidden, weights["model.language_model.norm.weight"], b.norm)
            self.linear_logits.launch(b.norm, weights["lm_head.weight"], b.logits)


@dataclass(slots=True)
class Qwen3_5DecodeGraphSlot:
    """One rank-local captured graph and every fixed address it references."""

    resources: tuple[ec.RankReplayResources, ...]
    body: _DecodeBody
    host: _HostBuffers
    stream: torch.cuda.Stream
    graph: torch.cuda.CUDAGraph

    @classmethod
    def capture(
        cls,
        *,
        model: Qwen3_5ReferenceModel,
        arena: CacheArena,
        resource: ec.RankReplayResources,
        captured_m: int,
        table_pages: int,
        mrope_tables: _MRoPETables | None = None,
    ) -> Qwen3_5DecodeGraphSlot:
        if resource.rank != arena.rank_id:
            raise Qwen3_5PerformanceRejected("capture resource rank differs from cache arena")
        device = model.weights.device
        with torch.cuda.device(device):
            body = _DecodeBody(
                model,
                arena,
                captured_m=captured_m,
                table_pages=table_pages,
                mrope_tables=mrope_tables,
            )
            body.prepare_capture_scratch(resource)
            stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                body.launch()
                body.launch()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                body.launch()
            stream.synchronize()
        return cls((resource,), body, _HostBuffers.like(body.buffers), stream, graph)

    def replay(self) -> None:
        with torch.cuda.device(self.body.arena.device), torch.cuda.stream(self.stream):
            self.graph.replay()

    def synchronize(self) -> None:
        self.stream.synchronize()


@dataclass(slots=True)
class _QwenDecodeGraphSession:
    slot: Qwen3_5DecodeGraphSlot
    plan: ec.LaunchPlan
    state: DecodeSession
    expected_guard: tuple[int, int]
    replayed: bool = False
    consumed: bool = False
    finished: bool = False
    output: Tensor | None = None

    def replay(self) -> None:
        if self.replayed or self.finished:
            raise RuntimeError("qwen decode graph session permits exactly one replay")
        self.slot.replay()
        self.replayed = True

    def consume(self) -> Mapping[str, Tensor]:
        if not self.replayed or self.consumed or self.finished:
            raise RuntimeError("qwen decode output is unavailable or already consumed")
        self.slot.synchronize()
        body = self.slot.body
        if tuple(body.buffers.guard.tolist()) != self.expected_guard:
            raise RuntimeError("qwen decode graph corrupted its static canary")
        physical = [0] * self.plan.logical_m
        for index, row in enumerate(self.plan.rows):
            if isinstance(row, ec.ActiveLaunchRow):
                physical[row.logical_index] = index
        output = torch.stack(tuple(body.buffers.logits[index] for index in physical))
        if not bool(torch.isfinite(output).all().item()):
            raise RuntimeError("qwen decode graph produced non-finite logical logits")
        self.state.accept_and_verify()
        self.output = output.clone()
        self.consumed = True
        return {"logits": self.output}

    def commit(self) -> None:
        if not self.consumed or self.finished:
            raise RuntimeError("qwen decode state commits only after durable output consumption")
        self.state.commit()
        self.finished = True

    def abort(self) -> None:
        if self.finished:
            return
        self.slot.synchronize()
        if self.state.status not in ("committed", "aborted"):
            self.state.abort()
        self.finished = True


class Qwen3_5DecodeGraphProgram:
    """Single-rank composite decode program; admission is graph-only and fail-closed."""

    program_id = PROGRAM_ID
    replay_abi_id = REPLAY_ABI_ID
    state_path_id = STATE_PATH_ID

    def __init__(self, model: Qwen3_5ReferenceModel, state: DecodeState) -> None:
        self.model, self.state = model, state
        self._binding: GraphRuntimeBinding | None = None
        self._slots: set[int] = set()
        self._mrope_tables: _MRoPETables | None = None

    @property
    def child_operator_closure(self) -> _ChildOperatorClosure:
        """Diagnostic semantic+shape projection; never sufficient for admission."""

        return _child_operator_closure(self.model.config.graph.text, self.state)

    def _primitive_program_manifest(
        self,
    ) -> tuple[tuple[ec.OpBinding, ...], ec.CompositeProgramManifest]:
        """Future exporter must derive this from body/config/plans/sealed weights, never callers."""

        raise CudaGraphRejected(
            "qwen decode primitives do not yet export a complete occurrence manifest"
        )

    def validate_binding(self, binding: GraphRuntimeBinding) -> None:
        self._validate_structure(binding)
        seal, plan = binding.execution_plan.operator_evidence, binding.execution_plan
        if type(seal) is not ec.OperatorEvidenceSeal or not ec.validate_operator_evidence(
            seal, plan.execution_plan_id, plan.bindings, plan.graph_buckets
        ):
            raise CudaGraphRejected("qwen decode requires complete typed arithmetic evidence")
        if (
            len(seal.arithmetic) != 1
            or type(seal.arithmetic[0]) is not ec.CompositeArithmeticEvidenceSeal
        ):
            raise CudaGraphRejected(
                "ordinary leaf arithmetic evidence cannot seal the qwen composite child closure"
            )
        composite = seal.arithmetic[0]
        children, manifest = self._primitive_program_manifest()
        if not ec.validate_composite_arithmetic_evidence(
            composite, plan.bindings[0], children, manifest
        ):
            raise CudaGraphRejected("qwen composite evidence differs from its primitive manifest")
        self._binding = binding

    def _validate_structure(self, binding: GraphRuntimeBinding) -> None:
        config, text = self.model.config, self.model.config.graph.text
        topology, partition, bucket = binding.topology, binding.partition, binding.bucket
        if (
            config.identity.model_type != "qwen3_5"
            or config.graph.variant is not Qwen3_5GraphVariant.DENSE
            or (topology.tp_size, topology.pp_size, topology.ep_size) != (1, 1, 1)
            or topology.world_size != 1
            or bucket.phase is not ec.BatchPhase.DECODE
        ):
            raise CudaGraphRejected("qwen dense decode performance supports one rank only")
        if self.state.device != self.model.weights.device:
            raise CudaGraphRejected("qwen model and physical state devices differ")
        if (
            self.model.weights.device.type == "cuda"
            and (self.model.weights.device.index or 0) != topology.placements[0].device_index
        ):
            raise CudaGraphRejected("qwen graph topology names a different CUDA device")
        architecture = config.architecture_plan()
        nodes = {
            architecture.token_embedding_node_id,
            *(layer.node_id for layer in architecture.decoder_layers),
            architecture.final_norm_node_id,
            architecture.lm_head_node_id,
        }
        if (
            {
                (item.node_id, item.pipeline_stage, item.owner_ranks)
                for item in partition.assignments
            }
            != {(node, 0, (0,)) for node in nodes}
            or partition.collectives
            or partition.pipeline_edges
            or partition.expert_assignments
        ):
            raise CudaGraphRejected("partition does not exactly close the single-rank text graph")
        expected_capacity = _capacity_items(self.state, text, bucket.captured_m)
        if bucket.capacity_items != expected_capacity:
            raise CudaGraphRejected("qwen graph bucket has a different fixed capacity profile")
        if (
            bucket.row_map_schema_id != ROW_MAP_SCHEMA_ID
            or bucket.scratch_schema_id != SCRATCH_SCHEMA_ID
            or bucket.graph_safety_contract_ids != (GRAPH_SAFETY_CONTRACT_ID,)
            or bucket.replay_abi_id != REPLAY_ABI_ID
        ):
            raise CudaGraphRejected("qwen graph bucket differs from its replay safety ABI")
        bindings = binding.execution_plan.bindings
        expected_signature = _decode_signature(text, topology.topology_id)
        if len(bindings) != 1 or bindings[0].signature != expected_signature:
            raise CudaGraphRejected("execution plan does not bind the exact qwen decode signature")
        candidate = bindings[0].candidate
        if (
            candidate is None
            or candidate.candidate_id != CANDIDATE_ID
            or candidate.implementation_id != PROGRAM_ID
            or candidate.backend is not ec.CandidateBackend.CUDA
            or candidate.launch_signature != REPLAY_ABI_ID
            or candidate.workspace_bytes != 0
            or candidate.graph_safety is None
            or candidate.graph_safety.contract_id != GRAPH_SAFETY_CONTRACT_ID
            or candidate.graph_safety.row_map_schema_id != ROW_MAP_SCHEMA_ID
        ):
            raise CudaGraphRejected("candidate identity does not close the qwen decode program")

    def capture(
        self,
        binding: GraphRuntimeBinding,
        resources: tuple[ec.RankReplayResources, ...],
    ) -> GraphProgramSlot:
        if binding is not self._binding or len(resources) != 1 or resources[0].rank != 0:
            raise CudaGraphRejected("qwen capture received a foreign binding or rank set")
        resource = resources[0]
        if resource.scratch_arena_id not in self.state.arena.spec.scratch_arena_ids:
            raise CudaGraphRejected("qwen capture resource has no physical scratch arena")
        if self.state.arena.spec.scratch_capacity < binding.bucket.captured_m:
            raise CudaGraphRejected("qwen capture scratch capacity is smaller than captured M")
        if self._mrope_tables is None:
            text = self.model.config.graph.text
            rotary = int(text.full_attention.head_dim * text.mrope.partial_rotary_factor)
            self._mrope_tables = build_bf16_mrope_tables(
                max_position=text.max_position_embeddings,
                device=self.model.weights.device,
                rotary_dim=rotary,
                section=text.mrope.section,
                rope_theta=text.mrope.rope_theta,
            )
        slot = Qwen3_5DecodeGraphSlot.capture(
            model=self.model,
            arena=self.state.arena,
            resource=resource,
            captured_m=binding.bucket.captured_m,
            table_pages=self.state.schema.max_pages,
            mrope_tables=self._mrope_tables,
        )
        self._slots.add(id(slot))
        return slot

    def qualification_state_factory(
        self,
        slot: GraphProgramSlot,
        plan: ec.LaunchPlan,
        intent: ReplayIntent,
    ) -> gp.GraphQualificationStateFactory:
        self._validate_slot_launch(slot, plan, intent)
        return QualificationFactory(self.state, intent.batch_plan)

    def fill(
        self,
        slot: GraphProgramSlot,
        plan: ec.LaunchPlan,
        intent: ReplayIntent,
        logical_inputs: Mapping[str, Tensor],
        state: object,
        padding: gp.GraphPaddingBuffers | None,
    ) -> GraphProgramSession:
        typed_slot = self._validate_slot_launch(slot, plan, intent)
        tokens, positions, pad_tokens, pad_positions, guard = self._validate_inputs(
            plan, intent, logical_inputs, padding
        )
        if isinstance(state, QualificationRun):
            if state.state is not self.state:
                raise CudaGraphRejected("qualification uses a foreign qwen state arena")
            decode = state.begin(intent.batch_plan, plan)
        elif state is self.state:
            decode = self.state.begin(intent.batch_plan, plan)
        else:
            raise CudaGraphRejected("qwen replay requires its production physical state")
        try:
            self._fill_static(
                typed_slot,
                plan,
                decode,
                tokens,
                positions,
                pad_tokens,
                pad_positions,
                guard,
                padding,
            )
            expected_guard = (
                _CANARY if guard is None else (int(guard[0].item()), int(guard[1].item()))
            )
            return _QwenDecodeGraphSession(typed_slot, plan, decode, expected_guard)
        except BaseException:
            try:
                typed_slot.synchronize()
            finally:
                decode.abort()
            raise

    def _validate_slot_launch(
        self, slot: GraphProgramSlot, plan: ec.LaunchPlan, intent: ReplayIntent
    ) -> Qwen3_5DecodeGraphSlot:
        if (
            not isinstance(slot, Qwen3_5DecodeGraphSlot)
            or id(slot) not in self._slots
            or self._binding is None
            or plan.bucket != self._binding.bucket
            or plan.source_plan_id != intent.batch_plan.plan_id
            or plan.rows != intent.rows
            or plan.replay_lease.rank_resources != slot.resources
        ):
            raise CudaGraphRejected("qwen replay slot, launch, or immutable intent differs")
        return slot

    def _validate_inputs(
        self,
        plan: ec.LaunchPlan,
        intent: ReplayIntent,
        inputs: Mapping[str, Tensor],
        padding: gp.GraphPaddingBuffers | None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]:
        if set(inputs) != {"position_ids", "token_ids"}:
            raise CudaGraphRejected("qwen decode inputs require only token_ids and position_ids")
        tokens, positions = inputs["token_ids"], inputs["position_ids"]
        logical_m = plan.logical_m
        segments = intent.batch_plan.segments
        axes = tuple(segment.positions[0].axes for segment in segments)
        expected_tokens = tuple(segment.token_ids[0] for segment in segments)
        axis_count = len(axes[0]) if axes else 0
        expected_positions = (
            tuple(axis[0] for axis in axes)
            if axis_count == 1
            else tuple(tuple(axis[row] for axis in axes) for row in range(3))
            if axis_count == 3
            else ()
        )
        if (
            len(segments) != logical_m
            or any(segment.token_count != 1 for segment in segments)
            or any(len(axis) != axis_count for axis in axes)
            or axis_count not in (1, 3)
            or not _host_tensor(tokens, torch.int64, (logical_m,))
            or not _host_tensor(
                positions, torch.int64, (logical_m,) if axis_count == 1 else (3, logical_m)
            )
            or tuple(tokens.tolist()) != expected_tokens
            or _nested_tuple(positions.tolist()) != expected_positions
        ):
            raise CudaGraphRejected("decode tensors differ from immutable token/position plans")
        text = self.model.config.graph.text
        if (
            min(expected_tokens) < 0
            or max(expected_tokens) >= text.vocabulary_size
            or any(not 0 <= value < text.max_position_embeddings for axis in axes for value in axis)
        ):
            raise CudaGraphRejected("decode token or position is outside captured model bounds")
        expanded = positions[None, :].expand(3, -1) if axis_count == 1 else positions
        padding_count = plan.captured_m - logical_m
        if padding is None:
            return tokens, expanded, None, None, None
        if set(padding.guards) != {"canary"}:
            raise CudaGraphRejected("qwen padding guard schema differs from capture")
        guard = padding.guards["canary"]
        if not _host_tensor(guard, torch.int64, (2,)):
            raise CudaGraphRejected("qwen padding canary differs from its static ABI")
        if padding_count == 0:
            if padding.padding_inputs or padding.scratch_state:
                raise CudaGraphRejected("full-M replay cannot carry padding tensors")
            return tokens, expanded, None, None, guard
        if set(padding.padding_inputs) != {"position_ids", "token_ids"}:
            raise CudaGraphRejected("qwen padding input schema is incomplete")
        pad_tokens, pad_positions = (
            padding.padding_inputs["token_ids"],
            padding.padding_inputs["position_ids"],
        )
        if (
            not _host_tensor(pad_tokens, torch.int64, (padding_count,))
            or not _host_tensor(pad_positions, torch.int64, (3, padding_count))
            or bool((pad_tokens < 0).any().item())
            or bool((pad_tokens >= text.vocabulary_size).any().item())
            or bool((pad_positions < 0).any().item())
            or bool((pad_positions >= text.max_position_embeddings).any().item())
        ):
            raise CudaGraphRejected("padding tokens and positions must be legal captured inputs")
        return tokens, expanded, pad_tokens, pad_positions, guard

    def _fill_static(
        self,
        slot: Qwen3_5DecodeGraphSlot,
        plan: ec.LaunchPlan,
        state: DecodeSession,
        tokens: Tensor,
        positions: Tensor,
        pad_tokens: Tensor | None,
        pad_positions: Tensor | None,
        guard: Tensor | None,
        padding: gp.GraphPaddingBuffers | None,
    ) -> None:
        body, host, arena = slot.body, slot.host, slot.body.arena
        padding_count = plan.captured_m - plan.logical_m
        if len(state.rows) != plan.captured_m:
            raise CudaGraphRejected("physical state rows differ from captured M")
        if padding_count:
            _validate_scratch_padding(arena, padding, padding_count)
        for target, value in zip(host.guard, _CANARY if guard is None else guard, strict=True):
            target.fill_(value)
        layers = len(body.gdn_layers)
        for physical, (launch_row, row) in enumerate(zip(plan.rows, state.rows, strict=True)):
            if isinstance(launch_row, ec.ActiveLaunchRow):
                logical = launch_row.logical_index
                if row.logical_index != logical or row.position is None:
                    raise CudaGraphRejected("decode state row changed the launch row map")
                host.token_ids[physical].copy_(tokens[logical])
                host.position_ids[:, physical].copy_(positions[:, logical])
            else:
                if row.logical_index is not None or row.position is not None:
                    raise CudaGraphRejected("padding acquired request identity")
                index = launch_row.scratch_index
                host.token_ids[physical].fill_(0 if pad_tokens is None else pad_tokens[index])
                host.position_ids[:, physical].fill_(0)
                if pad_positions is not None:
                    host.position_ids[:, physical].copy_(pad_positions[:, index])
            host.page_source[physical].fill_(row.page_cow[0])
            host.page_destination[physical].fill_(row.page_cow[1])
            host.token_offsets[physical].fill_(row.token_offset)
            for target, page in zip(host.page_table[physical], row.page_table, strict=True):
                target.fill_(page)
            host.sequence_lengths[physical].fill_(row.sequence_length)
            for state_kind, target in enumerate((host.conv_slots, host.recurrent_slots)):
                for local in range(layers):
                    for cow, index in zip(row.state_cow[state_kind], (0, 1), strict=True):
                        target[index, local, physical].fill_(cow * layers + local)
        slot.stream.wait_stream(torch.cuda.current_stream(arena.device))
        with torch.cuda.device(arena.device), torch.cuda.stream(slot.stream):
            host.copy_to(body.buffers)
            if padding_count and padding is not None:
                _copy_padding_scratch(state, padding.scratch_state, padding_count)


def _copy_padding_scratch(session: DecodeSession, values: Mapping[str, Tensor], count: int) -> None:
    """Bulk-copy qualification poison into this launch's private scratch."""

    binding = session.bindings[0]
    fields = (
        "arena_id",
        "scratch_arena_id",
        "scratch_lease_id",
        "scratch_lease_epoch",
        "scratch_launch_plan_id",
        "scratch_page_slots",
        "scratch_state_slots",
    )
    identity = tuple(getattr(binding, name) for name in fields)
    if any(
        tuple(getattr(item, name) for name in fields) != identity for item in session.bindings[1:]
    ):
        raise CudaGraphRejected("decode rows do not share one leased scratch mapping")

    page_slots = binding.scratch_page_slots[:count]
    state_slots = dict(binding.scratch_state_slots)
    if not page_slots or page_slots != tuple(range(page_slots[0], page_slots[0] + count)):
        raise CudaGraphRejected("padding KV scratch is not one contiguous leased slab")
    page_slice = slice(page_slots[0], page_slots[0] + count)
    binding.key[page_slice].copy_(values["kv_key"], non_blocking=True)
    binding.value[page_slice].copy_(values["kv_value"], non_blocking=True)
    binding.written[page_slice].copy_(values["kv_written"], non_blocking=True)
    tensors = dict(binding.state_tensors)
    for state_id in (CONV_STATE_ID, RECURRENT_STATE_ID):
        slots = state_slots[state_id][:count]
        if not slots or slots != tuple(range(slots[0], slots[0] + count)):
            raise CudaGraphRejected("padding recurrent scratch is not one contiguous leased slab")
        tensors[state_id][slots[0] : slots[0] + count].copy_(values[state_id], non_blocking=True)


def _capacity_items(
    state: DecodeState, text: Qwen3_5TextConfig, captured_m: int
) -> ec.CapacityItems:
    # Attribute access is intentionally centralized so the exact arena/program
    # allocation profile cannot drift from dispatch independently.
    return tuple(
        sorted(
            {
                "attention_layers": len(state.schema.kv_layout.layer_ids),
                "block_tokens": state.schema.kv_layout.block_size_tokens,
                "context_pages": state.schema.max_pages,
                "gdn_layers": len(state.schema.gdn_layer_ids),
                "hidden_size": text.hidden_size,
                "max_position": text.max_position_embeddings,
                "page_capacity": state.arena.spec.storage_page_capacity,
                "request_slots": captured_m,
                "scratch_capacity": state.arena.spec.scratch_capacity,
                "state_capacity": state.arena.spec.state_slot_capacity,
                "vocabulary_size": text.vocabulary_size,
            }.items()
        )
    )


def _decode_signature(text: Qwen3_5TextConfig, topology_id: str) -> ec.OpSignature:
    return ec.OpSignature(
        SEMANTIC_OP,
        ec.BatchPhase.DECODE,
        (
            ec.TensorSpec("token_ids", ScalarDType.INT64, ("logical_m",)),
            ec.TensorSpec("position_ids", ScalarDType.INT64, (3, "logical_m")),
        ),
        (ec.TensorSpec("logits", ScalarDType.BF16, ("logical_m", text.vocabulary_size)),),
        ScalarDType.FP32,
        _LAYOUT,
        (),
        _CACHE_FORM,
        _STATE_FORM,
        topology_id,
    )


def _child_operator_closure(text: Qwen3_5TextConfig, state: DecodeState) -> _ChildOperatorClosure:
    """Describe leaves only; this is not evidence and cannot admit a graph."""

    attention, gdn, intermediate = (
        text.full_attention,
        text.gated_delta_net,
        text.dense_intermediate_size,
    )
    if intermediate is None:
        raise Qwen3_5PerformanceRejected("dense child closure requires dense MLP dimensions")

    hidden, vocabulary = text.hidden_size, text.vocabulary_size
    query, kv, head = (
        attention.query_heads,
        attention.key_value_heads,
        attention.head_dim,
    )
    q_width, kv_width = query * head, kv * head
    gdn_width = gdn.value_heads * gdn.value_head_dim
    packed = 2 * gdn.key_heads * gdn.key_head_dim + gdn_width
    block, pages = state.schema.kv_layout.block_size_tokens, state.schema.max_pages
    raw: tuple[tuple[str, str, tuple[int, ...]], ...] = (
        (
            "attention.paged_gqa.decode.bf16",
            "block head kv pages query",
            (block, head, kv, pages, query),
        ),
        (
            "cache.page_cow.bf16",
            "block head kv layers",
            (block, head, kv, len(state.schema.kv_layout.layer_ids)),
        ),
        ("cache.page_token_write.bf16", "block head kv", (block, head, kv)),
        ("embedding.token.bf16", "hidden vocabulary", (hidden, vocabulary)),
        (
            "linear_attention.causal_conv1d.decode.bf16",
            "kernel width",
            (gdn.convolution_kernel_size, packed),
        ),
        (
            "linear_attention.gated_delta.decode",
            "key_dim key_heads value_dim value_heads",
            (gdn.key_head_dim, gdn.key_heads, gdn.value_head_dim, gdn.value_heads),
        ),
        ("normalization.gated_rms.bf16", "heads width", (gdn.value_heads, gdn.value_head_dim)),
        ("normalization.zero_centered_rms.bf16", "width", (head,)),
        ("normalization.zero_centered_rms.bf16", "width", (hidden,)),
        ("pointwise.attention_q_gate_split.bf16", "head query", (head, query)),
        ("pointwise.residual_add.bf16", "width", (hidden,)),
        ("pointwise.sigmoid_output_gate.bf16", "head query", (head, query)),
        ("pointwise.swiglu.bf16", "width", (intermediate,)),
        ("position.mrope_apply.bf16", "head kv query", (head, kv, query)),
        (
            "position.mrope_gather.bf16",
            "axes maximum rotary",
            (3, text.max_position_embeddings, int(head * text.mrope.partial_rotary_factor)),
        ),
    )
    linear_shapes = {
        (hidden, packed),
        (hidden, gdn.value_heads),
        (hidden, gdn_width),
        (gdn_width, hidden),
        (hidden, 2 * q_width),
        (hidden, kv_width),
        (q_width, hidden),
        (hidden, intermediate),
        (intermediate, hidden),
        (hidden, vocabulary),
    }
    raw += tuple(("linear.bf16", "input output", pair) for pair in linear_shapes)
    return tuple(
        sorted(
            {
                (semantic, tuple(sorted(zip(names.split(), values, strict=True))))
                for semantic, names, values in raw
            }
        )
    )


def _validate_scratch_padding(
    arena: CacheArena, padding: gp.GraphPaddingBuffers | None, count: int
) -> None:
    if padding is None:
        return
    values = padding.scratch_state
    specs = arena.state_specs
    page = (count, *arena.layout.page_shape)
    schema = {
        "kv_key": (page, torch.bfloat16),
        "kv_value": (page, torch.bfloat16),
        "kv_written": (
            (count, len(arena.layout.layer_ids), arena.layout.block_size_tokens),
            torch.bool,
        ),
        CONV_STATE_ID: ((count, *specs[CONV_STATE_ID].shape), specs[CONV_STATE_ID].dtype),
        RECURRENT_STATE_ID: (
            (count, *specs[RECURRENT_STATE_ID].shape),
            specs[RECURRENT_STATE_ID].dtype,
        ),
    }
    if set(values) != set(schema) or any(
        value.shape != shape
        or value.dtype is not dtype
        or value.device != arena.device
        or not value.is_contiguous()
        or (value.is_floating_point() and not bool(torch.isfinite(value).all().item()))
        for name, (shape, dtype) in schema.items()
        for value in (values[name],)
    ):
        raise CudaGraphRejected("qwen padding scratch differs from its physical arena ABI")


def _nested_tuple(value: object) -> object:
    return tuple(_nested_tuple(item) for item in value) if isinstance(value, list) else value


def _host_tensor(value: Tensor, dtype: torch.dtype, shape: tuple[int, ...]) -> bool:
    return (
        value.device.type == "cpu"
        and value.dtype is dtype
        and value.shape == shape
        and value.is_contiguous()
    )


class Qwen3_5ParallelPerformanceError(RuntimeError):
    """The TP2 x PP2 body or capture differs from its fixed rank-local ABI."""


def minimum_kv_page_capacity(table_pages: int, active_owners: int) -> int:
    """Cover one full sequence plus source/destination COW for active rows.

    The page table describes addressability, not a per-row physical reservation.
    Physical KV pages are shared across requests and consumed as tokens arrive.
    """

    if (
        type(table_pages) is not int
        or table_pages <= 0
        or type(active_owners) is not int
        or active_owners <= 0
    ):
        raise ValueError("KV capacity requires positive table pages and active owners")
    return table_pages + 2 * active_owners


@dataclass(frozen=True, slots=True)
class Qwen3_5LinearLeafEvent:
    """One eager-only BF16 linear leaf, valid only for its callback duration."""

    global_layer: int | None
    weight_name: str
    weight_shape: tuple[int, ...]
    artifact_digest: str
    launch_specialization_digest: str
    linear_input: Tensor
    local_weight: Tensor
    local_output: Tensor
    tp_partial: Tensor | None
    tp_reduced: Tensor | None


Qwen3_5LinearObserver = TypeAliasType("Qwen3_5LinearObserver", Callable[[Qwen3_5LinearLeafEvent], None])
Qwen3_5LayerObserver = TypeAliasType("Qwen3_5LayerObserver", Callable[[int, str, Tensor], None])


class _StageWeights:
    """Translate stage-local names and preserve semantic views after weight merging."""

    __slots__ = (
        "_layer_start",
        "_moe",
        "_pp_size",
        "_source",
        "_stage",
        "device",
        "inventory_digest",
        "semantic_plan_digest",
    )

    def __init__(
        self,
        source: Qwen3_5RankWeights,
        *,
        stage: int,
        pp_size: int,
        layer_start: int,
        semantic_plan_digest: str,
    ) -> None:
        self._source, self._stage, self._layer_start = source, stage, layer_start
        self._pp_size = pp_size
        self._moe = any(
            item.tensor_name.endswith("mlp.experts.gate_up_proj") for item in source.bindings
        )
        self.device, self.inventory_digest = source.device, source.inventory_digest
        self.semantic_plan_digest = semantic_plan_digest

    def _source_name(self, name: str) -> str:
        prefix = "model.language_model.layers."
        if name.startswith(prefix):
            local, separator, tail = name[len(prefix) :].partition(".")
            if not separator:
                raise KeyError(name)
            name = f"{prefix}{int(local) + self._layer_start}.{tail}"
        elif self._pp_size > 1 and self._stage == 0 and name == "model.language_model.norm.weight":
            name = f"{prefix}{self._layer_start}.input_layernorm.weight"
        elif self._pp_size > 1 and self._stage == 0 and name == "lm_head.weight":
            name = "model.language_model.embed_tokens.weight"
        elif (
            self._pp_size > 1
            and self._stage == self._pp_size - 1
            and name == "model.language_model.embed_tokens.weight"
        ):
            name = "lm_head.weight"
        if self._moe:
            for leaf in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
                suffix = f".mlp.{leaf}"
                if name.endswith(suffix):
                    name = f"{name[: -len(suffix)]}.mlp.shared_expert.{leaf}"
                    break
        return name

    def __getitem__(self, name: str) -> Tensor:
        return self._source[self._source_name(name)]

    def rebind_semantic_view(self, name: str, value: Tensor) -> None:
        """Replace one semantic tensor with an equal-shape view into merged storage."""

        source_name = self._source_name(name)
        current = self._source[source_name]
        if (
            value.shape != current.shape
            or value.dtype is not current.dtype
            or value.device != current.device
            or not value.is_contiguous()
        ):
            raise Qwen3_5ParallelPerformanceError(
                f"merged view differs from semantic tensor {source_name!r}"
            )
        self._source._tensors[source_name] = value

    @property
    def tensor_count(self) -> int:
        return self._source.tensor_count

    def named_tensors(self) -> tuple[tuple[str, Tensor], ...]:
        return self._source.named_tensors()


def _pack_gate_up_weights(weights: _StageWeights, layer_count: int) -> Tensor:
    """Merge local gate/up rows and rebind their semantic names to storage views."""

    if type(layer_count) is not int or layer_count <= 0:
        raise Qwen3_5ParallelPerformanceError("packed gate/up requires local layers")
    first = "model.language_model.layers.0.mlp"
    gate = weights[f"{first}.gate_proj.weight"]
    up = weights[f"{first}.up_proj.weight"]
    if gate.ndim != 2 or gate.shape != up.shape or min(gate.shape) <= 0:
        raise Qwen3_5ParallelPerformanceError("gate/up weights differ from the packed BF16 ABI")
    width, hidden = gate.shape
    packed = torch.empty(
        (layer_count, 2 * width, hidden), dtype=torch.bfloat16, device=weights.device
    )
    for layer in range(layer_count):
        prefix = f"model.language_model.layers.{layer}.mlp"
        gate = weights[f"{prefix}.gate_proj.weight"]
        up = weights[f"{prefix}.up_proj.weight"]
        if gate.shape != (width, hidden) or up.shape != (width, hidden):
            raise Qwen3_5ParallelPerformanceError(
                f"layer {layer} gate/up weights differ from the packed BF16 ABI"
            )
        packed[layer, :width].copy_(gate)
        packed[layer, width:].copy_(up)
    for layer in range(layer_count):
        prefix = f"model.language_model.layers.{layer}.mlp"
        weights.rebind_semantic_view(f"{prefix}.gate_proj.weight", packed[layer, :width])
        weights.rebind_semantic_view(f"{prefix}.up_proj.weight", packed[layer, width:])
    return packed


def _pack_gdn_abz_weights(
    weights: _StageWeights, layers: tuple[int, ...]
) -> tuple[Tensor, tuple[int, int, int]]:
    """Merge local GDN A/B/Z rows and rebind semantic names to storage views."""

    if not layers or any(type(layer) is not int or layer < 0 for layer in layers):
        raise Qwen3_5ParallelPerformanceError("packed GDN A/B/Z requires local layers")
    suffixes = ("in_proj_a.weight", "in_proj_b.weight", "in_proj_z.weight")
    first = f"model.language_model.layers.{layers[0]}.linear_attn"
    first_weights = tuple(weights[f"{first}.{suffix}"] for suffix in suffixes)
    if any(
        weight.ndim != 2 or min(weight.shape) <= 0 or weight.dtype is not torch.bfloat16
        for weight in first_weights
    ):
        raise Qwen3_5ParallelPerformanceError("GDN A/B/Z weights differ from the packed BF16 ABI")
    hidden = first_weights[0].shape[1]
    widths = cast(tuple[int, int, int], tuple(weight.shape[0] for weight in first_weights))
    if any(weight.shape[1] != hidden for weight in first_weights):
        raise Qwen3_5ParallelPerformanceError("GDN A/B/Z weights have different input widths")
    packed = torch.empty(
        (len(layers), sum(widths), hidden), dtype=torch.bfloat16, device=weights.device
    )
    for ordinal, layer in enumerate(layers):
        prefix = f"model.language_model.layers.{layer}.linear_attn"
        offset = 0
        for suffix, width in zip(suffixes, widths, strict=True):
            weight = weights[f"{prefix}.{suffix}"]
            if weight.shape != (width, hidden) or weight.dtype is not torch.bfloat16:
                raise Qwen3_5ParallelPerformanceError(
                    f"layer {layer} GDN A/B/Z weights differ from the packed BF16 ABI"
                )
            packed[ordinal, offset : offset + width].copy_(weight)
            offset += width
    for ordinal, layer in enumerate(layers):
        prefix = f"model.language_model.layers.{layer}.linear_attn"
        offset = 0
        for suffix, width in zip(suffixes, widths, strict=True):
            weights.rebind_semantic_view(
                f"{prefix}.{suffix}", packed[ordinal, offset : offset + width]
            )
            offset += width
    return packed, widths


@dataclass(slots=True)
class _ArenaView:
    """Expose stage-local physical layer ordinals to the reused single-rank body."""

    _initial_spec: CacheArenaSpec
    layout: KVLayout
    device: torch.device
    rank_id: int
    key: Tensor
    value: Tensor
    written: Tensor
    states: dict[str, Tensor]
    owner: CacheArena | None = None

    @property
    def spec(self) -> CacheArenaSpec:
        return self._initial_spec if self.owner is None else self.owner.spec

    @property
    def page_fields(self) -> dict[str, Tensor]:
        return {} if self.owner is None else self.owner.page_fields


@dataclass(slots=True)
class _CacheRuntimeBinding:
    """Fully constructed cache-bound state, ready for a no-fail attribute commit."""

    arena: _ArenaView
    conv_state: Tensor
    recurrent_state: Tensor
    page_copy: PageCOWCopyPrimitive
    conv: CausalConv1DDecodePrimitive
    gdn: GatedDeltaDecodePrimitive
    page_writes: tuple[PageTokenWritePrimitive, ...]
    attentions: tuple[BF16PagedGQADecodePrimitive, ...]


_CACHE_RUNTIME_ATTRIBUTES = (
    "arena",
    "conv_state",
    "recurrent_state",
    "page_copy",
    "conv",
    "gdn",
    "page_writes",
    "attentions",
)


def _build_cache_runtime(body: _DecodeBody, arena: _ArenaView) -> _CacheRuntimeBinding:
    """Construct every cache-address-bound object without mutating ``body``."""

    text = body.model.config.graph.text
    gdn = text.gated_delta_net
    buffers, weights = body.buffers, body.model.weights
    packed = 2 * gdn.key_heads * gdn.key_head_dim + gdn.value_heads * gdn.value_head_dim
    conv_state = (arena.states[CONV_STATE_ID].reshape(-1, packed, gdn.convolution_kernel_size - 1)
                  if arena.states[CONV_STATE_ID].is_contiguous() else arena.states[CONV_STATE_ID])
    recurrent_state = (arena.states[RECURRENT_STATE_ID].reshape(
        -1, gdn.value_heads, gdn.value_head_dim, gdn.key_head_dim
    ) if arena.states[RECURRENT_STATE_ID].is_contiguous() else arena.states[RECURRENT_STATE_ID])
    first_gdn = f"model.language_model.layers.{body.gdn_layers[0]}.linear_attn"
    page_copy = PageCOWCopyPrimitive(
        arena.key,
        arena.value,
        arena.written,
        buffers.page_source,
        buffers.page_destination,
    )
    conv = CausalConv1DDecodePrimitive(
        buffers.gdn_packed,
        weights[f"{first_gdn}.conv1d.weight"].squeeze(1),
        conv_state,
        buffers.conv_slots[0, 0],
        buffers.conv_slots[1, 0],
        buffers.segment_lengths,
        buffers.gdn_convolved,
        checkpoint_destinations=True,
    )
    gdn_decode = GatedDeltaDecodePrimitive(
        buffers.gdn_convolved,
        buffers.gdn_packed,
        buffers.gdn_a,
        buffers.gdn_b,
        weights[f"{first_gdn}.A_log"],
        weights[f"{first_gdn}.dt_bias"],
        recurrent_state,
        buffers.recurrent_slots[0, 0],
        buffers.recurrent_slots[1, 0],
        buffers.segment_lengths,
        buffers.gdn_core,
        q_heads=gdn.key_heads,
        scale=gdn.key_head_dim**-0.5,
        checkpoint_destinations=True,
        row_activity=buffers.row_activity,
    )
    page_writes = tuple(
        PageTokenWritePrimitive(
            arena.key,
            arena.value,
            arena.written,
            buffers.page_destination,
            buffers.token_offsets,
            buffers.rotated_key,
            buffers.value,
            layer=local,
        )
        for local in range(len(body.full_layers))
    )
    attentions = tuple(
        BF16PagedGQADecodePrimitive(
            buffers.rotated_query,
            arena.key,
            arena.value,
            buffers.page_table,
            buffers.sequence_lengths,
            buffers.attention_output,
            layer=local,
            partial=buffers.attention_partial,
            statistics=buffers.attention_statistics,
            arrivals=buffers.attention_arrivals,
        )
        for local in range(len(body.full_layers))
    )
    return _CacheRuntimeBinding(
        arena,
        conv_state,
        recurrent_state,
        page_copy,
        conv,
        gdn_decode,
        page_writes,
        attentions,
    )


def _commit_cache_runtime(body: _DecodeBody, binding: _CacheRuntimeBinding) -> None:
    """Publish one prebuilt binding; plain assignments below cannot partially fail."""

    body.arena = cast(CacheArena, binding.arena)
    body.conv_state = binding.conv_state
    body.recurrent_state = binding.recurrent_state
    body.page_copy = binding.page_copy
    body.conv = binding.conv
    body.gdn = binding.gdn
    body.page_writes = binding.page_writes
    body.attentions = binding.attentions


def _drop_cache_runtime(body: _DecodeBody) -> None:
    """Release every reference to bootstrap cache backing, retaining private tensors."""

    for name in _CACHE_RUNTIME_ATTRIBUTES:
        delattr(body, name)


def _local_config(
    config: Qwen3_5Config, *, stage: int, tp_size: int, pp_size: int
) -> Qwen3_5Config:
    text = config.graph.text
    intermediate = (
        text.dense_intermediate_size
        if text.moe is None
        else text.moe.shared_expert_intermediate_size
    )
    divisible = (
        text.layer_count == len(text.layer_schedule)
        and 0 <= stage < pp_size
        and text.layer_count % pp_size == 0
        and intermediate is not None
        and intermediate % tp_size == 0
        and text.vocabulary_size % tp_size == 0
        and text.full_attention.query_heads % tp_size == 0
        and (
            text.full_attention.key_value_heads % tp_size == 0
            or tp_size % text.full_attention.key_value_heads == 0
        )
        and text.gated_delta_net.key_heads % tp_size == 0
        and text.gated_delta_net.value_heads % tp_size == 0
    )
    if not divisible:
        raise Qwen3_5ParallelPerformanceError("Qwen config does not close the TP/PP slice")
    assert intermediate is not None
    layers = text.layer_count // pp_size
    start = stage * layers
    local_text = replace(
        text,
        vocabulary_size=text.vocabulary_size // tp_size,
        layer_count=layers,
        layer_schedule=text.layer_schedule[start : start + layers],
        dense_intermediate_size=intermediate // tp_size,
        full_attention=replace(
            text.full_attention,
            query_heads=text.full_attention.query_heads // tp_size,
            key_value_heads=max(1, text.full_attention.key_value_heads // tp_size),
        ),
        gated_delta_net=replace(
            text.gated_delta_net,
            key_heads=text.gated_delta_net.key_heads // tp_size,
            value_heads=text.gated_delta_net.value_heads // tp_size,
        ),
    )
    return replace(config, graph=replace(config.graph, text=local_text))


def _group_ranks(group: ProcessGroup, expected: tuple[int, ...], label: str) -> None:
    if not expected or tuple(dist.get_process_group_ranks(group)) != expected:
        raise Qwen3_5ParallelPerformanceError(f"{label} group differs from rank placement")


class Qwen3_5TP2PP2DecodeBody:
    """One fixed captured-M activation lane with native TP/PP communication."""

    def __init__(
        self,
        *,
        config: Qwen3_5Config,
        parallel_plan: Qwen3_5ParallelWeightPlan,
        weights: Qwen3_5RankWeights,
        tp_group: ProcessGroup,
        pp_group: ProcessGroup | None,
        captured_m: int,
        table_pages: int,
        page_capacity: int,
        state_capacity: int,
        scratch_arena_id: str,
        scratch_capacity: int | None = None,
        recurrent_state_dtype: torch.dtype = torch.float32,
        mrope_tables: _MRoPETables | None = None,
        packed_gate_up_weights: Tensor | None = None,
        packed_gdn_abz: tuple[Tensor, tuple[int, int, int]] | None = None,
    ) -> None:
        if not dist.is_initialized():
            raise Qwen3_5ParallelPerformanceError("NCCL process groups are not initialized")
        placement = next(
            (item for item in parallel_plan.topology.placements if item.rank == weights.rank),
            None,
        )
        topology = parallel_plan.topology
        resolved_scratch_capacity = captured_m if scratch_capacity is None else scratch_capacity
        if (
            placement is None
            or topology.tp_size % topology.ep_size
            or topology.pp_size not in (1, 2)
            or dist.get_world_size() != topology.world_size
            or dist.get_rank() != weights.rank
            or weights.device.type != "cuda"
            or torch.cuda.current_device() != weights.device.index
            or placement.device_index != weights.device.index
            or captured_m < 1
            or captured_m & (captured_m - 1)
            or min(table_pages, page_capacity, state_capacity) <= 0
            or page_capacity < minimum_kv_page_capacity(table_pages, captured_m)
            or state_capacity < captured_m
            or not scratch_arena_id
            or type(resolved_scratch_capacity) is not int
            or resolved_scratch_capacity < captured_m
            or recurrent_state_dtype not in (torch.float32, torch.bfloat16)
        ):
            raise Qwen3_5ParallelPerformanceError("body differs from the dense TP/PP multi-M ABI")
        rank, stage, tensor_rank = weights.rank, placement.pipeline_stage, placement.tensor_rank
        stage_ranks = tuple(
            item.rank for item in topology.placements if item.pipeline_stage == stage
        )
        lane_ranks = tuple(
            item.rank for item in topology.placements if item.tensor_rank == tensor_rank
        )
        _group_ranks(tp_group, stage_ranks, "TP")
        if topology.pp_size == 1:
            if pp_group is not None or lane_ranks != (rank,):
                raise Qwen3_5ParallelPerformanceError("PP1 must not create a pipeline group")
        else:
            if pp_group is None:
                raise Qwen3_5ParallelPerformanceError("PP topology requires a pipeline group")
            _group_ranks(pp_group, lane_ranks, "PP")
        self._validate_weights(config, parallel_plan, weights, stage)

        local_config = _local_config(
            config, stage=stage, tp_size=topology.tp_size, pp_size=topology.pp_size
        )
        text, local_text = config.graph.text, local_config.graph.text
        layer_start = stage * local_text.layer_count
        layer_ids = tuple(range(layer_start, layer_start + local_text.layer_count))
        global_full = tuple(
            layer
            for layer, kind in zip(layer_ids, local_text.layer_schedule, strict=True)
            if kind is AttentionKind.FULL_ATTENTION
        )
        global_gdn = tuple(
            layer
            for layer, kind in zip(layer_ids, local_text.layer_schedule, strict=True)
            if kind is AttentionKind.GATED_DELTA_NET
        )
        local_full = tuple(
            layer
            for layer, kind in enumerate(local_text.layer_schedule)
            if kind is AttentionKind.FULL_ATTENTION
        )
        local_gdn = tuple(
            layer
            for layer, kind in enumerate(local_text.layer_schedule)
            if kind is AttentionKind.GATED_DELTA_NET
        )
        stage_weights = _StageWeights(
            weights,
            stage=stage,
            pp_size=topology.pp_size,
            layer_start=layer_start,
            semantic_plan_digest=parallel_plan.semantic_plan_digest,
        )
        local_intermediate = local_text.dense_intermediate_size
        assert local_intermediate is not None
        # vLLM 8c51b926 (qwen3_5.py/qwen2_moe.py) and SGLang e51a3ae6
        # (qwen2.py) use gate-first merged column weights. llama.cpp 0b5be7e4
        # (qwen35.cpp) keeps the two semantic leaves explicit. Cubic uses the
        # merged launch while retaining the original leaves for qualification.
        if packed_gate_up_weights is None:
            packed_gate_up_weights = _pack_gate_up_weights(stage_weights, local_text.layer_count)
        elif (
            packed_gate_up_weights.shape
            != (local_text.layer_count, 2 * local_intermediate, local_text.hidden_size)
            or packed_gate_up_weights.dtype is not torch.bfloat16
            or packed_gate_up_weights.device != weights.device
            or not packed_gate_up_weights.is_contiguous()
        ):
            raise Qwen3_5ParallelPerformanceError(
                "reused gate/up weights differ from the packed BF16 ABI"
            )
        # vLLM 8c51b926 and SGLang e51a3ae6 merge QKV/Z and B/A projections;
        # llama.cpp 0b5be7e4 retains each checkpoint leaf. Cubic merges A/B/Z
        # while preserving the three original leaves at qualification time.
        if packed_gdn_abz is None:
            packed_gdn_abz_weights, gdn_abz_widths = _pack_gdn_abz_weights(stage_weights, local_gdn)
        else:
            packed_gdn_abz_weights, gdn_abz_widths = packed_gdn_abz
            if (
                len(gdn_abz_widths) != 3
                or packed_gdn_abz_weights.shape
                != (len(local_gdn), sum(gdn_abz_widths), local_text.hidden_size)
                or packed_gdn_abz_weights.dtype is not torch.bfloat16
                or packed_gdn_abz_weights.device != weights.device
                or not packed_gdn_abz_weights.is_contiguous()
            ):
                raise Qwen3_5ParallelPerformanceError(
                    "reused GDN A/B/Z weights differ from the packed BF16 ABI"
                )
        # Semantic names now point into merged storage. Release the superseded
        # allocator blocks before the cache arena consumes the remaining device memory.
        torch.cuda.empty_cache()
        gdn = local_text.gated_delta_net
        packed = 2 * gdn.key_heads * gdn.key_head_dim + gdn.value_heads * gdn.value_head_dim
        specs = (
            StateSlotSpec(
                CONV_STATE_ID,
                torch.bfloat16,
                (len(global_gdn), packed, gdn.convolution_kernel_size - 1),
            ),
            StateSlotSpec(
                RECURRENT_STATE_ID,
                recurrent_state_dtype,
                (len(global_gdn), gdn.value_heads, gdn.value_head_dim, gdn.key_head_dim),
            ),
        )
        arena_spec = CacheArenaSpec(
            f"qwen3_5.tp{topology.tp_size}pp{topology.pp_size}.rank{rank}",
            page_capacity,
            state_capacity,
            specs,
            resolved_scratch_capacity,
            (scratch_arena_id,),
        )
        arena = CacheArena(
            arena_spec,
            KVLayout(
                global_full,
                16,
                local_text.full_attention.key_value_heads,
                local_text.full_attention.head_dim,
            ),
            weights.device,
            rank_id=rank,
        )
        view = _ArenaView(
            arena_spec,
            KVLayout(
                local_full,
                16,
                local_text.full_attention.key_value_heads,
                local_text.full_attention.head_dim,
            ),
            weights.device,
            rank,
            arena.key,
            arena.value,
            arena.written,
            arena.states,
            owner=arena,
        )
        model = (
            Qwen3_5MoeStageModel(
                config=cast(object, local_config),
                weights=stage_weights,
                moe=lambda hidden, *, prefix, operators: hidden,
            )
            if config.graph.variant is Qwen3_5GraphVariant.MOE
            else Qwen3_5ReferenceModel(config=local_config, weights=stage_weights)
        )
        if mrope_tables is None:
            rotary = int(
                local_text.full_attention.head_dim * local_text.mrope.partial_rotary_factor
            )
            mrope_tables = build_bf16_mrope_tables(
                max_position=local_text.max_position_embeddings,
                device=weights.device,
                rotary_dim=rotary,
                section=local_text.mrope.section,
                rope_theta=local_text.mrope.rope_theta,
            )
        body = _DecodeBody(
            model,
            cast(CacheArena, view),
            captured_m=captured_m,
            table_pages=table_pages,
            mrope_tables=mrope_tables,
        )
        packed_mlp = torch.empty(
            (captured_m, 2 * local_intermediate),
            dtype=torch.bfloat16,
            device=weights.device,
        )
        packed_gdn_abz = torch.empty(
            (captured_m, sum(gdn_abz_widths)),
            dtype=torch.bfloat16,
            device=weights.device,
        )
        first_stage = stage == 0
        last_stage = stage == topology.pp_size - 1
        embedding_span = (
            next(
                item.spans[0]
                for item in weights.bindings
                if item.tensor_name == "model.language_model.embed_tokens.weight"
            )
            if first_stage
            else None
        )
        vocab_span = (
            next(item.spans[0] for item in weights.bindings if item.tensor_name == "lm_head.weight")
            if last_stage
            else None
        )
        self.rank, self.stage, self.tensor_rank = rank, stage, tensor_rank
        self.layer_ids, self.full_layer_ids, self.gdn_layer_ids = (
            layer_ids,
            global_full,
            global_gdn,
        )
        self.captured_m = captured_m
        self.lane_id = 0
        self.bucket_logical_ms = tuple(range(1, captured_m + 1))
        self.embedding_span = embedding_span
        self.vocab_span = vocab_span
        self._is_first_stage, self._is_last_stage = first_stage, last_stage
        self._is_moe = config.graph.variant is Qwen3_5GraphVariant.MOE
        self._moe = text.moe
        self._first_expert = topology.expert_coordinates(rank)[0] * (
            0 if text.moe is None else text.moe.expert_count // topology.ep_size
        )
        self._tp_size, self._pp_size = topology.tp_size, topology.pp_size
        self._vocab_size = text.vocabulary_size
        self._global_weight_shapes = {
            item.tensor_name: item.global_shape for item in weights.bindings
        }
        self._mrope_tables = mrope_tables
        self._packed_gate_up_weights = packed_gate_up_weights
        self._packed_gdn_abz_weights = packed_gdn_abz_weights
        self._local_intermediate = local_intermediate
        self._gdn_abz_widths = gdn_abz_widths
        self._tp_group, self._pp_group = tp_group, pp_group
        self._pp_peer = (
            None if topology.pp_size == 1 else lane_ranks[1] if first_stage else lane_ranks[0]
        )
        self._is_arena_primary = True
        self._bootstrap_siblings: list[Qwen3_5TP2PP2DecodeBody] = []
        self._graph_capture_started = False
        self._arena_rebind_failed = False
        self._arena_replacement_allowed = True
        self._active_rows = torch.full((1,), captured_m, dtype=torch.int32, device=weights.device)
        self._install_bucket_runtime(
            arena=arena,
            body=body,
            packed_mlp=packed_mlp,
            packed_gdn_abz=packed_gdn_abz,
        )

    def _install_bucket_runtime(
        self,
        *,
        arena: CacheArena,
        body: _DecodeBody,
        packed_mlp: Tensor,
        packed_gdn_abz: Tensor,
    ) -> None:
        """Calibrate ordinary projections independently of model interleaving.

        An enclosing overlap plan supplies its joint compute/transport cost.
        Otherwise the same linear candidates use their standalone compute cost;
        expert grouping and ownership never participate in this selection.
        """
        device = body.buffers.norm.device
        cost = getattr(self, "_standalone_linear_cost", None)
        standalone = device.type == "cuda" and current_linear_cost() is None
        if standalone and cost is None:
            cost = LinearLaunchCost(device_resources(device), 0.0, ())
            self._standalone_linear_cost: LinearLaunchCost | None = cost
        with linear_cost_scope(cost) if standalone and cost is not None else nullcontext():
            self._bind_bucket_runtime(arena=arena, body=body, packed_mlp=packed_mlp,
                                      packed_gdn_abz=packed_gdn_abz)

    def _bind_bucket_runtime(
        self,
        *,
        arena: CacheArena,
        body: _DecodeBody,
        packed_mlp: Tensor,
        packed_gdn_abz: Tensor,
    ) -> None:
        """Bind one bucket's private tensors and primitives to shared immutable inputs."""

        captured_m = self.captured_m
        if (
            packed_mlp.shape != (captured_m, 2 * self._local_intermediate)
            or packed_gdn_abz.shape != (captured_m, sum(self._gdn_abz_widths))
            or packed_mlp.dtype is not torch.bfloat16
            or packed_gdn_abz.dtype is not torch.bfloat16
            or packed_mlp.device != body.buffers.norm.device
            or packed_gdn_abz.device != body.buffers.norm.device
        ):
            raise Qwen3_5ParallelPerformanceError(
                "bucket workspaces differ from the fixed captured-M ABI"
            )
        local_intermediate = self._local_intermediate
        a_width, b_width, _ = self._gdn_abz_widths
        self._embedding_table = None
        if (getattr(self, "_is_first_stage", self.stage == 0)
                and body.buffers.hidden.is_cuda and collective_parameters().embedding_host_mirror):
            from cubic_serve.ops.cuda.embedding.immutable import ImmutableEmbeddingTable
            weight = body.model.weights["model.language_model.embed_tokens.weight"]
            self._embedding_table = ImmutableEmbeddingTable.create(
                weight, self.embedding_span, self._tp_group,
                body.buffers.hidden.device, body.buffers.hidden.dtype,
            )
            if self._embedding_table is not None:
                body.embedding = self._embedding_table.bind(
                    body.buffers.token_ids, body.buffers.hidden)
        self.arena, self._body = arena, body
        self._packed_mlp = packed_mlp
        self._mlp_gate_output = packed_mlp[:, :local_intermediate]
        self._mlp_up_output = packed_mlp[:, local_intermediate:]
        self._linear_mlp_gate_up = self._build_linear(
            body.buffers.norm, self._packed_gate_up_weights[0], packed_mlp
        )
        self._packed_mlp_gate = BF16PackedSwiGLUPrimitive(packed_mlp, body.buffers.mlp_active)
        self._packed_gdn_abz = packed_gdn_abz
        self._gdn_a_output = packed_gdn_abz[:, :a_width]
        self._gdn_b_output = packed_gdn_abz[:, a_width : a_width + b_width]
        self._gdn_z_output = packed_gdn_abz[:, a_width + b_width :]
        self._linear_gdn_abz = self._build_linear(
            body.buffers.norm, self._packed_gdn_abz_weights[0], packed_gdn_abz
        )
        self._gdn_coefficients: Tensor | None = None
        coefficient_args: dict[str, Tensor] = {}
        if body.buffers.hidden.device.type == "cuda":
            gdn = body.model.config.graph.text.gated_delta_net
            parameters = gdn_parameters(q_heads=gdn.key_heads, value_heads=gdn.value_heads,
                                        key_dim=gdn.key_head_dim, value_dim=gdn.value_head_dim)
            if parameters.reuse_coefficients:
                self._gdn_coefficients = torch.empty(
                    (captured_m, a_width, 2), dtype=torch.float32, device=packed_gdn_abz.device)
                first_gdn = f"model.language_model.layers.{body.gdn_layers[0]}.linear_attn"
                coefficient_args = dict(coefficients=self._gdn_coefficients,
                                        a_log=body.model.weights[f"{first_gdn}.A_log"],
                                        dt_bias=body.model.weights[f"{first_gdn}.dt_bias"])
        self._packed_gdn_abz_split = BF16PackedGDNProjectionSplitPrimitive(
            packed_gdn_abz, body.buffers.gdn_a, body.buffers.gdn_b,
            body.buffers.gdn_gate.reshape(captured_m, -1), **coefficient_args,
        )
        # CPU-only bucket ownership tests use a structural fake body and never
        # launch collectives.  Real parallel decode bodies are CUDA-only.
        self._tp_fixed_sum = (
            FixedRankBF16SumPrimitive(body.buffers.hidden, self._tp_group)
            if body.buffers.hidden.device.type == "cuda"
            else None
        )
        self._serialize_tp_sums = False
        self.embedding_mask = (
            torch.zeros(
                (captured_m, 1),
                dtype=torch.bfloat16,
                device=body.buffers.norm.device,
            )
            if getattr(self, "_is_first_stage", self.stage == 0)
            else None
        )
        self.feature_rows = (
            torch.zeros(
                (captured_m, body.buffers.hidden.shape[1]),
                dtype=torch.bfloat16,
                device=body.buffers.norm.device,
            )
            if getattr(self, "_is_first_stage", self.stage == 0)
            else None
        )
        # The lightweight arena-replacement harness deliberately substitutes a
        # cache-only body.  Real decode bodies always expose this complete set.
        self._grouped_gdn = None
        self._grouped_attention = None
        if hasattr(body, "linear_gdn_qkv"):
            self._linear_gdn_qkv = self._build_linear(
                body.buffers.norm,
                body.model.weights[
                    f"model.language_model.layers.{body.gdn_layers[0]}.linear_attn.in_proj_qkv.weight"
                ],
                body.buffers.gdn_packed,
                vendor=body.linear_gdn_qkv,
            )
            first_full = body.full_layers[0]
            full_prefix = f"model.language_model.layers.{first_full}.self_attn"
            self._linear_attention_q = self._build_linear(
                body.buffers.norm,
                body.model.weights[f"{full_prefix}.q_proj.weight"],
                body.buffers.attention_projected.reshape(captured_m, -1),
                vendor=body.linear_attention_q,
            )
            self._linear_attention_kv = self._build_linear(
                body.buffers.norm,
                body.model.weights[f"{full_prefix}.k_proj.weight"],
                body.buffers.key.reshape(captured_m, -1),
                vendor=body.linear_attention_kv,
            )
            self._grouped_gdn = bind_grouped_linear(
                body.buffers.norm,
                (body.model.weights[
                    f"model.language_model.layers.{body.gdn_layers[0]}.linear_attn.in_proj_qkv.weight"
                ], self._packed_gdn_abz_weights[0]),
                (body.buffers.gdn_packed, packed_gdn_abz), self._active_rows,
            )
            self._grouped_attention = bind_grouped_linear(
                body.buffers.norm,
                tuple(
                    body.model.weights[f"{full_prefix}.{name}_proj.weight"]
                    for name in ("q", "k", "v")
                ),
                (body.buffers.attention_projected.reshape(captured_m, -1),
                 body.buffers.key.reshape(captured_m, -1),
                 body.buffers.value.reshape(captured_m, -1)), self._active_rows,
            )
            first_kind = body.schedule[0]
            first_op = "linear_attn" if first_kind is AttentionKind.GATED_DELTA_NET else "self_attn"
            first_suffix = (
                "out_proj.weight"
                if first_kind is AttentionKind.GATED_DELTA_NET
                else "o_proj.weight"
            )
            self._linear_output = self._build_linear(
                body.buffers.gdn_normalized.reshape(captured_m, -1)
                if first_kind is AttentionKind.GATED_DELTA_NET
                else body.buffers.rotated_query.reshape(captured_m, -1),
                body.model.weights[f"model.language_model.layers.0.{first_op}.{first_suffix}"],
                body.buffers.update,
                vendor=body.linear_output,
            )
            self._linear_mlp_down = self._build_linear(
                body.buffers.mlp_active,
                body.model.weights["model.language_model.layers.0.mlp.down_proj.weight"],
                body.buffers.update,
                vendor=body.linear_mlp_down,
            )
            if (body.linear_logits is None
                    or isinstance(body.linear_logits, PackedProjectionDispatch)):
                raise Qwen3_5ParallelPerformanceError(
                    "packed distributed projection occurrences require their own collective binding"
                )
            self._linear_logits: BF16LinearKernel = body.linear_logits
            if self._is_last_stage:
                self._linear_logits = self._build_linear(
                    body.buffers.norm,
                    body.model.weights["lm_head.weight"],
                    body.buffers.logits,
                    vendor=body.linear_logits,
                )
            if getattr(self, "_is_moe", False):
                moe = self._moe
                if moe is None or not moe.normalize_topk_probability:
                    raise Qwen3_5ParallelPerformanceError("MoE routing contract is incomplete")
                first_mlp = "model.language_model.layers.0.mlp"
                self._moe_router_logits = torch.empty(
                    (captured_m, moe.expert_count),
                    dtype=torch.bfloat16,
                    device=body.buffers.norm.device,
                )
                self._moe_routed = torch.empty_like(body.buffers.update)
                self._moe_shared_gate = torch.empty(
                    (captured_m, 1),
                    dtype=torch.bfloat16,
                    device=body.buffers.norm.device,
                )
                self._linear_moe_router = self._build_linear(
                    body.buffers.norm,
                    body.model.weights[f"{first_mlp}.gate.weight"],
                    self._moe_router_logits,
                )
                self._linear_moe_shared_gate = self._build_linear(
                    body.buffers.norm,
                    body.model.weights[f"{first_mlp}.shared_expert_gate.weight"],
                    self._moe_shared_gate,
                )
                self._routed_experts = BF16RoutedExpertsPrimitive(
                    body.buffers.norm,
                    self._moe_router_logits,
                    body.buffers.row_activity,
                    self._active_rows,
                    body.model.weights[f"{first_mlp}.experts.gate_up_proj"],
                    body.model.weights[f"{first_mlp}.experts.down_proj"],
                    self._moe_routed,
                    first_expert=self._first_expert,
                    top_k=moe.experts_per_token,
                )
                self._moe_combine = BF16MoeSharedCombinePrimitive(
                    body.buffers.update,
                    self._moe_shared_gate,
                    self._moe_routed,
                    body.buffers.update,
                )
                self._moe_branches = MoeBranchStreams(body.buffers.norm.device, len(body.schedule))

    def _build_linear(
        self,
        linear_input: Tensor,
        local_weight: Tensor,
        local_output: Tensor,
        *,
        vendor: BF16LinearPrimitive | PackedProjectionDispatch | None = None,
    ) -> BF16LinearKernel:
        if isinstance(vendor, PackedProjectionDispatch):
            raise Qwen3_5ParallelPerformanceError(
                "packed projections cannot enter the BF16 distributed kernel selector"
            )
        return bind_bf16_linear_kernel(
            linear_input,
            local_weight,
            local_output,
            self._active_rows,
            vendor=vendor,
            reduction="fixed_tile",
        )

    def set_active_rows(self, logical_m: int) -> None:
        """Validate logical M while keeping dense math on the full physical bucket."""

        if type(logical_m) is not int or not 0 < logical_m <= self.captured_m:
            raise Qwen3_5ParallelPerformanceError("logical rows lie outside captured M")
        # Every operator executes the complete physical bucket. Padding, when
        # present, is isolated by the topology-wide scratch lease.
        self._active_rows.fill_(self.captured_m)

    def _launch_linear(
        self,
        primitive: BF16LinearKernel,
        linear_input: Tensor,
        local_weight: Tensor,
        local_output: Tensor,
    ) -> None:
        launch_bf16_linear_kernel(
            primitive, linear_input, local_weight, local_output, self._active_rows
        )

    def build_sibling(self, *, captured_m: int, lane_id: int = 0) -> Qwen3_5TP2PP2DecodeBody:
        """Build another fixed-M body over the final arena and packed weights.

        The final arena is externally sized for the largest sibling.  Only
        bucket-private activations, launch workspaces, primitives, and the
        embedding mask are allocated here; checkpoint weights, M-RoPE tables,
        model metadata, and all cache/state backing remain shared.
        """

        table_pages = self.buffers.page_table.shape[1]
        spec = self.arena.spec
        if (
            not self._is_arena_primary
            or not self._arena_replacement_allowed
            or self._graph_capture_started
            or type(captured_m) is not int
            or captured_m < 1
            or type(lane_id) is not int
            or lane_id < 0
            or (captured_m, lane_id) == (self.captured_m, self.lane_id)
            or any(
                (item.captured_m, item.lane_id) == (captured_m, lane_id)
                for item in self._bootstrap_siblings
            )
        ):
            raise Qwen3_5ParallelPerformanceError(
                "bootstrap arena cannot admit the sibling captured-M ABI"
            )

        sibling = object.__new__(type(self))
        sibling.rank, sibling.stage, sibling.tensor_rank = (
            self.rank,
            self.stage,
            self.tensor_rank,
        )
        sibling.layer_ids = self.layer_ids
        sibling.full_layer_ids = self.full_layer_ids
        sibling.gdn_layer_ids = self.gdn_layer_ids
        sibling.captured_m = captured_m
        sibling.lane_id = lane_id
        sibling.bucket_logical_ms = tuple(range(1, captured_m + 1))
        sibling.embedding_span = self.embedding_span
        sibling.vocab_span = self.vocab_span
        sibling._is_first_stage, sibling._is_last_stage = (
            getattr(self, "_is_first_stage", self.stage == 0),
            getattr(self, "_is_last_stage", self.stage == 1),
        )
        sibling._is_moe = getattr(self, "_is_moe", False)
        sibling._moe = getattr(self, "_moe", None)
        sibling._first_expert = getattr(self, "_first_expert", 0)
        sibling._tp_size, sibling._pp_size = (
            getattr(self, "_tp_size", 2),
            getattr(self, "_pp_size", 2),
        )
        sibling._vocab_size = self._vocab_size
        sibling._global_weight_shapes = self._global_weight_shapes
        sibling._standalone_linear_cost = getattr(self, "_standalone_linear_cost", None)
        sibling._mrope_tables = self._mrope_tables
        sibling._packed_gate_up_weights = self._packed_gate_up_weights
        sibling._packed_gdn_abz_weights = self._packed_gdn_abz_weights
        sibling._local_intermediate = self._local_intermediate
        sibling._gdn_abz_widths = self._gdn_abz_widths
        sibling._tp_group, sibling._pp_group = self._tp_group, self._pp_group
        sibling._pp_peer = self._pp_peer
        sibling._is_arena_primary = False
        sibling._bootstrap_siblings = []
        sibling._graph_capture_started = False
        sibling._arena_rebind_failed = False
        sibling._arena_replacement_allowed = True
        sibling._active_rows = torch.full(
            (1,), captured_m, dtype=torch.int32, device=self.hidden.device
        )

        local_view = self._body.arena
        view = _ArenaView(
            spec,
            local_view.layout,
            self.arena.device,
            self.arena.rank_id,
            self.arena.key,
            self.arena.value,
            self.arena.written,
            self.arena.states,
            owner=self.arena,
        )
        try:
            body = _DecodeBody(
                self._body.model,
                cast(CacheArena, view),
                captured_m=captured_m,
                table_pages=table_pages,
                mrope_tables=self._mrope_tables,
            )
            packed_mlp = torch.empty(
                (captured_m, 2 * self._local_intermediate),
                dtype=torch.bfloat16,
                device=self.hidden.device,
            )
            packed_gdn_abz = torch.empty(
                (captured_m, sum(self._gdn_abz_widths)),
                dtype=torch.bfloat16,
                device=self.hidden.device,
            )
            sibling._install_bucket_runtime(
                arena=self.arena,
                body=body,
                packed_mlp=packed_mlp,
                packed_gdn_abz=packed_gdn_abz,
            )
            if sibling._is_moe:
                # Every bucket replays under the same topology-wide lease.  A
                # single auxiliary stream therefore supplies the shared-expert
                # lane without multiplying CUDA Graph stream identities.
                sibling._moe_branches.shared = self._moe_branches.shared
        except Qwen3_5ParallelPerformanceError:
            raise
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise Qwen3_5ParallelPerformanceError(str(error)) from error
        self._bootstrap_siblings.append(sibling)
        return sibling

    @classmethod
    def build(
        cls,
        *,
        config: Qwen3_5Config,
        parallel_plan: Qwen3_5ParallelWeightPlan,
        weights: Qwen3_5RankWeights,
        tp_group: ProcessGroup,
        pp_group: ProcessGroup | None,
        captured_m: int,
        table_pages: int,
        page_capacity: int,
        state_capacity: int,
        scratch_arena_id: str,
        scratch_capacity: int | None = None,
        recurrent_state_dtype: torch.dtype = torch.float32,
        mrope_tables: _MRoPETables | None = None,
        packed_gate_up_weights: Tensor | None = None,
        packed_gdn_abz: tuple[Tensor, tuple[int, int, int]] | None = None,
    ) -> Qwen3_5TP2PP2DecodeBody:
        """Build and validate one rank body; never substitutes a reference replica."""

        try:
            return cls(
                config=config,
                parallel_plan=parallel_plan,
                weights=weights,
                tp_group=tp_group,
                pp_group=pp_group,
                captured_m=captured_m,
                table_pages=table_pages,
                page_capacity=page_capacity,
                state_capacity=state_capacity,
                scratch_arena_id=scratch_arena_id,
                scratch_capacity=scratch_capacity,
                recurrent_state_dtype=recurrent_state_dtype,
                mrope_tables=mrope_tables,
                packed_gate_up_weights=packed_gate_up_weights,
                packed_gdn_abz=packed_gdn_abz,
            )
        except Qwen3_5ParallelPerformanceError:
            raise
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise Qwen3_5ParallelPerformanceError(str(error)) from error

    def replace_shared_cache_arena(
        self,
        spec: CacheArenaSpec,
        siblings: tuple[Qwen3_5TP2PP2DecodeBody, ...],
    ) -> None:
        """Atomically rebind a complete bootstrap bucket family to one final arena.

        All bucket-private activations, workspaces, and non-cache primitives stay
        at their existing addresses.  Bootstrap cache references are dropped
        family-wide before allocating the final near-capacity backing.  Failure
        leaves every body uniformly detached and poisoned for worker shutdown;
        no subset can remain runnable on a different arena.
        """

        old_arena = self.arena
        registered = tuple(self._bootstrap_siblings)
        bodies = (self, *siblings)
        if (
            not self._is_arena_primary
            or not self._arena_replacement_allowed
            or self._graph_capture_started
            or not isinstance(spec, CacheArenaSpec)
            or type(siblings) is not tuple
            or len({id(item) for item in siblings}) != len(siblings)
            or {id(item) for item in siblings} != {id(item) for item in registered}
        ):
            raise Qwen3_5ParallelPerformanceError(
                "shared arena replacement requires the complete uncaptured bucket family"
            )
        table_pages = self.buffers.page_table.shape[1]
        largest_m = max(item.captured_m for item in bodies)
        current = old_arena.spec
        cache_runtime_complete = all(
            all(hasattr(item._body, name) for name in _CACHE_RUNTIME_ATTRIBUTES)
            and item._body.arena.key is old_arena.key
            and item._body.arena.value is old_arena.value
            and item._body.arena.written is old_arena.written
            and item._body.arena.states is old_arena.states
            for item in bodies
        )
        shared = all(
            isinstance(item, Qwen3_5TP2PP2DecodeBody)
            and item is not self
            and not item._is_arena_primary
            and item._arena_replacement_allowed
            and not item._graph_capture_started
            and not item._arena_rebind_failed
            and item.arena is old_arena
            and item._body.model is self._body.model
            and item._packed_gate_up_weights is self._packed_gate_up_weights
            and item._packed_gdn_abz_weights is self._packed_gdn_abz_weights
            and item._mrope_tables is self._mrope_tables
            and item.buffers.page_table.shape[1] == table_pages
            for item in siblings
        )
        if (
            self._arena_rebind_failed
            or not cache_runtime_complete
            or not shared
            or len({(item.captured_m, item.lane_id) for item in bodies}) != len(bodies)
            or spec.arena_id != current.arena_id
            or spec.state_specs != current.state_specs
            or spec.scratch_arena_ids != current.scratch_arena_ids
            or spec.scratch_capacity < largest_m
            or spec.state_slot_capacity < 1
            or spec.page_capacity < minimum_kv_page_capacity(table_pages, largest_m)
        ):
            raise Qwen3_5ParallelPerformanceError(
                "final shared arena differs from the complete bucket cache/state ABI"
            )

        global_layout = old_arena.layout
        device, rank_id = old_arena.device, old_arena.rank_id
        local_layouts = tuple(item._body.arena.layout for item in bodies)
        for item in bodies:
            item._arena_replacement_allowed = False
            item._arena_rebind_failed = True
            _drop_cache_runtime(item._body)
            del item.arena
        del old_arena
        torch.cuda.empty_cache()

        try:
            arena = CacheArena(
                spec,
                global_layout,
                device,
                rank_id=rank_id,
            )
            bindings = tuple(
                _build_cache_runtime(
                    item._body,
                    _ArenaView(
                        spec,
                        local_layout,
                        arena.device,
                        arena.rank_id,
                        arena.key,
                        arena.value,
                        arena.written,
                        arena.states,
                        owner=arena,
                    ),
                )
                for item, local_layout in zip(bodies, local_layouts, strict=True)
            )
        except (
            Qwen3_5ParallelPerformanceError,
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as error:
            for item in bodies:
                item._arena_rebind_failed = True
            if isinstance(error, Qwen3_5ParallelPerformanceError):
                raise
            raise Qwen3_5ParallelPerformanceError(str(error)) from error

        for item, binding in zip(bodies, bindings, strict=True):
            _commit_cache_runtime(item._body, binding)
            item.arena = arena
            item._arena_rebind_failed = False

    def replace_cache_arena(self, spec: CacheArenaSpec) -> None:
        """Compatibility entrypoint that coordinates every registered sibling."""

        self.replace_shared_cache_arena(spec, tuple(self._bootstrap_siblings))

    def _mark_graph_capture_started(self) -> None:
        """Permanently close arena replacement before the first capture action."""

        if self._arena_rebind_failed or not hasattr(self, "arena"):
            raise Qwen3_5ParallelPerformanceError(
                "graph capture cannot use a failed shared arena replacement"
            )
        self._graph_capture_started = True
        self._arena_replacement_allowed = False

    @staticmethod
    def _validate_weights(
        config: Qwen3_5Config,
        plan: Qwen3_5ParallelWeightPlan,
        weights: Qwen3_5RankWeights,
        stage: int,
    ) -> None:
        expected = plan.bindings_for_rank(weights.rank)
        if weights.bindings != expected or weights.tensor_count != len(expected):
            raise Qwen3_5ParallelPerformanceError("resident rank weights differ from their plan")
        for binding in expected:
            value = weights[binding.tensor_name]
            if (
                value.dtype is not binding.dtype
                or value.device != weights.device
                or tuple(value.shape) != binding.local_shape
                or not value.is_contiguous()
            ):
                raise Qwen3_5ParallelPerformanceError(
                    f"rank weight {binding.tensor_name!r} differs from its shard ABI"
                )
        names = {item.tensor_name for item in expected}
        endpoints = {
            "model.language_model.embed_tokens.weight",
            "model.language_model.norm.weight",
            "lm_head.weight",
        }
        topology = plan.topology
        wanted: set[str] = set()
        if stage == 0:
            wanted.add("model.language_model.embed_tokens.weight")
        if stage == topology.pp_size - 1:
            wanted.update(("model.language_model.norm.weight", "lm_head.weight"))
        if names & endpoints != wanted:
            raise Qwen3_5ParallelPerformanceError("PP endpoint weight ownership is open")
        layers_per_stage, remainder = divmod(config.graph.text.layer_count, topology.pp_size)
        if remainder:
            raise Qwen3_5ParallelPerformanceError("PP layer partition is not divisible")
        local_layers = set(range(stage * layers_per_stage, (stage + 1) * layers_per_stage))
        owned_layers = {
            int(name.split(".layers.", 1)[1].split(".", 1)[0])
            for name in names
            if ".layers." in name
        }
        if owned_layers != local_layers:
            raise Qwen3_5ParallelPerformanceError("rank does not own its contiguous PP stage")
        kv = tuple(item for item in plan.kv_heads if item.rank == weights.rank)
        gdn = tuple(item for item in plan.gdn_states if item.rank == weights.rank)
        schedule = config.graph.text.layer_schedule
        expected_full = tuple(
            layer
            for layer in sorted(local_layers)
            if schedule[layer] is AttentionKind.FULL_ATTENTION
        )
        expected_gdn = tuple(
            layer
            for layer in sorted(local_layers)
            if schedule[layer] is AttentionKind.GATED_DELTA_NET
        )
        local_kv_heads = max(
            1, config.graph.text.full_attention.key_value_heads // topology.tp_size
        )
        if (
            tuple(item.layer_index for item in kv) != expected_full
            or tuple(item.layer_index for item in gdn) != expected_gdn
            or any(item.head_span.size != local_kv_heads for item in kv)
            or any(item.convolution_dtype is not torch.bfloat16 for item in gdn)
            or any(item.recurrent_dtype is not torch.float32 for item in gdn)
        ):
            raise Qwen3_5ParallelPerformanceError("rank-local KV/GDN state ownership is open")

    @property
    def buffers(self) -> _Buffers:
        return self._body.buffers

    @property
    def hidden(self) -> Tensor:
        return self._body.buffers.hidden

    @property
    def logits(self) -> Tensor | None:
        is_last = getattr(self, "_is_last_stage", self.stage == 1)
        return self._body.buffers.logits if is_last else None

    def prepare_scratch(self, resource: RankReplayResources) -> None:
        if resource.rank != self.rank:
            raise Qwen3_5ParallelPerformanceError("scratch resource belongs to another rank")
        self._body.prepare_capture_scratch(resource)
        if self.embedding_mask is not None:
            self.embedding_mask.zero_()
        feature_rows = getattr(self, "feature_rows", None)
        if feature_rows is not None:
            feature_rows.zero_()

    def bind_scratch(self, resource: RankReplayResources) -> None:
        """Restore ownerless padding row bindings without clearing scratch state."""

        if resource.rank != self.rank:
            raise Qwen3_5ParallelPerformanceError("scratch resource belongs to another rank")
        self._body.bind_capture_scratch(resource)
        if self.embedding_mask is not None:
            self.embedding_mask.zero_()
        feature_rows = getattr(self, "feature_rows", None)
        if feature_rows is not None:
            feature_rows.zero_()

    @property
    def embedding_input_span(self) -> AxisSpan | None:
        table = getattr(self, "_embedding_table", None)
        return AxisSpan(0, table.vocabulary) if table is not None else self.embedding_span

    def fill_token_ids(self, global_token_ids: Tensor) -> None:
        """Map every fixed physical row from global ids into this embedding shard."""

        if not self._is_first_stage or self.embedding_span is None or self.embedding_mask is None:
            raise Qwen3_5ParallelPerformanceError("only the first PP stage accepts token ids")
        if (
            global_token_ids.shape != (self.captured_m,)
            or global_token_ids.dtype is not torch.int64
            or global_token_ids.device != self.hidden.device
            or not global_token_ids.is_contiguous()
        ):
            raise Qwen3_5ParallelPerformanceError(
                "token fill requires fixed contiguous CUDA int64 ids"
            )
        if bool(((global_token_ids < 0) | (global_token_ids >= self._vocab_size)).any().item()):
            raise Qwen3_5ParallelPerformanceError(
                "physical row contains an invalid global token id"
            )
        span = getattr(self, "embedding_input_span", self.embedding_span)
        assert span is not None
        owned = (global_token_ids >= span.start) & (global_token_ids < span.stop)
        local = torch.where(
            owned,
            global_token_ids - span.start,
            torch.zeros_like(global_token_ids),
        )
        self.buffers.token_ids.copy_(local)
        self.embedding_mask.copy_(owned[:, None])
        assert self.feature_rows is not None
        self.feature_rows.zero_()

    def _tp_sum(self, value: Tensor) -> None:
        if self._tp_fixed_sum is None:
            raise Qwen3_5ParallelPerformanceError(
                "target TP sum is unavailable outside the CUDA decode body"
            )
        self._tp_fixed_sum.launch(value)
        if self._serialize_tp_sums:
            torch.cuda.current_stream(value.device).synchronize()

    def _tp_sum_residual_norm(
        self,
        value: Tensor,
        residual: Tensor,
        summed: Tensor,
        weight: Tensor,
        output: Tensor,
    ) -> None:
        if self._tp_fixed_sum is None:
            raise Qwen3_5ParallelPerformanceError(
                "target TP fused residual RMSNorm is unavailable outside the CUDA decode body"
            )
        self._tp_fixed_sum.launch_residual_rms_norm(
            value,
            residual,
            summed,
            weight,
            output,
            epsilon=self._body.model.config.graph.text.rms_norm_epsilon,
        )
        if self._serialize_tp_sums:
            torch.cuda.current_stream(value.device).synchronize()

    def _linear_steps(
        self,
        primitive: BF16LinearKernel,
        linear_input: Tensor,
        local_weight: Tensor,
        local_output: Tensor,
        *,
        global_layer: int | None,
        weight_name: str,
        row_parallel: bool,
        observer: Qwen3_5LinearObserver | None,
        fused_residual_norm: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
    ) -> Generator[Callable[[], None], None, bool]:
        self._launch_linear(primitive, linear_input, local_weight, local_output)
        partial = local_output.clone() if row_parallel and observer is not None else None
        fused = False
        if row_parallel:
            if fused_residual_norm is not None and observer is None:
                yield _defer_collective(
                    self._tp_sum_residual_norm, local_output, *fused_residual_norm
                )
                fused = True
            else:
                (yield _defer_collective(self._tp_sum, local_output))
        if observer is not None:
            artifact = primitive.project_artifact()
            observer(
                Qwen3_5LinearLeafEvent(
                    global_layer=global_layer,
                    weight_name=weight_name,
                    weight_shape=self._global_weight_shapes[weight_name],
                    artifact_digest=artifact.artifact_digest,
                    launch_specialization_digest=artifact.launch_specialization_digest,
                    linear_input=linear_input,
                    local_weight=local_weight,
                    local_output=partial if partial is not None else local_output,
                    tp_partial=partial,
                    tp_reduced=local_output if row_parallel else None,
                )
            )
        return fused

    def _observe_packed_columns(
        self,
        observer: Qwen3_5LinearObserver | None,
        primitive: BF16LinearKernel | BF16GroupedLinearPrimitive,
        linear_input: Tensor,
        global_layer: int,
        leaves: tuple[tuple[str, Tensor, Tensor], ...],
    ) -> None:
        if observer is None:
            return
        artifact = primitive.project_artifact()
        for weight_name, local_weight, local_output in leaves:
            observer(
                Qwen3_5LinearLeafEvent(
                    global_layer=global_layer,
                    weight_name=weight_name,
                    weight_shape=self._global_weight_shapes[weight_name],
                    artifact_digest=artifact.artifact_digest,
                    launch_specialization_digest=artifact.launch_specialization_digest,
                    linear_input=linear_input,
                    local_weight=local_weight,
                    local_output=local_output,
                    tp_partial=None,
                    tp_reduced=None,
                )
            )

    def _pipeline(self, value: Tensor) -> None:
        if self._pp_group is None or self._pp_peer is None or self._pp_size == 1:
            raise Qwen3_5ParallelPerformanceError("PP1 cannot launch pipeline communication")
        operation = dist.isend if self._is_first_stage else dist.irecv
        requests = dist.batch_isend_irecv(
            [dist.P2POp(operation, value, self._pp_peer, group=self._pp_group)]
        )
        for request in requests:
            request.wait()

    def launch(self) -> None:
        """Launch one eager qualification step or record the same fixed graph ABI."""

        self._launch(None)

    def launch_capture_warmup(self) -> None:
        """Initialize a fresh stream without queuing work across TP boundaries."""

        self._serialize_tp_sums = True
        try:
            self._launch(None)
        finally:
            self._serialize_tp_sums = False

    def launch_qualification(
        self,
        observer: Qwen3_5LinearObserver | None = None,
        *,
        layer_observer: Qwen3_5LayerObserver | None = None,
    ) -> None:
        """Launch eager math and expose requested numerical boundaries."""

        if torch.cuda.is_current_stream_capturing():
            raise Qwen3_5ParallelPerformanceError("linear observation is eager-only")
        self._launch(observer, layer_observer=layer_observer)

    def _launch(
        self,
        observer: Qwen3_5LinearObserver | None,
        *,
        layer_observer: Qwen3_5LayerObserver | None = None,
    ) -> None:
        # Sequential execution consumes the identical arithmetic and collective
        # boundaries used by interleaved startup capture.
        for operation in self.launch_steps(observer, layer_observer=layer_observer):
            operation()

    def launch_steps(
        self,
        observer: Qwen3_5LinearObserver | None,
        *,
        layer_observer: Qwen3_5LayerObserver | None = None,
    ) -> Iterator[Callable[[], None]]:
        b, weights = self.buffers, self._body.model.weights
        if getattr(self, "_is_moe", False):
            self._moe_branches.begin()
        if self._is_first_stage:
            assert self.embedding_mask is not None
            assert self._body.embedding is not None
            self._body.embedding.launch(
                b.token_ids,
                weights["model.language_model.embed_tokens.weight"],
                b.hidden,
            )
            torch.mul(b.hidden, self.embedding_mask, out=b.hidden)
            assert self.feature_rows is not None
            torch.add(b.hidden, self.feature_rows, out=b.hidden)
            if self._embedding_table is None:
                (yield _defer_collective(self._tp_sum, b.hidden))
        elif self._pp_size > 1:
            self._pipeline(b.hidden)
        self._body.mrope_gather.launch(
            b.position_ids,
            self._body.cosine_table,
            self._body.sine_table,
            b.cosine,
            b.sine,
        )
        self._body.page_copy.launch(
            self._body.arena.key,
            self._body.arena.value,
            self._body.arena.written,
            b.page_source,
            b.page_destination,
        )
        for layer, kind in enumerate(self._body.schedule):
            prefix = f"model.language_model.layers.{layer}"
            global_prefix = f"model.language_model.layers.{self.layer_ids[layer]}"
            if layer_observer is not None:
                layer_observer(self.layer_ids[layer], "layer_input", b.hidden)
            if layer == 0:
                self._body.hidden_norm.launch(
                    b.hidden, weights[f"{prefix}.input_layernorm.weight"], b.norm
                )
            if layer_observer is not None:
                layer_observer(self.layer_ids[layer], "input_norm", b.norm)
            if kind is AttentionKind.GATED_DELTA_NET:
                op, local = f"{prefix}.linear_attn", self._body._gdn_local[layer]
                global_op = f"{global_prefix}.linear_attn"
                if self._grouped_gdn is not None:
                    self._grouped_gdn.launch(
                        b.norm, (weights[f"{op}.in_proj_qkv.weight"],
                                 self._packed_gdn_abz_weights[local]),
                        (b.gdn_packed, self._packed_gdn_abz), self._active_rows,
                    )
                    self._observe_packed_columns(
                        observer, self._grouped_gdn, b.norm, self.layer_ids[layer],
                        ((f"{global_op}.in_proj_qkv.weight",
                          weights[f"{op}.in_proj_qkv.weight"], b.gdn_packed),),
                    )
                else:
                    (yield from self._linear_steps(
                        self._linear_gdn_qkv,
                        b.norm,
                        weights[f"{op}.in_proj_qkv.weight"],
                        b.gdn_packed,
                        global_layer=self.layer_ids[layer],
                        weight_name=f"{global_op}.in_proj_qkv.weight",
                        row_parallel=False,
                        observer=observer,
                    ))
                self._body.conv.launch(
                    b.gdn_packed,
                    weights[f"{op}.conv1d.weight"].squeeze(1),
                    self._body.conv_state,
                    b.conv_slots[0, local],
                    b.conv_slots[1, local],
                    b.segment_lengths,
                    b.gdn_convolved,
                )
                if layer_observer is not None:
                    layer_observer(self.layer_ids[layer], "gdn_convolved", b.gdn_convolved)
                if self._grouped_gdn is None:
                    self._launch_linear(
                        self._linear_gdn_abz,
                        b.norm,
                        self._packed_gdn_abz_weights[local],
                        self._packed_gdn_abz,
                    )
                self._packed_gdn_abz_split.launch(
                    self._packed_gdn_abz,
                    b.gdn_a,
                    b.gdn_b,
                    b.gdn_gate.reshape(self.captured_m, -1),
                    **(dict(coefficients=self._gdn_coefficients,
                            a_log=weights[f"{op}.A_log"], dt_bias=weights[f"{op}.dt_bias"])
                       if self._gdn_coefficients is not None else {}),
                )
                if layer_observer is not None:
                    layer_observer(self.layer_ids[layer], "gdn_decay_input", b.gdn_a)
                    layer_observer(self.layer_ids[layer], "gdn_beta_input", b.gdn_b)
                self._observe_packed_columns(
                    observer,
                    self._grouped_gdn or self._linear_gdn_abz,
                    b.norm,
                    self.layer_ids[layer],
                    (
                        (
                            f"{global_op}.in_proj_a.weight",
                            weights[f"{op}.in_proj_a.weight"],
                            self._gdn_a_output,
                        ),
                        (
                            f"{global_op}.in_proj_b.weight",
                            weights[f"{op}.in_proj_b.weight"],
                            self._gdn_b_output,
                        ),
                    ),
                )
                self._body.gdn.launch(
                    b.gdn_convolved,
                    b.gdn_packed,
                    b.gdn_a,
                    b.gdn_b,
                    weights[f"{op}.A_log"],
                    weights[f"{op}.dt_bias"],
                    self._body.recurrent_state,
                    b.recurrent_slots[0, local],
                    b.recurrent_slots[1, local],
                    b.segment_lengths,
                    b.gdn_core,
                    coefficients=self._gdn_coefficients,
                )
                if layer_observer is not None:
                    layer_observer(self.layer_ids[layer], "gdn_normalized_qk", b.gdn_packed)
                    layer_observer(self.layer_ids[layer], "gdn_core", b.gdn_core)
                self._observe_packed_columns(
                    observer,
                    self._grouped_gdn or self._linear_gdn_abz,
                    b.norm,
                    self.layer_ids[layer],
                    (
                        (
                            f"{global_op}.in_proj_z.weight",
                            weights[f"{op}.in_proj_z.weight"],
                            self._gdn_z_output,
                        ),
                    ),
                )
                self._body.gdn_norm.launch(
                    b.gdn_core,
                    b.gdn_gate,
                    weights[f"{op}.norm.weight"],
                    b.gdn_normalized,
                )
                if layer_observer is not None:
                    layer_observer(self.layer_ids[layer], "gdn_normalized", b.gdn_normalized)
                row_input = b.gdn_normalized.reshape(self.captured_m, -1)
            else:
                op, local = f"{prefix}.self_attn", self._body._full_local[layer]
                global_op = f"{global_prefix}.self_attn"
                if self._grouped_attention is not None:
                    self._grouped_attention.launch(
                        b.norm,
                        tuple(weights[f"{op}.{name}_proj.weight"] for name in ("q", "k", "v")),
                        (b.attention_projected.reshape(self.captured_m, -1),
                         b.key.reshape(self.captured_m, -1),
                         b.value.reshape(self.captured_m, -1)), self._active_rows,
                    )
                    self._observe_packed_columns(
                        observer, self._grouped_attention, b.norm, self.layer_ids[layer],
                        ((f"{global_op}.q_proj.weight", weights[f"{op}.q_proj.weight"],
                          b.attention_projected.reshape(self.captured_m, -1)),),
                    )
                else:
                    (yield from self._linear_steps(
                        self._linear_attention_q,
                        b.norm,
                        weights[f"{op}.q_proj.weight"],
                        b.attention_projected.reshape(self.captured_m, -1),
                        global_layer=self.layer_ids[layer],
                        weight_name=f"{global_op}.q_proj.weight",
                        row_parallel=False,
                        observer=observer,
                    ))
                self._body.q_gate_split.launch(b.attention_projected, b.query, b.attention_gate)
                if self._grouped_attention is not None:
                    self._observe_packed_columns(
                        observer, self._grouped_attention, b.norm, self.layer_ids[layer],
                        ((f"{global_op}.k_proj.weight", weights[f"{op}.k_proj.weight"],
                          b.key.reshape(self.captured_m, -1)),
                         (f"{global_op}.v_proj.weight", weights[f"{op}.v_proj.weight"],
                          b.value.reshape(self.captured_m, -1))),
                    )
                else:
                    (yield from self._linear_steps(
                        self._linear_attention_kv,
                        b.norm,
                        weights[f"{op}.k_proj.weight"],
                        b.key.reshape(self.captured_m, -1),
                        global_layer=self.layer_ids[layer],
                        weight_name=f"{global_op}.k_proj.weight",
                        row_parallel=False,
                        observer=observer,
                    ))
                    (yield from self._linear_steps(
                        self._linear_attention_kv,
                        b.norm,
                        weights[f"{op}.v_proj.weight"],
                        b.value.reshape(self.captured_m, -1),
                        global_layer=self.layer_ids[layer],
                        weight_name=f"{global_op}.v_proj.weight",
                        row_parallel=False,
                        observer=observer,
                    ))
                self._body.q_norm.launch(
                    b.query.reshape(-1, b.query.shape[-1]),
                    weights[f"{op}.q_norm.weight"],
                    b.normalized_query.reshape(-1, b.query.shape[-1]),
                )
                self._body.k_norm.launch(
                    b.key.reshape(-1, b.key.shape[-1]),
                    weights[f"{op}.k_norm.weight"],
                    b.normalized_key.reshape(-1, b.key.shape[-1]),
                )
                self._body.mrope_apply.launch(
                    b.normalized_query,
                    b.normalized_key,
                    b.cosine,
                    b.sine,
                    b.rotated_query,
                    b.rotated_key,
                )
                if layer_observer is not None:
                    layer_observer(self.layer_ids[layer], "full_query", b.rotated_query)
                    layer_observer(self.layer_ids[layer], "full_key", b.rotated_key)
                    layer_observer(self.layer_ids[layer], "full_value", b.value)
                self._body.page_writes[local].launch(
                    self._body.arena.key,
                    self._body.arena.value,
                    self._body.arena.written,
                    b.page_destination,
                    b.token_offsets,
                    b.rotated_key,
                    b.value,
                )
                self._body.attentions[local].launch(
                    b.rotated_query,
                    self._body.arena.key,
                    self._body.arena.value,
                    b.page_table,
                    b.sequence_lengths,
                    b.attention_output,
                )
                if layer_observer is not None:
                    layer_observer(self.layer_ids[layer], "full_attention", b.attention_output)
                self._body.sigmoid_gate.launch(
                    b.attention_output, b.attention_gate, b.rotated_query
                )
                if layer_observer is not None:
                    layer_observer(self.layer_ids[layer], "full_gated", b.rotated_query)
                row_input = b.rotated_query.reshape(self.captured_m, -1)
            row_suffix = (
                "out_proj.weight" if kind is AttentionKind.GATED_DELTA_NET else "o_proj.weight"
            )
            row_weight = weights[f"{op}.{row_suffix}"]
            attention_norm_fused = (yield from self._linear_steps(
                self._linear_output,
                row_input,
                row_weight,
                b.update,
                global_layer=self.layer_ids[layer],
                weight_name=f"{global_op}.{row_suffix}",
                row_parallel=True,
                observer=observer,
                fused_residual_norm=(
                    None
                    if layer_observer is not None
                    else (
                        b.hidden,
                        b.after_attention,
                        weights[f"{prefix}.post_attention_layernorm.weight"],
                        b.norm,
                    )
                ),
            ))
            if not attention_norm_fused:
                self._body.residual_norm.launch(
                    b.hidden,
                    b.update,
                    b.after_attention,
                    weights[f"{prefix}.post_attention_layernorm.weight"],
                    b.norm,
                )
            if layer_observer is not None:
                layer_observer(self.layer_ids[layer], "attention_mixed", b.update)
                layer_observer(self.layer_ids[layer], "attention", b.after_attention)
                layer_observer(self.layer_ids[layer], "mlp_input", b.norm)
            mlp = f"{prefix}.mlp"
            global_mlp = f"{global_prefix}.mlp"
            row_weight = weights[f"{mlp}.down_proj.weight"]
            if layer + 1 < len(self._body.schedule):
                next_prefix = f"model.language_model.layers.{layer + 1}"
                mlp_residual_norm: tuple[Tensor, Tensor, Tensor, Tensor] | None = (
                    b.after_attention,
                    b.hidden,
                    weights[f"{next_prefix}.input_layernorm.weight"],
                    b.norm,
                )
            elif self._is_last_stage:
                mlp_residual_norm = (
                    b.after_attention,
                    b.hidden,
                    weights["model.language_model.norm.weight"],
                    b.norm,
                )
            else:
                mlp_residual_norm = None
            if getattr(self, "_is_moe", False):
                with self._moe_branches.fork_shared():
                    self._launch_linear(
                        self._linear_mlp_gate_up,
                        b.norm,
                        self._packed_gate_up_weights[layer],
                        self._packed_mlp,
                    )
                    self._packed_mlp_gate.launch(self._packed_mlp, b.mlp_active)
                    self._launch_linear(self._linear_mlp_down, b.mlp_active, row_weight, b.update)
                    self._launch_linear(
                        self._linear_moe_shared_gate,
                        b.norm,
                        weights[f"{mlp}.shared_expert_gate.weight"],
                        self._moe_shared_gate,
                    )
                if layer_observer is not None:
                    # These operands are still being produced on the shared
                    # branch. Snapshot on that stream; the normal join below
                    # orders the snapshots before any later buffer reuse.
                    with torch.cuda.stream(self._moe_branches.shared):
                        layer_observer(self.layer_ids[layer], "shared_active", b.mlp_active)
                        layer_observer(self.layer_ids[layer], "shared_down", b.update)
                        layer_observer(self.layer_ids[layer], "shared_gate", self._moe_shared_gate)
                self._launch_linear(
                    self._linear_moe_router,
                    b.norm,
                    weights[f"{mlp}.gate.weight"],
                    self._moe_router_logits,
                )
                routed_operands = (
                    b.norm,
                    self._moe_router_logits,
                    b.row_activity,
                    self._active_rows,
                    weights[f"{mlp}.experts.gate_up_proj"],
                    weights[f"{mlp}.experts.down_proj"],
                    self._moe_routed,
                )
                if layer_observer is None:
                    self._routed_experts.launch(*routed_operands)
                else:
                    self._routed_experts.launch_qualification(
                        *routed_operands,
                        lambda phase, value, global_layer=self.layer_ids[layer]: layer_observer(
                            global_layer, f"routed_{phase}", value
                        ),
                    )
                if layer_observer is not None:
                    layer_observer(self.layer_ids[layer], "router_logits", self._moe_router_logits)
                    layer_observer(
                        self.layer_ids[layer], "router_top_ids", self._routed_experts.top_ids
                    )
                    layer_observer(self.layer_ids[layer], "route", self._routed_experts.route_ids)
                    layer_observer(
                        self.layer_ids[layer],
                        "route_weights",
                        self._routed_experts.route_weights,
                    )
                    layer_observer(self.layer_ids[layer], "routed_partial", self._moe_routed)
                self._moe_branches.join_shared()
                self._observe_packed_columns(
                    observer,
                    self._linear_mlp_gate_up,
                    b.norm,
                    self.layer_ids[layer],
                    (
                        (
                            f"{global_mlp}.shared_expert.gate_proj.weight",
                            weights[f"{mlp}.gate_proj.weight"],
                            self._mlp_gate_output,
                        ),
                        (
                            f"{global_mlp}.shared_expert.up_proj.weight",
                            weights[f"{mlp}.up_proj.weight"],
                            self._mlp_up_output,
                        ),
                    ),
                )
                self._moe_combine.launch(
                    b.update,
                    self._moe_shared_gate,
                    self._moe_routed,
                    b.update,
                )
                if layer_observer is not None:
                    layer_observer(self.layer_ids[layer], "moe_local", b.update)
                if mlp_residual_norm is not None and observer is None and layer_observer is None:
                    yield _defer_collective(
                        self._tp_sum_residual_norm, b.update, *mlp_residual_norm
                    )
                    mlp_norm_fused = True
                else:
                    (yield _defer_collective(self._tp_sum, b.update))
                    mlp_norm_fused = False
            else:
                self._launch_linear(
                    self._linear_mlp_gate_up,
                    b.norm,
                    self._packed_gate_up_weights[layer],
                    self._packed_mlp,
                )
                self._observe_packed_columns(
                    observer,
                    self._linear_mlp_gate_up,
                    b.norm,
                    self.layer_ids[layer],
                    (
                        (
                            f"{global_mlp}.gate_proj.weight",
                            weights[f"{mlp}.gate_proj.weight"],
                            self._mlp_gate_output,
                        ),
                        (
                            f"{global_mlp}.up_proj.weight",
                            weights[f"{mlp}.up_proj.weight"],
                            self._mlp_up_output,
                        ),
                    ),
                )
                self._packed_mlp_gate.launch(self._packed_mlp, b.mlp_active)
                mlp_norm_fused = (yield from self._linear_steps(
                    self._linear_mlp_down,
                    b.mlp_active,
                    row_weight,
                    b.update,
                    global_layer=self.layer_ids[layer],
                    weight_name=f"{global_mlp}.down_proj.weight",
                    row_parallel=True,
                    observer=observer,
                    fused_residual_norm=(None if layer_observer is not None else mlp_residual_norm),
                ))
            if mlp_residual_norm is not None:
                if not mlp_norm_fused:
                    self._body.residual_norm.launch(
                        b.after_attention,
                        b.update,
                        b.hidden,
                        mlp_residual_norm[2],
                        b.norm,
                    )
            else:
                self._body.residual.launch(b.after_attention, b.update, b.hidden)
            if layer_observer is not None:
                layer_observer(self.layer_ids[layer], "mlp_output", b.update)
                layer_observer(self.layer_ids[layer], "mlp", b.hidden)
        if not self._is_last_stage:
            self._pipeline(b.hidden)
        else:
            (yield from self._linear_steps(
                self._linear_logits,
                b.norm,
                weights["lm_head.weight"],
                b.logits,
                global_layer=None,
                weight_name="lm_head.weight",
                row_parallel=False,
                observer=observer,
            ))


@dataclass(slots=True)
class Qwen3_5TP2PP2GraphSlot:
    """One captured rank graph; every tensor and NCCL launch address is fixed."""

    body: Qwen3_5TP2PP2DecodeBody
    resource: RankReplayResources
    stream: Stream
    graph: CUDAGraph

    _overlap: tuple[
        CUDAGraph, DecodeInterleave, tuple[Qwen3_5TP2PP2DecodeBody, ...]
    ] | None = None
    _use_overlap: bool = False
    overlap_measurements: list[tuple[tuple[int, ...], float]] | None = None

    @classmethod
    def capture(
        cls,
        body: Qwen3_5TP2PP2DecodeBody,
        resource: RankReplayResources,
        *,
        pool: tuple[int, int] | None = None,
        stream: Stream | None = None,
        retain: bool = True,
    ) -> Qwen3_5TP2PP2GraphSlot:
        if resource.rank != body.rank or body.captured_m < 1:
            raise Qwen3_5ParallelPerformanceError("capture resource differs from rank bucket")
        if retain:
            body._mark_graph_capture_started()
        try:
            body.prepare_scratch(resource)
            device = body.hidden.device
            dist.barrier()
            if stream is None:
                stream = Stream(device=device)  # type: ignore[no-untyped-call]
            stream.wait_stream(torch.cuda.current_stream(device))
            for _ in range(2):
                with torch.cuda.stream(stream):
                    body.launch_capture_warmup()
                stream.synchronize()
                dist.barrier()
            captured = CUDAGraph(keep_graph=True)
            with capture_graph(captured, pool=pool, stream=stream):
                body.launch()
            captured.instantiate()
            stream.synchronize()
            dist.barrier()
            slot = cls(body, resource, stream, captured)
            slot.prepare_decode_overlap(retain=retain)
            return slot
        except Qwen3_5ParallelPerformanceError:
            raise
        except (RuntimeError, ValueError) as error:
            raise Qwen3_5ParallelPerformanceError(f"CUDA Graph capture failed: {error}") from error

    def prepare_decode_overlap(self, *, retain: bool = True) -> None:
        """Select only startup-captured candidates; admission audits both variants."""
        candidates = getattr(self.body, "_decode_overlap_candidates", ())
        if not candidates:
            return
        policy = collective_parameters()
        prepare = decode_calibration_prepare(
            rows=self.body.captured_m, vocabulary_size=self.body._vocab_size,
            device=self.body.hidden.device,
            reset=lambda: self.body.prepare_scratch(self.resource),
            bind_tokens=self.body.fill_token_ids,
        )
        baseline = measure_decode_graph(
            self.graph, prepare, self.stream, policy.decode_overlap_samples
        )
        best = 1 + policy.decode_overlap_min_gain_percent / 100
        self.overlap_measurements = [((self.body.captured_m,), baseline)]
        for children in candidates:
            schedule = DecodeInterleave(len(children), self.body.hidden.device)

            def launch(
                children: tuple[Qwen3_5TP2PP2DecodeBody, ...] = children,
                schedule: DecodeInterleave = schedule,
            ) -> None:
                start = 0
                for child in children:
                    stop = start + child.captured_m
                    _copy_decode_input_rows(self.body.buffers, child.buffers, start, stop)
                    for name in ("embedding_mask", "feature_rows"):
                        source = getattr(self.body, name)
                        if source is not None:
                            getattr(child, name).copy_(source[start:stop])
                    start = stop
                schedule.launch([child.launch_steps(None) for child in children])
                start = 0
                for child in children:
                    stop = start + child.captured_m
                    for name in ("hidden", "norm", "logits"):
                        getattr(self.body.buffers, name)[start:stop].copy_(
                            getattr(child.buffers, name)
                        )
                    start = stop

            prepare()
            for child in children:
                child.prepare_scratch(self.resource)
                if retain:
                    child._mark_graph_capture_started()
            self.stream.wait_stream(torch.cuda.current_stream(self.body.hidden.device))
            for _ in range(2):
                with torch.cuda.stream(self.stream):
                    launch()
                self.synchronize()
                dist.barrier()
            graph = CUDAGraph(keep_graph=True)
            with capture_graph(graph, stream=self.stream):
                launch()
            graph.instantiate()
            self.synchronize()
            dist.barrier()
            control, elapsed = compare_decode_graphs(
                self.graph, graph, prepare, self.stream, policy.decode_overlap_samples
            )
            self.overlap_measurements.append(((self.body.captured_m,), control))
            self.overlap_measurements.append((tuple(c.captured_m for c in children), elapsed))
            gain = control / elapsed
            if gain > best:
                if self._overlap is not None:
                    self._overlap[0].reset()
                self._overlap = (graph, schedule, children)
                best = gain
            else:
                graph.reset()
        prepare()
        self.synchronize()
        dist.barrier()
        logging.getLogger(__name__).info(
            "decode overlap rank=%d M=%d calibrated_ms=%s selected=%s",
            self.body.rank, self.body.captured_m,
            format_decode_measurements(self.overlap_measurements),
            DecodeOverlapPlan(
                tuple(c.captured_m for c in self._overlap[2]),
                self._overlap[2][0]._decode_overlap_estimated,
                self._overlap[2][0]._decode_overlap_collective_threads,
            ) if self._overlap else DecodeOverlapPlan(),
        )

    def select_decode_overlap(self, independent_rows: bool) -> None:
        self._use_overlap = independent_rows and self._overlap is not None

    @property
    def owned_decode_graph(self) -> CUDAGraph:
        """The already-selected graph for independent one-token request owners."""
        return self._overlap[0] if self._overlap is not None else self.graph

    @property
    def graphs(self) -> tuple[CUDAGraph, ...]:
        return (self.graph, self._overlap[0]) if self._overlap else (self.graph,)

    @property
    def hidden(self) -> Tensor:
        return self.body.hidden

    @property
    def logits(self) -> Tensor | None:
        return self.body.logits

    @property
    def vocab_span(self) -> AxisSpan | None:
        return self.body.vocab_span

    def replay(self) -> None:
        try:
            with torch.cuda.device(self.body.hidden.device), torch.cuda.stream(self.stream):
                graph = self._overlap[0] if self._use_overlap and self._overlap else self.graph
                graph.replay()
        except RuntimeError as error:
            raise Qwen3_5ParallelPerformanceError(f"CUDA Graph replay failed: {error}") from error

    def synchronize(self) -> None:
        try:
            self.stream.synchronize()
        except RuntimeError as error:
            raise Qwen3_5ParallelPerformanceError(
                f"CUDA Graph completion failed: {error}"
            ) from error


def _private_activation_storage_ids(body: Qwen3_5TP2PP2DecodeBody) -> set[int]:
    """Return every lane-private tensor storage; cache and weights are excluded."""

    tensors: list[Tensor] = []
    for name in _Buffers.__annotations__:
        value = getattr(body.buffers, name, None)
        if not isinstance(value, Tensor):
            raise Qwen3_5ParallelPerformanceError(
                "wavefront lane does not own the complete fixed activation ABI"
            )
        tensors.append(value)
    tensors.extend((body._packed_mlp, body._packed_gdn_abz))
    tensors.extend(getattr(getattr(body, "_body", None), "packed_scratch", ()))
    coefficients = getattr(body, "_gdn_coefficients", None)
    if coefficients is not None:
        tensors.append(coefficients)
    if body.embedding_mask is not None:
        tensors.append(body.embedding_mask)
    feature_rows = getattr(body, "feature_rows", None)
    if feature_rows is not None:
        tensors.append(feature_rows)
    storage_ids = {value.untyped_storage().data_ptr() for value in tensors}
    if len(storage_ids) != len(tensors):
        raise Qwen3_5ParallelPerformanceError(
            "wavefront lane aliases its private activation or workspace storage"
        )
    return storage_ids


@dataclass(slots=True)
class Qwen3_5TP2PP2WavefrontGraphSlot:
    """One generic PP2 graph containing two fixed-M activation waves.

    Both waves share immutable weights, M-RoPE tables, and rank-local cache/state
    backing, but own every activation and launch workspace.  The request rows in
    A and B must be disjoint.  Both waves may reuse the same rank-local scratch,
    including for padding in both waves: one stream executes every A operation
    before B, and no operation reads A scratch after A completes.  Cross-stage
    overlap occurs on different ranks and therefore on different arenas.
    """

    lane_a: Qwen3_5TP2PP2DecodeBody
    lane_b: Qwen3_5TP2PP2DecodeBody
    resource: RankReplayResources
    stream: Stream
    graph: CUDAGraph
    _logits: Tensor | None

    @staticmethod
    def _validate_lanes(
        lane_a: Qwen3_5TP2PP2DecodeBody,
        lane_b: Qwen3_5TP2PP2DecodeBody,
        resource: RankReplayResources,
    ) -> None:
        if (
            not isinstance(lane_a, Qwen3_5TP2PP2DecodeBody)
            or not isinstance(lane_b, Qwen3_5TP2PP2DecodeBody)
            or lane_a is lane_b
            or lane_a.captured_m != lane_b.captured_m
            or lane_a.captured_m < 2
            or (lane_a.lane_id, lane_b.lane_id) != (0, 1)
            or resource.rank != lane_a.rank
            or lane_a.rank != lane_b.rank
            or lane_a.stage != lane_b.stage
            or lane_a.stage not in (0, 1)
            or lane_a.tensor_rank != lane_b.tensor_rank
            or lane_a.arena is not lane_b.arena
            or lane_a.arena.rank_id != lane_a.rank
            or lane_a._body is lane_b._body
            or lane_a._body.model is not lane_b._body.model
            or lane_a._body.model.weights is not lane_b._body.model.weights
            or lane_a._packed_gate_up_weights is not lane_b._packed_gate_up_weights
            or lane_a._packed_gdn_abz_weights is not lane_b._packed_gdn_abz_weights
            or lane_a._mrope_tables is not lane_b._mrope_tables
            or lane_a._tp_group is not lane_b._tp_group
            or lane_a._pp_group is not lane_b._pp_group
            or lane_a._pp_peer != lane_b._pp_peer
            or lane_a.layer_ids != lane_b.layer_ids
            or lane_a.full_layer_ids != lane_b.full_layer_ids
            or lane_a.gdn_layer_ids != lane_b.gdn_layer_ids
            or lane_a.embedding_span != lane_b.embedding_span
            or lane_a.vocab_span != lane_b.vocab_span
            or lane_a._vocab_size != lane_b._vocab_size
            or lane_a._graph_capture_started
            or lane_b._graph_capture_started
            or lane_a._arena_rebind_failed
            or lane_b._arena_rebind_failed
        ):
            raise Qwen3_5ParallelPerformanceError(
                "wavefront capture requires matching independent lane 0/lane 1 bodies"
            )
        for name in ("key", "value", "written", "states"):
            if getattr(lane_a._body.arena, name) is not getattr(lane_b._body.arena, name):
                raise Qwen3_5ParallelPerformanceError(
                    "wavefront lanes differ from their shared cache/state backing"
                )
        if _private_activation_storage_ids(lane_a) & _private_activation_storage_ids(lane_b):
            raise Qwen3_5ParallelPerformanceError(
                "wavefront lanes alias private activation or workspace storage"
            )
        for name in _Buffers.__annotations__:
            left, right = getattr(lane_a.buffers, name), getattr(lane_b.buffers, name)
            if (
                left.shape != right.shape
                or left.dtype is not right.dtype
                or left.device != right.device
                or not left.is_contiguous()
                or not right.is_contiguous()
            ):
                raise Qwen3_5ParallelPerformanceError(
                    "wavefront lanes differ from the fixed activation ABI"
                )
        spec = lane_a.arena.spec
        table_pages = lane_a.buffers.page_table.shape[1]
        single_stage = bool(
            getattr(lane_a, "_is_first_stage", lane_a.stage == 0)
            and getattr(lane_a, "_is_last_stage", lane_a.stage == 1)
        )
        required_owner_slots = lane_a.captured_m * (1 if single_stage else 2)
        if (
            lane_a.arena.device != lane_a.hidden.device
            or table_pages <= 0
            or spec.state_slot_capacity < required_owner_slots
            or spec.page_capacity < minimum_kv_page_capacity(table_pages, 2 * lane_a.captured_m)
            or spec.scratch_capacity < lane_a.captured_m
            or resource.scratch_arena_id not in spec.scratch_arena_ids
        ):
            raise Qwen3_5ParallelPerformanceError(
                "wavefront shared arena cannot cover its combined fixed-M ABI"
            )
        lane_a_logits = lane_a.logits
        expected_logits_shape = None if lane_a_logits is None else lane_a_logits.shape
        for lane in (lane_a, lane_b):
            logits = lane.logits
            if (
                lane.hidden.ndim != 2
                or lane.hidden.shape != lane_a.hidden.shape
                or lane.hidden.shape[0] != lane.captured_m
                or lane.hidden.dtype is not torch.bfloat16
                or lane.hidden.device != lane_a.hidden.device
                or not lane.hidden.is_contiguous()
            ):
                raise Qwen3_5ParallelPerformanceError(
                    "wavefront lane hidden buffer differs from fixed M"
                )
            if getattr(lane, "_is_first_stage", lane.stage == 0) and not getattr(
                lane, "_is_last_stage", lane.stage == 1
            ):
                if (
                    logits is not None
                    or lane.embedding_mask is None
                    or lane.embedding_mask.shape != (lane.captured_m, 1)
                    or lane.embedding_mask.dtype is not torch.bfloat16
                    or lane.embedding_mask.device != lane.hidden.device
                    or not lane.embedding_mask.is_contiguous()
                ):
                    raise Qwen3_5ParallelPerformanceError(
                        "first-stage wavefront lane differs from embedding/logits ABI"
                    )
            elif getattr(lane, "_is_last_stage", lane.stage == 1) and (
                logits is None
                or (
                    lane.embedding_mask is not None
                    and not getattr(lane, "_is_first_stage", lane.stage == 0)
                )
                or logits.ndim != 2
                or logits.shape != expected_logits_shape
                or logits.shape[0] != lane.captured_m
                or logits.dtype is not torch.bfloat16
                or logits.device != lane.hidden.device
                or not logits.is_contiguous()
            ):
                raise Qwen3_5ParallelPerformanceError(
                    "final-stage wavefront lane logits differ from fixed M"
                )

    @staticmethod
    def _launch_waves(
        lane_a: Qwen3_5TP2PP2DecodeBody,
        lane_b: Qwen3_5TP2PP2DecodeBody,
        logits: Tensor | None,
    ) -> None:
        micro_m = lane_a.captured_m
        lane_a.launch()
        if logits is not None:
            lane_a_logits = lane_a.logits
            assert lane_a_logits is not None
            logits[:micro_m].copy_(lane_a_logits)
        lane_b.launch()
        if logits is not None:
            lane_b_logits = lane_b.logits
            assert lane_b_logits is not None
            logits[micro_m:].copy_(lane_b_logits)

    @classmethod
    def capture(
        cls,
        lane_a: Qwen3_5TP2PP2DecodeBody,
        lane_b: Qwen3_5TP2PP2DecodeBody,
        resource: RankReplayResources,
    ) -> Qwen3_5TP2PP2WavefrontGraphSlot:
        cls._validate_lanes(lane_a, lane_b, resource)
        lane_a._mark_graph_capture_started()
        lane_b._mark_graph_capture_started()
        try:
            # The request rows are disjoint, while ownerless padding rows may be
            # shared: this rank's one stream finishes all A operations before B
            # and never reads A scratch again.  Other PP stages own other arenas.
            lane_a.prepare_scratch(resource)
            lane_b.prepare_scratch(resource)
            device = lane_a.hidden.device
            logits: Tensor | None = None
            if getattr(lane_a, "_is_last_stage", lane_a.stage == 1):
                lane_logits = lane_a.logits
                assert lane_logits is not None
                logits = torch.empty(
                    (2 * lane_a.captured_m, lane_logits.shape[1]),
                    dtype=torch.bfloat16,
                    device=device,
                )
            dist.barrier()
            stream = Stream(device=device)  # type: ignore[no-untyped-call]
            stream.wait_stream(torch.cuda.current_stream(device))
            for _ in range(2):
                with torch.cuda.stream(stream):
                    cls._launch_waves(lane_a, lane_b, logits)
                stream.synchronize()
                dist.barrier()
            captured = CUDAGraph()
            with capture_graph(captured, stream=stream):
                cls._launch_waves(lane_a, lane_b, logits)
            stream.synchronize()
            dist.barrier()
            return cls(lane_a, lane_b, resource, stream, captured, logits)
        except Qwen3_5ParallelPerformanceError:
            raise
        except (RuntimeError, ValueError) as error:
            raise Qwen3_5ParallelPerformanceError(f"CUDA Graph capture failed: {error}") from error

    @property
    def rank(self) -> int:
        return self.lane_a.rank

    @property
    def stage(self) -> int:
        return self.lane_a.stage

    @property
    def micro_m(self) -> int:
        return self.lane_a.captured_m

    @property
    def captured_m(self) -> int:
        return 2 * self.micro_m

    @property
    def hidden(self) -> Tensor:
        return self.lane_a.hidden

    @property
    def logits(self) -> Tensor | None:
        return self._logits

    @property
    def vocab_span(self) -> AxisSpan | None:
        return self.lane_a.vocab_span

    def replay(self) -> None:
        try:
            with torch.cuda.device(self.lane_a.hidden.device), torch.cuda.stream(self.stream):
                self.graph.replay()
        except RuntimeError as error:
            raise Qwen3_5ParallelPerformanceError(f"CUDA Graph replay failed: {error}") from error

    def synchronize(self) -> None:
        try:
            self.stream.synchronize()
        except RuntimeError as error:
            raise Qwen3_5ParallelPerformanceError(
                f"CUDA Graph completion failed: {error}"
            ) from error


__all__ = (
    "CANDIDATE_ID",
    "COMPOSITE_CHILD_CLOSURE_ID",
    "GDN_CONV_STATE_ID",
    "GDN_RECURRENT_STATE_ID",
    "GRAPH_SAFETY_CONTRACT_ID",
    "PROGRAM_ID",
    "REPLAY_ABI_ID",
    "ROW_MAP_SCHEMA_ID",
    "SCRATCH_SCHEMA_ID",
    "Qwen3_5DecodeGraphProgram",
    "Qwen3_5DecodeGraphSlot",
    "Qwen3_5LayerObserver",
    "Qwen3_5LinearLeafEvent",
    "Qwen3_5LinearObserver",
    "Qwen3_5ParallelPerformanceError",
    "Qwen3_5PerformanceRejected",
    "Qwen3_5TP2PP2DecodeBody",
    "Qwen3_5TP2PP2GraphSlot",
    "Qwen3_5TP2PP2WavefrontGraphSlot",
    "minimum_kv_page_capacity",
)
