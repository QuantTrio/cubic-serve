"""Fixed-address text-block bindings shared by model-owned graph traversals.

This module owns activation storage and bound calls. Models supply semantic
weights/geometry and order; the rank executor supplies state and collectives.
It does not capture, lease, admit, or silently fall back to another execution mode.
"""

from __future__ import annotations

from typing_extensions import TypeAliasType

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import Tensor

from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.ops.cuda.collective import FixedRankBF16SumPrimitive
from cubic_serve.ops.cuda.compiled_artifact import bind_tensor
from cubic_serve.ops.cuda.launch_cost import device_work_cost, scratch_ranges_disjoint
from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
from cubic_serve.ops.cuda.linear_attention import (
    CausalConv1DDecodePrimitive,
    GatedDeltaDecodePrimitive,
)
from cubic_serve.ops.cuda.moe.fused import MoeBranchStreams
from cubic_serve.ops.cuda.normalization.rms_norm import ZeroCenteredRMSNormPrimitive
from cubic_serve.ops.cuda.pointwise.qwen_decode import GatedRMSNormPrimitive
from cubic_serve.ops.cuda.pointwise.residual_streams import (
    bind_scaled_activation,
    bind_stream_combine,
    bind_stream_combine_norm,
    bind_stream_mix,
)
from cubic_serve.ops.cuda.position.qwen_mrope_table import (
    MRoPECosSinGatherPrimitive,
    MRoPETableEvidence,
)
from cubic_serve.ops.strategies import launch_parameters
from cubic_serve.quantization.registry import get_extension

if TYPE_CHECKING:
    from cubic_serve.runtime.cache.pages import CacheArena

Matrix = TypeAliasType("Matrix", Tensor | QuantizedMatrix)


@dataclass(frozen=True, slots=True)
class BoundTextBlock:
    output: Tensor
    calls: tuple[Callable[[], object], ...]

    def launch(self) -> None:
        for call in self.calls:
            call()


class GraphScratchPool(dict[tuple, Tensor]):
    """Private graph-family buffers plus explicitly serial topology scratch."""

    def __init__(self, serial: dict[tuple, Tensor] | None = None, *,
                 capacities: tuple[int, ...] = (), budgets: dict | None = None) -> None:
        super().__init__()
        if len(set(capacities)) != len(capacities) or any(
                type(m) is not int or m <= 0 for m in capacities):
            raise ValueError("scratch portfolio capacities must be distinct positive integers")
        self.serial = {} if serial is None else serial
        self.capacities = capacities
        self._budgets = {} if budgets is None else budgets
        self._owner = object()

    def sibling(self) -> GraphScratchPool:
        return GraphScratchPool(self.serial, capacities=self.capacities, budgets=self._budgets)

    def serial_budget(self, role: str, private_bytes: int, dtype, device) -> int:
        """Combine each serial owner's original footprint exactly once."""
        owners = self._budgets.setdefault((role, dtype, device), {})
        if owners.setdefault(self._owner, private_bytes) != private_bytes:
            raise ValueError("a graph family changed its declared scratch footprint")
        return sum(owners.values())

    def clear(self) -> None:
        super().clear()
        self.serial.clear()
        self._budgets.clear()


def full_attention_partitions(capacity: int, *, max_tokens: int, query_heads: int,
                              kv_heads: int, head_dim: int, device,
                              shared_scratch=None) -> int:
    """Spend the existing portfolio's bytes on a shared, fixed reduction plan."""
    from cubic_serve.ops.cuda.attention.paged_decode import paged_gqa_kv_splits
    from cubic_serve.ops.strategies import attention_precision

    programs = (kv_heads * ((query_heads // kv_heads + 15) // 16)
                if attention_precision() == "auto" else query_heads)
    sms = torch.cuda.get_device_properties(device).multi_processor_count

    def original(rows):
        return paged_gqa_kv_splits(rows, max_tokens=max_tokens, head_programs=programs,
                                  multiprocessors=sms, full_context=True)

    if not isinstance(shared_scratch, GraphScratchPool) or not shared_scratch.capacities:
        return original(capacity)
    if capacity not in shared_scratch.capacities:
        raise ValueError("attention capacity is outside its declared scratch portfolio")
    footprints: dict[int, int] = {}
    for rows in shared_scratch.capacities:
        splits = original(rows)
        footprints[splits] = max(footprints.get(splits, 0), rows * splits)
    private_rows = sum(footprints.values())
    budgets = []
    for role, width in (("attention.partial", head_dim), ("attention.statistics", 2)):
        row_bytes = query_heads * width * 4
        budgets.append(shared_scratch.serial_budget(
            role, private_rows * row_bytes, torch.float32, device) // row_bytes)
    partitions = min(budgets) // capacity
    return min(max(footprints), 1 << (partitions.bit_length() - 1))


class TextGraphBindings:
    """Startup-only storage/binding helpers, with explicitly reused activations.

    A role identifies a mathematical live range. Equal role/shape buffers may
    be reused across serial layers, never across overlapping graph leases.
    State is always supplied externally and never allocated by this scratch pool.
    """

    def __init__(
        self, operators: PackedLinearOperators, device: torch.device, dtype: torch.dtype,
        *, cache_arena: CacheArena | None = None,
        cache_source_slots: Tensor | None = None, cache_destination_slots: Tensor | None = None,
        shared_scratch: dict[tuple, Tensor] | None = None,
    ) -> None:
        if device.type != "cuda" or dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("text graph binding requires CUDA FP16/BF16 activations")
        self.operators, self.device, self.dtype = operators, device, dtype
        self.cache_encoding = None
        self._cache_arena = None
        self._cache_source_slots, self._cache_destination_slots = (
            cache_source_slots, cache_destination_slots)
        self._cache_metadata_identity = None
        self._cache_binding = None
        if cache_arena is not None and any(
            spec.dtype.quantized for spec in (cache_arena.layout.storage.key,
                                             cache_arena.layout.storage.value)
        ):
            from cubic_serve.ops.cuda.cache_quantization import paged_cache_encoding
            if cache_source_slots is None or cache_destination_slots is None:
                raise ValueError("encoded cache requires engine-owned state slot maps")
            self._cache_arena = cache_arena
            self.cache_encoding = paged_cache_encoding(cache_arena)
        self._buffers: dict[tuple[str, tuple[int, ...], torch.dtype, bool, bool], Tensor] = {}
        self._shared_scratch = shared_scratch
        self._claimed_scratch: set[tuple] = set()
        self._blocks: list[BoundTextBlock] = []
        self._extra_scratch: list[Tensor] = []
        self.local_tp_routers: set[int] = set()
        self.moe_branch_costs: dict[int, tuple[float, float, bool]] = {}
        self._moe_branches: MoeBranchStreams | None = None
        self._shared_expert_bindings: TextGraphBindings | None = None
        self._projection_branches: MoeBranchStreams | None = None
        self.projection_branch_costs: dict[tuple[int, ...], tuple[float, float, bool]] = {}
        self._residual_norm_choices: dict[tuple, bool] = {}
        self._rope_gathers: dict[tuple, tuple[Tensor, Tensor, MRoPECosSinGatherPrimitive]] = {}
        self._coalesced_sums: dict[
            tuple[FixedRankBF16SumPrimitive, FixedRankBF16SumPrimitive],
            tuple[Tensor, FixedRankBF16SumPrimitive] | None,
        ] = {}

    @property
    def scratch(self) -> tuple[Tensor, ...]:
        shared = self._shared_expert_bindings
        return (*self._buffers.values(), *self._extra_scratch,
                *(self._cache_binding.scratch if self._cache_binding is not None else ()),
                *(shared.scratch if shared is not None else ()))

    def buffer(self, role: str, *shape: int, dtype: torch.dtype | None = None,
               private: bool = False, serial: bool = False) -> Tensor:
        if not role or not shape or min(shape) <= 0:
            raise ValueError("text graph scratch requires a named positive shape")
        if type(serial) is not bool or (serial and private):
            raise ValueError("serial graph scratch requires an explicit shared live range")
        resolved_dtype = self.dtype if dtype is None else dtype
        key = role, shape, resolved_dtype, private, serial
        if key not in self._buffers:
            family = ((role, resolved_dtype, self.device) if serial else
                      (role, shape[1:], resolved_dtype, self.device))
            shared = self._shared_scratch
            if serial and isinstance(shared, GraphScratchPool):
                shared = shared.serial
            if shared is not None and not private and family not in self._claimed_scratch:
                # Only different graph owners borrow this backing. Distinct
                # shapes within this body keep their original disjoint storage.
                backing = shared.get(family)
                extent = math.prod(shape) if serial else shape[0]
                if backing is None or (backing.numel() if serial else backing.shape[0]) < extent:
                    backing = torch.empty((extent,) if serial else shape,
                                          dtype=resolved_dtype, device=self.device)
                    shared[family] = backing
                self._buffers[key] = (backing[:extent].view(shape) if serial else backing[:shape[0]])
                self._claimed_scratch.add(family)
            else:
                self._buffers[key] = torch.empty(shape, dtype=resolved_dtype, device=self.device)
        return self._buffers[key]

    def _rope_gather(self, positions, cosine_table, sine_table, cosine, sine, *, evidence):
        # The graph owner supplies immutable tables. Reuse their already
        # attested fixed-ABI gather across serial layers in this graph only.
        # New storage, evidence, ABI or a tracked mutation requires revalidation.
        try:
            versions = cosine_table._version, sine_table._version
        except RuntimeError:
            return MRoPECosSinGatherPrimitive(
                positions, cosine_table, sine_table, cosine, sine, evidence=evidence,
            )
        key = (
            bind_tensor(cosine_table), bind_tensor(sine_table), versions, evidence,
            tuple((tuple(t.shape), t.stride(), t.dtype, t.device)
                  for t in (positions, cosine, sine)),
        )
        if key not in self._rope_gathers:
            self._rope_gathers[key] = (
                cosine_table, sine_table, MRoPECosSinGatherPrimitive(
                    positions, cosine_table, sine_table, cosine, sine, evidence=evidence,
                ),
            )
        return self._rope_gathers[key][2]

    def _block(self, output: Tensor, *calls: Callable[[], object]) -> BoundTextBlock:
        block = BoundTextBlock(output, calls)
        self._blocks.append(block)
        return block

    def linear(
        self, x: Tensor, weight: Matrix, output: Tensor, *, private_scratch: bool = False,
    ) -> Callable[[], object]:
        binding = self.operators.bind_linear(x, weight, output)
        self._reuse_linear_scratch(binding, "linear.private", private_scratch)
        return binding.launch

    def _reuse_linear_scratch(self, binding, role, private):
        indices = (() if private else getattr(binding, "reusable_scratch_indices",
                    (0,) if binding.reusable_scratch else ()))
        for index in indices:
            old = binding.scratch[index]
            name = role if index == 0 and binding.reusable_scratch else role + ".temporary." + str(index)
            shared = self.buffer(name, *old.shape, dtype=old.dtype)
            if index == 0 and binding.reusable_scratch:
                binding.rebind_private_scratch(shared)
            else:
                binding.rebind_scratch(index, shared)
        self._extra_scratch.extend(value for i, value in enumerate(binding.scratch)
                                   if i not in indices)

    def linear_group(
        self, x: Tensor, weights: tuple[Matrix, ...], outputs: tuple[Tensor, ...],
        *, overlap: bool = False, private_scratch: bool = False,
    ) -> Callable[[], object]:
        """Bind same-input projections; optionally overlap two local bindings.

        The caller owns a serial group live range and a complete, read-only
        input. The projection role is separate from the MoE branch role and
        joins before any consumer or reuse. Grouped format bindings remain
        preferred; splitting an already grouped GEMM is not this operation.
        """
        bindings = self.operators.bind_linear_groups(x, weights, outputs)
        if (overlap and launch_parameters().projection_group_overlap
                and len(weights) == len(bindings) == 2
                and all(b.output is y for b, y in zip(bindings, outputs, strict=True))):
            def disjoint(a: Tensor, b: Tensor) -> bool:
                left, right = a.untyped_storage(), b.untyped_storage()
                return (left.data_ptr() >= right.data_ptr() + right.nbytes()
                        or right.data_ptr() >= left.data_ptr() + left.nbytes())

            writable = tuple((*b.scratch, b.output) for b in bindings)
            independent = (
                all(disjoint(a, b) for a in writable[0] for b in writable[1])
                and all(disjoint(a, x) for branch in writable for a in branch)
            )
            if independent:
                work = []
                for weight, binding in zip(weights, bindings, strict=True):
                    n, k = weight.shape
                    storage = (weight.numel() * weight.element_size()
                               if isinstance(weight, Tensor) else
                               sum(t.numel() * t.element_size() for t in weight.payload.values()))
                    work.append((2 * x.shape[0] * n * k,
                                 storage + x.shape[0] * (n + k) * x.element_size(),
                                 len(binding.project_artifacts())))
                serial, parallel = device_work_cost(self.device, self.dtype).branch_costs(
                    *work, fork_join_launches=launch_parameters().branch_fork_join_launches,
                )
                self.projection_branch_costs[tuple(id(w) for w in weights)] = (
                    serial, parallel, parallel < serial,
                )
                if parallel < serial:
                    # Each overlap lane owns its partials separately from the
                    # serial group and MoE roles. Joins permit serial layers
                    # to reuse a lane without aliasing the concurrent peer.
                    for lane, binding in enumerate(bindings):
                        self._reuse_linear_scratch(binding, f"linear.parallel.{lane}", private_scratch)
                    writable = tuple((*b.scratch, b.output) for b in bindings)
                    if (not all(disjoint(a, b) for a in writable[0] for b in writable[1])
                            or not all(disjoint(a, x) for branch in writable for a in branch)):
                        raise ValueError("parallel projection workspace aliases a live operand")
                    if self._projection_branches is None:
                        self._projection_branches = MoeBranchStreams(self.device, 1)
                    branches = self._projection_branches

                    def launch_parallel() -> None:
                        branches.begin()
                        with branches.fork_shared():  # type: ignore[no-untyped-call]
                            bindings[1].launch()
                        bindings[0].launch()
                        branches.join_shared()

                    return launch_parallel
        for binding in bindings:
            self._reuse_linear_scratch(binding, "linear.grouped.private", private_scratch)

        def launch():
            for binding in bindings:
                binding.launch()

        return launch

    @staticmethod
    def norm(
        x: Tensor, weight: Tensor, output: Tensor, *, epsilon: float, group_size: int | None = None
    ) -> Callable[[], object]:
        primitive = ZeroCenteredRMSNormPrimitive(
            x, weight, output, epsilon=epsilon, group_size=group_size
        )
        return partial(primitive.launch, x, weight, output)

    def mix(
        self,
        residual: Tensor,
        norm: Tensor,
        down: Matrix,
        up: Matrix,
        *,
        streams: int,
        epsilon: float,
        injection: tuple[Matrix, Tensor] | None = None,
        residual_update: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[BoundTextBlock, Tensor]:
        rows, full_width = residual.shape
        if streams <= 0 or full_width % streams:
            raise ValueError("residual streams do not divide the declared activation width")
        width = full_width // streams
        normalized = self.buffer("hc.normalized", rows, full_width)
        low = self.buffer("hc.low", rows, down.shape[0])
        activated = self.buffer("hc.activated", *low.shape)
        logits = self.buffer("hc.logits", rows, full_width)
        # Exported HC output crosses the provider alignment/head boundary;
        # its source and selected destination must remain disjoint.
        mixed = self.buffer("block.input", rows, width, private=True)
        activation_call = bind_scaled_activation(
            low, activated, divisor=float(streams)
        ).launch
        up_input = activated
        # Complete stream mixers have validated capacity-aware dense reductions.
        # The format owns the actual plan; other block bindings retain theirs.
        with self.operators.dense_capacity_scope():
            block = self._block(
                mixed,
                self.norm(residual, norm, normalized, epsilon=epsilon, group_size=width)
                if residual_update is None else self.combine_norm(
                    residual, *residual_update, norm, normalized,
                    epsilon=epsilon, group_size=width,
                ),
                self.linear(normalized, down, low) if injection is None else self.linear_group(
                    normalized, (down, injection[0]), (low, injection[1]), overlap=True,
                ),
                activation_call,
                self.linear(up_input, up, logits),
                bind_stream_mix(normalized, logits, mixed).launch,
            )
        return block, normalized

    def combine_norm(
        self, residual: Tensor, update: Tensor, logits: Tensor, weight: Tensor,
        normalized: Tensor, *, epsilon: float, group_size: int,
    ) -> Callable[[], object]:
        if update.shape[1] != group_size:
            raise ValueError("residual update width differs from the norm group")
        gates = self.buffer("hc.injection_gates", *logits.shape)
        return bind_stream_combine_norm(
            residual, update, logits, weight, normalized, gates,
            epsilon=epsilon, choices=self._residual_norm_choices,
        ).launch

    def inject(
        self, residual: Tensor, update: Tensor, normalized: Tensor, weight: Matrix, *, streams: int,
        logits: Tensor | None = None,
    ) -> BoundTextBlock:
        projection = ()
        if logits is None:
            logits = self.buffer("hc.injection_logits", residual.shape[0], streams)
            projection = (self.linear(normalized, weight, logits),)
        elif (
            logits.shape != (residual.shape[0], streams)
            or logits.dtype != residual.dtype or logits.device != residual.device
        ):
            raise ValueError(
                "precomputed injection logits differ from the residual stream contract"
            )
        gates = self.buffer("hc.injection_gates", *logits.shape)
        return self._block(
            residual,
            *projection,
            bind_scaled_activation(
                logits, gates, divisor=float(streams), multiplier=2.0, activation="sigmoid"
            ).launch,
            bind_stream_combine(residual, update, gates, residual).launch,
        )

    def ple(
        self,
        residual: Tensor,
        tokens: Tensor,
        *,
        shards: tuple[tuple[int, Matrix], ...],
        table_scale: Tensor,
        multipliers: Tensor,
        vocab_sizes: Tensor,
        offsets: Tensor,
        eos_token_id: int,
        hash_state: Tensor,
        hash_sources: Tensor,
        hash_destinations: Tensor,
        conv_state: Tensor,
        conv_sources: Tensor,
        conv_destinations: Tensor,
        segment_lengths: Tensor,
        row_activity: Tensor,
        key_projection: Matrix,
        value_projection: Matrix,
        key_norm: Tensor,
        query_norm: Tensor,
        conv_norm: Tensor,
        conv_weight: Tensor,
        streams: int,
        dilation: int,
        epsilon: float,
        reduce_embedding: Callable[[Tensor], object] | None = None,
    ) -> BoundTextBlock:
        """Bind lexical lookup and causal convolution into expanded residuals.

        Original table shard offsets/scales and state ownership are caller-owned.
        The optional reduction completes the sharded lookup before projections.
        """
        from cubic_serve.ops.cuda.embedding.ngram import bind_ngram_hash
        from cubic_serve.ops.cuda.embedding.token import bind_scaled_sharded_embedding
        from cubic_serve.ops.cuda.pointwise.dilated_conv import bind_dilated_depthwise_conv
        from cubic_serve.ops.cuda.pointwise.qwen_decode import ResidualAddPrimitive
        from cubic_serve.ops.cuda.pointwise.residual_streams import bind_signed_root_gate

        rows, full_width = residual.shape
        if streams <= 0 or full_width % streams or not shards:
            raise ValueError("lexical residual stream/table geometry differs")
        width = full_width // streams
        ids = self.buffer("ple.ids", rows, vocab_sizes.numel(), dtype=torch.int64)
        embedding = self.buffer("ple.embedding", rows, ids.shape[1] * shards[0][1].shape[1])
        key = self.buffer("ple.key", rows, full_width)
        normalized_key = self.buffer("ple.normalized_key", *key.shape)
        normalized_query = self.buffer("ple.normalized_query", *key.shape)
        value = self.buffer("ple.value", rows, width)
        gates = self.buffer("ple.gates", rows, streams)
        gated = self.buffer("ple.gated", *key.shape)
        conv_input = self.buffer("ple.conv_input", *key.shape)
        convolved = self.buffer("ple.convolved", *key.shape)
        combined = self.buffer("ple.combined", *key.shape)
        hashes = bind_ngram_hash(
            tokens,
            hash_state,
            hash_sources,
            hash_destinations,
            segment_lengths,
            row_activity,
            multipliers,
            vocab_sizes,
            offsets,
            ids,
            eos_token_id=eos_token_id,
        )
        lookup = bind_scaled_sharded_embedding(ids, shards, table_scale, row_activity, embedding)
        self._extra_scratch.extend(lookup.scratch)
        signed_gate = bind_signed_root_gate(normalized_key, normalized_query, value, gates, gated)
        convolution = bind_dilated_depthwise_conv(
            conv_input,
            conv_state,
            conv_sources,
            conv_destinations,
            segment_lengths,
            row_activity,
            conv_weight,
            convolved,
            dilation=dilation,
        )
        add_update = ResidualAddPrimitive(gated, convolved, combined)
        add_residual = ResidualAddPrimitive(residual, combined, residual)
        calls: tuple[Callable[[], object], ...] = (hashes.launch, lookup.launch)
        if reduce_embedding is not None:
            calls += (partial(reduce_embedding, embedding),)
        calls += (
            self.linear(embedding, key_projection, key),
            self.norm(key, key_norm, normalized_key, epsilon=epsilon, group_size=width),
            self.norm(residual, query_norm, normalized_query, epsilon=epsilon, group_size=width),
            self.linear(embedding, value_projection, value),
            signed_gate.launch,
            self.norm(gated, conv_norm, conv_input, epsilon=epsilon, group_size=width),
            convolution.launch,
            partial(add_update.launch, gated, convolved, combined),
            partial(add_residual.launch, residual, combined, residual),
        )
        return self._block(residual, *calls)

    def qsa(
        self,
        x: Tensor,
        projections: Mapping[str, Matrix],
        norms: Mapping[str, Tensor],
        *,
        query_heads: int,
        head_dim: int,
        index_heads: int,
        compression: int,
        token_budget: int,
        epsilon: float,
        key_cache: Tensor,
        value_cache: Tensor,
        written: Tensor,
        cache_layer: int,
        compressed_cache: Tensor,
        raw_history: Tensor,
        position_history: Tensor,
        sources: Tensor,
        destinations: Tensor,
        segment_lengths: Tensor,
        row_activity: Tensor,
        logical_positions: Tensor,
        positions: Tensor,
        page_table: Tensor,
        page_destinations: Tensor,
        token_offsets: Tensor,
        visible_lengths: Tensor,
        cosine_table: Tensor,
        sine_table: Tensor,
        rope_evidence: MRoPETableEvidence,
        reduce_output: Callable[[Tensor], object] | None = None,
        indexer_plan=None,
        indexer_group=None,
    ) -> BoundTextBlock:
        """Bind projections, causal compression and selected paged attention.

        Positions are exact INT64 [3,M]. The rank executor owns page/state COW,
        bounds, exclusive leases, and padding destinations in private scratch.
        Compressed keys are one contiguous page field for this layer.
        """
        from cubic_serve.ops.cuda.attention.paged_decode import (
            PagedGQADecodePrimitive,
            PageTokenWritePrimitive,
        )
        from cubic_serve.ops.cuda.attention.sparse_indexer import (
            PagedBlockSelection,
            IndexerPreparation,
            bind_causal_key_compression,
            bind_compressed_page_write,
        )
        from cubic_serve.ops.cuda.pointwise.qwen_decode import (
            QwenAttentionQGateSplitPrimitive,
            SigmoidOutputGatePrimitive,
        )
        from cubic_serve.ops.cuda.position.qwen_mrope import QwenMRoPEApplyPrimitive

        rows, hidden = x.shape
        encoding = self.cache_encoding
        if encoding is None:
            page_size, kv_heads, cache_dim = key_cache.shape[-3:]
        else:
            if self._cache_arena is None or (key_cache is not self._cache_arena.key
                                            or value_cache is not self._cache_arena.value):
                raise ValueError("attention cache differs from its engine-owned encoding")
            page_size, kv_heads, cache_dim = encoding.key.logical_shape
        index_dim = compressed_cache.shape[-1]
        index_parts = indexer_plan.size if indexer_plan is not None else 1
        index_rank = indexer_plan.rank if indexer_plan is not None else 0
        if (
            cache_dim != head_dim
            or raw_history.shape[1:] != (compression, index_dim // index_parts)
            or compressed_cache.shape[0] != key_cache.shape[0]
            or compressed_cache.shape[1] * compression * index_parts != page_size
        ):
            raise ValueError("attention and compressed page geometry differ")
        qg = self.buffer("qsa.qg", rows, query_heads, 2, head_dim)
        query = self.buffer("qsa.query", rows, query_heads, head_dim)
        gate = self.buffer("qsa.gate", *query.shape)
        key = self.buffer("qsa.key", rows, kv_heads, head_dim)
        value = self.buffer("qsa.value", *key.shape)
        nq = self.buffer("qsa.normalized_query", *query.shape)
        nk = self.buffer("qsa.normalized_key", *key.shape)
        rq = self.buffer("qsa.rotated_query", *query.shape)
        rk = self.buffer("qsa.rotated_key", *key.shape)
        cosine = self.buffer("qsa.cosine", rows, rope_evidence.rotary_dim)
        sine = self.buffer("qsa.sine", *cosine.shape)
        gather = self._rope_gather(
            positions, cosine_table, sine_table, cosine, sine, evidence=rope_evidence
        )
        split = QwenAttentionQGateSplitPrimitive(qg, query, gate)
        rotary = QwenMRoPEApplyPrimitive(nq, nk, cosine, sine, rq, rk)
        metadata_calls: tuple[Callable[[], object], ...] = ()
        append_calls: tuple[Callable[[], object], ...] = ()
        window = None
        if encoding is None:
            write = PageTokenWritePrimitive(
                key_cache, value_cache, written, page_destinations, token_offsets, rk, value,
                layer=cache_layer,
            )
            write_calls = (partial(write.launch, key_cache, value_cache, written,
                                   page_destinations, token_offsets, rk, value),)
        else:
            from cubic_serve.ops.cuda.cache_quantization import PagedKVGraphBinding
            assert self._cache_arena is not None
            assert self._cache_source_slots is not None
            assert self._cache_destination_slots is not None
            identity = tuple(bind_tensor(t) for t in (
                segment_lengths, row_activity, logical_positions, self._cache_source_slots))
            if self._cache_metadata_identity is None:
                self._cache_binding = PagedKVGraphBinding(
                    self._cache_arena, positions=logical_positions, segments=segment_lengths,
                    activity=row_activity, source_slots=self._cache_source_slots,
                    destination_slots=self._cache_destination_slots,
                )
                metadata_calls = (self._cache_binding.prepare_metadata,)
                self._cache_metadata_identity = identity
            elif self._cache_metadata_identity != identity:
                raise ValueError("serial attention blocks must share engine row metadata")
            assert self._cache_binding is not None
            window, append, checkpoint = self._cache_binding.bind_layer(
                rk, value, layer=cache_layer, pages=page_destinations, query=rq,
            )
            write_calls = (append.launch,)
            append_calls = (checkpoint.launch,)
        if index_parts > 1:
            packed = self.buffer("qsa.index_packed", rows,
                                 indexer_plan.local_query_heads * index_dim + indexer_plan.key_width)
            riq = self.buffer("qsa.rotated_index_query", rows, index_heads, index_dim)
            group_positions = self.buffer("qsa.group_positions", 3, rows, dtype=torch.int64)
            group_cosine = self.buffer("qsa.group_cosine", *cosine.shape)
            group_sine = self.buffer("qsa.group_sine", *sine.shape)
            preparation = IndexerPreparation(
                packed, norms["index_q"], norms["index_k"], cosine, sine,
                positions, logical_positions, raw_history, position_history,
                sources, destinations, segment_lengths, row_activity,
                token_offsets, page_destinations, compressed_cache, riq,
                group_positions, group_cosine, group_sine,
                plan=indexer_plan, group=indexer_group, epsilon=epsilon,
            )
            group_gather = self._rope_gather(
                group_positions, cosine_table, sine_table, group_cosine, group_sine,
                evidence=rope_evidence,
            )
            self._extra_scratch.extend(preparation.scratch)
            index_calls = (
                preparation.prepare,
                partial(group_gather.launch, group_positions, cosine_table, sine_table,
                        group_cosine, group_sine),
                preparation.finish,
            )
        else:
            packed = self.buffer("qsa.index_packed", rows, (index_heads + 1) * index_dim)
            iq = self.buffer("qsa.index_query", rows, index_heads, index_dim)
            ik = self.buffer("qsa.index_key", rows, index_dim)
            niq = self.buffer("qsa.normalized_index_query", *iq.shape)
            riq = self.buffer("qsa.rotated_index_query", *iq.shape)
            # Both source operands are identical; separate outputs avoid write aliasing.
            riq_copy = self.buffer("qsa.rotated_index_query_copy", *iq.shape)
            index_rotary = QwenMRoPEApplyPrimitive(niq, niq, cosine, sine, riq, riq_copy)
            pooled = self.buffer("qsa.pooled", rows, index_dim)
            normalized_pooled = self.buffer("qsa.normalized_pooled", rows, 1, index_dim)
            rotated_pooled = self.buffer("qsa.rotated_pooled", *normalized_pooled.shape)
            rotated_copy = self.buffer("qsa.rotated_pooled_copy", *normalized_pooled.shape)
            position_rows = self.buffer("qsa.position_rows", rows, 3, dtype=torch.int64)
            group_rows = self.buffer("qsa.group_rows", rows, 3, dtype=torch.int64)
            group_positions = self.buffer("qsa.group_positions", 3, rows, dtype=torch.int64)
            complete = self.buffer("qsa.complete", rows, dtype=torch.int32)
            compress = bind_causal_key_compression(
                ik,
                position_rows,
                logical_positions,
                raw_history,
                position_history,
                sources,
                destinations,
                segment_lengths,
                row_activity,
                pooled,
                group_rows,
                complete,
            )
            group_cosine = self.buffer("qsa.group_cosine", *cosine.shape)
            group_sine = self.buffer("qsa.group_sine", *sine.shape)
            group_gather = self._rope_gather(
                group_positions,
                cosine_table,
                sine_table,
                group_cosine,
                group_sine,
                evidence=rope_evidence,
            )
            group_rotary = QwenMRoPEApplyPrimitive(
                normalized_pooled,
                normalized_pooled,
                group_cosine,
                group_sine,
                rotated_pooled,
                rotated_copy,
            )
            publish = bind_compressed_page_write(
                rotated_pooled.view(rows, index_dim),
                compressed_cache,
                page_destinations,
                token_offsets,
                complete,
                row_activity,
                compression=compression,
            )
            index_calls = (
                partial(iq.view(rows, -1).copy_, packed[:, : index_heads * index_dim]),
                partial(ik.copy_, packed[:, index_heads * index_dim :]),
                self.norm(
                    iq.view(-1, index_dim), norms["index_q"], niq.view(-1, index_dim), epsilon=epsilon
                ),
                partial(index_rotary.launch, niq, niq, cosine, sine, riq, riq_copy),
                partial(position_rows.copy_, positions.T),
                # Compression leaves inactive outputs untouched; rotary gather still
                # runs on every physical row, so its private indices must be valid.
                group_rows.zero_,
                pooled.zero_,
                compress.launch,
                self.norm(
                    pooled, norms["index_k"], normalized_pooled.view(rows, index_dim), epsilon=epsilon
                ),
                partial(group_positions.copy_, group_rows.T),
                partial(
                    group_gather.launch,
                    group_positions,
                    cosine_table,
                    sine_table,
                    group_cosine,
                    group_sine,
                ),
                partial(
                    group_rotary.launch,
                    normalized_pooled,
                    normalized_pooled,
                    group_cosine,
                    group_sine,
                    rotated_pooled,
                    rotated_copy,
                ),
                publish.launch,
            )
        selected = self.buffer(
            "qsa.selected", rows, token_budget + compression - 1, dtype=torch.int32
        )
        counts = self.buffer("qsa.selected_counts", rows, dtype=torch.int32)
        selector = PagedBlockSelection(
            riq,
            compressed_cache,
            page_table,
            visible_lengths,
            row_activity,
            selected,
            counts,
            compression=compression,
            token_budget=token_budget,
            parts=index_parts, rank=index_rank, group=indexer_group,
            workspace=lambda role, shape, dtype: self.buffer(
                "qsa.selection." + role, *shape, dtype=dtype,
            ),
        )
        self._extra_scratch.extend(selector.scratch)
        core = self.buffer("qsa.core", *query.shape)
        gated = self.buffer("qsa.gated", *query.shape)
        from cubic_serve.ops.cuda.attention.paged_decode import paged_gqa_kv_splits

        # Bind the partition plan before capture and attest its actual split
        # count. Workspaces are private to this bound graph's lease.
        from cubic_serve.ops.strategies import attention_precision

        head_programs = (
            kv_heads * ((query.shape[1] // kv_heads + 15) // 16)
            if attention_precision() == "auto" else query.shape[1]
        )
        splits = paged_gqa_kv_splits(
            rows, max_tokens=selected.shape[1], head_programs=head_programs,
            multiprocessors=torch.cuda.get_device_properties(query.device).multi_processor_count,
        )
        attention_partial = self.buffer(
            "qsa.attention_partial", rows, query.shape[1], splits, head_dim,
            dtype=torch.float32,
        )
        attention_statistics = self.buffer(
            "qsa.attention_statistics", rows, query.shape[1], splits, 2,
            dtype=torch.float32,
        )
        attention_arrivals = self.buffer(
            "qsa.attention_arrivals", rows, query.shape[1], dtype=torch.int32,
        )
        attention_arrivals.zero_()
        attention_query = (rq if self._cache_binding is None else self._cache_binding.query_for(rq))
        attention = PagedGQADecodePrimitive(
            attention_query,
            key_cache,
            value_cache,
            page_table,
            counts,
            core,
            layer=cache_layer,
            partial=attention_partial,
            statistics=attention_statistics,
            arrivals=attention_arrivals,
            selected_indices=selected,
            cache_encoding=encoding,
            cache_window=window,
        )
        output_gate = SigmoidOutputGatePrimitive(core, gate, gated)
        output = self.buffer("block.output", rows, hidden)
        calls: tuple[Callable[[], object], ...] = (
            self.linear_group(
                x,
                tuple(projections[name] for name in ("q", "k", "v", "index_qk")),
                (qg.view(rows, -1), key.view(rows, -1), value.view(rows, -1), packed),
            ),
            partial(split.launch, qg, query, gate),
            self.norm(query.view(-1, head_dim), norms["q"], nq.view(-1, head_dim), epsilon=epsilon),
            self.norm(key.view(-1, head_dim), norms["k"], nk.view(-1, head_dim), epsilon=epsilon),
            partial(gather.launch, positions, cosine_table, sine_table, cosine, sine),
            partial(rotary.launch, nq, nk, cosine, sine, rq, rk),
            *metadata_calls,
            *write_calls,
            *index_calls,
            # Inactive selector rows do not write; never consume stale counts.
            counts.zero_,
            selector.launch,
            partial(attention.launch, attention_query, key_cache, value_cache,
                    page_table, counts, core),
            *append_calls,
            partial(output_gate.launch, core, gate, gated),
            self.linear(gated.view(rows, -1), projections["out"], output),
        )
        if reduce_output is not None:
            calls += (partial(reduce_output, output),)
        return self._block(output, *calls)

    def gdn(
        self,
        x: Tensor,
        projections: Mapping[str, Matrix],
        *,
        conv_weight: Tensor,
        decay: Tensor,
        bias: Tensor,
        norm: Tensor,
        conv_state: Tensor,
        recurrent_state: Tensor,
        conv_sources: Tensor,
        conv_destinations: Tensor,
        recurrent_sources: Tensor,
        recurrent_destinations: Tensor,
        segment_lengths: Tensor,
        row_activity: Tensor | None = None,
        key_heads: int,
        key_dim: int,
        value_heads: int,
        value_dim: int,
        epsilon: float,
        output_gate: str,
        reduce_output: Callable[[Tensor], object] | None = None,
    ) -> BoundTextBlock:
        """Bind shared segmented GDN; cache dtype and gate rule are explicit."""
        rows, hidden = x.shape
        packed_width = 2 * key_heads * key_dim + value_heads * value_dim
        packed = self.buffer("gdn.packed", rows, packed_width)
        convolved = self.buffer("gdn.convolved", *packed.shape)
        a = self.buffer("gdn.a", rows, value_heads)
        b = self.buffer("gdn.b", rows, value_heads)
        z = self.buffer("gdn.z", rows, value_heads, value_dim)
        core = self.buffer("gdn.core", *z.shape)
        normalized = self.buffer("gdn.normalized", *z.shape)
        output = self.buffer("block.output", rows, hidden)
        convolution = CausalConv1DDecodePrimitive(
            packed,
            conv_weight.squeeze(1),
            conv_state,
            conv_sources,
            conv_destinations,
            segment_lengths,
            convolved,
            checkpoint_destinations=True,
        )
        recurrence = GatedDeltaDecodePrimitive(
            convolved,
            packed,
            a,
            b,
            decay,
            bias,
            recurrent_state,
            recurrent_sources,
            recurrent_destinations,
            segment_lengths,
            core,
            q_heads=key_heads,
            scale=key_dim**-0.5,
            checkpoint_destinations=True,
            row_activity=row_activity,
        )
        gate_norm = GatedRMSNormPrimitive(
            core, z, norm, normalized, epsilon=epsilon, activation=output_gate
        )
        calls = (
            self.linear_group(
                x,
                tuple(projections[name] for name in ("qkv", "a", "b", "z")),
                (packed, a, b, z.reshape(rows, -1)),
            ),
            partial(
                convolution.launch,
                packed,
                conv_weight.squeeze(1),
                conv_state,
                conv_sources,
                conv_destinations,
                segment_lengths,
                convolved,
            ),
            partial(
                recurrence.launch,
                convolved,
                packed,
                a,
                b,
                decay,
                bias,
                recurrent_state,
                recurrent_sources,
                recurrent_destinations,
                segment_lengths,
                core,
            ),
            partial(gate_norm.launch, core, z, norm, normalized),
            self.linear(normalized.reshape(rows, -1), projections["out"], output),
        )
        return self._block(
            output, *calls, *((partial(reduce_output, output),) if reduce_output else ())
        )

    def moe(
        self,
        x: Tensor,
        *,
        router: Matrix,
        gate: tuple[Matrix, ...],
        up: tuple[Matrix, ...],
        down: tuple[Matrix, ...],
        shared_gate: Matrix,
        shared_up: Matrix,
        shared_down: Matrix,
        shared_router: Matrix,
        activity: Tensor,
        active_rows: Tensor,
        first_expert: int,
        top_k: int,
        reduce_routes: Callable[[Tensor], None] | None = None,
        reduce_shared: Callable[[Tensor], object] | None = None,
        reduce_experts: Callable[[Tensor], object] | None = None,
    ) -> BoundTextBlock:
        """Bind routed/shared branches with unchanged checkpoint expert ownership."""
        from cubic_serve.ops.cuda.moe import BF16MoeSharedCombinePrimitive
        from cubic_serve.ops.cuda.moe.quantized import PackedRoutedExpertsPrimitive
        from cubic_serve.ops.cuda.pointwise.qwen_decode import SwiGLUPrimitive

        rows, hidden = x.shape
        logits = self.buffer("moe.logits", rows, router.shape[0])
        shared_logits = self.buffer("moe.shared_logits", rows, 1)
        gate_values = self.buffer("moe.shared_gate", rows, shared_gate.shape[0])
        up_values = self.buffer("moe.shared_up", *gate_values.shape)
        activated = self.buffer("moe.shared_activated", *gate_values.shape)
        output = self.buffer("block.output", rows, hidden)
        swiglu = SwiGLUPrimitive(gate_values, up_values, activated)
        shared_activation = partial(swiglu.launch, gate_values, up_values, activated)
        complete_routed = reduce_routes or reduce_experts
        route_sum = getattr(complete_routed, "__self__", None)
        shared_sum = getattr(reduce_shared, "__self__", None)
        expert_sum = getattr(reduce_experts, "__self__", None)
        # Extensions opt in to this arithmetic contract independently. Moving
        # the TP sum changes low-precision rounding; it is not a transport-only
        # rewrite. Every gate is replicated, and no nonlinear transform consumes
        # an incomplete rank sum. Factored EP/ETP groups can complete the local
        # weighted output over the same TP group as the shared branch.
        matrices = (*gate, *up, *down, shared_gate, shared_up, shared_down)
        local_weighting = all(isinstance(weight, QuantizedMatrix) for weight in matrices) and all(
            getattr(policy, "LOCAL_WEIGHTED_TP_COMPLETION", False)
            and self.operators.config.compute_mode in getattr(
                policy, "LOCAL_WEIGHTED_TP_COMPLETION_MODES", (self.operators.config.compute_mode,),
            )
            for encoding in {weight.encoding.format for weight in matrices
                             if isinstance(weight, QuantizedMatrix)}
            for policy in (get_extension(encoding).module("policy"),)
        )
        same_tp_group = (
            (reduce_routes is None) != (reduce_experts is None)
            and type(route_sum) is FixedRankBF16SumPrimitive
            and type(shared_sum) is FixedRankBF16SumPrimitive
            and route_sum._group is shared_sum._group
        )
        factored_tp_group = False
        if (local_weighting and reduce_routes is not None and reduce_experts is not None
                and all(type(value) is FixedRankBF16SumPrimitive
                        for value in (route_sum, shared_sum, expert_sum))
                and getattr(reduce_experts, "__func__", None)
                is FixedRankBF16SumPrimitive.launch):
            # The stage has already validated these process groups against its
            # topology. Check their factorization before replacing ETP+EP with
            # one completion over the same TP ranks.
            route_ranks = set(dist.get_process_group_ranks(route_sum._group))
            expert_ranks = set(dist.get_process_group_ranks(expert_sum._group))
            shared_ranks = set(dist.get_process_group_ranks(shared_sum._group))
            factored_tp_group = (
                len(route_ranks) * len(expert_ranks) == len(shared_ranks)
                and route_ranks <= shared_ranks and expert_ranks <= shared_ranks
                and route_ranks & expert_ranks == {dist.get_rank()}
            )
        if (local_weighting and complete_routed is not None and reduce_shared is not None
                and type(route_sum) is FixedRankBF16SumPrimitive
                and type(shared_sum) is FixedRankBF16SumPrimitive
                and getattr(complete_routed, "__func__", None) is FixedRankBF16SumPrimitive.launch
                and getattr(reduce_shared, "__func__", None) is FixedRankBF16SumPrimitive.launch
                and (same_tp_group or factored_tp_group)):
            self.local_tp_routers.add(id(router))
            routed = self.buffer("moe.routed", rows, hidden)
            shared_output = output
            experts = PackedRoutedExpertsPrimitive(
                x, logits, activity, active_rows, gate, up, down, routed,
                first_expert=first_expert, top_k=top_k, operators=self.operators,
                projection_scratch=lambda t: self.buffer(
                    "moe.projection.partial", *t.shape, dtype=t.dtype,
                ),
                workspace=lambda role, shape, dtype: self.buffer(
                    "moe.activation." + role, *shape, dtype=dtype,
                ),
            )
            self._extra_scratch.extend(experts.scratch)
            combine = BF16MoeSharedCombinePrimitive(shared_output, shared_logits, routed, output)
            # One isolated owner for the shared-expert lane: serial layers reuse
            # format-opted-in partials, but never the concurrent main lane's.
            # The branch join below completes every consumer before next reuse.
            if self._shared_expert_bindings is None:
                self._shared_expert_bindings = TextGraphBindings(
                    self.operators, self.device, self.dtype,
                )
            shared = self._shared_expert_bindings
            shared_inputs = shared.linear_group(
                x, (shared_gate, shared_up), (gate_values, up_values),
            )
            shared_projection = shared.linear(
                activated, shared_down, shared_output,
            )
            route_inputs = self.linear_group(x, (router, shared_router), (logits, shared_logits))
            shared_scratch = shared.scratch
            main_scratch = (
                *self._buffers.values(), *self._extra_scratch,
                *(self._cache_binding.scratch if self._cache_binding is not None else ()),
            )
            overlap = False
            if (launch_parameters().shared_expert_overlap and not experts.has_collectives
                    and scratch_ranges_disjoint(shared_scratch, main_scratch)):
                rate = device_work_cost(self.device, self.dtype)

                def work(
                    weights: tuple[Matrix, ...], population: float, active: float = 1.0,
                ) -> tuple[float, float]:
                    operations = traffic = 0.0
                    for weight in weights:
                        n, k = weight.shape
                        storage = (weight.numel() * weight.element_size()
                                   if isinstance(weight, Tensor) else
                                   sum(t.numel() * t.element_size() for t in weight.payload.values()))
                        operations += 2 * population * n * k
                        traffic += active * storage + population * (n + k) * x.element_size()
                    return operations, traffic

                population = rows * top_k / len(gate)
                routed_work = work((*gate, *up, *down), population, -math.expm1(-population))
                router_work = work((router, shared_router), rows)
                main_work = (routed_work[0] + router_work[0], routed_work[1] + router_work[1])
                shared_work = work((shared_gate, shared_up, shared_down), rows)
                serial, parallel = rate.branch_costs(
                    (*shared_work, 4), (*main_work, 6),
                    fork_join_launches=launch_parameters().branch_fork_join_launches,
                )
                overlap = parallel < serial
                self.moe_branch_costs[id(router)] = serial, parallel, overlap
            local_shared_calls = (
                shared_inputs,
                shared_activation,
                shared_projection,
            )
            completion = (
                partial(combine.launch, shared_output, shared_logits, routed, output),
                partial(reduce_shared, output),
            )
            if not overlap:
                return self._block(output, route_inputs, experts.launch, *local_shared_calls, *completion)
            if self._moe_branches is None:
                self._moe_branches = MoeBranchStreams(self.device, 1)
            branches = self._moe_branches

            def begin_shared() -> None:
                branches.begin()
                with branches.fork_shared():  # type: ignore[no-untyped-call]
                    for call in local_shared_calls:
                        call()
                route_inputs()

            # Serial blocks reuse one role stream/event pair. The join precedes
            # combination, the only TP collective, and all following buffer reuse.
            return self._block(
                output, begin_shared, experts.launch, branches.join_shared, *completion,
            )
        joint = None
        if (type(route_sum) is FixedRankBF16SumPrimitive
                and type(shared_sum) is FixedRankBF16SumPrimitive
                and getattr(complete_routed, "__func__", None) is FixedRankBF16SumPrimitive.launch
                and getattr(reduce_shared, "__func__", None) is FixedRankBF16SumPrimitive.launch):
            # Preserve logical observer order even when the physical transport
            # changes: ETP partials precede shared output; EP follows it.
            key = (route_sum, shared_sum) if reduce_routes else (shared_sum, route_sum)
            if key not in self._coalesced_sums:
                storage = torch.empty(
                    (rows * ((top_k if reduce_routes else 1) + 1), hidden),
                    device=self.device, dtype=self.dtype,
                )
                reduction = FixedRankBF16SumPrimitive.coalesced(storage, key)
                self._coalesced_sums[key] = None if reduction is None else (storage, reduction)
                if reduction is not None:
                    self._extra_scratch.append(storage)
            joint = self._coalesced_sums[key]
        route_workspace = (
            None if joint is None or reduce_routes is None
            else joint[0][:rows * top_k].view(rows, top_k, hidden)
        )
        if joint is not None and reduce_routes is None:
            shared_output, routed = joint[0][:rows], joint[0][rows:]
        else:
            routed = self.buffer("moe.routed", rows, hidden)
            shared_output = output if joint is None else joint[0][rows * top_k:]

        def reduce_joint_routes(_value: Tensor) -> None:
            assert joint is not None
            # The route producer has completed. Produce the other independent
            # member, transport both, then let the original route combine run.
            for call in shared_calls:
                call()
            joint[1].launch(joint[0])

        experts = PackedRoutedExpertsPrimitive(
            x,
            logits,
            activity,
            active_rows,
            gate,
            up,
            down,
            routed,
            first_expert=first_expert,
            top_k=top_k,
            operators=self.operators,
            reduce_routes=(
                reduce_routes if joint is None or reduce_routes is None else reduce_joint_routes
            ),
            route_workspace=route_workspace,
            projection_scratch=lambda t: self.buffer(
                "moe.projection.partial", *t.shape, dtype=t.dtype,
            ),
            workspace=lambda role, shape, dtype: self.buffer(
                "moe.activation." + role, *shape, dtype=dtype,
            ),
        )
        self._extra_scratch.extend(experts.scratch)
        combine = BF16MoeSharedCombinePrimitive(shared_output, shared_logits, routed, output)
        shared_calls = (
            shared_activation,
            self.linear(activated, shared_down, shared_output),
        )
        return self._block(
            output,
            self.linear_group(
                x,
                (router, shared_router, shared_gate, shared_up),
                (logits, shared_logits, gate_values, up_values),
            ),
            experts.launch,
            *(shared_calls if joint is None or reduce_routes is None else ()),
            *((partial(joint[1].launch, joint[0]),)
              if joint is not None and reduce_routes is None else ()),
            *((partial(reduce_shared, output),) if reduce_shared and joint is None else ()),
            *((partial(reduce_experts, routed),)
              if reduce_experts and (joint is None or reduce_routes is not None) else ()),
            partial(combine.launch, shared_output, shared_logits, routed, output),
        )
