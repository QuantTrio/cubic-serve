"""Original-weight causal and free-trajectory witnesses for the typed text graph.

A disposable M1 graph retains immutable, stream-ordered producer snapshots.
Every forward and rank schedule completes before reference math, host
inspection or diagnostic collectives. The serving graphs are never instrumented.
Each candidate state transition is proved on its actual operands and rounded
writeback; the separate free reference advances from its own state and original
weights. The free path uses the same declared MoE reduction plan, while route
differences are checked against each path's logits and expert event schedule.
This bounded horizon still needs the separate cross-M, context and topology
qualification gates.
"""

import gc
import logging
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from typing import Any, Callable
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import Tensor

from cubic_serve.executor.runner import ReferenceExecutionContext
from cubic_serve.executor.text_graph import TextGraphBindings
from cubic_serve.executor.text_serving import FlashNextDecodeStage
from cubic_serve.guardrail.quantization import (
    ReferenceMathMemo,
    check_linear_enclosure,
    with_reference_integer_geometry,
)
from cubic_serve.guardrail.quantized_graph.bounds import (
    SwiGLUWitnessBatch,
    check,
    product,
    route_softmax_bounds,
)
from cubic_serve.guardrail.quantized_graph.ep import LinearRecorder as EPLinearRecorder
from cubic_serve.guardrail.quantized_graph.linear import Recorder, same
from cubic_serve.models.qwen4_exp import model as reference_module
from cubic_serve.models.qwen4_exp.model import Qwen4ExpReferenceModel
from cubic_serve.models.qwen4_exp.weights import logical_quantized_weights
from cubic_serve.ops.cuda.collective import FixedRankBF16SumPrimitive
from cubic_serve.ops.cuda.moe.quantized import PackedRoutedExpertsPrimitive
from cubic_serve.ops.eager.quantization import QuantizedReferenceOperators
from cubic_serve.ops.strategies import operator_strategy_scope
from cubic_serve.runtime.text_qualification import prepare_rows, state_audit_horizon


def _audit_cache_arena(source, pages, *, scratch_capacity=1):
    """Keep the production field pitches in a bounded single-owner witness.

    Page/state addressing and ownership at large offsets are separate byte
    proofs. The numerical transition needs the actual format fields and their
    production strides, without allocating a second serving-sized byte pool.
    """
    original = source.arena
    storage = getattr(original, "shared_storage", None)
    if storage is None:
        return None
    if (type(scratch_capacity) is not int or scratch_capacity < 1
            or scratch_capacity > original.spec.scratch_capacity):
        raise ValueError("numerical scratch exceeds the source graph envelope")
    from cubic_serve.runtime.cache.pages import CacheArena
    from cubic_serve.runtime.cache.shared_pool import SharedCacheStorage

    indices = [i for i in range(len(storage.schemas))
               if storage.views(i)[0]["key"].data_ptr() == original.key.data_ptr()]
    if len(indices) != 1:
        raise ValueError("numerical witness has ambiguous shared arena ownership")
    schemas = tuple((replace(spec, scratch_capacity=scratch_capacity), layout)
                    for spec, layout in storage.schemas)
    state_bytes = 0 if storage.state_layout is None else storage.state_layout.entry_bytes
    witness = SharedCacheStorage(
        schemas, budget_bytes=state_bytes + pages * storage.page_layout.entry_bytes,
        state_capacity=1 if state_bytes else 0, device=original.device,
        extra_state_fields=storage.extra_state_fields,
    )
    # Exact-input numerical fixtures own every byte they exercise. Other
    # potential page views intersect the live state and remain unowned.
    for page in range(pages):
        witness.allocator.reserve_at("page", page)
    if state_bytes:
        witness.allocator.reserve_at("state", 0)
    index = indices[0]
    return CacheArena(witness.arena_specs[index], schemas[index][1], original.device,
                      rank_id=original.rank_id, shared_storage=witness, shared_index=index)


def _completed_moe_combine_proof(
    shared, gate, routed, actual, *, expected_shared, expected_gate, expected_routed,
    activation_dtype,
):
    """Audit the stored shared combine after independently completed TP/EP sums."""
    from .bounds import moe_shared_combine_bounds

    for label, candidate, expected in (
        ("shared", shared, expected_shared), ("gate", gate, expected_gate),
        ("routed", routed, expected_routed),
    ):
        if not same(candidate, expected):
            raise AssertionError("unproved completed MoE " + label)
    if not all(bool(v.isfinite().all()) for v in (shared, gate, routed, actual)):
        raise AssertionError("nonfinite completed MoE boundary")
    lo, hi = moe_shared_combine_bounds(
        shared.cpu(), gate.cpu(), routed.cpu(), activation_dtype=activation_dtype,
    )
    proof = check(actual.cpu(), lo, hi)
    if proof["outside_enclosure"]:
        raise AssertionError("completed MoE shared combine: " + str(proof))
    return proof


def _expert_reference_schedule_error(schedules, topology, expert_count):
    """Validate completed oracle events against fixed checkpoint ownership.

    Shared operations agree across TP. Expert operations agree within each
    expert-tensor group and cover exactly the independently produced routes
    owned by that group. Different owners need not execute the same experts.
    """
    if len(schedules) != topology.tp_size or expert_count % topology.ep_size:
        return "independent reference topology inventory differs"
    common = None
    routes = schedules[0][1]
    owner_schedules = {}
    local_count = expert_count // topology.ep_size
    for rank, (events, decisions) in enumerate(schedules):
        owner, _ = topology.expert_coordinates(rank)
        if routes != decisions:
            return "independent reference producing route decisions differ across ranks"
        shared = [event for event in events if ".experts." not in event[0]]
        if common is None:
            common = shared
        elif common != shared:
            return "independent reference shared rank event schedules differ"
        expert_events = [event for event in events if ".experts." in event[0]]
        if owner in owner_schedules and owner_schedules[owner] != expert_events:
            return "independent reference expert-tensor event schedules differ"
        owner_schedules[owner] = expert_events
        expected = Counter()
        for prefix, rows in (decisions or {}).items():
            for row in rows:
                if len(row) != len(set(row)) or any(not 0 <= eid < expert_count for eid in row):
                    return "independent reference produced invalid expert IDs"
                for eid in row:
                    if owner * local_count <= eid < (owner + 1) * local_count:
                        for projection in ("gate_proj", "up_proj", "down_proj"):
                            expected[f"{prefix}.experts.{eid}.{projection}.weight"] += 1
        actual = Counter(event[0] for event in expert_events)
        if decisions is not None and actual != expected:
            return "independent reference events violate producing routes or expert ownership"
        occurrences = Counter()
        for name, occurrence, _, _ in expert_events:
            if occurrence != occurrences[name]:
                return "independent reference expert occurrence order differs"
            occurrences[name] += 1
    if set(owner_schedules) != set(range(topology.ep_size)):
        return "independent reference expert owner inventory differs"
    return None


@contextmanager
def preserve_observer_collections(collections):
    """Keep full-forward snapshots while a separate feedback graph is captured.

    Only Python references are saved/restored; no reference operator, tensor
    transfer or diagnostic synchronization occurs during the observed forward.
    """
    snapshots = tuple(value.copy() for value in collections)
    try:
        yield
    finally:
        for value, snapshot in zip(collections, snapshots, strict=True):
            if isinstance(value, list):
                value[:] = snapshot
            else:
                value.clear()
                value.update(snapshot)


class _TrajectoryDifferences:
    """Bounded post-forward diagnostic receipts; RMS keeps its original reduction."""

    def __init__(self, max_pending=128, max_bytes=16 << 20):
        if (type(max_pending) is not int or max_pending <= 0
                or type(max_bytes) is not int or max_bytes <= 0):
            raise ValueError("trajectory receipt capacity must be positive")
        self.max_pending, self.max_bytes = max_pending, max_bytes
        self.pending, self.bytes = [], 0

    def __call__(self, actual_value, reference_value):
        if actual_value.numel() != reference_value.numel():
            raise ValueError("independent trajectory boundary geometry differs")
        size = 16 * actual_value.numel()
        if self.pending and (self.bytes + size > self.max_bytes
                             or self.pending[-1][1].device != actual_value.device):
            self.flush()
        left = actual_value.detach().reshape(-1).to(dtype=torch.float64, copy=True)
        right = reference_value.detach().reshape(-1).to(dtype=torch.float64, copy=True)
        result = {"elements": left.numel(), "candidate_dtype": str(actual_value.dtype),
                  "reference_dtype": str(reference_value.dtype)}
        if not left.numel():
            result.update(different=0, max_abs=0.0, max_abs_flat_index=None,
                          candidate_at_peak=None, reference_at_peak=None,
                          rms=0.0, reference_rms=0.0)
            return result
        delta = left - right
        # Keep each original one-dimensional FP64 RMS reduction bit-for-bit.
        rms = delta.square().mean().sqrt()
        reference_rms = right.square().mean().sqrt()
        self.pending.append((result, left, right, rms, reference_rms))
        self.bytes += size
        if len(self.pending) >= self.max_pending or self.bytes >= self.max_bytes:
            self.flush()
        return result

    def flush(self):
        if not self.pending:
            return
        pending, self.pending = self.pending, []
        self.bytes = 0
        groups = {}
        for row in pending:
            groups.setdefault(row[1].numel(), []).append(row)
        for group in groups.values():
            left = torch.stack([row[1] for row in group])
            right = torch.stack([row[2] for row in group])
            delta = left - right
            maximum, peak = delta.abs().max(-1)
            integers = torch.stack((
                (torch.isfinite(left).all(-1) & torch.isfinite(right).all(-1)).to(torch.int64),
                (left != right).sum(-1), peak,
            ), -1).cpu().tolist()
            values = torch.stack((
                maximum, left.gather(1, peak[:, None]).squeeze(1),
                right.gather(1, peak[:, None]).squeeze(1),
                torch.stack([row[3] for row in group]),
                torch.stack([row[4] for row in group]),
            ), -1).cpu().tolist()
            for row, integer_values, float_values in zip(group, integers, values, strict=True):
                result = row[0]
                finite, different, index = integer_values
                error, actual, reference, rms, reference_rms = float_values
                if not finite:
                    raise AssertionError("independent trajectory contains nonfinite values")
                result.update(different=different, max_abs=error, max_abs_flat_index=index,
                              candidate_at_peak=actual, reference_at_peak=reference,
                              rms=rms, reference_rms=reference_rms)


@with_reference_integer_geometry
def audit_original(
    source: Any, admission: Any, groups: dict[str, Any],
    *, token_ids: tuple[int, ...] | None = None,
    diagnose_independent_trajectory: bool = False,
    input_feature_rows: tuple[bytes, ...] = (),
    position_axes: tuple[tuple[int, int, int], ...] = (),
    draft_provider: Any = None,
    feedback_rows: tuple[Tensor, ...] = (),
    column_completion_names: tuple[str, ...] = (),
    feedback_recurrence: bool = False,
    row_plan: dict | None = None,
    reference_math_memo: ReferenceMathMemo | None = None,
    completed_step_witness: Callable[[int, dict[str, Tensor], dict[str, Any]], None] | None = None,
    prefix_store: Any = None,
    prefix: Any = None,
    independent_feedback_rows: tuple[Tensor, ...] = (),
    completed_transition: Callable[[dict[str, Any]], None] | None = None,
    capture_pool: Any = None,
    operator_observers: Any = None,
) -> dict[str, Any]:
    """Audit before readiness; optional consumers receive completed CPU witnesses.

    The consumer runs after the complete forward, rank-schedule agreement and
    original-weight proofs. It never observes or alters in-flight model state.
    """
    if completed_step_witness is not None and not callable(completed_step_witness):
        raise ValueError("completed numerical witness consumer must be callable")
    if completed_transition is not None and (not callable(completed_transition)
                                            or prefix_store is None):
        raise ValueError("completed transition requires an admitted prefix store")
    if (prefix is not None or prefix_store is not None or independent_feedback_rows) and (
        not diagnose_independent_trajectory or completed_transition is None
    ):
        raise ValueError("live prefixes require complete independent transition witnesses")
    config, w, topology = source.config, source.weights, source.topology
    if draft_provider is not None:
        w = source.weights.original
    elif feedback_rows or feedback_recurrence or column_completion_names:
        raise ValueError("draft audit inputs require an explicit independent provider")
    t, rank, lane, tp = config.text, source.rank, source.tensor_rank, topology.tp_size
    if topology.pp_size != 1:
        raise ValueError("typed numerical oracle requires PP1")
    policy = source.operators.config
    if policy.softmax_dtype != "fp32":
        raise ValueError("text numerical oracle requires FP32 softmax")
    group = groups.get("tp")
    if tp > 1 and group is None:
        raise ValueError("typed numerical oracle requires its tensor process group")
    ep_group = group if topology.ep_size == tp else groups.get("ep")
    etp_group = group if topology.ep_size == 1 else groups.get("etp")
    if topology.ep_size > 1 and ep_group is None:
        raise ValueError("typed numerical oracle requires its expert owner process group")
    if topology.expert_tensor_size > 1 and etp_group is None:
        raise ValueError("typed numerical oracle requires its expert tensor process group")
    from cubic_serve.models.parallel_weights import IndexerTP
    index_plan = IndexerTP(t.qsa.query_heads, t.qsa.head_dim, t.qsa.compression_ratio, tp, lane)
    logical = logical_quantized_weights(
        config if draft_provider is None else source.original_model_config)
    subset = admission.parallel_plan.bindings_for_rank(rank)
    block = source.arena.layout.block_size_tokens
    prefix_binding = (id(source), id(w), repr(policy), repr(row_plan),
                      None if draft_provider is None else id(draft_provider),
                      repr(source.arena.layout.storage),
                      None if source.arena.shared_storage is None else (
                          source.arena.shared_storage.schemas,
                          source.arena.shared_storage.extra_state_fields))
    restored_prefix = (None if prefix is None else
                       prefix.restore(prefix_store, prefix_binding, w.device))
    if operator_observers is not None and (
        capture_pool is None or prefix is not None or completed_transition is not None
    ):
        raise ValueError('current operator witnesses require a cold, exclusively owned audit capture')
    start_position = 0 if restored_prefix is None else restored_prefix["position"]
    cache_storage = source.arena.layout.storage
    horizon = state_audit_horizon(
        2 * t.qsa.compression_ratio, admission.max_sequence_tokens, cache_storage,
    )
    if diagnose_independent_trajectory and token_ids is not None:
        # Explicit diagnostic corpora may cross compression/page boundaries.
        # This does not widen the default startup witness or grant qualification.
        horizon = admission.max_sequence_tokens
    if token_ids is not None:
        if not token_ids or len(token_ids) > horizon or any(
            type(token) is not int or not 0 <= token < t.vocabulary_size for token in token_ids
        ):
            raise ValueError("typed oracle token corpus exceeds its declared bounds")
        horizon = len(token_ids)
    if start_position + horizon > admission.max_sequence_tokens:
        raise ValueError("live audit prefix exceeds admitted context")
    if (input_feature_rows and len(input_feature_rows) != horizon) or (
        position_axes and len(position_axes) != horizon
    ):
        raise ValueError("typed oracle feature/position corpus differs from token extent")
    if draft_provider is not None and (
        len(feedback_rows) != horizon or any(
            row.shape != (1, 1, draft_provider.spec.feedback_width)
            or row.dtype is not torch.bfloat16 or row.device != w.device
            or not bool(torch.isfinite(row).all()) for row in feedback_rows
        )
    ):
        raise ValueError("draft oracle feedback corpus differs from its declared boundary")
    if independent_feedback_rows and (draft_provider is None
        or len(independent_feedback_rows) != horizon or any(
            row.shape != feedback_rows[i].shape or row.dtype != feedback_rows[i].dtype
            or row.device != w.device or not bool(torch.isfinite(row).all())
            for i, row in enumerate(independent_feedback_rows))):
        raise ValueError("independent feedback corpus differs from its producer boundary")
    if any(row and len(row) != 2 * t.hidden_size for row in input_feature_rows):
        raise ValueError("typed oracle feature width differs from decoder input")
    if any(len(axes) != 3 or any(type(x) is not int or not 0 <= x < t.max_position_embeddings
                               for x in axes) for axes in position_axes):
        raise ValueError("typed oracle position corpus exceeds declared axes")
    features = tuple(
        torch.frombuffer(bytearray(row), dtype=torch.bfloat16).to(w.device)[None, None]
        if row else None for row in input_feature_rows
    )
    pages = (start_position + horizon + block - 1) // block
    names = {id(w[b.tensor_name]): b.tensor_name for b in subset}
    row_weights = {
        id(w[b.tensor_name])
        for b in subset
        if b.shard_axis == 1 and b.local_shape[1] < b.global_shape[1]
    }
    records = []
    linear_scales = []
    ordered_events = []
    normalizations = []
    index_query_norms = set()
    index_key_norms = {}
    pre_routes = []
    route_events = []
    first_expert = topology.expert_coordinates(rank)[0] * (t.moe.expert_count // topology.ep_size)
    expert_range = (first_expert, first_expert + t.moe.expert_count // topology.ep_size)
    recorder = EPLinearRecorder(expert_range) if topology.ep_size > 1 else Recorder()
    original_linear = TextGraphBindings.linear
    original_linear_group = TextGraphBindings.linear_group
    original_norm = TextGraphBindings.norm
    original_combine_norm = TextGraphBindings.combine_norm
    from cubic_serve.ops.cuda.moe import MoeSharedCombinePrimitive
    original_moe_combine = MoeSharedCombinePrimitive.launch
    moe_combine_events = []

    def observe_moe_combine(self, shared, gate, routed, output):
        inputs = tuple(z.detach().clone() for z in (shared, gate, routed))
        result = original_moe_combine(self, shared, gate, routed, output)
        ordered_events.append(("moe_shared_combine", tuple(output.shape)))
        moe_combine_events.append((*inputs, output.detach().clone()))
        return result

    original_moe = TextGraphBindings.moe
    moe_events = {}

    def observe_moe(self, x, **options):
        block = original_moe(self, x, **options)
        router = options["router"]
        def launch():
            before = x.detach().clone()
            block.launch()
            ordered_events.append(("moe_completion", names[id(router)]))
            moe_events[id(router)] = (before, block.output.detach().clone())
        from cubic_serve.executor.text_graph import BoundTextBlock
        return BoundTextBlock(block.output, (launch,))
    original_sum = FixedRankBF16SumPrimitive.launch
    original_routed = PackedRoutedExpertsPrimitive.launch

    def observe_routes(self):
        result = original_routed(self)
        ordered_events.append(("route_probability", len(route_events)))
        route_events.append(tuple(
            value.detach().clone()
            for value in (self._operands[1], self.route_ids, self.route_weights)
        ))
        return result

    def observe_linear(self, x, weight, output, **kwargs):
        call = original_linear(self, x, weight, output, **kwargs)
        encoded = self.operators.int8_input(x) if x.dtype == torch.int8 else None

        def launch():
            call()
            ordered_events.append(("linear", names[id(weight)]))
            records.append((weight, x.detach().clone(), output.detach().clone()))
            linear_scales.append(None if encoded is None else encoded.scales.detach().clone())

        return launch

    def observe_linear_group(self, x, weights, outputs, **kwargs):
        call = original_linear_group(self, x, weights, outputs, **kwargs)
        encoded = self.operators.int8_input(x) if x.dtype == torch.int8 else None

        def launch():
            call()
            for weight, output in zip(weights, outputs, strict=True):
                ordered_events.append(("linear", names[id(weight)]))
                records.append((weight, x.detach().clone(), output.detach().clone()))
                linear_scales.append(None if encoded is None else encoded.scales.detach().clone())

        return launch

    def observe_norm(x, weight, output, *, epsilon, group_size=None):
        call = original_norm(x, weight, output, epsilon=epsilon, group_size=group_size)

        def launch():
            call()
            ordered_events.append(("norm", names[id(weight)]))
            normalizations.append(
                (x.detach().clone(), weight, output.detach().clone(), epsilon, group_size)
            )

        return launch

    from cubic_serve.ops.cuda.attention.sparse_indexer import IndexerPreparation
    original_index_prepare = IndexerPreparation.prepare
    original_index_finish = IndexerPreparation.finish

    def observe_index_prepare(self):
        original_index_prepare(self)
        heads, width = self.plan.local_query_heads, self.plan.head_dim
        ordered_events.append(("norm", names[id(self.qweight)]))
        index_query_norms.add(len(normalizations))
        normalizations.append((
            self.projected[:, :heads * width].detach().clone().reshape(
                self.projected.shape[0], heads, width
            ),
            self.qweight, self.normalized_query[:, :heads].detach().clone(), self.epsilon, None,
        ))

    def observe_index_finish(self):
        original_index_finish(self)
        ordered_events.append(("norm", names[id(self.kweight)]))
        index_key_norms[len(normalizations)] = self.offsets.detach().clone()
        normalizations.append((self.pooled.detach().clone(), self.kweight,
                               self.normalized_key.detach().clone(), self.epsilon, None))

    def observe_sum(self, value):
        for part in self.input_views(value):
            ordered_events.append(("sum", tuple(part.shape)))
            if part.ndim == 3:
                pre_routes.append(part.detach().clone())
        return original_sum(self, value)

    def observe_combine_norm(self, x, update, logits, weight, output, *, epsilon, group_size):
        call = original_combine_norm(
            self, x, update, logits, weight, output, epsilon=epsilon, group_size=group_size
        )

        def launch():
            before = x.detach().clone()
            call()
            ordered_events.append(("stream_combine", tuple(x.shape)))
            combine_events.append((before, update.detach().clone(), logits.detach().clone(),
                                   x.detach().clone(), x.shape[-1] // group_size))
            # The fused stage has materialized the same rounded residual that
            # its norm consumes. Snapshot only; reference work follows forward.
            ordered_events.append(("norm", names[id(weight)]))
            normalizations.append(
                (x.detach().clone(), weight, output.detach().clone(), epsilon, group_size)
            )

        return launch

    from cubic_serve.guardrail.quantized_graph.attention_bounds import enclose
    from cubic_serve.guardrail.quantized_graph.state_bounds import (
        conv_silu_range,
        library_interval,
        sigmoid_range,
    )
    from cubic_serve.ops.cuda.attention.paged_decode import PagedGQADecodePrimitive
    from cubic_serve.ops.cuda.pointwise.qwen_decode import SigmoidOutputGatePrimitive

    attention_events = []
    cache_events = []
    gate_events = []
    selections = {}
    gdn_norm_events = []
    from cubic_serve.ops.cuda.pointwise.qwen_decode import GatedRMSNormPrimitive

    gdn_events = []
    from cubic_serve.ops.cuda.linear_attention.gated_delta_decode import (
        CausalConv1DDecodePrimitive,
        GatedDeltaDecodePrimitive,
    )

    original_conv_launch = CausalConv1DDecodePrimitive.launch
    original_reference_conv = reference_module.depthwise_causal_conv1d
    conv_events = []

    def observe_conv(self, x, weight, state, sources, destinations, segments, out, **options):
        ordered_events.append(("conv", len(conv_events)))
        before = state[sources.long()].clone()
        result = original_conv_launch(
            self, x, weight, state, sources, destinations, segments, out, **options
        )
        conv_events.append(
            (x.clone(), weight, before, out.clone(), state[destinations.long()].clone())
        )
        return result

    original_gdn_launch = GatedDeltaDecodePrimitive.launch
    original_reference_gdn = reference_module.gated_delta_net_recurrent
    from types import SimpleNamespace

    from cubic_serve.executor import text_graph as shared_text_graph
    from cubic_serve.guardrail.quantized_graph.bounds import (
        fp32_sigmoid_bounds,
        moe_shared_combine_bounds,
        scaled_activation_bounds,
        scaled_silu_bounds,
        sigmoid_bounds,
        signed_root_sigmoid_bounds,
        stream_combine_bounds,
    )
    from cubic_serve.guardrail.quantized_graph.state_bounds import (
        fp32_gdn_output_range,
        fp32_gdn_state_range,
        native_gdn_output_range,
        native_gdn_state_range,
        normalized_range,
    )
    from cubic_serve.ops.eager import hybrid as hybrid_reference
    from cubic_serve.ops.eager.activation import compute_dtype as activation_compute_dtype

    scaled_events, mix_events, combine_events = [], [], []
    original_scaled = shared_text_graph.bind_scaled_activation
    original_mix = shared_text_graph.bind_stream_mix

    def observe_scaled(x, output, **options):
        binding = original_scaled(x, output, **options)
        def launch():
            result = binding.launch()
            ordered_events.append(("scaled_activation", tuple(x.shape)))
            scaled_events.append((x.detach().clone(), output.detach().clone(), dict(options)))
            return result
        return SimpleNamespace(launch=launch)

    def observe_mix(x, logits, output):
        binding = original_mix(x, logits, output)
        def launch():
            result = binding.launch()
            ordered_events.append(("stream_mix", tuple(output.shape)))
            mix_events.append(tuple(z.detach().clone() for z in (x, logits, output)))
            return result
        return SimpleNamespace(launch=launch)

    from cubic_serve.ops.cuda.pointwise import dilated_conv, residual_streams
    signed_events, dilated_events = [], []
    original_signed = residual_streams.bind_signed_root_gate
    original_dilated = dilated_conv.bind_dilated_depthwise_conv

    def observe_signed(key, query, value, gates, output, **options):
        binding = original_signed(key, query, value, gates, output, **options)
        def launch():
            result = binding.launch()
            ordered_events.append(("signed_root_gate", tuple(output.shape)))
            signed_events.append((*(z.detach().clone() for z in
                                    (key, query, value, gates, output)), dict(options)))
            return result
        return SimpleNamespace(launch=launch)

    def observe_dilated(x, states, sources, destinations, segments, activity, weight, output,
                        **options):
        binding = original_dilated(x, states, sources, destinations, segments, activity,
                                   weight, output, **options)
        def launch():
            before = states[sources.long()].detach().clone()
            result = binding.launch()
            ordered_events.append(("dilated_conv", tuple(output.shape)))
            after = states[destinations.long()].detach().clone()
            dilated_events.append((x.detach().clone(), before, weight,
                                   output.detach().clone(), after, dict(options)))
            return result
        return SimpleNamespace(launch=launch)

    original_l2 = hybrid_reference._l2_normalize_stored

    def observe_gdn(
        self,
        packed,
        normalized,
        a,
        b,
        alog,
        bias,
        state,
        sources,
        destinations,
        segments,
        out,
        **options,
    ):
        ordered_events.append(("gdn", names[id(alog)]))
        before = state[sources.long()].clone()
        result = original_gdn_launch(
            self,
            packed,
            normalized,
            a,
            b,
            alog,
            bias,
            state,
            sources,
            destinations,
            segments,
            out,
            **options,
        )
        gdn_events.append(
            (
                alog,
                packed.clone(),
                normalized.clone(),
                a.clone(),
                b.clone(),
                bias.clone(),
                before,
                out.clone(),
                state[destinations.long()].clone(),
            )
        )
        return result

    original_gdn_norm_launch = GatedRMSNormPrimitive.launch
    original_reference_gdn_norm = reference_module.gated_rms_norm

    def observe_gdn_norm(self, x, gate, weight, y):
        ordered_events.append(("gdn_norm", names[id(weight)]))
        result = original_gdn_norm_launch(self, x, gate, weight, y)
        gdn_norm_events.append(
            (weight, x.clone(), gate.clone(), y.clone(), self._meta[2], self._gate)
        )
        return result

    original_attention_init = PagedGQADecodePrimitive.__init__
    original_attention = PagedGQADecodePrimitive.launch
    original_gate = SigmoidOutputGatePrimitive.launch

    def observe_attention_init(self, *args, **kwargs):
        original_attention_init(self, *args, **kwargs)
        selections[id(self)] = kwargs["selected_indices"]
        assert self._attention_precision == policy.attention_dtype

    def observe_attention(self, q, k, v, table, length, y):
        ordered_events.append(("attention", self.meta.layer, self._grid[2]))
        result = original_attention(self, q, k, v, table, length, y)
        attention_events.append(
            (
                self.meta.layer,
                self.meta.scale,
                *(z.detach().clone() for z in (q, k, v, table, length, y, selections[id(self)])),
                self._grid[2],
            )
        )
        encoding, window = self._encoding, self._window
        cache_events.append(None if encoding is None else (
            replace(encoding, key_scales=(None if encoding.key_scales is None else
                                         encoding.key_scales.detach().clone()),
                    value_scales=(None if encoding.value_scales is None else
                                  encoding.value_scales.detach().clone())),
            replace(window, **{name: value.detach().clone() for name in (
                "key", "value", "start_positions", "start_rows", "sources",
                "key_tail", "value_tail", "activity", "producer_query", "producer_key",
                "producer_value") if (value := getattr(window, name)) is not None}),
        ))
        return result

    def observe_gate(self, x, gate, y):
        ordered_events.append(("gate", tuple(x.shape)))
        result = original_gate(self, x, gate, y)
        gate_events.append(tuple(z.detach().clone() for z in (x, gate, y)))
        return result

    from cubic_serve.ops.cuda.attention.sparse_indexer import PagedBlockSelection

    selector_inputs = {}
    selector_events = []
    original_selector_init = PagedBlockSelection.__init__
    original_selector_launch = PagedBlockSelection.launch

    def observe_selector_init(self, *args, **kwargs):
        original_selector_init(self, *args, **kwargs)
        selector_inputs[id(self)] = (kwargs["compression"], kwargs["token_budget"], args)

    def observe_selector(self):
        ordered_events.append(("selector", len(selector_events)))
        result = original_selector_launch(self)
        ratio, budget, args = selector_inputs[id(self)]
        selector_events.append(
            (ratio, budget, *(value.detach().clone() for value in args), self.scratch[0].clone())
        )
        return result

    from cubic_serve.ops.cuda.attention.paged_decode import paged_gqa_kv_splits
    capture_binding = (*prefix_binding, pages, paged_gqa_kv_splits(1))
    from .capture_session import OBSERVATION_FIELDS
    capture_payload = None if capture_pool is None else capture_pool.acquire(capture_binding)
    capture_retained = capture_payload is not None
    if capture_payload is None:
        if capture_pool is not None:
            capture_pool.prepare(w.device)
            # Exclude retirement of prior unreachable graph cycles from this
            # allocation delta. Live captures remain explicitly owned.
            gc.collect()
        before_capture_bytes = torch.cuda.memory_allocated(w.device)
        with ExitStack() as observer_context:
            if row_plan is not None:
                from cubic_serve.quantization.common.small_rows_cost import row_arithmetic_scope
                observer_context.enter_context(row_arithmetic_scope(row_plan))
            for context in (
                patch.object(shared_text_graph, "bind_scaled_activation", observe_scaled),
                patch.object(shared_text_graph, "bind_stream_mix", observe_mix),
                patch.object(residual_streams, "bind_signed_root_gate", observe_signed),
                patch.object(dilated_conv, "bind_dilated_depthwise_conv", observe_dilated),
                patch.object(IndexerPreparation, "prepare", observe_index_prepare),
                patch.object(IndexerPreparation, "finish", observe_index_finish),
                patch.object(CausalConv1DDecodePrimitive, "launch", observe_conv),
                patch.object(GatedDeltaDecodePrimitive, "launch", observe_gdn),
                patch.object(GatedRMSNormPrimitive, "launch", observe_gdn_norm),
                patch.object(PagedBlockSelection, "__init__", observe_selector_init),
                patch.object(PagedBlockSelection, "launch", observe_selector),
                patch.object(PagedGQADecodePrimitive, "__init__", observe_attention_init),
                patch.object(PagedGQADecodePrimitive, "launch", observe_attention),
                patch.object(SigmoidOutputGatePrimitive, "launch", observe_gate),
                patch.object(PackedRoutedExpertsPrimitive, "launch", observe_routes),
                recorder.observe(),
                patch.object(TextGraphBindings, "linear", observe_linear),
                patch.object(TextGraphBindings, "linear_group", observe_linear_group),
                patch.object(TextGraphBindings, "norm", staticmethod(observe_norm)),
                patch.object(TextGraphBindings, "combine_norm", observe_combine_norm),
                patch.object(TextGraphBindings, "moe", observe_moe),
                patch.object(MoeSharedCombinePrimitive, "launch", observe_moe_combine),
                patch.object(FixedRankBF16SumPrimitive, "launch", observe_sum),
            ):
                observer_context.enter_context(context)
            stage_constructor = (FlashNextDecodeStage if draft_provider is None
                                 else lambda config, w, topology, **options:
                                 draft_provider.build_stage(w, topology, **options))
            stage = stage_constructor(config, w, topology,
                tp_group=group,
                pp_group=None,
                ep_group=ep_group,
                expert_tp_group=etp_group,
                captured_m=1,
                table_pages=pages,
                page_size=block,
                arena=_audit_cache_arena(source, pages),
                operator_config=policy,
                recurrent_state_dtype=source.recurrent_state_dtype,
                cache_storage=cache_storage if cache_storage.quantized else None,
                mrope_tables=source.mrope_tables,
            )
            original_launch = stage.launch

            def observed_launch(*, produce_logits=True):
                if not produce_logits:
                    collections = (
                        records, linear_scales, ordered_events, normalizations, index_query_norms,
                        index_key_norms, pre_routes, route_events, recorder.experts,
                        attention_events, cache_events, gate_events, selector_events,
                        gdn_norm_events, gdn_events,
                        conv_events, scaled_events, mix_events, combine_events, moe_events,
                        moe_combine_events, signed_events, dilated_events,
                    )
                    with preserve_observer_collections(collections):
                        original_launch(produce_logits=False)
                    return
                records.clear()
                linear_scales.clear()
                ordered_events.clear()
                normalizations.clear()
                index_query_norms.clear()
                index_key_norms.clear()
                pre_routes.clear()
                route_events.clear()
                recorder.experts.clear()
                attention_events.clear()
                cache_events.clear()
                gate_events.clear()
                selector_events.clear()
                gdn_norm_events.clear()
                gdn_events.clear()
                conv_events.clear()
                scaled_events.clear()
                mix_events.clear()
                combine_events.clear()
                moe_events.clear()
                moe_combine_events.clear()
                signed_events.clear()
                dilated_events.clear()
                original_launch()

            stage.launch = observed_launch
            # Retired observers can own mapped tables whose finalizers synchronize
            # or unregister memory. Run those finalizers outside every CUDA capture.
            collect_enabled = gc.isenabled()
            if collect_enabled:
                gc.collect()
                gc.disable()
            try:
                stage.capture()
            finally:
                if collect_enabled:
                    gc.enable()
        captured_experts = tuple(recorder.experts)
        if capture_pool is not None:
            captured_locals = locals()
            capture_payload = {name: captured_locals[name] for name in OBSERVATION_FIELDS}
            capture_retained = capture_pool.retain(
                capture_binding, capture_payload,
                max(0, torch.cuda.memory_allocated(w.device) - before_capture_bytes),
            )
    else:
        (stage, records, linear_scales, ordered_events, normalizations, index_query_norms, index_key_norms, pre_routes, route_events, recorder, attention_events, cache_events, gate_events, selector_events, gdn_norm_events, gdn_events, conv_events, scaled_events, mix_events, combine_events, moe_events, moe_combine_events, signed_events, dilated_events, captured_experts) = (
            capture_payload[name] for name in OBSERVATION_FIELDS)
    audit_completed = False
    try:
        base_records = tuple(records)
        base_linear_scales = tuple(linear_scales)
        base_experts = captured_experts
        prepare_rows(stage, 1)
        stage.arena.reset_page_rows(torch.arange(pages, device=w.device))
        reference_state = None
        independent_state = None
        previous_feedback = None
        independent_feedback = None
        native_key_history, native_value_history = {}, {}
        if restored_prefix is not None:
            native = restored_prefix["native"]
            if native["states"].keys() != stage.arena.states.keys() or (
                native["pages"].keys() != stage.arena.page_fields.keys()
            ):
                raise ValueError("live prefix native storage schema differs")
            for name, value in native["states"].items():
                stage.arena.states[name][0].copy_(value)
            for name, value in native["pages"].items():
                stage.arena.page_fields[name][:value.shape[0]].copy_(value)
            for name in ("key", "value", "written"):
                value = native[name]
                getattr(stage.arena, name)[:value.shape[0]].copy_(value)
            reference_state = restored_prefix["causal"]
            independent_state = restored_prefix["independent"]
            previous_feedback = restored_prefix["feedback"]
            independent_feedback = restored_prefix["independent_feedback"]
            native_key_history = restored_prefix["native_key_history"]
            native_value_history = restored_prefix["native_value_history"]
            del restored_prefix, native
        reports = []

        audit_error_status = (
            torch.empty((), dtype=torch.int32, device=w.device) if tp > 1 else None
        )

        def coordinated(error):
            if audit_error_status is not None:
                audit_error_status.fill_(int(error is not None))
                dist.all_reduce(audit_error_status, op=dist.ReduceOp.MAX, group=group)
                if audit_error_status.item() == 0:
                    return
            errors = [None] * tp
            dist.all_gather_object(errors, error, group=group)
            if any(item is not None for item in errors):
                raise AssertionError(str(errors))

        def fixed_sum(value, process_group=group):
            size = dist.get_world_size(process_group)
            values = [torch.empty_like(value) for _ in range(size)]
            dist.all_gather(values, value.contiguous(), group=process_group)
            result = torch.zeros_like(value, dtype=torch.float32)
            for item in values:
                result.add_(item.float())
            return result.to(value.dtype)

        def gather_owned_normalized_rows(value, tokens, heads, parts, group):
            if dist.get_world_size(group) != parts or value.shape[0] != tokens.numel() * heads:
                raise ValueError("indexer normalization ownership geometry differs")
            pieces = [torch.empty_like(value) for _ in range(parts)]
            dist.all_gather(pieces, value.contiguous(), group=group)
            owners = tokens.remainder(parts).repeat_interleave(heads).long()
            rows = torch.arange(value.shape[0], device=value.device)
            return torch.stack(pieces)[owners, rows]

        def gather_index_queries(value):
            # This diagnostic collective is only called after schedule agreement.
            rows, _, width = value.shape
            capacity = (index_plan.query_heads + tp - 1) // tp
            padded = torch.zeros((rows, capacity, width), device=value.device, dtype=value.dtype)
            padded[:, :index_plan.local_query_heads].copy_(value)
            pieces = [torch.empty_like(padded) for _ in range(tp)]
            dist.all_gather(pieces, padded, group=group)
            return torch.cat([piece[:, :index_plan.query_heads * (r + 1) // tp
                                    - index_plan.query_heads * r // tp]
                              for r, piece in enumerate(pieces)], dim=1)

        def complete_index_projection(value, name):
            if tp > 1 and name in column_completion_names:
                pieces = [torch.empty_like(value) for _ in range(tp)]
                dist.all_gather(pieces, value.contiguous(), group=group)
                return torch.cat(pieces, dim=-1)
            if tp > 1 and name.endswith(".indexer.index_qk_proj.weight"):
                shape = value.shape[:-1]
                width = index_plan.head_dim
                heads = index_plan.local_query_heads
                queries = gather_index_queries(value[..., :heads * width].reshape(
                    value.numel() // value.shape[-1], heads, width
                ))
                key = value[..., heads * width:].contiguous()
                pieces = [torch.empty_like(key) for _ in range(tp)]
                dist.all_gather(pieces, key, group=group)
                return torch.cat((queries.reshape(*shape, -1), *pieces), dim=-1)
            return value

        local_prefixes = {
            names[identity].removesuffix(".gate.weight")
            for identity in stage.helper.local_tp_routers
        }
        in_local_mlp = False
        in_completed_mlp = False
        completed_mlp_terms = []
        local_mlp_terms = []
        current_moe_weights = None
        current_route_ids = None
        current_shared_gate = None
        moe_completion_proofs = []
        original_mlp = Qwen4ExpReferenceModel._mlp

        def causal_mlp(model, hidden, prefix):
            nonlocal in_local_mlp, local_mlp_terms, current_shared_gate, moe_combine_used
            nonlocal in_completed_mlp, completed_mlp_terms
            if prefix not in local_prefixes:
                # This plan completes each projection before weighting. The
                # local-weighted plan below instead completes the final fold.
                # Both must witness the same native shared-sigmoid boundary.
                completed_mlp_terms = []
                in_completed_mlp = True
                try:
                    result = original_mlp(model, hidden, prefix)
                finally:
                    in_completed_mlp = False
                terms = dict(completed_mlp_terms)
                selected = tuple(int(e) for e in current_route_ids.reshape(-1).tolist())
                expected_names = {
                    prefix + f".experts.{e}.down_proj.weight" for e in selected
                    if first_expert <= e < expert_range[1]
                } | {prefix + ".shared_expert.down_proj.weight",
                     prefix + ".shared_expert_gate.weight"}
                coordinated(None if len(terms) == len(completed_mlp_terms)
                            and set(terms) == expected_names else
                            "completed MoE projection ownership differs")
                routed = torch.zeros_like(hidden.reshape(-1, hidden.shape[-1]))
                for slot, expert in enumerate(selected):
                    if first_expert <= expert < expert_range[1]:
                        value = terms[prefix + f".experts.{expert}.down_proj.weight"]
                        product_value = (value.float() * current_moe_weights[:, slot:slot+1]
                                         .float()).to(routed.dtype)
                        routed = (routed.float() + product_value.float()).to(routed.dtype)
                if topology.ep_size > 1:
                    routed = fixed_sum(routed, ep_group)
                cs, cg, cr, actual = moe_combine_events[moe_combine_used]
                moe_combine_used += 1
                error = None
                try:
                    proof = _completed_moe_combine_proof(
                        cs, cg, cr, actual,
                        expected_shared=terms[prefix + ".shared_expert.down_proj.weight"]
                            .reshape_as(cs),
                        expected_gate=terms[prefix + ".shared_expert_gate.weight"].reshape_as(cg),
                        expected_routed=routed.reshape_as(cr),
                        activation_dtype=activation_compute_dtype(cg.dtype, cg.device),
                    )
                    observed_input, observed_output = moe_events[id(w[prefix + ".gate.weight"])]
                    assert same(hidden.reshape_as(observed_input), observed_input), (
                        "unproved completed MoE input"
                    )
                    assert same(actual.reshape_as(observed_output), observed_output), (
                        "completed MoE output schedule differs"
                    )
                except Exception as exc:
                    error = prefix + ": " + str(exc)
                coordinated(error)
                activation_proofs.append({"kind": "completed_moe_shared_combine", **proof})
                return actual.reshape_as(result).clone()
            local_mlp_terms = []
            current_shared_gate = None
            in_local_mlp = True
            try:
                partial = original_mlp(model, hidden, prefix)
            finally:
                in_local_mlp = False
            completed = fixed_sum(partial)
            # Independent early/late comparison on the same already-witnessed
            # local Marlin products and stored routing/gating values. No mixed
            # producer provenance and no reference launches in observer callbacks.
            assert (current_moe_weights is not None and current_route_ids is not None
                    and current_shared_gate is not None)
            top_k = current_moe_weights.shape[-1]
            routed_terms = {}
            shared_term = None
            for name, value in local_mlp_terms:
                if ".experts." in name:
                    expert_id = int(name.split(".experts.", 1)[1].split(".", 1)[0])
                    assert expert_id not in routed_terms
                    routed_terms[expert_id] = value
                else:
                    assert name.endswith(".shared_expert.down_proj.weight")
                    assert shared_term is None
                    shared_term = value
            assert shared_term is not None
            cs, cg, cr, local_actual = moe_combine_events[moe_combine_used]
            moe_combine_used += 1
            coordinated(None if same(shared_term.reshape_as(cs), cs)
                        and same(current_shared_gate.reshape_as(cg), cg)
                        else "unproved shared MoE combine inputs")
            selected = tuple(int(expert) for expert in current_route_ids.reshape(-1).tolist())
            assert len(selected) == top_k and len(set(selected)) == top_k
            assert set(routed_terms) == {
                expert for expert in selected
                if first_expert <= expert < first_expert + t.moe.expert_count // topology.ep_size
            }
            # An EP owner records only its checkpoint experts. Empty slots are
            # exact zero terms in the subsequent full-TP completion witness.
            zero = torch.zeros_like(shared_term)
            local_terms = torch.cat(
                [routed_terms.get(expert, zero) for expert in selected] + [shared_term], dim=0
            )
            all_terms = [torch.empty_like(local_terms) for _ in range(tp)]
            dist.all_gather(all_terms, local_terms, group=group)
            terms = torch.stack(all_terms).cpu()
            weighted = current_moe_weights.detach().cpu().reshape(top_k)
            routed_reference = torch.zeros_like(cr.cpu())
            for slot in range(top_k):
                term = local_terms[slot].cpu().reshape_as(routed_reference)
                product_value = (term.float() * weighted[slot].float()).to(cr.dtype)
                routed_reference = (routed_reference.float() + product_value.float()).to(cr.dtype)
            coordinated(None if same(cr.cpu(), routed_reference)
                        else "routed stored product/fold or ownership differs")
            local_lo, local_hi = moe_shared_combine_bounds(
                cs.cpu(), cg.cpu(), cr.cpu(),
                activation_dtype=activation_compute_dtype(cg.dtype, cg.device),
            )
            local_proof = check(local_actual.cpu(), local_lo, local_hi)
            coordinated(None if local_proof["outside_enclosure"] == 0
                        else f"shared MoE combine: {local_proof}")
            activation_proofs.append({"kind": "moe_shared_combine", **local_proof})
            gate_value = reference_module.sigmoid_stored(current_shared_gate).cpu().reshape(1)
            early_terms = torch.zeros_like(terms[0], dtype=torch.float32)
            for rank_terms in terms:
                early_terms = early_terms + rank_terms.float()
            early_terms = early_terms.to(local_terms.dtype)
            early = torch.zeros_like(early_terms[0])
            for slot in range(top_k):
                product = (early_terms[slot].float() * weighted[slot].float()).to(early.dtype)
                early = (early.float() + product.float()).to(early.dtype)
            shared = (early_terms[-1].float() * gate_value.float()).to(early.dtype)
            early = (early.float() + shared.float()).to(early.dtype)
            local_coefficients = torch.cat((
                current_moe_weights.reshape(-1).double(),
                reference_module.sigmoid_stored(current_shared_gate).reshape(-1).double(),
            ))
            coefficient_parts = [torch.empty_like(local_coefficients) for _ in range(tp)]
            dist.all_gather(coefficient_parts, local_coefficients, group=group)
            coefficients = torch.stack(coefficient_parts).cpu()
            local_coefficients = local_coefficients.cpu()
            # Replicated floating producers need not be bit-identical. Retain
            # their independently witnessed actual values and price their
            # coefficient variation explicitly in the cross-tree enclosure.
            absolute_terms = terms.double().abs()
            late_magnitude = (absolute_terms * coefficients.abs()[..., None]).sum((0, 1))
            early_magnitude = (absolute_terms * local_coefficients.abs()[None, :, None]).sum((0, 1))
            coefficient_difference = (coefficients - local_coefficients[None]).abs()
            coefficient_error = (absolute_terms * coefficient_difference[..., None]).sum((0, 1))
            # Both trees have at most K+4 low-precision rounding factors per
            # term, and K+TP+4 FP32 factors. Products of two stored BF16 inputs
            # are exactly representable in normal FP32; the FP32 allowance is
            # conservative and also covers the rank sums and casts.
            u = 2.0 ** (-8 if early.dtype == torch.bfloat16 else -11)
            bf_count, fp_count = top_k + 4, top_k + tp + 4
            assert bf_count * u < 1
            gb = bf_count * u / (1 - bf_count * u)
            gf = fp_count * 2.0**-24 / (1 - fp_count * 2.0**-24)
            gamma = (1 + gb) * (1 + gf) - 1
            tiny = 2.0 ** (-133 if early.dtype == torch.bfloat16 else -24)
            underflow = 2 * tp * (top_k + 4) * tiny * (1 + gamma) * (1 + coefficients.abs().max())
            budget = gamma * (early_magnitude + late_magnitude) + coefficient_error + underflow
            # Bound shared-sigmoid approximation separately from the existing
            # weighted-reduction allowance. Every gate input is a witnessed
            # original-weight projection, independently of the chosen TP tree.
            gate_input = current_shared_gate.detach().cpu().double()
            act = activation_compute_dtype(current_shared_gate.dtype, current_shared_gate.device)
            if act is torch.float32:
                gl, gh = fp32_sigmoid_bounds(gate_input)
            else:
                gl, gh = sigmoid_bounds(gate_input, dtype=act)
            gl = gl.to(current_shared_gate.dtype).double()
            gh = gh.to(current_shared_gate.dtype).double()
            gate_error = torch.maximum((gl.reshape(1) - gate_value.double()).abs(),
                                       (gh.reshape(1) - gate_value.double()).abs())
            gate_error_device = gate_error.to(current_shared_gate.device)
            gate_errors = [torch.empty_like(gate_error_device) for _ in range(tp)]
            dist.all_gather(gate_errors, gate_error_device, group=group)
            gate_budget = (absolute_terms[:, -1] * torch.stack(gate_errors).cpu()).sum(0)
            budget = budget + gate_budget * (1 + gamma)
            router = w[prefix + ".gate.weight"]
            observed_input, observed_output = moe_events[id(router)]
            coordinated(None if same(hidden.reshape_as(observed_input), observed_input)
                        else "unproved MoE block input")
            coordinated(None if same(fixed_sum(local_actual).reshape_as(observed_output),
                                     observed_output) else "MoE TP completion differs")
            actual = observed_output.detach().cpu().reshape_as(early)
            assert all(
                bool(torch.isfinite(value).all()) for value in (terms, early, actual, budget)
            ), (
                "nonfinite local-weighted TP witness", prefix,
            )
            proof = check(actual, early.double() - budget, early.double() + budget)
            coordinated(None if proof["outside_enclosure"] == 0 else
                        f"{prefix}: local-weighted TP enclosure {proof}")
            upper = early.double() + budget
            corrupted = torch.nextafter(upper, torch.full_like(upper, float("inf")))
            if check(corrupted, early.double() - budget, upper)["outside_enclosure"] == 0:
                raise RuntimeError("local-weighted TP counterexample harness failed")
            moe_completion_proofs.append(dict(
                layer=prefix, rule="local_stored_weighted_fold_then_fp32_tp_sum",
                top_k=top_k, ranks=tp, storage_roundings_per_term=bf_count,
                fp32_roundings_per_term=fp_count,
                raw_early_late_max_abs=float((actual.double() - early.double()).abs().max()),
                raw_early_late_changed=int((actual != early).sum()),
                max_allowed_error=float(budget.max()),
                max_replicated_coefficient_difference=float(coefficient_difference.max()),
                max_coefficient_variation_error=float(coefficient_error.max()), **proof,
            ))
            return observed_output.reshape_as(completed).clone()

        original_topk_routes = reference_module.topk_routes

        def causal_routes(logits, *, top_k, normalize):
            nonlocal route_used, current_moe_weights, current_route_ids
            cx, ids, weights = route_events[route_used]
            error = None
            try:
                assert normalize, "route probability witness requires normalized top-k"
                assert same(logits.reshape_as(cx), cx), "unproved router logits"
                expected_ids, eager_weights = original_topk_routes(
                    logits, top_k=top_k, normalize=normalize
                )
                assert same(ids.to(expected_ids.dtype), expected_ids), "route identities"
                assert weights.dtype == eager_weights.dtype, "route probability storage"
                # Reuse the named FP32 exp/sum/division allowance. The original
                # router linear has already proved these actual candidate logits.
                # All reference math is deferred until the observed forward and
                # topology-wide ordered-event check have completed.
                lo, hi = route_softmax_bounds(cx.gather(1, expected_ids), weights.dtype)
                proof = check(weights.cpu(), lo, hi)
                assert proof["outside_enclosure"] == 0, proof
                route_proofs.append({
                    "boundary": route_used,
                    "rule": "route_fp32_exp_sum_division",
                    "raw_eager_differences": int((weights != eager_weights).sum()),
                    **proof,
                })
            except Exception as exc:
                error = f"route probability {route_used}: {exc}"
            coordinated(error)
            route_used += 1
            current_route_ids = expected_ids.clone()
            current_moe_weights = weights.clone()
            return expected_ids, weights.clone()

        def causal_norm(x, weight, *, group_size=None, eps=None, epsilon=None, output_dtype=None):
            cx, cw, y, cepsilon, group = norm_events[id(weight)]
            error = None
            width = cx.shape[-1] if group is None else group
            try:
                assert (epsilon if epsilon is not None else eps) == cepsilon
                if group_size is not None:
                    assert group_size == width
                assert same(x.reshape_as(cx), cx), "unproved input " + names[id(weight)]
                xx = cx.cpu().double().reshape(-1, width)
                ww = cw.cpu().float().add(1).double().reshape(-1, width)
                variance = xx.square().sum(-1, keepdim=True) / width
                gamma = width * 2 ** (-24) / (1 - width * 2 ** (-24))
                eps32 = float(torch.tensor(cepsilon, dtype=torch.float32))
                vlo = (variance * (1 - gamma) + eps32).float().double()
                vhi = (variance * (1 + gamma) + eps32).float().double()
                lo, hi = product(
                    xx,
                    xx,
                    torch.rsqrt(vhi) * (1 - 3 * 2 ** (-23)),
                    torch.rsqrt(vlo) * (1 + 3 * 2 ** (-23)),
                )
                lo, hi = product(lo.float().double(), hi.float().double(), ww, ww)
                proof = check(
                    y.cpu().reshape_as(lo),
                    lo.float().to(y.dtype).double(),
                    hi.float().to(y.dtype).double(),
                )
                assert proof["outside_enclosure"] == 0, proof
            except Exception as exc:
                error = names[id(weight)] + ": " + str(exc)
            coordinated(error)
            norm_used.add(id(weight))
            return y.reshape_as(x).clone()

        def causal_conv(x, weight, state, bias=None, **options):
            nonlocal conv_used
            index = conv_used
            conv_used += 1
            error = None
            try:
                cx, cw, cs, actual, after = conv_events[index]
                assert bias is None and options["activation"] == "silu"
                assert options["output_dtype"] == actual.dtype
                assert same(x.reshape_as(cx), cx), "unproved convolution input"
                assert same(weight, cw), "unproved convolution weight"
                assert same(state, cs), "unproved convolution state"
                expected, expected_state = original_reference_conv(
                    x, weight, state, bias, **options
                )
                independent_state = torch.cat((state, x[:, 0, :, None]), dim=-1)[..., 1:]
                assert same(independent_state, after), "convolution state writeback"
                assert same(expected_state, after), "reference convolution state"
                lower, upper = conv_silu_range(
                    cx.cpu(), cw.cpu(), cs.cpu(), dtype=actual.dtype,
                    activation_dtype=activation_compute_dtype(actual.dtype, actual.device),
                )
                proof = check(actual.cpu(), lower, upper)
                assert proof["outside_enclosure"] == 0, proof
                corrupted = actual.cpu().clone()
                corrupted.flatten()[0] = upper.flatten()[0].abs() * 2 + 1
                assert check(corrupted, lower, upper)["outside_enclosure"] > 0, (
                    "convolution enclosure accepts corruption"
                )
                conv_proofs.append(
                    {
                        "index": index,
                        "raw_eager_differences": int((actual != expected.reshape_as(actual)).sum()),
                        **proof,
                    }
                )
            except Exception as exc:
                error = f"convolution{index} at position{position}: {exc}"
            coordinated(error)
            return actual.reshape_as(x).clone(), independent_state

        def diagnose_gdn(q, k, v, a, b, alog, bias, state, **options):
            error = None
            try:
                event = next(e for e in gdn_events if e[0] is alog)
                _, packed, normalized, ca, cb, cbias, before, out, after = event
                width = q.shape[-2] * q.shape[-1]
                pq, pk, pv = packed.split((width, width, v.shape[-2] * v.shape[-1]), -1)
                comparisons = {
                    name: same(left.reshape_as(right), right)
                    for name, left, right in (
                        ("q", q, pq),
                        ("k", k, pk),
                        ("v", v, pv),
                        ("a", a, ca),
                        ("b", b, cb),
                        ("bias", bias, cbias),
                        ("state", state, before),
                    )
                }
                assert all(comparisons.values()), (
                    "unproved GDN input",
                    names[id(alog)],
                    comparisons,
                )
                normalized_pairs = [
                    (q, normalized[:, :width]),
                    (k, normalized[:, width : 2 * width]),
                ]

                def causal_l2(values, *, epsilon, output_dtype):
                    original, target = normalized_pairs.pop(0)
                    assert same(values, original.reshape_as(values).float()), "unproved L2 producer"
                    assert epsilon == options["l2_epsilon"] and target.dtype == output_dtype
                    target = target.reshape_as(values)
                    lower, upper = normalized_range(
                        values.detach().cpu().double(), epsilon, dtype=output_dtype
                    )
                    proof = check(target.cpu(), lower, upper)
                    assert proof["outside_enclosure"] == 0, proof
                    corrupted = target.cpu().clone()
                    corrupted.flatten()[0] = upper.flatten()[0].abs() * 2 + 1
                    assert check(corrupted, lower, upper)["outside_enclosure"] > 0, (
                        "L2 enclosure accepts corruption"
                    )
                    canonical = original_l2(values, epsilon=epsilon, output_dtype=output_dtype)
                    l2_proofs.append(
                        {
                            "name": names[id(alog)],
                            "kind": "q" if normalized_pairs else "k",
                            "raw_eager_differences": int((canonical != target).sum()),
                            **proof,
                        }
                    )
                    return target.clone()

                with patch.object(hybrid_reference, "_l2_normalize_stored", causal_l2):
                    result, new = original_reference_gdn(
                        q, k, v, a, b, alog, bias, state, **options
                    )
                assert not normalized_pairs
                comparisons["output_exact"] = same(result.reshape_as(out), out)
                comparisons["state_exact"] = same(new.reshape_as(after), after)
                if not comparisons["state_exact"]:
                    query = normalized[:, :width].reshape(
                        q.shape[0], q.shape[-2], q.shape[-1]
                    )
                    key = normalized[:, width : 2 * width].reshape_as(query)
                    repeats = v.shape[-2] // q.shape[-2]
                    key = key.repeat_interleave(repeats, dim=1)
                    value = pv.reshape(v.shape[0], v.shape[-2], v.shape[-1])
                    if policy.attention_dtype == "auto":
                        lower, upper = native_gdn_state_range(
                            before, key, value, ca, cb, alog, cbias,
                            activation_dtype=activation_compute_dtype(cb.dtype, cb.device),
                        )
                    else:
                        lower, upper = fp32_gdn_state_range(
                            before, key, value, ca, cb, alog, cbias,
                            activation_dtype=activation_compute_dtype(cb.dtype, cb.device),
                        )
                    proof = check(after.cpu(), lower, upper)
                    eager_proof = check(new.reshape_as(after).cpu(), lower, upper)
                    assert proof["outside_enclosure"] == 0, proof
                    assert eager_proof["outside_enclosure"] == 0, eager_proof
                    gdn_state_proofs.append({
                        "name": names[id(alog)],
                        "kind": "declared_operands_fp32_reduction_before_state_rounding",
                        "attention_precision": policy.attention_dtype,
                        "operand_dtype": str(key.dtype),
                        "activation_dtype": str(activation_compute_dtype(cb.dtype, cb.device)),
                        "state_dtype": str(after.dtype),
                        "reduction_length": q.shape[-1],
                        "raw_eager_differences": int((new.reshape_as(after) != after).sum()),
                        **proof,
                    })
                    # This is a witnessed producer boundary, just like stored
                    # Q/K above. Both current output and next input must consume
                    # the SAME rounded state, never an unrounded temporary.
                    new = after.reshape_as(new).clone()
                    if policy.attention_dtype == "auto":
                        scale = torch.tensor(q.shape[-1] ** -0.5, device=query.device,
                                             dtype=query.dtype)
                        scaled = (query * scale).to(query.dtype).float()
                        read_state = new.to(query.dtype).float()
                    else:
                        scaled = query.float() * (q.shape[-1] ** -0.5)
                        read_state = new.float()
                    scaled = scaled.repeat_interleave(repeats, dim=1)
                    result = (
                        (read_state * scaled.unsqueeze(-2))
                        .sum(-1).to(out.dtype).reshape_as(result)
                    )
                    comparisons["state_exact"] = True
                    comparisons["output_exact"] = same(result.reshape_as(out), out)
                if comparisons["state_exact"] and not comparisons["output_exact"]:
                    # State and normalized Q are independently proved above.
                    # Only their final FP32 sum tree may differ, with complete
                    # K coverage and the declared output rounding boundary.
                    query = normalized[:, :width].reshape(
                        q.shape[0], q.shape[-2], q.shape[-1]
                    )
                    output_range = (native_gdn_output_range if policy.attention_dtype == "auto"
                                    else fp32_gdn_output_range)
                    lower, upper = output_range(new, query, dtype=out.dtype)
                    proof = check(out.cpu(), lower.reshape_as(out), upper.reshape_as(out))
                    eager_proof = check(
                        result.reshape_as(out).cpu(),
                        lower.reshape_as(out), upper.reshape_as(out),
                    )
                    assert proof["outside_enclosure"] == 0, proof
                    assert eager_proof["outside_enclosure"] == 0, eager_proof
                    gdn_output_proofs.append({
                        "name": names[id(alog)],
                        "kind": "fp32_reduction_order",
                        "reduction_length": q.shape[-1],
                        "raw_eager_differences": int((result.reshape_as(out) != out).sum()),
                        **proof,
                    })
                    # Advance the independent causal oracle only after this
                    # producer boundary has been witnessed in full.
                    result = out.reshape_as(result).clone()
                    comparisons["output_witnessed"] = True
                    del comparisons["output_exact"]
                error = None
                if not all(comparisons.values()):
                    error = str({"name": names[id(alog)], "position": position, **comparisons})
            except Exception as exc:
                error = f"GDN {names.get(id(alog), 'unknown')} at position{position}: {exc}"
            coordinated(error)
            return result, new

        def diagnose_gdn_norm(x, gate, weight, **options):
            event = next((e for e in gdn_norm_events if e[0] is weight), None)
            if event is None:
                coordinated("missing GDN norm witness " + names[id(weight)])
            _, cx, cgate, actual, epsilon, activation = event
            expected = original_reference_gdn_norm(x, gate, weight, **options)
            comparisons = {
                "core_exact": same(x.reshape_as(cx), cx),
                "gate_exact": same(gate.reshape_as(cgate), cgate),
                "output_exact": same(expected.reshape_as(actual), actual),
            }
            if (comparisons["core_exact"] and comparisons["gate_exact"]
                    and not comparisons["output_exact"]):
                from cubic_serve.guardrail.quantized_graph.bounds import gated_norm_enclosure

                proof = gated_norm_enclosure(
                    cx.cpu(), cgate.cpu(), weight.cpu(), actual.cpu(),
                    epsilon=epsilon, activation=activation,
                    activation_dtype=activation_compute_dtype(actual.dtype, actual.device),
                )
                coordinated(None if proof["outside_enclosure"] == 0 else str(proof))
                expected = actual.reshape_as(expected).clone()
                comparisons["output_exact"] = True
            error = None
            if not all(comparisons.values()):
                error = str({"name": names[id(weight)], "position": position, **comparisons})
            coordinated(error)
            return expected

        def causal_attention(q, k, v, indices):
            nonlocal attention_used, pending_gate
            index = attention_used
            attention_used += 1
            layer, scale, cq, ck, cv, table, counts, y, chosen, splits = attention_events[index]
            error = None
            try:
                length = k.shape[2]
                positions = torch.arange(length, device=k.device)
                rotate_value = False
                if cache_events[index] is None:
                    assert same(q.reshape_as(cq), cq), "unproved query"
                    keys = ck[
                        table[0, positions // ck.shape[2]].long(), layer, positions % ck.shape[2]
                    ][None]
                    values = cv[
                        table[0, positions // cv.shape[2]].long(), layer, positions % cv.shape[2]
                    ][None]
                    assert same(k.transpose(1, 2), keys), "unproved main keys"
                    assert same(v.transpose(1, 2), values), "unproved main values"
                else:
                    from cubic_serve.ops.eager.kv_cache import (
                        witness_cache_query, witness_cache_read,
                    )

                    encoding, window = cache_events[index]
                    assert window is not None, "encoded model audit requires native window"
                    witness_cache_query(q, cq, encoding=encoding, window=window)
                    rotate_value = encoding.enable_rotation and encoding.value.spec.dtype.quantized
                    keys, values = (item[None] for item in witness_cache_read(
                        k[0].transpose(0, 1), v[0].transpose(0, 1),
                        key_pages=ck, value_pages=cv, encoding=encoding, window=window,
                        table=table, layer=layer,
                    ))
                selected = indices[0, 0]
                selected = selected[selected >= 0].long().sort().values
                assert same(selected, chosen[0, : int(counts[0])].long()), "unproved selection"
                proof = enclose(
                    cq,
                    keys[:, selected],
                    values[:, selected],
                    y,
                    scale=scale,
                    splits=splits,
                    probability_dtype=(torch.float32
                                       if policy.attention_dtype == "fp32" else cq.dtype),
                    value_rotation=rotate_value,
                )
                assert proof["outside_enclosure"] == 0, proof
                attention_proofs.append(proof)
            except Exception as exc:
                error = f"attention{index}: " + str(exc)
            coordinated(error)
            pending_gate = index
            return y.reshape_as(q).clone()

        def causal_sigmoid(core, x, *, output_dtype):
            nonlocal pending_gate
            if pending_gate is not None:
                cx, cgate, y = gate_events[pending_gate]
                error = None
                try:
                    assert same(x.reshape_as(cgate), cgate), "unproved sigmoid gate"
                    assert same(core.reshape_as(cx), cx), "unproved attention output gate input"
                    assert output_dtype == y.dtype
                    assert same(cx, attention_events[pending_gate][7]), "unproved attention core"
                    gate = cgate.cpu().double()
                    operand = cx.cpu().double()
                    activation_math = activation_compute_dtype(x.dtype, x.device)
                    if activation_math is torch.float32:
                        lo, hi = library_interval(torch.exp, -gate, -gate, 2)
                        dlo, dhi = (1 + lo).float().double(), (1 + hi).float().double()
                        slo, shi = (1 / dhi).float().double(), (1 / dlo).float().double()
                    else:
                        slo, shi = sigmoid_range(gate, gate, dtype=activation_math)
                    slo, shi = slo.to(x.dtype).double(), shi.to(x.dtype).double()
                    lower, upper = product(operand, operand, slo, shi)
                    proof = check(
                        y.cpu(),
                        lower.float().to(y.dtype).double(),
                        upper.float().to(y.dtype).double(),
                    )
                    assert proof["outside_enclosure"] == 0, proof
                    gate_proofs.append(proof)
                except Exception as exc:
                    error = f"attention gate{pending_gate}: " + str(exc)
                coordinated(error)
                pending_gate = None
                return y.reshape_as(core).clone()
            raise RuntimeError("missing observed attention gate")

        def causal_signed(key, query, value, *, magnitude_floor=1e-6):
            nonlocal signed_used
            ck, cq, cv, cg, actual_gate, options = signed_events[signed_used]
            signed_used += 1
            error = None
            try:
                assert same(key.reshape_as(ck), ck), "unproved signed-root key"
                assert same(query.reshape_as(cq), cq), "unproved signed-root query"
                assert same(value.reshape_as(cv), cv), "unproved signed-root value"
                assert options.get("magnitude_floor", 1e-6) == magnitude_floor
                operands = (ck.reshape(-1, cg.shape[-1], cv.shape[-1]),
                            cq.reshape(-1, cg.shape[-1], cv.shape[-1]))
                lo, hi = signed_root_sigmoid_bounds(*(z.cpu() for z in operands),
                    activation_dtype=activation_compute_dtype(cg.dtype, cg.device),
                    magnitude_floor=magnitude_floor)
                proof = check(cg.cpu(), lo, hi)
                assert proof["outside_enclosure"] == 0, proof
                independently_multiplied = (cv.unsqueeze(1) * cg.unsqueeze(-1)).flatten(1)
                assert same(actual_gate, independently_multiplied), "signed-root stored multiply"
                activation_proofs.append({"kind": "signed_root_gate", **proof})
            except Exception as exc:
                error = "signed-root gate: " + str(exc)
            coordinated(error)
            return actual_gate.reshape(*value.shape[:-1], actual_gate.shape[-1]).clone()

        def causal_dilated(x, history, weight, *, dilation):
            nonlocal dilated_used
            cx, ch, cw, actual_conv, after, options = dilated_events[dilated_used]
            dilated_used += 1
            error = None
            try:
                assert same(x.reshape_as(cx), cx), "unproved dilated convolution input"
                assert same(history, ch), "unproved dilated history"
                assert weight is cw and options == {"dilation": dilation}
                combined = torch.cat((ch.cpu(), cx.cpu().unsqueeze(1)), 1)
                acc = torch.zeros_like(cx.cpu(), dtype=torch.float32)
                for tap in range(weight.shape[-1]):
                    tap_value = combined[:, tap * dilation].float()
                    acc = acc + tap_value * weight[:, 0, tap].cpu().float()
                lo, hi = scaled_silu_bounds(acc.to(cx.dtype), divisor=1.0,
                    activation_dtype=activation_compute_dtype(cx.dtype, cx.device))
                proof = check(actual_conv.cpu(), lo, hi)
                assert proof["outside_enclosure"] == 0, proof
                assert same(after, torch.cat((ch[:, 1:], cx.unsqueeze(1)), 1)), (
                    "dilated history writeback differs"
                )
                activation_proofs.append({"kind": "dilated_silu", **proof})
            except Exception as exc:
                error = "dilated convolution: " + str(exc)
            coordinated(error)
            return actual_conv.reshape_as(x).clone(), after.clone()

        def causal_hc_injection(normalized, weight, *, streams, project):
            nonlocal scaled_used, pending_injection_logits
            logits = project(normalized, weight)
            pending_injection_logits = logits
            # Adjacency fusion removes the standalone gate buffer. Its complete
            # gate/product/add contract is witnessed in causal_hc_combine below.
            if scaled_used < len(scaled_events):
                cx, gates, options = scaled_events[scaled_used]
                if options.get("activation") == "sigmoid":
                    scaled_used += 1
                    error = None
                    try:
                        assert same(logits.reshape_as(cx), cx), "unproved injection logits"
                        assert options == {"divisor": float(streams), "multiplier": 2.0,
                                           "activation": "sigmoid"}
                        lo, hi = scaled_activation_bounds(cx.cpu(), **options,
                            activation_dtype=activation_compute_dtype(cx.dtype, cx.device))
                        proof = check(gates.cpu(), lo, hi)
                        assert proof["outside_enclosure"] == 0, proof
                        activation_proofs.append({"kind": "scaled_sigmoid", **proof})
                    except Exception as exc:
                        error = "scaled injection: " + str(exc)
                    coordinated(error)
                    pending_injection_logits = None
                    return gates.reshape_as(logits).clone()
            return reference_module.hc.sigmoid_stored(logits / streams) * 2

        original_hc_combine = reference_module.hc.combine

        def causal_hc_combine(residual, output, gates):
            nonlocal combine_used, pending_injection_logits
            if pending_injection_logits is None:
                return original_hc_combine(residual, output, gates)
            cr, cu, cg, actual_combine, streams = combine_events[combine_used]
            combine_used += 1
            error = None
            try:
                assert same(residual.reshape_as(cr), cr), "unproved injection residual"
                assert same(output.reshape_as(cu), cu), "unproved injection update"
                assert same(pending_injection_logits.reshape_as(cg), cg), (
                    "unproved fused gate logits"
                )
                assert cg.shape[-1] == streams
                lower, upper = stream_combine_bounds(
                    cr.cpu(), cu.cpu(), cg.cpu(),
                    activation_dtype=activation_compute_dtype(cg.dtype, cg.device),
                )
                proof = check(actual_combine.cpu(), lower, upper)
                assert proof["outside_enclosure"] == 0, proof
                activation_proofs.append({"kind": "fused_stream_combine", **proof})
            except Exception as exc:
                error = "fused stream combine: " + str(exc)
            coordinated(error)
            pending_injection_logits = None
            return actual_combine.reshape_as(residual).clone()

        def causal_hc_mix(x, norm, down, up, *, streams, eps, project):
            nonlocal scaled_used, mix_used
            normalized = causal_norm(x, norm, group_size=x.shape[-1] // streams, eps=eps)
            low = project(normalized, down)
            cx, activated, options = scaled_events[scaled_used]
            scaled_used += 1
            error = None
            try:
                assert same(low.reshape_as(cx), cx), "unproved scaled activation input"
                assert options == {"divisor": float(streams)}
                lo, hi = scaled_silu_bounds(cx.cpu(), divisor=streams,
                    activation_dtype=activation_compute_dtype(cx.dtype, cx.device))
                proof = check(activated.cpu(), lo, hi)
                assert proof["outside_enclosure"] == 0, proof
                activation_proofs.append({"kind": "scaled_silu", **proof})
            except Exception as exc:
                error = "scaled activation: " + str(exc)
            coordinated(error)
            logits = project(activated.reshape_as(low).clone(), up)
            cn, cg, actual_mix = mix_events[mix_used]
            mix_used += 1
            error = None
            try:
                assert same(normalized.reshape_as(cn), cn), "unproved stream norm input"
                assert same(logits.reshape_as(cg), cg), "unproved stream gate input"
                act = activation_compute_dtype(cg.dtype, cg.device)
                gate = cg.cpu().double()
                if act is torch.float32:
                    gl, gh = fp32_sigmoid_bounds(gate)
                else:
                    gl, gh = sigmoid_bounds(gate, dtype=act)
                gl, gh = gl.to(cg.dtype).double(), gh.to(cg.dtype).double()
                nl = cn.cpu().double()
                pl, ph = product(nl, nl, gl, gh)
                pl, ph = pl.float().to(cn.dtype).double(), ph.float().to(cn.dtype).double()
                width = actual_mix.shape[-1]
                pl, ph = pl.reshape(-1, streams, width), ph.reshape(-1, streams, width)
                # Match ordered FP32 addition, without borrowing candidate intermediates.
                sl, sh = torch.zeros_like(pl[:, 0]), torch.zeros_like(ph[:, 0])
                for stream in range(streams):
                    sl = (sl + pl[:, stream]).float().double()
                    sh = (sh + ph[:, stream]).float().double()
                lower = (sl / streams).float().to(actual_mix.dtype).double()
                upper = (sh / streams).float().to(actual_mix.dtype).double()
                proof = check(actual_mix.cpu(), lower, upper)
                assert proof["outside_enclosure"] == 0, proof
                activation_proofs.append({"kind": "stream_mix", **proof})
            except Exception as exc:
                error = "stream mix: " + str(exc)
            coordinated(error)
            return actual_mix.reshape(*x.shape[:-1], actual_mix.shape[-1]).clone(), normalized

        class CausalOperators(QuantizedReferenceOperators):
            def __init__(self):
                super().__init__(compute_mode=policy.compute_mode)
                self.used = {}
                self.expert_errors = []
                self.events = {}
                self.activation_operands = {}
                for (weight, x, y), scales in zip(records, linear_scales, strict=True):
                    self.events.setdefault(id(weight), []).append((x, y, scales))

            def linear(self, x, weight):
                nonlocal current_shared_gate
                index = self.used.get(id(weight), 0)
                cx, y, scales = self.events[id(weight)][index]
                error = None
                try:
                    expected = x.reshape_as(cx)
                    if scales is None:
                        name = names[id(weight)]
                        if name.endswith(".down_proj.weight"):
                            prefix = name.removesuffix(".down_proj.weight")
                            operands = self.activation_operands.pop(prefix, {})
                            if set(operands) == {"gate", "up"}:
                                gate, up = operands["gate"], operands["up"]
                                eager_middle = reference_module.silu_stored(gate) * up
                                assert same(expected, eager_middle.reshape_as(cx)), (
                                    "unproved SwiGLU producing operands"
                                )
                                act = activation_compute_dtype(gate.dtype, gate.device)
                                activation_proofs.append(
                                    swiglu_witnesses.add(
                                        gate.reshape_as(cx), up.reshape_as(cx), cx,
                                        activation_dtype=act, consumer=name,
                                    )
                                )
                                expected = cx
                        if not same(expected, cx):
                            raise AssertionError(
                                {
                                    "producer": names[id(weight)],
                                    "different": int((expected != cx).sum()),
                                    "max_abs": float((expected.float() - cx.float()).abs().max()),
                                }
                            )
                    else:
                        import triton

                        from cubic_serve.quantization.common.int8 import quantize_rows

                        expected = expected.contiguous()
                        codes = torch.empty_like(cx)
                        token_scales = torch.empty_like(scales)
                        quantize_rows[(cx.shape[0],)](
                            expected, codes, token_scales,
                            cx.shape[1], triton.next_power_of_2(cx.shape[1]),
                            num_warps=4, enable_fp_fusion=False,
                        )
                        if not same(codes, cx) or not same(token_scales, scales):
                            raise AssertionError({
                                "producer": names[id(weight)],
                                "int8_codes_different": int((codes != cx).sum()),
                                "int8_scales_different": int((token_scales != scales).sum()),
                            })
                    proof = (check_linear_enclosure if reference_math_memo is None
                             else reference_math_memo.enclosure)(
                        cx, weight, y, int8_token_scales=scales
                    )
                    proofs.append({"name": names[id(weight)], **proof})
                except Exception as exc:
                    error = names[id(weight)] + ": " + str(exc)
                if topology.ep_size > 1 and ".experts." in names[id(weight)]:
                    # Owners execute different expert counts. Coordinate only at
                    # the common owner-reduction boundary, never per local GEMM.
                    if error is not None:
                        self.expert_errors.append(error)
                else:
                    coordinated(error)
                self.used[id(weight)] = index + 1
                out = y.clone().reshape(*x.shape[:-1], y.shape[-1])
                for role in ("gate", "up"):
                    suffix = f".{role}_proj.weight"
                    if names[id(weight)].endswith(suffix):
                        prefix = names[id(weight)].removesuffix(suffix)
                        self.activation_operands.setdefault(prefix, {})[role] = out.clone()
                if in_local_mlp and names[id(weight)].endswith(".shared_expert_gate.weight"):
                    current_shared_gate = out.clone()
                if in_local_mlp and (
                    id(weight) in row_weights
                    or (".experts." in names[id(weight)]
                        and names[id(weight)].endswith(".down_proj.weight"))
                ):
                    local_mlp_terms.append((names[id(weight)], out.clone()))
                    return out
                if id(weight) in row_weights:
                    completion = etp_group if ".experts." in names[id(weight)] else group
                    out = fixed_sum(out, completion)
                else:
                    out = complete_index_projection(out, names[id(weight)])
                if in_completed_mlp and names[id(weight)].endswith(
                    (".down_proj.weight", ".shared_expert_gate.weight")
                ):
                    completed_mlp_terms.append((names[id(weight)], out.clone()))
                return out

            def reduce_experts(self, value):
                coordinated(self.expert_errors or None)
                self.expert_errors.clear()
                if in_local_mlp:
                    return value
                return fixed_sum(value, ep_group)

            def embedding(self, ids, weight, *, dtype):
                start = lane * t.vocabulary_size // tp
                stop = (lane + 1) * t.vocabulary_size // tp
                mask = (ids >= start) & (ids < stop)
                local = (ids - start).clamp(0, weight.shape[0] - 1)
                value = super().embedding(local, weight, dtype=dtype).to(dtype)
                return fixed_sum(torch.where(mask[..., None], value, 0))

            def embedding_shards(self, ids, base, weights, *, parts, dtype):
                # Infer the lexical row width from the checkpoint's declared shard geometry.
                logical_by_name = {item.name: item for item in logical}
                width = logical_by_name[base + ".shard_0.weight"].shape[1]
                values = torch.zeros((ids.numel(), width), device=w.device, dtype=dtype)
                owned = {b.tensor_name: b for b in subset}
                offset = 0
                flat = ids.flatten().cpu().tolist()
                scale = (
                    weights[base + ".weight_scale"].float()
                    if t.ple_embedding_dtype == "float8_e4m3fn" else 1.0
                )
                for part in range(parts):
                    name = f"{base}.shard_{part}.weight"
                    shape = logical_by_name[name].shape
                    if name in owned:
                        begin = offset + owned[name].spans[0].start
                        table = weights[name]
                        for row, token in enumerate(flat):
                            if begin <= token < begin + table.shape[0]:
                                value = super().embedding(
                                    torch.tensor([token - begin], device=w.device),
                                    table,
                                    dtype=torch.float32,
                                )
                                values[row].copy_((value.float() * scale).to(dtype).reshape(-1))
                    offset += shape[0]
                return fixed_sum(values).reshape(*ids.shape, width)

        local_config = replace(
            config,
            text=replace(
                t,
                gated_delta_net=replace(
                    t.gated_delta_net,
                    key_heads=t.gated_delta_net.key_heads // tp,
                    value_heads=t.gated_delta_net.value_heads // tp,
                ),
                full_attention=replace(
                    t.full_attention,
                    query_heads=t.full_attention.query_heads // tp,
                    key_value_heads=max(1, t.full_attention.key_value_heads // tp),
                ),
            ),
        )

        trajectory_difference = _TrajectoryDifferences()

        independent_local_mlp = False

        def independent_mlp(model, hidden, prefix):
            nonlocal independent_local_mlp
            if prefix not in local_prefixes:
                return original_mlp(model, hidden, prefix)
            independent_local_mlp = True
            try:
                partial = original_mlp(model, hidden, prefix)
            finally:
                independent_local_mlp = False
            return fixed_sum(partial)

        class IndependentOperators(CausalOperators):
            """Reuse TP embeddings, but never candidate linear outputs or state.

            This runs outside every causal substitution patch. Immutable event
            snapshots are compared only after the whole independent forward.
            Different routes are reported as unmatched events, not FP failures.
            """

            def __init__(self):
                super().__init__()
                self.boundaries = []

            def linear(self, x, weight):
                out = (QuantizedReferenceOperators.linear(self, x, weight)
                       if reference_math_memo is None else
                       reference_math_memo.linear(self, x, weight))
                index = self.used.get(id(weight), 0)
                self.boundaries.append(
                    (id(weight), index, x.detach().clone(), out.detach().clone())
                )
                self.used[id(weight)] = index + 1
                if independent_local_mlp and (
                    id(weight) in row_weights
                    or (".experts." in names[id(weight)]
                        and names[id(weight)].endswith(".down_proj.weight"))
                ):
                    # The original-weight free trajectory uses the requested
                    # local weighted fold before the TP completion. No native
                    # output or candidate route is substituted here.
                    return out
                if id(weight) in row_weights:
                    completion = etp_group if ".experts." in names[id(weight)] else group
                    return fixed_sum(out, completion)
                return complete_index_projection(out, names[id(weight)])

        base_causal_norm = causal_norm

        def complete_group_norm(x, weight, **options):
            return base_causal_norm(x, weight, **options)

        def reference_selection(q, keys, visible, *, compression, token_budget):
            nonlocal selector_used
            index = selector_used
            selector_used += 1
            (ratio, budget, cq, ck, table, lengths,
             active, chosen, counts, order) = completed_selectors[index]
            compressed_expected.append(keys.clone())
            error = None
            proof = None
            try:
                assert ratio == compression and budget == token_budget
                assert same(q.reshape_as(cq), cq), "unproved index query"
                assert same(visible.reshape_as(lengths).to(lengths.dtype), lengths), (
                    "unproved visible length"
                )
                length = int(visible[0, 0])
                groups = length // ratio
                keep = budget // ratio
                ids = torch.arange(groups, device=w.device)
                cached = ck[table[0, ids // ck.shape[1]].long(), ids % ck.shape[1]][None]
                assert same(keys, cached), "unproved compressed keys"
                assert bool((active == 1).all())
                if groups > keep:
                    # The no-ranking path synthesizes every visible block ID;
                    # score scratch is not a producing boundary on that path.
                    encoded = order[0, :groups].cpu()
                    assert same(
                        0xFFFFFFFF - (encoded & 0xFFFFFFFF),
                        torch.arange(groups, dtype=torch.int64),
                    ), "encoded original block IDs changed"
                    scores = (encoded >> 32).to(torch.int32).contiguous().view(torch.float32)
                    assert bool(torch.isfinite(scores).all()) and bool((scores >= 0).all())
                    products = cq[0].cpu().double()[:, None, :] * keys[0].cpu().double()[None, :, :]
                    neg = torch.full_like(products, float("-inf"), dtype=torch.float32)
                    pos = torch.full_like(neg, float("inf"))
                    pl = torch.nextafter(products.float(), neg).double()
                    ph = torch.nextafter(products.float(), pos).double()
                    u = 2**-24
                    width = products.shape[-1]
                    g = (width - 1) * u / (1 - (width - 1) * u)
                    magnitude = torch.maximum(pl.abs(), ph.abs()).sum(-1)
                    dl = (pl.sum(-1) - g * magnitude).clamp_min(0)
                    dh = (ph.sum(-1) + g * magnitude).clamp_min(0)
                    heads = cq.shape[1]
                    gh = heads * u / (1 - heads * u)
                    upper_sum = dh.sum(0)
                    lower = dl.sum(0) - gh * upper_sum
                    upper = upper_sum * (1 + gh)
                    scale = float(torch.tensor(width**0.5, dtype=torch.float32))
                    lower = torch.nextafter(
                        (lower / scale).float(), torch.full_like(scores, float("-inf"))
                    ).double()
                    upper = torch.nextafter(
                        (upper / scale).float(), torch.full_like(scores, float("inf"))
                    ).double()
                    proof = check(scores, lower, upper)
                    assert proof["outside_enclosure"] == 0, proof
                    corrupted = scores.clone()
                    corrupted[0] = 2 * upper[0].abs() + 1
                    assert check(corrupted, lower, upper)["outside_enclosure"] > 0, (
                        "score enclosure failed to reject injected corruption"
                    )
                    blocks = sorted(
                        sorted(range(groups), key=lambda i: (-float(scores[i]), i))[:keep]
                    )
                else:
                    blocks = list(range(groups))
                expected_ids = [
                    block * ratio + offset for block in blocks for offset in range(ratio)
                ] + list(range(groups * ratio, length))
                expected_selection = torch.full_like(chosen, -1)
                expected_selection[0, : len(expected_ids)] = torch.tensor(
                    expected_ids, device=w.device, dtype=chosen.dtype
                )
                assert int(counts[0]) == len(expected_ids) and torch.equal(
                    chosen, expected_selection
                ), "selection does not follow independently proven scores and stable block order"
                selection_proofs.append(
                    {
                        "groups": groups,
                        "keep": keep,
                        "score_enclosure": proof,
                        "selection_exact": True,
                    }
                )
            except Exception as exc:
                error = f"selection{index}: " + str(exc)
            coordinated(error)
            return chosen[:, None, :].clone()

        tokens = token_ids or tuple(
            (17 + ordinal * 7919) % t.vocabulary_size for ordinal in range(horizon)
        )
        def reference_forward(model, token, state, feature, axes, feedback):
            ids = torch.tensor([[token]], device=w.device, dtype=torch.int64)
            if draft_provider is None:
                if completed_transition is not None:
                    return model.forward_with_feedback(ids, state, input_embeddings=feature,
                                                       position_ids=axes)
                logits, state = model.forward(ids, state, input_embeddings=feature,
                                              position_ids=axes)
                return logits, state, None
            step = draft_provider.build_reference(model).forward(
                ids, feedback, state, input_embeddings=feature, position_ids=axes)
            return step.logits, step.state, step.feedback

        if operator_observers is not None:
            operator_observers.start(stage, capture_payload)
        for ordinal, token in enumerate(tokens):
            position = start_position + ordinal
            conv_used = 0
            scaled_used = mix_used = combine_used = 0
            pending_injection_logits = None
            moe_combine_used = signed_used = dilated_used = 0
            activation_proofs = []
            swiglu_witnesses = SwiGLUWitnessBatch()
            conv_proofs = []
            compressed_expected = []
            selector_used = 0
            selection_proofs = []
            l2_proofs = []
            gdn_output_proofs = []
            gdn_state_proofs = []
            attention_used = 0
            pending_gate = None
            attention_proofs = []
            gate_proofs = []
            records = list(base_records)
            linear_scales = list(base_linear_scales)
            b = stage.buffers
            b.token_ids.fill_(token)
            b.position_ids.fill_(position)
            axes = (torch.tensor(position_axes[ordinal], device=w.device, dtype=torch.int64)
                    .reshape(3, 1, 1) if position_axes else None)
            if axes is not None:
                b.position_ids.copy_(axes.reshape(3, 1))
            feature = features[ordinal] if features else None
            stage.feature_rows.zero_()
            stage.embedding_mask.fill_(feature is None)
            if feature is not None:
                stage.feature_rows.copy_(feature.reshape(1, t.hidden_size))
            feedback = None if draft_provider is None else feedback_rows[ordinal]
            if feedback_recurrence and ordinal:
                feedback = previous_feedback
            if feedback is not None:
                b.target_hidden.copy_(feedback[:, 0])
            b.sequence_lengths.fill_(position + 1)
            b.token_offsets.fill_(position % block)
            b.page_source.fill_(position // block)
            b.page_destination.fill_(position // block)
            b.page_table[0].copy_(
                torch.arange(b.page_table.shape[1], device=w.device, dtype=b.page_table.dtype)
            )
            page_count = (position + block) // block
            if operator_observers is not None:
                operator_observers.before_step(stage, position)
            torch.cuda.synchronize()
            dist.barrier(group=group)
            stage.replay()
            stage.synchronize()
            dist.barrier(group=group)
            actual = stage.logits.detach().clone()
            if draft_provider is not None or completed_transition is not None:
                previous_feedback = stage.draft_feedback.detach().clone().reshape(
                    1, 1, -1)
            actual_states = {
                name: value[0].detach().clone() for name, value in stage.arena.states.items()
            }
            if cache_storage.quantized:
                for event, cache in zip(attention_events, cache_events, strict=True):
                    assert cache is not None
                    layer = event[0]
                    window = cache[1]
                    # Captured clone destinations are static replay buffers.
                    # Retain a durable copy now, after completion, before the
                    # next replay overwrites that observer destination.
                    key = window.key if window.producer_key is None else window.producer_key
                    value = window.value if window.producer_value is None else window.producer_value
                    native_key_history.setdefault(layer, []).append(key.detach().clone())
                    native_value_history.setdefault(layer, []).append(value.detach().clone())
                def native_history(history):
                    result = torch.zeros(
                        (len(stage.arena.layout.layer_ids), page_count * block,
                         stage.arena.layout.num_kv_heads, stage.arena.layout.head_dim),
                        device=w.device, dtype=stage.arena.layout.dtype,
                    )
                    for layer, values in history.items():
                        result[layer, :position + 1] = torch.cat(values, dim=0)
                    return result
                actual_key = native_history(native_key_history)
                actual_value = native_history(native_value_history)
            else:
                actual_key = (
                    stage.arena.key[:page_count]
                    .transpose(0, 1)
                    .reshape(
                        len(stage.arena.layout.layer_ids),
                        page_count * block,
                        *stage.arena.key.shape[-2:],
                    )
                    .detach()
                    .clone()
                )
                actual_value = (
                    stage.arena.value[:page_count]
                    .transpose(0, 1)
                    .reshape_as(actual_key)
                    .detach()
                    .clone()
                )
            actual_pages = {
                name: value[:page_count].reshape(-1, value.shape[-1]).detach().clone()
                for name, value in stage.arena.page_fields.items()
            }
            actual_written = (
                stage.arena.written[:page_count]
                .transpose(0, 1)
                .reshape(len(stage.arena.layout.layer_ids), page_count * block)
                .clone()
            )
            recorder.experts = list(base_experts)
            if topology.expert_tensor_size > 1:
                early_partials = iter(pre_routes)
                for event_index, event in enumerate(base_experts):
                    # EP records checkpoint-ID maps; pure TP records ordered tuples.
                    gate_weights = event[0][0]
                    first_gate = (next(iter(gate_weights.values()))
                                  if isinstance(gate_weights, dict) else gate_weights[0])
                    prefix = names[id(first_gate)].split(".experts.", 1)[0]
                    if prefix not in local_prefixes:
                        recorder.experts[event_index] = (*event[:-1], next(early_partials))
                if next(early_partials, None) is not None:
                    raise ValueError("unmatched early TP route witness")
            schedule = list(ordered_events)
            schedule += [
                (names[id(weight)], tuple(x.shape),
                 (*y.shape[:-1], (t.qsa.query_heads + 1) * t.qsa.head_dim)
                 if names[id(weight)].endswith(".indexer.index_qk_proj.weight") else tuple(y.shape))
                for weight, x, y in records
            ]
            schedule += [
                ("norm", names[id(weight)],
                 (x.shape[0], t.qsa.query_heads, t.qsa.head_dim)
                 if i in index_query_norms else tuple(x.shape))
                for i, (x, weight, y, epsilon, group) in enumerate(normalizations)
            ]
            schedule += [
                ("attention", event[0], tuple(event[2].shape), event[-1])
                for event in attention_events
            ]
            if cache_storage.quantized:
                schedule += [("cache_encoding", event[0],
                              cache[0].key.spec.identity, cache[0].value.spec.identity,
                              tuple(cache[1].key.shape), tuple(cache[1].value.shape))
                             for event, cache in zip(attention_events, cache_events, strict=True)]
            schedule += [("gate", tuple(event[0].shape)) for event in gate_events]
            schedule += [
                ("gdn_norm", names[id(event[0])], tuple(event[1].shape))
                for event in gdn_norm_events
            ]
            schedule += [
                ("gdn_core", names[id(event[0])], tuple(event[1].shape)) for event in gdn_events
            ]
            schedule += [
                ("selector", event[0], event[1], tuple(event[2].shape)) for event in selector_events
            ]
            schedule += [
                ("routed", tuple(event[2].flatten().cpu().tolist())) for event in recorder.experts
            ]
            schedules = [None] * tp
            dist.all_gather_object(schedules, schedule, group=group)
            assert all(other == schedule for other in schedules)
            recorder.freeze()
            for events in recorder.events.values():
                records.extend(events)
                linear_scales.extend([None] * len(events))

            # Captured clone destinations are replay-owned. Never replace their
            # references with one-step reconstruction tensors: subsequent graph
            # replays update the original destinations, not the reconstructed copy.
            completed_normalizations = list(normalizations)
            for i in sorted(index_query_norms):
                x, weight, y, epsilon, norm_group = normalizations[i]
                completed_normalizations[i] = (
                    gather_index_queries(x), weight, gather_index_queries(y), epsilon, norm_group,
                )
            for i, offsets in index_key_norms.items():
                x, weight, y, epsilon, norm_group = normalizations[i]
                completed_normalizations[i] = (
                    x, weight, gather_owned_normalized_rows(
                        y, offsets // index_plan.compression, 1, tp, group), epsilon, norm_group,
                )
            completed_selectors = list(selector_events)
            if tp > 1:
                for i, event in enumerate(selector_events):
                    ratio, budget, cq, ck, table, lengths, active, chosen, counts, order = event
                    fields = []
                    for local in (ck, order):
                        pieces = [torch.empty_like(local) for _ in range(tp)]
                        dist.all_gather(pieces, local, group=group)
                        fields.append(torch.stack(pieces, dim=2).flatten(1, 2))
                    completed_selectors[i] = (
                        ratio, budget, cq, fields[0], table, lengths,
                        active, chosen, counts, fields[1],
                    )
            norm_events = {
                id(weight): (x, weight, y, epsilon, group)
                for x, weight, y, epsilon, group in completed_normalizations
            }
            norm_used = set()
            proofs = []
            route_used = 0
            route_proofs = []
            moe_completion_proofs = []
            operators = CausalOperators()
            reference = Qwen4ExpReferenceModel(
                local_config, w, operators, recurrent_state_dtype=stage.recurrent_state_dtype,
                expert_range=expert_range,
                reduce_experts=operators.reduce_experts if topology.ep_size > 1 else None,
            )
            with ExitStack() as causal_context:
                for context in (
                    operator_strategy_scope(policy, device="", model=""),
                    ReferenceExecutionContext(),
                    torch.inference_mode(),
                    patch.object(reference_module.hc, "grouped_rms_norm", causal_norm),
                    patch.object(reference_module.hc, "mix", causal_hc_mix),
                    patch.object(reference_module.hc, "injection", causal_hc_injection),
                    patch.object(reference_module.hc, "combine", causal_hc_combine),
                    patch.object(reference_module.hc, "signed_root_gate", causal_signed),
                    patch.object(reference_module, "dilated_depthwise_conv", causal_dilated),
                    patch.object(reference_module, "topk_routes", causal_routes),
                    patch.object(Qwen4ExpReferenceModel, "_mlp", causal_mlp),
                    patch.object(reference_module, "zero_centered_rms_norm", complete_group_norm),
                    patch.object(reference_module, "select_blocks", reference_selection),
                    patch.object(reference_module, "depthwise_causal_conv1d", causal_conv),
                    patch.object(reference_module, "gated_delta_net_recurrent", diagnose_gdn),
                    patch.object(reference_module, "gated_rms_norm", diagnose_gdn_norm),
                    patch.object(reference_module, "attend_selected", causal_attention),
                    patch.object(reference_module, "sigmoid_output_gate", causal_sigmoid),
                ):
                    causal_context.enter_context(context)
                expected, reference_state, expected_feedback = reference_forward(
                    reference, token, reference_state, feature, axes, feedback)
            error = None
            try:
                swiglu_witnesses.finish()
            except Exception as exc:
                error = str(exc)
            coordinated(error)
            coordinated(
                None
                if same(actual, expected.reshape_as(actual))
                else f"position{position}: final logits differ"
            )
            if expected_feedback is not None:
                coordinated(None if same(stage.draft_feedback,
                                         expected_feedback.reshape_as(stage.draft_feedback))
                            else "draft feedback producer differs")
            assert all(
                operators.used.get(key, 0) == len(events)
                for key, events in operators.events.items()
            )
            assert selector_used == len(selector_events) and attention_used == len(attention_events)
            assert conv_used == len(conv_events)
            assert signed_used == len(signed_events) and dilated_used == len(dilated_events)
            assert scaled_used == len(scaled_events) and mix_used == len(mix_events)
            assert combine_used == len(combine_events) and pending_injection_logits is None
            assert route_used == len(route_events), "route probability producer coverage"
            state_differences = {}

            def verify(name, left, right, differences=state_differences):
                right = right.reshape_as(left)
                if not same(left, right):
                    differences[name] = {
                        "different": int((left != right).sum()),
                        "max_abs": float((left.double() - right.double()).abs().max()),
                    }

            for layer, previous in enumerate(reference_state.layers):
                prefix = f"model.language_model.layers.{layer}.state."
                if previous.convolution is not None:
                    verify(
                        prefix + "gdn-convolution",
                        actual_states[prefix + "gdn-convolution"],
                        previous.convolution,
                    )
                    verify(
                        prefix + "gdn-recurrent-matrix",
                        actual_states[prefix + "gdn-recurrent-matrix"],
                        previous.recurrent,
                    )
                if previous.ple_history is not None:
                    verify(
                        prefix + "ple-ngram-token-context",
                        actual_states[prefix + "ple-ngram-token-context"],
                        previous.ple_history,
                    )
                    verify(
                        prefix + "ple-convolution",
                        actual_states[prefix + "ple-convolution"],
                        previous.ple_convolution,
                    )
                if previous.key is not None:
                    index = stage.arena.layout.layer_ids.index(layer)
                    verify(
                        prefix + "key",
                        actual_key[index, : position + 1],
                        previous.key[0].transpose(0, 1),
                    )
                    verify(
                        prefix + "value",
                        actual_value[index, : position + 1],
                        previous.value[0].transpose(0, 1),
                    )
                    if cache_storage.quantized:
                        from cubic_serve.runtime.cache.quantization import CacheGroupAxis
                        for side, native, spec in (
                            ("key", actual_key[index, :position + 1], cache_storage.key),
                            ("value", actual_value[index, :position + 1], cache_storage.value),
                        ):
                            if spec.dtype.quantized and spec.axis is CacheGroupAxis.TOKENS:
                                actual_tail = actual_states[f"kv_{side}_tail"][index]
                                expected_tail = torch.zeros_like(actual_tail)
                                count = (position + 1) % spec.group_size
                                if count:
                                    tail = native[-count:]
                                    if cache_storage.enable_rotation:
                                        from cubic_serve.ops.eager.rotation import (
                                            rotate_cache_reference,
                                        )

                                        with ReferenceExecutionContext():
                                            tail = rotate_cache_reference(tail)
                                    expected_tail[:count] = tail
                                verify(prefix + side + "-native-tail", actual_tail, expected_tail)
                    wanted = torch.zeros_like(actual_states[prefix + "qsa-raw-key-ring"])
                    count = min(position + 1, wanted.shape[0])
                    wanted[-count:].copy_(
                        previous.index_keys[
                            :, -count:, lane * index_plan.key_width:
                            (lane + 1) * index_plan.key_width
                        ].reshape_as(wanted[-count:])
                    )
                    verify(
                        prefix + "qsa-raw-key-ring",
                        actual_states[prefix + "qsa-raw-key-ring"],
                        wanted,
                    )
                    positions = torch.zeros_like(actual_states[prefix + "qsa-position-ring"])
                    positions[-count:] = (
                        torch.arange(position + 1 - count, position + 1, device=w.device)[:, None]
                        if previous.index_positions is None
                        else previous.index_positions[:, 0, -count:].transpose(0, 1)
                    )
                    verify(
                        prefix + "qsa-position-ring",
                        actual_states[prefix + "qsa-position-ring"],
                        positions,
                    )
                    field = prefix + "qsa-compressed-key-cache"
                    compressed = torch.zeros_like(actual_pages[field])
                    assert same(previous.compressed_index_keys, compressed_expected[index])
                    owned_keys = previous.compressed_index_keys[:, lane::tp]
                    groups = owned_keys.shape[1]
                    compressed[:groups] = owned_keys.reshape(
                        groups, compressed.shape[-1]
                    )
                    verify(field, actual_pages[field], compressed)
            wanted_written = torch.zeros_like(actual_written)
            wanted_written[:, : position + 1] = True
            verify("written", actual_written, wanted_written)
            coordinated(None if not state_differences else str(state_differences))
            independent_diagnostic = None
            if diagnose_independent_trajectory:
                independent_operators = IndependentOperators()
                if cache_storage.quantized:
                    from cubic_serve.ops.eager.kv_cache import ReferenceCacheOperators
                    independent_operators = ReferenceCacheOperators(
                        independent_operators, cache_storage)
                independent_model = Qwen4ExpReferenceModel(
                    local_config, w, independent_operators,
                    recurrent_state_dtype=stage.recurrent_state_dtype,
                    expert_range=expert_range,
                    reduce_experts=(
                        independent_operators.reduce_experts if topology.ep_size > 1 else None
                    ),
                    )
                # No causal operator substitutions here. Both trajectories see
                # the same tokens, but maintain separate recursively rounded state.
                with (
                    operator_strategy_scope(policy, device="", model=""),
                    ReferenceExecutionContext(),
                    torch.inference_mode(),
                    patch.object(Qwen4ExpReferenceModel, "_mlp", independent_mlp),
                ):
                    independent_logits, independent_state, independent_feedback = reference_forward(
                        independent_model, token, independent_state, feature, axes,
                        (independent_feedback_rows[ordinal] if independent_feedback_rows else
                         independent_feedback if feedback_recurrence and ordinal else feedback))
                # Reference observers obey the same deferred-inspection rule.
                # TP ranks own different heads but share the ordered linear schedule.
                reference_schedule = [
                    (names[key], occurrence, tuple(x.shape), tuple(y.shape))
                    for key, occurrence, x, y in independent_operators.boundaries
                ]
                schedules = [None] * tp
                if topology.ep_size == 1:
                    dist.all_gather_object(schedules, reference_schedule, group=group)
                    if not all(schedule == reference_schedule for schedule in schedules):
                        raise AssertionError("independent reference rank event schedules differ")
                else:
                    # The oracle has completed. Inspect its own producing router
                    # values now, outside all model/operator observer callbacks.
                    # First prove the structural rank schedule before reference
                    # math. Route ownership is proved separately from its values.
                    dist.all_gather_object(schedules, (reference_schedule, None), group=group)
                    error = _expert_reference_schedule_error(
                        schedules, topology, t.moe.expert_count,
                    )
                    if error is not None:
                        raise AssertionError(error)
                    reference_routes = {}
                    for key, _, _, value in independent_operators.boundaries:
                        if names[key].endswith(".mlp.gate.weight"):
                            with ReferenceExecutionContext():
                                ids, _ = reference_module.topk_routes(
                                    value.reshape(-1, value.shape[-1]),
                                    top_k=t.moe.experts_per_token,
                                    normalize=t.moe.normalize_topk_probability,
                                )
                            reference_routes[names[key].removesuffix(".gate.weight")] = (
                                tuple(tuple(row) for row in ids.cpu().tolist())
                            )
                    dist.all_gather_object(
                        schedules, (reference_schedule, reference_routes), group=group,
                    )
                    error = _expert_reference_schedule_error(
                        schedules, topology, t.moe.expert_count,
                    )
                    if error is not None:
                        raise AssertionError(error)
                independent_boundaries = []
                route_decisions = []
                for key, occurrence, x, y in independent_operators.boundaries:
                    boundary = {"name": names[key], "occurrence": occurrence}
                    events = independent_operators.events.get(key, ())
                    if occurrence < len(events):
                        cx, cy, token_scales = events[occurrence]
                        if names[key].endswith(".mlp.gate.weight"):
                            candidate_logits = cy.float().reshape(-1)
                            reference_logits = y.float().reshape(-1)
                            candidate_order = torch.argsort(
                                candidate_logits, descending=True, stable=True
                            )
                            reference_order = torch.argsort(
                                reference_logits, descending=True, stable=True
                            )
                            selected = t.moe.experts_per_token
                            candidate_ids = candidate_order[:selected].sort().values
                            reference_ids = reference_order[:selected].sort().values
                            if not torch.equal(candidate_ids, reference_ids):
                                route_decisions.append({
                                    "name": names[key],
                                    "candidate_ids": candidate_ids.cpu().tolist(),
                                    "reference_ids": reference_ids.cpu().tolist(),
                                    "candidate_topk_margin": float(
                                        candidate_logits[candidate_order[selected - 1]]
                                        - candidate_logits[candidate_order[selected]]
                                    ),
                                    "reference_topk_margin": float(
                                        reference_logits[reference_order[selected - 1]]
                                        - reference_logits[reference_order[selected]]
                                    ),
                                    "max_logit_delta": float(
                                        (candidate_logits - reference_logits).abs().max()
                                    ),
                                })
                        reference_input = x
                        candidate_input = cx
                        if token_scales is not None:
                            from cubic_serve.quantization.common.reference import wna8_activation
                            candidate_input = cx.float() * token_scales.reshape(-1, 1)
                            reference_input = wna8_activation(x)
                            boundary["input_boundary"] = "dequantized-int8-with-fp32-token-scale"
                            boundary["candidate_storage_dtype"] = str(cx.dtype)
                        boundary["input"] = trajectory_difference(candidate_input, reference_input)
                        boundary["local_output"] = trajectory_difference(cy, y)
                    else:
                        boundary["candidate_event_present"] = False
                    independent_boundaries.append(boundary)
                state_metrics = {}
                for layer, previous in enumerate(independent_state.layers):
                    prefix = f"model.language_model.layers.{layer}.state."
                    for field, value in (
                        ("gdn-convolution", previous.convolution),
                        ("gdn-recurrent-matrix", previous.recurrent),
                        ("ple-ngram-token-context", previous.ple_history),
                        ("ple-convolution", previous.ple_convolution),
                    ):
                        if value is not None:
                            state_metrics[prefix + field] = trajectory_difference(
                                actual_states[prefix + field], value,
                            )
                    if previous.key is not None:
                        index = stage.arena.layout.layer_ids.index(layer)
                        for field, actual_cache, expected_cache in (
                            ("key", actual_key, previous.key),
                            ("value", actual_value, previous.value),
                        ):
                            state_metrics[prefix + field + (
                                "-native-producer" if cache_storage.quantized else "")
                            ] = trajectory_difference(
                                actual_cache[index, : position + 1],
                                expected_cache[0].transpose(0, 1),
                            )
                candidate_only = [
                    {"name": names[key], "count": len(events) -
                     independent_operators.used.get(key, 0)}
                    for key, events in independent_operators.events.items()
                    if len(events) > independent_operators.used.get(key, 0)
                ]
                reference_only = [
                    {"name": names[key], "count": count -
                     len(independent_operators.events.get(key, ()))}
                    for key, count in independent_operators.used.items()
                    if count > len(independent_operators.events.get(key, ()))
                ]

                def expected_route_events(side, decisions):
                    expected = Counter()
                    other = "reference" if side == "candidate" else "candidate"
                    for decision in decisions:
                        prefix = decision["name"].removesuffix(".gate.weight")
                        selected = set(decision[side + "_ids"]) - set(decision[other + "_ids"])
                        for expert in selected:
                            if not expert_range[0] <= expert < expert_range[1]:
                                continue
                            for projection in ("gate_proj", "up_proj", "down_proj"):
                                expected[f"{prefix}.experts.{expert}.{projection}.weight"] += 1
                    return expected

                candidate_events = Counter(
                    {entry["name"]: entry["count"] for entry in candidate_only}
                )
                reference_events = Counter(
                    {entry["name"]: entry["count"] for entry in reference_only}
                )
                route_error = None
                if (
                    candidate_events != expected_route_events("candidate", route_decisions)
                    or reference_events != expected_route_events("reference", route_decisions)
                ):
                    route_error = "independent route branch events do not match top-k decisions"
                elif any(
                    decision["candidate_topk_margin"] > 2 * decision["max_logit_delta"]
                    or decision["reference_topk_margin"] > 2 * decision["max_logit_delta"]
                    for decision in route_decisions
                ):
                    route_error = "route changed beyond its observed logit perturbation"
                coordinated(route_error)
                independent_diagnostic = {
                    "qualification": "pathwise original-weight and independent free trajectory",
                    "reference_compute_mode": independent_operators.compute_mode,
                    "reference_moe_reduction_plan": (
                        "local_stored_weighted_fold_then_fp32_tp_sum"
                        if local_prefixes else "single_rank"
                    ),
                    "reference_recurrent_state_dtype": str(independent_model.recurrent_state_dtype),
                    "logits": trajectory_difference(actual, independent_logits),
                    "draft_feedback": (
                        None if draft_provider is None else
                        trajectory_difference(previous_feedback, independent_feedback)
                    ),
                    "linear_boundaries": independent_boundaries,
                    "states": state_metrics,
                    "changed_route_decisions": route_decisions,
                    "route_branch_schedule_exact": True,
                    "candidate_only_events": candidate_only,
                    "reference_only_events": reference_only,
                }
            trajectory_difference.flush()
            report = {
                "position": position,
                "linear_boundaries": len(proofs),
                "activation_boundary_proofs": activation_proofs,
                "moe_completion_proofs": moe_completion_proofs,
                "route_probability_proofs": route_proofs,
                "norm_boundaries": len(norm_used),
                "attention_boundaries": len(attention_proofs),
                "attention_reduction_proofs": attention_proofs,
                "selection_boundaries": len(selection_proofs),
                "stored_l2_boundaries": len(l2_proofs),
                "gdn_output_reduction_proofs": gdn_output_proofs,
                "gdn_state_rounding_proofs": gdn_state_proofs,
                "state_raw_eager_exact": not gdn_state_proofs,
                "convolution_boundaries": len(conv_proofs),
                "convolution_eager_differences": sum(
                    p["raw_eager_differences"] for p in conv_proofs
                ),
                "candidate_boundary_continuation_state_and_logits_exact": True,
            }
            if independent_diagnostic is not None:
                report["independent_trajectory_diagnostic"] = independent_diagnostic
            from cubic_serve.guardrail.compare import tensor_digest
            from cubic_serve.runtime.text_qualification import boundary_snapshot

            boundaries = boundary_snapshot(stage, 1)
            report["boundary_digests"] = {
                name: tensor_digest(value) for name, value in boundaries.items()
            }
            if operator_observers is not None:
                operator_observers.completed_step(position, capture_payload, report)
            if completed_step_witness is not None:
                completed_step_witness(position, boundaries, report)
            if completed_transition is not None:
                next_prefix = prefix_store.freeze(prefix_binding, dict(
                    position=position + 1,
                    native=dict(
                        states={name: value[0] for name, value in stage.arena.states.items()},
                        pages={name: value[:page_count]
                               for name, value in stage.arena.page_fields.items()},
                        key=stage.arena.key[:page_count], value=stage.arena.value[:page_count],
                        written=stage.arena.written[:page_count],
                    ),
                    causal=reference_state, independent=independent_state,
                    feedback=previous_feedback, independent_feedback=independent_feedback,
                    native_key_history=native_key_history, native_value_history=native_value_history,
                ))
                completed_transition(dict(
                    position=position, token=token, prefix=next_prefix,
                    boundaries=boundaries, report=report,
                    independent_logits=independent_logits.detach().to("cpu", copy=True),
                    independent_feedback=(None if independent_feedback is None else
                                          independent_feedback.detach().to("cpu", copy=True)),
                ))
            reports.append(report)
            logging.getLogger(__name__).info(
                "operator-local text numerical witness rank=%d position=%d", rank, position
            )
        audit_completed = True
        return {
            "scope": "original-weight causal and free-reference M1 cold/warm",
            "reference_math_memo": (None if reference_math_memo is None else dict(
                hits=reference_math_memo.hits, misses=reference_math_memo.misses,
                tensor_payload_bytes=reference_math_memo.bytes,
                entries=len(reference_math_memo.entries),
            )),
            "numerical_contract": {
                "activation_dtype": policy.activation_dtype,
                "attention_dtype": policy.attention_dtype,
                "softmax_dtype": policy.softmax_dtype,
                "candidate_recurrent_state_dtype": str(stage.recurrent_state_dtype),
                "local_oracle_recurrent_state_dtype": str(stage.recurrent_state_dtype),
                "checkpoint_recurrent_state_dtype": t.gated_delta_net.recurrent_state_dtype,
                "state_storage_matches_checkpoint": stage.recurrent_state_dtype
                is getattr(torch, t.gated_delta_net.recurrent_state_dtype),
                "state_storage_scope": (
                    "the selected BF16 or FP32 runtime state is rounded before current output "
                    "and durable writeback in both audited trajectories"
                ),
            },
            "independent_trajectory": diagnose_independent_trajectory,
            "draft_method": None if draft_provider is None else draft_provider.spec.method,
            "feedback_recurrence": feedback_recurrence,
            "reference_column_completion_names": column_completion_names,
            "feature_rows_witnessed": sum(bool(row) for row in input_feature_rows),
            "explicit_position_axes": bool(position_axes),
            "independent_trajectory_diagnostic_recorded": diagnose_independent_trajectory,
            "cross_step_error_propagation_qualified": (
                diagnose_independent_trajectory and len(reports) > 1
                and all(
                    step["candidate_boundary_continuation_state_and_logits_exact"]
                    and step["independent_trajectory_diagnostic"]["route_branch_schedule_exact"]
                    for step in reports
                )
            ),
            "propagation_contract": (
                "inductive original-weight boundary enclosures, exact durable state "
                "continuation, and an independently advancing reference; route differences "
                "must follow each trajectory's audited logits and complete expert event set"
                if diagnose_independent_trajectory else None
            ),
            "steps": reports,
        }
    finally:
        try:
            if operator_observers is not None:
                operator_observers.finish()
        finally:
            if capture_retained:
                if audit_completed:
                    capture_pool.release(capture_binding)
                else:
                    capture_pool.discard(capture_binding)
            else:
                stage.close()
                # Retire observer closures outside any later CUDA capture.
                del stage.launch
