"""Graph-only TP2 x PP2 continuous serving for dense Qwen3.5.

Request ownership is independent of scheduler rows and physical graph rows.
The dispatcher selects a permanent audited C1 graph or the smallest qualified
multi-M graph bucket that can cover the current runnable set. Padding has no
request identity and writes only shared-arena private scratch. Startup alone
may launch the same bodies eagerly to qualify graph equality and padding
isolation. There is no execution fallback, dynamic exact-M capture, or implicit
compilation.
"""

from __future__ import annotations

from typing_extensions import Self

import gc
import faulthandler
import hashlib
import heapq
import inspect
import json
import logging
import math
import os
import pickle
import queue
import signal
import struct
import sys
import tempfile
import threading
import time
import traceback
from bisect import bisect_left
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass, field, replace
from datetime import datetime
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
from torch import Tensor
from torch.cuda import Stream
from torch.distributed import ProcessGroup

from cubic_serve.ops.cuda.collective import all_reduce, barrier

from cubic_serve.contracts.execution import (
    GroupKind,
    ParallelTopology,
    RankGroup,
    RankReplayResources,
)
from cubic_serve.contracts.model import WeightVariantMeta
from cubic_serve.contracts.request import SamplingMeta
from cubic_serve.executor.decode_overlap import DecodeOverlapPlan, common_overlap_thread_candidates
from cubic_serve.executor.draft import DraftReplayBinding, DraftReplayRow
from cubic_serve.executor.model_runner import (
    Qwen3_5TP2PP2DecodeBody,
    Qwen3_5TP2PP2GraphSlot,
    Qwen3_5TP2PP2WavefrontGraphSlot,
    minimum_kv_page_capacity,
)
from cubic_serve.executor.mtp_runner import (
    Qwen3_5MTPArena,
    Qwen3_5TPMTPBody,
    Qwen3_5TPMTPGraphSlot,
)
from cubic_serve.executor.parallel import SealedPartition, initialize_eager_nccl
from cubic_serve.executor.runner import configure_reference_math
from cubic_serve.executor.serving_body import ServingDecodeBody
from cubic_serve.generation.sampler import (
    GREEDY_SAMPLING,
    SamplingParameters,
    resolve_sampling,
)
from cubic_serve.generation.tp_sampling import (
    FUSED_ROOT_SAMPLING_PROFILE_METRIC,
    ShardedSamplingParameters,
    ShardedSamplingWorkspace,
    gather_greedy_broadcast_tp,
    sample_sharded_broadcast_tp,
)
from cubic_serve.guardrail.compare import stable_digest
from cubic_serve.models.builtin import builtin_text_config
from cubic_serve.models.checkpoint import CheckpointInventory
from cubic_serve.models.parallel_weights import AxisSpan
from cubic_serve.models.qwen3_5.model import (
    CONV_STATE_ID,
    RECURRENT_STATE_ID,
    Qwen3_5Config,
)
from cubic_serve.models.qwen3_5.weights import (
    Qwen3_5ParallelWeightPlan,
    load_qwen3_5_rank_weights,
)
from cubic_serve.models.qwen3_5_moe import (
    Qwen3_5MoeConfig,
    load_qwen3_5_moe_rank_weights,
)
from cubic_serve.models.weights import CheckpointWeightSource
from cubic_serve.ops.cuda.launch_cost import LinearLaunchCost, linear_cost_scope
from cubic_serve.ops.cuda.sampling import RootTopKTopPSamplingPrimitive
from cubic_serve.ops.cuda.sampling.greedy import POST_METRIC, PRE_METRIC, CapturedGreedySampler
from cubic_serve.ops.cuda.sampling.speculative import verify_one_draft_candidates
from cubic_serve.ops.strategies import (
    OperatorConfig,
    collective_parameters,
    collective_thread_scope,
    operator_strategy_scope,
    overlap_collective_threads,
)
from cubic_serve.runtime.cache import (
    CacheArenaSpec,
    CacheCapacityError,
    CacheCapacityPlan,
    DeviceMemoryBudget,
    DeviceMemorySnapshot,
    KVLayout,
    estimate_cache_arena_bytes,
    plan_cache_capacity,
    read_cuda_memory_snapshot,
)
from cubic_serve.runtime.continuous_batch import (
    RESIDENT_STEP_CAPACITY,
    BatchProtocolError,
    DraftWindowBinding,
    InputEncoderExecutionLease,
    SharedPublicationInterrupt,
    batch_segments,
    deserialize_worker_command,
    independent_decode_partitions,
    owned_segment_rows,
    owned_single_token_rows,
    prepared_input_fields,
    resident_batch_ownership,
    serialize_worker_command,
)
from cubic_serve.runtime.continuous_batch import (
    BatchCommand as _BatchCommand,
)
from cubic_serve.runtime.continuous_batch import (
    BatchResult as _BatchResult,
)
from cubic_serve.runtime.continuous_batch import (
    BatchRow as _BatchRow,
)
from cubic_serve.runtime.continuous_batch import (
    CloseCommand as _CloseCommand,
)
from cubic_serve.runtime.continuous_batch import (
    ContinueBatchCommand as _ContinueBatchCommand,
)
from cubic_serve.runtime.continuous_batch import (
    GenerateCommand as _GenerateCommand,
)
from cubic_serve.runtime.continuous_batch import (
    SegmentResult as _SegmentResult,
)
from cubic_serve.runtime.continuous_batch import (
    WorkerCommand as _WorkerCommand,
)
from cubic_serve.runtime.continuous_batch import (
    run_resident_continuation as _run_resident_continuation,
)
from cubic_serve.executor.readiness import execution_ready
from cubic_serve.runtime.logging import (
    diagnostic_event,
    configure_logging,
    format_weight_progress,
    serving_status,
    startup_progress,
    startup_sampling_qualified,
)
from cubic_serve.runtime.profiling import (
    PROFILE_SAMPLE_LIMIT,
    CudaStepProfiler,
    StepProfile,
    new_step_profile,
)
from cubic_serve.runtime.speculative import (
    DraftPolicyConfig,
    DynamicDraftPlanner,
    DynamicDraftPolicy,
    draft_decision_horizon,
)
from cubic_serve.runtime.token_stream import (
    TokenGenerationOverloaded, TokenSink, TokenStream, TokenWakeBatch,
)

_LOGGER = logging.getLogger(__name__)
_WORLD_SIZE = 4
_PHYSICAL_GRAPH_BUCKET_CAPACITIES = (1, 2, 4, 8, 16, 32, 64, 128)
_PP1_PHYSICAL_GRAPH_BUCKET_CAPACITIES = (*_PHYSICAL_GRAPH_BUCKET_CAPACITIES, 256)
# PP2 covers C256 with its qualified 2 x C128 pipeline wave.  PP1 has no
# pipeline bubble to overlap, so C256 is a real physical graph there.
_GRAPH_BUCKET_CAPACITIES = (*_PHYSICAL_GRAPH_BUCKET_CAPACITIES, 256)
_CAPTURED_M = _GRAPH_BUCKET_CAPACITIES[0]
_MAX_CAPTURED_M = _GRAPH_BUCKET_CAPACITIES[-1]
_MAX_PHYSICAL_CAPTURED_M = _PHYSICAL_GRAPH_BUCKET_CAPACITIES[-1]
_MAX_LOGICAL_CAPTURED_M = 2 * _PP1_PHYSICAL_GRAPH_BUCKET_CAPACITIES[-1]
# A dual PP wave pays one additional pipeline bubble versus one physical wave.
# Keep the portable scheduler on C128 until the 2M graph is at least 7/8 full;
# this shape ratio is topology-derived and contains no device/SKU branch.
_WAVE_MIN_LOGICAL_M = 7 * _MAX_CAPTURED_M // 8
_BLOCK_SIZE = 16
_WORKER_RESPONSE_TIMEOUT_SECONDS = 600.0
_PREFIX_BLOCK_PACKER = struct.Struct(f"<{_BLOCK_SIZE}I")
_COMMAND_NOTICE = struct.Struct("<Q")
_FEEDING_PROFILE_SAMPLE_LIMIT = PROFILE_SAMPLE_LIMIT
_SERVING_STATS_INTERVAL_SECONDS = 10.0
_DRAFT_TRACE_ENV = "CUBIC_SERVE_DRAFT_TRACE"
_StepFeedingProfile = StepProfile
_CudaStepProfiler = CudaStepProfiler


def _mixed_prefill_token_quota(batch_limit: int, pending_requests: int) -> int:
    """Grow cold-fill work while preserving at least half a mixed batch for decode."""

    if batch_limit < 2 or pending_requests < 1:
        raise ValueError("mixed prefill requires decode and pending prompt capacity")
    return min(batch_limit // 2, max(batch_limit // 8, pending_requests))


_CUDA_PROFILE_METRICS = (
    PRE_METRIC,
    POST_METRIC,
    "production_bind_gpu_ms",
    "graph_replay_gpu_ms",
    "local_greedy_max_gpu_ms",
    "final_tp_all_gather_gpu_ms",
    "final_tp_p2p_gpu_ms",
    "world_broadcast_gpu_ms",
    "guard_rank_logits_finite_gpu_ms",
    "rank_local_topk_gpu_ms",
    "rank_greedy_max_gpu_ms",
    "tp_candidate_all_gather_gpu_ms",
    "guard_root_candidates_gpu_ms",
    "root_global_topk_gpu_ms",
    "root_temperature_softmax_gpu_ms",
    "root_top_p_gpu_ms",
    "root_sample_gpu_ms",
    FUSED_ROOT_SAMPLING_PROFILE_METRIC,
    "tp_token_broadcast_gpu_ms",
    "guard_final_token_gpu_ms",
)
_MTP_CUDA_PROFILE_METRICS = (
    *_CUDA_PROFILE_METRICS,
    "mtp_proposal_bind_gpu_ms",
    "mtp_proposal_graph_gpu_ms",
    "mtp_verify_gpu_ms",
    "mtp_catchup_bind_gpu_ms",
    "mtp_catchup_graph_gpu_ms",
)


def _new_step_feeding_profile(role: str) -> StepProfile | None:
    try:
        return new_step_profile(
            role,
            schema="cubic.qwen3_5.feeding-profile.v1",
            logger=_LOGGER,
        )
    except ValueError as error:
        raise Qwen3_5TP2PP2ServingError(str(error)) from error


def _physical_graph_capacities(
    max_speculative_tokens: int = 0,
    *,
    pipeline_parallel_size: int = 2,
    max_running_requests: int | None = None,
    include_speculative: bool = False,
) -> tuple[int, ...]:
    """Cover PP1 admission with permanent startup power-of-two anchors.

    Request capacity and speculative token capacity remain distinct. PP has
    its separately qualified wave portfolio; this does not change that plan.
    """

    if type(pipeline_parallel_size) is not int or pipeline_parallel_size <= 0:
        raise ValueError("pipeline parallel size must be positive")
    if max_running_requests is not None and (
        type(max_running_requests) is not int or max_running_requests <= 0
    ):
        raise ValueError("request capacity must be positive")
    anchors = (
        _PP1_PHYSICAL_GRAPH_BUCKET_CAPACITIES
        if pipeline_parallel_size == 1
        else _PHYSICAL_GRAPH_BUCKET_CAPACITIES
    )
    if pipeline_parallel_size == 1 and max_running_requests is not None:
        limit = max(anchors[-1], max_running_requests)
        if include_speculative and max_speculative_tokens:
            # Probe one adjacent token bucket, bounded by the configured K.
            limit = min(2 * limit, max_running_requests * (max_speculative_tokens + 1))
        anchors = tuple(1 << bit for bit in range((limit - 1).bit_length() + 1))
    exact_through = max_speculative_tokens + 1 if max_speculative_tokens else 1
    if exact_through > anchors[-1]:
        raise ValueError("speculative review batch exceeds the physical graph pool")
    return anchors


def _graph_capture_order(capacities: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(reversed(capacities))


def _serving_graph_capacities(
    physical_capacities: tuple[int, ...],
    max_running_requests: int,
    *,
    pipeline_parallel_size: int,
    max_speculative_tokens: int = 0,
) -> tuple[int, ...]:
    """Keep CUDA graph capacity independent from admitted request capacity.

    PP2 may use its qualified dual-lane wave to overlap pipeline work. PP1
    anchors cover admitted requests; speculative review rows beyond their
    token budget still use consecutive transactions, never a logical wave
    masquerading as a larger physical graph.
    """

    if (
        type(max_running_requests) is not int
        or max_running_requests <= 0
        or type(pipeline_parallel_size) is not int
        or pipeline_parallel_size <= 0
    ):
        raise ValueError("request and pipeline capacities must be positive")
    logical_limit = max_running_requests * (max_speculative_tokens + 1)
    needs_wave = (
        logical_limit > physical_capacities[-1]
        and pipeline_parallel_size > 1
    )
    return (
        (*physical_capacities, 2 * physical_capacities[-1])
        if needs_wave
        else physical_capacities
    )


def _uses_wavefront(
    physical_capacities: tuple[int, ...],
    graph_capacities: tuple[int, ...],
    pipeline_parallel_size: int,
) -> bool:
    """Distinguish a PP logical wave from an equally sized physical graph."""

    return graph_capacities[-1] > physical_capacities[-1]


def _effective_max_sequence_tokens(
    model_max_sequence_tokens: int,
    page_capacity: int,
    largest_graph_capacity: int,
    *, block_size: int = _BLOCK_SIZE, scratch_pages: int | None = None,
) -> int:
    """Fit one request context after source/destination graph COW pages."""

    if (
        type(model_max_sequence_tokens) is not int
        or model_max_sequence_tokens <= 0
        or type(page_capacity) is not int
        or page_capacity <= 0
        or type(largest_graph_capacity) is not int
        or largest_graph_capacity <= 0
        or type(block_size) is not int
        or block_size <= 0
    ):
        raise ValueError("context capacity inputs must be positive integers")
    if scratch_pages is not None and (type(scratch_pages) is not int or scratch_pages < 0):
        raise ValueError("context scratch pages must be a nonnegative integer")
    context_pages = page_capacity - (
        2 * largest_graph_capacity if scratch_pages is None else scratch_pages)
    if context_pages < 1:
        raise Qwen3_5TP2PP2ServingError(
            "rank cache leaves no page for an admitted request context"
        )
    return min(model_max_sequence_tokens, context_pages * block_size)


def _logical_m_values(
    captured_m: int,
    capacities: tuple[int, ...] = _GRAPH_BUCKET_CAPACITIES,
) -> tuple[int, ...]:
    """Return a bucket's audited domain, including M1 padding in physical M2."""

    try:
        index = capacities.index(captured_m)
    except ValueError as error:
        raise ValueError("captured M is outside the serving graph pool") from error
    lower = 1 if index == 0 or captured_m == 2 else capacities[index - 1] + 1
    return tuple(range(lower, captured_m + 1))


def _bucket_for_logical_m(
    logical_m: int,
    capacities: tuple[int, ...] = _GRAPH_BUCKET_CAPACITIES,
) -> int:
    if type(logical_m) is not int or logical_m <= 0:
        raise ValueError("logical M must be a positive integer")
    for captured_m in capacities:
        if logical_m <= captured_m:
            return captured_m
    raise Qwen3_5TP2PP2Overloaded("runnable rows exceed the qualified graph pool")


def _owner_state_capacity(
    max_running_requests: int,
    physical_capacities: tuple[int, ...],
    *,
    max_speculative_tokens: int,
    use_prefix_cache: bool,
) -> int:
    """Reserve request owners, graph witnesses and enabled audit fork owners.

    These are storage requirements, not an increase in admitted concurrency.
    Prefix and speculative fork audits retain four distinct owners; ordinary
    COW needs two. Graph rows round up to an existing anchor independently.
    """

    if type(max_running_requests) is not int or max_running_requests <= 0:
        raise ValueError("request capacity must be a positive integer")
    if type(max_speculative_tokens) is not int or max_speculative_tokens < 0:
        raise ValueError("speculative capacity must be a nonnegative integer")
    audit_owners = 4 if use_prefix_cache or max_speculative_tokens else 2
    required_rows = max(max_running_requests, max_speculative_tokens + 1, audit_owners)
    graph_owners = _bucket_for_logical_m(
        min(required_rows, physical_capacities[-1]), physical_capacities,
    )
    return max(max_running_requests, audit_owners, graph_owners)


def _cold_audit_logical_ms(captured_m: int) -> tuple[int, ...]:
    """Bound cold-start cost while covering lower, interior, and full bucket M."""

    return _bounded_audit_logical_ms(_logical_m_values(captured_m))


def _bounded_audit_logical_ms(domain: tuple[int, ...]) -> tuple[int, ...]:
    if len(domain) <= 4:
        return domain
    lower, upper = domain[0], domain[-1]
    midpoint = lower + (upper - lower) // 2
    return tuple(dict.fromkeys((lower, midpoint, upper)))


class Qwen3_5TP2PP2ServingError(RuntimeError):
    """The qualified graph topology cannot safely serve this request."""


class Qwen3_5TP2PP2Overloaded(Qwen3_5TP2PP2ServingError, TokenGenerationOverloaded):
    """The request cannot be admitted within the qualified cache envelope."""


@dataclass(frozen=True, slots=True)
class Qwen3_5TP2PP2RankCacheCapacity:
    """Observed and allocated rank-local cache capacity at startup."""

    rank: int
    device: str
    page_capacity: int
    state_slot_capacity: int
    scratch_slot_count: int
    bytes_per_page: int
    bytes_per_state_bundle: int
    arena_backing_bytes: int
    bootstrap_arena_backing_bytes: int
    reserve_bytes: int
    free_after_bootstrap_bytes: int
    replaceable_free_bytes: int
    free_after_final_capture_bytes: int
    total_bytes: int
    # Display-only estimate, independent of request/context allocation caps.
    vram_page_capacity_estimate: int | None = None
    shared_pool_bytes: int | None = None

    @property
    def arena_gib(self) -> float:
        return self.arena_backing_bytes / (1 << 30)


def _plan_replacement_cache_capacity(
    *,
    snapshot: DeviceMemorySnapshot,
    layout: KVLayout,
    bootstrap_spec: CacheArenaSpec,
    state_slot_capacity: int,
    minimum_page_capacity: int,
    reserve_bytes: int,
    auxiliary_bytes_per_page: int = 0,
    scratch_capacity: int | None = None,
) -> tuple[CacheCapacityPlan, int]:
    """Credit only backing that the final arena replaces after graph reset."""

    bootstrap_bytes = estimate_cache_arena_bytes(layout, bootstrap_spec)
    replaceable_free = snapshot.free_bytes + bootstrap_bytes
    if replaceable_free > snapshot.total_bytes:
        raise Qwen3_5TP2PP2ServingError(
            "bootstrap arena backing exceeds the observed device allocation"
        )
    try:
        plan = plan_cache_capacity(
            snapshot=DeviceMemorySnapshot(
                snapshot.device,
                replaceable_free,
                snapshot.total_bytes,
            ),
            arena_id=bootstrap_spec.arena_id,
            layout=layout,
            state_specs=bootstrap_spec.state_specs,
            state_slot_capacity=state_slot_capacity,
            reserve_bytes=reserve_bytes,
            auxiliary_bytes_per_page=auxiliary_bytes_per_page,
            scratch_capacity=(
                bootstrap_spec.scratch_capacity
                if scratch_capacity is None
                else scratch_capacity
            ),
            scratch_arena_ids=bootstrap_spec.scratch_arena_ids,
        )
    except CacheCapacityError as error:
        raise Qwen3_5TP2PP2ServingError(f"rank cache capacity rejected: {error}") from error
    if plan.arena_spec.page_capacity < minimum_page_capacity:
        raise Qwen3_5TP2PP2ServingError(
            "rank cache capacity cannot hold the graph-pool context envelope: "
            f"pages={plan.arena_spec.page_capacity} required_pages={minimum_page_capacity} "
            f"free={snapshot.free_bytes / (1 << 30):.3f}GiB "
            f"replaceable_free={replaceable_free / (1 << 30):.3f}GiB "
            f"reserve={reserve_bytes / (1 << 30):.3f}GiB "
            f"state_slots={state_slot_capacity} "
            f"state_bundle={plan.bytes_per_state_bundle / (1 << 20):.3f}MiB "
            f"scratch_slots={plan.scratch_slot_count}"
        )
    return plan, bootstrap_bytes


def _largest_fitting_physical_graph(
    *,
    snapshot: DeviceMemorySnapshot,
    layout: KVLayout,
    bootstrap_spec: CacheArenaSpec,
    model_table_pages: int,
    auto_max_sequence_tokens: bool,
    max_running_requests: int,
    state_slot_capacity: int,
    max_speculative_tokens: int,
    pipeline_parallel_size: int,
    reserve_bytes: int,
    auxiliary_bytes_per_page: int = 0,
) -> int:
    """Choose the largest portable graph anchor that fits rank-local memory."""

    available = _physical_graph_capacities(
        max_speculative_tokens,
        pipeline_parallel_size=pipeline_parallel_size,
        max_running_requests=max_running_requests,
        include_speculative=True,
    )
    exact_through = max_speculative_tokens + 1 if max_speculative_tokens else 1
    required = (
        _bucket_for_logical_m(max_running_requests, available)
        if pipeline_parallel_size == 1
        else exact_through
    )
    rejection: Qwen3_5TP2PP2ServingError | None = None
    for candidate in reversed(available):
        if candidate < exact_through or (
            candidate > state_slot_capacity
            and (not max_speculative_tokens or pipeline_parallel_size != 1)
        ):
            continue
        physical = tuple(
            capacity
            for capacity in available
            if capacity <= candidate
        )
        serving = _serving_graph_capacities(
            physical,
            max_running_requests,
            pipeline_parallel_size=pipeline_parallel_size,
            max_speculative_tokens=max_speculative_tokens,
        )
        minimum_pages = minimum_kv_page_capacity(
            1 if auto_max_sequence_tokens else model_table_pages,
            serving[-1],
        )
        try:
            _plan_replacement_cache_capacity(
                snapshot=snapshot,
                layout=layout,
                bootstrap_spec=bootstrap_spec,
                state_slot_capacity=state_slot_capacity,
                minimum_page_capacity=minimum_pages,
                reserve_bytes=reserve_bytes,
                auxiliary_bytes_per_page=auxiliary_bytes_per_page,
                scratch_capacity=candidate,
            )
        except Qwen3_5TP2PP2ServingError as error:
            rejection = error
            if candidate <= required:
                break
            continue
        return candidate
    detail = f": {rejection}" if rejection is not None else ""
    raise Qwen3_5TP2PP2ServingError(
        f"rank memory cannot fit required physical C{required} graph{detail}"
    ) from rejection


def _is_digest(value: object) -> bool:
    if type(value) is not str or not value.startswith("b2s256:") or len(value) != 71:
        return False
    try:
        int(value[7:], 16)
    except ValueError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class Qwen3_5TP2PP2ServingQualification:
    """Instance-bound multi-M cold-start evidence.

    This is a trusted-process integrity record, not an arbitrary-code security
    boundary.  ``start`` always reruns the graph checks and only accepts an
    optional caller record when it exactly matches the new result.
    """

    model_id: str
    inventory_digest: str
    semantic_plan_digest: str
    topology_digest: str
    partition_digest: str
    max_sequence_tokens: int
    rank_graph_evidence_ids: tuple[str, ...]
    graph_evidence_id: str
    graph_buckets: tuple[tuple[int, tuple[int, ...]], ...] = tuple(
        (captured_m, _logical_m_values(captured_m, _PHYSICAL_GRAPH_BUCKET_CAPACITIES))
        for captured_m in _PHYSICAL_GRAPH_BUCKET_CAPACITIES
    )

    def __post_init__(self) -> None:
        if (
            type(self.model_id) is not str
            or not self.model_id
            or self.model_id.strip() != self.model_id
        ):
            raise ValueError("serving qualification requires a canonical model id")
        digests = (
            self.inventory_digest,
            self.semantic_plan_digest,
            self.topology_digest,
            self.partition_digest,
            *self.rank_graph_evidence_ids,
            self.graph_evidence_id,
        )
        if any(not _is_digest(item) for item in digests):
            raise ValueError("serving qualification identities must use content_digest digests")
        graph_capacities = tuple(captured_m for captured_m, _ in self.graph_buckets)
        canonical_graph_buckets = tuple(
            (captured_m, _logical_m_values(captured_m, graph_capacities))
            for captured_m in graph_capacities
        )
        if (
            type(self.rank_graph_evidence_ids) is not tuple
            or not self.rank_graph_evidence_ids
            or self.max_sequence_tokens <= 0
            or not graph_capacities
            or tuple(sorted(set(graph_capacities))) != graph_capacities
            or graph_capacities[-1] > _MAX_LOGICAL_CAPTURED_M
            or self.graph_buckets != canonical_graph_buckets
            or self.graph_evidence_id
            != _graph_evidence_id(
                model_id=self.model_id,
                inventory_digest=self.inventory_digest,
                semantic_plan_digest=self.semantic_plan_digest,
                topology_digest=self.topology_digest,
                partition_digest=self.partition_digest,
                max_sequence_tokens=self.max_sequence_tokens,
                rank_graph_evidence_ids=self.rank_graph_evidence_ids,
                graph_buckets=self.graph_buckets,
            )
        ):
            raise ValueError("serving qualification differs from the graph-pool evidence boundary")


def _graph_evidence_id(
    *,
    model_id: str,
    inventory_digest: str,
    semantic_plan_digest: str,
    topology_digest: str,
    partition_digest: str,
    max_sequence_tokens: int,
    rank_graph_evidence_ids: tuple[str, ...],
    graph_buckets: tuple[tuple[int, tuple[int, ...]], ...],
) -> str:
    return stable_digest(
        {
            "schema": "cubic.qwen3_5.tp2_pp2.serving-qualification.v2",
            "model_id": model_id,
            "inventory_digest": inventory_digest,
            "semantic_plan_digest": semantic_plan_digest,
            "topology_digest": topology_digest,
            "partition_digest": partition_digest,
            "max_sequence_tokens": max_sequence_tokens,
            "graph_buckets": graph_buckets,
            "rank_graph_evidence_ids": rank_graph_evidence_ids,
        }
    )


@dataclass(frozen=True, slots=True)
class _Admission:
    config: Qwen3_5Config | Qwen3_5MoeConfig
    inventory: CheckpointInventory
    weight_plan: Any
    parallel_plan: Any
    sealed: SealedPartition
    max_sequence_tokens: int
    auto_max_sequence_tokens: bool
    max_running_requests: int
    num_speculative_tokens: int
    max_speculative_tokens: int
    dynamic_max_speculative_tokens: int
    dynamic_speculation: bool
    use_prefix_cache: bool
    mamba_ssm_cache_dtype: str
    prefill_chunk_tokens: int | None = None
    operator_config: OperatorConfig = field(default_factory=OperatorConfig)
    operator_model_id: str = ""
    draft_policy: DraftPolicyConfig = field(default_factory=DraftPolicyConfig)
    block_size_tokens: int = _BLOCK_SIZE
    device_memory_budgets: dict[int, DeviceMemoryBudget] = field(default_factory=dict)
    input_encoder_reserves: dict[int, int] = field(default_factory=dict)
    input_encoder_factory: Any = None
    input_encoder_connection: Any = None

    @property
    def model_root(self) -> Path:
        return self.inventory.model_root


def _topology(
    devices: tuple[int, ...], *, tp_size: int, pp_size: int, ep_size: int
) -> ParallelTopology:
    topology = ParallelTopology.single_node(
        topology_id=f"qwen3_5.performance.tp{tp_size}_pp{pp_size}_ep{ep_size}.bucketed.v1",
        device_indices=devices,
        tp_size=tp_size,
        pp_size=pp_size,
        ep_size=ep_size,
    )
    if tp_size == 1:
        # Preserve the same ordered reference and sampling group contract at
        # width one; a singleton collective has identity semantics.
        topology = replace(
            topology,
            groups=topology.groups + tuple(
                RankGroup(f"tp:{p.pipeline_stage}", GroupKind.TENSOR, (p.rank,))
                for p in topology.placements
            ),
        )
    return topology


def _admit(
    checkpoint: Path,
    devices: tuple[int, ...],
    max_sequence_tokens: int | None,
    max_running_requests: int,
    tp_size: int = 2,
    pp_size: int = 2,
    ep_size: int = 1,
    num_speculative_tokens: int = 0,
    max_speculative_tokens: int = 0,
    dynamic_max_speculative_tokens: int = 0,
    dynamic_speculation: bool = False,
    use_prefix_cache: bool = True,
    mamba_ssm_cache_dtype: str = "bfloat16",
    prefill_chunk_tokens: int | None = None,
    checkpoint_weights: CheckpointWeightSource | None = None,
) -> _Admission:
    """Compatibility entry for the model-owned legacy startup declaration."""
    from cubic_serve.models.qwen3_5.serving import admit_legacy_performance

    return admit_legacy_performance(
        checkpoint, devices,
        max_sequence_tokens=max_sequence_tokens,
        max_running_requests=max_running_requests,
        tp_size=tp_size,
        pp_size=pp_size,
        ep_size=ep_size,
        num_speculative_tokens=num_speculative_tokens,
        max_speculative_tokens=max_speculative_tokens,
        dynamic_max_speculative_tokens=dynamic_max_speculative_tokens,
        dynamic_speculation=dynamic_speculation,
        use_prefix_cache=use_prefix_cache,
        mamba_ssm_cache_dtype=mamba_ssm_cache_dtype,
        prefill_chunk_tokens=prefill_chunk_tokens,
        checkpoint_weights=checkpoint_weights,
    )


def _resolve_max_sequence_tokens(config: Qwen3_5Config, requested: int | None) -> int:
    """Resolve an omitted serving cap from the admitted model configuration."""

    model_max = config.graph.text.max_position_embeddings
    if requested is None:
        return model_max
    if type(requested) is not int or requested <= 0:
        raise ValueError("max_sequence_tokens must be a positive integer")
    if requested > model_max:
        raise ValueError("serving capacity exceeds the model position table")
    return requested


def _mtp_kv_bytes_per_page(admission: _Admission) -> int:
    """Return the proposal cache charged alongside every target KV page."""

    if not admission.num_speculative_tokens:
        return 0
    text = admission.config.graph.text
    local_heads = max(
        1,
        text.full_attention.key_value_heads
        // admission.parallel_plan.topology.tp_size,
    )
    return _BLOCK_SIZE * (
        2 * local_heads * text.full_attention.head_dim * torch.bfloat16.itemsize
        + torch.bool.itemsize
    )


def _load_sampling_defaults(checkpoint: Path) -> SamplingParameters:
    from cubic_serve.generation.sampler import load_sampling_defaults

    try:
        return load_sampling_defaults(checkpoint)
    except ValueError as error:
        raise Qwen3_5TP2PP2ServingError(str(error)) from error


def _scratch_arena_id(rank: int) -> str:
    return f"qwen3_5-serving-rank{rank}-scratch"


def _resource(rank: int, captured_m: int) -> RankReplayResources:
    stem = f"qwen3_5-serving-c{captured_m}-rank{rank}"
    return RankReplayResources(
        rank,
        stem + "-graph",
        stem + "-stream",
        stem + "-buffers",
        stem + "-workspace",
        _scratch_arena_id(rank),
    )


def _final_ranks(plan: Qwen3_5ParallelWeightPlan) -> tuple[int, ...]:
    final_stage = plan.topology.pp_size - 1
    return tuple(
        item.rank for item in plan.topology.placements if item.pipeline_stage == final_stage
    )


def _sampler_rank(plan: Qwen3_5ParallelWeightPlan) -> int:
    return _final_ranks(plan)[0]


def _final_vocab_spans(plan: Qwen3_5ParallelWeightPlan) -> tuple[AxisSpan, ...]:
    spans: list[AxisSpan] = []
    for rank in _final_ranks(plan):
        bindings = tuple(
            item for item in plan.bindings_for_rank(rank) if item.tensor_name == "lm_head.weight"
        )
        if len(bindings) != 1 or len(bindings[0].spans) != 1:
            raise Qwen3_5TP2PP2ServingError("final PP rank has no single vocab shard")
        spans.append(bindings[0].spans[0])
    return tuple(spans)


def _build_body(
    admission: _Admission,
    weights: Any,
    runtime: Any,
    *,
    max_sequence_tokens: int,
    page_capacity: int,
    state_capacity: int = _CAPTURED_M,
    scratch_capacity: int = _MAX_PHYSICAL_CAPTURED_M,
    mrope_tables: Any = None,
    packed_gate_up_weights: Tensor | None = None,
    packed_gdn_abz: tuple[Tensor, tuple[int, int, int]] | None = None,
) -> tuple[Qwen3_5TP2PP2DecodeBody, RankReplayResources]:
    placement = admission.parallel_plan.topology.placements[weights.rank]
    pages = (max_sequence_tokens + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    resource = _resource(weights.rank, _CAPTURED_M)
    pp_group = (
        runtime._groups[f"pp:{placement.tensor_rank}"]
        if admission.parallel_plan.topology.pp_size > 1
        else None
    )
    body = Qwen3_5TP2PP2DecodeBody.build(
        config=admission.config,
        parallel_plan=admission.parallel_plan,
        weights=weights,
        tp_group=runtime._groups[f"tp:{placement.pipeline_stage}"],
        pp_group=pp_group,
        captured_m=_CAPTURED_M,
        table_pages=pages,
        page_capacity=page_capacity,
        state_capacity=state_capacity,
        recurrent_state_dtype=(
            torch.bfloat16
            if admission.mamba_ssm_cache_dtype == "bfloat16"
            else torch.float32
        ),
        scratch_arena_id=resource.scratch_arena_id,
        scratch_capacity=scratch_capacity,
        mrope_tables=mrope_tables,
        packed_gate_up_weights=packed_gate_up_weights,
        packed_gdn_abz=packed_gdn_abz,
    )
    return body, resource


def _decode_overlap_lanes(
    bodies: Mapping[int, Qwen3_5TP2PP2DecodeBody],
) -> tuple[Qwen3_5TP2PP2DecodeBody, ...]:
    return tuple({
        id(child): child
        for body in bodies.values()
        for child in getattr(body, "_decode_overlap_auxiliary", ())
    }.values())


def _prepare_decode_overlap_family(
    primary: Qwen3_5TP2PP2DecodeBody,
    bodies: Mapping[int, Qwen3_5TP2PP2DecodeBody],
    plans: Mapping[int, DecodeOverlapPlan] | None = None,
) -> None:
    # Native launch_steps exposes row-independent work and collective yields,
    # including routed experts. Complete-model calibration prices repeated
    # expert reads; model identity is not an overlap admission criterion.
    # PP retains its separate wave dependency contract.
    if primary._tp_size == 1 or primary._pp_size != 1:
        return
    if plans is not None and not plans:
        primary._decode_overlap_auxiliary = ()
        return
    policy = collective_parameters()
    thread_candidates = (
        common_overlap_thread_candidates(overlap_collective_threads(), primary.hidden.device)
        if plans is None else ()
    )
    pool: dict[tuple[int, int, bool, int | None], Qwen3_5TP2PP2DecodeBody] = {}
    costs: dict[int, LinearLaunchCost] = {}
    initial = torch.cuda.memory_allocated(primary.hidden.device)
    common = torch.zeros(1, dtype=torch.int64, device=primary.hidden.device)
    next_lane = max((b.lane_id for b in (primary, *primary._bootstrap_siblings)), default=0) + 1
    for body in reversed(tuple(bodies.values())):
        if plans is not None and body.captured_m not in plans:
            continue
        # A compiled binding declares its row geometry. Do not guess the tile
        # of an opaque vendor GEMM from dtype, GPU name or model identity.
        bindings = [
            value for name, value in vars(body).items()
            if name.startswith(("_linear_", "_grouped_")) and value is not None
        ]
        counts = [getattr(binding, "row_tile_count", 0) for binding in bindings]
        common.fill_(min(counts, default=0))
        all_reduce(common, op=dist.ReduceOp.MIN)
        row_tiles = int(common.item())
        if not row_tiles:
            continue
        candidates = []
        upper = _graph_runtime_reserve_bytes({body.captured_m: body}, ())
        resources = (
            tuple((estimated, threads) for threads in thread_candidates
                  for estimated in (False, True))
            if plans is None else
            ((plans[body.captured_m].estimated_linear, plans[body.captured_m].collective_threads),)
        )
        for estimated, threads in resources:
            partitions = (
                policy.overlap_partitions(body.captured_m, row_tiles)
                if plans is None
                else (plans[body.captured_m].parts,)
            )
            for parts in partitions:
                occurrences: dict[int, int] = {}
                children = []
                for size in parts:
                    occurrence = occurrences.get(size, 0)
                    occurrences[size] = occurrence + 1
                    key = (size, occurrence, estimated, threads)
                    if key not in pool:
                        used = torch.cuda.memory_allocated(primary.hidden.device) - initial
                        free, _ = torch.cuda.mem_get_info(primary.hidden.device)
                        common.fill_(int(
                            used + upper <= policy.decode_overlap_max_bytes and upper < free // 2
                        ))
                        all_reduce(common, op=dist.ReduceOp.MIN)
                        if plans is None and not int(common.item()):
                            break
                        if estimated:
                            if size not in costs:
                                source = bodies[size]
                                if source._tp_fixed_sum is None:
                                    raise RuntimeError(
                                        "overlap calibration requires a bound TP sum"
                                    )
                                costs[size] = LinearLaunchCost.calibrate(
                                    source.hidden, source._tp_fixed_sum.launch,
                                    group=source._tp_group, samples=policy.decode_overlap_samples,
                                )
                            with linear_cost_scope(costs[size]), collective_thread_scope(threads):
                                pool[key] = primary.build_sibling(
                                    captured_m=size, lane_id=next_lane,
                                )
                        else:
                            with collective_thread_scope(threads):
                                pool[key] = primary.build_sibling(
                                    captured_m=size, lane_id=next_lane,
                                )
                        pool[key]._decode_overlap_estimated = estimated
                        pool[key]._decode_overlap_collective_threads = threads
                        next_lane += 1
                    children.append(pool[key])
                if len(children) != len(parts):
                    break
                candidates.append(tuple(children))
        body._decode_overlap_candidates = tuple(candidates)
    # A partially allocated final plan still belongs to the bootstrap arena
    # family; keep it in the reserve/rebind closure without exposing a graph.
    primary._decode_overlap_auxiliary = tuple(pool.values())


def _retain_profiled_decode_overlap(
    bodies: Mapping[int, Qwen3_5TP2PP2DecodeBody],
) -> dict[int, DecodeOverlapPlan]:
    """Discard cold-only lanes before measuring the cache's available memory."""
    plans = {}
    retained = set()
    for m, body in bodies.items():
        candidates = getattr(body, "_decode_overlap_candidates", ())
        if candidates:
            if len(candidates) != 1:
                raise RuntimeError("overlap memory sizing requires a completed selection")
            children = candidates[0]
            plans[m] = DecodeOverlapPlan(
                tuple(c.captured_m for c in children), children[0]._decode_overlap_estimated,
                children[0]._decode_overlap_collective_threads,
            )
            retained.update(id(child) for child in children)
    for body in bodies.values():
        auxiliary = getattr(body, "_decode_overlap_auxiliary", ())
        if auxiliary:
            discarded = {id(child) for child in auxiliary} - retained
            body._decode_overlap_auxiliary = tuple(c for c in auxiliary if id(c) in retained)
            body._bootstrap_siblings = [
                c for c in body._bootstrap_siblings if id(c) not in discarded
            ]
    return plans


def _build_body_family(
    admission: _Admission,
    weights: Any,
    runtime: Any,
    physical_capacities: tuple[int, ...],
    *,
    max_sequence_tokens: int,
    page_capacity: int,
    state_capacity: int,
    scratch_capacity: int,
    wave_enabled: bool,
    mrope_tables: Any = None,
    packed_gate_up_weights: Tensor | None = None,
    packed_gdn_abz: tuple[Tensor, tuple[int, int, int]] | None = None,
    overlap_plans: Mapping[int, DecodeOverlapPlan] | None = None,
) -> tuple[
    Qwen3_5TP2PP2DecodeBody,
    dict[int, Qwen3_5TP2PP2DecodeBody],
    tuple[Qwen3_5TP2PP2DecodeBody, ...],
]:
    transient_bytes = 0

    def finish_construction() -> None:
        nonlocal transient_bytes
        transient_bytes = max(
            transient_bytes,
            torch.cuda.max_memory_allocated(runtime.device)
            - torch.cuda.memory_allocated(runtime.device),
        )
        torch.cuda.reset_peak_memory_stats(runtime.device)

    torch.cuda.reset_peak_memory_stats(runtime.device)
    body, _ = _build_body(
        admission,
        weights,
        runtime,
        max_sequence_tokens=max_sequence_tokens,
        page_capacity=page_capacity,
        state_capacity=state_capacity,
        scratch_capacity=scratch_capacity,
        mrope_tables=mrope_tables,
        packed_gate_up_weights=packed_gate_up_weights,
        packed_gdn_abz=packed_gdn_abz,
    )
    finish_construction()
    bodies = {body.captured_m: body}
    for captured_m in physical_capacities[1:]:
        bodies[captured_m] = body.build_sibling(
            captured_m=captured_m,
            lane_id=2 if captured_m == physical_capacities[-1] else 0,
        )
        finish_construction()
    for captured_m, bucket_body in bodies.items():
        bucket_body.bucket_logical_ms = _logical_m_values(
            captured_m, physical_capacities
        )
    wave_lanes = (
        (
            body.build_sibling(captured_m=physical_capacities[-1], lane_id=0),
            body.build_sibling(captured_m=physical_capacities[-1], lane_id=1),
        )
        if wave_enabled
        else ()
    )
    _prepare_decode_overlap_family(body, bodies, overlap_plans)
    finish_construction()
    # Later resident candidates must not hide an earlier body's temporary
    # calibration peak. The final family repeats these operator constructors.
    body._construction_transient_bytes = transient_bytes
    return body, bodies, wave_lanes


def _warm_body_family(
    bodies: Mapping[int, Qwen3_5TP2PP2DecodeBody],
    wave_lanes: tuple[Qwen3_5TP2PP2DecodeBody, ...],
    physical_capacities: tuple[int, ...],
    *,
    rank: int,
    device: torch.device,
) -> None:
    for captured_m in reversed(physical_capacities):
        diagnostic_event(_LOGGER, 'performance.startup_probe_rank_d_warm_c_d',
            rank,
            captured_m,
        )
        body = bodies[captured_m]
        body.prepare_scratch(_resource(rank, captured_m))
        barrier()
        body.launch()
        torch.cuda.synchronize(device)
        barrier()
        diagnostic_event(_LOGGER, 'performance.startup_probe_rank_d_warm_c_d_2',
            rank,
            captured_m,
        )
    for body in (*wave_lanes, *_decode_overlap_lanes(bodies)):
        body.prepare_scratch(_resource(rank, 2 * body.captured_m))
        barrier()
        body.launch()
        torch.cuda.synchronize(device)
        barrier()


def _build_mtp_body_family(
    admission: _Admission,
    weights: Any,
    runtime: Any,
    physical_capacities: tuple[int, ...],
    *,
    table_pages: int,
    request_page_capacity: int,
) -> dict[int, Qwen3_5TPMTPBody]:
    mtp_page_capacity = request_page_capacity + physical_capacities[-1]
    mtp_bodies: dict[int, Qwen3_5TPMTPBody] = {}
    shared_arena: Qwen3_5MTPArena | None = None
    packed_gate_up: Tensor | None = None
    packed_experts: tuple[Tensor, Tensor] | None = None
    mrope_tables: tuple[Tensor, Tensor, Any] | None = None
    for captured_m in reversed(physical_capacities):
        mtp_body = Qwen3_5TPMTPBody(
            config=admission.config,
            parallel_plan=admission.parallel_plan,
            weights=weights,
            tp_group=runtime._groups["tp:0"],
            captured_m=captured_m,
            table_pages=table_pages,
            page_capacity=mtp_page_capacity,
            max_proposal_tokens=admission.max_speculative_tokens,
            arena=shared_arena,
            packed_gate_up=packed_gate_up,
            packed_experts=packed_experts,
            mrope_tables=mrope_tables,
        )
        if shared_arena is None:
            shared_arena = mtp_body.arena
            packed_gate_up = mtp_body._gate_up
            packed_experts = (
                (mtp_body._expert_gate_up, mtp_body._expert_down)
                if isinstance(admission.config, Qwen3_5MoeConfig)
                else None
            )
            mrope_tables = mtp_body.mrope_tables
        _prepare_mtp_capture(mtp_body, request_page_capacity)
        mtp_bodies[captured_m] = mtp_body
    return mtp_bodies


def _profile_cuda_graph_memory(
    bodies: Mapping[int, Qwen3_5TP2PP2DecodeBody],
    capture_order: tuple[int, ...],
    *,
    rank: int,
    device: torch.device,
    runtime: Any,
    admission: _Admission,
    weights: Any,
    progress: Callable[[str, int, int], None] | None = None,
) -> tuple[int, int, dict[int, DecodeOverlapPlan]]:
    """Measure all graph buckets and their rank staging before KV sizing."""

    if not capture_order:
        return 0, 0, {}
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    pool = torch.cuda.graph_pool_handle()
    stream = torch.cuda.Stream(device=device)
    slots: list[Qwen3_5TP2PP2GraphSlot] = []
    executors: list[_CapturedGraphRank] = []
    constraint_arena = None
    constraint_capacity = max(_serving_graph_capacities(
        tuple(sorted(bodies)), admission.max_running_requests,
        pipeline_parallel_size=admission.parallel_plan.topology.pp_size,
        max_speculative_tokens=admission.dynamic_max_speculative_tokens,
    ))
    with ExitStack() as sampling_lifetime:
        samples: list[int] = []
        for captured_m in capture_order:
            free_before, _ = torch.cuda.mem_get_info(device)
            slot = Qwen3_5TP2PP2GraphSlot.capture(
                bodies[captured_m],
                _resource(rank, captured_m),
                pool=pool,
                stream=stream,
                retain=False,
            )
            slots.append(slot)
            if getattr(bodies[captured_m], "_decode_overlap_candidates", ()):
                bodies[captured_m]._decode_overlap_candidates = (
                    (slot._overlap[2],) if slot._overlap is not None else ()
                )
            executors.append(_CapturedGraphRank(
                bodies[captured_m], _resource(rank, captured_m), slot, runtime, admission
            ))
            if constraint_arena is None:
                constraint_arena = _ConstraintSamplingArena(executors[-1], constraint_capacity)
            _prepare_rank_greedy_sampling(
                executors[-1], sampling_lifetime, qualify_resident=False,
                constraint_arena=constraint_arena,
            )
            torch.cuda.synchronize(device)
            free_after, _ = torch.cuda.mem_get_info(device)
            samples.append(max(int(free_before) - int(free_after), 0))
            if progress is not None:
                progress("graph_measurement", len(samples), len(capture_order))
        assert constraint_arena is not None
        transient_bytes = max(
            torch.cuda.max_memory_allocated(device) - torch.cuda.memory_allocated(device), 0,
        )
        if admission.num_speculative_tokens:
            torch.cuda.reset_peak_memory_stats(device)
            free_before, _ = torch.cuda.mem_get_info(device)
            body = bodies[capture_order[0]]
            request_pages = body.arena.spec.page_capacity
            mtp_bodies = _build_mtp_body_family(
                admission, weights, runtime, tuple(sorted(bodies)),
                table_pages=body.buffers.page_table.shape[1],
                request_page_capacity=request_pages,
            )
            hidden_slots = torch.empty_like(body.hidden)
            mtp_executors = []
            for captured_m in capture_order:
                mtp_body = mtp_bodies[captured_m]
                mtp_slot = Qwen3_5TPMTPGraphSlot.capture(mtp_body)
                mtp_executors.append(_CapturedMTPRank(
                    mtp_body, mtp_slot, runtime, admission,
                    request_page_capacity=request_pages,
                    target_hidden_slots=hidden_slots,
                ))
                _prepare_rank_greedy_sampling(
                    mtp_executors[-1], sampling_lifetime, qualify_resident=False,
                    constraint_arena=constraint_arena,
                )
            torch.cuda.synchronize(device)
            free_after, _ = torch.cuda.mem_get_info(device)
            samples.append(max(int(free_before) - int(free_after), 0))
            transient_bytes = max(transient_bytes,
                torch.cuda.max_memory_allocated(device) - torch.cuda.memory_allocated(device))
            del mtp_executors, mtp_slot, mtp_body, mtp_bodies, hidden_slots
        estimate = sum(samples)
        from cubic_serve.executor.resident_window import (
            ResidentDecodeWindow,
            prepare_resident_compactions,
        )

        if any(isinstance(getattr(e, "_resident_window", None), ResidentDecodeWindow)
               for e in executors):
            before_compaction, _ = torch.cuda.mem_get_info(device)
            prepare_resident_compactions({e.captured_m: e for e in executors}, sampling_lifetime)
            torch.cuda.synchronize(device)
            after_compaction, _ = torch.cuda.mem_get_info(device)
            estimate += max(int(before_compaction) - int(after_compaction), 0)
        overlap_plans = _retain_profiled_decode_overlap(bodies)
        sampling_lifetime.close()
        executors.clear()
        slots.clear()
        del slot, stream, pool, constraint_arena
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        diagnostic_event(_LOGGER, 'performance.graph_and_rank_staging_memory_rank_d',
            rank,
            estimate / (1 << 20),
            transient_bytes / (1 << 20),
            len(capture_order),
        )
        return estimate, transient_bytes, overlap_plans


def _context_staging_growth_bytes(
    capacities: tuple[int, ...],
    wave_lanes: tuple[Qwen3_5TP2PP2DecodeBody, ...],
    bootstrap_pages: int,
    final_pages: int,
    *,
    mtp: bool,
) -> int:
    """Charge final-width tables missing from the small bootstrap profile."""

    extra_pages = max(final_pages - bootstrap_pages, 0)
    # Target: int32 body table, int32 rank metadata, int64 reset indices,
    # and one int32 page-id vector per bucket. MTP: body and bind tables.
    per_page = sum(capacities) * (16 + (8 if mtp else 0)) + 4 * len(capacities)
    per_page += 16 * sum(body.captured_m for body in wave_lanes) + 4 * len(wave_lanes)
    return extra_pages * per_page


def _graph_runtime_reserve_bytes(
    bodies: Mapping[int, Qwen3_5TP2PP2DecodeBody],
    wave_lanes: tuple[Qwen3_5TP2PP2DecodeBody, ...],
) -> int:
    """Bound graph-private allocations by the resident bucket working set."""

    tensors: list[Tensor] = []
    for body in (*bodies.values(), *wave_lanes, *_decode_overlap_lanes(bodies)):
        tensors.extend(
            getattr(body.buffers, name) for name in type(body.buffers).__annotations__
        )
        tensors.extend(
            value
            for name in (
                "_active_rows",
                "_packed_mlp",
                "_packed_gdn_abz",
                "embedding_mask",
                "feature_rows",
                "_moe_router_logits",
                "_moe_routed",
                "_moe_shared_gate",
            )
            if isinstance((value := getattr(body, name, None)), Tensor)
        )
        routed = getattr(body, "_routed_experts", None)
        if routed is not None:
            tensors.extend(value for value in vars(routed).values() if isinstance(value, Tensor))
    storage_bytes: dict[tuple[torch.device, int], int] = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storage_bytes[(tensor.device, storage.data_ptr())] = storage.nbytes()
    return sum(storage_bytes.values())


def _audit_working_set_bytes(body: Qwen3_5TP2PP2DecodeBody) -> int:
    """Reserve simultaneous state witnesses retained by startup qualification."""

    state_bundle_bytes = sum(
        state[0].numel() * state.element_size()
        for state in body.arena.states.values()
    )
    # The steady audit materializes three sampled owner states concurrently.
    return 3 * state_bundle_bytes


def _set_active_rows(body: Qwen3_5TP2PP2DecodeBody, logical_m: int) -> None:
    _bind_audit_rows(body, logical_m, position=0, parity=0, preserve_first=False)


def _audit_cow_pair(row: int, parity: int) -> tuple[int, int]:
    """Return one deterministic, non-aliasing source/destination audit pair."""

    if type(row) is not int or row < 0 or parity not in (0, 1):
        raise ValueError("audit COW pair requires a non-negative row and parity 0 or 1")
    first = 2 * row
    return (first, first + 1) if parity == 0 else (first + 1, first)


def _audit_state_pair(
    body: Qwen3_5TP2PP2DecodeBody,
    row: int,
    parity: int,
) -> tuple[int, int]:
    """Use COW when owner capacity permits, otherwise the production in-place ABI."""

    storage=getattr(body.arena,"shared_storage",None)
    pair_fits=(storage is None or 2*body.captured_m*(storage.page_layout.entry_bytes+
               (0 if storage.state_layout is None else storage.state_layout.entry_bytes))<=storage.budget_bytes)
    if 2*body.captured_m<=body.arena.spec.state_slot_capacity and pair_fits:
        return _audit_cow_pair(row, parity)
    if row >= body.arena.spec.state_slot_capacity:
        # Independent cold graph witnesses use lease-private physical rows;
        # they are never admitted as additional persistent request owners.
        slot = body.arena.spec.state_slot_capacity + row
        return slot, slot
    return row, row


def _bind_audit_rows(
    body: Qwen3_5TP2PP2DecodeBody,
    logical_m: int,
    *,
    position: int,
    parity: int,
    preserve_first: bool,
) -> None:
    """Bind active rows with the same page/state COW ABI used by serving."""

    buffers = body.buffers
    captured_m = body.captured_m
    logical_ms = getattr(body, "bucket_logical_ms", None)
    if logical_ms is None:
        logical_ms = _logical_m_values(captured_m)
    if logical_m not in logical_ms:
        raise ValueError("audit logical M lies outside the serving bucket")
    if (
        type(position) is not int
        or not 0 <= position < body.arena.layout.block_size_tokens
        or type(preserve_first) is not bool
        or 2 * captured_m > body.arena.spec.page_capacity
        or captured_m > body.arena.spec.scratch_capacity
    ):
        raise ValueError("audit COW resources cannot cover the fixed bucket")
    body.set_active_rows(logical_m)
    device = body.hidden.device
    local_layers = torch.arange(buffers.conv_slots.shape[1], dtype=torch.int32, device=device)
    page_pairs = 2 * torch.arange(logical_m, dtype=torch.int64, device=device)
    page_sources = page_pairs + parity
    page_destinations = page_pairs + (1 - parity)
    state_pairs = tuple(_audit_state_pair(body, row, parity) for row in range(logical_m))
    storage = getattr(body.arena, "shared_storage", None)
    if storage is not None:
        extent = 1 + max(slot for pair in state_pairs for slot in pair)
        state_bytes = 0 if storage.state_layout is None else storage.state_layout.entry_bytes
        if (extent > body.arena.spec.state_slot_capacity or
                extent * state_bytes + 2 * logical_m * storage.page_layout.entry_bytes
                > storage.budget_bytes):
            raise ValueError("audit independent owners exceed the shared byte budget")
        state_pairs = tuple(tuple(body.arena.spec.state_slot_capacity - 1 - slot
                                  for slot in pair) for pair in state_pairs)
    state_sources = torch.tensor(
        tuple(source for source, _ in state_pairs), dtype=torch.int64, device=device
    )
    state_destinations = torch.tensor(
        tuple(destination for _, destination in state_pairs), dtype=torch.int64, device=device
    )
    reset_pages = page_sources[1:] if preserve_first else page_sources
    reset_states = state_sources[1:] if preserve_first else state_sources
    if reset_pages.numel():
        body.arena.key.index_fill_(0, reset_pages, 0)
        body.arena.value.index_fill_(0, reset_pages, 0)
        body.arena.written.index_fill_(0, reset_pages, False)
        for payload in getattr(body.arena, "page_fields", {}).values():
            payload.index_fill_(0, reset_pages, 0)
    if reset_states.numel():
        for state in body.arena.states.values():
            state.index_fill_(0, reset_states, 0)
    # Recycled destinations are intentionally dirty: COW must overwrite them.
    body.arena.key.index_fill_(0, page_destinations, 11)
    body.arena.value.index_fill_(
        0, page_destinations, 243 if body.arena.value.dtype == torch.uint8 else -13)
    body.arena.written.index_fill_(0, page_destinations, True)
    for payload in getattr(body.arena, "page_fields", {}).values():
        payload.index_fill_(0, page_destinations, 11)
    distinct_state_destinations = state_destinations[state_destinations != state_sources]
    if distinct_state_destinations.numel():
        for state in body.arena.states.values():
            state.index_fill_(0, distinct_state_destinations, 0.25)
    page_source_i32 = page_sources.to(torch.int32)
    page_destination_i32 = page_destinations.to(torch.int32)
    state_source_i32 = state_sources.to(torch.int32)
    state_destination_i32 = state_destinations.to(torch.int32)
    buffers.position_ids[:, :logical_m].fill_(position)
    buffers.segment_lengths[:logical_m].fill_(1)
    buffers.segment_lengths[logical_m:].zero_()
    buffers.row_activity[:logical_m].fill_(1)
    buffers.row_activity[logical_m:].zero_()
    buffers.page_source[:logical_m].copy_(page_source_i32)
    buffers.page_destination[:logical_m].copy_(page_destination_i32)
    buffers.token_offsets[:logical_m].fill_(position)
    buffers.page_table[:logical_m].copy_(
        page_destination_i32[:, None].expand(-1, buffers.page_table.shape[1])
    )
    buffers.sequence_lengths[:logical_m].fill_(position + 1)
    source_slots = state_source_i32[:, None] * local_layers.numel() + local_layers[None, :]
    destination_slots = (
        state_destination_i32[:, None] * local_layers.numel() + local_layers[None, :]
    )
    buffers.conv_slots[0, :, :logical_m].copy_(source_slots.T)
    buffers.conv_slots[1, :, :logical_m].copy_(destination_slots.T)
    buffers.recurrent_slots[0, :, :logical_m].copy_(source_slots.T)
    buffers.recurrent_slots[1, :, :logical_m].copy_(destination_slots.T)
    if body.stage == 0:
        tokens = torch.zeros(captured_m, dtype=torch.int64, device=body.hidden.device)
        tokens[:logical_m] = torch.arange(
            17 + position,
            17 + position + logical_m,
            dtype=torch.int64,
            device=body.hidden.device,
        )
        body.fill_token_ids(tokens)


def _storage_poison_value(tensor: Tensor, value: int | float) -> int | float:
    """Express an audit pattern in its actual physical storage domain."""
    if tensor.dtype == torch.uint8:
        return int(value) % 256 if math.isfinite(value) else 255
    return value


def _set_padding_poison(body: Qwen3_5TP2PP2DecodeBody, poison: int, *, logical_m: int) -> None:
    if poison not in (0, 1):
        raise ValueError("padding poison must be zero or one")
    arena, buffers = body.arena, body.buffers
    captured_m = body.captured_m
    value = (37.0, -73.0)[poison]
    padding_m = captured_m - logical_m
    if padding_m:
        device = body.hidden.device
        local_layers = torch.arange(buffers.conv_slots.shape[1], dtype=torch.int32, device=device)
        if getattr(arena, "shared_storage", None) is not None:
            # prepare_scratch already installed this lease's actual folded
            # padding addresses. Physical row offsets are not storage slots.
            pages = buffers.page_destination[logical_m:].to(torch.int64)
            states = torch.div(buffers.conv_slots[0, 0, logical_m:],
                               local_layers.numel(), rounding_mode="floor").to(torch.int64)
            if (bool((pages < arena.spec.storage_page_capacity).any()) or
                    bool((states < arena.spec.state_slot_capacity).any())):
                raise ValueError("padding poison requires bound ownerless storage")
        else:
            offsets = torch.arange(logical_m, captured_m, dtype=torch.int64, device=device)
            pages = offsets + arena.spec.storage_page_capacity
            states = offsets + arena.spec.state_slot_capacity
        page_i32, state_i32 = pages.to(torch.int32), states.to(torch.int32)
        rows = slice(logical_m, captured_m)
        buffers.position_ids[:, rows].fill_(31 + 32 * poison)
        buffers.page_source[rows].copy_(page_i32)
        buffers.page_destination[rows].copy_(page_i32)
        buffers.token_offsets[rows].zero_()
        buffers.page_table[rows].copy_(page_i32[:, None].expand(-1, buffers.page_table.shape[1]))
        buffers.sequence_lengths[rows].fill_(1)
        buffers.segment_lengths[rows].zero_()
        buffers.row_activity[rows].zero_()
        aggregate = state_i32[:, None] * local_layers.numel() + local_layers[None, :]
        slots = aggregate.T.unsqueeze(0).expand(2, -1, -1)
        buffers.conv_slots[:, :, rows].copy_(slots)
        buffers.recurrent_slots[:, :, rows].copy_(slots)
        pages, states = torch.unique(pages), torch.unique(states)
        arena.key.index_fill_(0, pages, _storage_poison_value(arena.key, value))
        arena.value.index_fill_(
            0, pages, _storage_poison_value(arena.value, -value))
        arena.written.index_fill_(0, pages, True)
        arena.states[CONV_STATE_ID].index_fill_(0, states, value / 4)
        arena.states[RECURRENT_STATE_ID].index_fill_(0, states, value / 16)
        for name, state in arena.states.items():
            if name not in (CONV_STATE_ID, RECURRENT_STATE_ID):
                state.index_fill_(0, states, value / 8)
        for payload in getattr(arena, "page_fields", {}).values():
            payload.index_fill_(0, pages, float("nan") if payload.is_floating_point() else 31)
    if body.stage == 0:
        tokens = torch.zeros(captured_m, dtype=torch.int64, device=body.hidden.device)
        tokens[:logical_m] = torch.arange(
            17, 17 + logical_m, dtype=torch.int64, device=body.hidden.device
        )
        tokens[logical_m:] = torch.arange(
            1001 + 997 * poison,
            1001 + 997 * poison + captured_m - logical_m,
            dtype=torch.int64,
            device=body.hidden.device,
        )
        body.fill_token_ids(tokens)


def _reset_owner(body: ServingDecodeBody, table_pages: int) -> None:
    del table_pages
    body.buffers.segment_lengths.zero_()
    body.buffers.row_activity.zero_()
    owner_pages = 2 * body.captured_m
    body.arena.key[:owner_pages].zero_()
    body.arena.value[:owner_pages].zero_()
    body.arena.written[:owner_pages].zero_()
    for payload in getattr(body.arena, "page_fields", {}).values():
        payload[:owner_pages].zero_()
    for state in body.arena.states.values():
        count=min(2*body.captured_m,body.arena.spec.state_slot_capacity)
        if getattr(body.arena,"shared_storage",None) is None:state[:count].zero_()
        else:state[body.arena.spec.state_slot_capacity-count:body.arena.spec.state_slot_capacity].zero_()


def _prepare_audit(
    body: Qwen3_5TP2PP2DecodeBody,
    resource: RankReplayResources,
    *,
    poison: int,
    logical_m: int,
) -> None:
    body.prepare_scratch(resource)
    _set_active_rows(body, logical_m)
    _set_padding_poison(body, poison, logical_m=logical_m)
    torch.cuda.current_stream(body.hidden.device).synchronize()


def _active_snapshot(body: Qwen3_5TP2PP2DecodeBody, logical_m: int) -> dict[str, Tensor]:
    """Copy one boundary row; sweeping logical M covers every physical row."""

    buffers = body.buffers
    row = logical_m - 1
    owner_page = int(buffers.page_destination[row].item())
    local_layers = buffers.conv_slots.shape[1]
    owner_state = int(buffers.conv_slots[1, 0, row].item()) // local_layers
    values = {
        "hidden": body.hidden[row : row + 1],
        "kv_key": body.arena.key[owner_page : owner_page + 1],
        "kv_value": body.arena.value[owner_page : owner_page + 1],
        "kv_written": body.arena.written[owner_page : owner_page + 1],
        "conv": body.arena.states[CONV_STATE_ID][owner_state : owner_state + 1],
        "recurrent": body.arena.states[RECURRENT_STATE_ID][owner_state : owner_state + 1],
        "guard": body.buffers.guard,
    }
    if body.logits is not None:
        values["logits"] = body.logits[row : row + 1]
    values.update({f"state:{name}": value[owner_state:owner_state + 1]
                   for name, value in body.arena.states.items()
                   if name not in (CONV_STATE_ID, RECURRENT_STATE_ID)})
    values.update({f"page:{name}": value[owner_page:owner_page + 1]
                   for name, value in getattr(body.arena, "page_fields", {}).items()})
    return {name: value.detach().to("cpu", copy=True) for name, value in values.items()}


def _padding_snapshot(body: Qwen3_5TP2PP2DecodeBody, logical_m: int) -> dict[str, Tensor]:
    values = {"hidden": body.hidden[logical_m:]}
    if body.logits is not None:
        values["logits"] = body.logits[logical_m:]
    return {name: value.detach().to("cpu", copy=True) for name, value in values.items()}


def _snapshot_is_finite(value: Tensor) -> bool:
    """Classify immutable host snapshots without widening FP16/BF16 state."""
    if not value.is_floating_point():
        return True
    if value.device.type == "cpu" and value.dtype in (torch.float16, torch.bfloat16):
        exponent = 0x7C00 if value.dtype is torch.float16 else 0x7F80
        bits = value.detach().contiguous().view(torch.int16).numpy()
        return bool(np.all(np.bitwise_and(bits, exponent) != exponent))
    return bool(torch.isfinite(value).all())


def _exact_snapshot(left: dict[str, Tensor], right: dict[str, Tensor]) -> bool:
    return left.keys() == right.keys() and all(
        left[name].shape == right[name].shape
        and left[name].dtype is right[name].dtype
        and _snapshot_is_finite(right[name])
        and torch.equal(
            left[name].contiguous().view(torch.uint8),
            right[name].contiguous().view(torch.uint8),
        )
        for name in left
    )


def _cold_start_graph_audit(
    body: Qwen3_5TP2PP2DecodeBody,
    resource: RankReplayResources,
    slot: Qwen3_5TP2PP2GraphSlot,
    admission: _Admission,
    eager_snapshots: dict[int, dict[str, Tensor]],
    cost_samples: dict[int, list[float]] | None = None,
) -> str:
    slot.select_decode_overlap(False)
    baseline = _cold_start_graph_variant_audit(
        body, resource, slot, admission, eager_snapshots,
        cost_samples if slot._overlap is None else None,
    )
    if slot._overlap is None:
        return baseline
    slot.select_decode_overlap(True)
    candidate = _cold_start_graph_variant_audit(
        body, resource, slot, admission, eager_snapshots, cost_samples,
    )
    return stable_digest({
        "schema": "cubic.decode-overlap-audit.v1",
        "baseline": baseline, "candidate": candidate,
        "parts": tuple(child.captured_m for child in slot._overlap[2]),
    })


def _cold_start_graph_variant_audit(
    body: Qwen3_5TP2PP2DecodeBody,
    resource: RankReplayResources,
    slot: Qwen3_5TP2PP2GraphSlot,
    admission: _Admission,
    eager_snapshots: Mapping[int, dict[str, Tensor]],
    cost_samples: dict[int, list[float]] | None = None,
) -> str:
    captured_m = body.captured_m
    logical_m_values = tuple(body.bucket_logical_ms)
    domains: dict[int, dict[str, bool]] = {}
    audited_logical_ms = _bounded_audit_logical_ms(logical_m_values)
    replay_started = (
        torch.cuda.Event(enable_timing=True) if cost_samples is not None else None
    )
    replay_finished = (
        torch.cuda.Event(enable_timing=True) if cost_samples is not None else None
    )
    for logical_m in audited_logical_ms:
        eager = eager_snapshots[logical_m]

        _prepare_audit(body, resource, poison=0, logical_m=logical_m)
        if replay_started is not None:
            replay_started.record(slot.stream)
        slot.replay()
        if replay_finished is not None:
            replay_finished.record(slot.stream)
        slot.synchronize()
        if cost_samples is not None and replay_started is not None and replay_finished is not None:
            cost_samples.setdefault(captured_m, []).append(
                replay_started.elapsed_time(replay_finished)
            )
        barrier()
        graph_zero = _active_snapshot(body, logical_m)
        padding_zero = _padding_snapshot(body, logical_m)
        if not _exact_snapshot(eager, graph_zero):
            raise Qwen3_5TP2PP2ServingError(
                f"C{captured_m}/M{logical_m} graph differs from same-body eager"
            )

        _prepare_audit(body, resource, poison=1, logical_m=logical_m)
        if replay_started is not None:
            replay_started.record(slot.stream)
        slot.replay()
        if replay_finished is not None:
            replay_finished.record(slot.stream)
        slot.synchronize()
        if cost_samples is not None and replay_started is not None and replay_finished is not None:
            cost_samples.setdefault(captured_m, []).append(
                replay_started.elapsed_time(replay_finished)
            )
        barrier()
        graph_one = _active_snapshot(body, logical_m)
        padding_one = _padding_snapshot(body, logical_m)
        if not _exact_snapshot(graph_zero, graph_one):
            raise Qwen3_5TP2PP2ServingError(
                f"C{captured_m}/M{logical_m} active state depends on padding poison"
            )
        if logical_m < captured_m and (
            padding_zero.keys() != padding_one.keys()
            or not all(
                not torch.equal(padding_zero[name], padding_one[name]) for name in padding_zero
            )
        ):
            raise Qwen3_5TP2PP2ServingError(
                f"C{captured_m}/M{logical_m} padding poison missed a rank output"
            )
        domains[logical_m] = {
            "same_body_eager_exact": True,
            "padding_poison_independent": True,
            "physical_boundary_row_covered": True,
        }

    evidence_id = stable_digest(
        {
            "schema": "cubic.qwen3_5.tp2_pp2.bucket-cold-start.v1",
            "rank": body.rank,
            "inventory_digest": admission.inventory.inventory_digest,
            "semantic_plan_digest": admission.parallel_plan.semantic_plan_digest,
            "topology_digest": admission.sealed.topology_digest,
            "partition_digest": admission.sealed.partition_digest,
            "max_sequence_tokens": admission.max_sequence_tokens,
            "cache_page_capacity": body.arena.spec.page_capacity,
            "cache_state_slot_capacity": body.arena.spec.state_slot_capacity,
            "cache_scratch_capacity": body.arena.spec.scratch_capacity,
            "captured_m": captured_m,
            "logical_m_values": logical_m_values,
            "audited_logical_m_values": audited_logical_ms,
            "domains": domains,
        }
    )
    _reset_owner(body, body.buffers.page_table.shape[1])
    body.prepare_scratch(resource)
    torch.cuda.current_stream(body.hidden.device).synchronize()
    return evidence_id


class _CapturedGraphSlot(Protocol):
    """Rank-local captured graph surface required by the serving executor."""

    stream: Stream

    @property
    def graph(self) -> torch.cuda.CUDAGraph: ...

    @property
    def owned_decode_graph(self) -> torch.cuda.CUDAGraph: ...

    @property
    def logits(self) -> Tensor | None: ...

    @property
    def vocab_span(self) -> AxisSpan | None: ...

    def select_decode_overlap(self, independent_rows: bool) -> None: ...

    def replay(self) -> None: ...

    def synchronize(self) -> None: ...


def _batch_segments(rows: tuple[_BatchRow, ...]) -> tuple[tuple[int, int], ...]:
    try:
        return batch_segments(rows)
    except BatchProtocolError as error:
        raise Qwen3_5TP2PP2ServingError(str(error)) from error


@dataclass(frozen=True, slots=True)
class _RaggedSpeculativeSegment:
    """One request's compact target segment and its independent draft depth."""

    index: int
    start: int
    stop: int

    @property
    def draft_count(self) -> int:
        return self.stop - self.start - 1


def _ragged_speculative_plan(
    rows: tuple[_BatchRow, ...],
) -> tuple[
    tuple[_RaggedSpeculativeSegment, ...],
    tuple[tuple[_RaggedSpeculativeSegment, ...], ...],
]:
    """Build compact per-depth proposal waves over one packed target batch."""

    boundaries = tuple(
        (start, stop)
        for start, stop in _batch_segments(rows)
        if rows[start].is_decode and stop - start > 1
    )
    speculative = tuple(
        _RaggedSpeculativeSegment(index, start, stop)
        for index, (start, stop) in enumerate(boundaries)
    )
    max_depth = max((segment.draft_count for segment in speculative), default=0)
    levels = tuple(
        tuple(segment for segment in speculative if segment.draft_count > depth)
        for depth in range(max_depth)
    )
    return speculative, levels


def _ragged_acceptance_rows(
    speculative: tuple[_RaggedSpeculativeSegment, ...],
    levels: tuple[tuple[_RaggedSpeculativeSegment, ...], ...],
    decisions: tuple[tuple[tuple[int, int], ...], ...],
) -> tuple[tuple[int, int], ...]:
    """Reduce compact depth decisions back to independent request commits."""

    if len(decisions) != len(levels):
        raise Qwen3_5TP2PP2ServingError("MTP decision depth differs from its ragged plan")
    by_segment: dict[int, list[tuple[int, int]]] = {
        segment.index: [] for segment in speculative
    }
    for level, decision_row in zip(levels, decisions, strict=True):
        if len(level) != len(decision_row):
            raise Qwen3_5TP2PP2ServingError("MTP decision rows differ from their ragged level")
        for segment, decision in zip(level, decision_row, strict=True):
            by_segment[segment.index].append(decision)

    reduced: list[tuple[int, int]] = []
    for segment in speculative:
        segment_decisions = by_segment[segment.index]
        if len(segment_decisions) != segment.draft_count:
            raise Qwen3_5TP2PP2ServingError("MTP request decision chain is incomplete")
        accepted_count = 0
        next_token = -1
        for accepted, candidate in segment_decisions:
            if accepted not in (0, 1) or candidate < 0:
                raise Qwen3_5TP2PP2ServingError(
                    "MTP verification produced an invalid decision"
                )
            next_token = int(candidate)
            if not accepted:
                break
            accepted_count += 1
        reduced.append((accepted_count, next_token))
    return tuple(reduced)


def _repair_speculative_state(
    executor: Any,
    target_hidden_slots: Tensor,
    rows: tuple[_BatchRow, ...],
    segments: tuple[_RaggedSpeculativeSegment, ...],
    accepted_counts: tuple[int, ...],
    *, draft_state_fields: Mapping[str, Tensor] | None = None,
) -> None:
    """Commit partially accepted recurrent checkpoints into durable owner slots."""

    bindings = getattr(executor, "_state_transfer_bindings", None)
    identity = id(draft_state_fields) if draft_state_fields else 0
    if bindings is not None and identity not in bindings:
        raise Qwen3_5TP2PP2ServingError("state repair differs from its bound provider storage")
    plan = None if bindings is None else bindings[identity]
    if plan is not None:
        if not execution_ready(plan):
            raise Qwen3_5TP2PP2ServingError("state repair has no byte qualification")
        if (target_hidden_slots is not executor._state_transfer_hidden_slots
                or executor.body.arena.states is not executor._state_transfer_state_fields):
            raise Qwen3_5TP2PP2ServingError("state repair differs from its bound target storage")
        pairs = []
        for segment, accepted in zip(segments, accepted_counts, strict=True):
            if not 0 <= accepted <= segment.draft_count:
                raise Qwen3_5TP2PP2ServingError("state repair accepted prefix is invalid")
            if accepted == segment.draft_count:
                continue
            source = rows[segment.start + accepted].state_destination_slot
            destination = rows[segment.stop - 1].state_destination_slot
            if source != destination:
                pairs.append((source, destination))
        plan.run(tuple(pairs))
        return

    for segment, accepted in zip(segments, accepted_counts, strict=True):
        if accepted == segment.draft_count:
            continue
        source = rows[segment.start + accepted].state_destination_slot
        destination = rows[segment.stop - 1].state_destination_slot
        if source == destination:
            continue
        for state in executor.body.arena.states.values():
            state[destination].copy_(state[source])
        if draft_state_fields is not None:
            for state in draft_state_fields.values():
                state[destination].copy_(state[source])
        target_hidden_slots[destination].copy_(target_hidden_slots[source])


def _ragged_proposal_buckets(
    levels: tuple[tuple[_RaggedSpeculativeSegment, ...], ...],
    capacities: tuple[int, ...] = _PHYSICAL_GRAPH_BUCKET_CAPACITIES,
) -> tuple[int, ...]:
    """Select the smallest qualified physical proposal graph at every depth."""

    return tuple(_bucket_for_logical_m(len(level), capacities) for level in levels)


def _mtp_maintenance_pairs(
    rows: tuple[_BatchRow, ...],
    speculative: tuple[_RaggedSpeculativeSegment, ...],
) -> tuple[tuple[int, _BatchRow], ...]:
    """Return prefill rows that establish MTP state before decode policy starts."""

    speculative_rows = {
        physical
        for segment in speculative
        for physical in range(segment.start, segment.stop)
    }
    return tuple(
        (physical, row)
        for physical, row in enumerate(rows)
        if physical not in speculative_rows and not row.is_decode and row.mtp_position >= 0
    )


def _cache_take(free):
    return heapq.heappop(free) if isinstance(free,list) else free.allocate()


def _cache_return(free,index):
    if isinstance(free,list):heapq.heappush(free,index)
    else:free.release(index)


def _serialize_worker_command(command: _WorkerCommand) -> bytes:
    try:
        return serialize_worker_command(command)
    except BatchProtocolError as error:
        raise Qwen3_5TP2PP2ServingError(str(error)) from error


def _deserialize_worker_command(payload: bytes) -> _WorkerCommand:
    try:
        return deserialize_worker_command(payload)
    except BatchProtocolError as error:
        raise Qwen3_5TP2PP2ServingError(str(error)) from error


class _SamplingState:
    """Caller-owned fixed-capacity TP candidate sampling resources."""

    _MAX_TOP_K = 20
    _constraint_arena: _ConstraintSamplingArena
    _constraint_mask: Tensor
    _constraint_activity: Tensor

    def __init__(
        self,
        capacity: int,
        *,
        tp_size: int,
        device: torch.device,
        final_rank: bool,
        sampling_root: bool,
        token_ids: Tensor,
        score_dtype: torch.dtype = torch.bfloat16,
        asynchronous_bind: bool = False,
    ) -> None:
        self.temperatures = torch.empty(capacity, dtype=torch.float32, device=device)
        self.top_ks = torch.empty(capacity, dtype=torch.int64, device=device)
        self.top_ps = torch.empty(capacity, dtype=torch.float32, device=device)
        self.seeds = torch.empty(capacity, dtype=torch.int64, device=device)
        self.ordinals = torch.empty(capacity, dtype=torch.int64, device=device)
        self.parameters = ShardedSamplingParameters(
            self.temperatures,
            self.top_ks,
            self.top_ps,
            self._MAX_TOP_K,
            self.seeds,
            self.ordinals,
        )
        self.workspace = (
            ShardedSamplingWorkspace(
                torch.empty(
                    (capacity, self._MAX_TOP_K),
                    dtype=score_dtype,
                    device=device,
                ),
                torch.empty((capacity, self._MAX_TOP_K), dtype=torch.int64, device=device),
                torch.empty(
                    (tp_size, capacity, self._MAX_TOP_K),
                    dtype=score_dtype,
                    device=device,
                ),
                torch.empty(
                    (tp_size, capacity, self._MAX_TOP_K),
                    dtype=torch.int64,
                    device=device,
                ),
            )
            if final_rank
            else None
        )
        self.root_valid = (
            torch.empty(capacity, dtype=torch.bool, device=device) if sampling_root else None
        )
        self.root_sampler = (
            RootTopKTopPSamplingPrimitive(
                self.workspace.gathered_scores,
                self.workspace.gathered_token_ids,
                self.temperatures,
                self.top_ps,
                self.top_ks,
                self.seeds,
                self.ordinals,
                token_ids,
                self.root_valid,
            )
            if sampling_root and self.workspace is not None and self.root_valid is not None
            else None
        )
        self.greedy_plans: list[CapturedGreedySampler] = []
        self.masked_plans: list[CapturedGreedySampler] = []
        self.constraint_bound = False
        self.all_greedy = False
        self._temperature_host = torch.empty(capacity, dtype=torch.float32, pin_memory=True)
        self._top_k_host = torch.empty(capacity, dtype=torch.int64, pin_memory=True)
        self._top_p_host = torch.empty(capacity, dtype=torch.float32, pin_memory=True)
        self._seed_host = torch.empty(capacity, dtype=torch.int64, pin_memory=True)
        self._ordinal_host = torch.empty(capacity, dtype=torch.int64, pin_memory=True)
        self._temperature_host_array = self._temperature_host.numpy()
        self._top_k_host_array = self._top_k_host.numpy()
        self._top_p_host_array = self._top_p_host.numpy()
        self._seed_host_array = self._seed_host.numpy()
        self._ordinal_host_array = self._ordinal_host.numpy()
        self._parameter_hosts = None
        if asynchronous_bind:
            first = (self._temperature_host, self._top_k_host, self._top_p_host,
                     self._seed_host, self._ordinal_host)
            self._parameter_hosts = (first, tuple(torch.empty_like(t, pin_memory=True)
                                                 for t in first))
            self._parameter_events = tuple(torch.cuda.Event() for _ in self._parameter_hosts)
            self._parameter_pending = [False] * len(self._parameter_hosts)
            self._next_parameter_host = 0

    def greedy_plan(self, logits: Tensor | None) -> CapturedGreedySampler | None:
        return next((plan for plan in self.greedy_plans if plan.local_logits is logits), None)

    def bound_plan(self, logits: Tensor | None) -> CapturedGreedySampler | None:
        if not self.constraint_bound:
            return self.greedy_plan(logits)
        return next((plan for plan in self.masked_plans if plan.local_logits is logits), None)

    def bind_constraints(self, masks: list[Tensor | None], owners: list[bool]) -> None:
        if not self.masked_plans or not 0 < len(masks) <= self.temperatures.numel():
            raise Qwen3_5TP2PP2ServingError("grammar mask exceeds qualified sampling capacity")
        if any(
            plan.token_mask is not self._constraint_mask
            or plan.constrained_rows is not self._constraint_activity
            for plan in self.masked_plans
        ):
            raise Qwen3_5TP2PP2ServingError("grammar operands changed after graph qualification")
        self._constraint_arena.bind(masks, owners)
        self.constraint_bound = True

    def bind(
        self,
        rows: tuple[_BatchRow, ...],
    ) -> bool:
        if self._parameter_hosts is not None:
            host_slot = self._next_parameter_host
            if self._parameter_pending[host_slot]:
                self._parameter_events[host_slot].synchronize()
            (self._temperature_host, self._top_k_host, self._top_p_host,
             self._seed_host, self._ordinal_host) = self._parameter_hosts[host_slot]
            (self._temperature_host_array, self._top_k_host_array, self._top_p_host_array,
             self._seed_host_array, self._ordinal_host_array) = tuple(
                t.numpy() for t in self._parameter_hosts[host_slot])
        count = len(rows)
        temperatures = np.fromiter(
            (row.sampling.temperature for row in rows), dtype=np.float32, count=count
        )
        self._temperature_host_array[:count] = temperatures
        self._top_k_host_array[:count] = np.fromiter(
            (row.sampling.top_k for row in rows), dtype=np.int64, count=count
        )
        self._top_p_host_array[:count] = np.fromiter(
            (row.sampling.top_p for row in rows), dtype=np.float32, count=count
        )
        self._seed_host_array[:count] = np.fromiter(
            (row.sampling.seed for row in rows), dtype=np.int64, count=count
        )
        self._ordinal_host_array[:count] = np.fromiter(
            (row.sample_ordinal + 1 for row in rows), dtype=np.int64, count=count
        )
        self.temperatures[:count].copy_(self._temperature_host[:count], non_blocking=True)
        self.top_ks[:count].copy_(self._top_k_host[:count], non_blocking=True)
        self.top_ps[:count].copy_(self._top_p_host[:count], non_blocking=True)
        self.seeds[:count].copy_(self._seed_host[:count], non_blocking=True)
        self.ordinals[:count].copy_(self._ordinal_host[:count], non_blocking=True)
        if self._parameter_hosts is not None:
            # Protect pinned bytes until their H2D reads finish. Deferred
            # proposal guards no longer provide a per-depth host barrier.
            self._parameter_events[host_slot].record(
                torch.cuda.current_stream(self.temperatures.device))
            self._parameter_pending[host_slot] = True
            self._next_parameter_host = (host_slot + 1) % len(self._parameter_hosts)
        self.all_greedy = bool(np.all(temperatures == 0.0))
        return bool(np.all(temperatures > 0.0))


class _ConstraintSamplingArena:
    """Rank-local scratch under the existing exclusive topology replay lease.

    No candidate result, proposal probability/history, or request matcher lives
    here. A masked-logits producer is consumed by top-k in the same graph;
    captured allocation-pool temporaries also have no live cross-graph output.
    All masked graph captures use one stream/pool. Runtime execution is serial
    under the lease, so buckets and target/provider phases may reuse scratch in
    arbitrary order. Addresses and capacity are fixed before the first capture.
    """

    def __init__(self, executor: Any, capacity: int, *, qualification=None,
                 consume_logits: bool = False) -> None:
        if capacity < executor.captured_m:
            raise Qwen3_5TP2PP2ServingError("grammar arena is smaller than its graph bucket")
        self.device = executor._sampling.temperatures.device
        self.mask = torch.full(
            (capacity, (executor.vocab_size + 31) // 32), -1,
            dtype=torch.int32, device=self.device,
        )
        self.activity = torch.zeros(capacity, dtype=torch.bool, device=self.device)
        logits, span = executor.slot.logits, executor.slot.vocab_span
        self.logits_width = logits.shape[1] if logits is not None else 0
        self.logits_dtype = None if logits is None else logits.dtype
        self.consume_logits = consume_logits
        self.visible_width = (max(0, min(span.stop, executor.vocab_size) - span.start)
                              if span is not None else 0)
        self.masked_logits = (
            torch.empty((capacity, self.visible_width), dtype=logits.dtype, device=self.device)
            if logits is not None and self.visible_width and not consume_logits else None
        )
        self.hosts = tuple(torch.empty_like(self.mask, device="cpu", pin_memory=True)
                           for _ in range(2))
        self.activity_hosts = tuple(torch.empty_like(self.activity, device="cpu", pin_memory=True)
                                    for _ in range(2))
        self.events = tuple(torch.cuda.Event() for _ in range(2))
        self.pending = [False, False]
        self.next_host = 0
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.capture_stream = executor.slot.stream
        self.graph_pool = torch.cuda.graph_pool_handle()
        self.budget = None
        boundaries = getattr(executor, "_reasoning_token_boundaries", ())
        if boundaries:
            from cubic_serve.ops.cuda.sampling.reasoning_budget import ReasoningBudgetProgram
            self.budget = ReasoningBudgetProgram(
                boundaries, self.mask, self.activity,
                stop_token_ids=getattr(executor, "_constraint_stop_token_ids", ()),
            )
            if qualification is None:
                self.budget.qualify(executor.vocab_size)
            else:
                qualification.component(
                    self.budget, (boundaries, self.budget.stop_token_ids,
                                  capacity, executor.vocab_size), args=(executor.vocab_size,))
            diagnostic_event(_LOGGER, 'performance.reasoning_budget_scratch_rank_d_device_f',
                executor._rank,
                sum(t.numel() * t.element_size() for t in
                    (self.budget.state, self.budget.scratch, self.budget.markers,
                     self.budget.offsets)) / (1 << 20),
                self.budget.host.numel() * self.budget.host.element_size() / (1 << 20),
            )
        diagnostic_event(_LOGGER, 'performance.grammar_sampling_scratch_rank_d_capacity_d',
            executor._rank,
            capacity,
            sum(t.numel() * t.element_size() for t in
                (self.mask, self.activity, *((self.masked_logits,)
                                            if self.masked_logits is not None else ()))) / (1 << 20),
            sum(t.numel() * t.element_size() for t in (*self.hosts, *self.activity_hosts)) / (1 << 20),
        )

    def logits_view(self, logits: Tensor | None) -> Tensor | None:
        if logits is None or not self.visible_width:
            return None
        if self.consume_logits:
            if (logits.device != self.device or logits.dtype != self.logits_dtype
                    or logits.ndim != 2 or logits.shape[0] > self.mask.shape[0]
                    or logits.shape[1] != self.logits_width):
                raise Qwen3_5TP2PP2ServingError("terminal logits exceed qualified geometry")
            return logits[:, :self.visible_width]
        scratch = self.masked_logits
        if (scratch is None or logits.device != scratch.device or logits.dtype != scratch.dtype
                or logits.ndim != 2 or logits.shape[0] > scratch.shape[0]
                or logits.shape[1] != self.logits_width):
            raise Qwen3_5TP2PP2ServingError("grammar logits exceed qualified scratch geometry")
        return scratch[:logits.shape[0]]

    def bind(self, masks: list[Tensor | None], owners: list[bool]) -> None:
        count = len(masks)
        if not 0 < count <= len(self.mask) or len(owners) != count:
            raise Qwen3_5TP2PP2ServingError("grammar mask and row ownership differ")
        slot = self.next_host
        if self.pending[slot] and not self.events[slot].query():
            self.events[slot].synchronize()
        host, activity = self.hosts[slot], self.activity_hosts[slot]
        host[:count].fill_(-1)
        activity[:count].copy_(torch.tensor(owners, dtype=torch.bool))
        for row, mask in enumerate(masks):
            if mask is not None:
                if (mask.device.type != "cpu" or mask.dtype != torch.int32
                        or mask.shape != host[row].shape):
                    raise Qwen3_5TP2PP2ServingError("grammar mask differs from qualified vocabulary")
                host[row].copy_(mask)
        current = torch.cuda.current_stream(self.device)
        # Shared destination: wait for preceding graph reads before overwriting.
        # The host planes remain double buffered until their H2D event completes.
        self.copy_stream.wait_stream(current)
        with torch.cuda.stream(self.copy_stream):
            self.mask[:count].copy_(host[:count], non_blocking=True)
            self.activity[:count].copy_(activity[:count], non_blocking=True)
            self.events[slot].record(self.copy_stream)
        current.wait_event(self.events[slot])
        self.pending[slot] = True
        self.next_host = (slot + 1) % len(self.hosts)


def _prepare_rank_greedy_sampling(
    executor: Any, lifetime: ExitStack, *, constraint_arena: _ConstraintSamplingArena,
    qualify_resident: bool = True, qualification=None,
) -> None:
    """Own the sampler portfolio; qualify resident composition on final arenas.

    Memory-pricing and fixed-address bootstrap graphs allocate the same
    resources but never grant readiness. A retained fixed-address instance
    completes its resident checks after physical KV admission; a temporary
    pricing instance is destroyed. Final instances run their complete checks.
    """
    sources = [executor.slot.logits]
    if hasattr(executor, "_logical_logits"):
        sources.append(executor._logical_logits)
    for logits in sources:
        plan = CapturedGreedySampler(
            local_logits=logits, local_vocab_span=executor.slot.vocab_span,
            final_ranks=executor._final_ranks, final_vocab_spans=executor._final_spans,
            true_vocab_size=executor.vocab_size, token_ids=executor._token_ids,
            parameters=executor._sampling.parameters, workspace=executor._sampling.workspace,
            final_tp_group=executor._final_tp_group, world_group=executor._world_group,
            root_valid=executor._sampling.root_valid,
            capture_stream=executor.slot.stream,
        )
        lifetime.callback(plan.close)
        if qualification is None:
            plan.qualify()
        else:
            qualification.component(plan, tuple(call.signature for call in plan.calls),
                                    attributes=("qualified", "evidence"))
        executor._sampling.greedy_plans.append(plan)
        startup_sampling_qualified(
            executor._rank, executor.captured_m, len(plan.evidence.get("cases", ())), masked=False,
        )
    sampling = executor._sampling
    sampling._constraint_arena = constraint_arena
    sampling._constraint_mask = constraint_arena.mask[:executor.captured_m]
    sampling._constraint_activity = constraint_arena.activity[:executor.captured_m]
    for logits in sources:
        masked = CapturedGreedySampler(
            local_logits=logits,
            local_vocab_span=executor.slot.vocab_span,
            final_ranks=executor._final_ranks,
            final_vocab_spans=executor._final_spans,
            true_vocab_size=executor.vocab_size,
            token_ids=executor._token_ids,
            parameters=sampling.parameters,
            workspace=sampling.workspace,
            final_tp_group=executor._final_tp_group,
            world_group=executor._world_group,
            root_valid=sampling.root_valid,
            capture_stream=constraint_arena.capture_stream,
            token_mask=sampling._constraint_mask,
            constrained_rows=sampling._constraint_activity,
            masked_logits=constraint_arena.logits_view(logits),
            graph_pool=constraint_arena.graph_pool,
            consume_logits=getattr(constraint_arena, "consume_logits", False),
        )
        lifetime.callback(masked.close)
        if qualification is None:
            masked.qualify()
        else:
            qualification.component(masked, tuple(call.signature for call in masked.calls),
                                    attributes=("qualified", "evidence"))
        sampling.masked_plans.append(masked)
        startup_sampling_qualified(
            executor._rank, executor.captured_m, len(masked.evidence.get("cases", ())), masked=True,
        )
    if isinstance(executor, _CapturedGraphRank) and executor._target_hidden_slots is None:
        from cubic_serve.executor.resident_window import ResidentDecodeWindow, SharedResidentJournal

        journal = None
        if executor._rank == executor._sampler_rank:
            elements = 64 + RESIDENT_STEP_CAPACITY * (executor.captured_m + 3)
            journal = SharedResidentJournal(elements, executor.body.hidden.device)
            lifetime.callback(journal.close)
        window = ResidentDecodeWindow(
            executor, executor._sampling.greedy_plan(executor.slot.logits), journal=journal,
        )
        lifetime.callback(window.close)
        if qualify_resident:
            if qualification is None:
                window.qualify()
            else:
                qualification.window(window)
        executor._resident_window = window
        stochastic_window = ResidentDecodeWindow(
            executor, executor._sampling.greedy_plan(executor.slot.logits),
            journal=journal, stochastic=True,
        )
        lifetime.callback(stochastic_window.close)
        if qualify_resident:
            if qualification is None:
                stochastic_window.qualify()
            else:
                qualification.window(stochastic_window)
        executor._resident_stochastic_window = stochastic_window


class _CapturedGraphRank:
    _block_size = _BLOCK_SIZE

    def __init__(
        self,
        body: ServingDecodeBody,
        resource: RankReplayResources,
        slot: _CapturedGraphSlot,
        runtime: Any,
        admission: _Admission,
        target_hidden_slots: Tensor | None = None,
        *, private_state_sources: bool = False,
    ) -> None:
        self._block_size = getattr(admission, "block_size_tokens", _BLOCK_SIZE)
        self._reasoning_token_boundaries = getattr(admission, "reasoning_token_boundaries", ())
        self._constraint_stop_token_ids = getattr(admission, "constraint_stop_token_ids", ())
        if body.arena.layout.block_size_tokens != self._block_size:
            raise Qwen3_5TP2PP2ServingError(
                "decode body KV page size differs from serving metadata"
            )
        self.body, self.resource, self.slot = body, resource, slot
        self._private_state_sources = private_state_sources
        self._reserved_scratch_states: frozenset[int] = frozenset()
        if (private_state_sources
                and body.arena.spec.scratch_capacity < 2 * body.captured_m - 1):
            raise Qwen3_5TP2PP2ServingError(
                "draft scratch cannot cover live checkpoints and graph padding"
            )
        self.vocab_size = builtin_text_config(admission.config).vocabulary_size
        self.max_sequence_tokens = admission.max_sequence_tokens
        self.captured_m = body.captured_m
        self._table_pages = body.buffers.page_table.shape[1]
        self._rank = body.rank
        self._world_group = cast(ProcessGroup, dist.group.WORLD)
        self._final_spans = _final_vocab_spans(admission.parallel_plan)
        self._final_ranks = _final_ranks(admission.parallel_plan)
        self._sampler_rank = _sampler_rank(admission.parallel_plan)
        final_stage = admission.parallel_plan.topology.pp_size - 1
        self._final_tp_group = (
            runtime._groups[f"tp:{final_stage}"] if self._rank in self._final_ranks else None
        )
        self._page_ids = torch.arange(
            self._table_pages, dtype=torch.int32, device=body.hidden.device
        )
        state_maps = body.buffers.state_slot_maps
        if not state_maps or any(
            value.dtype is not torch.int32
            or value.device != body.hidden.device
            or value.ndim != 3
            or value.shape[0] != 2
            or value.shape[1] <= 0
            or value.shape[2] != self.captured_m
            or value.shape != state_maps[0].shape
            or not value.is_contiguous()
            for value in state_maps
        ):
            raise Qwen3_5TP2PP2ServingError("state slot maps differ from the serving ownership ABI")
        self._local_state_layers = torch.arange(
            body.buffers.state_slot_maps[0].shape[1],
            dtype=torch.int32,
            device=body.hidden.device,
        )
        self._scratch_pages = body.arena.padding_page_slots(
            self.captured_m, resource.scratch_arena_id,
        )
        self._scratch_states = body.arena.padding_state_slots(self.captured_m, resource.scratch_arena_id)
        # Startup witnesses may leave any physical row pointing at live owners.
        # The first bind must rewrite every row, including inactive destinations:
        # a zero activity flag does not mask every fixed-shape cache writer.
        # Subsequent binds can transfer only the previous/current active prefix.
        self._bound_active_rows = self.captured_m
        # Captured capacity is address space, not executable GDN work.  Active
        # rows are installed by the first bulk metadata bind; ownerless rows
        # remain zero so the graph's segment-aware kernels return immediately.
        body.buffers.segment_lengths.zero_()
        body.buffers.row_activity.zero_()
        self._token_ids = torch.empty(self.captured_m, dtype=torch.int64, device=body.hidden.device)
        self._sampling = _SamplingState(
            self.captured_m,
            tp_size=len(self._final_ranks),
            device=body.hidden.device,
            final_rank=self._rank in self._final_ranks,
            sampling_root=self._rank == self._sampler_rank,
            token_ids=self._token_ids,
            score_dtype=body.hidden.dtype,
        )
        self._sampling_bound_all_random: bool | None = None
        self._continuation_updates_host = torch.empty(
            (self.captured_m, 3), dtype=torch.int32, device="cpu", pin_memory=True
        )
        self._continuation_updates_host_array = self._continuation_updates_host.numpy()
        self._continuation_updates_device = torch.empty(
            (self.captured_m, 3), dtype=torch.int32, device=body.hidden.device
        )
        self._local_max_scores = (
            torch.empty(
                (self.captured_m,),
                dtype=body.hidden.dtype,
                device=body.hidden.device,
            )
            if self._rank in self._final_ranks
            else None
        )
        self._gathered_max_scores = (
            torch.empty(
                (len(self._final_ranks), self.captured_m),
                dtype=body.hidden.dtype,
                device=body.hidden.device,
            )
            if self._rank in self._final_ranks
            else None
        )
        self._gathered_token_ids = (
            torch.empty(
                (len(self._final_ranks), self.captured_m),
                dtype=torch.int64,
                device=body.hidden.device,
            )
            if self._rank in self._final_ranks
            else None
        )
        # Row assembly stays on the CPU.  One merged async transfer feeds fixed
        # device staging, then a constant number of D2D copies binds the graph
        # inputs.  A two-slot pinned ring prevents the CPU from rewriting a
        # buffer whose prior H2D is still in flight.
        local_layers = int(self._local_state_layers.numel())
        self._state_destination_column = 8
        base_columns = 9 + 2 * local_layers + self._table_pages
        self._position_axes_slice = slice(base_columns, base_columns + 3)
        self._metadata_columns = base_columns + 3
        self._source_slots_slice = slice(9, 9 + local_layers)
        self._destination_slots_slice = slice(9 + local_layers, 9 + 2 * local_layers)
        self._page_table_slice = slice(9 + 2 * local_layers, base_columns)
        self._metadata_hosts = tuple(
            torch.empty(
                (self.captured_m, self._metadata_columns),
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(2)
        )
        self._metadata_host_arrays = tuple(item.numpy() for item in self._metadata_hosts)
        self._local_slot_offsets = np.arange(local_layers, dtype=np.int32)
        self._feature_hosts = tuple(
            torch.zeros(
                (self.captured_m, getattr(body, "feature_width", body.hidden.shape[1])),
                dtype=body.hidden.dtype,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(2)
        )
        self._features_bound = False
        self._staged_copy_features = False
        self._staged_has_features = False
        self._staged_page_table_pages = 1
        self._metadata_device = torch.empty(
            (self.captured_m, self._metadata_columns),
            dtype=torch.int32,
            device=body.hidden.device,
        )
        self._state_destinations = torch.empty(
            self.captured_m,
            dtype=torch.int64,
            device=body.hidden.device,
        )
        self._index_page_offset = self.captured_m
        self._index_capacity = self.captured_m * (self._table_pages + 1)
        self._index_hosts = tuple(
            torch.empty(
                self._index_capacity,
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(2)
        )
        self._index_host_arrays = tuple(item.numpy() for item in self._index_hosts)
        self._index_device = torch.empty(
            self._index_capacity,
            dtype=torch.int64,
            device=body.hidden.device,
        )
        self._host_events = tuple(
            torch.cuda.Event()  # type: ignore[no-untyped-call]
            for _ in range(2)
        )
        self._host_event_recorded = [False, False]
        self._next_host_slot = 0
        self._replay_started = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        self._replay_finished = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        self._replay_timing_pending = False
        self._feeding_profiler: _CudaStepProfiler | None = None
        self._batch_bound = False
        self._batch_phase_failed = False
        self._target_hidden_slots = target_hidden_slots

        self._input_update_graph = self._capture_input_update()
        from cubic_serve.ops.cuda.pointwise.qwen_decode import update_resident_pages

        self._update_resident_pages = update_resident_pages
        buffers = self.body.buffers
        update_resident_pages(
            self._continuation_updates_device, self.body.arena.key, self.body.arena.value,
            self.body.arena.written, buffers.token_offsets, buffers.page_source,
            buffers.page_destination, buffers.page_table, compile_only=True,
            page_fields=tuple(self.body.arena.page_fields.values()),
        )

    def _capture_input_update(self) -> torch.cuda.CUDAGraph:
        """Capture the bucket's owned input preparation before readiness."""
        from cubic_serve.ops.cuda.pointwise.qwen_decode import advance_resident_inputs

        buffers = self.body.buffers
        first = self.body.stage == 0
        span = (
            getattr(self.body, "embedding_input_span", self.body.embedding_span) if first else None
        )
        mask = self.body.embedding_mask if first else None
        if first and (span is None or mask is None):
            raise Qwen3_5TP2PP2ServingError("first stage has no embedding shard")
        changed = [
            buffers.position_ids, buffers.token_offsets, buffers.sequence_lengths,
            buffers.token_ids, self._sampling.ordinals,
        ]
        if mask is not None:
            changed.append(mask)

        def launch() -> None:
            advance_resident_inputs(
                buffers.position_ids, buffers.token_offsets, buffers.sequence_lengths,
                self._token_ids, buffers.token_ids, self._sampling.ordinals,
                buffers.row_activity, mask,
                vocab_start=span.start if span is not None else 0,
                vocab_stop=span.stop if span is not None else 0,
            )

        with torch.cuda.device(self.body.hidden.device):
            snapshots = [tensor.clone() for tensor in changed]
            try:
                launch()
                torch.cuda.synchronize(self.body.hidden.device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self.slot.stream):
                    launch()
            finally:
                for tensor, snapshot in zip(changed, snapshots, strict=True):
                    tensor.copy_(snapshot)
                torch.cuda.current_stream(self.body.hidden.device).synchronize()
        return graph

    def enable_feeding_profile(self, aggregate: _StepFeedingProfile) -> None:
        """Enable diagnostics after cold-start qualification has completed."""

        if self._feeding_profiler is not None:
            raise Qwen3_5TP2PP2ServingError("feeding profile is already enabled")
        self._feeding_profiler = _CudaStepProfiler(
            aggregate, self.body.hidden.device, _CUDA_PROFILE_METRICS
        )

    def reset(self) -> None:
        self.body.set_active_rows(1)
        self.body.prepare_scratch(self.resource)
        self._features_bound = False
        _reset_owner(self.body, self._table_pages)
        buffers = self.body.buffers
        buffers.page_table[0].copy_(self._page_ids)
        buffers.segment_lengths[0].fill_(1)
        buffers.row_activity[0].fill_(1)
        for slots in buffers.state_slot_maps:
            slots[:, :, 0].copy_(self._local_state_layers.expand(2, -1))
        torch.cuda.current_stream(self.body.hidden.device).synchronize()

    def replay(self, token_id: int, position: int) -> None:
        if not 0 <= position < self.max_sequence_tokens:
            raise Qwen3_5TP2PP2ServingError("request exceeds captured context capacity")
        buffers = self.body.buffers
        page = position // self._block_size
        buffers.position_ids[:, 0].fill_(position)
        buffers.page_source[0].fill_(page)
        buffers.page_destination[0].fill_(page)
        buffers.token_offsets[0].fill_(position % self._block_size)
        buffers.sequence_lengths[0].fill_(position + 1)
        if self.body.stage == 0:
            span = getattr(self.body, "embedding_input_span", self.body.embedding_span)
            mask = self.body.embedding_mask
            if span is None or mask is None:
                raise Qwen3_5TP2PP2ServingError("first stage has no embedding shard")
            owned = span.start <= token_id < span.stop
            buffers.token_ids[0].fill_(token_id - span.start if owned else 0)
            mask[0].fill_(owned)
        current = torch.cuda.current_stream(self.body.hidden.device)
        self.slot.stream.wait_stream(current)
        self.slot.replay()
        current.wait_stream(self.slot.stream)

    def _stage_batch_rows(
        self,
        rows: tuple[_BatchRow, ...],
        *,
        host_slot: int,
    ) -> tuple[int, int, int]:
        """Validate rows and assemble host metadata without launching GPU work."""

        if not 0 < len(rows) <= self.captured_m:
            raise Qwen3_5TP2PP2ServingError("batch rows exceed captured M")
        arena = self.body.arena
        if (getattr(arena,"shared_storage",None) is not None and any(
                not 0<=slot<arena.spec.state_slot_capacity for row in rows
                for slot in (row.state_source_slot,row.state_destination_slot))):
            raise Qwen3_5TP2PP2ServingError("owned row references a padding-only state")
        metadata = self._metadata_host_arrays[host_slot]
        indices = self._index_host_arrays[host_slot]
        local_layers = int(self._local_state_layers.numel())
        transfer_rows = max(len(rows), self._bound_active_rows)
        reset_count = 0
        new_page_count = 0
        position_axes_slice = getattr(self, "_position_axes_slice", None)
        feature_hosts = getattr(self, "_feature_hosts", None)
        has_features = feature_hosts is not None and any(row.input_feature for row in rows)
        copy_features = getattr(self, "_features_bound", False) or has_features
        _batch_segments(rows)
        page_table_pages = max(1, max(len(row.page_table) for row in rows))
        steady_decode = len(rows) >= self._bound_active_rows and all(
            row.is_decode
            and row.segment_length == 1
            and not row.reset_state
            and not row.new_pages
            and row.state_source_slot == row.state_destination_slot
            and row.page_source == row.page_destination
            and not row.input_feature
            for row in rows
        )
        if steady_decode:
            packed_rows: list[tuple[int, ...]] = []
            page_tables: list[tuple[int, ...]] = []
            position_axes: list[tuple[int, int, int]] = []
            uniform_page_tables = all(
                len(row.page_table) == page_table_pages for row in rows
            )
            span = (
                getattr(self.body, "embedding_input_span", self.body.embedding_span)
                if self.body.stage == 0 else None
            )
            mask = self.body.embedding_mask if self.body.stage == 0 else None
            if self.body.stage == 0 and (span is None or mask is None):
                raise Qwen3_5TP2PP2ServingError("first stage has no embedding shard")
            for row in rows:
                if (
                    not 0 <= row.position < self.max_sequence_tokens
                    or not 0 <= row.token_id < self.vocab_size
                    or not row.page_table
                    or len(row.page_table) > self._table_pages
                    or not 0 <= row.state_source_slot < arena.spec.state_slot_capacity
                    or not 0 <= row.page_source < arena.spec.page_capacity
                    or any(
                        not 0 <= page < arena.spec.page_capacity
                        for page in row.page_table
                    )
                ):
                    raise Qwen3_5TP2PP2ServingError(
                        "request row lies outside cache capacity"
                    )
                owned = bool(
                    span is not None and span.start <= row.token_id < span.stop
                )
                packed_rows.append(
                    (
                        row.position,
                        row.page_source,
                        row.page_destination,
                        row.position % self._block_size,
                        row.position + 1,
                        row.token_id - span.start if owned and span is not None else 0,
                        int(owned),
                        1,
                        row.state_destination_slot,
                    )
                )
                page_tables.append(row.page_table)
                axes = row.position_axes or (row.position, row.position, row.position)
                if len(axes) != 3 or any(
                    type(axis) is not int or axis < 0 for axis in axes
                ):
                    raise Qwen3_5TP2PP2ServingError("request row has invalid MRoPE axes")
                position_axes.append(axes)

            active = len(rows)
            metadata[:active, :9] = np.asarray(packed_rows, dtype=np.int32)
            states = metadata[:active, 8:9] * local_layers
            slots = states + self._local_slot_offsets
            metadata[:active, self._source_slots_slice] = slots
            metadata[:active, self._destination_slots_slice] = slots
            if uniform_page_tables:
                metadata[
                    :active,
                    self._page_table_slice.start : self._page_table_slice.start
                    + page_table_pages,
                ] = np.asarray(page_tables, dtype=np.int32)
            else:
                for physical, row in enumerate(rows):
                    metadata[
                        physical,
                        self._page_table_slice.start : self._page_table_slice.start
                        + page_table_pages,
                    ] = row.page_destination
                    metadata[
                        physical,
                        self._page_table_slice.start : self._page_table_slice.start
                        + len(row.page_table),
                    ] = row.page_table
            if self._position_axes_slice is not None:
                metadata[:active, self._position_axes_slice] = np.asarray(
                    position_axes, dtype=np.int32
                )
            self._staged_copy_features = False
            self._staged_has_features = False
            self._staged_page_table_pages = page_table_pages
            return active, 0, 0
        active_scratch_states: set[int] = set(getattr(self, "_reserved_scratch_states", ()))
        scratch_start = self._scratch_states[0]
        scratch_stop = scratch_start + (
            arena.spec.state_scratch_capacity_per_arena if getattr(self, "_private_state_sources", False)
            else min(len(self._scratch_states), arena.spec.state_scratch_capacity_per_arena)
        )
        for physical, row in enumerate(rows):
            if (
                not 0 <= row.position < self.max_sequence_tokens
                or not 0 <= row.token_id < self.vocab_size
                or not row.page_table
                or len(row.page_table) > self._table_pages
                or not (
                    0 <= row.state_source_slot < arena.spec.state_slot_capacity
                    or (getattr(self, "_private_state_sources", False)
                        and not row.reset_state
                        and scratch_start <= row.state_source_slot < (
                            scratch_start + arena.spec.state_scratch_capacity_per_arena))
                )
                or not 0 <= row.state_destination_slot < (
                    arena.spec.state_slot_capacity
                    + arena.spec.state_scratch_capacity_per_arena * len(arena.spec.scratch_arena_ids)
                )
                or not 0 <= row.page_source < arena.spec.page_capacity
                or not 0 <= row.page_destination < arena.spec.page_capacity
                or any(not 0 <= page < arena.spec.page_capacity for page in row.page_table)
                or len(row.new_pages) > self._table_pages
                or any(not 0 <= page < arena.spec.page_capacity for page in row.new_pages)
            ):
                raise Qwen3_5TP2PP2ServingError("request row lies outside cache capacity")
            if row.reset_state:
                indices[reset_count] = row.state_source_slot
                reset_count += 1
            for page in row.new_pages:
                indices[self._index_page_offset + new_page_count] = page
                new_page_count += 1
            source = row.state_source_slot * local_layers
            destination = row.state_destination_slot * local_layers
            metadata[physical, 0:5] = (
                row.position,
                row.page_source,
                row.page_destination,
                row.position % self._block_size,
                row.position + 1,
            )
            metadata[physical, 7] = row.segment_length
            axes = row.position_axes or (row.position, row.position, row.position)
            if len(axes) != 3 or any(type(axis) is not int or axis < 0 for axis in axes):
                raise Qwen3_5TP2PP2ServingError("request row has invalid MRoPE axes")
            if position_axes_slice is not None:
                metadata[physical, position_axes_slice] = axes
            metadata[physical, self._state_destination_column] = row.state_destination_slot
            metadata[physical, self._source_slots_slice] = range(source, source + local_layers)
            metadata[physical, self._destination_slots_slice] = range(
                destination, destination + local_layers
            )
            metadata[
                physical,
                self._page_table_slice.start : self._page_table_slice.start
                + page_table_pages,
            ] = row.page_destination
            metadata[
                physical,
                self._page_table_slice.start : self._page_table_slice.start + len(row.page_table),
            ] = row.page_table
            if self.body.stage == 0:
                span = getattr(self.body, "embedding_input_span", self.body.embedding_span)
                if span is None or self.body.embedding_mask is None:
                    raise Qwen3_5TP2PP2ServingError("first stage has no embedding shard")
                owned = span.start <= row.token_id < span.stop
                has_feature = bool(row.input_feature)
                if has_feature and len(row.input_feature) != getattr(
                    self.body, "feature_width", self.body.hidden.shape[1]
                ) * 2:
                    raise Qwen3_5TP2PP2ServingError("input feature row differs from hidden size")
                metadata[physical, 5] = row.token_id - span.start if owned else 0
                metadata[physical, 6] = owned and not has_feature
                if feature_hosts is not None and copy_features:
                    feature = feature_hosts[host_slot][physical]
                    feature.zero_()
                    if has_feature and (
                        self.body.tensor_rank == 0
                        or getattr(self.body, "_embedding_table", None) is not None
                        or getattr(self.body, "features_replicated", False)
                    ):
                        feature.copy_(
                            torch.frombuffer(
                                bytearray(row.input_feature), dtype=torch.bfloat16
                            )
                        )
                elif has_feature:
                    raise Qwen3_5TP2PP2ServingError(
                        "input feature staging is unavailable"
                    )
            else:
                metadata[physical, 5:7] = 0
            for state_slot in (row.state_source_slot, row.state_destination_slot):
                if scratch_start <= state_slot < scratch_stop:
                    active_scratch_states.add(state_slot)
        padding_count = transfer_rows - len(rows)
        available_padding_states = tuple(
            state for state in range(scratch_start, scratch_stop)
            if state not in active_scratch_states
        )
        if padding_count and (not available_padding_states or (
                arena.spec.state_scratch_capacity is None and len(available_padding_states) < padding_count)):
            raise Qwen3_5TP2PP2ServingError(
                "active transaction state exhausts graph-private padding scratch"
            )
        padding_states = tuple(available_padding_states[i % len(available_padding_states)]
                               for i in range(padding_count))
        for offset, physical in enumerate(range(len(rows), transfer_rows)):
            page = self._scratch_pages[physical]
            scratch_state = padding_states[offset]
            aggregate = scratch_state * local_layers
            metadata[physical, 0:8] = (0, page, page, 0, 1, 0, 0, 0)
            metadata[physical, self._state_destination_column] = scratch_state
            metadata[physical, self._source_slots_slice] = range(
                aggregate, aggregate + local_layers
            )
            metadata[physical, self._destination_slots_slice] = range(
                aggregate, aggregate + local_layers
            )
            metadata[
                physical,
                self._page_table_slice.start : self._page_table_slice.start
                + page_table_pages,
            ] = page
            if position_axes_slice is not None:
                metadata[physical, position_axes_slice] = 0
            if feature_hosts is not None and copy_features:
                feature_hosts[host_slot][physical].zero_()
        self._staged_copy_features = copy_features
        self._staged_has_features = has_features
        self._staged_page_table_pages = page_table_pages
        return transfer_rows, reset_count, new_page_count

    def _bind_staged_metadata(
        self, metadata: Tensor, transfer_rows: int, page_table_pages: int | None = None
    ) -> None:
        """Bulk-copy one staged prefix into the fixed graph input buffers."""

        buffers = self.body.buffers
        position_axes_slice = getattr(self, "_position_axes_slice", None)
        positions = (
            metadata[:, 0].unsqueeze(0).expand(buffers.position_ids.shape[0], -1)
            if position_axes_slice is None
            else metadata[:, position_axes_slice].transpose(0, 1)
        )
        buffers.position_ids[:, :transfer_rows].copy_(positions)
        buffers.page_source[:transfer_rows].copy_(metadata[:, 1])
        buffers.page_destination[:transfer_rows].copy_(metadata[:, 2])
        buffers.token_offsets[:transfer_rows].copy_(metadata[:, 3])
        buffers.sequence_lengths[:transfer_rows].copy_(metadata[:, 4])
        buffers.segment_lengths[:transfer_rows].copy_(metadata[:, 7])
        copied_pages = self._table_pages if page_table_pages is None else page_table_pages
        buffers.page_table[:transfer_rows, :copied_pages].copy_(
            metadata[:, self._page_table_slice.start : self._page_table_slice.start + copied_pages]
        )
        slots = (
            metadata[:, self._source_slots_slice.start : self._destination_slots_slice.stop]
            .view(transfer_rows, 2, -1)
            .permute(1, 2, 0)
        )
        for target in buffers.state_slot_maps:
            target[:, :, :transfer_rows].copy_(slots)
        if self.body.stage == 0:
            buffers.token_ids[:transfer_rows].copy_(metadata[:, 5])
            assert self.body.embedding_mask is not None
            self.body.embedding_mask[:transfer_rows].copy_(metadata[:, 6:7])

    def bind_batch(self, rows: tuple[_BatchRow, ...]) -> None:
        """Bind one batch completely without launching its captured graph."""

        if self._batch_phase_failed:
            raise Qwen3_5TP2PP2ServingError("batch bind/replay phase is poisoned")
        if self._batch_bound:
            self._batch_phase_failed = True
            raise Qwen3_5TP2PP2ServingError("a bound batch must replay before another bind")
        profiler = self._feeding_profiler
        if profiler is not None and not profiler.active:
            profiler = None
        bind_started_ns = time.monotonic_ns() if profiler is not None else 0
        try:
            host_slot = self._next_host_slot
            if self._host_event_recorded[host_slot]:
                self._host_events[host_slot].synchronize()
            transfer_rows, reset_count, new_page_count = self._stage_batch_rows(
                rows, host_slot=host_slot
            )
            copy_features = getattr(self, "_staged_copy_features", False)
            has_features = getattr(self, "_staged_has_features", False)
            page_table_pages = getattr(
                self, "_staged_page_table_pages", getattr(self, "_table_pages", 1)
            )
            independent = owned_single_token_rows(rows)
            overlap = getattr(self.slot, "_overlap", None)
            if not independent and overlap is not None:
                independent = independent_decode_partitions(
                    rows, tuple(child.captured_m for child in overlap[2])
                )
            self.slot.select_decode_overlap(independent)
            self.body.set_active_rows(len(rows))
            arena = self.body.arena
            metadata = self._metadata_device[:transfer_rows]
            current = torch.cuda.current_stream(self.body.hidden.device)
            if profiler is not None:
                profiler.begin_cuda("production_bind_gpu_ms")
            page_table_slice = getattr(self, "_page_table_slice", None)
            position_axes_slice = getattr(self, "_position_axes_slice", None)
            if page_table_slice is None:
                self._metadata_device[:transfer_rows].copy_(
                    self._metadata_hosts[host_slot][:transfer_rows], non_blocking=True
                )
            else:
                prefix_stop = page_table_slice.start + page_table_pages
                self._metadata_device[:transfer_rows, :prefix_stop].copy_(
                    self._metadata_hosts[host_slot][:transfer_rows, :prefix_stop],
                    non_blocking=True,
                )
                if position_axes_slice is not None:
                    self._metadata_device[:transfer_rows, position_axes_slice].copy_(
                        self._metadata_hosts[host_slot][
                            :transfer_rows, position_axes_slice
                        ],
                        non_blocking=True,
                    )
            feature_rows = getattr(self.body, "feature_rows", None)
            if feature_rows is not None and copy_features:
                feature_rows[:transfer_rows].copy_(
                    self._feature_hosts[host_slot][:transfer_rows], non_blocking=True
                )
            if reset_count:
                self._index_device[:reset_count].copy_(
                    self._index_hosts[host_slot][:reset_count], non_blocking=True,
                )
            if new_page_count:
                page_indices = slice(self._index_page_offset,
                                     self._index_page_offset + new_page_count)
                self._index_device[page_indices].copy_(
                    self._index_hosts[host_slot][page_indices], non_blocking=True,
                )
            self._host_events[host_slot].record(current)
            self._host_event_recorded[host_slot] = True
            self._next_host_slot = (host_slot + 1) % len(self._metadata_hosts)
            if reset_count:
                reset_slots = self._index_device[:reset_count]
                arena.reset_state_rows(reset_slots)
                target_hidden_slots = self._target_hidden_slots
                if target_hidden_slots is not None:
                    target_hidden_slots.index_fill_(0, reset_slots, 0)
            if new_page_count:
                new_pages = self._index_device[
                    self._index_page_offset : self._index_page_offset + new_page_count
                ]
                arena.reset_page_rows(new_pages)
            self._bind_staged_metadata(metadata, transfer_rows, page_table_pages)
            self.body.buffers.row_activity[: len(rows)].fill_(1)
            self.body.buffers.row_activity[len(rows) : transfer_rows].zero_()
            self._features_bound = has_features
            if profiler is not None:
                profiler.end_cuda("production_bind_gpu_ms")
            self._bound_active_rows = len(rows)
            if profiler is not None:
                profiler.record_cpu(
                    "production_bind_cpu_ms",
                    (time.monotonic_ns() - bind_started_ns) / 1_000_000,
                )
        except BaseException:
            self._batch_phase_failed = True
            raise
        self._batch_bound = True

    def complete_bound(self, *, replay: bool, state_only: bool = False) -> None:
        """Consume one successful bind, optionally through this rank's slot."""

        if self._batch_phase_failed:
            raise Qwen3_5TP2PP2ServingError("batch bind/replay phase is poisoned")
        if not self._batch_bound:
            self._batch_phase_failed = True
            raise Qwen3_5TP2PP2ServingError(
                "captured graph replay requires one successful batch bind"
            )
        self._batch_bound = False
        if not replay:
            return
        profiler = self._feeding_profiler
        if profiler is not None and not profiler.active:
            profiler = None
        current = torch.cuda.current_stream(self.body.hidden.device)
        try:
            self.slot.stream.wait_stream(current)
            replay_started = getattr(self, "_replay_started", None)
            replay_finished = getattr(self, "_replay_finished", None)
            if replay_started is not None:
                replay_started.record(self.slot.stream)
            if profiler is not None:
                with torch.cuda.stream(self.slot.stream):
                    profiler.begin_cuda("graph_replay_gpu_ms")
            if state_only:
                self.slot.replay_state()
            else:
                self.slot.replay()
            if profiler is not None:
                with torch.cuda.stream(self.slot.stream):
                    profiler.end_cuda("graph_replay_gpu_ms")
            if replay_finished is not None:
                replay_finished.record(self.slot.stream)
                self._replay_timing_pending = True
            current.wait_stream(self.slot.stream)
        except BaseException:
            self._batch_phase_failed = True
            raise

    def replay_bound(self) -> None:
        """Replay exactly once after a successful ``bind_batch``."""

        self.complete_bound(replay=True)

    def replay_elapsed_ms(self) -> float:
        """Return the last target graph's GPU time at its existing sync boundary."""

        if not self._replay_timing_pending:
            raise Qwen3_5TP2PP2ServingError("target graph timing has no completed replay")
        milliseconds = self._replay_started.elapsed_time(self._replay_finished)
        self._replay_timing_pending = False
        return milliseconds

    def replay_batch(self, rows: tuple[_BatchRow, ...], *, commit_hidden: bool = True) -> None:
        """Bind independent request state and replay it in the original order."""

        self.bind_batch(rows)
        self.replay_bound()
        if commit_hidden:
            self._commit_target_hidden(rows)

    def _commit_target_hidden(self, rows: tuple[_BatchRow, ...]) -> None:
        """Publish replayed row hidden states to their durable transaction slots."""

        target_hidden_slots = getattr(self, "_target_hidden_slots", None)
        if target_hidden_slots is not None:
            destinations = tuple(row.state_destination_slot for row in rows)
            if len(set(destinations)) == len(destinations):
                self._commit_owned_target_hidden(len(rows))
            else:
                for physical, destination in enumerate(destinations):
                    feedback = getattr(self.body, "draft_feedback", self.body.buffers.norm)
                    target_hidden_slots[destination].copy_(feedback[physical])

    def _commit_owned_target_hidden(self, active_rows: int) -> None:
        target_hidden_slots = getattr(self, "_target_hidden_slots", None)
        if target_hidden_slots is not None:
            self._state_destinations[:active_rows].copy_(
                self._metadata_device[:active_rows, self._state_destination_column]
            )
            target_hidden_slots.index_copy_(
                0, self._state_destinations[:active_rows],
                getattr(self.body, "draft_feedback", self.body.buffers.norm)[:active_rows]
            )

    def replay_steady_continuation(
        self,
        active_rows: int,
        page_updates: tuple[tuple[int, int, int], ...],
    ) -> None:
        """Advance an unchanged owned decode batch without rebuilding metadata."""

        if (
            self._batch_phase_failed
            or self._batch_bound
            or active_rows != self._bound_active_rows
            or not 0 < active_rows <= self.captured_m
        ):
            raise Qwen3_5TP2PP2ServingError(
                "steady continuation differs from the resident graph batch"
            )
        profiler = self._feeding_profiler
        if profiler is not None and not profiler.active:
            profiler = None
        bind_started_ns = time.monotonic_ns() if profiler is not None else 0
        buffers = self.body.buffers
        if profiler is not None:
            profiler.begin_cuda("production_bind_gpu_ms")
        self._input_update_graph.replay()
        if page_updates:
            count = len(page_updates)
            if count > active_rows or any(
                not 0 <= physical < active_rows
                or not 0 <= page_index < self._table_pages
                or not 0 <= page < self.body.arena.spec.page_capacity
                for physical, page_index, page in page_updates
            ):
                raise Qwen3_5TP2PP2ServingError(
                    "steady continuation page update is outside cache capacity"
                )
            self._continuation_updates_host_array[:count] = page_updates
            self._continuation_updates_device[:count].copy_(
                self._continuation_updates_host[:count], non_blocking=True
            )
            self._update_resident_pages(
                self._continuation_updates_device[:count], self.body.arena.key,
                self.body.arena.value, self.body.arena.written, buffers.token_offsets,
                buffers.page_source, buffers.page_destination, buffers.page_table,
                page_fields=tuple(self.body.arena.page_fields.values()),
            )
        if profiler is not None:
            profiler.end_cuda("production_bind_gpu_ms")
            profiler.record_cpu(
                "production_bind_cpu_ms",
                (time.monotonic_ns() - bind_started_ns) / 1_000_000,
            )
        self._batch_bound = True
        self.replay_bound()
        self._commit_owned_target_hidden(active_rows)

    def sample_tokens(self, batch: int | tuple[_BatchRow, ...]) -> tuple[int, ...]:
        greedy = type(batch) is int
        rows = None if greedy else cast(tuple[_BatchRow, ...], batch)
        active_rows = cast(int, batch) if rows is None else len(rows)
        profiler = self._feeding_profiler
        if profiler is not None and not profiler.active:
            profiler = None
        if greedy:
            token_ids = gather_greedy_broadcast_tp(
                local_logits=self.slot.logits,
                local_vocab_span=self.slot.vocab_span,
                final_ranks=self._final_ranks,
                final_vocab_spans=self._final_spans,
                true_vocab_size=self.vocab_size,
                active_rows=active_rows,
                token_ids=self._token_ids,
                local_max_scores=self._local_max_scores,
                gathered_max_scores=self._gathered_max_scores,
                gathered_token_ids=self._gathered_token_ids,
                final_tp_group=self._final_tp_group,
                world_group=self._world_group,
                device=self.body.hidden.device,
                profiler=profiler,
            )
        else:
            assert rows is not None
            all_random_rows = self._sampling.bind(rows)
            self._sampling_bound_all_random = all_random_rows
            token_ids = sample_sharded_broadcast_tp(
                local_logits=self.slot.logits,
                local_vocab_span=self.slot.vocab_span,
                final_ranks=self._final_ranks,
                final_vocab_spans=self._final_spans,
                true_vocab_size=self.vocab_size,
                active_rows=active_rows,
                token_ids=self._token_ids,
                parameters=self._sampling.parameters,
                workspace=self._sampling.workspace,
                final_tp_group=self._final_tp_group,
                world_group=self._world_group,
                device=self.body.hidden.device,
                root_sampler=self._sampling.root_sampler,
                root_valid=self._sampling.root_valid,
                all_random_rows=all_random_rows,
                all_greedy_rows=self._sampling.all_greedy,
                greedy_plan=self._sampling.bound_plan(self.slot.logits),
                profiler=profiler,
            )
        if profiler is not None:
            profiler.flush_cuda(("production_bind_gpu_ms", "graph_replay_gpu_ms"))
        if self._rank != self._sampler_rank:
            return ()
        d2h_started_ns = time.monotonic_ns() if profiler is not None else 0
        result = tuple(int(token) for token in token_ids[:active_rows].cpu().tolist())
        if profiler is not None:
            profiler.record_cpu(
                "sampled_token_d2h_cpu_ms",
                (time.monotonic_ns() - d2h_started_ns) / 1_000_000,
            )
        return result

    def sample_steady_continuation(self, active_rows: int) -> tuple[int, ...]:
        """Sample with request-stable parameters already resident on this rank."""

        all_random_rows = self._sampling_bound_all_random
        if all_random_rows is None or active_rows != self._bound_active_rows:
            raise Qwen3_5TP2PP2ServingError(
                "steady continuation has no resident sampling parameters"
            )
        profiler = self._feeding_profiler
        if profiler is not None and not profiler.active:
            profiler = None
        token_ids = sample_sharded_broadcast_tp(
            local_logits=self.slot.logits,
            local_vocab_span=self.slot.vocab_span,
            final_ranks=self._final_ranks,
            final_vocab_spans=self._final_spans,
            true_vocab_size=self.vocab_size,
            active_rows=active_rows,
            token_ids=self._token_ids,
            parameters=self._sampling.parameters,
            workspace=self._sampling.workspace,
            final_tp_group=self._final_tp_group,
            world_group=self._world_group,
            device=self.body.hidden.device,
            root_sampler=self._sampling.root_sampler,
            root_valid=self._sampling.root_valid,
            all_random_rows=all_random_rows,
            all_greedy_rows=self._sampling.all_greedy,
            greedy_plan=self._sampling.bound_plan(self.slot.logits),
            profiler=profiler,
        )
        if profiler is not None:
            profiler.flush_cuda(("production_bind_gpu_ms", "graph_replay_gpu_ms"))
        if self._rank != self._sampler_rank:
            return ()
        result = tuple(int(token) for token in token_ids[:active_rows].cpu().tolist())
        return result

    def gather_candidates(self, rows: tuple[_BatchRow, ...]) -> None:
        """Collect TP candidates for device-side speculative verification."""

        profiler = self._feeding_profiler
        if profiler is not None and not profiler.active:
            profiler = None
        all_random_rows = self._sampling.bind(rows)
        sample_sharded_broadcast_tp(
            local_logits=self.slot.logits,
            local_vocab_span=self.slot.vocab_span,
            final_ranks=self._final_ranks,
            final_vocab_spans=self._final_spans,
            true_vocab_size=self.vocab_size,
            active_rows=len(rows),
            token_ids=self._token_ids,
            parameters=self._sampling.parameters,
            workspace=self._sampling.workspace,
            final_tp_group=self._final_tp_group,
            world_group=self._world_group,
            device=self.body.hidden.device,
            root_sampler=self._sampling.root_sampler,
            root_valid=self._sampling.root_valid,
            all_random_rows=all_random_rows,
            profiler=profiler,
            candidates_only=True,
            greedy_plan=(
                self._sampling.bound_plan(self.slot.logits)
                if self._sampling.constraint_bound
                else None
            ),
        )
        if profiler is not None:
            profiler.flush_cuda(("production_bind_gpu_ms", "graph_replay_gpu_ms"))


_MTPReplayRow = DraftReplayRow


@dataclass(frozen=True, slots=True)
class _MTPBindHost:
    tensor: Tensor
    array: np.ndarray[Any, np.dtype[np.int32]]


class CapturedDraftRank:
    """Shared proposal sampling/history/verification over a provider graph.

    Model and format arithmetic are supplied by the body; state ownership and
    metadata binding are supplied by the transaction binding. The legacy MTP
    binder remains an adapter for its already-qualified stateless KV body.
    """

    def __init__(
        self,
        body: Any,
        slot: Any,
        runtime: Any,
        admission: _Admission,
        *,
        request_page_capacity: int,
        target_hidden_slots: Tensor,
        binding: DraftReplayBinding | None = None,
    ) -> None:
        self.body, self.slot = body, slot
        self._draft_binding = binding
        self._feedback = body.buffers.norm if binding is None else binding.feedback
        self._max_draft_tokens = admission.max_speculative_tokens
        self._max_request_owners = getattr(
            admission, "graph_request_capacity", admission.max_running_requests)
        self.captured_m = body.captured_m
        self.vocab_size = builtin_text_config(admission.config).vocabulary_size
        self._request_page_capacity = request_page_capacity
        self._target_hidden_slots = target_hidden_slots
        self._rank = body.rank
        self._final_ranks = _final_ranks(admission.parallel_plan)
        self._final_spans = _final_vocab_spans(admission.parallel_plan)
        self._sampler_rank = _sampler_rank(admission.parallel_plan)
        self._world_group = cast(ProcessGroup, dist.group.WORLD)
        self._tp_group = runtime._groups["tp:0"]
        self._final_tp_group = self._tp_group
        self._token_ids = torch.empty(
            self.captured_m, dtype=torch.int64, device=body.buffers.hidden.device
        )
        self._sampling = _SamplingState(
            self.captured_m,
            tp_size=len(self._final_ranks),
            device=body.buffers.hidden.device,
            final_rank=True,
            sampling_root=self._rank == self._sampler_rank,
            token_ids=self._token_ids,
            asynchronous_bind=True,
        )
        candidate_workspace = self._sampling.workspace
        if candidate_workspace is None:
            raise Qwen3_5TP2PP2ServingError("MTP rank has no TP candidate workspace")
        history_shape = (
            self._max_draft_tokens,
            *candidate_workspace.gathered_scores.shape,
        )
        self._verification_q_scores = torch.empty(
            history_shape, dtype=torch.bfloat16, device=body.buffers.hidden.device
        )
        self._verification_q_token_ids = torch.empty(
            history_shape, dtype=torch.int64, device=body.buffers.hidden.device
        )
        self._verification_constraint_rows = torch.zeros(
            (self._max_draft_tokens, self.captured_m),
            dtype=torch.bool,
            device=body.buffers.hidden.device,
        )
        row_history_shape = (self._max_draft_tokens, self.captured_m)
        self._verification_draft_tokens = torch.empty(
            row_history_shape, dtype=torch.int64, device=body.buffers.hidden.device
        )
        self._verification_ordinals = torch.empty(
            row_history_shape, dtype=torch.int64, device=body.buffers.hidden.device
        )
        self._verification_temperatures = torch.empty(
            row_history_shape, dtype=torch.float32, device=body.buffers.hidden.device
        )
        self._verification_top_ks = torch.empty(
            row_history_shape, dtype=torch.int64, device=body.buffers.hidden.device
        )
        self._verification_top_ps = torch.empty(
            row_history_shape, dtype=torch.float32, device=body.buffers.hidden.device
        )
        self._verification_seeds = torch.empty(
            row_history_shape, dtype=torch.int64, device=body.buffers.hidden.device
        )
        self._verification_target_rows = torch.empty(
            row_history_shape, dtype=torch.int32, device=body.buffers.hidden.device
        )
        self._verification_target_rows_host = torch.empty(
            row_history_shape, dtype=torch.int32, pin_memory=True
        )
        self._verification_decisions = torch.empty(
            (self._max_draft_tokens, self.captured_m, 2),
            dtype=torch.int64,
            device=body.buffers.hidden.device,
        )
        self._proposal_hidden = torch.empty_like(body.buffers.target_hidden)
        self._proposal_input_tokens = torch.empty_like(self._token_ids)
        self._verification_validity = torch.empty(
            self._max_draft_tokens, dtype=torch.bool, device=body.buffers.hidden.device
        )
        self._verifier_qualified = False
        # Dynamic K consumes completed proposal costs, never asynchronous
        # submission times. These depth events belong to the round's lease.
        self._proposal_cost_events = tuple(
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(self._max_draft_tokens)
        ) if admission.dynamic_speculation else ()
        for start, end in self._proposal_cost_events:
            start.record()
            end.record()
        self._feeding_profiler: _CudaStepProfiler | None = None
        if binding is not None:
            return
        table_pages = body.buffers.page_table.shape[1]
        self._bind_tokens = torch.empty(
            self.captured_m, dtype=torch.int64, device=body.buffers.hidden.device
        )
        self._bind_metadata = torch.empty(
            (self.captured_m, 5 + table_pages),
            dtype=torch.int32,
            device=body.buffers.hidden.device,
        )
        bind_hosts: list[_MTPBindHost] = []
        for _ in range(2):
            tensor = torch.zeros(
                (self.captured_m, 5 + table_pages),
                dtype=torch.int32,
                pin_memory=True,
            )
            array = tensor.numpy()
            scratch = self._request_page_capacity + np.arange(
                self.captured_m, dtype=np.int32
            )
            array[:, 2] = scratch
            array[:, 4] = 1
            array[:, 5:] = scratch[:, None]
            bind_hosts.append(_MTPBindHost(tensor, array))
        self._bind_hosts = tuple(bind_hosts)
        self._bind_host_events = tuple(
            torch.cuda.Event()
            for _ in self._bind_hosts  # type: ignore[no-untyped-call]
        )
        self._bind_host_event_recorded = [False] * len(self._bind_hosts)
        self._bind_host_counts = [0] * len(self._bind_hosts)
        self._next_bind_host = 0
        self._feeding_profiler: _CudaStepProfiler | None = None

    @property
    def verification_decisions(self) -> Tensor:
        return self._verification_decisions

    @property
    def state_fields(self) -> Mapping[str, Tensor]:
        return {} if self._draft_binding is None else self._draft_binding.state_fields

    def stash_proposal(
        self,
        depth: int,
        active_rows: int,
        *,
        source: CapturedDraftRank | None = None,
    ) -> None:
        """Retain one proposal level from its depth-specific graph executor."""

        source_executor = self if source is None else source
        if (
            not 0 <= depth < self._max_draft_tokens
            or not 0 < active_rows <= self.captured_m
            or active_rows > source_executor.captured_m
        ):
            raise Qwen3_5TP2PP2ServingError("MTP proposal history exceeds capacity")
        workspace = source_executor._sampling.workspace
        if workspace is None:
            raise Qwen3_5TP2PP2ServingError("MTP proposal history requires TP candidates")
        self._verification_q_scores[depth, :, :active_rows].copy_(
            workspace.gathered_scores[:, :active_rows]
        )
        self._verification_q_token_ids[depth, :, :active_rows].copy_(
            workspace.gathered_token_ids[:, :active_rows]
        )
        self._verification_draft_tokens[depth, :active_rows].copy_(
            source_executor._token_ids[:active_rows]
        )
        if source_executor._sampling.constraint_bound:
            self._verification_constraint_rows[depth, :active_rows].copy_(
                source_executor._sampling._constraint_activity[:active_rows]
            )
        self._verification_ordinals[depth, :active_rows].copy_(
            source_executor._sampling.ordinals[:active_rows]
        )
        self._verification_temperatures[depth, :active_rows].copy_(
            source_executor._sampling.temperatures[:active_rows]
        )
        self._verification_top_ks[depth, :active_rows].copy_(
            source_executor._sampling.top_ks[:active_rows]
        )
        self._verification_top_ps[depth, :active_rows].copy_(
            source_executor._sampling.top_ps[:active_rows]
        )
        self._verification_seeds[depth, :active_rows].copy_(
            source_executor._sampling.seeds[:active_rows]
        )

    def chain_hidden(self, source_rows: tuple[int, ...]) -> Tensor:
        """Compact selected MTP hidden rows into caller-stable scratch."""

        if (
            not source_rows
            or len(source_rows) > self.captured_m
            or any(not 0 <= physical < self.captured_m for physical in source_rows)
        ):
            raise Qwen3_5TP2PP2ServingError("MTP hidden history exceeds capacity")
        for destination, source in enumerate(source_rows):
            self._proposal_hidden[destination].copy_(self._feedback[source])
        return self._proposal_hidden[: len(source_rows)]

    def chain_tokens(self, tokens: Tensor, source_rows: tuple[int, ...]) -> Tensor:
        """Keep ragged proposal IDs on device with bounded embedding inputs.

        The original IDs and all sampler guards remain in proposal history.
        Clamping only protects private speculative work until the common
        transaction observes every guard; it never repairs a published token.
        """
        if (tokens.ndim != 1 or tokens.dtype is not torch.int64
            or tokens.device != self._token_ids.device or not source_rows
            or len(source_rows) > self.captured_m
            or any(type(row) is not int or not 0 <= row < tokens.numel()
                   for row in source_rows)):
            raise Qwen3_5TP2PP2ServingError("draft token chain differs from its capacity")
        self._proposal_input_tokens.zero_()
        for destination, source in enumerate(source_rows):
            self._proposal_input_tokens[destination].copy_(tokens[source])
        self._proposal_input_tokens.clamp_(0, self.vocab_size - 1)
        return self._proposal_input_tokens

    def materialize_proposals(self, level_counts: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        """Consume every rank's immutable depth guard before target review."""
        if (not 0 < len(level_counts) <= self._max_draft_tokens
            or any(type(count) is not int or not 0 < count <= self.captured_m
                   for count in level_counts)):
            raise Qwen3_5TP2PP2ServingError("draft result history exceeds capacity")
        if not bool(self._verification_validity[:len(level_counts)].all().item()):
            raise Qwen3_5TP2PP2ServingError("draft chain sampling guard rejected")
        history = self._verification_draft_tokens[:len(level_counts)].cpu().tolist()
        result = tuple(tuple(int(token) for token in values[:count])
                       for values, count in zip(history, level_counts, strict=True))
        if any(not 0 <= token < self.vocab_size for level in result for token in level):
            raise Qwen3_5TP2PP2ServingError("draft chain token range guard rejected")
        return result

    def launch_verification(
        self,
        target: Any,
        physical_rows_by_depth: tuple[tuple[int, ...], ...],
    ) -> Tensor:
        """Verify compact request subsets at each independent draft depth."""

        if not execution_ready(self, "_verifier_qualified"):
            raise Qwen3_5TP2PP2ServingError("draft verifier has no startup qualification")
        if (
            not 1 <= len(physical_rows_by_depth) <= self._max_draft_tokens
            or any(
                not 0 < len(physical_rows) <= self.captured_m
                for physical_rows in physical_rows_by_depth
            )
        ):
            raise Qwen3_5TP2PP2ServingError("MTP verification rows exceed capacity")
        if self._rank != self._sampler_rank:
            return self._verification_decisions
        q_workspace = self._sampling.workspace
        p_workspace = target._sampling.workspace
        if q_workspace is None or p_workspace is None:
            raise Qwen3_5TP2PP2ServingError("MTP verifier requires TP candidates")
        for depth, physical_rows in enumerate(physical_rows_by_depth):
            for index, physical in enumerate(physical_rows):
                self._verification_target_rows_host[depth, index] = physical + depth
            self._verification_target_rows[depth, : len(physical_rows)].copy_(
                self._verification_target_rows_host[depth, : len(physical_rows)],
                non_blocking=True,
            )
        profiler = self._feeding_profiler
        if profiler is not None and profiler.active:
            profiler.begin_cuda("mtp_verify_gpu_ms")
        for depth, physical_rows in enumerate(physical_rows_by_depth):
            verify_one_draft_candidates(
                q_scores=self._verification_q_scores[depth],
                q_token_ids=self._verification_q_token_ids[depth],
                p_scores=p_workspace.gathered_scores,
                p_token_ids=p_workspace.gathered_token_ids,
                target_rows=self._verification_target_rows[depth],
                draft_tokens=self._verification_draft_tokens[depth],
                temperatures=self._verification_temperatures[depth],
                top_ps=self._verification_top_ps[depth],
                top_ks=self._verification_top_ks[depth],
                seeds=self._verification_seeds[depth],
                ordinals=self._verification_ordinals[depth],
                decisions=self._verification_decisions[depth],
                active_rows=len(physical_rows),
                allow_underfilled=target._sampling.constraint_bound,
                constrained_rows=(
                    self._verification_constraint_rows[depth]
                    if target._sampling.constraint_bound
                    else None
                ),
            )
        if profiler is not None and profiler.active:
            profiler.end_cuda("mtp_verify_gpu_ms")
        return self._verification_decisions

    def flush_verification_profile(self) -> None:
        profiler = self._feeding_profiler
        if self._rank == self._sampler_rank and profiler is not None and profiler.active:
            profiler.flush_cuda(("mtp_verify_gpu_ms",))

    def enable_feeding_profile(self, aggregate: _StepFeedingProfile) -> None:
        if self._feeding_profiler is not None:
            raise Qwen3_5TP2PP2ServingError("MTP feeding profile is already enabled")
        self._feeding_profiler = _CudaStepProfiler(
            aggregate,
            self.body.buffers.hidden.device,
            _MTP_CUDA_PROFILE_METRICS,
        )

    def reserve_state_slots(self, slots: tuple[int, ...]) -> None:
        if self._draft_binding is not None:
            self._draft_binding.reserve_state_slots(slots)
        else:
            # Inline rounds also bypass every legacy host staging slot.
            self._bind_host_counts[:] = [self.captured_m] * len(self._bind_host_counts)

    def _bind(self, rows: tuple[_MTPReplayRow, ...], target_hidden: Tensor) -> None:
        count = len(rows)
        if (
            not 0 < count <= self.captured_m
            or target_hidden.shape != (count, self.body.buffers.target_hidden.shape[1])
            or target_hidden.dtype is not torch.bfloat16
            or target_hidden.device != self.body.buffers.hidden.device
        ):
            raise Qwen3_5TP2PP2ServingError("MTP rows differ from the captured ABI")
        if self._draft_binding is not None:
            self._draft_binding.bind(rows, target_hidden)
            return
        b = self.body.buffers
        host_slot = self._next_bind_host
        if self._bind_host_event_recorded[host_slot]:
            self._bind_host_events[host_slot].synchronize()
        host = self._bind_hosts[host_slot]
        metadata = host.array
        reset_stop = max(count, self._bind_host_counts[host_slot])
        page_table_pages = max(1, max(len(row.page_table) for row in rows))
        for physical in range(reset_stop):
            if physical < count:
                row = rows[physical]
                if (
                    not 0 <= row.position < self.body._cosine_table.shape[0]
                    or not 0 <= row.page_destination < self._request_page_capacity
                    or not row.page_table
                    or len(row.page_table) > b.page_table.shape[1]
                ):
                    raise Qwen3_5TP2PP2ServingError("MTP row lies outside cache capacity")
                metadata[physical, :5] = (
                    row.token_id,
                    row.position,
                    row.page_destination,
                    row.position % _BLOCK_SIZE,
                    row.position + 1,
                )
                metadata[physical, 5 : 5 + page_table_pages] = row.page_destination
                metadata[physical, 5 : 5 + len(row.page_table)] = row.page_table
            else:
                scratch = self._request_page_capacity + physical
                metadata[physical, :5] = (0, 0, scratch, 0, 1)
                metadata[physical, 5 : 5 + page_table_pages] = scratch
        self._bind_host_counts[host_slot] = count
        live_columns = 5 + page_table_pages
        self._bind_metadata[:, :live_columns].copy_(
            host.tensor[:, :live_columns], non_blocking=True
        )
        device_metadata = self._bind_metadata
        self._bind_tokens.copy_(device_metadata[:, 0])
        b.position_ids.copy_(device_metadata[:, 1].unsqueeze(0).expand(3, -1))
        b.page_destination.copy_(device_metadata[:, 2])
        b.token_offsets.copy_(device_metadata[:, 3])
        b.sequence_lengths.copy_(device_metadata[:, 4])
        self.body._active_rows.fill_(count)
        # The captured ABI reserves the maximum context width, but attention
        # can only read blocks covered by sequence_lengths.  Copy that live
        # prefix instead of retransmitting the entire auto-context table on
        # every proposal/catch-up wave.
        b.page_table[:, :page_table_pages].copy_(
            device_metadata[:, 5:live_columns]
        )
        self.body.bind_token_ids(self._bind_tokens)
        b.target_hidden[:count].copy_(target_hidden)
        if count < self.captured_m:
            b.target_hidden[count:].zero_()
        self._bind_host_events[host_slot].record(torch.cuda.current_stream(b.hidden.device))
        self._bind_host_event_recorded[host_slot] = True
        self._next_bind_host = (host_slot + 1) % len(self._bind_hosts)

    def replay(
        self,
        rows: tuple[_MTPReplayRow, ...],
        target_hidden: Tensor,
        *,
        sample: bool = True,
        input_tokens: Tensor | None = None,
        validity_out: Tensor | None = None,
    ) -> tuple[int, ...]:
        if validity_out is not None and not sample:
            raise Qwen3_5TP2PP2ServingError("draft maintenance has no sampling guard")
        if input_tokens is not None and (
            input_tokens.shape != self._token_ids.shape
            or input_tokens.dtype is not torch.int64
            or input_tokens.device != self._token_ids.device
        ):
            raise Qwen3_5TP2PP2ServingError("draft device token input differs from its graph")
        profiler = self._feeding_profiler
        if profiler is not None and not profiler.active:
            profiler = None
        phase = "proposal" if sample else "catchup"
        bind_cpu_started_ns = time.monotonic_ns() if profiler is not None else 0
        if profiler is not None:
            profiler.begin_cuda(f"mtp_{phase}_bind_gpu_ms")
        self._bind(rows, target_hidden)
        if input_tokens is not None:
            if self._draft_binding is None:
                self.body.bind_token_ids(input_tokens)
            else:
                self.body.fill_token_ids(input_tokens)
        if profiler is not None:
            profiler.end_cuda(f"mtp_{phase}_bind_gpu_ms")
            profiler.record_cpu(
                f"mtp_{phase}_bind_cpu_ms",
                (time.monotonic_ns() - bind_cpu_started_ns) / 1_000_000,
            )
        if sample:
            self._sampling.bind(
                tuple(
                    _BatchRow(
                        row.request_id,
                        row.token_id,
                        row.position,
                        row.state_source_slot,
                        row.state_source_slot,
                        0,
                        0,
                        (0,),
                        sampling=row.sampling,
                        sample_ordinal=row.sample_ordinal,
                    )
                    for row in rows
                )
            )
        if sample:
            local_logits = self.slot.logits
            local_vocab_span = self.slot.vocab_span
            final_ranks = self._final_ranks
            final_spans = self._final_spans
            vocabulary_size = self.vocab_size
            active_rows = len(rows)
            token_buffer = self._token_ids
            parameters = self._sampling.parameters
            workspace = self._sampling.workspace
            tp_group = self._tp_group
            world_group = self._world_group
            device = self.body.buffers.hidden.device
            root_sampler = self._sampling.root_sampler
            root_valid = self._sampling.root_valid
            all_random_rows = all(row.sampling.temperature > 0.0 for row in rows)
        # Replay and post-graph sampling share the caller stream.  The MTP graph
        # contains NCCL collectives, so an event hand-off from its capture stream
        # can block before the sampler is even entered.
        current = torch.cuda.current_stream(self.body.buffers.hidden.device)
        if profiler is not None:
            profiler.begin_cuda(f"mtp_{phase}_graph_gpu_ms")
        if self._draft_binding is not None:
            self._draft_binding.replay(sample=sample)
        elif sample:
            self.slot.replay()
        else:
            self.slot.replay_state()
        if profiler is not None:
            profiler.end_cuda(f"mtp_{phase}_graph_gpu_ms")
        if not sample:
            if profiler is not None:
                wait_started_ns = time.monotonic_ns()
                current.synchronize()
                profiler.flush_cuda(("mtp_catchup_bind_gpu_ms", "mtp_catchup_graph_gpu_ms"))
                profiler.record_cpu(
                    "mtp_catchup_host_wait_ms",
                    (time.monotonic_ns() - wait_started_ns) / 1_000_000,
                )
            return ()
        assert workspace is not None
        token_ids = sample_sharded_broadcast_tp(
            local_logits=local_logits,
            local_vocab_span=local_vocab_span,
            final_ranks=final_ranks,
            final_vocab_spans=final_spans,
            true_vocab_size=vocabulary_size,
            active_rows=active_rows,
            token_ids=token_buffer,
            parameters=parameters,
            workspace=workspace,
            final_tp_group=tp_group,
            world_group=world_group,
            device=device,
            root_sampler=root_sampler,
            root_valid=root_valid,
            all_random_rows=all_random_rows,
            all_greedy_rows=self._sampling.all_greedy,
            greedy_plan=self._sampling.bound_plan(local_logits),
            profiler=profiler,
            groups_prevalidated=True,
            prevalidated_rank=self._rank,
            validity_out=validity_out,
        )
        if profiler is not None:
            profiler.flush_cuda(("mtp_proposal_bind_gpu_ms", "mtp_proposal_graph_gpu_ms"))
        result_cpu_started_ns = time.monotonic_ns() if profiler is not None else 0
        tokens = (() if validity_out is not None else
                  tuple(int(token) for token in token_ids[: len(rows)].cpu().tolist()))
        if profiler is not None:
            profiler.record_cpu(
                "mtp_proposal_result_cpu_ms",
                (time.monotonic_ns() - result_cpu_started_ns) / 1_000_000,
            )
        return tokens

    def sample_token(self) -> int:
        return self.sample_tokens(1)[0]


_CapturedMTPRank = CapturedDraftRank


def _qualify_draft_verification(
    drafts: dict[int, CapturedDraftRank], targets: dict[int, Any],
) -> None:
    """Qualify the actual verifier ABIs before readiness and cost learning.

    The independent fixture has q=p=delta(token 0), so both greedy and
    stochastic rules accept with probability one and return bonus token 0.
    An owned invalid pair rejects even when another shard supplies that delta.
    This operator witness is not an independent whole-model speculative audit.
    """
    for capacity, draft in sorted(drafts.items()):
        target_capacity = min(max(targets), 2 * capacity)
        target = targets[target_capacity]
        workspace = target._sampling.workspace
        if workspace is None:
            raise Qwen3_5TP2PP2ServingError("draft verification has no target candidates")
        counts = sorted({1, min(capacity, target_capacity // 2)})
        for count in counts:
            for temperature, failure in ((0., None), (1., None), (1., "q"),
                                         (1., "p0"), (1., "p1")):
                if draft._rank == draft._sampler_rank:
                    qs, qi = draft._verification_q_scores[0], draft._verification_q_token_ids[0]
                    ps, pi = workspace.gathered_scores, workspace.gathered_token_ids
                    qs.fill_(float("nan")); qi.fill_(-1)
                    ps.fill_(float("nan")); pi.fill_(-1)
                    qs[0, :count, 0].zero_(); qi[0, :count, 0].zero_()
                    ps[0, :2 * count, 0].zero_(); pi[0, :2 * count, 0].zero_()
                    if failure is not None:
                        scores, ids, row = ((qs, qi, 0) if failure == "q" else
                                            (ps, pi, int(failure == "p1")))
                        scores[-1, row, 0] = float("nan")
                        ids[-1, row, 0] = -2
                    draft._verification_target_rows[0].zero_()
                    draft._verification_target_rows[0, :count].copy_(
                        2 * torch.arange(count, device=qs.device, dtype=torch.int32))
                    draft._verification_draft_tokens[0].zero_()
                    draft._verification_temperatures[0].fill_(temperature)
                    draft._verification_top_ps[0].fill_(1)
                    draft._verification_top_ks[0].fill_(1)
                    draft._verification_seeds[0].fill_(1)
                    draft._verification_ordinals[0].zero_()
                    verify_one_draft_candidates(
                        q_scores=qs, q_token_ids=qi, p_scores=ps, p_token_ids=pi,
                        target_rows=draft._verification_target_rows[0],
                        draft_tokens=draft._verification_draft_tokens[0],
                        temperatures=draft._verification_temperatures[0],
                        top_ps=draft._verification_top_ps[0], top_ks=draft._verification_top_ks[0],
                        seeds=draft._verification_seeds[0], ordinals=draft._verification_ordinals[0],
                        decisions=draft._verification_decisions[0], active_rows=count,
                    )
                    torch.cuda.current_stream(qs.device).synchronize()
                schedule = (capacity, target_capacity, count, temperature, failure)
                schedules = [None] * dist.get_world_size(draft._world_group)
                dist.all_gather_object(schedules, schedule, group=draft._world_group)
                if any(item != schedule for item in schedules):
                    raise Qwen3_5TP2PP2ServingError("draft verifier audit schedule differs")
                if draft._rank == draft._sampler_rank:
                    expected = [[1, 0] for _ in range(count)]
                    if failure is not None:
                        expected[0] = [-1, -1]
                    expected.extend([[-1, -1] for _ in range(capacity - count)])
                    if draft._verification_decisions[0].cpu().tolist() != expected:
                        raise Qwen3_5TP2PP2ServingError("draft verifier differs from its fixed oracle")
        barrier(group=draft._world_group)
        draft._verifier_qualified = True
        diagnostic_event(_LOGGER, 'performance.draft_verifier_qualified_rank_d_physical_m',
            draft._rank,
            capacity,
            target_capacity,
        )


class _CapturedWavefrontRank:
    """Two balanced physical-M binders feeding one combined 2M graph replay."""

    def __init__(
        self,
        lane_a: Qwen3_5TP2PP2DecodeBody,
        lane_b: Qwen3_5TP2PP2DecodeBody,
        slot: Qwen3_5TP2PP2WavefrontGraphSlot,
        runtime: Any,
        admission: _Admission,
        target_hidden_slots: Tensor | None = None,
    ) -> None:
        if (
            slot.lane_a is not lane_a
            or slot.lane_b is not lane_b
            or lane_a.captured_m != lane_b.captured_m
            or slot.micro_m != lane_a.captured_m
        ):
            raise Qwen3_5TP2PP2ServingError(
                "wavefront binders differ from the captured physical lanes"
            )
        self.slot = slot
        self.body = lane_a
        self.micro_m = lane_a.captured_m
        self.captured_m = 2 * self.micro_m
        self.vocab_size = builtin_text_config(admission.config).vocabulary_size
        self.max_sequence_tokens = admission.max_sequence_tokens
        self.speculative = bool(admission.num_speculative_tokens)
        self._lane_a = _CapturedGraphRank(
            lane_a, slot.resource, slot, runtime, admission, target_hidden_slots
        )
        self._lane_b = _CapturedGraphRank(
            lane_b, slot.resource, slot, runtime, admission, target_hidden_slots
        )
        self._rank = lane_a.rank
        self._world_group = cast(ProcessGroup, dist.group.WORLD)
        self._final_spans = _final_vocab_spans(admission.parallel_plan)
        self._final_ranks = _final_ranks(admission.parallel_plan)
        self._sampler_rank = _sampler_rank(admission.parallel_plan)
        final_stage = admission.parallel_plan.topology.pp_size - 1
        self._final_tp_group = (
            runtime._groups[f"tp:{final_stage}"] if self._rank in self._final_ranks else None
        )
        device = lane_a.hidden.device
        self._token_ids = torch.empty(self.captured_m, dtype=torch.int64, device=device)
        self._sampling = _SamplingState(
            self.captured_m,
            tp_size=len(self._final_ranks),
            device=device,
            final_rank=self._rank in self._final_ranks,
            sampling_root=self._rank == self._sampler_rank,
            token_ids=self._token_ids,
        )
        self._local_max_scores = (
            torch.empty(self.captured_m, dtype=torch.bfloat16, device=device)
            if self._rank in self._final_ranks
            else None
        )
        self._gathered_max_scores = (
            torch.empty(
                (len(self._final_ranks), self.captured_m),
                dtype=torch.bfloat16,
                device=device,
            )
            if self._rank in self._final_ranks
            else None
        )
        self._gathered_token_ids = (
            torch.empty(
                (len(self._final_ranks), self.captured_m),
                dtype=torch.int64,
                device=device,
            )
            if self._rank in self._final_ranks
            else None
        )
        self._logical_logits = torch.empty_like(slot.logits) if slot.logits is not None else None
        self._feeding_profiler: _CudaStepProfiler | None = None
        self._replay_started = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        self._replay_finished = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        self._replay_timing_pending = False
        self._phase_failed = False
        self._active_rows: int | None = None
        self._lane_a_rows: int | None = None

    def _split_rows(
        self, rows: tuple[_BatchRow, ...]
    ) -> tuple[tuple[_BatchRow, ...], tuple[_BatchRow, ...]]:
        """Split only between request segments while keeping both lanes bounded."""

        segments = _batch_segments(rows)
        request_ids = tuple(rows[start].request_id for start, _stop in segments)
        if len(set(request_ids)) != len(request_ids):
            raise Qwen3_5TP2PP2ServingError(
                "wavefront lanes require request-disjoint contiguous segments"
            )
        boundaries = tuple(stop for _start, stop in segments)
        lower = len(rows) - self.micro_m
        eligible = tuple(stop for stop in boundaries if lower <= stop <= self.micro_m)
        split = min(
            eligible,
            key=lambda stop: (abs(2 * stop - len(rows)), -stop),
            default=0,
        )
        if split == 0:
            raise Qwen3_5TP2PP2ServingError(
                "wavefront request segments cannot fit two physical lanes: "
                f"logical_m={len(rows)} micro_m={self.micro_m} "
                f"segment_lengths={tuple(stop - start for start, stop in segments)}"
            )
        return rows[:split], rows[split:]

    def enable_feeding_profile(self, aggregate: _StepFeedingProfile) -> None:
        if self._feeding_profiler is not None:
            raise Qwen3_5TP2PP2ServingError("feeding profile is already enabled")
        self._feeding_profiler = _CudaStepProfiler(
            aggregate, self.body.hidden.device, _CUDA_PROFILE_METRICS
        )
        # Each binder owns a distinct CUDA event pair.  Recording both under the
        # existing bind key is two honest lane measurements, never one fabricated
        # combined interval spanning CPU preparation between the lanes.
        self._lane_a.enable_feeding_profile(aggregate)
        self._lane_b.enable_feeding_profile(aggregate)

    def replay_batch(self, rows: tuple[_BatchRow, ...], *, commit_hidden: bool = True) -> None:
        if self._phase_failed:
            raise Qwen3_5TP2PP2ServingError("wavefront rank is poisoned")
        if self._active_rows is not None:
            self._phase_failed = True
            raise Qwen3_5TP2PP2ServingError(
                "wavefront replay output must be sampled before the next bind"
            )
        logical_m = len(rows)
        if not self.micro_m < logical_m <= self.captured_m:
            self._phase_failed = True
            raise Qwen3_5TP2PP2ServingError("wavefront rows must lie above M and at or below 2M")
        lane_a_rows, lane_b_rows = self._split_rows(rows)
        lane_a_bound = False
        try:
            self._lane_a.bind_batch(lane_a_rows)
            lane_a_bound = True
            self._lane_b.bind_batch(lane_b_rows)
        except BaseException:
            self._phase_failed = True
            if lane_a_bound:
                self._lane_a.complete_bound(replay=False)
            raise
        try:
            self._lane_a.complete_bound(replay=False)
            self._lane_b.complete_bound(replay=False)
            profiler = self._feeding_profiler
            if profiler is not None and not profiler.active:
                profiler = None
            current = torch.cuda.current_stream(self.body.hidden.device)
            self.slot.stream.wait_stream(current)
            replay_started = getattr(self, "_replay_started", None)
            replay_finished = getattr(self, "_replay_finished", None)
            if replay_started is not None:
                replay_started.record(self.slot.stream)
            if profiler is not None:
                with torch.cuda.stream(self.slot.stream):
                    profiler.begin_cuda("graph_replay_gpu_ms")
            self.slot.replay()
            if profiler is not None:
                with torch.cuda.stream(self.slot.stream):
                    profiler.end_cuda("graph_replay_gpu_ms")
            if replay_finished is not None:
                replay_finished.record(self.slot.stream)
                self._replay_timing_pending = True
            current.wait_stream(self.slot.stream)
        except BaseException:
            self._phase_failed = True
            raise
        self._active_rows = logical_m
        self._lane_a_rows = len(lane_a_rows)
        if commit_hidden:
            self._commit_target_hidden(rows)

    def _commit_target_hidden(self, rows: tuple[_BatchRow, ...]) -> None:
        self._lane_a._commit_target_hidden(rows[:self._lane_a_rows])
        self._lane_b._commit_target_hidden(rows[self._lane_a_rows:])

    def replay_elapsed_ms(self) -> float:
        """Return the last wavefront graph's GPU time at its existing sync boundary."""

        if not self._replay_timing_pending:
            raise Qwen3_5TP2PP2ServingError("target graph timing has no completed replay")
        milliseconds = self._replay_started.elapsed_time(self._replay_finished)
        self._replay_timing_pending = False
        return milliseconds

    def _sampling_logits(self, active_rows: int) -> Tensor | None:
        """Project two fixed physical lanes into request-logical row order."""

        physical = self.slot.logits
        if physical is None or active_rows == self.captured_m:
            return physical
        logical = self._logical_logits
        if logical is None:
            raise Qwen3_5TP2PP2ServingError("final wave rank has no logical-logits workspace")
        first = self._lane_a_rows
        if first is None:
            raise Qwen3_5TP2PP2ServingError("wavefront sampling has no bound lane split")
        second = active_rows - first
        logical[:first].copy_(physical[:first])
        logical[first:active_rows].copy_(physical[self.micro_m : self.micro_m + second])
        return logical

    def sample_tokens(self, batch: int | tuple[_BatchRow, ...]) -> tuple[int, ...]:
        greedy = type(batch) is int
        rows = None if greedy else cast(tuple[_BatchRow, ...], batch)
        active_rows = cast(int, batch) if rows is None else len(rows)
        if self._phase_failed:
            raise Qwen3_5TP2PP2ServingError("wavefront rank is poisoned")
        if self._active_rows != active_rows:
            self._phase_failed = True
            raise Qwen3_5TP2PP2ServingError(
                "wavefront sampling rows differ from the completed replay"
            )
        profiler = self._feeding_profiler
        if profiler is not None and not profiler.active:
            profiler = None
        try:
            if greedy:
                token_ids = gather_greedy_broadcast_tp(
                    local_logits=self._sampling_logits(active_rows),
                    local_vocab_span=self.slot.vocab_span,
                    final_ranks=self._final_ranks,
                    final_vocab_spans=self._final_spans,
                    true_vocab_size=self.vocab_size,
                    active_rows=active_rows,
                    token_ids=self._token_ids,
                    local_max_scores=self._local_max_scores,
                    gathered_max_scores=self._gathered_max_scores,
                    gathered_token_ids=self._gathered_token_ids,
                    final_tp_group=self._final_tp_group,
                    world_group=self._world_group,
                    device=self.body.hidden.device,
                    profiler=profiler,
                )
            else:
                assert rows is not None
                all_random_rows = self._sampling.bind(rows)
                sampling_logits = self._sampling_logits(active_rows)
                token_ids = sample_sharded_broadcast_tp(
                    local_logits=sampling_logits,
                    local_vocab_span=self.slot.vocab_span,
                    final_ranks=self._final_ranks,
                    final_vocab_spans=self._final_spans,
                    true_vocab_size=self.vocab_size,
                    active_rows=active_rows,
                    token_ids=self._token_ids,
                    parameters=self._sampling.parameters,
                    workspace=self._sampling.workspace,
                    final_tp_group=self._final_tp_group,
                    world_group=self._world_group,
                    device=self.body.hidden.device,
                    root_sampler=self._sampling.root_sampler,
                    root_valid=self._sampling.root_valid,
                    all_random_rows=all_random_rows,
                    all_greedy_rows=self._sampling.all_greedy,
                    greedy_plan=self._sampling.bound_plan(sampling_logits),
                    profiler=profiler,
                )
            if profiler is not None:
                for binder in (self._lane_a, self._lane_b):
                    binder_profiler = binder._feeding_profiler
                    assert binder_profiler is not None
                    binder_profiler.flush_cuda(("production_bind_gpu_ms",))
                profiler.flush_cuda(("graph_replay_gpu_ms",))
            self._active_rows = None
            self._lane_a_rows = None
            if self._rank != self._sampler_rank:
                return ()
            d2h_started_ns = time.monotonic_ns() if profiler is not None else 0
            result = tuple(int(token) for token in token_ids[:active_rows].cpu().tolist())
            if profiler is not None:
                profiler.record_cpu(
                    "sampled_token_d2h_cpu_ms",
                    (time.monotonic_ns() - d2h_started_ns) / 1_000_000,
                )
            return result
        except BaseException:
            self._phase_failed = True
            raise

    def gather_candidates(self, rows: tuple[_BatchRow, ...]) -> None:
        """Collect target candidates in logical segment order for MTP verification."""

        if self._phase_failed or self._active_rows != len(rows):
            self._phase_failed = True
            raise Qwen3_5TP2PP2ServingError(
                "wavefront candidate rows differ from the completed replay"
            )
        profiler = self._feeding_profiler
        if profiler is not None and not profiler.active:
            profiler = None
        all_random_rows = self._sampling.bind(rows)
        sample_sharded_broadcast_tp(
            local_logits=self._sampling_logits(len(rows)),
            local_vocab_span=self.slot.vocab_span,
            final_ranks=self._final_ranks,
            final_vocab_spans=self._final_spans,
            true_vocab_size=self.vocab_size,
            active_rows=len(rows),
            token_ids=self._token_ids,
            parameters=self._sampling.parameters,
            workspace=self._sampling.workspace,
            final_tp_group=self._final_tp_group,
            world_group=self._world_group,
            device=self.body.hidden.device,
            root_sampler=self._sampling.root_sampler,
            root_valid=self._sampling.root_valid,
            all_random_rows=all_random_rows,
            profiler=profiler,
            candidates_only=True,
            greedy_plan=(
                self._sampling.bound_plan(self._sampling_logits(len(rows)))
                if self._sampling.constraint_bound
                else None
            ),
        )
        if profiler is not None:
            for binder in (self._lane_a, self._lane_b):
                binder_profiler = binder._feeding_profiler
                assert binder_profiler is not None
                binder_profiler.flush_cuda(("production_bind_gpu_ms",))
            profiler.flush_cuda(("graph_replay_gpu_ms",))
        self._active_rows = None

    def norm_row(self, logical_row: int) -> Tensor:
        """Return one logical target row from its physical lane."""

        split = self._lane_a_rows
        if split is None or not 0 <= logical_row < self.captured_m:
            raise Qwen3_5TP2PP2ServingError("wavefront norm row has no live lane mapping")
        return (
            self._lane_a.body.buffers.norm[logical_row]
            if logical_row < split
            else self._lane_b.body.buffers.norm[logical_row - split]
        )


def _migration_logical_ms(
    capacities: tuple[int, ...] = _PHYSICAL_GRAPH_BUCKET_CAPACITIES,
) -> tuple[int, ...]:
    """Use each bucket's first dispatchable M while one owner crosses all buckets."""

    return (1, *(capacity + 1 for capacity in capacities[:-1]))


def _tracked_snapshot(body: Qwen3_5TP2PP2DecodeBody) -> dict[str, Tensor]:
    """Snapshot only migration row zero's durable destination and output."""

    buffers = body.buffers
    page = int(buffers.page_destination[0].item())
    local_layers = buffers.conv_slots.shape[1]
    state = int(buffers.conv_slots[1, 0, 0].item()) // local_layers
    values = {
        "hidden": body.hidden[:1],
        "kv_key": body.arena.key[page : page + 1],
        "kv_value": body.arena.value[page : page + 1],
        "kv_written": body.arena.written[page : page + 1],
        "conv": body.arena.states[CONV_STATE_ID][state : state + 1],
        "recurrent": body.arena.states[RECURRENT_STATE_ID][state : state + 1],
        "guard": buffers.guard,
    }
    values.update({f"kv_field:{name}": value[page:page + 1]
                   for name, value in body.arena.page_fields.items()})
    values.update({f"state:{name}": value[state:state + 1]
                   for name, value in body.arena.states.items()
                   if name not in (CONV_STATE_ID, RECURRENT_STATE_ID)})
    if body.logits is not None:
        values["logits"] = body.logits[:1]
    return {name: value.detach().to("cpu", copy=True) for name, value in values.items()}


def _continuation_audit_rows(body: Qwen3_5TP2PP2DecodeBody, length: int) -> tuple[_BatchRow, ...]:
    """Build one real ``(L, 0, ...)`` prefill transaction."""

    if type(length) is not int or not 1 <= length <= body.captured_m:
        raise ValueError("continuation audit length exceeds its graph capacity")
    block_size = body.arena.layout.block_size_tokens
    page_count = (length + block_size - 1) // block_size
    destinations = tuple(2 * page + 1 for page in range(page_count))
    table = (
        *destinations,
        *(destinations[-1] for _ in range(body.buffers.page_table.shape[1] - page_count)),
    )
    rows: list[_BatchRow] = []
    for position in range(length):
        page = position // block_size
        first_on_page = position % block_size == 0
        rows.append(
            _BatchRow(
                request_id=f"audit:continuation:L{length}",
                token_id=17 + position,
                position=position,
                state_source_slot=0,
                state_destination_slot=1,
                page_source=2 * page if first_on_page else destinations[page],
                page_destination=destinations[page],
                page_table=table,
                reset_state=position == 0,
                new_pages=destinations if position == 0 else (),
                segment_length=length if position == 0 else 0,
            )
        )
    return tuple(rows)


def _continuation_snapshot(body: Qwen3_5TP2PP2DecodeBody, length: int) -> dict[str, Tensor]:
    block_size = body.arena.layout.block_size_tokens
    pages = torch.tensor(
        tuple(2 * page + 1 for page in range((length + block_size - 1) // block_size)),
        dtype=torch.int64,
        device=body.hidden.device,
    )
    values = {
        "hidden": body.hidden[:length],
        "kv_key": body.arena.key.index_select(0, pages),
        "kv_value": body.arena.value.index_select(0, pages),
        "kv_written": body.arena.written.index_select(0, pages),
        "conv_source": body.arena.states[CONV_STATE_ID][:1],
        "conv_final": body.arena.states[CONV_STATE_ID][1:2],
        "recurrent_source": body.arena.states[RECURRENT_STATE_ID][:1],
        "recurrent_final": body.arena.states[RECURRENT_STATE_ID][1:2],
        "guard": body.buffers.guard,
    }
    values.update({f"kv_field:{name}": value.index_select(0, pages)
                   for name, value in body.arena.page_fields.items()})
    for name, value in body.arena.states.items():
        if name not in (CONV_STATE_ID, RECURRENT_STATE_ID):
            values[f"state:{name}:source"] = value[:1]
            values[f"state:{name}:final"] = value[1:2]
    if body.logits is not None:
        values["logits"] = body.logits[:length]
    return {name: value.detach().to("cpu", copy=True) for name, value in values.items()}


def _run_continuation_audit(
    executor: _CapturedGraphRank, length: int, *, graph: bool
) -> dict[str, Tensor]:
    body = executor.body
    _reset_owner(body, body.buffers.page_table.shape[1])
    _reset_production_audit_executor(executor)
    rows = _continuation_audit_rows(body, length)
    _dirty_production_audit_destinations(body, rows)
    if graph:
        executor.replay_batch(rows)
        executor.slot.synchronize()
    else:
        body.bind_scratch(executor.resource)
        _direct_bind_production_audit_rows(body, rows)
        barrier()
        body.launch()
        torch.cuda.synchronize(body.hidden.device)
    barrier()
    expected_segments = torch.tensor(
        (length, *(0 for _ in range(length - 1))),
        dtype=torch.int32,
        device=body.hidden.device,
    )
    if not torch.equal(body.buffers.segment_lengths[:length], expected_segments):
        raise Qwen3_5TP2PP2ServingError("continuation segment layout changed")
    snapshot = _continuation_snapshot(body, length)
    if any(
        bool(snapshot[name].view(torch.int16).numpy().any())
        for name in ("conv_source", "recurrent_source")
    ):
        raise Qwen3_5TP2PP2ServingError("continuation mutated its source state")
    return snapshot


def _steady_audit_rows(
    body: Qwen3_5TP2PP2DecodeBody,
    logical_m: int,
    *,
    position: int,
) -> tuple[_BatchRow, ...]:
    """Build one owned decode row per request for steady-path qualification."""

    block_size = body.arena.layout.block_size_tokens
    if not 0 < logical_m <= body.captured_m or position not in (0, 1, block_size - 1, block_size):
        raise ValueError("steady audit case lies outside its fixed corpus")
    rows: list[_BatchRow] = []
    for physical in range(logical_m):
        first_page, next_page = 2 * physical, 2 * physical + 1
        crossed_page = position >= block_size
        current_page = next_page if crossed_page else first_page
        rows.append(
            _BatchRow(
                request_id=f"audit:steady:row:{physical}",
                token_id=17 + position + physical,
                position=position,
                state_source_slot=physical,
                state_destination_slot=physical,
                page_source=current_page,
                page_destination=current_page,
                page_table=(first_page, next_page) if crossed_page else (first_page,),
                reset_state=position in (0, block_size - 1),
                new_pages=(current_page,) if position in (0, block_size - 1, block_size) else (),
                segment_length=1,
                is_decode=True,
            )
        )
    return tuple(rows)


def _steady_audit_snapshot(
    body: Qwen3_5TP2PP2DecodeBody,
    logical_m: int,
    *,
    crossed_page: bool,
) -> dict[str, Tensor]:
    device = body.hidden.device
    buffers = body.buffers
    page_columns = 2 if crossed_page else 1
    pages = slice(0, 2 * logical_m)
    sampled_states = torch.tensor(
        tuple(dict.fromkeys((0, logical_m // 2, logical_m - 1))),
        dtype=torch.int64,
        device=device,
    )
    values = {
        "hidden": body.hidden[:logical_m],
        "kv_key": body.arena.key[pages],
        "kv_value": body.arena.value[pages],
        "kv_written": body.arena.written[pages],
        "conv": body.arena.states[CONV_STATE_ID].index_select(0, sampled_states),
        "recurrent": body.arena.states[RECURRENT_STATE_ID].index_select(
            0, sampled_states
        ),
        "position_ids": buffers.position_ids[:, :logical_m],
        "page_source": buffers.page_source[:logical_m],
        "page_destination": buffers.page_destination[:logical_m],
        "token_offsets": buffers.token_offsets[:logical_m],
        "sequence_lengths": buffers.sequence_lengths[:logical_m],
        "segment_lengths": buffers.segment_lengths[:logical_m],
        "page_table": buffers.page_table[:logical_m, :page_columns],
        "guard": buffers.guard,
    }
    values.update({f"kv_field:{name}": value[pages]
                   for name, value in body.arena.page_fields.items()})
    values.update({f"state:{name}": value.index_select(0, sampled_states)
                   for name, value in body.arena.states.items()
                   if name not in (CONV_STATE_ID, RECURRENT_STATE_ID)})
    if body.logits is not None:
        values["logits"] = body.logits[:logical_m]
    return {name: value.detach().to("cpu", copy=True) for name, value in values.items()}


def _run_steady_continuation_audit(
    executor: _CapturedGraphRank,
    logical_m: int,
    *,
    crossed_page: bool,
    continuation: bool,
) -> dict[str, Tensor]:
    """Run two identical decode steps through full bind or sparse continuation."""

    body = executor.body
    first_position = body.arena.layout.block_size_tokens - 1 if crossed_page else 0
    second_position = first_position + 1
    first = _steady_audit_rows(body, logical_m, position=first_position)
    second = _steady_audit_rows(body, logical_m, position=second_position)
    _reset_owner(body, body.buffers.page_table.shape[1])
    _reset_production_audit_executor(executor)
    _dirty_production_audit_destinations(body, first)
    executor.replay_batch(first)
    executor.slot.synchronize()
    if continuation:
        tokens = torch.tensor(
            tuple(row.token_id for row in second),
            dtype=torch.int64,
            device=body.hidden.device,
        )
        executor._token_ids[:logical_m].copy_(tokens)
        updates = (
            tuple((physical, 1, 2 * physical + 1) for physical in range(logical_m))
            if crossed_page
            else ()
        )
        executor.replay_steady_continuation(logical_m, updates)
    else:
        executor.replay_batch(second)
    executor.slot.synchronize()
    barrier()
    return _steady_audit_snapshot(body, logical_m, crossed_page=crossed_page)


def _shared_fixture_rows(body,rows):
    """Use exact disjoint byte domains for pre-readiness fixture resources."""
    storage=getattr(body.arena,"shared_storage",None)
    if storage is None:return rows
    owners=sorted({v for row in rows for v in (row.state_source_slot,row.state_destination_slot)})
    mapping={old:body.arena.spec.state_slot_capacity-1-i for i,old in enumerate(owners)}
    page_end=1+max(p for row in rows for p in (row.page_source,row.page_destination,*row.page_table))
    state_bytes=0 if storage.state_layout is None else storage.state_layout.entry_bytes
    if (len(owners)>body.arena.spec.state_slot_capacity
            or len(owners)*state_bytes+page_end*storage.page_layout.entry_bytes>storage.budget_bytes):
        raise ValueError("fixture resource ownership exceeds the shared byte budget")
    return tuple(replace(row,state_source_slot=mapping[row.state_source_slot],
                         state_destination_slot=mapping[row.state_destination_slot]) for row in rows)


def _production_audit_rows(
    body: Qwen3_5TP2PP2DecodeBody,
    logical_m: int,
    *,
    position: int,
    parity: int,
    initialize_from: int | None,
    allow_underfilled: bool = False,
) -> tuple[_BatchRow, ...]:
    """Materialize request rows without sharing production staging code."""

    domain = getattr(body, "bucket_logical_ms", None)
    logical_ms = tuple(domain) if domain is not None else _logical_m_values(body.captured_m)
    if not 0 < logical_m <= body.captured_m or (
        not allow_underfilled and logical_m not in logical_ms
    ):
        raise ValueError("production audit logical M lies outside the bucket")
    if initialize_from is not None and not 0 <= initialize_from <= logical_m:
        raise ValueError("production audit initialization boundary is invalid")
    table_pages = body.buffers.page_table.shape[1]
    rows: list[_BatchRow] = []
    for physical in range(logical_m):
        page_source, page_destination = _audit_cow_pair(physical, parity)
        state_source, state_destination = _audit_state_pair(body, physical, parity)
        initialize = initialize_from is not None and physical >= initialize_from
        rows.append(
            _BatchRow(
                request_id=f"audit:row:{physical}",
                token_id=17 + physical,
                position=position,
                state_source_slot=state_source,
                state_destination_slot=state_destination,
                page_source=page_source,
                page_destination=page_destination,
                page_table=(page_destination,) * table_pages,
                reset_state=initialize,
                new_pages=(page_destination,) if initialize else (),
            )
        )
    return _shared_fixture_rows(body,tuple(rows))


def _dirty_production_audit_destinations(
    body: Qwen3_5TP2PP2DecodeBody,
    rows: tuple[_BatchRow, ...],
) -> None:
    """Poison recycled destinations before either independent execution path."""

    device = body.hidden.device
    pages = torch.tensor(
        tuple(row.page_destination for row in rows), dtype=torch.int64, device=device
    )
    states = torch.tensor(
        tuple(
            row.state_destination_slot
            for row in rows
            if row.state_destination_slot != row.state_source_slot
        ),
        dtype=torch.int64,
        device=device,
    )
    body.arena.key.index_fill_(0, pages, 11)
    body.arena.value.index_fill_(
        0, pages, 243 if body.arena.value.dtype == torch.uint8 else -13)
    body.arena.written.index_fill_(0, pages, True)
    for cache_field in body.arena.page_fields.values():
        cache_field.index_fill_(0, pages, 17)
    if states.numel():
        for state in body.arena.states.values():
            state.index_fill_(0, states, 0.25)


def _review_audit_rows(body: ServingDecodeBody, logical_m: int) -> tuple[_BatchRow, ...]:
    """Fresh multi-token transactions, with one durable owner per request."""

    owners = body.arena.spec.state_slot_capacity
    depth = (body.captured_m + owners - 1) // owners
    rows: list[_BatchRow] = []
    for start in range(0, logical_m, depth):
        owner = start // depth
        length = min(depth, logical_m - start)
        for offset in range(length):
            physical = start + offset
            rows.append(_BatchRow(
                request_id=f"audit:review:{owner}", token_id=17 + physical,
                position=offset, state_source_slot=owner,
                state_destination_slot=(owner if offset == length - 1 else owners + physical),
                page_source=owner, page_destination=owner,
                page_table=(owner,) * body.buffers.page_table.shape[1],
                reset_state=offset == 0, new_pages=(owner,) if offset == 0 else (),
                segment_length=length if offset == 0 else 0, is_decode=True,
            ))
    return tuple(rows)


def _audit_review_capacity(executor: _CapturedGraphRank) -> str:
    """Qualify wide review graphs without inventing extra request owners."""

    body = executor.body
    cases = _bounded_audit_logical_ms(tuple(body.bucket_logical_ms))
    for logical_m in cases:
        snapshots: list[dict[str, Tensor]] = []
        rows = _review_audit_rows(body, logical_m)
        for graph, poison in ((False, 0), (True, 0), (True, 1)):
            _reset_owner(body, body.buffers.page_table.shape[1])
            _reset_production_audit_executor(executor)
            _poison_production_audit_scratch(executor, poison)
            if graph:
                executor.replay_batch(rows)
                executor.slot.synchronize()
                if not _padding_is_bound_to_scratch(executor, logical_m):
                    raise Qwen3_5TP2PP2ServingError("review padding escaped private scratch")
            else:
                body.bind_scratch(executor.resource)
                _direct_bind_production_audit_rows(body, rows)
                barrier()
                body.launch()
                torch.cuda.synchronize(body.hidden.device)
            barrier()
            snapshots.append({**_tracked_snapshot(body), **{
                f"boundary:{key}": value
                for key, value in _active_snapshot(body, logical_m).items()
            }})
        if not all(_exact_snapshot(snapshots[0], value) for value in snapshots[1:]):
            raise Qwen3_5TP2PP2ServingError("wide review differs from eager or padding poison")
    _reset_owner(body, body.buffers.page_table.shape[1])
    _reset_production_audit_executor(executor)
    return stable_digest({"schema": "cubic.review-capacity.v1", "capacity": body.captured_m,
                          "owners": body.arena.spec.state_slot_capacity, "logical_ms": cases})


def _direct_bind_production_audit_rows(
    body: Qwen3_5TP2PP2DecodeBody,
    rows: tuple[_BatchRow, ...],
) -> None:
    """Bind the eager oracle directly, independent of bulk host staging."""

    buffers, arena = body.buffers, body.arena
    local_layers = buffers.conv_slots.shape[1]
    local_layer_offsets = torch.arange(local_layers, dtype=torch.int32)
    page_tables = torch.tensor(tuple(row.page_table for row in rows), dtype=torch.int32)
    state_sources = torch.tensor(
        tuple(row.state_source_slot for row in rows), dtype=torch.int32
    )
    state_destinations = torch.tensor(
        tuple(row.state_destination_slot for row in rows), dtype=torch.int32
    )
    source_slots = state_sources[:, None] * local_layers + local_layer_offsets[None, :]
    destination_slots = (
        state_destinations[:, None] * local_layers + local_layer_offsets[None, :]
    )
    buffers.page_table[: len(rows)].copy_(page_tables)
    buffers.conv_slots[0, :, : len(rows)].copy_(source_slots.T)
    buffers.conv_slots[1, :, : len(rows)].copy_(destination_slots.T)
    buffers.recurrent_slots[0, :, : len(rows)].copy_(source_slots.T)
    buffers.recurrent_slots[1, :, : len(rows)].copy_(destination_slots.T)
    for physical, row in enumerate(rows):
        if row.reset_state:
            for state in arena.states.values():
                state[row.state_source_slot].zero_()
        for page in row.new_pages:
            arena.key[page].zero_()
            arena.value[page].zero_()
            arena.written[page].zero_()
            for cache_field in arena.page_fields.values():
                cache_field[page].zero_()
        buffers.position_ids[:, physical].fill_(row.position)
        buffers.page_source[physical].fill_(row.page_source)
        buffers.page_destination[physical].fill_(row.page_destination)
        buffers.token_offsets[physical].fill_(row.position % arena.layout.block_size_tokens)
        buffers.sequence_lengths[physical].fill_(row.position + 1)
        buffers.segment_lengths[physical].fill_(row.segment_length)
        if body.stage == 0:
            span = getattr(body, "embedding_input_span", body.embedding_span)
            if span is None or body.embedding_mask is None:
                raise Qwen3_5TP2PP2ServingError("first stage has no embedding shard")
            owned = span.start <= row.token_id < span.stop
            buffers.token_ids[physical].fill_(row.token_id - span.start if owned else 0)
            body.embedding_mask[physical].fill_(owned)
    buffers.segment_lengths[len(rows) :].zero_()
    buffers.row_activity[: len(rows)].fill_(1)
    buffers.row_activity[len(rows) :].zero_()


def _padding_is_bound_to_scratch(
    executor: _CapturedGraphRank,
    logical_m: int,
) -> bool:
    """Verify every ownerless row names only this bucket's private scratch."""

    body, buffers = executor.body, executor.body.buffers
    if logical_m == executor.captured_m:
        return True
    rows = slice(logical_m, executor.captured_m)
    pages = torch.tensor(executor._scratch_pages[logical_m:], dtype=torch.int32)
    local_layers = torch.arange(buffers.conv_slots.shape[1], dtype=torch.int32)
    active_rows = slice(0, logical_m)
    active_pages = torch.cat(
        (
            buffers.page_source[active_rows].detach().to("cpu", copy=True),
            buffers.page_destination[active_rows].detach().to("cpu", copy=True),
            buffers.page_table[active_rows].detach().to("cpu", copy=True).reshape(-1),
        )
    )
    position_ids = buffers.position_ids[:, rows].detach().to("cpu", copy=True)
    page_source = buffers.page_source[rows].detach().to("cpu", copy=True)
    page_destination = buffers.page_destination[rows].detach().to("cpu", copy=True)
    token_offsets = buffers.token_offsets[rows].detach().to("cpu", copy=True)
    sequence_lengths = buffers.sequence_lengths[rows].detach().to("cpu", copy=True)
    segment_lengths = buffers.segment_lengths[rows].detach().to("cpu", copy=True)
    page_table = buffers.page_table[rows].detach().to("cpu", copy=True)
    conv_slots = buffers.conv_slots[:, :, rows].detach().to("cpu", copy=True)
    recurrent_slots = buffers.recurrent_slots[:, :, rows].detach().to("cpu", copy=True)
    states = torch.div(
        conv_slots[0, 0], local_layers.numel(), rounding_mode="floor"
    )
    slots = (states[:, None] * local_layers.numel() + local_layers[None, :]).T
    active_state_slots = torch.cat(
        (
            buffers.conv_slots[:, :, active_rows].detach().to("cpu", copy=True).reshape(-1),
            buffers.recurrent_slots[:, :, active_rows]
            .detach()
            .to("cpu", copy=True)
            .reshape(-1),
        )
    )
    scratch_start = executor._scratch_states[0]
    scratch_stop = scratch_start + len(executor._scratch_states)
    checks = {
        "positions": not bool(position_ids.any()),
        "page_source": torch.equal(page_source, pages),
        "page_destination": torch.equal(page_destination, pages),
        "token_offsets": not bool(token_offsets.any()),
        "sequence_lengths": bool(torch.all(sequence_lengths == 1)),
        "segment_lengths": not bool(segment_lengths.any()),
        "unique_pages": int(torch.unique(pages).numel()) == pages.numel(),
        "scratch_pages": bool(torch.all(pages >= body.arena.spec.storage_page_capacity)),
        "active_pages": bool(torch.all(active_pages < body.arena.spec.page_capacity)),
        "page_table": torch.equal(
            page_table,
            pages[:, None].expand(-1, buffers.page_table.shape[1]),
        ),
        "unique_states": int(torch.unique(states).numel()) == states.numel(),
        "scratch_states": bool(
            torch.all((states >= scratch_start) & (states < scratch_stop))
        ),
        "active_states": not bool(torch.isin(slots.reshape(-1), active_state_slots).any()),
        "conv_slots": torch.equal(conv_slots, slots.unsqueeze(0).expand(2, -1, -1)),
        "recurrent_slots": torch.equal(
            recurrent_slots, slots.unsqueeze(0).expand(2, -1, -1)
        ),
    }
    valid = all(checks.values())
    if body.stage == 0:
        assert body.embedding_mask is not None
        token_ids = buffers.token_ids[rows].detach().to("cpu", copy=True)
        embedding_mask = body.embedding_mask[rows].detach().to("cpu", copy=True)
        valid = (
            valid
            and not bool(token_ids.any())
            and not bool(embedding_mask.any())
        )
        checks["token_ids"] = not bool(token_ids.any())
        checks["embedding_mask"] = not bool(embedding_mask.any())
    if not valid:
        diagnostic_event(_LOGGER, 'performance.padding_scratch_binding_failed_rank_d_c',
            body.rank,
            body.captured_m,
            logical_m,
            ",".join(name for name, passed in checks.items() if not passed),
        )
    return valid


def _bind_direct_padding_to_unused_scratch(
    executor: _CapturedGraphRank,
    logical_m: int,
) -> None:
    """Give the independent eager oracle the same ownerless-row guarantee."""

    if logical_m == executor.captured_m:
        return
    buffers = executor.body.buffers
    local_layers = buffers.conv_slots.shape[1]
    active_slots = buffers.conv_slots[:, :, :logical_m].reshape(-1)
    candidates = torch.tensor(
        executor._scratch_states,
        dtype=torch.int32,
        device=buffers.conv_slots.device,
    )
    active_states = torch.div(active_slots, local_layers, rounding_mode="floor")
    available = candidates[~torch.isin(candidates, active_states)]
    padding_count = executor.captured_m - logical_m
    if available.numel() < padding_count:
        raise Qwen3_5TP2PP2ServingError(
            "eager audit transaction state exhausts padding scratch"
        )
    states = available[:padding_count]
    layers = torch.arange(local_layers, dtype=torch.int32, device=states.device)
    slots = (states[:, None] * local_layers + layers[None, :]).T
    rows = slice(logical_m, executor.captured_m)
    buffers.conv_slots[:, :, rows].copy_(slots.unsqueeze(0).expand(2, -1, -1))
    buffers.recurrent_slots[:, :, rows].copy_(slots.unsqueeze(0).expand(2, -1, -1))


def _bound_wave_padding_scratch_indices(
    executor: _CapturedWavefrontRank,
    split: tuple[int, int],
) -> tuple[Tensor, Tensor]:
    """Read the exact private padding backing installed for this wave."""

    page_parts: list[Tensor] = []
    state_parts: list[Tensor] = []
    for binder, logical_m in zip(
        (executor._lane_a, executor._lane_b), split, strict=True
    ):
        if logical_m == binder.captured_m:
            continue
        buffers = binder.body.buffers
        rows = slice(logical_m, binder.captured_m)
        page_parts.append(buffers.page_destination[rows].to(torch.int64))
        state_parts.append(
            torch.div(
                buffers.conv_slots[0, 0, rows],
                buffers.conv_slots.shape[1],
                rounding_mode="floor",
            ).to(torch.int64)
        )
    device = executor.body.hidden.device
    if not page_parts:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return empty, empty
    return torch.unique(torch.cat(page_parts)), torch.unique(torch.cat(state_parts))


def _poison_production_audit_scratch(
    executor: _CapturedGraphRank,
    poison: int,
) -> None:
    """Change only private scratch backing, never active graph input metadata."""

    if poison not in (0, 1):
        raise ValueError("production audit padding poison must be zero or one")
    body = executor.body
    device = body.hidden.device
    pages = torch.tensor(executor._scratch_pages, device=device, dtype=torch.int64)
    states = torch.tensor(executor._scratch_states, device=device, dtype=torch.int64)
    value = (37.0, -73.0)[poison]
    body.arena.key.index_fill_(0, pages, _storage_poison_value(body.arena.key, value))
    body.arena.value.index_fill_(0, pages, _storage_poison_value(body.arena.value, -value))
    body.arena.written.index_fill_(0, pages, True)
    body.arena.states[CONV_STATE_ID].index_fill_(0, states, value / 4)
    body.arena.states[RECURRENT_STATE_ID].index_fill_(0, states, value / 16)
    for name, state in body.arena.states.items():
        if name not in (CONV_STATE_ID, RECURRENT_STATE_ID):
            state.index_fill_(0, states, _storage_poison_value(state, value / 4))
    for cache_field in body.arena.page_fields.values():
        cache_field.index_fill_(0, pages, _storage_poison_value(cache_field, value / 8))


def _reset_production_audit_executor(executor: _CapturedGraphRank) -> None:
    """Return host-ring and graph inputs to their pre-serving startup state."""

    executor.slot.synchronize()
    executor._bound_active_rows = 0
    executor._next_host_slot = 0
    executor._host_event_recorded[:] = (False, False)
    executor._batch_bound = False
    executor._batch_phase_failed = False
    executor.body.prepare_scratch(executor.resource)
    executor.body.buffers.segment_lengths.zero_()
    executor.body.buffers.row_activity.zero_()


def _wave_audit_rows(
    executor: _CapturedWavefrontRank,
    logical_m: int,
) -> tuple[_BatchRow, ...]:
    """Build disjoint global-owner rows independently of production staging."""

    if not executor.micro_m < logical_m <= executor.captured_m:
        raise ValueError("wave audit logical M lies outside the dual-lane domain")
    table_pages = executor.body.buffers.page_table.shape[1]
    if not executor.speculative:
        return tuple(
            _BatchRow(
                request_id=f"audit:wave:row:{physical}",
                token_id=17 + physical,
                position=0,
                state_source_slot=2 * physical,
                state_destination_slot=2 * physical + 1,
                page_source=2 * physical,
                page_destination=2 * physical + 1,
                page_table=(2 * physical + 1,) * table_pages,
                reset_state=True,
                new_pages=(2 * physical + 1,),
            )
            for physical in range(logical_m)
        )
    rows: list[_BatchRow] = []
    scratch = executor.body.arena.spec.state_slot_capacity
    for request in range((logical_m + 1) // 2):
        length = min(2, logical_m - len(rows))
        source_page, destination_page = 2 * request, 2 * request + 1
        for offset in range(length):
            rows.append(
                _BatchRow(
                    request_id=f"audit:wave:request:{request}",
                    token_id=17 + len(rows),
                    position=offset,
                    state_source_slot=request,
                    state_destination_slot=(
                        scratch + request if length == 2 and offset == 0 else request
                    ),
                    page_source=source_page if offset == 0 else destination_page,
                    page_destination=destination_page,
                    page_table=(destination_page,) * table_pages,
                    reset_state=offset == 0,
                    new_pages=(destination_page,) if offset == 0 else (),
                    segment_length=length if offset == 0 else 0,
                    is_decode=True,
                )
            )
    return tuple(rows)


def _reset_wave_audit_executor(executor: _CapturedWavefrontRank) -> None:
    """Reset only startup audit state; production never recovers a poisoned wave."""

    executor.slot.synchronize()
    for binder in (executor._lane_a, executor._lane_b):
        binder._bound_active_rows = 0
        binder._next_host_slot = 0
        binder._host_event_recorded[:] = (False, False)
        binder._batch_bound = False
        binder._batch_phase_failed = False
        # Rebind fixed padding metadata without clearing the full physical-M
        # scratch arena.  The following logical-M owner reset clears precisely
        # the shared scratch suffix reachable by either balanced lane.
        binder.body.bind_scratch(binder.resource)
        binder.body.buffers.segment_lengths.zero_()
        binder.body.buffers.row_activity.zero_()
    executor._phase_failed = False
    executor._active_rows = None
    executor._lane_a_rows = None


def _wave_audit_scratch_indices(
    executor: _CapturedWavefrontRank,
    logical_m: int,
) -> tuple[Tensor, Tensor]:
    """Return the shared scratch union addressed by either balanced lane."""

    if not executor.micro_m < logical_m <= executor.captured_m:
        raise ValueError("wave audit logical M lies outside the dual-lane domain")
    # lane B has floor(M/2) active rows and therefore the larger padding suffix;
    # lane A's ceil(M/2) padding is a subset of the same physical scratch rows.
    lane_a_rows = getattr(executor, "_lane_a_rows", None)
    first_padding = (
        min(lane_a_rows, logical_m - lane_a_rows)
        if lane_a_rows is not None
        else logical_m // 2
    )
    device = executor.body.hidden.device
    return (
        torch.tensor(
            executor._lane_a._scratch_pages[first_padding:],
            device=device,
            dtype=torch.int64,
        ),
        torch.tensor(
            executor._lane_a._scratch_states[first_padding:],
            device=device,
            dtype=torch.int64,
        ),
    )


def _reset_wave_audit_owners(
    executor: _CapturedWavefrontRank,
    logical_m: int,
    *,
    scratch_logical_m: int | None = None,
) -> None:
    """Clear only owners and shared scratch reachable by this audit M."""

    body = executor.body
    pages, states = _wave_audit_scratch_indices(
        executor, logical_m if scratch_logical_m is None else scratch_logical_m
    )
    state_owner_count = (
        (logical_m + 1) // 2 if executor.speculative else 2 * logical_m
    )
    page_owner_count = (
        2 * ((logical_m + 1) // 2) if executor.speculative else 2 * logical_m
    )
    if (
        body.arena.spec.state_slot_capacity < state_owner_count
        or body.arena.spec.page_capacity < page_owner_count
    ):
        raise Qwen3_5TP2PP2ServingError(
            "dual-wave cache cannot cover committed and COW owner slots"
        )
    body.arena.key[:page_owner_count].zero_()
    body.arena.value[:page_owner_count].zero_()
    body.arena.written[:page_owner_count].zero_()
    for state in body.arena.states.values():
        state[:state_owner_count].zero_()
    if pages.numel():
        body.arena.key.index_fill_(0, pages, 0)
        body.arena.value.index_fill_(0, pages, 0)
        body.arena.written.index_fill_(0, pages, False)
        for state in body.arena.states.values():
            state.index_fill_(0, states, 0)


def _poison_wave_audit_scratch(
    executor: _CapturedWavefrontRank,
    split: tuple[int, int],
    poison: int,
) -> None:
    """Poison only the exact padding scratch installed by both lanes."""

    if poison not in (0, 1):
        raise ValueError("wave audit padding poison must be zero or one")
    pages, states = _bound_wave_padding_scratch_indices(executor, split)
    if not pages.numel():
        return
    value = (37.0, -73.0)[poison]
    arena = executor.body.arena
    arena.key.index_fill_(0, pages, _storage_poison_value(arena.key, value))
    arena.value.index_fill_(0, pages, _storage_poison_value(arena.value, -value))
    arena.written.index_fill_(0, pages, True)
    arena.states[CONV_STATE_ID].index_fill_(0, states, value / 4)
    arena.states[RECURRENT_STATE_ID].index_fill_(0, states, value / 16)


def _poison_wave_audit_padding_inputs(
    executor: _CapturedWavefrontRank,
    split: tuple[int, int],
    poison: int,
) -> None:
    """Vary only ownerless stage-zero inputs so padding execution is observable."""

    if poison not in (0, 1):
        raise ValueError("wave audit padding poison must be zero or one")
    for binder, logical_m in zip((executor._lane_a, executor._lane_b), split, strict=True):
        body = binder.body
        if body.stage != 0 or logical_m == body.captured_m:
            continue
        token_ids = torch.zeros(body.captured_m, dtype=torch.int64, device=body.hidden.device)
        token_ids[logical_m:] = torch.arange(
            1001 + 997 * poison,
            1001 + 997 * poison + body.captured_m - logical_m,
            dtype=torch.int64,
            device=body.hidden.device,
        )
        body.fill_token_ids(token_ids)


def _wave_active_snapshot(
    executor: _CapturedWavefrontRank,
    split: tuple[int, int],
) -> dict[str, Tensor]:
    """Copy one bounded active boundary witness from each physical lane."""

    result: dict[str, Tensor] = {}
    for label, binder, logical_m in zip(
        ("a", "b"), (executor._lane_a, executor._lane_b), split, strict=True
    ):
        for name, value in _active_snapshot(binder.body, logical_m).items():
            result[f"{label}:{name}"] = value
    combined = executor.slot.logits
    if combined is not None:
        first, second = split
        result["combined:a"] = combined[first - 1 : first].detach().clone()
        result["combined:b"] = (
            combined[executor.micro_m + second - 1 : executor.micro_m + second].detach().clone()
        )
    return result


def _wave_padding_snapshot(
    executor: _CapturedWavefrontRank,
    split: tuple[int, int],
) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    for label, binder, logical_m in zip(
        ("a", "b"), (executor._lane_a, executor._lane_b), split, strict=True
    ):
        for name, value in _padding_snapshot(binder.body, logical_m).items():
            result[f"{label}:{name}"] = value
    return result


def _wave_sampling_layout_is_exact(
    executor: _CapturedWavefrontRank,
    logical_m: int,
    split: tuple[int, int],
) -> bool:
    """Witness both lane segments in the exact row order consumed by sampling."""

    logical = executor._sampling_logits(logical_m)
    if logical is None:
        return executor.slot.logits is None
    first, second = split
    lane_a_logits = executor._lane_a.body.logits
    lane_b_logits = executor._lane_b.body.logits
    if lane_a_logits is None or lane_b_logits is None:
        return False
    return torch.equal(logical[:first], lane_a_logits[:first]) and torch.equal(
        logical[first:logical_m], lane_b_logits[:second]
    )


def _run_wave_audit_once(
    executor: _CapturedWavefrontRank,
    logical_m: int,
    *,
    graph: bool,
    poison: int,
) -> tuple[dict[str, Tensor], dict[str, Tensor], bool, bool]:
    """Execute independent sequential eager or the production combined graph."""

    _reset_wave_audit_executor(executor)
    _reset_wave_audit_owners(executor, logical_m)
    rows = _wave_audit_rows(executor, logical_m)
    lane_a_rows, lane_b_rows = executor._split_rows(rows)
    split = (len(lane_a_rows), len(lane_b_rows))
    lane_rows = (lane_a_rows, lane_b_rows)
    executor._lane_a_rows = split[0]
    _dirty_production_audit_destinations(executor.body, rows)
    if graph:
        for binder, active_rows in zip(
            (executor._lane_a, executor._lane_b), lane_rows, strict=True
        ):
            binder.bind_batch(active_rows)
    else:
        # This oracle deliberately shares no production row-staging logic.
        for binder, active_rows in zip(
            (executor._lane_a, executor._lane_b), lane_rows, strict=True
        ):
            binder.body.bind_scratch(binder.resource)
            _direct_bind_production_audit_rows(binder.body, active_rows)
            _bind_direct_padding_to_unused_scratch(binder, len(active_rows))
    # Inspect metadata only after its independent or production binder has
    # installed the current rows.  Checking before bind merely examines stale
    # all-scratch startup state and cannot prove this transaction's ownership.
    padding_safe = all(
        _padding_is_bound_to_scratch(binder, lane_m)
        for binder, lane_m in zip((executor._lane_a, executor._lane_b), split, strict=True)
    )
    _poison_wave_audit_scratch(executor, split, poison)
    _poison_wave_audit_padding_inputs(executor, split, poison)
    torch.cuda.current_stream(executor.body.hidden.device).synchronize()
    if graph:
        for binder in (executor._lane_a, executor._lane_b):
            binder.complete_bound(replay=False)
        current = torch.cuda.current_stream(executor.body.hidden.device)
        executor.slot.stream.wait_stream(current)
        executor.slot.replay()
        current.wait_stream(executor.slot.stream)
        executor.slot.synchronize()
    else:
        # A device completion and world rendezvous between independent eager
        # lanes prevents graph overlap from becoming its own oracle.
        barrier()
        executor._lane_a.body.launch()
        torch.cuda.synchronize(executor.body.hidden.device)
        barrier()
        executor._lane_b.body.launch()
        torch.cuda.synchronize(executor.body.hidden.device)
        barrier()
        combined = executor.slot.logits
        if combined is not None:
            first_logits = executor._lane_a.body.logits
            second_logits = executor._lane_b.body.logits
            assert first_logits is not None and second_logits is not None
            combined[: executor.micro_m].copy_(first_logits)
            combined[executor.micro_m :].copy_(second_logits)
    active_snapshot = _wave_active_snapshot(executor, split)
    padding = _wave_padding_snapshot(executor, split)
    sampling_layout_safe = _wave_sampling_layout_is_exact(executor, logical_m, split)
    torch.cuda.current_stream(executor.body.hidden.device).synchronize()
    # Keep the three retained oracle snapshots out of the nearly-full graph
    # arena and perform exact/finite checks on CPU, where no captured NCCL
    # stream can become an implicit dependency of the verdict.
    return (
        {name: value.cpu() for name, value in active_snapshot.items()},
        {name: value.cpu() for name, value in padding.items()},
        padding_safe,
        sampling_layout_safe,
    )


def _wave_audit_consensus(
    statuses: Any,
    barrier: Any,
    *,
    rank: int,
    phase_id: int,
    failure_code: int,
    timeout_seconds: float,
) -> None:
    """Make every rank accept or reject one verdict through startup-only IPC."""

    world_size = len(statuses) // 2
    if not 0 <= rank < world_size:
        raise ValueError("wave audit rank lies outside the world")
    with statuses.get_lock():
        statuses[2 * rank] = phase_id
        statuses[2 * rank + 1] = failure_code
    try:
        barrier.wait(timeout_seconds)
        with statuses.get_lock():
            observed = tuple(int(statuses[index]) for index in range(2 * world_size))
        # Prevent a fast rank from overwriting its slot for the next phase while
        # another rank is still reading this one.
        barrier.wait(timeout_seconds)
    except threading.BrokenBarrierError as error:
        raise Qwen3_5TP2PP2ServingError(
            f"wave audit phase {phase_id} IPC consensus timed out"
        ) from error
    phases = tuple(observed[2 * peer_rank] for peer_rank in range(world_size))
    failures = tuple(
        (peer_rank, observed[2 * peer_rank + 1])
        for peer_rank in range(world_size)
        if observed[2 * peer_rank + 1] != 0
    )
    if phases != (phase_id,) * world_size:
        raise Qwen3_5TP2PP2ServingError(
            f"wave audit rank phase diverged: expected {phase_id}, observed {phases}"
        )
    if failures:
        raise Qwen3_5TP2PP2ServingError(
            f"wave audit phase {phase_id} failed on rank/code {failures}"
        )


def _cold_start_wavefront_audit(
    executor: _CapturedWavefrontRank,
    admission: _Admission,
    audit_statuses: Any,
    audit_barrier: Any,
    audit_timeout_seconds: float,
) -> str:
    """Qualify the real 2 x C128 production binder against eager lane math."""

    logical_ms = _bounded_audit_logical_ms(
        tuple(range(executor.micro_m + 1, executor.captured_m + 1))
    )
    domains: dict[int, dict[str, bool]] = {}
    for phase_id, logical_m in enumerate(logical_ms, start=1):
        eager, _, _, eager_sampling_safe = _run_wave_audit_once(
            executor, logical_m, graph=False, poison=0
        )
        graph_zero, padding_zero, safe_zero, sampling_zero = _run_wave_audit_once(
            executor, logical_m, graph=True, poison=0
        )
        graph_one, padding_one, safe_one, sampling_one = _run_wave_audit_once(
            executor, logical_m, graph=True, poison=1
        )
        failure_code = 0
        failure_detail: list[str] = []
        padding_exercised = logical_m < executor.captured_m
        try:
            if not _exact_snapshot(eager, graph_zero):
                failure_code |= 1
                failure_detail.append("eager_exact")
            if not _exact_snapshot(graph_zero, graph_one) or not (safe_zero and safe_one):
                failure_code |= 2
                changed = tuple(
                    name
                    for name in graph_zero.keys() & graph_one.keys()
                    if not torch.equal(graph_zero[name], graph_one[name])
                )
                failure_detail.append(
                    "scratch_isolation:"
                    f"changed={'|'.join(changed) or 'none'}:"
                    f"binding={int(safe_zero)}/{int(safe_one)}"
                )
            unchanged_padding = tuple(
                name
                for name in padding_zero
                if name in padding_one and torch.equal(padding_zero[name], padding_one[name])
            )
            if padding_exercised and (
                padding_zero.keys() != padding_one.keys() or unchanged_padding
            ):
                failure_code |= 4
                failure_detail.append("padding_poison:" + "|".join(unchanged_padding))
            if not (eager_sampling_safe and sampling_zero and sampling_one):
                failure_code |= 16
                failure_detail.append("sampling_layout")
        except (RuntimeError, ValueError) as error:
            failure_code |= 8
            failure_detail.append(type(error).__name__)
        if failure_code:
            diagnostic_event(_LOGGER, 'performance.wave_audit_local_verdict_failed_rank_d',
                executor.body.rank,
                logical_m,
                failure_code,
                ",".join(failure_detail),
            )
        _wave_audit_consensus(
            audit_statuses,
            audit_barrier,
            rank=executor.body.rank,
            phase_id=phase_id,
            failure_code=failure_code,
            timeout_seconds=audit_timeout_seconds,
        )
        domains[logical_m] = {
            "sequential_lane_eager_exact": True,
            "production_dual_bind_exact": True,
            "shared_scratch_isolated": True,
            "padding_poison_exercised": padding_exercised,
            "logical_sampling_layout_exact": True,
        }
    _reset_wave_audit_executor(executor)
    _reset_wave_audit_owners(
        executor,
        max(logical_ms),
        scratch_logical_m=min(logical_ms),
    )
    torch.cuda.current_stream(executor.body.hidden.device).synchronize()
    return stable_digest(
        {
            "schema": "cubic.qwen3_5.tp2_pp2.wavefront-cold-start.v1",
            "rank": executor.body.rank,
            "inventory_digest": admission.inventory.inventory_digest,
            "semantic_plan_digest": admission.parallel_plan.semantic_plan_digest,
            "topology_digest": admission.sealed.topology_digest,
            "partition_digest": admission.sealed.partition_digest,
            "micro_m": executor.micro_m,
            "logical_capacity": executor.captured_m,
            "logical_m_values": logical_ms,
            "balanced_split": True,
            "rank_local_serial_shared_scratch": True,
            "domains": domains,
        }
    )


def _run_production_bulk_sequence(
    executor: _CapturedGraphRank,
    logical_ms: tuple[int, ...],
    *,
    graph: bool,
    snapshot_ordinal: int | None,
) -> tuple[dict[int, dict[str, Tensor]], bool]:
    """Run one eager or production-staged COW sequence over a fixed bucket."""

    body = executor.body
    _reset_owner(body, body.buffers.page_table.shape[1])
    _reset_production_audit_executor(executor)
    snapshots: dict[int, dict[str, Tensor]] = {}
    padding_scratch = True
    previous_m = 0
    for ordinal, logical_m in enumerate(logical_ms):
        initialize_from = previous_m if logical_m > previous_m else None
        rows = _production_audit_rows(
            body,
            logical_m,
            position=ordinal,
            parity=ordinal & 1,
            initialize_from=initialize_from,
        )
        _dirty_production_audit_destinations(body, rows)
        if graph:
            _poison_production_audit_scratch(executor, ordinal & 1)
            executor.replay_batch(rows)
            executor.slot.synchronize()
            padding_scratch = padding_scratch and _padding_is_bound_to_scratch(executor, logical_m)
        else:
            body.bind_scratch(executor.resource)
            _direct_bind_production_audit_rows(body, rows)
            barrier()
            body.launch()
            torch.cuda.synchronize(body.hidden.device)
        barrier()
        if snapshot_ordinal is None or ordinal == snapshot_ordinal:
            snapshots[ordinal] = _tracked_snapshot(body)
        previous_m = logical_m
    ring_wrapped = (
        not graph
        or len(logical_ms) < 3
        or (
            executor._host_event_recorded == [True, True]
            and executor._next_host_slot == len(logical_ms) % 2
        )
    )
    return snapshots, padding_scratch and ring_wrapped


def _run_bucket_migration_chain(
    executors: dict[int, _CapturedGraphRank],
    *,
    graph: bool,
    poison: int,
    snapshot_capacity: int | None,
) -> tuple[dict[int, dict[str, Tensor]], bool]:
    """Move one request owner through every physical bucket without sampling."""

    capacities = tuple(sorted(executors))
    if not capacities or capacities[0] != _CAPTURED_M:
        raise Qwen3_5TP2PP2ServingError("migration audit requires the complete graph pool")
    largest = executors[capacities[-1]].body
    _reset_owner(largest, largest.buffers.page_table.shape[1])
    for executor in executors.values():
        _reset_production_audit_executor(executor)
    torch.cuda.current_stream(largest.hidden.device).synchronize()
    snapshots: dict[int, dict[str, Tensor]] = {}
    padding_scratch = True
    previous_m = 0
    for ordinal, (captured_m, logical_m) in enumerate(
        zip(capacities, _migration_logical_ms(capacities), strict=True)
    ):
        executor = executors[captured_m]
        body = executor.body
        initialize_from = previous_m if logical_m > previous_m else None
        rows = _production_audit_rows(
            body,
            logical_m,
            position=ordinal,
            parity=ordinal & 1,
            initialize_from=initialize_from,
        )
        _dirty_production_audit_destinations(body, rows)
        _poison_production_audit_scratch(executor, poison)
        torch.cuda.current_stream(body.hidden.device).synchronize()
        if graph:
            executor.replay_batch(rows)
            executor.slot.synchronize()
            padding_scratch = padding_scratch and _padding_is_bound_to_scratch(executor, logical_m)
        else:
            body.bind_scratch(executor.resource)
            _direct_bind_production_audit_rows(body, rows)
            barrier()
            body.launch()
            torch.cuda.synchronize(body.hidden.device)
        barrier()
        if snapshot_capacity is None or captured_m == snapshot_capacity:
            snapshots[captured_m] = _tracked_snapshot(body)
        if captured_m == snapshot_capacity:
            break
        previous_m = logical_m
    return snapshots, padding_scratch


@contextmanager
def _startup_check_phase(rank: int, phase: str) -> Iterator[None]:
    """Report real work boundaries without introducing CUDA synchronization."""
    started = time.perf_counter()
    if rank == 0:
        diagnostic_event(_LOGGER, 'performance.startup_check_begin_phase_s',
            phase,
        )
    try:
        yield
    except Exception:
        diagnostic_event(_LOGGER, 'performance.startup_check_failed_rank_d_phase_s',
            rank,
            phase,
            time.perf_counter() - started,
        )
        raise
    else:
        if rank == 0:
            diagnostic_event(_LOGGER, 'performance.startup_check_complete_phase_s_duration_s',
                phase,
                time.perf_counter() - started,
            )


def _run_bucket_migration_audit(
    executors: dict[int, _CapturedGraphRank],
    admission: _Admission,
) -> str:
    """Qualify request-owned COW state across every serving bucket transition."""

    owner_executors = {m: x for m, x in executors.items()
                       if m <= x.body.arena.spec.state_slot_capacity}
    if len(owner_executors) != len(executors):
        ordinary = _run_bucket_migration_audit(owner_executors, admission)
        review = tuple((m, _audit_review_capacity(x)) for m, x in executors.items()
                       if m not in owner_executors)
        return stable_digest({"owner_migration": ordinary, "review_capacity": review})

    capacities = tuple(sorted(executors))
    rank = executors[capacities[0]].body.rank
    production_domains: dict[int, dict[str, bool]] = {}
    for length in sorted({min(length, capacities[-1]) for length in (2, 3, 17)}):
        with _startup_check_phase(rank, f"continuation L{length}"):
            executor = executors[_bucket_for_logical_m(length, capacities)]
            eager = _run_continuation_audit(executor, length, graph=False)
            graph = _run_continuation_audit(executor, length, graph=True)
            if not _exact_snapshot(eager, graph):
                raise Qwen3_5TP2PP2ServingError(
                    f"L{length} continuation graph differs from direct eager"
                )
            del eager, graph
            torch.cuda.empty_cache()
    for captured_m, executor in executors.items():
        with _startup_check_phase(rank, f"steady continuation M{captured_m}"):
            representatives = _bounded_audit_logical_ms(
                tuple(executor.body.bucket_logical_ms)
            )
            for logical_m in (representatives[0], representatives[-1]):
                for crossed_page in (False, True):
                    full = _run_steady_continuation_audit(
                        executor,
                        logical_m,
                        crossed_page=crossed_page,
                        continuation=False,
                    )
                    sparse = _run_steady_continuation_audit(
                        executor,
                        logical_m,
                        crossed_page=crossed_page,
                        continuation=True,
                    )
                    if not _exact_snapshot(full, sparse):
                        boundary = "page" if crossed_page else "token"
                        raise Qwen3_5TP2PP2ServingError(
                            f"C{captured_m}/M{logical_m} steady {boundary} continuation "
                            "differs from full bind"
                        )
                    del full, sparse
            torch.cuda.empty_cache()
    for captured_m, executor in executors.items():
        with _startup_check_phase(rank, f"batch grow/shrink M{captured_m}"):
            representatives = _bounded_audit_logical_ms(
                tuple(executor.body.bucket_logical_ms)
            )
            logical_ms = (
                representatives[0],
                representatives[-1],
                representatives[0],
                representatives[-1],
            )
            eager, _ = _run_production_bulk_sequence(
                executor, logical_ms, graph=False, snapshot_ordinal=None
            )
            graph, production_safe = _run_production_bulk_sequence(
                executor, logical_ms, graph=True, snapshot_ordinal=None
            )
            production_exact = eager.keys() == graph.keys() and all(
                _exact_snapshot(eager[target], graph[target]) for target in eager
            )
            del eager, graph
            torch.cuda.empty_cache()
            if not production_exact or not production_safe:
                raise Qwen3_5TP2PP2ServingError(
                    f"C{captured_m} production bulk replay differs from direct eager"
                )
            production_domains[captured_m] = {
                "production_bulk_replay_exact": True,
                "reset_and_new_page_index_exact": True,
                "host_ring_wrap_exact": True,
                "active_to_padding_shrink_exact": True,
                "grow_shrink_grow_exact": True,
            }

    domains: dict[int, dict[str, bool]] = {}
    capacities = tuple(sorted(executors))
    with _startup_check_phase(rank, "bucket migration and padding poison"):
        eager, _ = _run_bucket_migration_chain(
            executors, graph=False, poison=0, snapshot_capacity=None
        )
        graph_zero, graph_zero_padding = _run_bucket_migration_chain(
            executors, graph=True, poison=0, snapshot_capacity=None
        )
        graph_one, graph_one_padding = _run_bucket_migration_chain(
            executors, graph=True, poison=1, snapshot_capacity=None
        )
        for captured_m in capacities:
            if not _exact_snapshot(eager[captured_m], graph_zero[captured_m]):
                raise Qwen3_5TP2PP2ServingError(
                    f"C{captured_m} migration graph differs from direct eager"
                )
            if not _exact_snapshot(graph_zero[captured_m], graph_one[captured_m]):
                raise Qwen3_5TP2PP2ServingError(
                    f"C{captured_m} migration state depends on padding poison"
                )
            if not graph_zero_padding or not graph_one_padding:
                raise Qwen3_5TP2PP2ServingError(
                    f"C{captured_m} migration padding escaped private scratch"
                )
            domains[captured_m] = {
                "migration_eager_exact": True,
                "padding_poison_independent": True,
                "production_replay_batch": True,
            }
        del eager, graph_zero, graph_one
        torch.cuda.empty_cache()
        largest = executors[capacities[-1]]
        _reset_owner(largest.body, largest.body.buffers.page_table.shape[1])
        for executor in executors.values():
            _reset_production_audit_executor(executor)
        torch.cuda.current_stream(largest.body.hidden.device).synchronize()
        torch.cuda.empty_cache()
    return stable_digest(
        {
            "schema": "cubic.qwen3_5.tp2_pp2.bucket-migration-cow.v1",
            "rank": largest.body.rank,
            "inventory_digest": admission.inventory.inventory_digest,
            "semantic_plan_digest": admission.parallel_plan.semantic_plan_digest,
            "topology_digest": admission.sealed.topology_digest,
            "partition_digest": admission.sealed.partition_digest,
            "max_sequence_tokens": admission.max_sequence_tokens,
            "captured_ms": capacities,
            "logical_ms": _migration_logical_ms(capacities),
            "domains": domains,
            "production_bulk_domains": production_domains,
            "production_bulk_replay_exact": True,
        }
    )


def _qualification_cache_root() -> Path:
    from cubic_serve import cache_root

    return cache_root() / "qualification"


def _qualification_source_digest() -> str:
    from cubic_serve.contracts.implementation import implementation_sources

    return stable_digest(implementation_sources(__name__))


def _migration_execution_plan(executors: Mapping[int, _CapturedGraphRank]) -> tuple[object, ...]:
    """Fingerprint selected launches, excluding instance addresses and timings."""

    def body_plan(body: Any) -> tuple[object, ...]:
        bindings = dict(vars(body))
        inner = getattr(body, "_body", None)
        if inner is not None:
            bindings.update((f"_body.{name}", value) for name, value in vars(inner).items())
        launches = tuple(
            (name, signature)
            for name, binding in sorted(bindings.items())
            if isinstance(signature := getattr(
                getattr(binding, "__self__", binding), "launch_signature", None,
            ), str)
        )
        return (body.captured_m, launches)

    result = []
    for capacity, executor in sorted(executors.items()):
        slot = executor.slot
        overlap = getattr(slot, "_overlap", None)
        packed_overlap = getattr(slot, "_overlap_resources", None)
        children = overlap[2] if overlap is not None else (
            packed_overlap[1] if packed_overlap is not None else ()
        )
        waves = tuple(
            body_plan(lane)
            for name in ("lane_a", "lane_b")
            if (lane := getattr(slot, name, None)) is not None
        )
        result.append((
            capacity, body_plan(executor.body), waves,
            tuple(body_plan(child) for child in children),
        ))
    return tuple(result)


def _migration_topology_plan_digest(executors: Mapping[int, _CapturedGraphRank]) -> str:
    # Cold-start only. Independently cached rank plans from different runs must
    # not compose into a topology that has never passed a distributed audit.
    plans: list[object] = [None] * dist.get_world_size()
    dist.all_gather_object(plans, _migration_execution_plan(executors))
    return stable_digest(plans)


def _migration_audit_cache_key(
    executors: dict[int, _CapturedGraphRank], admission: _Admission
) -> str:
    """Collectively bind receipts to the actual startup-selected topology plan."""

    largest = executors[tuple(executors)[-1]].body
    topology_plan = _migration_topology_plan_digest(executors)
    implementation = tuple(
        inspect.getsource(item)
        for item in (
            _CapturedGraphRank,
            independent_decode_partitions,
            owned_single_token_rows,
            batch_segments,
            _production_audit_rows,
            _review_audit_rows,
            _audit_review_capacity,
            _direct_bind_production_audit_rows,
            _steady_audit_rows,
            _steady_audit_snapshot,
            _run_steady_continuation_audit,
            _run_production_bulk_sequence,
            _run_bucket_migration_chain,
            _run_continuation_audit,
            _run_bucket_migration_audit,
            _migration_execution_plan,
            _migration_topology_plan_digest,
        )
    )
    return stable_digest(
        {
            "schema": "cubic.graph-migration-receipt-key.v2",
            "selected_topology_plan": topology_plan,
            "operator_config": admission.operator_config.digest,
            "operator_model_id": admission.operator_model_id,
            "inventory": admission.inventory.inventory_digest,
            "semantic_plan": admission.parallel_plan.semantic_plan_digest,
            "topology": admission.sealed.topology_digest,
            "partition": admission.sealed.partition_digest,
            "max_sequence_tokens": admission.max_sequence_tokens,
            "captured_ms": tuple(executors),
            "arena": {
                "pages": largest.arena.spec.page_capacity,
                "states": largest.arena.spec.state_slot_capacity,
                "scratch": largest.arena.spec.scratch_capacity,
                "state_dtypes": {
                    name: str(state.dtype).removeprefix("torch.")
                    for name, state in sorted(largest.arena.states.items())
                },
            },
            "runtime": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(largest.hidden.device),
                "capability": torch.cuda.get_device_capability(largest.hidden.device),
            },
            "implementation": implementation,
            "source_digest": _qualification_source_digest(),
        }
    )


def _read_execution_audit_receipt(key: str, rank: int, *, check: str) -> str | None:
    path = _qualification_cache_root() / f"{check}-rank{rank}-{key.removeprefix('b2s256:')}.json"
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return None
    if (
        type(raw) is not dict
        or raw.get("schema") != f"cubic.graph-{check}-receipt.v1"
        or raw.get("key") != key
        or type(raw.get("evidence_id")) is not str
        or (check != "migration" and raw.get("digest") != stable_digest(raw["evidence_id"]))
    ):
        return None
    return cast(str, raw["evidence_id"])


def _write_execution_audit_receipt(key: str, rank: int, evidence_id: str, *, check: str) -> None:
    root = _qualification_cache_root()
    path = root / f"{check}-rank{rank}-{key.removeprefix('b2s256:')}.json"
    temporary: Path | None = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=root, prefix=f"{check}-", suffix=".tmp", delete=False
        ) as output:
            temporary = Path(output.name)
            json.dump(
                {
                    "schema": f"cubic.graph-{check}-receipt.v1",
                    "key": key,
                    "evidence_id": evidence_id,
                    "digest": stable_digest(evidence_id),
                },
                output,
                separators=(",", ":"),
                sort_keys=True,
            )
        os.replace(temporary, path)
    except OSError:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink()
        diagnostic_event(_LOGGER, 'performance.qualification_cache_write_failed_rank_d',
            rank,
            exc_info=True,
        )


def _read_migration_audit_receipt(key: str, rank: int) -> str | None:
    return _read_execution_audit_receipt(key, rank, check="migration")


def _write_migration_audit_receipt(key: str, rank: int, evidence_id: str) -> None:
    _write_execution_audit_receipt(key, rank, evidence_id, check="migration")


def _cold_start_bucket_migration_audit(
    executors: dict[int, _CapturedGraphRank],
    admission: _Admission,
) -> str:
    """Run once per exact implementation/checkpoint/topology, then reuse its receipt.

    Every process instance has already performed live eager-versus-graph and
    padding-poison checks for each captured bucket before reaching this point.
    The receipt therefore caches only the expensive cross-bucket/COW sequence,
    never graph capture or the instance-bound numerical smoke check.
    """

    rank = executors[tuple(executors)[-1]].body.rank
    key = _migration_audit_cache_key(executors, admission)
    cached = _read_migration_audit_receipt(key, rank)
    device = executors[tuple(executors)[-1]].body.hidden.device
    cache_consensus = torch.tensor(
        (int(cached is not None),), dtype=torch.int32, device=device
    )
    dist.all_reduce(cache_consensus, op=dist.ReduceOp.MIN)
    all_ranks_cached = bool(cache_consensus.item())
    del cache_consensus
    if all_ranks_cached:
        assert cached is not None
        diagnostic_event(_LOGGER, 'performance.qualification_cache_hit_rank_d_check_migration',
            rank,
        )
        return cached
    if cached is not None:
        diagnostic_event(_LOGGER, 'performance.qualification_cache_partial_miss_rank_d_check',
            rank,
        )
    evidence_id = _run_bucket_migration_audit(executors, admission)
    _write_migration_audit_receipt(key, rank, evidence_id)
    return evidence_id


@dataclass(frozen=True, slots=True)
class _Ready:
    rank: int
    vocab_size: int
    max_sequence_tokens: int
    max_physical_graph_capacity: int
    graph_evidence_id: str
    cross_row_state_evidence_id: str
    cache_capacity: Qwen3_5TP2PP2RankCacheCapacity
    target_cost_samples: tuple[tuple[int, float], ...]
    proposal_cost_samples: tuple[tuple[int, float], ...]
    target_depth_samples: tuple[tuple[int, int, float], ...] = ()
    # Actual logical target rows, segment depth, greedy flag, milliseconds.
    compound_cost_samples: tuple[tuple[int, int, bool, float], ...] = ()
    replay_resources: RankReplayResources | None = None
    resident_publications: tuple[str, ...] = ()
    resident_input_program_capacities: tuple[int, ...] = ()
    resident_compaction_capacities: tuple[int, ...] = ()
    resident_segment_program_capacities: tuple[int, ...] = ()
    resident_draft_capacities: tuple[int, ...] = ()
    resident_stochastic_input_program_capacities: tuple[int, ...] = ()
    constraint_graphs_qualified: bool = False
    grammar_draft_continuation_capacities: tuple[int, ...] = ()
    grammar_continuation_capacities: tuple[int, ...] = ()
    constraint_compound_cost_samples: tuple[tuple[int, int, bool, int, float], ...] = ()
    # Qualified masked sampling tail only: physical rows, greedy, wall ms.
    constraint_sampling_cost_samples: tuple[tuple[int, bool, float], ...] = ()
    input_encoder_admission: tuple[int, int, int] | None = None


@dataclass(frozen=True, slots=True)
class _LoadProgress:
    rank: int
    section: str
    completed: int
    total: int
    unit: str = "bytes"
    capacity: int | None = None


@dataclass(frozen=True, slots=True)
class _Failure:
    rank: int
    request_id: str | None
    detail: str


@contextmanager
def _report_before_teardown(report: Callable[[], None]) -> Iterator[None]:
    """Wake the controller before distributed cleanup can wait on peer ranks."""
    try:
        yield
    except BaseException:
        report()
        raise


def _warm_sampling(
    executors: dict[int, _CapturedGraphRank],
    *, price_sampling_through: int = 0,
    full_bucket_only: bool = False,
) -> tuple[tuple[int, bool, float], ...]:
    """Materialize every resident physical bucket's public sampling path."""

    sampling = SamplingParameters(1.0, 0.95, 20, 1)
    prices = []
    for captured_m, executor in executors.items():
        # Model/capture checks may intentionally leave inactive logits as NaN.
        # Sampling warmup owns synthetic inputs, never those partial forwards;
        # the first real model replay replaces every request-owned output row.
        if executor.slot.logits is not None:
            executor.slot.logits.zero_()
        for logical_m in ((captured_m,) if full_bucket_only else
                          _bounded_audit_logical_ms(tuple(executor.body.bucket_logical_ms))):
            rows = tuple(
                _BatchRow(
                    f"startup:sampling:{captured_m}:{row}",
                    0,
                    0,
                    0,
                    1,
                    0,
                    1,
                    (1,),
                    sampling=sampling,
                )
                for row in range(logical_m)
            )
            executor.sample_tokens(rows)
        if captured_m <= price_sampling_through and len(rows) == captured_m:
            # Reuse the sealed, qualified sampler. This is a startup prior,
            # not a new language forward or a serving-throughput measurement.
            state = executor._sampling
            was_bound = state.constraint_bound
            try:
                for greedy in (False, True):
                    priced_rows = tuple(replace(row, sampling=GREEDY_SAMPLING if greedy else sampling)
                                        for row in rows)
                    samples = []
                    for ordinal in range(4):
                        torch.cuda.synchronize(executor.body.hidden.device)
                        started = time.monotonic_ns()
                        state.bind_constraints([None] * captured_m, [True] * captured_m)
                        executor.sample_tokens(priced_rows)
                        torch.cuda.synchronize(executor.body.hidden.device)
                        if ordinal:
                            samples.append((time.monotonic_ns() - started) / 1_000_000)
                    prices.append((captured_m, greedy, _median_cost_samples(samples)))
            finally:
                state.constraint_bound = was_bound
                executor.sample_tokens(rows)
    device = next(iter(executors.values())).body.hidden.device
    torch.cuda.current_stream(device).synchronize()
    return tuple(prices)


def _cold_start_cross_row_state_audit(
    target: _CapturedGraphRank,
    mtp: _CapturedMTPRank,
    target_hidden_slots: Tensor,
    admission: _Admission,
) -> str:
    """Require equal inputs at different row/state/page addresses to stay exact."""

    body, device = target.body, target.body.hidden.device
    target_states = torch.tensor((0, 1, 2, 3), dtype=torch.int64, device=device)
    target_pages = torch.tensor((0, 1, 2, 3), dtype=torch.int64, device=device)
    for state in body.arena.states.values():
        state.index_fill_(0, target_states, 0)
    body.arena.key.index_fill_(0, target_pages, 0)
    body.arena.value.index_fill_(0, target_pages, 0)
    body.arena.written.index_fill_(0, target_pages, False)
    target_hidden_slots.index_fill_(0, target_states, 0)
    barrier()
    rows = tuple(
        _BatchRow(
            request_id=f"audit:cross-row:{physical}",
            token_id=101,
            position=0,
            state_source_slot=source,
            state_destination_slot=destination,
            page_source=source,
            page_destination=destination,
            page_table=(destination,),
            reset_state=True,
            new_pages=(destination,),
            segment_length=1,
        )
        for physical, (source, destination) in enumerate(((0, 1), (2, 3)))
    )
    target.replay_batch(rows)
    torch.cuda.current_stream(device).synchronize()

    target_pairs = {
        "target_output": (body.hidden[0], body.hidden[1]),
        "target_hidden": (target_hidden_slots[1], target_hidden_slots[3]),
        "target_kv_key": (body.arena.key[1], body.arena.key[3]),
        "target_kv_value": (body.arena.value[1], body.arena.value[3]),
        "target_kv_written": (body.arena.written[1], body.arena.written[3]),
        "target_conv": (
            body.arena.states[CONV_STATE_ID][1],
            body.arena.states[CONV_STATE_ID][3],
        ),
        "target_recurrent": (
            body.arena.states[RECURRENT_STATE_ID][1],
            body.arena.states[RECURRENT_STATE_ID][3],
        ),
    }
    failures = tuple(
        name for name, (left, right) in target_pairs.items() if not torch.equal(left, right)
    )
    if failures:
        raise Qwen3_5TP2PP2ServingError(
            f"target cross-row state differs at {', '.join(failures)}"
        )

    scratch_state = target._scratch_states[0]
    rollback_states = torch.tensor(
        (0, 1, 2, 3, scratch_state), dtype=torch.int64, device=device
    )
    rollback_pages = torch.tensor((6, 7, 8, 9), dtype=torch.int64, device=device)
    for state in body.arena.states.values():
        state.index_fill_(0, rollback_states, 0)
    body.arena.key.index_fill_(0, rollback_pages, 0)
    body.arena.value.index_fill_(0, rollback_pages, 0)
    body.arena.written.index_fill_(0, rollback_pages, False)
    target_hidden_slots.index_fill_(0, rollback_states, 0)
    speculative_rows = (
        _BatchRow(
            "audit:rollback",
            101,
            0,
            0,
            scratch_state,
            6,
            7,
            (7,),
            reset_state=True,
            new_pages=(7,),
            segment_length=2,
            is_decode=True,
        ),
        _BatchRow(
            "audit:rollback",
            102,
            1,
            0,
            0,
            7,
            7,
            (7,),
            segment_length=0,
            is_decode=True,
        ),
    )
    target.replay_batch(speculative_rows)
    _repair_speculative_state(
        target,
        target_hidden_slots,
        speculative_rows,
        (_RaggedSpeculativeSegment(0, 0, 2),),
        (0,),
    )
    oracle_row = (
        _BatchRow(
            "audit:rollback:oracle",
            101,
            0,
            2,
            2,
            8,
            9,
            (9,),
            reset_state=True,
            new_pages=(9,),
            segment_length=1,
            is_decode=True,
        ),
    )
    target.replay_batch(oracle_row)
    torch.cuda.current_stream(device).synchronize()
    rollback_pairs = {
        "rollback_hidden": (target_hidden_slots[0], target_hidden_slots[2]),
        "rollback_conv": (
            body.arena.states[CONV_STATE_ID][0],
            body.arena.states[CONV_STATE_ID][2],
        ),
        "rollback_recurrent": (
            body.arena.states[RECURRENT_STATE_ID][0],
            body.arena.states[RECURRENT_STATE_ID][2],
        ),
    }
    failures = tuple(
        name for name, (left, right) in rollback_pairs.items() if not torch.equal(left, right)
    )
    if failures:
        raise Qwen3_5TP2PP2ServingError(
            f"MTP scratch rollback differs at {', '.join(failures)}"
        )

    mtp_pages = torch.tensor((4, 5), dtype=torch.int64, device=device)
    mtp.body.arena.key.index_fill_(0, mtp_pages, 0)
    mtp.body.arena.value.index_fill_(0, mtp_pages, 0)
    mtp.body.arena.written.index_fill_(0, mtp_pages, False)
    mtp.replay(
        tuple(
            _MTPReplayRow(
                request_id=f"audit:cross-row:{physical}",
                token_id=101,
                position=0,
                page_destination=page,
                page_table=(page,),
                state_source_slot=source,
                sampling=GREEDY_SAMPLING,
                sample_ordinal=0,
            )
            for physical, (page, source) in enumerate(((4, 1), (5, 3)))
        ),
        torch.stack((target_hidden_slots[1], target_hidden_slots[3])),
        sample=False,
    )
    torch.cuda.current_stream(device).synchronize()
    mtp_pairs = {
        "mtp_hidden": (mtp.body.buffers.hidden[0], mtp.body.buffers.hidden[1]),
        "mtp_kv_key": (mtp.body.arena.key[4], mtp.body.arena.key[5]),
        "mtp_kv_value": (mtp.body.arena.value[4], mtp.body.arena.value[5]),
        "mtp_kv_written": (mtp.body.arena.written[4], mtp.body.arena.written[5]),
    }
    failures = tuple(
        name for name, (left, right) in mtp_pairs.items() if not torch.equal(left, right)
    )
    if failures:
        raise Qwen3_5TP2PP2ServingError(
            f"MTP cross-row state differs at {', '.join(failures)}"
        )

    evidence = stable_digest(
        {
            "schema": "cubic.qwen3_5.tp4.cross-row-state.v1",
            "rank": body.rank,
            "inventory_digest": admission.inventory.inventory_digest,
            "semantic_plan_digest": admission.parallel_plan.semantic_plan_digest,
            "target_fields": tuple(target_pairs),
            "mtp_fields": tuple(mtp_pairs),
            "rollback_fields": tuple(rollback_pairs),
            "bf16_exact": True,
            "recurrent_state_storage_dtype": str(
                body.arena.states[RECURRENT_STATE_ID].dtype
            ).removeprefix("torch."),
            "recurrent_state_exact": True,
            "partial_rejection_rollback_exact": True,
            "distinct_state_and_page_addresses": True,
        }
    )
    for state in body.arena.states.values():
        state.index_fill_(0, rollback_states, 0)
    all_target_pages = torch.cat((target_pages, rollback_pages))
    body.arena.key.index_fill_(0, all_target_pages, 0)
    body.arena.value.index_fill_(0, all_target_pages, 0)
    body.arena.written.index_fill_(0, all_target_pages, False)
    target_hidden_slots.index_fill_(0, rollback_states, 0)
    mtp.body.arena.key.index_fill_(0, mtp_pages, 0)
    mtp.body.arena.value.index_fill_(0, mtp_pages, 0)
    mtp.body.arena.written.index_fill_(0, mtp_pages, False)
    torch.cuda.current_stream(device).synchronize()
    barrier()
    return evidence


def _replay_target_with_mtp_maintenance(
    executor: _CapturedGraphRank | _CapturedWavefrontRank,
    rows: tuple[_BatchRow, ...],
    speculative: tuple[_RaggedSpeculativeSegment, ...],
    mtp_executors: dict[int, _CapturedMTPRank],
    target_hidden_slots: Tensor,
) -> None:
    """Consume previous-prefix hidden before publishing into an owned slot."""

    pairs = _mtp_maintenance_pairs(rows, speculative) if mtp_executors else ()
    executor.replay_batch(rows, commit_hidden=not pairs)
    if not pairs:
        return
    mtp_rows = tuple(
        _MTPReplayRow(
            row.request_id, row.token_id, row.mtp_position, row.mtp_page_destination,
            row.mtp_page_table, row.state_source_slot, row.sampling, row.sample_ordinal,
            state_destination_slot=row.state_destination_slot,
            position_axes=row.position_axes, input_feature=row.input_feature,
            segment_length=row.segment_length,
        )
        for _, row in pairs
    )
    hidden_rows: list[Tensor] = []
    for physical, row in pairs:
        previous = physical - 1
        if (
            previous >= 0
            and rows[previous].request_id == row.request_id
            and rows[previous].position + 1 == row.position
        ):
            hidden_rows.append(
                executor.norm_row(previous)
                if isinstance(executor, _CapturedWavefrontRank)
                else getattr(executor.body, "draft_feedback", executor.body.buffers.norm)[previous]
            )
        else:
            hidden_rows.append(target_hidden_slots[row.state_source_slot])
    mtp = mtp_executors[_bucket_for_logical_m(len(mtp_rows), tuple(mtp_executors))]
    mtp.replay(mtp_rows, torch.stack(tuple(hidden_rows)), sample=False)
    executor._commit_target_hidden(rows)


def _cold_start_prefix_cache_audit(
    executors: dict[int, _CapturedGraphRank],
    mtp_executors: dict[int, _CapturedMTPRank],
    target_hidden_slots: Tensor,
    admission: _Admission,
) -> str:
    """Fork one aligned production prefix and require exact durable state."""

    prefix_length = _BLOCK_SIZE
    continuation_executor = executors[1]
    target_arena = continuation_executor.body.arena
    device = continuation_executor.body.hidden.device
    state_ids = torch.arange(4, dtype=torch.int64, device=device)
    page_ids = torch.arange(6, dtype=torch.int64, device=device)
    for state in target_arena.states.values():
        state.index_fill_(0, state_ids, 0)
    target_arena.key.index_fill_(0, page_ids, 0)
    target_arena.value.index_fill_(0, page_ids, 0)
    target_arena.written.index_fill_(0, page_ids, False)
    target_hidden_slots.index_fill_(0, state_ids, 0)

    token_ids = tuple(101 + position for position in range(prefix_length + 1))
    prefix_rows = tuple(
        _BatchRow(
            request_id="audit:prefix",
            token_id=token_ids[position],
            position=position,
            state_source_slot=0,
            state_destination_slot=1,
            page_source=0 if position == 0 else 1,
            page_destination=1,
            page_table=(1,),
            reset_state=position == 0,
            new_pages=(1,) if position == 0 else (),
            segment_length=prefix_length if position == 0 else 0,
        )
        for position in range(prefix_length)
    )
    mtp_arena = None
    if mtp_executors:
        mtp_arena = mtp_executors[1].body.arena
        mtp_pages = torch.arange(3, dtype=torch.int64, device=device)
        mtp_arena.key.index_fill_(0, mtp_pages, 0)
        mtp_arena.value.index_fill_(0, mtp_pages, 0)
        mtp_arena.written.index_fill_(0, mtp_pages, False)

    def replay_prefix(*, owned: bool) -> None:
        # A page-aligned prefix can span multiple captured prefill batches.
        # Carry both recurrent state and the previous target hidden across seams.
        capacity = min(max(executors), max(mtp_executors)) if mtp_executors else max(executors)
        destination = 0 if owned else 1
        for start in range(0, prefix_length, capacity):
            stop = min(start + capacity, prefix_length)
            rows = tuple(
                replace(
                    row,
                    state_source_slot=0 if start == 0 else destination,
                    state_destination_slot=destination,
                    segment_length=stop - start if position == start else 0,
                    mtp_position=row.position,
                    mtp_page_destination=0,
                    mtp_page_table=(0,),
                )
                for position, row in enumerate(prefix_rows[start:stop], start)
            )
            executor = executors[_bucket_for_logical_m(len(rows), tuple(executors))]
            _replay_target_with_mtp_maintenance(
                executor, rows, (), mtp_executors, target_hidden_slots,
            )
            executor.slot.synchronize()

    replay_prefix(owned=False)

    def prefix_snapshot() -> dict[str, Tensor]:
        snapshot = {
            "target_key": target_arena.key[1],
            "target_value": target_arena.value[1],
            "target_written": target_arena.written[1],
            "target_conv": target_arena.states[CONV_STATE_ID][1],
            "target_recurrent": target_arena.states[RECURRENT_STATE_ID][1],
            "target_hidden": target_hidden_slots[1],
        }
        if mtp_arena is not None:
            snapshot.update(
                mtp_key=mtp_arena.key[0],
                mtp_value=mtp_arena.value[0],
                mtp_written=mtp_arena.written[0],
            )
        return {name: value.detach().to("cpu", copy=True) for name, value in snapshot.items()}

    def continue_from_prefix(
        *, state_destination: int, page_source: int, page_destination: int, mtp_page: int
    ) -> dict[str, Tensor]:
        row = _BatchRow(
            request_id=f"audit:prefix:fork:{state_destination}",
            token_id=token_ids[-1],
            position=prefix_length,
            state_source_slot=1,
            state_destination_slot=state_destination,
            page_source=page_source,
            page_destination=page_destination,
            page_table=(1, page_destination),
            new_pages=(page_source,),
        )
        continuation_executor.replay_batch((row,))
        continuation_executor.slot.synchronize()
        snapshot = {
            "target_output": continuation_executor.body.hidden[0],
            "target_key": target_arena.key[page_destination],
            "target_value": target_arena.value[page_destination],
            "target_written": target_arena.written[page_destination],
            "target_conv": target_arena.states[CONV_STATE_ID][state_destination],
            "target_recurrent": target_arena.states[RECURRENT_STATE_ID][state_destination],
            "target_hidden": target_hidden_slots[state_destination],
        }
        logits = continuation_executor.body.logits
        if logits is not None:
            snapshot["target_logits"] = logits[0]
        if mtp_executors:
            mtp = mtp_executors[1]
            mtp.replay(
                (
                    _MTPReplayRow(
                        row.request_id,
                        row.token_id,
                        row.position,
                        mtp_page,
                        (0, mtp_page),
                        row.state_source_slot,
                        GREEDY_SAMPLING,
                        0,
                    ),
                ),
                target_hidden_slots[1:2],
                sample=False,
            )
            assert mtp_arena is not None
            snapshot.update(
                mtp_key=mtp_arena.key[mtp_page],
                mtp_value=mtp_arena.value[mtp_page],
                mtp_written=mtp_arena.written[mtp_page],
            )
        return {name: value.detach().to("cpu", copy=True) for name, value in snapshot.items()}

    immutable = prefix_snapshot()
    cold = continue_from_prefix(state_destination=2, page_source=2, page_destination=3, mtp_page=1)
    warm = continue_from_prefix(state_destination=3, page_source=4, page_destination=5, mtp_page=2)
    if not _exact_snapshot(cold, warm) or not _exact_snapshot(immutable, prefix_snapshot()):
        raise Qwen3_5TP2PP2ServingError(
            "prefix cache fork differs across target, recurrent, or MTP state"
        )
    # Exercise the production owned-slot seam, not just a fork into new slots.
    if mtp_arena is not None:
        mtp_arena.key[:1].zero_()
        mtp_arena.value[:1].zero_()
        mtp_arena.written[:1].zero_()
    replay_prefix(owned=True)
    torch.cuda.synchronize(device)
    owned = prefix_snapshot()
    for name, value in (
        ("target_conv", target_arena.states[CONV_STATE_ID][0]),
        ("target_recurrent", target_arena.states[RECURRENT_STATE_ID][0]),
        ("target_hidden", target_hidden_slots[0]),
    ):
        owned[name] = value.detach().to("cpu", copy=True)
    if not _exact_snapshot(immutable, owned):
        raise Qwen3_5TP2PP2ServingError(
            "owned prefill differs from COW target or MTP state"
        )
    evidence = stable_digest(
        {
            "schema": "cubic.qwen3_5.prefix-cache-fork.v1",
            "rank": continuation_executor.body.rank,
            "prefix_tokens": prefix_length,
            "target_kv_bf16_exact": True,
            "target_conv_bf16_exact": True,
            "target_recurrent_storage_dtype": str(
                target_arena.states[RECURRENT_STATE_ID].dtype
            ).removeprefix("torch."),
            "target_recurrent_exact": True,
            "target_hidden_bf16_exact": True,
            "mtp_kv_bf16_exact": bool(mtp_executors),
            "prefix_immutable_after_fork": True,
            "owned_prefill_matches_cow": True,
            "inventory_digest": admission.inventory.inventory_digest,
            "semantic_plan_digest": admission.parallel_plan.semantic_plan_digest,
        }
    )
    for state in target_arena.states.values():
        state.index_fill_(0, state_ids, 0)
    target_arena.key.index_fill_(0, page_ids, 0)
    target_arena.value.index_fill_(0, page_ids, 0)
    target_arena.written.index_fill_(0, page_ids, False)
    target_hidden_slots.index_fill_(0, state_ids, 0)
    if mtp_arena is not None:
        mtp_pages = torch.arange(3, dtype=torch.int64, device=device)
        mtp_arena.key.index_fill_(0, mtp_pages, 0)
        mtp_arena.value.index_fill_(0, mtp_pages, 0)
        mtp_arena.written.index_fill_(0, mtp_pages, False)
    torch.cuda.current_stream(device).synchronize()
    barrier()
    return evidence


def _prepare_mtp_capture(body: Qwen3_5TPMTPBody, request_page_capacity: int) -> None:
    b = body.buffers
    tokens = torch.zeros(body.captured_m, dtype=torch.int64, device=b.hidden.device)
    body.bind_token_ids(tokens)
    b.position_ids.zero_()
    b.token_offsets.zero_()
    b.sequence_lengths.fill_(1)
    b.target_hidden.zero_()
    for row in range(body.captured_m):
        scratch = request_page_capacity + row
        b.page_destination[row].fill_(scratch)
        b.page_table[row].fill_(scratch)


def _median_cost_samples(samples: list[float]) -> float:
    ordered = sorted(samples)
    midpoint = len(ordered) // 2
    if not ordered:
        raise Qwen3_5TP2PP2ServingError("startup graph cost sample is empty")
    if len(ordered) & 1:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _cold_start_owner_limit(body, *, state_slots=1, pages=1):
    """One common physically valid cold owner extent, before any forward."""
    limit = body.arena.spec.state_slot_capacity // state_slots
    storage = getattr(body.arena, "shared_storage", None)
    if storage is not None:
        limit = min(limit, storage.allocator.cold_owner_capacity(
            state_slots=state_slots, pages=pages))
        if dist.is_initialized():
            common = torch.tensor([limit], dtype=torch.int64, device=body.hidden.device)
            all_reduce(common, op=dist.ReduceOp.MIN)
            limit = int(common.item())
    if limit <= 0:
        raise ValueError("startup calibration cannot fit one owned transaction")
    return limit


def _cold_start_owned_rows(body, logical_m, *, position, reset):
    """Ordinary exclusive decode appends, without audit-only COW resources."""
    block = body.arena.layout.block_size_tokens
    page_count = (4 + block - 1) // block
    rows = tuple(_BatchRow(
        request_id=f"startup:owned:{i}", token_id=17 + i, position=position,
        state_source_slot=i, state_destination_slot=i,
        page_source=i * page_count + position // block,
        page_destination=i * page_count + position // block,
        page_table=tuple(range(i * page_count, (i + 1) * page_count)),
        reset_state=reset, new_pages=(tuple(range(i * page_count, (i + 1) * page_count))
                                     if reset else ()),
    ) for i in range(logical_m))
    return _shared_fixture_rows(body, rows)


def _cold_start_depth_rows(body, logical_m, depth):
    """Retain every review checkpoint within a physically owned segment."""
    block = body.arena.layout.block_size_tokens
    pages = (depth + block - 1) // block
    rows = tuple(_BatchRow(
        request_id=f"cost:segment:{physical // depth}", token_id=17 + physical,
        position=physical % depth, state_source_slot=physical // depth,
        state_destination_slot=(physical // depth
            if physical % depth == depth - 1 or physical == logical_m - 1
            else body.arena.spec.state_slot_capacity + physical),
        page_source=(physical // depth) * pages + (physical % depth) // block,
        page_destination=(physical // depth) * pages + (physical % depth) // block,
        page_table=tuple(range((physical // depth) * pages, (physical // depth + 1) * pages)),
        reset_state=physical % depth == 0,
        new_pages=(tuple(range((physical // depth) * pages, (physical // depth + 1) * pages))
                   if physical % depth == 0 else ()),
        segment_length=min(depth, logical_m - physical) if physical % depth == 0 else 0,
    ) for physical in range(logical_m))
    return _shared_fixture_rows(body, rows)


def _cold_start_calibration_signature(label, capacity, body, entry):
    """Bind changed cold ownership/corpus extents before reusing measured prices."""
    storage = getattr(getattr(body, "arena", None), "shared_storage", None)
    if storage is None or label not in ("target-costs", "target-depth-costs", "provider-costs"):
        return None
    block = body.arena.layout.block_size_tokens
    if label == "target-depth-costs":
        extent = tuple((d, min(capacity, d * _cold_start_owner_limit(
            body, state_slots=d, pages=(d + block - 1) // block)))
            for d in range(2, min(capacity, entry) + 1))
        corpus = "checkpoint-segments-v1"
    else:
        pages = (4 + block - 1) // block if label == "target-costs" else 1
        extent = min(capacity, _cold_start_owner_limit(body, pages=pages))
        corpus = ("exclusive-owned-append-v1" if label == "target-costs"
                  else "schema-initialized-proposal-v1")
    return (corpus, extent, storage.page_layout.entry_bytes,
            0 if storage.state_layout is None else storage.state_layout.entry_bytes)


def _cold_start_target_costs(
    executors: dict[int, _CapturedGraphRank],
) -> tuple[tuple[int, float], ...]:
    """Price qualified graphs after every rank has finished input preparation."""

    costs: list[tuple[int, float]] = []
    for capacity, executor in sorted(executors.items()):
        body = executor.body
        shared = getattr(body.arena, "shared_storage", None) is not None
        logical_m = min(capacity, _cold_start_owner_limit(
            body, pages=(4 + body.arena.layout.block_size_tokens - 1)
                        // body.arena.layout.block_size_tokens)) if shared else min(
                            capacity, body.arena.spec.state_slot_capacity)
        _reset_owner(body, body.buffers.page_table.shape[1])
        _reset_production_audit_executor(executor)
        samples: list[float] = []
        for ordinal in range(4):
            rows = (_cold_start_owned_rows(body, logical_m, position=ordinal, reset=ordinal == 0)
                    if shared else _production_audit_rows(
                body,
                logical_m,
                position=ordinal,
                parity=ordinal & 1,
                initialize_from=0 if ordinal == 0 else None,
                allow_underfilled=capacity > body.arena.spec.state_slot_capacity,
            ))
            executor.bind_batch(rows)
            torch.cuda.synchronize(body.hidden.device)
            barrier()
            torch.cuda.synchronize(body.hidden.device)
            executor.replay_bound()
            executor._commit_target_hidden(rows)
            executor.slot.synchronize()
            elapsed = executor.replay_elapsed_ms()
            if ordinal:
                samples.append(elapsed)
        costs.append((capacity, _median_cost_samples(samples)))
        _reset_owner(body, body.buffers.page_table.shape[1])
        _reset_production_audit_executor(executor)
    torch.cuda.synchronize()
    barrier()
    return tuple(costs)


def _cold_start_target_depth_costs(
    executors: dict[int, _CapturedGraphRank], maximum_depth: int,
) -> tuple[tuple[int, int, float], ...]:
    """Price serial segment work on sealed graphs, without additional capture."""

    costs: list[tuple[int, int, float]] = []
    for capacity, executor in sorted(executors.items()):
        body = executor.body
        for depth in range(2, min(capacity, maximum_depth) + 1):
            _reset_owner(body, body.buffers.page_table.shape[1])
            _reset_production_audit_executor(executor)
            logical_m = min(capacity, depth * body.arena.spec.state_slot_capacity)
            if getattr(body.arena, "shared_storage", None) is not None:
                logical_m = min(capacity, depth * _cold_start_owner_limit(
                    body, state_slots=depth,
                    pages=(depth + body.arena.layout.block_size_tokens - 1)
                          // body.arena.layout.block_size_tokens))
            rows = (_cold_start_depth_rows(body, logical_m, depth)
                    if getattr(body.arena, "shared_storage", None) is not None else tuple(
                _BatchRow(
                    request_id=f"cost:segment:{physical // depth}",
                    token_id=17 + physical,
                    position=physical % depth,
                    state_source_slot=physical // depth,
                    state_destination_slot=(physical // depth
                        if physical % depth == depth - 1 or physical == logical_m - 1
                        else body.arena.spec.state_slot_capacity + physical),
                    page_source=physical // depth,
                    page_destination=physical // depth,
                    page_table=(physical // depth,),
                    reset_state=physical % depth == 0,
                    new_pages=(physical // depth,) if physical % depth == 0 else (),
                    segment_length=(min(depth, logical_m - physical)
                                    if physical % depth == 0 else 0),
                )
                for physical in range(logical_m)
            ))
            samples: list[float] = []
            for ordinal in range(4):
                executor.bind_batch(rows)
                torch.cuda.synchronize(body.hidden.device)
                barrier()
                torch.cuda.synchronize(body.hidden.device)
                executor.replay_bound()
                executor._commit_target_hidden(rows)
                executor.slot.synchronize()
                if ordinal:
                    samples.append(executor.replay_elapsed_ms())
            costs.append((capacity, depth, _median_cost_samples(samples)))
        _reset_owner(body, body.buffers.page_table.shape[1])
        _reset_production_audit_executor(executor)
    torch.cuda.synchronize()
    barrier()
    return tuple(costs)


def _cold_start_mtp_proposal_costs(
    executors: dict[int, _CapturedMTPRank],
) -> tuple[tuple[int, float], ...]:
    """Price each resident proposal graph on its public sampling boundary."""

    if not executors:
        return ()
    largest = executors[max(executors)]
    device = largest.body.buffers.hidden.device
    hidden = torch.zeros(
        (largest.captured_m, largest.body.buffers.target_hidden.shape[1]),
        dtype=torch.bfloat16,
        device=device,
    )
    costs: list[tuple[int, float]] = []
    prepared_pages = 0
    for capacity, executor in executors.items():
        # A wide target-review/prefill bucket need not have that many durable
        # draft owners. Price its existing physical graph with the admitted
        # independent-owner extent, retaining private padding for the rest.
        storage = getattr(executor.body.arena, "shared_storage", None)
        logical_m = min(capacity, _cold_start_owner_limit(executor.body))
        prepared_pages = max(prepared_pages, logical_m)
        pages = torch.arange(logical_m, dtype=torch.int64, device=device)
        owner_ids = (tuple(range(logical_m)) if storage is None else
                     tuple(storage.state_capacity - 1 - i for i in range(logical_m)))
        if storage is not None:
            executor.body.arena.reset_state_rows(torch.tensor(
                owner_ids, dtype=torch.int64, device=device))
        executor.body.arena.key.index_fill_(0, pages, 0)
        executor.body.arena.value.index_fill_(0, pages, 0)
        executor.body.arena.written.index_fill_(0, pages, False)
        rows = tuple(
            _MTPReplayRow(
                f"startup:mtp-cost:{capacity}:{physical}",
                1,
                0,
                physical,
                (physical,),
                owner_ids[physical],
                GREEDY_SAMPLING,
                0,
            )
            for physical in range(logical_m)
        )
        samples: list[float] = []
        # The first public proposal replay materializes lazy sampling/runtime
        # state even though its CUDA Graph is already resident.  It is a
        # warmup, not a steady-state price for every request that follows.
        for sample_index in range(4):
            barrier()
            torch.cuda.current_stream(device).synchronize()
            started_ns = time.monotonic_ns()
            executor.replay(rows, hidden[:logical_m])
            torch.cuda.current_stream(device).synchronize()
            if sample_index:
                samples.append((time.monotonic_ns() - started_ns) / 1_000_000)
            barrier()
        costs.append((capacity, _median_cost_samples(samples)))
        diagnostic_event(_LOGGER, 'performance.draft_proposal_calibration_rank_d_physical_m',
            executor.body.rank,
            capacity,
            logical_m,
            costs[-1][1],
        )
    pages = torch.arange(prepared_pages, dtype=torch.int64, device=device)
    largest.body.arena.key.index_fill_(0, pages, 0)
    largest.body.arena.value.index_fill_(0, pages, 0)
    largest.body.arena.written.index_fill_(0, pages, False)
    torch.cuda.current_stream(device).synchronize()
    barrier()
    return tuple(costs)


def _rank_weight_progress(
    rank: int,
    results: Any,
    *,
    unit: str = "bytes",
) -> Callable[[str, int, int], None]:
    def report(section: str, completed: int, total: int) -> None:
        results.send(_LoadProgress(rank, section, completed, total, unit))

    return report


def _seed_provider_state_pair(rows,start,stop,ordinal,*,scratch_state,physical,depth,shared):
    root,last=rows[start],rows[stop-1]
    if shared:
        checkpoints=tuple(row.state_destination_slot for row in rows[start:stop-1])
        source=(root.state_source_slot if ordinal==0 else last.state_destination_slot if ordinal==1
                else checkpoints[ordinal-2])
        destination=last.state_destination_slot if ordinal==0 else checkpoints[ordinal-1]
    else:
        source=(root.state_source_slot if ordinal==0 else last.state_destination_slot if ordinal==1
                else scratch_state+physical*(depth+1)+ordinal-2)
        destination=(last.state_destination_slot if ordinal==0 else
                     scratch_state+physical*(depth+1)+ordinal-1)
    return source,destination


def _execute_pipelined_draft_batch(
    command, executor, providers, hidden, admission, capacities,
    *, rank, sampler_rank, grammar=None,
):
    """Carried proposal/review/realignment using the common sealed portfolio.

    A finite pending record belongs to each durable request owner. First ingress
    seeds it with the provider's existing qualified graphs; steady rounds use
    one captured transaction. Ragged retirement, a different current depth or
    constraints restore the pre-bonus provider checkpoint before the general path.
    """
    first = providers[min(providers)] if providers else None
    pool = getattr(first, "_pending_draft_pool", None)
    if pool is None:
        return None
    speculative, levels = _ragged_speculative_plan(command.rows)
    portfolio = getattr(executor, "_pipelined_draft_rounds", {})
    round_ = portfolio.get(len(levels)) if portfolio else getattr(executor, "_pipelined_draft_round", None)
    depth = round_.depth if round_ is not None else pool.depth
    budget = bool(
        command.constraints and grammar is not None and round_ is not None
        and execution_ready(round_, "budget_qualified")
        and getattr(round_, "budget", None) is not None
        and all(binding.meta.kind.value == "identity" and binding.reasoning_control is not None
                for binding in command.constraints)
    )
    grammar_mode = bool(command.constraints and grammar is not None
                        and execution_ready(round_, "grammar_qualified") and not budget)
    target_constraints = bool(grammar_mode and execution_ready(round_, "target_constraints_qualified"))
    eligible = (
        pool is not None and round_ is not None
        and (not command.constraints or budget or grammar_mode) and len(levels) == depth
        and len(speculative) == len(_batch_segments(command.rows))
        and len(speculative) <= round_.C
        and all(row.sampling == command.rows[segment.start].sampling
                for segment in speculative for row in command.rows[segment.start:segment.stop])
        and all(segment.draft_count == depth
                and command.rows[segment.start].mtp_position + 2 * depth
                    < admission.max_sequence_tokens
                and (command.rows[segment.start].mtp_position + 2 * depth)
                    // executor._block_size
                    < len(command.rows[segment.start].mtp_page_table)
                for segment in speculative)
    )
    if not eligible:
        if pool is not None:
            pool.restore(first, command.rows)
        return None
    executor._pipelined_draft_round = round_
    started = time.monotonic_ns()
    reserved = tuple({slot for row in command.rows for slot in
                      (row.state_source_slot, row.state_destination_slot)})
    for provider in providers.values():
        provider.reserve_state_slots(reserved)
    rows = list(command.rows)
    seeds = []
    for physical, segment in enumerate(speculative):
        root, last = rows[segment.start], rows[segment.stop - 1]
        pending = pool.metadata[root.state_source_slot]
        key = (root.request_id, root.mtp_position, root.token_id, root.sampling,
               root.sample_ordinal, root.position_axes)
        if (pending is not None and pending[:6] == key and len(pending[6]) == depth
            and root.state_source_slot == last.state_destination_slot):
            for ordinal, token in enumerate(pending[6]):
                index = segment.start + ordinal + 1
                rows[index] = replace(rows[index], token_id=token)
        else:
            pool.restore(first, (root,))
            seeds.append((physical, segment))
    if seeds:
        seed = providers[_bucket_for_logical_m(len(seeds), capacities)]
        if target_constraints:
            seed._sampling.constraint_bound = False
        feedback = torch.stack(tuple(hidden[rows[s.start].state_source_slot]
                                     for _, s in seeds))
        for ordinal in range(depth):
            replay_rows = []
            for physical, segment in seeds:
                root, last = rows[segment.start], rows[segment.stop - 1]
                source,destination=_seed_provider_state_pair(
                    rows,segment.start,segment.stop,ordinal,scratch_state=round_.scratch_state,
                    physical=physical,depth=depth,
                    shared=getattr(executor.body.arena,"shared_storage",None) is not None)
                replay_rows.append(_MTPReplayRow(
                    root.request_id, root.token_id, root.mtp_position + ordinal,
                    rows[segment.start + ordinal].mtp_page_destination,
                    root.mtp_page_table, source, root.sampling,
                    root.sample_ordinal + ordinal,
                    state_destination_slot=destination,
                    position_axes=rows[segment.start + ordinal].position_axes,
                ))
            tokens = (None if ordinal == 0 else seed.chain_tokens(
                seed._verification_draft_tokens[ordinal - 1], tuple(range(len(seeds)))))
            if budget or grammar_mode and not target_constraints:
                owners = [row.request_id in grammar.bindings for row in replay_rows]
                seed._sampling.bind_constraints(
                    [grammar.mask(row.request_id, ordinal) if owned else None
                     for row, owned in zip(replay_rows, owners, strict=True)], owners,
                )
            seed.replay(tuple(replay_rows), feedback, input_tokens=tokens,
                        validity_out=seed._verification_validity[ordinal])
            seed.stash_proposal(ordinal, len(seeds))
            if budget or grammar_mode and not target_constraints:
                sampled = seed._verification_draft_tokens[ordinal, :len(seeds)].cpu().tolist()
                for row, token in zip(replay_rows, sampled, strict=True):
                    if row.request_id in grammar.bindings:
                        grammar.propose(row.request_id, int(token), ordinal)
            if ordinal + 1 < depth:
                feedback = seed.chain_hidden(tuple(range(len(seeds))))
            for index, (_, segment) in enumerate(seeds):
                owner = rows[segment.stop - 1].state_destination_slot
                pool.scores[ordinal, :, owner].copy_(seed._verification_q_scores[ordinal, :, index])
                pool.ids[ordinal, :, owner].copy_(seed._verification_q_token_ids[ordinal, :, index])
                pool.tokens[ordinal, owner].copy_(seed._verification_draft_tokens[ordinal, index])
        proposals = seed.materialize_proposals((len(seeds),) * depth)
        for ordinal in range(depth):
            for (_, segment), token in zip(seeds, proposals[ordinal], strict=True):
                index = segment.start + ordinal + 1
                rows[index] = replace(rows[index], token_id=token)
    if budget:
        def stage_budget():
            sessions = [grammar.pool.sessions.get(rows[segment.start].request_id)
                        for segment in speculative]
            for session in sessions:
                if session is not None and session.stop_ids not in ((), round_.budget.stop_token_ids):
                    raise Qwen3_5TP2PP2ServingError("budget stop policy differs from its sealed graph")
            round_.budget.bind(
                [None if session is None else session.matcher.integer_snapshot()
                 for session in sessions],
                stop_enabled=[session is not None and bool(session.stop_ids) for session in sessions],
                terminated=[session is not None and session.terminated for session in sessions],
            )
        stage_budget()
    if target_constraints:
        completed = _execute_target_constrained_carried_round(
            round_, tuple(rows), speculative, grammar,
        )
    elif grammar_mode:
        completed = _execute_masked_carried_round(
            round_, tuple(rows), speculative, grammar,
            seeded={rows[segment.start].request_id for _, segment in seeds},
        )
    else:
        completed = round_.run(tuple(rows), speculative, **({"budget": True} if budget else {}))
    results = _pipelined_results(tuple(rows), speculative, completed, pool=pool)
    if budget:
        grammar.commit({part.request_id: part.output_token_ids for part in results
                        if part.request_id in grammar.bindings})
        stage_budget()
        executor._sampling.constraint_bound = False
        for provider in providers.values():
            provider._sampling.constraint_bound = False
    elif grammar_mode:
        for provider in providers.values():
            provider._sampling.constraint_bound = False
    wall_ms = (time.monotonic_ns() - started) / 1_000_000
    window = round_.active_draft_window
    target_ms = executor.replay_elapsed_ms()
    if target_constraints:
        round_._constraint_forward_ms = target_ms
    return _BatchResult(
        rank, command.batch_id, tuple(row.request_id for row in command.rows), (),
        tuple(results) if rank == sampler_rank else (), target_ms,
        (), wall_ms,
        resident_draft_depth=depth if execution_ready(window) else 0,
        draft_ingress=bool(seeds),
        pipelined_draft_depth=depth if grammar_mode else 0,
        grammar_cost=grammar.cost_witness() if grammar is not None else None,
    )


def _pipelined_results(rows, segments, completed, *, pool):
    """Install one immutable carried proposal and its durable owner coordinates."""
    results = []
    for segment, result in zip(segments, completed, strict=True):
        accepted, bonus = result[:2]
        next_proposals = tuple(result[2:-1])
        root, last = rows[segment.start], rows[segment.stop - 1]
        step = (None if root.position_axes is None else tuple(
            rows[segment.start + 1].position_axes[i] - root.position_axes[i]
            for i in range(3)))
        axes = (None if step is None else tuple(
            root.position_axes[i] + step[i] * (accepted + 1) for i in range(3)))
        pool.metadata[last.state_destination_slot] = (
            root.request_id, root.mtp_position + accepted + 1, bonus, root.sampling,
            root.sample_ordinal + accepted + 1, axes, next_proposals,
        )
        results.append(_SegmentResult(
            root.request_id, accepted + 1,
            (*(rows[segment.start + i + 1].token_id for i in range(accepted)), bonus),
        ))
    return tuple(results)


def _constraint_mask_worker_count(state, pool, owners):
    maximum = min(getattr(pool, "batch_mask_workers", 1), owners)
    work = getattr(state, "_constraint_mask_work_per_owner_ns", 0.) * owners
    forward_ms = getattr(state, "_constraint_forward_ms", 0.)
    if maximum > 1 and work > 0 and forward_ms > 0:
        elapsed = time.monotonic_ns() - state._constraint_forward_queued_ns
        budget = max(1, int(forward_ms * 1e6) - elapsed
                     - getattr(state, "_constraint_mask_upload_ns", 0))
        desired = min(maximum, max(1, math.ceil(work / budget)))
        return min(maximum, 1 << (desired - 1).bit_length())
    return 1


def _constraint_prefix_masks_with_overlap(state, grammar, proposals, *, workers=None):
    """Budget existing CPU cohorts against this graph's remaining target span."""
    if not proposals:
        return {}
    if workers is None:
        workers = _constraint_mask_worker_count(state, grammar.pool, len(proposals))
    # Active CPU counts include runtime helpers and omit service waits.
    # Reserve whole-phase wall time across the assigned CPU workers too;
    # neither quantity is isolated native matcher utilization.
    wall_started = time.monotonic_ns()
    cpu_started = time.process_time_ns()
    prepared = grammar.target_prefix_masks_many(proposals, workers=workers)
    reserved_work = max(time.process_time_ns() - cpu_started,
                        (time.monotonic_ns() - wall_started) * workers)
    state._constraint_mask_work_per_owner_ns = reserved_work / len(proposals)
    state._constraint_mask_workers = workers
    return prepared


def _execute_target_constrained_carried_round(round_, rows, segments, grammar):
    """Overlap target forward with p masks; retain the actual unmasked q."""
    valid_prefixes = {}

    def prepare_constraints():
        masks, owners = [None] * len(rows), [False] * len(rows)
        constrained = tuple(segment for segment in segments
                            if rows[segment.start].request_id in grammar.bindings)
        proposals = {
            rows[segment.start].request_id: tuple(
                rows[segment.start + depth + 1].token_id for depth in range(round_.depth))
            for segment in constrained
        }
        if len(proposals) != len(constrained):
            raise Qwen3_5TP2PP2ServingError("target grammar prefixes have duplicate request ownership")
        prepared = _constraint_prefix_masks_with_overlap(round_, grammar, proposals)
        for segment in segments:
            request = rows[segment.start].request_id
            if request not in prepared:
                continue
            prefix_masks, valid_prefixes[request] = prepared[request]
            for depth, mask in enumerate(prefix_masks):
                masks[segment.start + depth], owners[segment.start + depth] = mask, True
        upload_started = time.monotonic_ns()
        round_.target._sampling.bind_constraints(masks, owners)
        round_._constraint_mask_upload_ns = time.monotonic_ns() - upload_started

    completed = round_.run(rows, segments, target_constraints=True,
                           prepare_constraints=prepare_constraints)
    committed = {}
    for segment, result in zip(segments, completed, strict=True):
        request = rows[segment.start].request_id
        if request not in valid_prefixes:
            continue
        accepted, bonus = result[:2]
        if accepted > valid_prefixes[request]:
            raise Qwen3_5TP2PP2ServingError("target accepted an invalid grammar proposal")
        committed[request] = (
            *(rows[segment.start + depth + 1].token_id for depth in range(accepted)), bonus,
        )
    grammar.commit(committed)
    round_.target._sampling.constraint_bound = False
    round_.head._sampling.constraint_bound = False
    return completed


def _execute_masked_carried_round(round_, rows, segments, grammar, *, seeded):
    """Use carried q for one masked review, then sample the next private q.

    The entire model review/repair/alignment/head projection is one sealed
    graph. CPU grammar work occurs only at its real token dependency boundary.
    The q and p distributions always use the same matcher prefix.
    """
    head, target, depth = round_.head, round_.target, round_.depth
    masks, owners = [None] * len(rows), [False] * len(rows)
    for segment in segments:
        request = rows[segment.start].request_id
        if request not in grammar.bindings:
            continue
        for d in range(depth + 1):
            physical = segment.start + d
            masks[physical], owners[physical] = grammar.mask(request, d), True
            if d < depth and request not in seeded:
                grammar.propose(request, rows[physical + 1].token_id, d)
    target._sampling.bind_constraints(masks, owners)
    decisions = round_.run(rows, segments, review_only=True)
    committed = {
        rows[segment.start].request_id: (
            *(rows[segment.start + d + 1].token_id for d in range(decision[0])), decision[1],
        ) for segment, decision in zip(segments, decisions, strict=True)
        if rows[segment.start].request_id in grammar.bindings
    }
    future = grammar.fork_after(committed)
    sampled = []
    n = len(segments)
    stochastic = not all(rows[s.start].sampling.temperature == 0. for s in segments)
    for d in range(depth):
        head._sampling.bind_constraints(
            [future[rows[segment.start].request_id].mask()
             if rows[segment.start].request_id in future else None for segment in segments],
            [rows[segment.start].request_id in future for segment in segments],
        )
        tokens = round_.run_grammar_proposal(d, n, stochastic=stochastic)
        sampled.append(tokens)
        for segment, token in zip(segments, tokens, strict=True):
            request = rows[segment.start].request_id
            if request in future:
                future[request].accept(token)
    # Every rank consumes its complete immutable guard history before publishing.
    # The captured owner-copy phase may fill private pending planes early, but
    # their durable metadata is installed only after these checks and commit.
    head.materialize_proposals((n,) * depth)
    grammar.commit(committed)
    target._sampling.constraint_bound = False
    head._sampling.constraint_bound = False
    return tuple((decision[0], decision[1], *(tokens[r] for tokens in sampled), 1)
                 for r, decision in enumerate(decisions))


def _grammar_mask_worker_budget() -> int:
    """Bound optional CPU work by process affinity, quota and rank multiplicity."""
    cores = len(os.sched_getaffinity(0))
    try:
        quota, period = (Path("/sys/fs/cgroup/cpu.max").read_text().split())
        if quota != "max":
            cores = min(cores, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    return max(1, min(4, cores // world))


def _request_grammar_pool(pool: Any, command: _BatchCommand, admission: Any, vocab_size: int):
    """A reasoning ledger needs no tokenizer/compiler; load those on first grammar."""
    if command.constraints:
        from cubic_serve.generation.structured_sampling import GrammarPool
        from cubic_serve.contracts.request import StructuredOutputKind

        stop_ids = getattr(admission, "constraint_stop_token_ids", ())
        if pool is None and stop_ids:
            pool = GrammarPool(None, vocab_size=vocab_size, stop_token_ids=stop_ids,
                               batch_mask_workers=_grammar_mask_worker_budget())
        needs_factory = any(binding.meta.kind is not StructuredOutputKind.IDENTITY
                            for binding in command.constraints)
    else:
        needs_factory = False
    if command.constraints and (pool is None or needs_factory and pool.factory is None):
        from cubic_serve.generation.structured_sampling import GrammarPool
        from cubic_serve.generation.xgrammar import frontend_constraint_factory
        from cubic_serve.models.builtin import resolve_model_definition

        source = getattr(admission, "model_definition_source", None)
        definition = resolve_model_definition(
            admission.config,
            definition=None if source is None else source.definition,
        )
        input_factory, output_factory = definition.frontend_factories(multimodal=False)
        processor, codec = input_factory(admission.model_root), output_factory(admission.model_root)
        factory = frontend_constraint_factory(processor, codec, vocab_size)
        if pool is None:
            pool = GrammarPool(factory, vocab_size=vocab_size,
                               stop_token_ids=codec.eos_token_ids,
                               batch_mask_workers=_grammar_mask_worker_budget())
        else:
            if pool.stop_ids != codec.eos_token_ids:
                raise Qwen3_5TP2PP2ServingError("grammar stop policy differs from admission")
            pool.factory = factory
    if pool is not None and command.released_constraints:
        pool.release(command.released_constraints)
    return pool


def execute_draft_batch(
    command: _BatchCommand,
    executor: Any,
    mtp_executors: dict[int, CapturedDraftRank],
    target_hidden_slots: Tensor,
    admission: Any,
    physical_capacities: tuple[int, ...],
    *,
    rank: int,
    sampler_rank: int,
    mtp_feeding_profile: _StepFeedingProfile | None = None,
    grammar_pool: Any = None,
) -> _BatchResult:
    """Provider-independent proposal, target review and accepted-state commit.

    The common scheduler supplies the complete transaction and exclusive lease.
    Models supply captured bodies and feedback/state fields, never verification
    or scheduling policy. Caller publishes only after every rank completes.
    """
    grammar = grammar_pool.begin(command.constraints) if command.constraints else None
    pipelined = _execute_pipelined_draft_batch(
        command,
        executor,
        mtp_executors,
        target_hidden_slots,
        admission,
        physical_capacities,
        rank=rank,
        sampler_rank=sampler_rank,
        grammar=grammar,
    )
    if pipelined is not None:
        return pipelined
    step_wall_started_ns = time.monotonic_ns()
    # A ragged proposal wave may be smaller than the review graph. Padding
    # must preserve checkpoints from every depth, including retired waves,
    # until acceptance and catch-up have completed under this common lease.
    reserved_slots = tuple({slot for row in command.rows for slot in
                            (row.state_source_slot, row.state_destination_slot)})
    for draft_executor in mtp_executors.values():
        draft_executor.reserve_state_slots(reserved_slots)
    proposal_cost_samples: list[tuple[int, float]] = []
    proposal_wall_ms = 0.0
    segments = _batch_segments(command.rows)
    speculative_segments, proposal_levels = _ragged_speculative_plan(command.rows)
    draft_chains: tuple[tuple[int, ...], ...] = ()
    proposal_history: _CapturedMTPRank | None = None
    target_rows = list(command.rows)
    if admission.num_speculative_tokens and speculative_segments:
        proposal_wall_started_ns = (
            time.monotonic_ns()
            if mtp_feeding_profile is not None and mtp_feeding_profile.active
            else 0
        )
        if len(proposal_levels) > admission.max_speculative_tokens:
            raise Qwen3_5TP2PP2ServingError(
                "MTP batch draft depth exceeds provider capacity"
            )
        proposal_history = mtp_executors[
            _bucket_for_logical_m(len(speculative_segments), physical_capacities)
        ]
        proposal_buckets = _ragged_proposal_buckets(
            proposal_levels, physical_capacities
        )
        proposal_hidden = torch.stack(
            tuple(
                target_hidden_slots[target_rows[segment.start].state_source_slot]
                for segment in speculative_segments
            )
        )
        for depth, (level, proposal_bucket) in enumerate(
            zip(proposal_levels, proposal_buckets, strict=True)
        ):
            proposal_executor = mtp_executors[proposal_bucket]
            mtp_rows = tuple(
                _MTPReplayRow(
                    row.request_id,
                    row.token_id,
                    row.mtp_position,
                    row.mtp_page_destination,
                    row.mtp_page_table,
                    (row.state_source_slot if depth == 0 else
                     target_rows[segment.start + depth - 1].state_destination_slot),
                    row.sampling,
                    row.sample_ordinal,
                    state_destination_slot=row.state_destination_slot,
                    position_axes=row.position_axes,
                )
                for segment in level
                for row in (target_rows[segment.start + depth],)
            )
            if admission.dynamic_speculation:
                proposal_history._proposal_cost_events[depth][0].record()
            input_tokens = None
            if depth:
                previous_rows = {
                    segment.index: physical
                    for physical, segment in enumerate(proposal_levels[depth - 1])
                }
                input_tokens = proposal_executor.chain_tokens(
                    proposal_history._verification_draft_tokens[depth - 1],
                    tuple(previous_rows[segment.index] for segment in level),
                )
            if grammar is not None:
                owners = [row.request_id in grammar.bindings for row in mtp_rows]
                proposal_executor._sampling.bind_constraints(
                    [
                        grammar.mask(row.request_id, depth) if owned else None
                        for row, owned in zip(mtp_rows, owners, strict=True)
                    ],
                    owners,
                )
            proposal_executor.replay(
                mtp_rows, proposal_hidden, input_tokens=input_tokens,
                validity_out=proposal_history._verification_validity[depth],
            )
            if admission.dynamic_speculation:
                # Match the startup replay/sampler price. History retention
                # and feedback compaction belong to remaining round overhead.
                proposal_history._proposal_cost_events[depth][1].record()
            proposal_history.stash_proposal(
                depth,
                len(mtp_rows),
                source=proposal_executor,
            )
            if grammar is not None:
                values = (
                    proposal_history._verification_draft_tokens[depth, : len(level)].cpu().tolist()
                )
                for row, token in zip(mtp_rows, values, strict=True):
                    if row.request_id in grammar.bindings:
                        grammar.propose(row.request_id, int(token), depth)
            if depth + 1 < len(proposal_levels):
                compact_rows = {
                    segment.index: physical
                    for physical, segment in enumerate(level)
                }
                proposal_hidden = proposal_executor.chain_hidden(
                    tuple(
                        compact_rows[segment.index]
                        for segment in proposal_levels[depth + 1]
                    )
                )
        # One completed proposal chain crosses the host boundary. Until this
        # point every proposal and validity predicate stays lease-private on
        # device, including levels whose compact row order has changed.
        proposal_tokens = proposal_history.materialize_proposals(
            tuple(len(level) for level in proposal_levels)
        )
        if admission.dynamic_speculation:
            # The guards/history D2H above already completed every event.
            # Reading durations here adds no per-depth wait or synchronization.
            proposal_cost_samples.extend(
                (len(level), start.elapsed_time(end))
                for level, (start, end) in zip(
                    proposal_levels,
                    proposal_history._proposal_cost_events[:len(proposal_levels)],
                    strict=True,
                )
            )
        chains: list[list[int]] = [[] for _ in speculative_segments]
        for depth, (level, tokens) in enumerate(zip(proposal_levels, proposal_tokens, strict=True)):
            for segment, token in zip(level, tokens, strict=True):
                chains[segment.index].append(token)
                target_rows[segment.start + depth + 1] = replace(
                    target_rows[segment.start + depth + 1], token_id=token
                )
        draft_chains = tuple(tuple(chain) for chain in chains)
        if proposal_wall_started_ns:
            proposal_wall_ms = (
                time.monotonic_ns() - proposal_wall_started_ns
            ) / 1_000_000
    bound_rows = tuple(target_rows)
    fully_speculative = bool(speculative_segments) and len(
        speculative_segments
    ) == len(segments)
    target_wall_started_ns = time.monotonic_ns()
    _replay_target_with_mtp_maintenance(
        executor, bound_rows, speculative_segments, mtp_executors,
        target_hidden_slots,
    )
    if grammar is not None:
        masks: list[Tensor | None] = [None] * len(bound_rows)
        owners = [False] * len(bound_rows)
        for start, stop in segments:
            request_id = bound_rows[start].request_id
            if request_id not in grammar.bindings:
                continue
            indices = range(start, stop) if bound_rows[start].is_decode else (stop - 1,)
            for depth, physical in enumerate(indices):
                masks[physical] = grammar.mask(request_id, depth)
                owners[physical] = True
        executor._sampling.bind_constraints(masks, owners)
    if (
        fully_speculative
        and proposal_wall_ms
        and mtp_feeding_profile is not None
        and mtp_feeding_profile.active
    ):
        mtp_feeding_profile.record("mtp_proposal_wall_cpu_ms", proposal_wall_ms)
    if fully_speculative:
        executor.gather_candidates(bound_rows)
        token_ids = ()
    else:
        token_ids = executor.sample_tokens(bound_rows)
    target_wall_ms = (time.monotonic_ns() - target_wall_started_ns) / 1_000_000
    segment_results: tuple[_SegmentResult, ...] = ()
    if admission.num_speculative_tokens and speculative_segments:
        verify_wall_started_ns = (
            time.monotonic_ns()
            if mtp_feeding_profile is not None and mtp_feeding_profile.active
            else 0
        )
        if proposal_history is None:
            raise Qwen3_5TP2PP2ServingError("MTP proposal executor is unavailable")
        decisions = proposal_history.launch_verification(
            executor,
            tuple(
                tuple(segment.start for segment in level)
                for level in proposal_levels
            ),
        )
        broadcast_started_ns = (
            time.monotonic_ns() if mtp_feeding_profile is not None else 0
        )
        dist.broadcast(
            decisions[: len(proposal_levels)],
            src=sampler_rank,
            group=cast(ProcessGroup, dist.group.WORLD),
        )
        if mtp_feeding_profile is not None:
            mtp_feeding_profile.record(
                "mtp_acceptance_broadcast_wall_cpu_ms",
                (time.monotonic_ns() - broadcast_started_ns) / 1_000_000,
            )
        decision_depths = tuple(
            tuple(
                (int(accepted), int(candidate))
                for accepted, candidate in decisions[depth, : len(level)]
                .cpu()
                .tolist()
            )
            for depth, level in enumerate(proposal_levels)
        )
        decision_rows = _ragged_acceptance_rows(
            speculative_segments,
            proposal_levels,
            decision_depths,
        )
        # Fully-speculative target replay is asynchronous until
        # verification consumes its logits.  Price the target
        # bucket only at this real completion boundary; ending
        # the timer before verification hides graph-padding cost
        # inside a shape-independent overhead estimate.
        target_wall_ms = (time.monotonic_ns() - target_wall_started_ns) / 1_000_000
        if (
            fully_speculative
            and mtp_feeding_profile is not None
            and mtp_feeding_profile.active
        ):
            mtp_feeding_profile.record(
                "mtp_target_wall_cpu_ms",
                target_wall_ms,
            )
        if fully_speculative and verify_wall_started_ns:
            mtp_feeding_profile.record(
                "mtp_verify_wall_cpu_ms",
                (time.monotonic_ns() - verify_wall_started_ns) / 1_000_000,
            )
        proposal_history.flush_verification_profile()
        accepted_counts = tuple(accepted for accepted, _ in decision_rows)
        if rank == sampler_rank or grammar is not None:
            built_results = tuple(
                _SegmentResult(
                    bound_rows[segment.start].request_id,
                    accepted + 1,
                    (*draft_chain[:accepted], next_token),
                )
                for segment, draft_chain, (accepted, next_token) in zip(
                    speculative_segments,
                    draft_chains,
                    decision_rows,
                    strict=True,
                )
            )
        catchup_rows: list[_MTPReplayRow] = []
        catchup_hidden: list[Tensor] = []
        for segment, accepted in zip(
            speculative_segments, accepted_counts, strict=True
        ):
            if accepted != segment.draft_count:
                continue
            previous_row = bound_rows[segment.stop - 2]
            draft_row = bound_rows[segment.stop - 1]
            catchup_rows.append(
                _MTPReplayRow(
                    draft_row.request_id,
                    draft_row.token_id,
                    draft_row.mtp_position,
                    draft_row.mtp_page_destination,
                    draft_row.mtp_page_table,
                    previous_row.state_destination_slot,
                    draft_row.sampling,
                    draft_row.sample_ordinal,
                    state_destination_slot=draft_row.state_destination_slot,
                    position_axes=draft_row.position_axes,
                )
            )
            catchup_hidden.append(
                target_hidden_slots[previous_row.state_destination_slot]
            )
        if catchup_rows:
            catchup_wall_started_ns = (
                time.monotonic_ns()
                if mtp_feeding_profile is not None and mtp_feeding_profile.active
                else 0
            )
            mtp_executor = mtp_executors[
                _bucket_for_logical_m(len(catchup_rows), physical_capacities)
            ]
            mtp_executor.replay(
                tuple(catchup_rows),
                torch.stack(tuple(catchup_hidden)),
                sample=False,
            )
            if catchup_wall_started_ns:
                mtp_feeding_profile.record(
                    "mtp_catchup_wall_cpu_ms",
                    (time.monotonic_ns() - catchup_wall_started_ns) / 1_000_000,
                )
        _repair_speculative_state(
            executor,
            target_hidden_slots,
            bound_rows,
            speculative_segments,
            accepted_counts,
            draft_state_fields=proposal_history.state_fields,
        )
        if rank == sampler_rank:
            segment_results = built_results
    if grammar is not None:
        committed = {
            part.request_id: part.output_token_ids
            for part in (built_results if speculative_segments else ())
        }
        if not fully_speculative:
            local_tokens = (
                token_ids
                if token_ids
                else tuple(int(t) for t in executor._token_ids[: len(bound_rows)].cpu().tolist())
            )
            for start, stop in segments:
                request_id = bound_rows[start].request_id
                if request_id in grammar.bindings and request_id not in committed:
                    committed[request_id] = (local_tokens[stop - 1],)
        grammar.commit({key: value for key, value in committed.items() if key in grammar.bindings})
        executor._sampling.constraint_bound = False
        for provider in mtp_executors.values():
            provider._sampling.constraint_bound = False
    return _BatchResult(
        rank, command.batch_id, tuple(row.request_id for row in command.rows),
        token_ids, segment_results, executor.replay_elapsed_ms(),
        tuple(proposal_cost_samples),
        (time.monotonic_ns() - step_wall_started_ns) / 1_000_000,
        grammar_cost=grammar.cost_witness() if grammar is not None else None,
    )



def _worker(
    rank: int,
    init_method: str,
    admission: _Admission,
    commands: Connection,
    shared_commands: Any,
    shared_command_capacity: int,
    results: Any,
    nccl_timeout_seconds: int,
    audit_statuses: Any,
    audit_barrier: Any,
    continuation_interrupt: Any = None,
) -> None:
    # The controller owns terminal interrupts and the bounded shutdown sequence.
    # SIGINT stays blocked through spawn/import until this handler is installed.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})
    configure_logging()
    # Spawned ranks inherit explicit OpenMP/MKL settings from their launcher.
    if not any(name in os.environ for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")):
        torch.set_num_threads(4)
    from cubic_serve.executor.resident_window import prepare_resident_communication

    prepare_resident_communication()
    request_id: str | None = None
    failure_reported = False

    def report_failure() -> None:
        nonlocal failure_reported
        if not failure_reported:
            failure_reported = True
            results.send(_Failure(rank, request_id, traceback.format_exc()))

    sampler_rank = _sampler_rank(admission.parallel_plan)
    feeding_profile = _new_step_feeding_profile(f"worker-rank-{rank}")
    mtp_feeding_profile = (
        _new_step_feeding_profile(f"mtp-worker-rank-{rank}")
        if admission.num_speculative_tokens
        else None
    )
    shared_command_view = memoryview(shared_commands).cast("B")
    if len(shared_command_view) != shared_command_capacity:
        raise Qwen3_5TP2PP2ServingError("shared command arena differs from its capacity")
    try:
        configure_reference_math()
        placement = admission.parallel_plan.topology.placements[rank]
        memory_budget = admission.device_memory_budgets.get(placement.device_index)
        if memory_budget is None:
            memory_budget = DeviceMemoryBudget(
                read_cuda_memory_snapshot(f"cuda:{placement.device_index}"))
        with (
            initialize_eager_nccl(
                admission.sealed,
                rank=rank,
                init_method=init_method,
                timeout_seconds=nccl_timeout_seconds,
            ) as runtime,
            _report_before_teardown(report_failure),
            operator_strategy_scope(
                admission.operator_config,
                device=torch.cuda.get_device_name(runtime.device),
                model=admission.operator_model_id,
            ),
            ExitStack() as input_lifetime,
        ):
            memory_budget.apply_allocator_limit()
            from cubic_serve.executor.vision import (
                finish_owned_input_encoder,
                prepare_owned_input_encoder,
                start_owned_input_encoder,
            )
            owned_encoder = prepare_owned_input_encoder(
                admission, topology=admission.parallel_plan.topology,
                rank=rank, device=runtime.device)
            diagnostic_event(_LOGGER, 'performance.operator_strategy_rank_d_model_s_device',
                rank,
                admission.operator_model_id,
                torch.cuda.get_device_name(runtime.device),
                admission.operator_config.digest,
            )
            physical_capacities = _physical_graph_capacities(
                admission.dynamic_max_speculative_tokens,
                pipeline_parallel_size=admission.parallel_plan.topology.pp_size,
                max_running_requests=admission.max_running_requests,
                include_speculative=True,
            )
            graph_capacities = _serving_graph_capacities(
                physical_capacities,
                admission.max_running_requests,
                pipeline_parallel_size=admission.parallel_plan.topology.pp_size,
                max_speculative_tokens=admission.dynamic_max_speculative_tokens,
            )
            wave_enabled = _uses_wavefront(
                physical_capacities,
                graph_capacities,
                admission.parallel_plan.topology.pp_size,
            )
            capture_order = _graph_capture_order(physical_capacities)
            load_weights = (
                load_qwen3_5_moe_rank_weights
                if isinstance(admission.config, Qwen3_5MoeConfig)
                else load_qwen3_5_rank_weights
            )
            weights = load_weights(
                inventory=admission.inventory,
                weight_plan=admission.weight_plan,
                parallel_plan=admission.parallel_plan,
                rank=rank,
                device=runtime.device,
                progress=_rank_weight_progress(rank, results),
            )
            # EP ranks own different expert shards and can finish loading at
            # different times.  Do not let an early rank enter collective
            # model warmup while another rank is still materializing weights.
            barrier()
            model_table_pages = (admission.max_sequence_tokens + _BLOCK_SIZE - 1) // _BLOCK_SIZE
            bootstrap_sequence_tokens = (
                _BLOCK_SIZE if admission.auto_max_sequence_tokens else admission.max_sequence_tokens
            )
            bootstrap_table_pages = (bootstrap_sequence_tokens + _BLOCK_SIZE - 1) // _BLOCK_SIZE
            bootstrap_pages = minimum_kv_page_capacity(
                bootstrap_table_pages,
                graph_capacities[-1],
            )
            # Final cache allocation must leave room to repeat target-family
            # construction, including private operator calibration fixtures.
            torch.cuda.reset_peak_memory_stats(runtime.device)
            diagnostic_event(_LOGGER, 'performance.startup_probe_rank_d_body_build_begin',
                rank,
            )
            body, bodies, wave_lanes = _build_body_family(
                admission,
                weights,
                runtime,
                physical_capacities,
                max_sequence_tokens=bootstrap_sequence_tokens,
                page_capacity=bootstrap_pages,
                state_capacity=_CAPTURED_M,
                scratch_capacity=physical_capacities[-1],
                wave_enabled=wave_enabled,
            )
            diagnostic_event(_LOGGER, 'performance.startup_probe_rank_d_body_build_complete',
                rank,
            )
            construction_transient_bytes = max(
                torch.cuda.max_memory_allocated(runtime.device)
                - torch.cuda.memory_allocated(runtime.device),
                body._construction_transient_bytes,
                0,
            )
            # Materialize shape-dependent CUDA/NCCL runtime allocations before
            # giving all remaining memory to the final BF16 cache arena.
            _warm_body_family(
                bodies,
                wave_lanes,
                physical_capacities,
                rank=rank,
                device=runtime.device,
            )
            diagnostic_event(_LOGGER, 'performance.startup_probe_rank_d_bootstrap_warm_complete',
                rank,
            )
            qualification_progress = _rank_weight_progress(rank, results, unit="buckets")
            profiled_graph_bytes, transient_graph_bytes, overlap_plans = _profile_cuda_graph_memory(
                bodies,
                capture_order,
                rank=rank,
                device=runtime.device,
                runtime=runtime,
                admission=admission,
                weights=weights,
                progress=qualification_progress,
            )
            transient_graph_bytes = max(transient_graph_bytes, construction_transient_bytes)
            graph_memory = torch.tensor(
                (profiled_graph_bytes, transient_graph_bytes),
                dtype=torch.int64, device=runtime.device,
            )
            dist.all_reduce(graph_memory, op=dist.ReduceOp.MAX)
            graph_reserve_bytes, transient_reserve_bytes = map(int, graph_memory.tolist())
            del graph_memory
            # Weight staging and eager warmup may leave dead allocations in
            # PyTorch's cache.  They are not serving state and must not reduce
            # the BF16 KV capacity inferred from driver-visible free memory.
            diagnostic_event(_LOGGER, 'performance.startup_probe_rank_d_gc_begin',
                rank,
            )
            gc.collect()
            diagnostic_event(_LOGGER, 'performance.startup_probe_rank_d_gc_complete',
                rank,
            )
            torch.cuda.empty_cache()
            torch.cuda.synchronize(runtime.device)
            diagnostic_event(_LOGGER, 'performance.startup_probe_rank_d_allocator_cleanup_complete',
                rank,
            )
            barrier()
            diagnostic_event(_LOGGER, 'performance.startup_probe_rank_d_cleanup_barrier_complete',
                rank,
            )
            post_bootstrap = read_cuda_memory_snapshot(runtime.device)
            runtime_reserve_bytes = (
                _graph_runtime_reserve_bytes(
                    bodies,
                    wave_lanes,
                )
                + graph_reserve_bytes
                + _audit_working_set_bytes(body)
                + transient_reserve_bytes
            )
            runtime_reserve_bytes += _context_staging_growth_bytes(
                physical_capacities,
                (*wave_lanes, *_decode_overlap_lanes(bodies)),
                bootstrap_table_pages,
                model_table_pages,
                mtp=bool(admission.num_speculative_tokens),
            )
            mtp_kv_bytes_per_page = _mtp_kv_bytes_per_page(admission)
            runtime_reserve_bytes += physical_capacities[-1] * mtp_kv_bytes_per_page
            runtime_reserve_bytes += memory_budget.withheld_bytes
            runtime_reserve_bytes += admission.input_encoder_reserves.get(placement.device_index, 0)
            owner_state_capacity = _owner_state_capacity(
                admission.max_running_requests,
                physical_capacities,
                max_speculative_tokens=admission.dynamic_max_speculative_tokens,
                use_prefix_cache=admission.use_prefix_cache,
            )
            # Probe the complete candidate family before cache admission. Graph
            # pools, sampler staging and MTP buffers are nonlinear in row count;
            # extrapolating the smaller family's tensor bytes omits allocations.
            local_physical_max = _largest_fitting_physical_graph(
                snapshot=post_bootstrap,
                layout=body.arena.layout,
                bootstrap_spec=body.arena.spec,
                model_table_pages=model_table_pages,
                auto_max_sequence_tokens=admission.auto_max_sequence_tokens,
                max_running_requests=admission.max_running_requests,
                state_slot_capacity=owner_state_capacity,
                max_speculative_tokens=admission.dynamic_max_speculative_tokens,
                pipeline_parallel_size=admission.parallel_plan.topology.pp_size,
                reserve_bytes=runtime_reserve_bytes,
                auxiliary_bytes_per_page=mtp_kv_bytes_per_page,
            )
            selected_physical_max = torch.tensor(
                (local_physical_max,), dtype=torch.int64, device=runtime.device
            )
            dist.all_reduce(selected_physical_max, op=dist.ReduceOp.MIN)
            physical_max = int(selected_physical_max.item())
            del selected_physical_max
            physical_capacities = tuple(1 << bit for bit in range(physical_max.bit_length()))
            graph_capacities = _serving_graph_capacities(
                physical_capacities,
                admission.max_running_requests,
                pipeline_parallel_size=admission.parallel_plan.topology.pp_size,
                max_speculative_tokens=admission.dynamic_max_speculative_tokens,
            )
            wave_enabled = _uses_wavefront(
                physical_capacities,
                graph_capacities,
                admission.parallel_plan.topology.pp_size,
            )
            capture_order = _graph_capture_order(physical_capacities)
            minimum_final_pages = minimum_kv_page_capacity(
                1 if admission.auto_max_sequence_tokens else model_table_pages,
                graph_capacities[-1],
            )
            capacity_plan, bootstrap_backing = _plan_replacement_cache_capacity(
                snapshot=post_bootstrap,
                layout=body.arena.layout,
                bootstrap_spec=body.arena.spec,
                state_slot_capacity=owner_state_capacity,
                minimum_page_capacity=minimum_final_pages,
                reserve_bytes=runtime_reserve_bytes,
                auxiliary_bytes_per_page=mtp_kv_bytes_per_page,
                scratch_capacity=physical_max,
            )
            diagnostic_event(_LOGGER, 'performance.startup_probe_rank_d_cache_plan_complete',
                rank,
            )
            # A peer must not launch the persistent NCCL reduction while
            # another rank is still inside the CUDA driver memory query above.
            # On PCIe-only topologies that can starve the late rank before it
            # has enqueued the matching collective.
            try:
                audit_barrier.wait(nccl_timeout_seconds)
            except threading.BrokenBarrierError as error:
                raise Qwen3_5TP2PP2ServingError(
                    "rank-local cache planning IPC barrier timed out"
                ) from error
            served_max_sequence_tokens = _effective_max_sequence_tokens(
                admission.max_sequence_tokens,
                capacity_plan.arena_spec.page_capacity,
                graph_capacities[-1],
            )
            context_limit = torch.tensor(
                (served_max_sequence_tokens,),
                dtype=torch.int64,
                device=runtime.device,
            )
            dist.all_reduce(context_limit, op=dist.ReduceOp.MIN)
            diagnostic_event(_LOGGER, 'performance.startup_probe_rank_d_context_reduce_complete',
                rank,
            )
            served_max_sequence_tokens = int(context_limit.item())
            del context_limit
            if admission.auto_max_sequence_tokens or set(bodies) != set(physical_capacities):
                # Keep immutable packed weights and M-RoPE tables, but release
                # every temporary cache, page table, activation, and workspace
                # before allocating the final context-width family.
                mrope_tables = body._mrope_tables
                packed_gate_up_weights = body._packed_gate_up_weights
                packed_gdn_abz = (
                    body._packed_gdn_abz_weights,
                    body._gdn_abz_widths,
                )
                bodies.clear()
                wave_lanes = ()
                del body
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize(runtime.device)
                barrier()
                diagnostic_event(_LOGGER, 'performance.startup_probe_rank_d_final_body_build',
                    rank,
                )
                body, bodies, wave_lanes = _build_body_family(
                    admission,
                    weights,
                    runtime,
                    physical_capacities,
                    max_sequence_tokens=served_max_sequence_tokens,
                    page_capacity=capacity_plan.arena_spec.page_capacity,
                    state_capacity=capacity_plan.arena_spec.state_slot_capacity,
                    scratch_capacity=physical_max,
                    wave_enabled=wave_enabled,
                    mrope_tables=mrope_tables,
                    packed_gate_up_weights=packed_gate_up_weights,
                    packed_gdn_abz=packed_gdn_abz,
                    overlap_plans=overlap_plans,
                )
                diagnostic_event(_LOGGER, 'performance.startup_probe_rank_d_final_body_build_2',
                    rank,
                )
            else:
                body.replace_shared_cache_arena(
                    capacity_plan.arena_spec,
                    (
                        *(bodies[captured_m] for captured_m in physical_capacities[1:]),
                        *wave_lanes,
                        *_decode_overlap_lanes(bodies),
                    ),
                )
            barrier()
            table_pages = (served_max_sequence_tokens + _BLOCK_SIZE - 1) // _BLOCK_SIZE
            hidden_slot_count = body.arena.states[CONV_STATE_ID].shape[0]
            target_hidden_slots = torch.zeros(
                (hidden_slot_count, builtin_text_config(admission.config).hidden_size),
                dtype=torch.bfloat16,
                device=runtime.device,
            )
            physical_executors: dict[int, _CapturedGraphRank] = {}
            constraint_arena = None
            bucket_evidence: dict[int, str] = {}
            target_cost_observations: dict[int, list[float]] = {}
            eager_snapshots: dict[int, dict[int, dict[str, Tensor]]] = {}
            # NCCL communicators that have entered CUDA Graph capture must not
            # return to eager collectives on another stream.  Materialize every
            # eager oracle before the first capture, then compare graph replays
            # against these small boundary-row snapshots.
            for captured_m in capture_order:
                bucket_body = bodies[captured_m]
                bucket_resource = _resource(rank, captured_m)
                snapshots: dict[int, dict[str, Tensor]] = {}
                for logical_m in _bounded_audit_logical_ms(tuple(bucket_body.bucket_logical_ms)):
                    _prepare_audit(
                        bucket_body,
                        bucket_resource,
                        poison=0,
                        logical_m=logical_m,
                    )
                    barrier()
                    bucket_body.launch()
                    torch.cuda.synchronize(runtime.device)
                    barrier()
                    snapshots[logical_m] = _active_snapshot(bucket_body, logical_m)
                eager_snapshots[captured_m] = snapshots
            # Capture largest-first so allocator/capture pools are established by
            # the maximal portable shape and smaller buckets can reuse them.
            target_graph_pool = torch.cuda.graph_pool_handle()
            target_graph_stream = torch.cuda.Stream(device=runtime.device)
            for captured_m in capture_order:
                bucket_body = bodies[captured_m]
                bucket_resource = _resource(rank, captured_m)
                diagnostic_event(_LOGGER, 'performance.capturing_graph_rank_d_c_d',
                    rank,
                    captured_m,
                )
                slot = Qwen3_5TP2PP2GraphSlot.capture(
                    bucket_body,
                    bucket_resource,
                    pool=target_graph_pool,
                    stream=target_graph_stream,
                )
                bucket_evidence[captured_m] = _cold_start_graph_audit(
                    bucket_body,
                    bucket_resource,
                    slot,
                    admission,
                    eager_snapshots[captured_m],
                    target_cost_observations,
                )
                diagnostic_event(_LOGGER, 'performance.graph_cold_audit_ready_rank_d_c',
                    rank,
                    captured_m,
                )
                physical_executors[captured_m] = _CapturedGraphRank(
                    bucket_body,
                    bucket_resource,
                    slot,
                    runtime,
                    admission,
                    target_hidden_slots,
                )
                if constraint_arena is None:
                    constraint_arena = _ConstraintSamplingArena(
                        physical_executors[captured_m], max(graph_capacities),
                    )
                _prepare_rank_greedy_sampling(
                    physical_executors[captured_m], input_lifetime,
                    constraint_arena=constraint_arena,
                )
                qualification_progress(
                    "graph_capture", len(physical_executors), len(physical_capacities),
                )
            assert constraint_arena is not None
            physical_executors = {
                captured_m: physical_executors[captured_m] for captured_m in physical_capacities
            }
            try:
                migration_evidence_id = _cold_start_bucket_migration_audit(
                    physical_executors,
                    admission,
                )
            except BaseException:
                diagnostic_event(_LOGGER, 'performance.bucket_migration_audit_failed_before_nccl_teardown',
                    rank,
                )
                raise
            executors: dict[int, _CapturedGraphRank | _CapturedWavefrontRank] = {
                **physical_executors
            }
            if wave_enabled:
                wave_lane_a, wave_lane_b = wave_lanes
                wave_capacity = 2 * wave_lane_a.captured_m
                wave_resource = _resource(rank, wave_capacity)
                diagnostic_event(_LOGGER, 'performance.capturing_graph_rank_d_c_d',
                    rank,
                    wave_capacity,
                )
                wave_slot = Qwen3_5TP2PP2WavefrontGraphSlot.capture(
                    wave_lane_a,
                    wave_lane_b,
                    wave_resource,
                )
                wave_executor = _CapturedWavefrontRank(
                    wave_lane_a,
                    wave_lane_b,
                    wave_slot,
                    runtime,
                    admission,
                    target_hidden_slots,
                )
                _prepare_rank_greedy_sampling(
                    wave_executor, input_lifetime, constraint_arena=constraint_arena,
                )
                bucket_evidence[wave_capacity] = _cold_start_wavefront_audit(
                    wave_executor,
                    admission,
                    audit_statuses,
                    audit_barrier,
                    nccl_timeout_seconds,
                )
                diagnostic_event(_LOGGER, 'performance.graph_cold_audit_ready_rank_d_c',
                    rank,
                    wave_capacity,
                )
                executors[wave_capacity] = wave_executor
            executors = {captured_m: executors[captured_m] for captured_m in graph_capacities}
            mtp_executors: dict[int, _CapturedMTPRank] = {}
            if admission.num_speculative_tokens:
                request_page_capacity = body.arena.spec.page_capacity
                mtp_bodies = _build_mtp_body_family(
                    admission,
                    weights,
                    runtime,
                    physical_capacities,
                    table_pages=table_pages,
                    request_page_capacity=request_page_capacity,
                )
                for captured_m in capture_order:
                    mtp_body = mtp_bodies[captured_m]
                    diagnostic_event(_LOGGER, 'performance.capturing_mtp_graph_rank_d_c_d',
                        rank,
                        captured_m,
                    )
                    mtp_slot = Qwen3_5TPMTPGraphSlot.capture(mtp_body)
                    mtp_executors[captured_m] = _CapturedMTPRank(
                        mtp_body,
                        mtp_slot,
                        runtime,
                        admission,
                        request_page_capacity=request_page_capacity,
                        target_hidden_slots=target_hidden_slots,
                    )
                    _prepare_rank_greedy_sampling(
                        mtp_executors[captured_m], input_lifetime, constraint_arena=constraint_arena,
                    )
                mtp_executors = {
                    captured_m: mtp_executors[captured_m] for captured_m in physical_capacities
                }
                from cubic_serve.executor.draft_graph import prepare_state_transfers
                prepare_state_transfers(mtp_executors, physical_executors, input_lifetime)
                _qualify_draft_verification(mtp_executors, physical_executors)
                if mtp_feeding_profile is not None:
                    for mtp_executor in mtp_executors.values():
                        mtp_executor.enable_feeding_profile(mtp_feeding_profile)
            cross_row_state_evidence_id = (
                _cold_start_cross_row_state_audit(
                    physical_executors[2],
                    mtp_executors[2],
                    target_hidden_slots,
                    admission,
                )
                if mtp_executors
                else stable_digest(
                    {
                        "schema": "cubic.qwen3_5.tp4.cross-row-state.v1",
                        "rank": rank,
                        "enabled": False,
                    }
                )
            )
            prefix_evidence_id = (
                _cold_start_prefix_cache_audit(
                    physical_executors,
                    mtp_executors,
                    target_hidden_slots,
                    admission,
                )
                if admission.use_prefix_cache
                else None
            )
            if not mtp_executors:
                # Prefix qualification has consumed the hidden-state witness.
                # A target-only server has no draft model that can read this
                # per-request stash, so do not copy every decode row into it.
                for target_executor in physical_executors.values():
                    target_executor._target_hidden_slots = None
            target_cost_samples = (
                _cold_start_target_costs(physical_executors)
                if admission.dynamic_speculation
                else tuple(
                    (capacity, _median_cost_samples(samples))
                    for capacity, samples in sorted(target_cost_observations.items())
                )
            )
            target_depth_samples = (
                _cold_start_target_depth_costs(
                    physical_executors, admission.dynamic_max_speculative_tokens + 1
                )
                if admission.dynamic_speculation else ()
            )
            proposal_cost_samples = (
                _cold_start_mtp_proposal_costs(mtp_executors)
                if admission.dynamic_speculation
                else ()
            )
            constraint_sampling_cost_samples = _warm_sampling(
                physical_executors,
                price_sampling_through=(admission.max_running_requests
                                        if admission.dynamic_speculation else 0),
            )
            if feeding_profile is not None:
                for profiled_executor in executors.values():
                    profiled_executor.enable_feeding_profile(feeding_profile)
            from cubic_serve.executor.resident_window import prepare_resident_compactions

            prepare_resident_compactions(physical_executors, input_lifetime)
            graph_evidence_id = stable_digest(
                {
                    "schema": "cubic.qwen3_5.tp2_pp2.graph-pool.v1",
                    "operator_config": admission.operator_config.digest,
                    "operator_model_id": admission.operator_model_id,
                    "rank": rank,
                    "buckets": tuple(
                        (captured_m, bucket_evidence[captured_m]) for captured_m in graph_capacities
                    ),
                    "greedy_sampling": tuple(
                        (capacity, tuple(plan.evidence for plan in executor._sampling.greedy_plans))
                        for capacity, executor in executors.items()
                    ),
                    "migration_cow": migration_evidence_id,
                    "num_speculative_tokens": admission.num_speculative_tokens,
                    "prefix_cache_fork": prefix_evidence_id,
                }
            )
            torch.cuda.synchronize(runtime.device)
            torch.cuda.empty_cache()
            final_snapshot = read_cuda_memory_snapshot(runtime.device)
            memory_budget.require_headroom(final_snapshot.free_bytes)
            minimum_runtime_free = _audit_working_set_bytes(body) // 2
            if final_snapshot.free_bytes < minimum_runtime_free:
                raise Qwen3_5TP2PP2ServingError(
                    "runtime headroom rejected after graph qualification: "
                    f"free={final_snapshot.free_bytes / (1 << 20):.1f}MiB "
                    f"required={minimum_runtime_free / (1 << 20):.1f}MiB"
                )
            cache_capacity = Qwen3_5TP2PP2RankCacheCapacity(
                rank=rank,
                device=final_snapshot.device,
                page_capacity=capacity_plan.arena_spec.page_capacity,
                state_slot_capacity=capacity_plan.arena_spec.state_slot_capacity,
                scratch_slot_count=capacity_plan.scratch_slot_count,
                bytes_per_page=capacity_plan.bytes_per_page,
                bytes_per_state_bundle=capacity_plan.bytes_per_state_bundle,
                arena_backing_bytes=capacity_plan.backing_bytes,
                bootstrap_arena_backing_bytes=bootstrap_backing,
                reserve_bytes=capacity_plan.reserve_bytes,
                free_after_bootstrap_bytes=post_bootstrap.free_bytes,
                replaceable_free_bytes=capacity_plan.snapshot.free_bytes,
                free_after_final_capture_bytes=final_snapshot.free_bytes,
                total_bytes=final_snapshot.total_bytes,
                vram_page_capacity_estimate=(
                    capacity_plan.arena_spec.page_capacity
                    + max(0, min(final_snapshot.free_bytes, memory_budget.startup.free_bytes)
                          - capacity_plan.reserve_bytes)
                    // (capacity_plan.bytes_per_page + capacity_plan.auxiliary_bytes_per_page)
                ),
            )
            diagnostic_event(_LOGGER, 'performance.cache_ready_rank_d_device_s_pages',
                rank,
                cache_capacity.device,
                cache_capacity.page_capacity,
                cache_capacity.arena_gib,
                cache_capacity.state_slot_capacity,
                cache_capacity.scratch_slot_count,
                cache_capacity.reserve_bytes / (1 << 30),
                cache_capacity.free_after_final_capture_bytes / (1 << 30),
            )
            encoder_thread = start_owned_input_encoder(
                owned_encoder, admission.input_encoder_connection,
                lambda failure: results.send(_Failure(rank, None, failure)))
            input_lifetime.callback(finish_owned_input_encoder, encoder_thread)
            from cubic_serve.runtime.heap_lifetime import frozen_startup_heap

            input_lifetime.enter_context(frozen_startup_heap())
            from cubic_serve.executor.resident_window import (
                resident_compaction_capacities,
                resident_input_program_capacities,
                resident_publication_names,
                resident_segment_program_capacities,
            )
            from cubic_serve.runtime.quantized_qualification import portfolio_resources

            results.send(
                _Ready(
                    rank,
                    executors[_CAPTURED_M].vocab_size,
                    served_max_sequence_tokens,
                    physical_max,
                    graph_evidence_id,
                    cross_row_state_evidence_id,
                    cache_capacity,
                    target_cost_samples,
                    proposal_cost_samples,
                    target_depth_samples,
                    constraint_sampling_cost_samples=constraint_sampling_cost_samples,
                    resident_publications=resident_publication_names(executors),
                    resident_input_program_capacities=resident_input_program_capacities(executors),
                    resident_stochastic_input_program_capacities=(
                        resident_input_program_capacities(executors, stochastic=True)
                    ),
                    resident_compaction_capacities=resident_compaction_capacities(executors),
                    constraint_graphs_qualified=all(
                        item._sampling.masked_plans
                        and all(plan.qualified for plan in item._sampling.masked_plans)
                        for item in (*executors.values(), *mtp_executors.values())
                    ),
                    resident_segment_program_capacities=resident_segment_program_capacities(
                        executors
                    ),
                    replay_resources=portfolio_resources(executors, mtp_executors),
                )
            )
            resident_executor: _CapturedGraphRank | None = None
            resident_request_ids: tuple[str, ...] = ()
            resident_batch_id = -1
            resident_offsets: tuple[int, ...] = ()
            resident_page_counts: tuple[int, ...] = ()
            grammar_pool = None
            while True:
                command_get_started_ns = time.monotonic_ns() if feeding_profile is not None else 0
                notice = commands.recv_bytes()
                if len(notice) != _COMMAND_NOTICE.size:
                    raise Qwen3_5TP2PP2ServingError("worker command notice is invalid")
                (payload_bytes,) = _COMMAND_NOTICE.unpack(notice)
                if not 0 < payload_bytes <= shared_command_capacity:
                    raise Qwen3_5TP2PP2ServingError(
                        "worker command exceeds the shared command arena"
                    )
                command = _deserialize_worker_command(bytes(shared_command_view[:payload_bytes]))
                if type(command) is _CloseCommand:
                    # Captured NCCL operations retain communicators. Release
                    # completed graphs before the runtime destroys its groups.
                    torch.cuda.synchronize(runtime.device)
                    graphs = [
                        executor.slot.graph
                        for executor in (*physical_executors.values(), *executors.values())
                    ]
                    for executor in physical_executors.values():
                        graphs.extend(executor.slot.graphs)
                    for executor in mtp_executors.values():
                        graphs.extend((executor.slot.graph, executor.slot.state_graph))
                    for graph in {id(graph): graph for graph in graphs}.values():
                        graph.reset()
                    return
                if type(command) is _ContinueBatchCommand:
                    request_id = command.request_ids[0] if command.request_ids else None
                    try:
                        if resident_executor is None:
                            raise Qwen3_5TP2PP2ServingError("steady continuation has no resident batch")
                        resident_offsets, resident_page_counts = _run_resident_continuation(
                            command, resident_executor, previous_batch_id=resident_batch_id,
                            request_ids=resident_request_ids, offsets=resident_offsets,
                            page_counts=resident_page_counts, block_size=_BLOCK_SIZE,
                            rank=rank, sampler_rank=sampler_rank,
                            completion_statuses=audit_statuses, completion_barrier=audit_barrier,
                            timeout=nccl_timeout_seconds, interrupt=continuation_interrupt,
                            publish=results.send,
                        )
                        resident_batch_id = command.batch_id
                        resident_request_ids = command.request_ids
                    except BaseException:
                        detail = traceback.format_exc()
                        diagnostic_event(_LOGGER, 'performance.worker_continuation_failed_before_nccl_teardown_rank',
                            rank,
                            request_id,
                            detail,
                        )
                        report_failure()
                        raise
                    request_id = None
                    continue
                assert type(command) is _BatchCommand
                if feeding_profile is not None:
                    feeding_profile.record(
                        "command_queue_get_cpu_ms",
                        (time.monotonic_ns() - command_get_started_ns) / 1_000_000,
                    )
                request_id = command.rows[0].request_id if command.rows else None
                try:
                    if command.captured_m != _bucket_for_logical_m(
                        len(command.rows), graph_capacities
                    ):
                        raise Qwen3_5TP2PP2ServingError(
                            "batch selected a non-canonical graph bucket"
                        )
                    executor = executors.get(command.captured_m)
                    if executor is None:
                        raise Qwen3_5TP2PP2ServingError("batch graph bucket is not resident")
                    round_started_ns = time.monotonic_ns() if mtp_executors else 0
                    if command.constraints or command.released_constraints:
                        grammar_pool = _request_grammar_pool(
                            grammar_pool, command, admission, executor.vocab_size
                        )
                    result = execute_draft_batch(
                        command,
                        executor,
                        mtp_executors,
                        target_hidden_slots,
                        admission,
                        physical_capacities,
                        rank=rank,
                        sampler_rank=sampler_rank,
                        mtp_feeding_profile=mtp_feeding_profile,
                        grammar_pool=grammar_pool,
                    )
                    speculative_segments, _ = _ragged_speculative_plan(command.rows)
                    ownership = (
                        resident_batch_ownership(
                            executor, command.rows, block_size=executor._block_size
                        )
                        if not command.constraints
                        and not speculative_segments
                        and isinstance(executor, _CapturedGraphRank)
                        else None
                    )
                    if mtp_executors:
                        torch.cuda.current_stream(runtime.device).synchronize()
                        audit_barrier.wait(nccl_timeout_seconds)
                        result = replace(
                            result, step_ms=(time.monotonic_ns() - round_started_ns) / 1e6,
                        )
                    if rank == sampler_rank:
                        results.send(result)
                    if ownership is not None:
                        resident_executor = executor
                        resident_request_ids, resident_offsets, resident_page_counts = ownership
                        resident_batch_id = command.batch_id
                    else:
                        resident_executor = None
                        resident_request_ids = ()
                        resident_batch_id = -1
                        resident_offsets = ()
                        resident_page_counts = ()
                except BaseException:
                    detail = traceback.format_exc()
                    diagnostic_event(_LOGGER, 'performance.worker_batch_failed_before_nccl_teardown_rank',
                        rank,
                        request_id,
                        detail,
                    )
                    report_failure()
                    raise
                request_id = None
    except BaseException:
        report_failure()
        raise
    finally:
        if feeding_profile is not None:
            feeding_profile.emit_once()
        if mtp_feeding_profile is not None:
            mtp_feeding_profile.emit_once()


_TokenSink = TokenSink


class _Controller(Protocol):
    vocab_size: int
    max_sequence_tokens: int
    rank_graph_evidence_ids: tuple[str, ...]
    rank_cache_capacities: tuple[Qwen3_5TP2PP2RankCacheCapacity, ...]

    @property
    def kv_vram_capacity_tokens(self) -> int | None: ...

    @property
    def ready(self) -> bool: ...

    def submit(self, command: _GenerateCommand, output: _TokenSink) -> None: ...

    def cancel(self, request_id: str) -> None: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class _ScheduledRequest:
    command: _GenerateCommand
    output: _TokenSink
    admitted_at: float
    prefill_cursor: int = 0
    generated_count: int = 0
    last_token_id: int | None = None
    state_slot: int | None = None
    pages: list[int] = field(default_factory=list)
    mtp_pages: list[int] = field(default_factory=list)
    state_initialized: bool = False
    cancelled: bool = False
    output_closed: bool = False
    stats_terminal: bool = False
    draft_policy: DynamicDraftPolicy | None = None
    draft_tokens: int | None = None
    draft_stopped: bool = False
    pending_cow_pages: tuple[int, ...] = ()

    @property
    def decode_ready(self) -> bool:
        return self.prefill_cursor == len(self.command.input_ids)


def _effective_request_draft_tokens(
    record: _ScheduledRequest,
    *,
    default: int,
    maximum: int,
) -> int:
    """Resolve this round's request-owned K and clip only at its output boundary."""

    requested = default if record.draft_tokens is None else record.draft_tokens
    if type(requested) is not int or not 0 <= requested <= maximum:
        raise Qwen3_5TP2PP2ServingError("request draft depth exceeds provider capacity")
    if record.draft_stopped:
        if requested:
            raise Qwen3_5TP2PP2ServingError("K0 request cannot resume drafting")
        return 0
    if requested == 0 and default > 0:
        record.draft_stopped = True
        return 0
    remaining = record.command.max_tokens - record.generated_count
    return min(requested, max(0, remaining - 1))


def _advance_dynamic_draft(
    record: _ScheduledRequest,
    *,
    accepted: int,
    proposed: int,
    terminal: bool,
) -> None:
    """Commit one request-local observation and select its next K."""

    policy = record.draft_policy
    if policy is None or terminal:
        return
    if proposed != policy.current_tokens:
        return
    policy.observe(accepted=accepted, proposed=proposed)


@dataclass(frozen=True, slots=True)
class _PrefixSnapshot:
    token_ids: tuple[int, ...]
    state_slot: int
    pages: tuple[int, ...]
    mtp_pages: tuple[int, ...]


@dataclass(slots=True)
class _ServingStats:
    """Exact event-time counters for admitted streaming generation work.

    Rates use separate completed dispatch-phase durations (command publication
    through worker result), never request-active wall time. Prefill's first
    output is not a decode token. Fused mixed batches have a shared duration:
    retain separate internal accounting rather than invent a prefill/decode time split.
    Request-active time only controls reporting cadence and request accounting.
    """

    active_requests: int = 0
    prefill_tokens_total: int = 0
    completion_tokens_total: int = 0
    mtp_accepted_total: int = 0
    mtp_proposed_total: int = 0
    mtp_rounds_total: int = 0
    prefix_hit_tokens_total: int = 0
    prefix_query_tokens_total: int = 0
    active_seconds_closed: float = 0.0
    active_started_at: float | None = None
    last_report_at: float | None = None
    last_report_active_seconds: float = 0.0
    last_report_prefill_tokens: int = 0
    last_report_tokens: int = 0
    last_report_mtp_accepted: int = 0
    last_report_mtp_proposed: int = 0
    last_report_mtp_rounds: int = 0
    last_report_prefix_hit_tokens: int = 0
    last_report_prefix_query_tokens: int = 0
    interval_active_peak: int = 0

    def admit(self, now: float) -> None:
        if self.active_requests == 0:
            self.active_started_at = now
            self.last_report_at = now
        self.active_requests += 1
        self.interval_active_peak = max(self.interval_active_peak, self.active_requests)

    def publish(self, count: int, *, is_prefill: bool = False) -> None:
        if count <= 0 or self.active_requests <= 0:
            raise Qwen3_5TP2PP2ServingError(
                "serving token accounting requires active positive output"
            )
        # Count visible output, including mixed steps and the first sampled
        # prefill token. Execution phases never remove published work.
        self.completion_tokens_total += count

    def step(self, prefill_tokens: int, decode_rows: int, seconds: float) -> None:
        if (prefill_tokens < 0 or decode_rows < 0 or not math.isfinite(seconds)
                or seconds < 0 or self.active_requests <= 0):
            raise Qwen3_5TP2PP2ServingError("invalid phase accounting")
        self.prefill_tokens_total += prefill_tokens

    def prefix_lookup(self, hit_tokens: int, query_tokens: int) -> None:
        if not 0 <= hit_tokens <= query_tokens:
            raise Qwen3_5TP2PP2ServingError("invalid prefix cache accounting")
        self.prefix_hit_tokens_total += hit_tokens
        self.prefix_query_tokens_total += query_tokens

    def maybe_report(
        self,
        now: float,
        *,
        kv_usage_percent: float = 0.0,
    ) -> None:
        if (
            self.last_report_at is not None
            and now - self.last_report_at >= _SERVING_STATS_INTERVAL_SECONDS
        ):
            self._emit(now, kv_usage_percent)

    def speculate(self, accepted: int, proposed: int, rounds: int) -> None:
        if not 0 <= accepted <= proposed or not 0 < rounds <= proposed:
            raise Qwen3_5TP2PP2ServingError("invalid MTP acceptance accounting")
        self.mtp_accepted_total += accepted
        self.mtp_proposed_total += proposed
        self.mtp_rounds_total += rounds

    def terminal(
        self,
        now: float,
        *,
        kv_usage_percent: float = 0.0,
    ) -> None:
        if self.active_requests <= 0 or self.active_started_at is None:
            raise Qwen3_5TP2PP2ServingError("serving request accounting has no active owner")
        self.active_requests -= 1
        if self.active_requests == 0:
            self.active_seconds_closed += now - self.active_started_at
            self.active_started_at = None
        if self.active_requests == 0:
            self._emit(now, kv_usage_percent)

    def _active_seconds(self, now: float) -> float:
        seconds = self.active_seconds_closed
        if self.active_started_at is not None:
            seconds += now - self.active_started_at
        return seconds

    def _emit(
        self,
        now: float,
        kv_usage_percent: float,
    ) -> None:
        active_seconds = self._active_seconds(now)
        interval_seconds = active_seconds - self.last_report_active_seconds
        interval_prefill_tokens = self.prefill_tokens_total - self.last_report_prefill_tokens
        interval_tokens = self.completion_tokens_total - self.last_report_tokens
        interval_mtp_accepted = self.mtp_accepted_total - self.last_report_mtp_accepted
        interval_mtp_proposed = self.mtp_proposed_total - self.last_report_mtp_proposed
        interval_mtp_rounds = self.mtp_rounds_total - self.last_report_mtp_rounds
        interval_prefix_hits = self.prefix_hit_tokens_total - self.last_report_prefix_hit_tokens
        interval_prefix_queries = (
            self.prefix_query_tokens_total - self.last_report_prefix_query_tokens
        )
        if not (interval_prefill_tokens or interval_tokens or interval_mtp_proposed):
            self._mark_reported(now, active_seconds)
            return
        serving_status(
            _LOGGER, self.interval_active_peak, kv_usage_percent,
            interval_prefix_hits, interval_prefix_queries, interval_prefill_tokens, interval_seconds,
            interval_tokens, interval_mtp_accepted, interval_mtp_proposed, interval_mtp_rounds,
        )
        self._mark_reported(now, active_seconds)

    def _mark_reported(self, now: float, active_seconds: float) -> None:
        self.last_report_at = now
        self.last_report_active_seconds = active_seconds
        self.last_report_prefill_tokens = self.prefill_tokens_total
        self.last_report_tokens = self.completion_tokens_total
        self.last_report_mtp_accepted = self.mtp_accepted_total
        self.last_report_mtp_proposed = self.mtp_proposed_total
        self.last_report_mtp_rounds = self.mtp_rounds_total
        self.last_report_prefix_hit_tokens = self.prefix_hit_tokens_total
        self.last_report_prefix_query_tokens = self.prefix_query_tokens_total
        self.interval_active_peak = self.active_requests


class _StartupProgress:
    def __init__(self, world_size: int = 4) -> None:
        self._world_size = world_size
        self._by_section: dict[str, dict[int, tuple[int, int]]] = {}
        self._units: dict[tuple[str, int], str] = {}
        self._rendered = False
        self._rendered_lines = 0
        self._dirty = False
        self._last_render_at = 0.0

    def update(self, item: _LoadProgress) -> bool:
        qualification = item.section in {"graph_measurement", "graph_capture", "numerical_audit",
                                         "mtp_composition", "target_numerical_audit",
                                         "provider_numerical_audit"}
        if (
            not ((item.section in {"base", "mtp"} and item.unit in {"bytes", "tensors"})
                 or (item.section in {"graph_measurement", "graph_capture"}
                     and item.unit == "buckets")
                 or (item.section in {"numerical_audit", "mtp_composition",
                                      "target_numerical_audit", "provider_numerical_audit"}
                     and item.unit == "witnesses"))
            or type(item.rank) is not int or item.rank not in range(self._world_size)
            or type(item.completed) is not int or type(item.total) is not int
            or item.total <= 0
            or not 0 <= item.completed <= item.total
        ):
            raise Qwen3_5TP2PP2ServingError("rank reported invalid startup progress")
        by_rank = self._by_section.setdefault(item.section, {})
        previous = by_rank.get(item.rank)
        if previous is not None and (item.total != previous[1] or item.completed < previous[0]):
            raise Qwen3_5TP2PP2ServingError("rank startup progress regressed")
        advanced = item.completed > 0 and (previous is None or item.completed > previous[0])
        by_rank[item.rank] = (item.completed, item.total)
        key = item.section, item.rank
        if key in self._units and self._units[key] != item.unit:
            raise Qwen3_5TP2PP2ServingError("weight progress unit changed during loading")
        self._units[key] = item.unit
        if qualification:
            if advanced:
                startup_progress(item.rank, item.section, item.completed, item.total,
                                 capacity=item.capacity)
            return advanced
        self._dirty = True
        base = self._by_section.get("base", {})
        if len(base) != self._world_size:
            return advanced
        now = time.monotonic()
        all_complete = all(
            len(section) == self._world_size
            and all(completed == total for completed, total in section.values())
            for name, section in self._by_section.items() if name in {"base", "mtp"}
        )
        if self._rendered and not all_complete and now - self._last_render_at < 0.1:
            return advanced
        self._render(now)
        return advanced

    def _render(self, now: float) -> None:
        if self._rendered:
            sys.stderr.write(f"\x1b[{self._rendered_lines}A")
        self._rendered_lines = 0
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        for section in ("base", "mtp"):
            if section not in self._by_section:
                continue
            label = "main_model" if section == "base" else "mtp"
            by_rank = self._by_section.get(section, {})
            for rank in range(self._world_size):
                sys.stderr.write("\r\x1b[2K")
                state = by_rank.get(rank)
                if state is None:
                    sys.stderr.write(
                        f"{timestamp} | INFO | weights/{label} rank={rank} [waiting]"
                    )
                else:
                    completed, total = state
                    bar = format_weight_progress(completed, total, self._units[(section, rank)])
                    sys.stderr.write(
                        f"{timestamp} | INFO | weights/{label} rank={rank} {bar}"
                    )
                sys.stderr.write("\n")
                self._rendered_lines += 1
        sys.stderr.flush()
        self._rendered = True
        self._dirty = False
        self._last_render_at = now

    def finish(self) -> None:
        if self._rendered and self._dirty:
            self._render(time.monotonic())
        elif not self._rendered:
            sys.stderr.write("\n")
            sys.stderr.flush()


class _ControllerAdmission(Protocol):
    @property
    def block_size_tokens(self) -> int: ...

    @property
    def config(self) -> Qwen3_5Config | Qwen3_5MoeConfig: ...

    @property
    def parallel_plan(self) -> Any: ...

    @property
    def max_sequence_tokens(self) -> int: ...

    @property
    def auto_max_sequence_tokens(self) -> bool: ...

    @property
    def max_running_requests(self) -> int: ...

    @property
    def num_speculative_tokens(self) -> int: ...

    @property
    def max_speculative_tokens(self) -> int: ...

    @property
    def dynamic_max_speculative_tokens(self) -> int: ...

    @property
    def dynamic_speculation(self) -> bool: ...

    @property
    def use_prefix_cache(self) -> bool: ...

    @property
    def prefill_chunk_tokens(self) -> int | None: ...

    @property
    def draft_policy(self) -> DraftPolicyConfig: ...

def _worker_from_metadata(
    worker_target: Any, metadata_path: str, rank: int, init_method: str,
    commands: Any, shared_commands: Any, shared_capacity: int, results: Any,
    timeout: int, audit_statuses: Any, audit_barrier: Any, encoder_connection: Any = None,
    continuation_interrupt: Any = None,
) -> None:
    """Keep spawn pipes small so worker imports can run concurrently.

    The controller owns this private, locally written metadata file for the
    entire worker lifetime. It is never a user-supplied pickle or a weight file.
    Live pipes/shared-memory handles still use multiprocessing's spawn protocol.
    """
    if not faulthandler.is_enabled():
        faulthandler.enable(file=sys.__stderr__, all_threads=True)
    configure_logging()
    diagnostic_event(_LOGGER, 'performance.worker_bootstrap_entered_rank_d',
        rank,
    )
    admission = None
    try:
        started = time.perf_counter()
        with open(metadata_path, "rb") as stream:
            admission = pickle.load(stream)
        if encoder_connection is not None:
            if isinstance(admission, _Admission):
                admission = replace(admission, input_encoder_connection=encoder_connection)
            else:
                admission.input_encoder_connection = encoder_connection
        diagnostic_event(_LOGGER, 'performance.worker_metadata_restored_rank_d_bytes_d',
            rank,
            Path(metadata_path).stat().st_size,
            time.perf_counter() - started,
        )
    except BaseException:
        source = getattr(admission, "model_definition_source", None)
        if source is not None:
            source.close()
        results.send(_Failure(rank, None, traceback.format_exc()))
        return
    try:
        worker_target(
            rank, init_method, admission, commands, shared_commands, shared_capacity,
            results, timeout, audit_statuses, audit_barrier,
            **({"continuation_interrupt": continuation_interrupt}
               if continuation_interrupt is not None else {}),
        )
    finally:
        source = getattr(admission, "model_definition_source", None)
        if source is not None:
            source.close()


class _ProcessController:
    _block_size = _BLOCK_SIZE
    _prefix_block_packer = _PREFIX_BLOCK_PACKER

    @property
    def kv_vram_capacity_tokens(self) -> int | None:
        """Startup estimate from the tightest rank, never summed across TP."""
        pages = tuple(item.vram_page_capacity_estimate for item in self.rank_cache_capacities)
        if not pages or any(value is None for value in pages):
            return None
        return min(value for value in pages if value is not None) * self._block_size

    def __init__(
        self,
        admission: _ControllerAdmission,
        *,
        startup_timeout_seconds: float,
        worker_timeout_seconds: float = _WORKER_RESPONSE_TIMEOUT_SECONDS,
        worker_target: Any = None,
        graph_capacities: tuple[int, ...] | None = None,
        encoder_group: Any = None,
        decode_window_seconds: float | None = None,
    ) -> None:
        if (not isinstance(worker_timeout_seconds, (int, float))
                or isinstance(worker_timeout_seconds, bool)
                or not math.isfinite(worker_timeout_seconds) or worker_timeout_seconds <= 0):
            raise ValueError("worker response timeout must be finite and positive")
        if decode_window_seconds is not None and (
            not math.isfinite(decode_window_seconds) or decode_window_seconds < 0
        ):
            raise ValueError("decode window time budget must be finite and nonnegative")
        self._decode_window_seconds = decode_window_seconds
        self._resident_step_seconds = 0.0
        world_size = len(admission.parallel_plan.topology.placements)
        self._block_size = getattr(admission, "block_size_tokens", _BLOCK_SIZE)
        if type(self._block_size) is not int or self._block_size <= 0:
            raise ValueError("serving page size must be a positive integer")
        self._prefix_block_packer = struct.Struct(f"<{self._block_size}I")
        if graph_capacities is not None and (
            not graph_capacities or graph_capacities[0] != 1
            or graph_capacities != tuple(1 << i for i in range(len(graph_capacities)))
        ):
            raise ValueError("explicit graph capacities must contain every power-of-two anchor")
        if not any(name in os.environ for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")):
            torch.set_num_threads(4)
        self._feeding_profile = _new_step_feeding_profile("controller")
        self._sampler_rank = _sampler_rank(admission.parallel_plan)
        # A missing worker result is a transport/execution failure. This
        # budget restarts for every validated result, not for each request.
        self._worker_timeout = worker_timeout_seconds
        self._max_running_requests = admission.max_running_requests
        self._num_speculative_tokens = admission.num_speculative_tokens
        self._max_speculative_tokens = admission.max_speculative_tokens
        self._dynamic_speculation = admission.dynamic_speculation
        self._draft_policy = admission.draft_policy
        self._dynamic_max_speculative_tokens = admission.dynamic_max_speculative_tokens
        self._physical_graph_capacities = _physical_graph_capacities(
            admission.dynamic_max_speculative_tokens,
            pipeline_parallel_size=admission.parallel_plan.topology.pp_size,
            max_running_requests=admission.max_running_requests,
            include_speculative=True,
        )
        self._graph_capacities = _serving_graph_capacities(
            self._physical_graph_capacities,
            admission.max_running_requests,
            pipeline_parallel_size=admission.parallel_plan.topology.pp_size,
            max_speculative_tokens=admission.dynamic_max_speculative_tokens,
        )
        if graph_capacities is not None:
            self._physical_graph_capacities = graph_capacities
            self._graph_capacities = graph_capacities
        self._draft_planner = (
            DynamicDraftPlanner(self._physical_graph_capacities, config=self._draft_policy)
            if admission.dynamic_speculation
            else None
        )
        self._draft_trace = os.environ.get(_DRAFT_TRACE_ENV, "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._prefix_cache_enabled = admission.use_prefix_cache
        self._prefill_chunk_tokens = admission.prefill_chunk_tokens
        self._closed = False
        self._poisoned = False
        self._close_lock = threading.Lock()
        self._command_send_lock = threading.Lock()
        self._condition = threading.Condition()
        self._requests: dict[str, _ScheduledRequest] = {}
        self._request_order: list[str] = []
        self._serving_stats = _ServingStats()
        self._next_request_index = 0
        self._batch_id = 0
        self._resident_steady: tuple[_ScheduledRequest, ...] = ()
        self._resident_steady_batch_id = -1
        self._resident_steady_captured_m = 0
        self._resident_features_bound = False
        self._resident_owner_lengths: tuple[int, ...] = ()
        self._directory = tempfile.TemporaryDirectory(prefix="cubic-qwen-serving-")
        init_method = (Path(self._directory.name) / "nccl-store").resolve().as_uri()
        context = multiprocessing.get_context("spawn")
        encoder_pipes = ({index: context.Pipe() for index in encoder_group.devices}
                         if encoder_group is not None else {})
        self.input_encoder_connections = tuple(pair[0] for pair in encoder_pipes.values())
        result_pipes = tuple(context.Pipe(duplex=False) for _ in range(world_size))
        self._results = tuple(receiver for receiver, _ in result_pipes)
        result_senders = tuple(sender for _, sender in result_pipes)
        self._audit_statuses = context.Array("i", 2 * world_size, lock=True)
        self._audit_barrier = context.Barrier(world_size)
        self._continuation_interrupt = SharedPublicationInterrupt(context.Event())
        self._input_encoder_execution_lock = (
            InputEncoderExecutionLease(self._continuation_interrupt)
            if encoder_group is not None else None
        )
        if encoder_group is not None:
            encoder_group.execution_lease = self._input_encoder_execution_lock
        command_pipes = tuple(context.Pipe(duplex=False) for _ in range(world_size))
        command_receivers = tuple(receiver for receiver, _ in command_pipes)
        self._commands = tuple(sender for _, sender in command_pipes)
        table_pages = (admission.max_sequence_tokens + self._block_size - 1) // self._block_size
        table_copies = 1 + int(bool(admission.num_speculative_tokens))
        largest_batch = self._graph_capacities[-1]
        self._shared_command_capacity = (
            4 * 1024 * 1024
            + largest_batch * table_pages * table_copies * 8
            + largest_batch * builtin_text_config(admission.config).hidden_size * 2
        )
        self._shared_commands = context.RawArray("B", self._shared_command_capacity)
        self._shared_command_view = memoryview(self._shared_commands).cast("B")
        nccl_timeout = max(1, int(startup_timeout_seconds))
        metadata_path = str(Path(self._directory.name) / "admission.pkl")
        metadata_started = time.perf_counter()
        definition_source = getattr(admission, "model_definition_source", None)
        with open(metadata_path, "wb") as stream:
            pickle.dump(
                (admission if definition_source is None
                 else definition_source.bind_metadata(admission)),
                stream, protocol=pickle.HIGHEST_PROTOCOL,
            )
        diagnostic_event(_LOGGER, 'performance.worker_metadata_prepared_bytes_d_duration_s',
            Path(metadata_path).stat().st_size,
            time.perf_counter() - metadata_started,
        )
        self._processes = tuple(
            context.Process(
                target=_worker_from_metadata,
                args=(
                    _worker if worker_target is None else worker_target,
                    metadata_path,
                    rank,
                    init_method,
                    command_receivers[rank],
                    self._shared_commands,
                    self._shared_command_capacity,
                    result_senders[rank],
                    nccl_timeout,
                    self._audit_statuses,
                    self._audit_barrier,
                    (encoder_pipes[admission.parallel_plan.topology.placements[rank].device_index][1]
                     if admission.parallel_plan.topology.placements[rank].device_index in encoder_pipes
                     else None),
                    self._continuation_interrupt,
                ),
                name=(
                    f"cubic-qwen-tp{admission.parallel_plan.topology.tp_size}"
                    f"pp{admission.parallel_plan.topology.pp_size}-rank{rank}"
                ),
            )
            for rank in range(world_size)
        )
        self._startup_progress = _StartupProgress(world_size)
        try:
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
            try:
                spawn_started = time.perf_counter()
                for process in self._processes:
                    process.start()
                diagnostic_event(_LOGGER, 'performance.worker_processes_spawned_count_d_duration_s',
                    world_size,
                    time.perf_counter() - spawn_started,
                )
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                for receiver in command_receivers:
                    receiver.close()
                for sender in result_senders:
                    sender.close()
                for _, child in encoder_pipes.values():
                    child.close()
            ready = self._receive_ready(startup_timeout_seconds)
            self.input_encoder_admissions = {
                item.input_encoder_admission[0]: item.input_encoder_admission[1:]
                for item in ready if item.input_encoder_admission is not None
            }
            self._continuation_interrupt.attach(tuple(
                name for item in ready for name in item.resident_publications
            ))
            self._resident_input_program_capacities = tuple(sorted(set.intersection(
                *(set(item.resident_input_program_capacities) for item in ready)
            )))
            self._resident_stochastic_input_program_capacities = tuple(sorted(set.intersection(
                *(set(item.resident_stochastic_input_program_capacities) for item in ready)
            )))
            self._resident_compaction_capacities = tuple(sorted(set.intersection(
                *(set(item.resident_compaction_capacities) for item in ready)
            )))
            self._resident_segment_program_capacities = tuple(sorted(set.intersection(
                *(set(item.resident_segment_program_capacities) for item in ready)
            )))
            self._resident_draft_capacities = tuple(sorted(set.intersection(
                *(set(item.resident_draft_capacities) for item in ready)
            )))
            self._grammar_draft_continuation_capacities = tuple(sorted(set.intersection(
                *(set(item.grammar_draft_continuation_capacities) for item in ready)
            )))
            self._grammar_continuation_capacities = tuple(sorted(set.intersection(
                *(set(item.grammar_continuation_capacities) for item in ready)
            )))
            vocab = {item.vocab_size for item in ready}
            capacities = {item.max_sequence_tokens for item in ready}
            physical_maxima = {item.max_physical_graph_capacity for item in ready}
            if (
                len(vocab) != 1
                or not capacities
                or (not admission.auto_max_sequence_tokens and len(capacities) != 1)
                or len(physical_maxima) != 1
            ):
                raise Qwen3_5TP2PP2ServingError("rank startup metadata differs")
            self.vocab_size = vocab.pop()
            self.constraint_graphs_qualified = all(
                item.constraint_graphs_qualified for item in ready
            )
            self.max_sequence_tokens = min(capacities)
            physical_max = physical_maxima.pop()
            self._physical_graph_capacities = tuple(
                1 << bit for bit in range(physical_max.bit_length())
            )
            self._graph_capacities = _serving_graph_capacities(
                self._physical_graph_capacities,
                admission.max_running_requests,
                pipeline_parallel_size=admission.parallel_plan.topology.pp_size,
                max_speculative_tokens=admission.dynamic_max_speculative_tokens,
            )
            if graph_capacities is not None:
                self._graph_capacities = self._physical_graph_capacities
            self._draft_planner = (
                DynamicDraftPlanner(self._physical_graph_capacities, config=self._draft_policy)
                if admission.dynamic_speculation
                else None
            )
            by_rank = {item.rank: item.graph_evidence_id for item in ready}
            if tuple(sorted(by_rank)) != tuple(range(world_size)):
                raise Qwen3_5TP2PP2ServingError("rank startup evidence is incomplete")
            self.rank_graph_evidence_ids = cast(
                tuple[str, ...], tuple(by_rank[rank] for rank in range(world_size))
            )
            state_by_rank = {item.rank: item.cross_row_state_evidence_id for item in ready}
            self.rank_cross_row_state_evidence_ids = cast(
                tuple[str, ...],
                tuple(state_by_rank[rank] for rank in range(world_size)),
            )
            # A portfolio may share cache/scratch across buckets. Lease the
            # entire rank portfolio as one indivisible topology-wide set.
            # Existing admission providers without this resource descriptor
            # retain the dispatcher's existing single-in-flight ownership.
            self._replay_leases = None
            resources = tuple(item.replay_resources for item in sorted(ready, key=lambda x: x.rank))
            if any(resource is not None for resource in resources):
                if any(resource is None for resource in resources):
                    raise Qwen3_5TP2PP2ServingError("incomplete topology replay resources")
                from cubic_serve.executor.replay_lease import ReplayLeasePool

                self._replay_leases = ReplayLeasePool(
                    (cast(tuple[RankReplayResources, ...], resources),)
                )
                self._replay_execution_id = stable_digest(self.rank_graph_evidence_ids)
                self._replay_topology_id = admission.parallel_plan.topology.topology_id
            if self._draft_planner is not None:
                target_by_rank = {item.rank: dict(item.target_cost_samples) for item in ready}
                proposal_by_rank = {item.rank: dict(item.proposal_cost_samples) for item in ready}
                target_samples: list[tuple[int, float]] = []
                proposal_samples: list[tuple[int, float]] = []
                for capacity in self._physical_graph_capacities:
                    target_values = tuple(
                        target_by_rank[rank].get(capacity) for rank in range(world_size)
                    )
                    proposal_values = tuple(
                        proposal_by_rank[rank].get(capacity) for rank in range(world_size)
                    )
                    if any(value is None for value in (*target_values, *proposal_values)):
                        raise Qwen3_5TP2PP2ServingError(
                            "dynamic draft startup cost profile is incomplete"
                        )
                    target_samples.append((capacity, max(cast(tuple[float, ...], target_values))))
                    proposal_samples.append(
                        (capacity, max(cast(tuple[float, ...], proposal_values)))
                    )
                depth_by_rank = {
                    item.rank: {(capacity, depth): ms for capacity, depth, ms
                                in item.target_depth_samples}
                    for item in ready
                }
                depth_samples: list[tuple[int, int, float]] = []
                for capacity in self._physical_graph_capacities:
                    maximum_depth = min(capacity, admission.dynamic_max_speculative_tokens + 1)
                    for depth in range(2, maximum_depth + 1):
                        values = tuple(depth_by_rank[rank].get((capacity, depth))
                                       for rank in range(world_size))
                        if any(value is None for value in values):
                            raise Qwen3_5TP2PP2ServingError(
                                "dynamic draft target segment profile is incomplete"
                            )
                        depth_samples.append(
                            (capacity, depth, max(cast(tuple[float, ...], values)))
                        )
                compound_by_rank = {
                    item.rank: {(capacity, depth, greedy): milliseconds
                                for capacity, depth, greedy, milliseconds
                                in item.compound_cost_samples}
                    for item in ready
                }
                compound_shapes = set(compound_by_rank[0])
                if any(set(values) != compound_shapes for values in compound_by_rank.values()):
                    raise Qwen3_5TP2PP2ServingError("draft round cold profile differs across ranks")
                self._cold_compound_samples = {
                    greedy: tuple((capacity, depth, max(
                        compound_by_rank[rank][capacity, depth, greedy]
                        for rank in range(world_size)))
                        for capacity, depth, mode in sorted(compound_shapes) if mode == greedy)
                    for greedy in (False, True)
                }
                constraint_by_rank = {
                    item.rank: {(rows, depth, greedy, processing): milliseconds
                                for rows, depth, greedy, processing, milliseconds
                                in item.constraint_compound_cost_samples}
                    for item in ready
                }
                constraint_shapes = set(constraint_by_rank[0])
                if any(set(values) != constraint_shapes for values in constraint_by_rank.values()):
                    raise Qwen3_5TP2PP2ServingError(
                        "constraint round cold profile differs across ranks")
                self._cold_constraint_compound_samples = {
                    (processing, greedy): tuple((rows, depth, max(
                        constraint_by_rank[rank][rows, depth, greedy, processing]
                        for rank in range(world_size)))
                        for rows, depth, mode, kind in sorted(constraint_shapes)
                        if mode == greedy and kind == processing)
                    for processing in (1, 2) for greedy in (False, True)
                }
                sampling_by_rank = {
                    item.rank: {(rows, greedy): milliseconds
                                for rows, greedy, milliseconds in item.constraint_sampling_cost_samples
                                if rows in self._physical_graph_capacities}
                    for item in ready
                }
                sampling_shapes = set(sampling_by_rank[0])
                if any(set(values) != sampling_shapes for values in sampling_by_rank.values()):
                    raise Qwen3_5TP2PP2ServingError("masked sampling profile differs across ranks")
                self._cold_constraint_sampling_samples = {
                    greedy: tuple((rows, max(sampling_by_rank[rank][rows, mode]
                                            for rank in range(world_size)))
                                  for rows, mode in sorted(sampling_shapes) if mode == greedy)
                    for greedy in (False, True)
                }
                self._draft_planner.seed_costs(
                    target_samples=target_samples,
                    proposal_samples=proposal_samples,
                    target_depth_samples=depth_samples,
                    compound_samples=self._cold_compound_samples[False],
                )
            cache_by_rank = {item.rank: item.cache_capacity for item in ready}
            if tuple(sorted(cache_by_rank)) != tuple(range(world_size)):
                raise Qwen3_5TP2PP2ServingError("rank cache capacity metadata is incomplete")
            self.rank_cache_capacities = tuple(cache_by_rank[rank] for rank in range(world_size))
            state_capacity = min(item.state_slot_capacity for item in self.rank_cache_capacities)
            page_capacity = min(item.page_capacity for item in self.rank_cache_capacities)
            table_pages = (self.max_sequence_tokens + self._block_size - 1) // self._block_size
            shared = [item.shared_pool_bytes is not None for item in self.rank_cache_capacities]
            if any(shared) and not all(shared):
                raise Qwen3_5TP2PP2ServingError("rank cache ownership modes differ")
            if ((not all(shared) and state_capacity < self._max_running_requests)
                    or state_capacity < 1
                    or page_capacity < (table_pages if all(shared) else
                        minimum_kv_page_capacity(table_pages,self._graph_capacities[-1]))):
                raise Qwen3_5TP2PP2ServingError("rank cache cannot cover one context and its transaction")
            self._state_slot_capacity=state_capacity
            self._state_scratch_capacity=min(item.scratch_slot_count for item in self.rank_cache_capacities)
            self._page_capacity=page_capacity
            self._shared_cache_pool=None
            self._checkpoint_bank=[]
            self._checkpoint_bank_set=set()
            self._checkpoint_bank_used=0
            if all(shared):
                from cubic_serve.runtime.cache.shared_pool import SharedCachePool,SharedCacheFreeSlots
                geometry=tuple((item.shared_pool_bytes,item.bytes_per_page,item.bytes_per_state_bundle)
                               for item in self.rank_cache_capacities)
                budget,page,state=geometry[0]
                self._shared_cache_pool=SharedCachePool(
                    pool_id="shared-cache",budget_bytes=budget,page_bytes=page,state_bytes=state,
                    state_capacity=state_capacity,rank_geometries=geometry[1:])
                if not self._shared_cache_pool.cold_owner_capacity(
                        state_slots=1+self._max_speculative_tokens+int(self._prefix_cache_enabled),
                        pages=table_pages):
                    raise Qwen3_5TP2PP2ServingError(
                        "rank cache cannot cover one context and its transaction")
                self._state_scratch_capacity=min(self._state_scratch_capacity,state_capacity-1)
                self._free_state_slots=SharedCacheFreeSlots(self._shared_cache_pool,"state")
                self._free_pages=SharedCacheFreeSlots(self._shared_cache_pool,"page")
                self._free_mtp_pages=SharedCacheFreeSlots(self._shared_cache_pool,"page")
            else:
                self._free_state_slots=list(range(state_capacity))
                self._free_pages=list(range(page_capacity))
                self._free_mtp_pages=list(range(page_capacity))
            self._shared_state_slots: dict[int, int] = {}
            self._shared_pages: dict[int, int] = {}
            self._shared_mtp_pages: dict[int, int] = {}
            self._prefix_cache: OrderedDict[tuple[int, bytes], _PrefixSnapshot] = OrderedDict()
            if self._shared_cache_pool is None:
                heapq.heapify(self._free_state_slots)
                heapq.heapify(self._free_pages)
                heapq.heapify(self._free_mtp_pages)
            self._dispatcher = threading.Thread(
                target=self._dispatch,
                name="cubic-continuous-batch-dispatch",
                daemon=True,
            )
            from cubic_serve.runtime.heap_lifetime import frozen_startup_heap

            self._heap_lifetime = ExitStack()
            self._heap_lifetime.enter_context(frozen_startup_heap())
            self._dispatcher.start()
        except BaseException:
            self.close()
            raise

    def _startup_worker_failure(self, message: str) -> Qwen3_5TP2PP2ServingError:
        """Read worker status before controller cleanup can terminate peers."""
        states = []
        signals = set()
        for rank, process in enumerate(self._processes):
            code = process.exitcode
            state = f"rank={rank} pid={getattr(process, 'pid', None)} exitcode={code}"
            if code is not None and code < 0:
                try:
                    name = signal.Signals(-code).name
                except ValueError:
                    name = f"signal {-code}"
                signals.add(name)
                state += f" ({name})"
            states.append(state)
        detail = message + "; " + "; ".join(states)
        if 'SIGKILL' in signals:
            detail += ". Check the kernel log for an OOM kill or external termination."
        if signals & {'SIGSEGV', 'SIGABRT', 'SIGBUS'}:
            detail += (". A native worker crashed; see its fault traceback above. "
                       "If absent, relaunch with PYTHONFAULTHANDLER=1.")
        return Qwen3_5TP2PP2ServingError(detail)

    def _receive_ready(self, timeout_seconds: float) -> list[_Ready]:
        pending = {channel: rank for rank, channel in enumerate(self._results)}
        # Each rank owns an inactivity lease renewed only by validated useful
        # progress. An audit-runtime target is not a total startup deadline.
        # Peers and duplicate notifications cannot renew a stalled rank.
        now = time.monotonic()
        deadlines = dict.fromkeys(pending.values(), now + timeout_seconds)
        received: list[_Ready] = []
        progress = self._startup_progress
        try:
            while pending:
                now = time.monotonic()
                remaining = min(deadlines[rank] for rank in pending.values()) - now
                if remaining <= 0:
                    raise Qwen3_5TP2PP2ServingError("worker startup timed out")
                channels = wait(tuple(pending), timeout=min(remaining, 1.0))
                if not channels:
                    if any(process.exitcode not in (None, 0) for process in self._processes):
                        raise self._startup_worker_failure("a graph rank exited during startup")
                    continue
                for channel in channels:
                    rank = pending[channel]
                    try:
                        item = channel.recv()
                    except (EOFError, OSError, ValueError) as error:
                        raise self._startup_worker_failure(
                            f"rank {rank} graph startup channel closed") from error
                    if type(item) is _LoadProgress:
                        if item.rank != rank:
                            raise Qwen3_5TP2PP2ServingError("rank startup progress identity differs")
                        if progress.update(item):
                            deadlines[rank] = time.monotonic() + timeout_seconds
                        continue
                    if type(item) is _Failure:
                        raise Qwen3_5TP2PP2ServingError(
                            f"rank {item.rank} failed without fallback:\n{item.detail}"
                        )
                    if type(item) is not _Ready or item.rank != rank:
                        raise Qwen3_5TP2PP2ServingError(
                            "rank startup result differs from its channel"
                        )
                    received.append(item)
                    del pending[channel]
        finally:
            progress.finish()
        return received

    def _receive(
        self,
        expected: type[Any],
        request_id: str | None,
        timeout_seconds: float,
    ) -> list[Any]:
        deadline = time.monotonic() + timeout_seconds
        received: list[Any] = []
        for rank, channel in enumerate(self._results):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise Qwen3_5TP2PP2ServingError("four-rank graph operation timed out")
            while True:
                try:
                    if not channel.poll(min(remaining, 1.0)):
                        raise queue.Empty
                    item = channel.recv()
                    break
                except queue.Empty:
                    if any(process.exitcode not in (None, 0) for process in self._processes):
                        raise Qwen3_5TP2PP2ServingError(
                            "a graph rank exited without a result"
                        ) from None
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise Qwen3_5TP2PP2ServingError(
                            "four-rank graph operation timed out"
                        ) from None
                except (EOFError, OSError, ValueError) as error:
                    raise Qwen3_5TP2PP2ServingError("graph result channel closed") from error
            if type(item) is _Failure:
                raise Qwen3_5TP2PP2ServingError(
                    f"rank {item.rank} failed without fallback:\n{item.detail}"
                )
            if type(item) is not expected or getattr(item, "request_id", None) != request_id:
                raise Qwen3_5TP2PP2ServingError("rank result differs from the active operation")
            if item.rank != rank:
                raise Qwen3_5TP2PP2ServingError("rank result channel identity differs")
            received.append(item)
        return received

    @property
    def ready(self) -> bool:
        return (
            not self._closed
            and not self._poisoned
            and all(process.is_alive() for process in self._processes)
            and self._dispatcher.is_alive()
        )

    def submit(self, command: _GenerateCommand, output: _TokenSink) -> None:
        record = _ScheduledRequest(
            command,
            output,
            time.monotonic(),
            draft_policy=(
                DynamicDraftPolicy(
                    self._num_speculative_tokens,
                    self._dynamic_max_speculative_tokens,
                    self._draft_policy,
                )
                if getattr(self, "_dynamic_speculation", False)
                else None
            ),
            draft_tokens=self._num_speculative_tokens,
        )
        with self._condition:
            if self._closed:
                raise Qwen3_5TP2PP2ServingError("graph controller is closed")
            if self._poisoned:
                raise Qwen3_5TP2PP2ServingError("graph controller is poisoned")
            if len(self._requests) >= self._max_running_requests:
                raise Qwen3_5TP2PP2Overloaded("maximum running request capacity exhausted")
            if command.request_id in self._requests:
                raise Qwen3_5TP2PP2ServingError("request id is already active")
            # Account for other admitted requests that have not acquired their
            # first state/page yet. The condition serializes concurrent arrivals.
            # Decode growth still uses the ordinary per-round allocation checks.
            pending = sum(
                record.state_slot is None or record.state_slot in self._shared_state_slots
                for record in self._requests.values()
            )
            owners = pending + 1
            provider_pages = owners if self._num_speculative_tokens else 0
            self._evict_prefix_locked(owners, owners, provider_pages)
            if not self._cache_fits(owners, owners, provider_pages):
                raise Qwen3_5TP2PP2Overloaded("KV/state admission capacity exhausted")
            if not self._requests and self._draft_planner is not None:
                for planner in getattr(self, "_draft_cost_domains", {None: self._draft_planner}).values():
                    planner.reset_runtime_costs()
            self._requests[command.request_id] = record
            if command.structured_output is not None:
                self._active_structured = getattr(self, "_active_structured", 0) + 1
            self._request_order.append(command.request_id)
            self._serving_stats.admit(record.admitted_at)
            self._restore_prefix_locked(record)
            interrupt = getattr(self, "_continuation_interrupt", None)
            if interrupt is not None:
                interrupt.set()
            self._condition.notify()

    def _draft_cost_planner(self, selected, *, constrained=None, bindings=None):
        """Keep ordinary, integer-ledger and host-grammar round histories private."""
        base = getattr(self, "_draft_planner", None)
        if base is None:
            return None
        if constrained is None:
            constrained = any(r.command.structured_output is not None for r in selected)
        processing = bool(constrained)
        if constrained:
            from cubic_serve.contracts.request import StructuredOutputKind

            controls = (tuple((b.meta, b.reasoning_control) for b in bindings)
                        if bindings is not None else tuple(
                            (r.command.structured_output,
                             getattr(r.command, "reasoning_control", None)) for r in selected))
            if (any(meta is not None for meta, _ in controls)
                    and all(meta is None or (
                        getattr(meta, "kind", None) is StructuredOutputKind.IDENTITY
                        and control is not None) for meta, control in controls)):
                # The integer ledger has no CPU grammar matcher. Its actual
                # per-M/K costs must neither inherit nor train that path.
                processing = 2
        key = (processing, all(r.command.sampling.temperature == 0.0 for r in selected))
        if key == (False, False):
            return base
        domains = getattr(self, "_draft_cost_domains", None)
        if domains is None:
            domains = self._draft_cost_domains = {(False, False): base}
        if key not in domains:
            domain = domains[key] = base.new_cost_domain()
            # A model-only K0 seed omits masked sampling and request-owned
            # matcher work. It cannot authorize an irreversible request stop.
            domain.require_complete_rounds = processing is True
            if not constrained:
                domain.seed_compound_costs(
                    getattr(self, "_cold_compound_samples", {}).get(key[1], ()))
            else:
                domain.seed_compound_costs(
                    getattr(self, "_cold_constraint_compound_samples", {}).get(key, ()))
                if processing is True:
                    domain.seed_constraint_sampling_costs(
                        getattr(self, "_cold_constraint_sampling_samples", {}).get(key[1], ()),
                        greedy_samples=getattr(self, "_cold_constraint_sampling_samples", {}).get(True, ()))
        return domains[key]

    def cancel(self, request_id: str) -> None:
        if type(request_id) is not str or not request_id:
            raise ValueError("controller cancel requires a request id")
        with self._condition:
            record = self._requests.get(request_id)
            if record is None:
                return
            record.cancelled = True
            interrupt = getattr(self, "_continuation_interrupt", None)
            if interrupt is not None:
                interrupt.set()
            self._close_output(record)
            self._condition.notify()

    def _resident_prompt_program_supported(
        self, captured_m: int, selected: tuple[_ScheduledRequest, ...],
    ) -> bool:
        """Use only the actual policy's all-rank startup qualification."""
        stochastic = any(record.command.sampling.temperature != 0.0 for record in selected)
        attribute = ("_resident_stochastic_input_program_capacities" if stochastic
                     else "_resident_input_program_capacities")
        return captured_m in getattr(self, attribute, ())

    def _steady_continuation_page_updates_locked(
        self,
    ) -> tuple[tuple[int, int, int], ...] | None:
        selected = getattr(self, "_resident_steady", ())
        if (
            getattr(self, "_resident_owner_lengths", ())
            or
            (self._num_speculative_tokens and not all(r.draft_stopped for r in selected))
            or not selected
            or len(selected) != len(self._requests)
            or any(
                self._requests.get(record.command.request_id) is not record for record in selected
            )
        ):
            return None
        program_supported = self._resident_prompt_program_supported(
            self._resident_steady_captured_m, selected)
        crossings: list[tuple[int, _ScheduledRequest, int]] = []
        for physical, record in enumerate(selected):
            position = self._next_input_position(record)
            page_index = position // self._block_size
            if (
                record.cancelled
                or (not record.decode_ready and not program_supported)
                or (not record.decode_ready and any(record.command.feature_rows[
                        record.prefill_cursor:record.prefill_cursor + RESIDENT_STEP_CAPACITY
                    ]))
                or record.state_slot is None
                or record.state_slot in self._shared_state_slots
                or not self._resident_decode_positions_safe(record)
            ):
                return None
            if position % self._block_size == 0:
                if page_index != len(record.pages):
                    return None
                crossings.append((physical, record, page_index))
            elif page_index >= len(record.pages) or record.pages[page_index] in self._shared_pages:
                return None
        self._evict_prefix_locked(0, len(crossings), 0)
        if len(self._free_pages) < len(crossings):
            return None
        updates: list[tuple[int, int, int]] = []
        for physical, record, page_index in crossings:
            page = _cache_take(self._free_pages)
            record.pages.append(page)
            updates.append((physical, page_index, page))
        return tuple(updates)

    @staticmethod
    def _resident_steady_rows(
        rows: tuple[_BatchRow, ...], selected: tuple[_ScheduledRequest, ...]
    ) -> bool:
        return len(rows) == len(selected) and owned_single_token_rows(rows, allow_prompt=True)

    @staticmethod
    def _resident_decode_positions_safe(record: _ScheduledRequest) -> bool:
        """The ingress increments axes; it cannot invent a declared transition."""
        command = record.command
        if (command.decode_position_step not in (None, (1, 1, 1))
                or (command.decode_position_step is None and bool(command.positions))):
            return False
        if not record.decode_ready and command.decode_position_first is not None:
            last = (command.positions[-1] if command.positions else
                    (len(command.input_ids) - 1,) * 3)
            return command.decode_position_first == tuple(axis + 1 for axis in last)
        return True

    @staticmethod
    def _next_input_position(record: _ScheduledRequest) -> int:
        return (len(record.command.input_ids) + record.generated_count - 1
                if record.decode_ready else record.prefill_cursor)

    def _resident_survivor_rows_locked(self) -> tuple[int, ...]:
        """Keep an unchanged greedy owner subset inside its qualified bucket."""
        previous = getattr(self, "_resident_steady", ())
        capacity = getattr(self, "_resident_steady_captured_m", 0)
        if (getattr(self, "_resident_grammar", False)
                or not previous or len(previous) == len(self._requests)
                or getattr(self, "_resident_owner_lengths", ())
                or capacity not in getattr(self, "_resident_compaction_capacities", ())
                or getattr(self, "_resident_features_bound", False)
                or any(record.command.sampling.temperature != 0.0 for record in previous)):
            return ()
        rows = tuple(i for i, record in enumerate(previous)
                     if self._requests.get(record.command.request_id) is record)
        if (not rows or len(rows) != len(self._requests)
                or _bucket_for_logical_m(len(rows), self._graph_capacities) != capacity):
            return ()
        selected = tuple(previous[i] for i in rows)
        if any(not record.decode_ready or record.cancelled
               or record.state_slot in self._shared_state_slots for record in selected):
            return ()
        self._resident_steady = selected
        return rows

    def _reserve_resident_window_pages_locked(
        self, selected: tuple[_ScheduledRequest, ...], steps: int,
    ) -> tuple[int, tuple[tuple[int, int, int, int], ...]]:
        crossings = []
        for physical, record in enumerate(selected):
            position = self._next_input_position(record)
            crossing_step = self._block_size - position % self._block_size
            for crossing in range(crossing_step, steps, self._block_size):
                crossings.append((crossing, physical, record))
        if not crossings:
            return steps, ()
        self._evict_prefix_locked(0, len(crossings), 0)
        if len(self._free_pages) < len(crossings):
            return min(step for step, _physical, _record in crossings), ()
        updates = []
        for step, physical, record in crossings:
            index = len(record.pages)
            page = _cache_take(self._free_pages)
            record.pages.append(page)
            updates.append((step, physical, index, page))
        return steps, tuple(updates)

    def _trim_unconsumed_window_pages_locked(
        self, selected: tuple[_ScheduledRequest, ...],
    ) -> None:
        # Called after the final all-rank completion receipt, before its last
        # token is committed. No future page was installed without its step.
        for record in selected:
            if record.cancelled:
                continue  # final commit releases the entire cancelled owner
            consumed_position = self._next_input_position(record)
            visible_pages = consumed_position // self._block_size + 1
            while len(record.pages) > visible_pages:
                page = record.pages.pop()
                self._release_shared(self._free_pages, self._shared_pages, page)

    def _stage_segment_page_program_locked(self, selected, widths, steps):
        """Reserve ordinary COW coordinates without changing live owner state.

        The existing one-step planner supplies every source, destination and
        new-page reset. Original pages remain leased until final consumption.
        A private free-heap copy makes rejection/exception rollback atomic;
        no request, prefix-cache or state-slot mutation is permitted here.
        """
        prompt_draft = bool(self._num_speculative_tokens and selected and all(
            not r.decode_ready and not r.draft_stopped for r in selected))
        if (len(selected) != len(widths) or not selected
                or type(steps) is not int or not 2 <= steps <= RESIDENT_STEP_CAPACITY
                or any(type(width) is not int or width <= 0 for width in widths)
                or any(width > 1 and (r.decode_ready or
                       r.prefill_cursor + steps * width >= len(r.command.input_ids))
                       for r, width in zip(selected, widths, strict=True))
                or any(self._next_input_position(r) + steps * width > self.max_sequence_tokens
                       for r, width in zip(selected, widths, strict=True))
                or any(not r.state_initialized or r.state_slot is None
                       or r.state_slot in self._shared_state_slots
                       or r.pending_cow_pages for r in selected)
                or (self._num_speculative_tokens and not prompt_draft and
                    any(not r.draft_stopped for r in selected))):
            return None
        shadows = tuple(replace(r, pages=list(r.pages), mtp_pages=list(r.mtp_pages))
                        for r in selected)
        original_heap = self._free_pages
        original_provider_heap = self._free_mtp_pages if prompt_draft else None
        committed = False
        program = []
        reserved = [set() for _ in selected]
        try:
            from cubic_serve.runtime.cache.shared_pool import SharedCacheJournal
            self._free_pages=(list(original_heap) if isinstance(original_heap,list)
                              else SharedCacheJournal(original_heap))
            if prompt_draft:
                self._free_mtp_pages=(list(original_provider_heap) if isinstance(original_provider_heap,list)
                                     else SharedCacheJournal(original_provider_heap))
            for _ in range(steps):
                # Price the complete step before invoking the ordinary
                # planner. It therefore cannot evict shared prefixes or fail
                # an actual request's output while operating on its shadow.
                needed = 0
                for record, width in zip(shadows, widths, strict=True):
                    position = self._next_input_position(record)
                    first = position // self._block_size
                    last = (position + width - 1) // self._block_size
                    needed += max(0, last + 1 - len(record.pages))
                    needed += (last - first + 1 if width > 1 else sum(
                        record.pages[index] in self._shared_pages
                        for index in range(first, min(last + 1, len(record.pages)))
                    ))
                provider_needed = (sum(max(0,
                    (self._next_input_position(record) + width + self._block_size - 1)
                    // self._block_size - len(record.mtp_pages))
                    for record, width in zip(shadows, widths, strict=True)) if prompt_draft else 0)
                if len(self._free_pages) < needed or (prompt_draft and len(self._free_mtp_pages) < provider_needed):
                    break
                coordinates = []
                for owner, (record, width) in enumerate(zip(shadows, widths, strict=True)):
                    rows = self._reserve_segment_locked(record, width, running=len(shadows))
                    if rows is None or len(rows) != width:
                        raise Qwen3_5TP2PP2ServingError("page program preflight differs from allocation")
                    resets = set(rows[0].new_pages)
                    reserved[owner].update(resets)
                    seen = set()
                    for row in rows:
                        if (row.state_source_slot != record.state_slot
                                or row.state_destination_slot != record.state_slot):
                            raise Qwen3_5TP2PP2ServingError("page program changed its private state owner")
                        index = row.position // self._block_size
                        first_on_page = index not in seen
                        coordinates.append((
                            row.page_source, row.page_destination, index,
                            int(first_on_page and row.page_source in resets),
                        ))
                        if first_on_page:
                            seen.add(index)
                            record.pages[index] = row.page_destination
                        if row.page_source != row.page_destination:
                            reserved[owner].add(row.page_destination)
                    was_prefill = not record.decode_ready
                    if was_prefill:
                        record.prefill_cursor += width
                    if not was_prefill or record.decode_ready:
                        record.generated_count += 1
                        # Only page metadata is retained. Sampled token
                        # feedback remains owned by the existing GPU ingress.
                        record.last_token_id = 0
                program.append(tuple(coordinates))
            if len(program) < 2:
                return None
            pending = tuple(tuple(sorted(pages)) for pages in reserved)
            for record, shadow, pages in zip(selected, shadows, pending, strict=True):
                record.pending_cow_pages = pages
                if prompt_draft:
                    record.mtp_pages[:] = shadow.mtp_pages
            committed = True
            return tuple(program)
        finally:
            if not isinstance(original_heap,list):
                self._free_pages.finish(commit=committed)
                self._free_pages=original_heap
                if prompt_draft:
                    self._free_mtp_pages.finish(commit=committed)
                    self._free_mtp_pages=original_provider_heap
            if not committed:
                self._free_pages = original_heap
                if prompt_draft:
                    self._free_mtp_pages = original_provider_heap
                for record in selected:
                    record.pending_cow_pages = ()

    def _next_segment_continuation_locked(self):
        """Prepare one fixed-layout segment window on named private COW pages."""
        if getattr(self, "_active_structured", 0):
            return None
        widths = getattr(self, "_resident_owner_lengths", ())
        selected = getattr(self, "_resident_steady", ())
        capacity = getattr(self, "_resident_steady_captured_m", 0)
        if (not widths or len(widths) != len(selected) or not selected
                or capacity not in getattr(self, "_resident_segment_program_capacities", ())
                or len(selected) != len(self._requests)
                or getattr(self, "_resident_features_bound", False)
                or any(self._requests.get(r.command.request_id) is not r for r in selected)):
            return None
        prompt_draft = bool(self._num_speculative_tokens and all(
            not r.decode_ready and not r.draft_stopped for r in selected))
        if self._num_speculative_tokens and not prompt_draft and any(not r.draft_stopped for r in selected):
            return None
        if any(r.cancelled or not r.state_initialized or r.state_slot is None
               or r.state_slot in self._shared_state_slots
               or r.command.sampling.temperature != 0.0 or r.pending_cow_pages
               or (width > 1 and r.decode_ready)
               or ((r.decode_ready or width == 1)
                   and not self._resident_decode_positions_safe(r))
               for r, width in zip(selected, widths, strict=True)):
            return None
        if sum(len(r.command.eos_token_ids) for r in selected) > self.vocab_size:
            return None
        duration = self._worker_timeout
        configured = getattr(self, "_decode_window_seconds", None)
        if configured is not None:
            duration = min(duration, configured)
        if duration <= 0:
            return None
        steps = RESIDENT_STEP_CAPACITY
        cost = getattr(self, "_resident_step_seconds", 0.0)
        if cost > 0:
            steps = min(steps, max(1, int(duration / cost)))
        remaining, offsets = [], []
        for record, width in zip(selected, widths, strict=True):
            position = self._next_input_position(record)
            if position <= 0 or len(record.pages) != (position - 1) // self._block_size + 1:
                return None
            offset = (position - 1) % self._block_size
            offsets.append(offset)
            steps = min(steps, (self.max_sequence_tokens - position) // width)
            count = 0 if record.decode_ready else len(record.command.input_ids) - record.prefill_cursor
            remaining.append(count)
            if width > 1 or prompt_draft:
                steps = min(steps, (count - 1) // width)
            else:
                if record.pages[-1] in self._shared_pages:
                    return None
                steps = min(steps, record.command.max_tokens - record.generated_count
                            + max(0, count - 1))
            if count and getattr(self, "_prefix_cache_enabled", False):
                boundary = self._cacheable_prefix_length(record.command.input_ids)
                if record.prefill_cursor < boundary:
                    steps = min(steps, (boundary - record.prefill_cursor) // width)
        if steps < 2:
            return None
        # Reject future media before reserving any pages. The fixed window
        # consumes only already-declared token/position producers.
        for record, width, count in zip(selected, widths, remaining, strict=True):
            begin = record.prefill_cursor
            if count and record.command.feature_rows and any(
                record.command.feature_rows[begin:begin + min(count, steps * width)]
            ):
                return None
        if prompt_draft and any(page in self._shared_mtp_pages
            for record in selected for page in record.mtp_pages[record.prefill_cursor // self._block_size:]):
            return None
        source_pages = tuple(record.pages[-1] for record in selected)
        program = self._stage_segment_page_program_locked(selected, widths, steps)
        if program is None:
            return None
        steps = len(program)
        ids, positions = [], []
        for record, width, count in zip(selected, widths, remaining, strict=True):
            begin = record.prefill_cursor
            stop = begin + min(count, steps * width)
            ids.append(record.command.input_ids[begin:stop] if count else ())
            positions.append(tuple(
                record.command.positions[p] if record.command.positions else (p, p, p)
                for p in range(begin, stop)
            ) if count else ())
        command = _ContinueBatchCommand(
            self._batch_id, self._resident_steady_batch_id, capacity,
            tuple(r.command.request_id for r in selected), max_steps=steps,
            eos_token_ids=tuple(r.command.eos_token_ids for r in selected),
            max_duration_seconds=duration, prompt_remaining=tuple(remaining),
            prompt_tokens=tuple(ids), prompt_positions=tuple(positions),
            owner_lengths=widths, source_pages=source_pages, page_program=program,
            provider_tables=tuple(tuple(r.mtp_pages) for r in selected) if prompt_draft else (),
        )
        interrupt = getattr(self, "_continuation_interrupt", None)
        if interrupt is not None:
            interrupt.clear()
        self._batch_id += 1
        self._resident_steady_batch_id = command.batch_id
        return command, (), selected

    def _finish_segment_pages_locked(self, command, selected, steps):
        """Adopt only completed program writes after the all-rank receipt."""
        if (not 1 <= steps <= len(command.page_program)
                or len(selected) != len(command.owner_lengths)
                or len(selected) != len(command.source_pages)):
            raise Qwen3_5TP2PP2ServingError("resident page program completion differs")
        updates = []
        first = 0
        for record, width, source in zip(selected, command.owner_lengths,
                                         command.source_pages, strict=True):
            if not record.pages or record.pages[-1] != source:
                raise Qwen3_5TP2PP2ServingError("resident page program source lease differs")
            owned = set(record.pages) | set(record.pending_cow_pages)
            final = list(record.pages)
            for coordinates in command.page_program[:steps]:
                for src, target, index, _reset in coordinates[first:first + width]:
                    if src not in owned or target not in owned or not 0 <= index <= len(final):
                        raise Qwen3_5TP2PP2ServingError("resident page program write-set differs")
                    if index == len(final):
                        final.append(target)
                    else:
                        final[index] = target
            updates.append((record, final, owned - set(final)))
            first += width
        for record, final, released in updates:
            if record.cancelled:
                continue  # Existing deferred release owns original + all pending pages.
            record.pages[:] = final
            record.pending_cow_pages = ()
            for page in released:
                self._release_shared(self._free_pages, self._shared_pages, page)

    def _next_draft_continuation_locked(self):
        resident = getattr(self, "_resident_draft", None)
        if resident is None:
            return None
        previous, captured_m, depth, selected = resident
        grammar = getattr(self, "_resident_grammar_draft", False)
        capacities = getattr(self, "_grammar_draft_continuation_capacities", ()) if grammar else (
            getattr(self, "_resident_draft_capacities", ()))
        dynamic = getattr(self, "_dynamic_speculation", False)
        decision_horizon = (draft_decision_horizon(
            tuple(cast(DynamicDraftPolicy, record.draft_policy) for record in selected),
            coherent_rounds=True)
            if dynamic and all(record.draft_policy is not None for record in selected) else 0)
        if (captured_m not in capacities
                or (not dynamic and self._num_speculative_tokens != depth)
                or (dynamic and any(
                    record.draft_policy is None or record.draft_policy.stopped
                    or record.draft_policy.current_tokens != depth
                    or record.draft_tokens != depth
                    for record in selected))
                or any(record.cancelled or record.draft_stopped or not record.decode_ready
                       or record.state_slot is None or record.state_slot in self._shared_state_slots
                       or record.pending_cow_pages
                       or not self._resident_decode_positions_safe(record)
                       for record in selected)
                or any(all(record is not owner for owner in selected) and not record.cancelled
                       for record in self._requests.values())):
            return None
        policies = tuple(record.draft_policy for record in selected) if dynamic else ()
        planner = (self._draft_cost_planner(selected)
                   if dynamic and getattr(self, "_draft_planner", None) is not None else None)
        exact_budget = bool(
            planner is not None and len(selected) == len(self._requests)
            and all(self._requests.get(r.command.request_id) is r for r in selected)
            and planner.permits_coherent_budget(policies, row_limit=self._graph_capacities[-1]))
        if dynamic and decision_horizon < 2 and not (
                exact_budget and not all(policy.ready for policy in policies)):
            policies = tuple(record.draft_policy for record in selected)
            if (len(selected) != len(self._requests)
                    or any(self._requests.get(r.command.request_id) is not r for r in selected)
                    or not policies or not all(policy.ready for policy in policies)):
                return None
            planner = self._draft_cost_planner(selected)
            if planner is None or not planner.has_compound_cost(len(selected), depth):
                return None
            targets = planner.plan(
                policies, row_limit=self._graph_capacities[-1],
                fixed_decode_requests=0, target_extra_rows=0,
                available_decode_requests=len(selected), coherent_rounds=True,
            )
            if tuple(targets) != (depth,) * len(selected):
                return None
            for policy in policies:
                policy.move_toward(depth, complete_round=True)
            decision_horizon = draft_decision_horizon(policies, coherent_rounds=True)
            if decision_horizon < 2:
                return None
        decision_tokens = tuple(policy.decision_budget[0] for policy in policies) if exact_budget else ()
        decision_rounds = tuple(policy.decision_budget[1] for policy in policies) if exact_budget else ()
        positions = tuple(self._next_input_position(record) for record in selected)
        target_limits = tuple(min(len(record.pages) * self._block_size, self.max_sequence_tokens)
                              for record in selected)
        provider_limits = tuple(min(len(record.mtp_pages) * self._block_size, self.max_sequence_tokens)
                                for record in selected)
        # Cached prefix pages are immutable read operands. Only the append
        # write-set must be exclusively owned by this transaction.
        if any(page in self._shared_pages
               for record, position in zip(selected, positions)
               for page in record.pages[position // self._block_size:]) or any(
            page in self._shared_mtp_pages
            for record, position in zip(selected, positions)
            for page in record.mtp_pages[position // self._block_size:]
        ):
            return None
        remaining = tuple(record.command.max_tokens - record.generated_count for record in selected)
        seconds = self._worker_timeout
        configured = getattr(self, "_decode_window_seconds", 0.1)
        if configured is not None:
            seconds = min(seconds, configured)
        step_seconds = getattr(self, "_resident_draft_step_seconds", 0.)
        if seconds <= 0 or step_seconds <= 0:
            return None
        stride = depth + 1
        # Only already-granted pages are reachable. Provider realignment can
        # consume another K inputs beyond the next root.
        steps = min(RESIDENT_STEP_CAPACITY, int(seconds / step_seconds),
                    *(value // stride for value in remaining),
                    *((self.max_sequence_tokens - position - 2 * depth + stride - 1) // stride
                      for position in positions))
        if dynamic:
            # Uniform candidates cannot transition before all owners are ready.
            # Admit the worst real acceptance trace; workers stop on actual work.
            # Other cohorts retain the original conservative checkpoints.
            steps = min(steps, max((*decision_tokens, *decision_rounds))
                        if exact_budget else decision_horizon)
        if grammar:
            # Host-matcher continuation borrows only the completed transaction's
            # existing grants. At a page boundary the ordinary allocator resumes.
            steps = min(steps, *((limit - position - depth + stride - 1) // stride
                                for limit, position in zip(target_limits, positions)),
                        *((limit - position - 2 * depth + stride - 1) // stride
                          for limit, position in zip(provider_limits, positions)))
        def page_needs(count):
            target = tuple(max(0, (position + count * stride + self._block_size - 1)
                                  // self._block_size - len(record.pages))
                           for position, record in zip(positions, selected))
            provider = tuple(max(0, (position + count * stride + depth + self._block_size - 1)
                                    // self._block_size - len(record.mtp_pages))
                             for position, record in zip(positions, selected))
            return target, provider
        # Bound admission against both real pools before mutating either one.
        # Prefix eviction/COW remains in the ordinary allocator; this lease
        # already proves private append ownership and only grants fresh pages.
        low, high = 0, max(0, steps)
        while low < high:
            middle = (low + high + 1) // 2
            target_need, provider_need = page_needs(middle)
            if (sum(target_need) <= len(getattr(self, "_free_pages", ()))
                    and sum(provider_need) <= len(getattr(self, "_free_mtp_pages", ()))):
                low = middle
            else:
                high = middle - 1
        steps = low
        if steps < 2:
            return None
        target_need, provider_need = page_needs(steps)
        for record, target_count, provider_count in zip(selected, target_need, provider_need):
            record.pages.extend(_cache_take(self._free_pages) for _ in range(target_count))
            record.mtp_pages.extend(_cache_take(self._free_mtp_pages) for _ in range(provider_count))
        target_limits = tuple(min(len(record.pages) * self._block_size, self.max_sequence_tokens)
                              for record in selected)
        provider_limits = tuple(min(len(record.mtp_pages) * self._block_size, self.max_sequence_tokens)
                                for record in selected)
        binding = DraftWindowBinding(depth, tuple(record.state_slot for record in selected),
                                     positions, target_limits, provider_limits, remaining,
                                     tuple(tuple(record.pages) for record in selected),
                                     tuple(tuple(record.mtp_pages) for record in selected),
                                     decision_tokens, decision_rounds)
        interrupt = getattr(self, "_continuation_interrupt", None)
        if interrupt is not None:
            interrupt.clear()
        command = _ContinueBatchCommand(
            self._batch_id, previous, captured_m,
            tuple(record.command.request_id for record in selected),
            max_steps=steps, eos_token_ids=tuple(record.command.eos_token_ids for record in selected),
            max_duration_seconds=seconds, draft=binding,
        )
        self._batch_id += 1
        return command, (), selected

    def _remember_draft_continuation_locked(self, batch, selected, result):
        grammar = result.pipelined_draft_depth > 0
        depth = result.pipelined_draft_depth if grammar else result.resident_draft_depth
        capacities = (getattr(self, "_grammar_draft_continuation_capacities", ()) if grammar
                      else getattr(self, "_resident_draft_capacities", ()))
        if (depth > 0 and batch.captured_m in capacities
                and all(self._requests.get(record.command.request_id) is record
                        and not record.cancelled for record in selected)
                and (not grammar or type(batch) is not _ContinueBatchCommand or all(
                    tuple(record.pages) == target and tuple(record.mtp_pages) == provider
                    for record, target, provider in zip(
                        selected, batch.draft.target_tables, batch.draft.provider_tables,
                        strict=True)))):
            self._resident_draft = (batch.batch_id, batch.captured_m, depth, selected)
            self._resident_grammar_draft = grammar
            if result.step_ms > 0:
                self._resident_draft_step_seconds = result.step_ms / 1000
        else:
            self._resident_draft = None

    def _next_dispatch_locked(
        self,
    ) -> tuple[_WorkerCommand | None, tuple[_BatchRow, ...], tuple[_ScheduledRequest, ...]]:
        draft = self._next_draft_continuation_locked()
        if draft is not None:
            return draft
        resident = getattr(self, "_resident_draft", None)
        if resident is not None and getattr(self, "_resident_grammar_draft", False):
            self._trim_draft_grants_locked(resident[3], resident[2])
        self._resident_draft = None
        segmented = self._next_segment_continuation_locked()
        if segmented is not None:
            return segmented
        source_rows = self._resident_survivor_rows_locked()
        page_updates = self._steady_continuation_page_updates_locked()
        if page_updates is not None:
            selected = self._resident_steady
            # Bound each transaction by its resources and worker wait budget,
            # never by a request's age. Admission, cancellation and encoder
            # work revoke it directly; progress may continue across windows.
            window_seconds = self._worker_timeout
            configured_seconds = getattr(self, "_decode_window_seconds", 0.1)
            if configured_seconds is not None:
                window_seconds = min(window_seconds, configured_seconds)
            step_seconds = getattr(self, "_resident_step_seconds", 0.0)
            max_steps = 1
            if step_seconds > 0:
                max_steps = min(
                    max(1, int(window_seconds / step_seconds)),
                    *(record.command.max_tokens - record.generated_count
                      + max(0, len(record.command.input_ids) - record.prefill_cursor - 1)
                      for record in selected),
                    RESIDENT_STEP_CAPACITY,
                )
            if getattr(self, "_resident_grammar", False):
                # The existing allocator resumes at page/owner boundaries.
                # No future page or CPU matcher belongs to a padded row.
                max_steps = min(max_steps, *(self._block_size -
                    self._next_input_position(record) % self._block_size
                    for record in selected))
            if getattr(self, "_prefix_cache_enabled", False):
                # A cached recurrent snapshot belongs to this exact prefix,
                # before subsequent tokens mutate the privately owned state.
                # Complete the joint lease here, then publish the snapshot.
                checkpoints = tuple(
                    self._cacheable_prefix_length(record.command.input_ids) - record.prefill_cursor
                    for record in selected if not record.decode_ready
                    and record.prefill_cursor < self._cacheable_prefix_length(
                        record.command.input_ids,
                    )
                )
                if checkpoints:
                    max_steps = min(max_steps, *checkpoints)
            max_steps, future_pages = self._reserve_resident_window_pages_locked(
                selected, max_steps,
            )
            interrupt = getattr(self, "_continuation_interrupt", None)
            if interrupt is not None:
                interrupt.clear()
            prompt_remaining = (
                tuple(0 if record.decode_ready else
                      len(record.command.input_ids) - record.prefill_cursor
                      for record in selected)
                if any(not record.decode_ready for record in selected) else ()
            )
            prompt_tokens = tuple(
                record.command.input_ids[record.prefill_cursor:
                                         record.prefill_cursor + min(count, max_steps)]
                for record, count in zip(selected, prompt_remaining, strict=True)
            ) if prompt_remaining else ()
            prompt_positions = tuple(
                tuple(record.command.positions[position] if record.command.positions
                      else (position, position, position)
                      for position in range(record.prefill_cursor,
                                            record.prefill_cursor + min(count, max_steps)))
                for record, count in zip(selected, prompt_remaining, strict=True)
            ) if prompt_remaining else ()
            command = _ContinueBatchCommand(
                self._batch_id,
                self._resident_steady_batch_id,
                self._resident_steady_captured_m,
                tuple(record.command.request_id for record in selected),
                page_updates,
                max_steps,
                tuple(record.command.eos_token_ids for record in selected),
                window_seconds,
                future_pages,
                prompt_remaining,
                prompt_tokens,
                prompt_positions,
                source_rows,
            )
            self._batch_id += 1
            self._resident_steady_batch_id = command.batch_id
            return command, (), selected
        rows, selected = self._next_batch_locked()
        if not rows:
            self._resident_steady = ()
            return None, (), ()
        captured_m = _bucket_for_logical_m(len(rows), self._graph_capacities)
        command = _BatchCommand(self._batch_id, captured_m, rows)
        if getattr(self, "_active_structured", 0) or getattr(self, "_released_constraints", ()):
            from cubic_serve.generation.structured_sampling import ConstraintBinding

            last_rows = {row.request_id: row for row in rows}
            bindings = tuple(
                ConstraintBinding(
                    record.command.request_id,
                    record.command.structured_output,
                    record.generated_count,
                    record.command.max_tokens - record.generated_count,
                    record.command.constraint_reasoning_end,
                    record.command.reasoning_control,
                    (() if record.command.structured_output.kind.value == "identity"
                     and not record.command.eos_token_ids else None),
                )
                for record in selected
                if record.command.structured_output is not None
                and (
                    record.decode_ready
                    or last_rows[record.command.request_id].position
                    == len(record.command.input_ids) - 1
                )
            )
            command = replace(
                command,
                constraints=bindings,
                released_constraints=tuple(getattr(self, "_released_constraints", ())),
            )
            self._released_constraints = []
        self._batch_id += 1
        has_constraints = bool(getattr(self, "_active_structured", 0)) and any(
            record.command.structured_output is not None for record in selected
        )
        grammar_resident = (
            has_constraints and bool(command.constraints)
            and captured_m in getattr(self, "_grammar_continuation_capacities", ())
            and all(row.is_decode for row in rows)
        )
        owner_lengths = (
            owned_segment_rows(rows, allow_draft_prompt=bool(self._num_speculative_tokens))
            if not has_constraints
            and captured_m in getattr(self, "_resident_segment_program_capacities", ())
            and (not self._num_speculative_tokens or all(r.draft_stopped for r in selected)
                 or all(not r.decode_ready and not r.draft_stopped for r in selected))
            else ()
        )
        if (
            (not has_constraints or grammar_resident)
            and self._resident_steady_rows(rows, selected)
            and captured_m in getattr(self, "_physical_graph_capacities", self._graph_capacities)
            and (not self._num_speculative_tokens or all(r.draft_stopped for r in selected))
            and (all(row.is_decode for row in rows)
                 or self._resident_prompt_program_supported(captured_m, selected))
        ):
            self._resident_steady = selected
            self._resident_steady_batch_id = command.batch_id
            self._resident_steady_captured_m = captured_m
            # Ordinary binding clears the prior bucket's feature prefix when
            # transitioning to feature-free rows. A resident window never
            # introduces features; keep this producing boundary explicit.
            self._resident_features_bound = any(row.input_feature for row in rows)
            self._resident_owner_lengths = ()
            self._resident_grammar = grammar_resident
        elif owner_lengths:
            self._resident_steady = selected
            self._resident_steady_batch_id = command.batch_id
            self._resident_steady_captured_m = captured_m
            self._resident_features_bound = False
            self._resident_owner_lengths = owner_lengths
            self._resident_grammar = False
        else:
            self._resident_steady = ()
            self._resident_steady_batch_id = -1
            self._resident_steady_captured_m = 0
            self._resident_owner_lengths = ()
            self._resident_grammar = False
        return command, rows, selected

    def _commit_steady_records_locked(
        self,
        selected: tuple[_ScheduledRequest, ...],
        token_ids: tuple[int, ...],
        output_wakes: TokenWakeBatch,
        *,
        profile: _StepFeedingProfile | None,
        defer_cancelled_release: bool = False,
        allow_prefill: bool = False,
        prefill_counts: tuple[int, ...] = (),
    ) -> tuple[int, int]:
        if len(token_ids) != len(selected):
            raise Qwen3_5TP2PP2ServingError(
                "steady decode returned a different request count"
            )
        state_commit_ns = 0
        output_commit_ns = 0
        widths = prefill_counts or (1,) * len(selected)
        if len(widths) != len(selected) or any(type(width) is not int or width <= 0 for width in widths):
            raise Qwen3_5TP2PP2ServingError("resident receipt has invalid owner widths")
        decode_count = sum(not record.cancelled and record.decode_ready for record in selected)
        prefill_count = sum(
            not record.cancelled and not record.decode_ready
            and record.prefill_cursor + width == len(record.command.input_ids)
            for record, width in zip(selected, widths, strict=True)
        )
        if decode_count:
            self._serving_stats.publish(decode_count)
        if prefill_count:
            self._serving_stats.publish(prefill_count, is_prefill=True)
        for record, token_id, width in zip(selected, token_ids, widths, strict=True):
            record_started_ns = time.monotonic_ns() if profile is not None else 0
            if record.cancelled:
                self._finish_request_stats_locked(record)
                if not defer_cancelled_release:
                    self._release_request_locked(record.command.request_id)
                if profile is not None:
                    state_commit_ns += time.monotonic_ns() - record_started_ns
                continue
            record.state_initialized = True
            if not record.decode_ready:
                if not allow_prefill:
                    raise Qwen3_5TP2PP2ServingError("decode receipt contains a prompt row")
                record.prefill_cursor += width
                if record.prefill_cursor > len(record.command.input_ids):
                    raise Qwen3_5TP2PP2ServingError("resident prefill crossed its prompt boundary")
                if record.prefill_cursor < len(record.command.input_ids):
                    if not defer_cancelled_release:
                        self._publish_prefix_locked(record)
                    continue
            if profile is not None:
                state_commit_ns += time.monotonic_ns() - record_started_ns
                output_started_ns = time.monotonic_ns()
            record.generated_count += 1
            record.last_token_id = token_id
            terminal = (
                token_id in record.command.eos_token_ids
                or record.generated_count >= record.command.max_tokens
            )
            if terminal:
                # A resident draft receipt can publish several tokens through
                # this scalar policy. Drain its preceding deferred prefix
                # before the immediate terminal token and mailbox close.
                output_wakes.flush()
                output_wakes.publish_one(record.output, token_id)
                self._close_output(record)
                self._finish_request_stats_locked(record)
                self._release_request_locked(record.command.request_id)
            else:
                output_wakes.defer_one(record.output, token_id)
            if profile is not None:
                output_commit_ns += time.monotonic_ns() - output_started_ns
        return state_commit_ns, output_commit_ns

    def _commit_steady_decode_locked(
        self,
        selected: tuple[_ScheduledRequest, ...],
        rows: tuple[_BatchRow, ...],
        token_ids: tuple[int, ...],
        output_wakes: TokenWakeBatch,
        *,
        profile: _StepFeedingProfile | None,
    ) -> tuple[int, int] | None:
        """Commit the allocation-free MTP0 decode case without transaction sets."""

        if (
            (self._num_speculative_tokens and not all(r.draft_stopped for r in selected))
            or len(selected) != len(rows)
            or len(token_ids) != len(rows)
            or not all(
                row.is_decode
                and row.segment_length == 1
                and row.request_id == record.command.request_id
                and row.state_source_slot == record.state_slot
                and row.state_destination_slot == record.state_slot
                and record.state_slot not in self._shared_state_slots
                and row.page_source == row.page_destination
                and row.page_source not in self._shared_pages
                for record, row in zip(selected, rows, strict=True)
            )
        ):
            return None

        return self._commit_steady_records_locked(
            selected, token_ids, output_wakes, profile=profile
        )

    def _observe_resident_step(self, result: _BatchResult) -> None:
        if result.step_ms > 0:
            self._resident_step_seconds = max(
                result.step_ms / 1000,
                getattr(self, "_resident_step_seconds", 0.0) * 0.95,
            )

    def _trim_draft_grants_locked(self, selected, depth):
        """Release unused private grants after declining continuation reuse."""
        for record in selected:
            if self._requests.get(record.command.request_id) is not record:
                continue
            position = self._next_input_position(record)
            # Target committed before the bonus; the provider has also
            # consumed the carried proposal prefix. Keep those distinct
            # horizons, release unused grants only after all-rank consume.
            for pages, needed, free, shared in (
                (record.pages, (position + self._block_size - 1) // self._block_size,
                 self._free_pages, self._shared_pages),
                (record.mtp_pages, (position + depth + self._block_size - 1)
                 // self._block_size, self._free_mtp_pages, self._shared_mtp_pages),
            ):
                while len(pages) > needed:
                    self._release_shared(free, shared, pages.pop())

    def _commit_draft_receipt_locked(self, batch, selected, result, wakes, *, profile):
        """Reuse token publication policy without reallocating committed owners."""
        segments = result.segment_results
        if (tuple(segment.request_id for segment in segments) != batch.request_ids
                or any(segment.committed_row_count != len(segment.output_token_ids)
                       or not 1 <= segment.committed_row_count <= batch.draft.depth + 1
                       for segment in segments)):
            raise Qwen3_5TP2PP2ServingError("draft window returned invalid owner prefixes")
        if not result.continuation_complete and any(
            not record.cancelled and (
                record.generated_count + len(segment.output_token_ids) >= record.command.max_tokens
                or any(token in record.command.eos_token_ids for token in segment.output_token_ids)
            ) for record, segment in zip(selected, segments, strict=True)
        ):
            raise Qwen3_5TP2PP2ServingError("draft window advanced beyond a terminal receipt")
        self._serving_stats.speculate(
            sum(segment.committed_row_count - 1 for segment in segments),
            len(selected) * batch.draft.depth, len(selected),
        )
        state_ns = output_ns = 0
        for index in range(max(segment.committed_row_count for segment in segments)):
            active = tuple((record, segment.output_token_ids[index])
                           for record, segment in zip(selected, segments, strict=True)
                           if index < segment.committed_row_count
                           and self._requests.get(record.command.request_id) is record
                           and (index == 0 or not record.cancelled))
            if active:
                state, output = self._commit_steady_records_locked(
                    tuple(record for record, _token in active),
                    tuple(token for _record, token in active), wakes, profile=profile,
                    defer_cancelled_release=not result.continuation_complete,
                )
                state_ns += state
                output_ns += output
        if getattr(self, "_dynamic_speculation", False):
            for record, segment in zip(selected, segments, strict=True):
                _advance_dynamic_draft(
                    record, accepted=segment.committed_row_count - 1,
                    proposed=batch.draft.depth,
                    terminal=(record.cancelled or
                              self._requests.get(record.command.request_id) is not record),
                )
        if result.continuation_complete and not (
                result.pipelined_draft_depth > 0
                and len(selected) == len(self._requests)
                and all(self._requests.get(record.command.request_id) is record
                        and not record.cancelled and record.decode_ready
                        for record in selected)):
            self._trim_draft_grants_locked(selected, batch.draft.depth)
        return state_ns, output_ns

    def _consume_resident_window(
        self, batch: _ContinueBatchCommand, selected: tuple[_ScheduledRequest, ...],
        result: _BatchResult, *, stats_started_at: float,
        profile: _StepFeedingProfile | None,
    ) -> tuple[_BatchResult, float]:
        """Consume immutable intermediate receipts while retaining all ownership.

        The caller holds the topology lease throughout. Cancellation closes
        output immediately but cannot free pages/state until the final receipt.
        Workers check EOS, window budget and new work at each completed step.
        """
        widths = batch.owner_lengths or (1,) * len(selected)
        # Resident target-only execution publishes its measured steps while
        # the topology lease is still held. Price those steady intervals;
        # the final receipt also includes draining and the completion barrier.
        # Prompt recurrence is a different workload; draft windows retain a
        # separate complete-round price instead of adding their stages twice.
        cost_planner = (
            self._draft_cost_planner(selected)
            if not batch.prompt_remaining else None
        )
        first_receipt_ns = last_receipt_ns = 0
        first_step_ms = worker_ms = target_ms = 0.0
        cost_steps = 0
        grammar_window_cost = None
        while not result.continuation_complete:
            if cost_planner is not None and result.step_ms > 0.0:
                last_receipt_ns = time.monotonic_ns()
                if not cost_steps:
                    first_receipt_ns, first_step_ms = last_receipt_ns, result.step_ms
                cost_steps += 1
                worker_ms += result.step_ms
                target_ms += result.target_ms
                if result.grammar_cost is not None:
                    if grammar_window_cost is None:
                        grammar_window_cost = result.grammar_cost
                    else:
                        previous = grammar_window_cost
                        grammar_window_cost = replace(previous, **{
                            name: getattr(previous, name) + getattr(result.grammar_cost, name)
                            for name in ("begin_ms", "mask_ms", "first_mask_ms", "advance_ms",
                                         "fork_ms", "accept_ms", "publish_ms", "mask_calls",
                                         "output_tokens")
                        })
            self._observe_resident_step(result)
            with self._condition:
                if batch.draft is None and any(not record.cancelled and (
                    token in record.command.eos_token_ids
                    or record.generated_count + 1 >= record.command.max_tokens
                ) and (record.decode_ready or record.prefill_cursor + 1 ==
                       len(record.command.input_ids))
                       for record, token in zip(selected, result.token_ids, strict=True)):
                    raise Qwen3_5TP2PP2ServingError(
                        "resident window advanced beyond a terminal receipt"
                    )
                wakes = TokenWakeBatch()
                prefills = sum(width for record, width in zip(selected, widths, strict=True)
                               if not record.decode_ready)
                self._serving_stats.step(
                    prefills, sum(record.decode_ready for record in selected),
                    time.monotonic() - stats_started_at,
                )
                if batch.draft is not None:
                    self._commit_draft_receipt_locked(batch, selected, result, wakes, profile=profile)
                else:
                    self._commit_steady_records_locked(
                        selected, result.token_ids, wakes, profile=profile,
                        defer_cancelled_release=True,
                        allow_prefill=bool(batch.prompt_remaining),
                        prefill_counts=batch.owner_lengths,
                    )
                self._condition.notify_all()
            # The next sealed step is worker-owned; public output does not
            # wait for another controller dispatch or release the active lease.
            wakes.flush()
            stats_started_at = time.monotonic()
            result = self._receive_batch(batch, window_index=result.continuation_index + 1)
        if cost_planner is not None and cost_steps:
            # Publication/IPC can bunch neighboring receipts. Sum first, then
            # divide by actual executed steps; per-step maxima/clamps otherwise
            # turn delivery jitter into fictitious steady execution cost.
            controller_ms = (last_receipt_ns - first_receipt_ns) / 1_000_000 + first_step_ms
            milliseconds = max(worker_ms, controller_ms, target_ms) / cost_steps
            if batch.draft is None:
                cost_planner.observe_resident_cost(
                    target_rows=batch.logical_m, milliseconds=milliseconds,
                    target_ms=(target_ms / cost_steps
                               if target_ms > 0
                               and cost_steps >= cost_planner.config.cost_sample_window else None),
                )
            else:
                if grammar_window_cost is not None:
                    cost_planner.observe_request_round(
                        target_rows=batch.logical_m,
                        proposal_rows=(len(batch.request_ids),) * batch.draft.depth,
                        target_ms=target_ms / cost_steps, worker_ms=worker_ms / cost_steps,
                        round_ms=milliseconds, grammar_cost=grammar_window_cost,
                        steps=cost_steps,
                    )
                cost_planner.observe_compound_cost(
                    target_rows=batch.logical_m, segment_depth=batch.draft.depth + 1,
                    milliseconds=milliseconds, completed_steps=cost_steps,
                    target_ms=(target_ms / cost_steps
                               if grammar_window_cost is None
                               and cost_steps >= cost_planner.config.cost_sample_window else None),
                )
        self._observe_resident_step(result)
        if batch.owner_lengths:
            with self._condition:
                self._finish_segment_pages_locked(batch, selected, result.continuation_index + 1)
                if batch.provider_tables:
                    for record,width in zip(selected,batch.owner_lengths,strict=True):
                        if record.cancelled:
                            continue
                        needed=(record.prefill_cursor+width+self._block_size-1)//self._block_size
                        while len(record.mtp_pages)>needed:
                            self._release_shared(self._free_mtp_pages,self._shared_mtp_pages,record.mtp_pages.pop())
        if batch.future_page_updates:
            with self._condition:
                self._trim_unconsumed_window_pages_locked(selected)
        return result, stats_started_at

    def _dispatch(self) -> None:
        pending_output_wakes: TokenWakeBatch | None = None
        replay_leases = getattr(self, "_replay_leases", None)
        replay_lease = None
        input_encoder_execution_lock = getattr(self, "_input_encoder_execution_lock", None)
        input_encoder_execution_held = False
        try:
            previous_cycle_ended_ns = 0
            previous_result_ns = 0
            previous_result_decode = False
            while True:
                cycle_started_ns = (
                    time.monotonic_ns()
                    if self._feeding_profile is not None and self._feeding_profile.active
                    else 0
                )
                if cycle_started_ns and previous_cycle_ended_ns:
                    self._feeding_profile.record(
                        "dispatch_gap_cpu_ms",
                        (cycle_started_ns - previous_cycle_ended_ns) / 1_000_000,
                    )
                with self._condition:
                    schedule_started_ns = (
                        time.monotonic_ns()
                        if self._feeding_profile is not None and self._feeding_profile.active
                        else 0
                    )
                    self._retire_cancelled_locked()
                    batch, rows, selected = self._next_dispatch_locked()
                    while batch is None and not self._closed:
                        self._trim_checkpoint_bank(0)
                        previous_result_ns = 0
                        previous_result_decode = False
                        if pending_output_wakes is not None:
                            pending_output_wakes.flush()
                            pending_output_wakes = None
                        self._condition.wait(timeout=0.25)
                        self._retire_cancelled_locked()
                        batch, rows, selected = self._next_dispatch_locked()
                    if self._closed:
                        if pending_output_wakes is not None:
                            pending_output_wakes.flush()
                        return
                    assert batch is not None
                    if type(batch) is _BatchCommand and getattr(self,"_shared_cache_pool",None) is not None:
                        self._trim_checkpoint_bank(self._checkpoint_bank_used)
                    if schedule_started_ns:
                        self._feeding_profile.record(
                            "schedule_cpu_ms",
                            (time.monotonic_ns() - schedule_started_ns) / 1_000_000,
                        )
                feeding_profile = self._feeding_profile
                if feeding_profile is not None and not feeding_profile.active:
                    self._feeding_profile = None
                    feeding_profile = None
                stats_started_at = time.monotonic()
                serialize_started_ns = time.monotonic_ns() if feeding_profile is not None else 0
                # Pinned vLLM 73fac1e... and SGLang ba433bb... broadcast one
                # encoded command through local shared memory; llama.cpp
                # c841aeeb... keeps its batch/slot loop process-local.  Keep the
                # payload single-copy and use pipes only as rank wake-up notices.
                payload = _serialize_worker_command(batch)
                if getattr(self,"_shared_cache_pool",None) is not None:
                    from cubic_serve.runtime.cache.shared_pool import SharedCacheOwnership
                    payload=SharedCacheOwnership.pack(payload,self._shared_cache_pool)
                if feeding_profile is not None:
                    feeding_profile.record(
                        "command_serialize_cpu_ms",
                        (time.monotonic_ns() - serialize_started_ns) / 1_000_000,
                    )
                if input_encoder_execution_lock is not None:
                    input_encoder_execution_lock.acquire()
                    input_encoder_execution_held = True
                if replay_leases is not None:
                    replay_lease = replay_leases.acquire(
                        launch_plan_id=f"batch:{batch.batch_id}",
                        execution_plan_id=self._replay_execution_id,
                        bucket_id=f"M{batch.captured_m}",
                        template_id="resident-graph-portfolio",
                        topology_id=self._replay_topology_id,
                        partition_id=self._replay_execution_id,
                    )
                self._publish_worker_command(payload, feeding_profile)
                # Do not wake every HTTP/SSE coroutine until the next GPU step
                # is already in flight.  Otherwise the newly runnable token
                # consumers contend with this dispatcher for the GIL exactly
                # while it is building the next batch.  vLLM similarly keeps
                # engine execution and public-output delivery on separate
                # queues; this one-step handoff preserves that boundary without
                # introducing another process or delaying terminal output.
                if pending_output_wakes is not None:
                    pending_output_wakes.flush()
                    pending_output_wakes = None
                wait_started_ns = time.monotonic_ns() if feeding_profile is not None else 0
                result = self._receive_batch(batch)
                if type(batch) is _ContinueBatchCommand:
                    result, stats_started_at = self._consume_resident_window(
                        batch, selected, result, stats_started_at=stats_started_at,
                        profile=feeding_profile,
                    )
                result_received_ns = time.monotonic_ns()
                round_ms = max(
                    result.step_ms,
                    result.target_ms,
                    (result_received_ns - previous_result_ns) / 1_000_000
                    if previous_result_ns else result.step_ms,
                )
                decode_round = all(row.is_decode for row in rows)
                consecutive_decode = decode_round and previous_result_decode
                previous_result_ns = result_received_ns
                previous_result_decode = decode_round
                draft_planner = self._draft_cost_planner(
                    selected, constrained=bool(getattr(batch, "constraints", ())),
                    bindings=getattr(batch, "constraints", None),
                )
                compound_depth = max(result.resident_draft_depth, result.pipelined_draft_depth)
                if (draft_planner is not None and consecutive_decode
                        and type(batch) is not _ContinueBatchCommand
                        and result.grammar_cost is not None):
                    segments = _batch_segments(rows)
                    proposal_rows = tuple(
                        sum(stop - start > depth for start, stop in segments)
                        for depth in range(1, max(stop - start for start, stop in segments))
                    )
                    draft_planner.observe_request_round(
                        target_rows=len(rows), proposal_rows=proposal_rows,
                        target_ms=result.target_ms, worker_ms=result.step_ms,
                        round_ms=round_ms, grammar_cost=result.grammar_cost,
                        ingress=result.draft_ingress,
                    )
                if (draft_planner is not None and result.step_ms > 0.0
                        and compound_depth > 0
                        and consecutive_decode
                        and type(batch) is not _ContinueBatchCommand):
                    # This price already contains proposal, review, repair,
                    # sampling, communication and the controller handoff.
                    draft_planner.observe_compound_cost(
                        target_rows=len(rows),
                        segment_depth=compound_depth + 1,
                        milliseconds=round_ms,
                        ingress=result.draft_ingress,
                        target_ms=(result.target_ms if result.grammar_cost is None else None),
                    )
                elif (draft_planner is not None and result.step_ms > 0.0
                        and not result.draft_ingress
                        and type(batch) is not _ContinueBatchCommand):
                    draft_planner.observe_cost(
                        target_rows=len(rows) if rows else batch.logical_m,
                        is_decode=consecutive_decode,
                        target_ms=result.target_ms,
                        segment_depth=(batch.draft.depth + 1
                                       if type(batch) is _ContinueBatchCommand and batch.draft is not None
                                       else max((row.segment_length for row in rows), default=1)),
                        proposal_samples=result.proposal_samples,
                        overhead_ms=max(
                            0.0,
                            round_ms
                            - result.target_ms
                            - sum(sample[1] for sample in result.proposal_samples),
                        ),
                    )
                if feeding_profile is not None:
                    feeding_profile.record(
                        "result_wait_cpu_ms",
                        (time.monotonic_ns() - wait_started_ns) / 1_000_000,
                    )
                    feeding_profile.record("worker_step_wall_ms", result.step_ms)
                    fully_speculative_result = bool(result.segment_results) and len(
                        result.segment_results
                    ) == len(selected)
                    if fully_speculative_result:
                        feeding_profile.record("decode_worker_step_wall_ms", result.step_ms)
                        feeding_profile.record("decode_target_graph_gpu_ms", result.target_ms)
                        feeding_profile.record(
                            "decode_proposal_wall_ms",
                            sum(milliseconds for _, milliseconds in result.proposal_samples),
                        )
                        feeding_profile.record_value(
                            "decode_scheduled_requests", float(len(selected))
                        )
                        feeding_profile.record_value(
                            "decode_target_rows", float(len(rows) if rows else batch.logical_m),
                        )
                        feeding_profile.record_value(
                            "decode_proposal_rows",
                            float(sum(count for count, _ in result.proposal_samples)),
                        )
                token_ids = result.token_ids
                commit_started_ns = time.monotonic_ns() if feeding_profile is not None else 0
                with self._condition:
                    output_wakes = TokenWakeBatch()
                    resident_prefills = (
                        sum(width for record, width in zip(
                            selected, batch.owner_lengths or (1,) * len(selected), strict=True,
                        ) if not record.decode_ready)
                        if type(batch) is _ContinueBatchCommand else 0
                    )
                    self._serving_stats.step(
                        resident_prefills if type(batch) is _ContinueBatchCommand
                        else sum(not row.is_decode for row in rows),
                        sum(record.decode_ready for record in selected)
                        if type(batch) is _ContinueBatchCommand
                        else sum(row.is_decode for row in rows),
                        time.monotonic() - stats_started_at,
                    )
                    if type(batch) is _ContinueBatchCommand:
                        if batch.draft is not None:
                            steady_commit = self._commit_draft_receipt_locked(
                                batch, selected, result, output_wakes, profile=feeding_profile,
                            )
                        else:
                            steady_commit = self._commit_steady_records_locked(
                                selected, token_ids, output_wakes, profile=feeding_profile,
                                allow_prefill=bool(batch.prompt_remaining),
                                prefill_counts=batch.owner_lengths,
                            )
                    else:
                        steady_commit = self._commit_steady_decode_locked(
                            selected, rows, token_ids, output_wakes, profile=feeding_profile,
                        )
                    if steady_commit is not None:
                        self._remember_draft_continuation_locked(batch, selected, result)
                        state_commit_ns, output_commit_ns = steady_commit
                        pending_output_wakes = output_wakes
                        self._serving_stats.maybe_report(
                            time.monotonic(),
                            kv_usage_percent=self._kv_usage_percent_locked(),
                        )
                        self._condition.notify_all()
                        if feeding_profile is not None:
                            cycle_ended_ns = time.monotonic_ns()
                            feeding_profile.record(
                                "result_state_commit_cpu_ms", state_commit_ns / 1_000_000
                            )
                            feeding_profile.record(
                                "result_output_commit_cpu_ms", output_commit_ns / 1_000_000
                            )
                            feeding_profile.record(
                                "dispatch_cycle_cpu_ms",
                                (cycle_ended_ns - cycle_started_ns) / 1_000_000,
                            )
                            feeding_profile.record(
                                "result_commit_cpu_ms",
                                (time.monotonic_ns() - commit_started_ns) / 1_000_000,
                            )
                            if not feeding_profile.active:
                                self._feeding_profile = None
                        previous_cycle_ended_ns = time.monotonic_ns()
                        if replay_lease is not None:
                            replay_leases.release(replay_lease)
                            replay_lease = None
                        if input_encoder_execution_held:
                            input_encoder_execution_lock.release()
                            input_encoder_execution_held = False
                        continue
                    segments = _batch_segments(rows)
                    if len(selected) != len(segments):
                        raise Qwen3_5TP2PP2ServingError(
                            "selected requests differ from batch segments"
                        )
                    speculative_results = {item.request_id: item for item in result.segment_results}
                    if len(speculative_results) != len(result.segment_results):
                        raise Qwen3_5TP2PP2ServingError(
                            "batch has duplicate speculative segment results"
                        )
                    if result.segment_results:
                        self._serving_stats.speculate(
                            sum(item.committed_row_count - 1 for item in result.segment_results),
                            sum(
                                stop - start - 1
                                for start, stop in segments
                                if rows[start].request_id in speculative_results
                            ),
                            len(result.segment_results),
                        )
                    if feeding_profile is not None:
                        feeding_profile.record(
                            "result_prepare_cpu_ms",
                            (time.monotonic_ns() - commit_started_ns) / 1_000_000,
                        )
                    state_commit_ns = 0
                    output_commit_ns = 0
                    step_output_tokens = 0
                    step_accepted_draft_tokens = 0
                    step_proposed_draft_tokens = 0
                    for record, (start, stop) in zip(selected, segments, strict=True):
                        record_started_ns = (
                            time.monotonic_ns() if feeding_profile is not None else 0
                        )
                        segment = rows[start:stop]
                        head = segment[0]
                        destination_slots = {row.state_destination_slot for row in segment}
                        speculative = speculative_results.pop(record.command.request_id, None)
                        if speculative is not None:
                            step_accepted_draft_tokens += speculative.committed_row_count - 1
                            step_proposed_draft_tokens += stop - start - 1
                        if record.cancelled:
                            for slot in destination_slots - {record.state_slot}:
                                if self._private_checkpoint(slot):
                                    continue
                                self._release_shared(
                                    self._free_state_slots,
                                    self._shared_state_slots,
                                    slot,
                                )
                            for page in {
                                row.page_destination
                                for row in segment
                                if row.page_destination != row.page_source
                            }:
                                self._release_shared(self._free_pages, self._shared_pages, page)
                            self._finish_request_stats_locked(record)
                            self._release_request_locked(record.command.request_id)
                            if feeding_profile is not None:
                                state_commit_ns += time.monotonic_ns() - record_started_ns
                            continue
                        assert record.state_slot == head.state_source_slot
                        committed_rows = (
                            speculative.committed_row_count
                            if speculative is not None
                            else len(segment)
                        )
                        if not 1 <= committed_rows <= len(segment):
                            raise Qwen3_5TP2PP2ServingError(
                                "speculative commit depth differs from its segment"
                            )
                        committed_state = (
                            segment[-1].state_destination_slot
                            if speculative is not None
                            else segment[committed_rows - 1].state_destination_slot
                        )
                        if not 0 <= committed_state < self._state_slot_capacity or self._private_checkpoint(committed_state):
                            raise Qwen3_5TP2PP2ServingError(
                                "request committed a graph-private scratch state"
                            )
                        if committed_state != head.state_source_slot:
                            self._release_shared(
                                self._free_state_slots,
                                self._shared_state_slots,
                                head.state_source_slot,
                            )
                        for slot in destination_slots - {
                            committed_state,
                            head.state_source_slot,
                        }:
                            if self._private_checkpoint(slot):
                                continue
                            self._release_shared(
                                self._free_state_slots,
                                self._shared_state_slots,
                                slot,
                            )
                        record.state_slot = committed_state
                        committed_pages: set[int] = set()
                        for row in segment:
                            page_index = row.position // self._block_size
                            if page_index in committed_pages:
                                continue
                            committed_pages.add(page_index)
                            if record.pages[page_index] != row.page_source:
                                raise Qwen3_5TP2PP2ServingError(
                                    "request KV source changed during replay"
                                )
                            if row.page_destination != row.page_source:
                                record.pages[page_index] = row.page_destination
                                self._release_shared(
                                    self._free_pages,
                                    self._shared_pages,
                                    row.page_source,
                                )
                        was_prefill = record.prefill_cursor < len(record.command.input_ids)
                        if was_prefill:
                            record.prefill_cursor += stop - start
                        record.state_initialized = True
                        self._publish_prefix_locked(record)
                        produces_output = (not was_prefill) or record.decode_ready
                        if feeding_profile is not None:
                            state_commit_ns += time.monotonic_ns() - record_started_ns
                        if not produces_output:
                            continue
                        output_started_ns = (
                            time.monotonic_ns() if feeding_profile is not None else 0
                        )
                        output_tokens = (
                            speculative.output_token_ids
                            if speculative is not None
                            else (token_ids[stop - 1],)
                        )
                        published: list[int] = []
                        terminal = False
                        for token_id in output_tokens:
                            record.generated_count += 1
                            record.last_token_id = token_id
                            published.append(token_id)
                            terminal = (
                                token_id in record.command.eos_token_ids
                                or record.generated_count >= record.command.max_tokens
                            )
                            if terminal:
                                break
                        if speculative is not None:
                            _advance_dynamic_draft(
                                record,
                                accepted=committed_rows - 1,
                                proposed=len(segment) - 1,
                                terminal=terminal,
                            )
                        if terminal:
                            output_wakes.publish(record.output, tuple(published))
                        else:
                            output_wakes.defer(record.output, tuple(published))
                        self._serving_stats.publish(len(published), is_prefill=was_prefill)
                        step_output_tokens += len(published)
                        if terminal:
                            self._close_output(record)
                            self._finish_request_stats_locked(record)
                            self._release_request_locked(record.command.request_id)
                        if feeding_profile is not None:
                            output_commit_ns += time.monotonic_ns() - output_started_ns
                    if speculative_results:
                        raise Qwen3_5TP2PP2ServingError(
                            "speculative result does not own a scheduled segment"
                        )
                    self._remember_draft_continuation_locked(batch, selected, result)
                    if feeding_profile is not None and fully_speculative_result:
                        feeding_profile.record_value(
                            "decode_published_output_tokens", float(step_output_tokens)
                        )
                        feeding_profile.record_value(
                            "decode_accepted_draft_tokens",
                            float(step_accepted_draft_tokens),
                        )
                        feeding_profile.record_value(
                            "decode_proposed_draft_tokens",
                            float(step_proposed_draft_tokens),
                        )
                    pending_output_wakes = output_wakes
                    self._serving_stats.maybe_report(
                        time.monotonic(),
                        kv_usage_percent=self._kv_usage_percent_locked(),
                    )
                    self._condition.notify_all()
                if feeding_profile is not None:
                    cycle_ended_ns = time.monotonic_ns()
                    if fully_speculative_result:
                        feeding_profile.record(
                            "decode_dispatch_cycle_cpu_ms",
                            (cycle_ended_ns - cycle_started_ns) / 1_000_000,
                        )
                    feeding_profile.record(
                        "result_state_commit_cpu_ms", state_commit_ns / 1_000_000
                    )
                    feeding_profile.record(
                        "result_output_commit_cpu_ms", output_commit_ns / 1_000_000
                    )
                    feeding_profile.record(
                        "dispatch_cycle_cpu_ms",
                        (cycle_ended_ns - cycle_started_ns) / 1_000_000,
                    )
                    feeding_profile.record(
                        "result_commit_cpu_ms",
                        (time.monotonic_ns() - commit_started_ns) / 1_000_000,
                    )
                    if not feeding_profile.active:
                        self._feeding_profile = None
                previous_cycle_ended_ns = time.monotonic_ns()
                if replay_lease is not None:
                    replay_leases.release(replay_lease)
                    replay_lease = None
                if input_encoder_execution_held:
                    input_encoder_execution_lock.release()
                    input_encoder_execution_held = False
        except BaseException as error:
            # A failed replay/consumption keeps its lease unavailable. Teardown
            # destroys the portfolio; no later request may reuse partial state.
            if pending_output_wakes is not None:
                pending_output_wakes.flush()
            if self._closed:
                if input_encoder_execution_held:
                    input_encoder_execution_lock.release()
                return
            diagnostic_event(_LOGGER, 'performance.continuous_batch_dispatcher_failed')
            wrapped = (
                error
                if isinstance(error, Qwen3_5TP2PP2ServingError)
                else Qwen3_5TP2PP2ServingError(f"continuous batch dispatcher failed: {error}")
            )
            with self._condition:
                self._poisoned = True
                for record in tuple(self._requests.values()):
                    if not record.output_closed:
                        record.output.fail(wrapped)
                        record.output_closed = True
                    self._finish_request_stats_locked(record)
                self._condition.notify_all()
            if input_encoder_execution_held:
                input_encoder_execution_lock.release()
            self.close()

    def _cacheable_prefix_length(self, input_ids: tuple[int, ...]) -> int:
        return ((len(input_ids) - 1) // self._block_size) * self._block_size

    def _prefix_keys(
        self, input_ids: tuple[int, ...], prefix_length: int, feature_digest: str = ""
    ) -> Iterator[tuple[int, bytes]]:
        digest = hashlib.blake2s()
        if feature_digest:
            digest.update(feature_digest.encode())
        for stop in range(self._block_size, prefix_length + 1, self._block_size):
            digest.update(self._prefix_block_packer.pack(*input_ids[stop - self._block_size : stop]))
            yield stop, digest.digest()

    @staticmethod
    def _retain_shared(shared: dict[int, int], index: int) -> None:
        shared[index] = shared.get(index, 0) + 1

    @staticmethod
    def _release_shared(free: list[int], shared: dict[int, int], index: int) -> None:
        extra = shared.get(index, 0)
        if extra:
            if extra == 1:
                del shared[index]
            else:
                shared[index] = extra - 1
            return
        _cache_return(free,index)

    def _restore_prefix_locked(self, record: _ScheduledRequest) -> None:
        if not getattr(self, "_prefix_cache_enabled", False):
            return
        query_tokens = self._cacheable_prefix_length(record.command.input_ids)
        if query_tokens == 0:
            return
        candidates = tuple(
            (key, snapshot)
            for key in self._prefix_keys(
                record.command.input_ids, query_tokens, record.command.feature_digest
            )
            if (snapshot := self._prefix_cache.get(key)) is not None
        )
        selected = None
        for key, snapshot in reversed(candidates):
            if snapshot.token_ids == record.command.input_ids[: len(snapshot.token_ids)]:
                selected = key, snapshot
                break
        snapshot = selected[1] if selected is not None else None
        if selected is not None:
            self._prefix_cache.move_to_end(selected[0])
        hit_tokens = len(snapshot.token_ids) if snapshot is not None else 0
        self._serving_stats.prefix_lookup(hit_tokens, query_tokens)
        if snapshot is None:
            return
        self._retain_shared(self._shared_state_slots, snapshot.state_slot)
        for page in snapshot.pages:
            self._retain_shared(self._shared_pages, page)
        for page in snapshot.mtp_pages:
            self._retain_shared(self._shared_mtp_pages, page)
        record.prefill_cursor = hit_tokens
        record.state_slot = snapshot.state_slot
        record.pages.extend(snapshot.pages)
        record.mtp_pages.extend(snapshot.mtp_pages)
        record.state_initialized = True

    def _publish_prefix_locked(self, record: _ScheduledRequest) -> None:
        if not getattr(self, "_prefix_cache_enabled", False):
            return
        prefix_length = self._cacheable_prefix_length(record.command.input_ids)
        if (prefix_length == 0 or record.prefill_cursor != prefix_length
            or record.generated_count != 0):
            return
        key = None
        for candidate in self._prefix_keys(
            record.command.input_ids, prefix_length, record.command.feature_digest
        ):
            key = candidate
        assert key is not None
        if key in self._prefix_cache:
            self._prefix_cache.move_to_end(key)
            return
        if record.state_slot is None:
            raise Qwen3_5TP2PP2ServingError("prefix boundary has no recurrent state")
        page_count = prefix_length // self._block_size
        pages = tuple(record.pages[:page_count])
        mtp_pages = tuple(record.mtp_pages[:page_count])
        if len(pages) != page_count or (
            self._num_speculative_tokens and len(mtp_pages) != page_count
        ):
            raise Qwen3_5TP2PP2ServingError("prefix boundary cache is incomplete")
        self._retain_shared(self._shared_state_slots, record.state_slot)
        for page in pages:
            self._retain_shared(self._shared_pages, page)
        for page in mtp_pages:
            self._retain_shared(self._shared_mtp_pages, page)
        self._prefix_cache[key] = _PrefixSnapshot(
            record.command.input_ids[:prefix_length],
            record.state_slot,
            pages,
            mtp_pages,
        )

    def _private_checkpoint(self,index):
        return (index>=self._state_slot_capacity or
                index in getattr(self,"_checkpoint_bank_set",()))

    def _cache_fits(self,states,pages,provider_pages):
        if getattr(self,"_shared_cache_pool",None) is None:
            return (len(self._free_state_slots)>=states and len(self._free_pages)>=pages
                    and len(self._free_mtp_pages)>=provider_pages)
        from cubic_serve.runtime.cache.shared_pool import shared_reservation_fits
        return shared_reservation_fits(((self._free_state_slots,states),
            (self._free_pages,pages),(self._free_mtp_pages,provider_pages)))

    def _trim_checkpoint_bank(self,keep):
        if getattr(self,"_shared_cache_pool",None) is not None:
            while len(self._checkpoint_bank)>keep:
                index=self._checkpoint_bank.pop()
                self._checkpoint_bank_set.remove(index)
                self._free_state_slots.release(index)

    def _evict_prefix_locked(self, state_slots: int, pages: int, mtp_pages: int) -> None:
        if (getattr(self,"_shared_cache_pool",None) is not None
                and not self._cache_fits(state_slots,pages,mtp_pages)):
            self._trim_checkpoint_bank(self._checkpoint_bank_used)
        while self._prefix_cache and not self._cache_fits(state_slots,pages,mtp_pages):
            _, snapshot = self._prefix_cache.popitem(last=False)
            self._release_shared(
                self._free_state_slots,
                self._shared_state_slots,
                snapshot.state_slot,
            )
            for page in snapshot.pages:
                self._release_shared(self._free_pages, self._shared_pages, page)
            for page in snapshot.mtp_pages:
                self._release_shared(self._free_mtp_pages, self._shared_mtp_pages, page)

    def _next_batch_locked(
        self,
    ) -> tuple[tuple[_BatchRow, ...], tuple[_ScheduledRequest, ...]]:
        if getattr(self,"_shared_cache_pool",None) is not None:
            self._checkpoint_bank_used=0
        profile = getattr(self, "_feeding_profile", None)
        if profile is not None and not profile.active:
            profile = None
        phase_started_ns = time.monotonic_ns() if profile is not None else 0
        if not self._request_order:
            return (), ()
        self._retire_cancelled_locked()
        if not self._request_order:
            return (), ()

        count = len(self._request_order)
        rotated = [
            self._requests[self._request_order[(self._next_request_index + offset) % count]]
            for offset in range(count)
        ]
        decoded = [record for record in rotated if record.decode_ready]
        prefills = [record for record in rotated if not record.decode_ready]
        selected: list[_ScheduledRequest] = []
        rows: list[_BatchRow] = []
        segment_boundaries: list[int] = []
        speculative_tokens = int(getattr(self, "_num_speculative_tokens", 0))
        max_speculative_tokens = int(getattr(self, "_max_speculative_tokens", speculative_tokens))
        running = sum(record.state_slot is not None for record in self._requests.values())
        if profile is not None:
            profile.record(
                "schedule_partition_cpu_ms",
                (time.monotonic_ns() - phase_started_ns) / 1_000_000,
            )
            phase_started_ns = time.monotonic_ns()
        # The admitted request count is intentionally independent from the
        # largest resident graph.  Larger concurrency is drained in repeated
        # fixed-capacity graph steps, preserving the qualified graph ABI.
        batch_limit = self._graph_capacities[-1]
        prefill_chunk_tokens = (getattr(self, "_prefill_chunk_tokens", None) or batch_limit)
        wave_micro_m = (
            self._physical_graph_capacities[-1]
            if batch_limit > self._physical_graph_capacities[-1]
            else None
        )

        # A recurrence segment stays inside one admitted physical wave.
        prefill_chunk_tokens = min(prefill_chunk_tokens, wave_micro_m or batch_limit)

        def wave_accepts(length: int) -> bool:
            if wave_micro_m is None or len(rows) + length <= wave_micro_m:
                return True
            lower = len(rows) + length - wave_micro_m
            boundary = bisect_left(segment_boundaries, lower)
            return (
                boundary < len(segment_boundaries) and segment_boundaries[boundary] <= wave_micro_m
            )

        mixed_prefill_quota = (
            _mixed_prefill_token_quota(batch_limit, len(prefills))
            if decoded and prefills
            else batch_limit
        )
        if decoded and prefills:
            ready_decode_rows = sum(
                _effective_request_draft_tokens(
                    record, default=speculative_tokens, maximum=max_speculative_tokens
                ) + 1
                for record in decoded
            )
            # Keep the existing minimum prefill quota and use capacity that
            # currently ready decode transactions cannot consume.
            mixed_prefill_quota = max(
                mixed_prefill_quota, batch_limit - ready_decode_rows
            )
        # MTP state maintenance uses the proposal body, whose qualified graph
        # pool is physical-M.  A logical target wave may therefore carry more
        # decode/review rows, but never more prefill-maintenance rows than the
        # largest proposal graph can consume in one transaction.
        if speculative_tokens:
            mixed_prefill_quota = min(mixed_prefill_quota, self._physical_graph_capacities[-1])
        prefill_reserve = min(
            mixed_prefill_quota,
            sum(len(record.command.input_ids) - record.prefill_cursor for record in prefills),
        )
        decode_limit = batch_limit - prefill_reserve
        if (
            (not speculative_tokens or all(record.draft_stopped for record in decoded))
            and not prefills
            and len(decoded) <= decode_limit
        ):
            steady_rows: list[_BatchRow] = []
            for record in decoded:
                row = self._owned_single_token_row_locked(record)
                if row is None:
                    break
                steady_rows.append(row)
            else:
                if decoded and self._request_order:
                    last = decoded[-1].command.request_id
                    self._next_request_index = (self._request_order.index(last) + 1) % len(
                        self._request_order
                    )
                if profile is not None:
                    elapsed_ms = (time.monotonic_ns() - phase_started_ns) / 1_000_000
                    profile.record("schedule_draft_plan_cpu_ms", 0.0)
                    profile.record("schedule_reserve_cpu_ms", elapsed_ms)
                return tuple(steady_rows), tuple(decoded)
        draft_planner = getattr(self, "_draft_planner", None)
        if draft_planner is not None:
            candidates: list[_ScheduledRequest] = []
            mandatory_rows = 0
            for record in decoded:
                policy = record.draft_policy
                minimum_k = (
                    0
                    if policy is None or policy.stopped
                    else max(0, policy.current_tokens - int(policy.ready))
                )
                if mandatory_rows + minimum_k + 1 > decode_limit:
                    continue
                mandatory_rows += minimum_k + 1
                candidates.append(record)
            planned = tuple(
                record
                for record in candidates
                if record.draft_policy is not None and not record.draft_policy.stopped
            )
            if planned:
                draft_planner = self._draft_cost_planner(candidates)
                policies = tuple(
                    cast(DynamicDraftPolicy, record.draft_policy) for record in planned
                )
                trace_decision = self._draft_trace and any(policy.ready for policy in policies)
                targets = draft_planner.plan(
                    policies,
                    row_limit=decode_limit,
                    fixed_decode_requests=len(candidates) - len(planned),
                    target_extra_rows=prefill_reserve,
                    available_decode_requests=len(decoded),
                    coherent_rounds=True,
                )
                if trace_decision:
                    diagnostic_event(_LOGGER, 'performance.draft_s',
                        json.dumps(
                            {
                                "requests": [
                                    {
                                        "id": record.command.request_id,
                                        "k": policy.current_tokens,
                                        "survival": [
                                            round(value, 4)
                                            for value in policy.survival_probabilities()
                                        ],
                                        "target": target,
                                    }
                                    for record, policy, target in zip(
                                        planned, policies, targets, strict=True
                                    )
                                    if policy.ready
                                ],
                                "row_limit": decode_limit,
                                "fixed_requests": len(candidates) - len(planned),
                                "prefill_rows": prefill_reserve,
                                "target_ms": {
                                    capacity: round(value, 3)
                                    for capacity, value in draft_planner.target_cost.values.items()
                                },
                                "target_depth_ms": {
                                    depth: {capacity: round(value, 3)
                                            for capacity, value in curve.values.items()}
                                    for depth, curve in draft_planner.target_depth_cost.items()
                                },
                                "proposal_ms": {
                                    capacity: round(value, 3)
                                    for capacity, value in (
                                        draft_planner.proposal_cost.values.items()
                                    )
                                },
                                "overhead_ms": {
                                    capacity: round(value, 3)
                                    for capacity, value in (
                                        draft_planner.overhead_cost.values.items()
                                    )
                                },
                            },
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                    )
                complete_cohort = (
                    not prefills and len(planned) == len(candidates) == len(decoded)
                    and len(set(targets)) == 1
                    and (targets[0] == 0
                         or draft_planner.has_compound_cost(len(planned), targets[0]))
                )
                cohort_ready = all(policy.ready for policy in policies)
                for record, target in zip(planned, targets, strict=True):
                    policy = cast(DynamicDraftPolicy, record.draft_policy)
                    if complete_cohort and not cohort_ready:
                        # Keep ready counters intact until the complete cohort
                        # can consume the same qualified round. Tokens continue
                        # at their current depth; no request waits for decoding.
                        record.draft_tokens = policy.current_tokens
                        record.draft_stopped = policy.stopped
                        continue
                    decision = policy.move_toward(
                        target, complete_round=complete_cohort and target > 0,
                    )
                    record.draft_tokens = decision.next_tokens
                    record.draft_stopped = policy.stopped
            decoded = candidates
        if profile is not None:
            profile.record(
                "schedule_draft_plan_cpu_ms",
                (time.monotonic_ns() - phase_started_ns) / 1_000_000,
            )
            phase_started_ns = time.monotonic_ns()
        cache_blocked=[]
        def reserve_segment(record,length,*,running,scratch_offset=0):
            segment=self._reserve_segment_locked(record,length,running=running,scratch_offset=scratch_offset)
            if (segment is not None or getattr(self,"_shared_cache_pool",None) is None
                    or record.cancelled or record.output_closed
                    or record.state_slot is None and running>=self._max_running_requests):
                return segment
            if length>1 and (not record.decode_ready or getattr(self,"_dynamic_speculation",False)):
                if record.decode_ready:
                    # Preserve the existing terminal-K0 provider contract. No
                    # fake timing/acceptance observations or implicit resume.
                    if record.draft_policy is not None:
                        record.draft_policy.stop_for_capacity()
                    record.draft_tokens=0
                    record.draft_stopped=True
                    for page in record.mtp_pages:
                        self._release_shared(self._free_mtp_pages,self._shared_mtp_pages,page)
                    record.mtp_pages.clear()
                segment=self._reserve_segment_locked(record,1,running=running)
            if segment is None:cache_blocked.append(record)
            return segment

        scheduled_decode_ids: set[str] = set()
        speculative_scratch_rows = 0
        for record in decoded:
            record_draft_tokens = _effective_request_draft_tokens(
                record,
                default=speculative_tokens,
                maximum=max_speculative_tokens,
            )
            length = record_draft_tokens + 1
            scratch_need = max(0, length - 1)
            if (
                len(rows) + length > decode_limit
                or not wave_accepts(length)
                or speculative_scratch_rows + scratch_need > self._state_scratch_capacity
            ):
                continue
            was_new = record.state_slot is None
            segment = reserve_segment(
                record,
                length,
                running=running,
                scratch_offset=speculative_scratch_rows,
            )
            if segment is not None:
                running += int(was_new)
                rows.extend(segment)
                segment_boundaries.append(len(rows))
                selected.append(record)
                scheduled_decode_ids.add(record.command.request_id)
                speculative_scratch_rows += max(0,len(segment)-1)
        prefill_budget = min(
            batch_limit - len(rows),
            mixed_prefill_quota,
        )
        for index, record in enumerate(prefills):
            if prefill_budget <= 0:
                break
            remaining = len(record.command.input_ids) - record.prefill_cursor
            fair_share = max(1, prefill_budget // (len(prefills) - index))
            length = min(remaining, fair_share, prefill_chunk_tokens)
            cache_boundary = self._cacheable_prefix_length(record.command.input_ids)
            if (
                getattr(self, "_prefix_cache_enabled", False)
                and record.prefill_cursor < cache_boundary
            ):
                length = min(length, cache_boundary - record.prefill_cursor)
            if not wave_accepts(length):
                continue
            was_new = record.state_slot is None
            segment = reserve_segment(record,length,running=running)
            if segment is not None:
                running += int(was_new)
                rows.extend(segment)
                segment_boundaries.append(len(rows))
                selected.append(record)
                prefill_budget -= len(segment)
        for record in decoded:
            if record.command.request_id in scheduled_decode_ids:
                continue
            record_draft_tokens = _effective_request_draft_tokens(
                record,
                default=speculative_tokens,
                maximum=max_speculative_tokens,
            )
            length = record_draft_tokens + 1
            scratch_need = max(0, length - 1)
            if (
                len(rows) + length > batch_limit
                or not wave_accepts(length)
                or speculative_scratch_rows + scratch_need > self._state_scratch_capacity
            ):
                continue
            was_new = record.state_slot is None
            segment = reserve_segment(
                record,
                length,
                running=running,
                scratch_offset=speculative_scratch_rows,
            )
            if segment is not None:
                running += int(was_new)
                rows.extend(segment)
                segment_boundaries.append(len(rows))
                selected.append(record)
                speculative_scratch_rows += max(0,len(segment)-1)
        if not rows and cache_blocked:
            # There is no in-flight round whose retirement could free space.
            # Preserve fixed-depth semantics and fail a physical capacity
            # boundary explicitly instead of waiting indefinitely.
            victim=max(cache_blocked, key=lambda record: (
                record.generated_count == 0, record.admitted_at,
            ))
            victim.output.fail(Qwen3_5TP2PP2Overloaded("shared KV/state capacity exhausted"))
            victim.output_closed=True
            victim.cancelled=True
        self._retire_cancelled_locked()
        if selected and self._request_order:
            # Prefill is appended after decode. Advancing from that last
            # prefill would repeatedly skip the same capacity-excluded decodes
            # until the entire prompt completed. Rotate from a served decode.
            last = next(
                (record.command.request_id for record in reversed(selected) if record.decode_ready),
                selected[-1].command.request_id,
            )
            self._next_request_index = (self._request_order.index(last) + 1) % len(
                self._request_order
            )
        if profile is not None:
            profile.record(
                "schedule_reserve_cpu_ms",
                (time.monotonic_ns() - phase_started_ns) / 1_000_000,
            )
        scheduled_rows = self._grant_grammar_draft_lookahead_locked(tuple(rows), tuple(selected))
        return scheduled_rows, tuple(selected)

    def _grant_grammar_draft_lookahead_locked(self, rows, selected):
        """Fund an existing host-matcher window before its immutable table bind.

        The worker still consumes only already-granted, initialized pages.
        Fund the complete worst-case observation window, bounded by output, context,
        decision time, both free pools and existing input initialization staging.
        Optional grants never evict prefixes or displace another admitted row.
        """
        from cubic_serve.contracts.request import StructuredOutputKind

        if (not rows or not selected
                or not any(getattr(record.command, "structured_output", None) is not None
                           and record.command.structured_output.kind is not StructuredOutputKind.IDENTITY
                           for record in selected)
                or any(not row.is_decode for row in rows)
                or len(selected) != sum(not record.cancelled for record in self._requests.values())):
            return rows
        segments = _batch_segments(rows)
        lengths = {stop - start for start, stop in segments}
        if len(segments) != len(selected) or len(lengths) != 1:
            return rows
        stride = next(iter(lengths))
        depth = stride - 1
        capacities = getattr(self, "_grammar_draft_continuation_capacities", ())
        if (not 1 <= depth <= getattr(self, "_max_speculative_tokens", 0)
                or _bucket_for_logical_m(len(rows), self._graph_capacities) not in capacities
                or any(record.state_slot is None or record.draft_stopped
                       or record.state_slot in self._shared_state_slots
                       or record.pending_cow_pages
                       or not self._resident_decode_positions_safe(record)
                       or record.command.max_tokens - record.generated_count < 3 * stride
                       or (getattr(self, "_dynamic_speculation", False) and (
                           record.draft_policy is None or record.draft_policy.stopped
                           or record.draft_policy.current_tokens != depth))
                       for record in selected)):
            return rows
        if (getattr(self, "_dynamic_speculation", False)
                and draft_decision_horizon(
                    tuple(cast(DynamicDraftPolicy, record.draft_policy) for record in selected),
                    coherent_rounds=True) < 3):
            return rows
        for record, (start, stop) in zip(selected, segments, strict=True):
            root = rows[start]
            if (root.request_id != record.command.request_id
                    or root.state_source_slot != rows[stop - 1].state_destination_slot
                    or len(root.page_table) != len(record.pages)
                    or len(root.mtp_page_table) != len(record.mtp_pages)
                    or root.position + 3 * stride > self.max_sequence_tokens
                    or root.mtp_position + 3 * stride + depth > self.max_sequence_tokens):
                return rows
        rounds = min(
            RESIDENT_STEP_CAPACITY + 1,
            *((r.command.max_tokens - r.generated_count) // stride for r in selected),
            *((self.max_sequence_tokens - rows[start].position) // stride
              for start, _ in segments),
            *((self.max_sequence_tokens - rows[start].mtp_position - depth) // stride
              for start, _ in segments),
        )
        if getattr(self, "_dynamic_speculation", False):
            rounds = min(rounds, max(
                cast(DynamicDraftPolicy, r.draft_policy).maximum_rounds_until_decision
                for r in selected))
        seconds = self._worker_timeout
        configured = getattr(self, "_decode_window_seconds", None)
        if configured is not None:
            seconds = min(seconds, configured)
        if seconds <= 0.:
            return rows
        step_seconds = getattr(self, "_resident_draft_step_seconds", 0.)
        # Unpriced startup admission retains only the previous minimum grant.
        rounds = min(rounds, 1 + int(max(0., seconds) / step_seconds)
                     if step_seconds > 0. else 3)
        if rounds < 3:
            return rows
        capacity = _bucket_for_logical_m(len(rows), self._graph_capacities)
        reset_capacity = capacity * (
            (self.max_sequence_tokens + self._block_size - 1) // self._block_size)
        already_new = sum(len(row.new_pages) for row in rows)

        def page_needs(count):
            return tuple((
                max(0, (rows[start].position + count * stride + self._block_size - 1)
                    // self._block_size - len(record.pages)),
                max(0, (rows[start].mtp_position + count * stride + depth
                        + self._block_size - 1) // self._block_size - len(record.mtp_pages)),
            ) for record, (start, _) in zip(selected, segments, strict=True))

        def fits(count):
            needs = page_needs(count)
            target = sum(t for t, _ in needs)
            return (target <= len(self._free_pages)
                    and sum(p for _, p in needs) <= len(self._free_mtp_pages)
                    and target + already_new <= reset_capacity)

        if not fits(3):
            return rows
        low, high = 3, rounds
        while low < high:
            middle = (low + high + 1) // 2
            if fits(middle):
                low = middle
            else:
                high = middle - 1
        needs = page_needs(low)
        target_need = sum(t for t, _ in needs)
        provider_need = sum(p for _, p in needs)
        if not target_need and not provider_need:
            return rows
        granted = list(rows)
        for record, (start, stop), (target_count, provider_count) in zip(
                selected, segments, needs, strict=True):
            target = tuple(_cache_take(self._free_pages) for _ in range(target_count))
            provider = tuple(_cache_take(self._free_mtp_pages) for _ in range(provider_count))
            record.pages.extend(target)
            record.mtp_pages.extend(provider)
            for physical in range(start, stop):
                row = rows[physical]
                granted[physical] = replace(
                    row, page_table=(*row.page_table, *target),
                    mtp_page_table=(*row.mtp_page_table, *provider),
                    new_pages=(*row.new_pages, *target) if physical == start else row.new_pages,
                )
        return tuple(granted)

    def _reserve_segment_locked(
        self,
        record: _ScheduledRequest,
        length: int,
        *,
        running: int,
        scratch_offset: int = 0,
    ) -> tuple[_BatchRow, ...] | None:
        """Reserve one atomic recurrent/KV COW transaction for contiguous tokens."""

        if length <= 0:
            raise Qwen3_5TP2PP2ServingError("request segment must be non-empty")
        new_state = record.state_slot is None
        if new_state and running >= self._max_running_requests:
            return None
        decode = record.decode_ready
        speculative = bool(getattr(self, "_num_speculative_tokens", 0) and decode and length > 1)
        state_shared = record.state_slot in self._shared_state_slots
        # A single-token append to exclusively owned state and KV needs no
        # allocation or COW, whether its input comes from a prompt or sampler.
        # Shared prefixes, new owners and page growth retain full admission.
        if (
            length == 1
            and (not getattr(self, "_num_speculative_tokens", 0) or record.draft_stopped)
            and not new_state
            and not state_shared
        ):
            row = self._owned_single_token_row_locked(record)
            if row is not None:
                return (row,)
        # The CUDA recurrence kernels explicitly support source == destination
        # for one exclusively owned segment.  Only shared prefix state and
        # speculative checkpoints need COW owner slots; graph padding already
        # has its own scratch slab outside this owner capacity.
        reuse_state_exclusive = not speculative and not state_shared
        # Target verification appends into positions at or beyond the logical
        # sequence length.  A rejection shortens only that logical length and
        # the next replay overwrites the rejected suffix, so an exclusively
        # owned KV page needs no copy-on-write.  Shared prefix pages still COW.
        reuse_exclusive = not speculative and (decode or length == 1)
        destination_count = 1
        state_need = int(new_state) + int(speculative and state_shared)
        if not speculative:
            state_need += int(not reuse_state_exclusive or (not new_state and state_shared))
        if speculative and scratch_offset + length - 1 > self._state_scratch_capacity:
            raise Qwen3_5TP2PP2ServingError(
                "speculative rows exceed the graph transaction scratch capacity"
            )
        start = (
            len(record.command.input_ids) + record.generated_count - 1
            if decode
            else record.prefill_cursor
        )
        stop = start + length
        first_page = start // self._block_size
        last_page = (stop - 1) // self._block_size
        missing_pages = max(0, last_page + 1 - len(record.pages))
        touched_pages = last_page - first_page + 1
        destination_pages = (
            sum(
                record.pages[index] in self._shared_pages
                for index in range(first_page, min(last_page + 1, len(record.pages)))
            )
            if reuse_exclusive
            else touched_pages
        )
        admission_mtp = (
            bool(getattr(self, "_num_speculative_tokens", 0)) and not record.draft_stopped
        )
        # A target-first round prepares K pending proposals after the bonus.
        # Price that provider lookahead in the common page transaction; target
        # KV retains its original review horizon and ownership.
        mtp_stop = min(stop + (length - 1 if speculative else 0), self.max_sequence_tokens)
        mtp_missing_pages = max(
            0,
            (mtp_stop + self._block_size - 1) // self._block_size - len(record.mtp_pages),
        )
        pooled=getattr(self,"_shared_cache_pool",None) is not None
        checkpoint_end=scratch_offset+length-1 if speculative else self._checkpoint_bank_used if pooled else 0
        checkpoint_need=max(0,checkpoint_end-len(self._checkpoint_bank)) if pooled else 0
        self._evict_prefix_locked(
            state_need+checkpoint_need,
            missing_pages + destination_pages,
            mtp_missing_pages if admission_mtp else 0,
        )
        # Eviction can remove the last cache co-owner without freeing a slot:
        # the request then owns its original storage exclusively. Admission
        # must price that current ownership, rather than a now-stale COW plan.
        state_shared = record.state_slot in self._shared_state_slots
        reuse_state_exclusive = not speculative and not state_shared
        state_need = int(new_state) + int(speculative and state_shared)
        if not speculative:
            state_need += int(not reuse_state_exclusive or (not new_state and state_shared))
        if reuse_exclusive:
            destination_pages = sum(
                record.pages[index] in self._shared_pages
                for index in range(first_page, min(last_page + 1, len(record.pages)))
            )
        if pooled:
            checkpoint_need=max(0,checkpoint_end-len(self._checkpoint_bank))
            if not self._cache_fits(state_need+checkpoint_need,missing_pages+destination_pages,
                                   mtp_missing_pages if admission_mtp else 0):
                return None
            for _ in range(checkpoint_need):
                index=_cache_take(self._free_state_slots)
                self._checkpoint_bank.append(index);self._checkpoint_bank_set.add(index)
            self._checkpoint_bank_used=max(self._checkpoint_bank_used,checkpoint_end)
        if len(self._free_state_slots) < state_need:
            # Shared-prefix COW is transient pressure, not request overload.
            # Earlier rows in this batch detach from the shared source when
            # they commit; defer this request to the next scheduler turn so
            # the final branch can then update its exclusively owned source
            # in place.  max_running_requests owner slots are therefore
            # sufficient for max_running_requests live requests.
            return None
        if len(self._free_pages) < missing_pages + destination_pages:
            record.output.fail(Qwen3_5TP2PP2Overloaded("BF16 KV COW capacity exhausted"))
            record.output_closed = True
            record.cancelled = True
            return None
        if admission_mtp and len(self._free_mtp_pages) < mtp_missing_pages:
            record.output.fail(Qwen3_5TP2PP2Overloaded("BF16 MTP KV capacity exhausted"))
            record.output_closed = True
            record.cancelled = True
            return None

        if new_state:
            record.state_slot = _cache_take(self._free_state_slots)
        assert record.state_slot is not None
        source_state = record.state_slot
        if speculative:
            final_state = (
                _cache_take(self._free_state_slots)
                if source_state in self._shared_state_slots
                else source_state
            )
            destination_states = (
                *(
                    (self._checkpoint_bank[scratch_offset+offset] if pooled else
                     self._state_slot_capacity+scratch_offset+offset)
                    for offset in range(length-1)
                ),
                final_state,
            )
        else:
            destination_states = (
                tuple(_cache_take(self._free_state_slots) for _ in range(destination_count))
                if not reuse_state_exclusive or source_state in self._shared_state_slots
                else (source_state,)
            )
        new_pages = tuple(_cache_take(self._free_pages) for _ in range(missing_pages))
        record.pages.extend(new_pages)
        if admission_mtp:
            record.mtp_pages.extend(
                _cache_take(self._free_mtp_pages) for _ in range(mtp_missing_pages)
            )
        destinations = {
            index: (
                _cache_take(self._free_pages)
                if not reuse_exclusive or record.pages[index] in self._shared_pages
                else record.pages[index]
            )
            for index in range(first_page, last_page + 1)
        }
        replay_table = list(record.pages)
        for index, destination in destinations.items():
            replay_table[index] = destination

        rows: list[_BatchRow] = []
        copied_pages: set[int] = set()
        for offset, position in enumerate(range(start, stop)):
            page_index = position // self._block_size
            source_page = record.pages[page_index]
            destination_page = destinations[page_index]
            first_on_page = page_index not in copied_pages
            copied_pages.add(page_index)
            if (
                decode
                and record.command.decode_position_first is not None
                and record.command.decode_position_step is not None
            ):
                decode_first = record.command.decode_position_first
                decode_step = record.command.decode_position_step
                generation_offset = position - len(record.command.input_ids)
                position_axes = cast(
                    tuple[int, int, int],
                    tuple(
                        first + generation_offset * step
                        for first, step in zip(decode_first, decode_step, strict=True)
                    ),
                )
            elif record.command.positions:
                position_axes = record.command.positions[position]
            else:
                position_axes = None
            rows.append(
                _BatchRow(
                    request_id=record.command.request_id,
                    token_id=(
                        cast(int, record.last_token_id)
                        if decode
                        else record.command.input_ids[position]
                    ),
                    position=position,
                    state_source_slot=source_state,
                    state_destination_slot=(
                        destination_states[offset] if speculative else destination_states[0]
                    ),
                    page_source=source_page if first_on_page else destination_page,
                    page_destination=destination_page,
                    page_table=tuple(replay_table),
                    reset_state=not record.state_initialized and offset == 0,
                    new_pages=new_pages if offset == 0 else (),
                    segment_length=length if offset == 0 else 0,
                    sampling=(
                        record.command.sampling
                        if speculative or offset == length - 1
                        else GREEDY_SAMPLING
                    ),
                    sample_ordinal=record.generated_count + (offset if speculative else 0),
                    mtp_position=position if admission_mtp else -1,
                    mtp_page_destination=(
                        record.mtp_pages[position // self._block_size] if admission_mtp else -1
                    ),
                    mtp_page_table=tuple(record.mtp_pages) if admission_mtp else (),
                    is_decode=decode,
                    position_axes=position_axes,
                    input_feature=(
                        record.command.feature_rows[position]
                        if not decode and record.command.feature_rows
                        else b""
                    ),
                )
            )
        return tuple(rows)

    def _owned_single_token_row_locked(
        self, record: _ScheduledRequest
    ) -> _BatchRow | None:
        """Project one private single-token append into the shared row ABI."""

        decode = record.decode_ready
        state_slot = record.state_slot
        token_id = (record.last_token_id if decode
                    else record.command.input_ids[record.prefill_cursor])
        if (
            state_slot is None
            or token_id is None
            or state_slot in self._shared_state_slots
        ):
            return None
        position = (len(record.command.input_ids) + record.generated_count - 1
                    if decode else record.prefill_cursor)
        page_index = position // self._block_size
        if page_index >= len(record.pages):
            return None
        page = record.pages[page_index]
        if page in self._shared_pages:
            return None
        if (
            decode
            and record.command.decode_position_first is not None
            and record.command.decode_position_step is not None
        ):
            generation_offset = position - len(record.command.input_ids)
            position_axes = cast(
                tuple[int, int, int],
                tuple(
                    first + generation_offset * step
                    for first, step in zip(
                        record.command.decode_position_first,
                        record.command.decode_position_step,
                        strict=True,
                    )
                ),
            )
        elif record.command.positions:
            position_axes = record.command.positions[position]
        else:
            position_axes = None
        return _BatchRow(
            request_id=record.command.request_id,
            token_id=token_id,
            position=position,
            state_source_slot=state_slot,
            state_destination_slot=state_slot,
            page_source=page,
            page_destination=page,
            page_table=tuple(record.pages),
            reset_state=not record.state_initialized,
            segment_length=1,
            sampling=record.command.sampling,
            sample_ordinal=record.generated_count,
            is_decode=decode,
            position_axes=position_axes,
            input_feature=(
                record.command.feature_rows[position]
                if not decode and record.command.feature_rows else b""
            ),
        )

    def _receive_batch(
        self,
        batch: _BatchCommand | _ContinueBatchCommand,
        *, window_index: int = 0,
    ) -> _BatchResult:
        # Each call consumes the next validated receipt, including intermediate
        # resident steps. A healthy long stream never reaches a total deadline.
        deadline = time.monotonic() + self._worker_timeout
        channel = self._results[self._sampler_rank]
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise Qwen3_5TP2PP2ServingError("four-rank graph operation timed out")
            try:
                get_started_ns = time.monotonic_ns() if self._feeding_profile is not None else 0
                if not channel.poll(min(remaining, 1.0)):
                    raise queue.Empty
                item = channel.recv()
                if self._feeding_profile is not None:
                    self._feeding_profile.record(
                        "result_queue_get_cpu_ms",
                        (time.monotonic_ns() - get_started_ns) / 1_000_000,
                    )
                break
            except queue.Empty:
                if any(process.exitcode not in (None, 0) for process in self._processes):
                    raise Qwen3_5TP2PP2ServingError(
                        "a graph rank exited without a result"
                    ) from None
        expected_ids = (
            batch.request_ids
            if type(batch) is _ContinueBatchCommand
            else tuple(row.request_id for row in batch.rows)
        )
        if type(item) is _Failure:
            raise Qwen3_5TP2PP2ServingError(
                f"rank {item.rank} failed without fallback:\n{item.detail}"
            )
        from cubic_serve.generation.structured_sampling import GrammarRoundCost

        cost_cursor = getattr(self, "_grammar_cost_cursor", None)
        expected_grammar_owners = ()
        if type(item) is _BatchResult and item.grammar_cost is not None:
            if type(batch) is _BatchCommand:
                expected_grammar_owners = tuple(
                    (binding.request_id, binding.revision) for binding in batch.constraints)
            elif window_index == 0:
                expected_grammar_owners = tuple(
                    (request_id, self._requests[request_id].generated_count)
                    for request_id in batch.request_ids
                    if self._requests[request_id].command.structured_output is not None)
            else:
                expected_grammar_owners = (cost_cursor[2] if cost_cursor is not None
                                          and cost_cursor[:2] == (batch.batch_id, window_index)
                                          else None)

        if (
            type(item) is not _BatchResult
            or item.rank != self._sampler_rank
            or item.batch_id != batch.batch_id
            or type(item.continuation_index) is not int
            or item.continuation_index != window_index
            or type(item.continuation_complete) is not bool
            or type(item.draft_ingress) is not bool
            or (type(batch) is _ContinueBatchCommand and item.grammar_cost is None
                and cost_cursor is not None
                and cost_cursor[:2] == (batch.batch_id, window_index))
            or (item.grammar_cost is not None and (
                type(item.grammar_cost) is not GrammarRoundCost
                or not item.grammar_cost.valid()
                or item.grammar_cost.owners != expected_grammar_owners
            ))
            or (item.draft_ingress and type(batch) is not _BatchCommand)
            or (item.gpu_step_intervals_ms is not None and (
                type(batch) is not _ContinueBatchCommand
                or type(item.gpu_step_intervals_ms) is not tuple
                or len(item.gpu_step_intervals_ms) != (
                    window_index if item.continuation_complete else 0
                )
                or any(
                    type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                    for value in item.gpu_step_intervals_ms
                )
            ))
            or (type(batch) is _ContinueBatchCommand and (
                not 0 <= window_index < batch.max_steps
                or (window_index + 1 == batch.max_steps and not item.continuation_complete)
            ))
            or (type(batch) is _BatchCommand and (
                window_index != 0 or not item.continuation_complete
            ))
            or item.request_ids != expected_ids
            or len(item.token_ids) not in (0, len(expected_ids))
            or (
                not item.token_ids
                and {segment.request_id for segment in item.segment_results} != set(expected_ids)
            )
            or any(
                segment.request_id not in expected_ids
                or not 1 <= segment.committed_row_count <= self._max_speculative_tokens + 1
                or not 1 <= len(segment.output_token_ids) <= self._max_speculative_tokens + 1
                for segment in item.segment_results
            )
        ):
            raise Qwen3_5TP2PP2ServingError("sampler batch result differs from dispatch")
        if (item.grammar_cost is not None and type(batch) is _ContinueBatchCommand
                and not item.continuation_complete):
            counts = {part.request_id: len(part.output_token_ids) for part in item.segment_results}
            prefixes = tuple((request_id, revision + counts.get(request_id, 1))
                             for request_id, revision in item.grammar_cost.owners)
            # Public counts stop on cancellation; an already leased GPU round
            # must still drain. Cost witnesses follow actual immutable receipts.
            self._grammar_cost_cursor = (batch.batch_id, window_index + 1, prefixes)
        elif cost_cursor is not None:
            self._grammar_cost_cursor = None
        return item

    def _retire_cancelled_locked(self) -> None:
        for request_id in tuple(self._request_order):
            record = self._requests[request_id]
            if record.cancelled:
                self._finish_request_stats_locked(record)
                self._release_request_locked(request_id)

    def _finish_request_stats_locked(self, record: _ScheduledRequest) -> None:
        if record.stats_terminal:
            return
        self._serving_stats.terminal(
            time.monotonic(),
            kv_usage_percent=self._kv_usage_percent_locked(),
        )
        record.stats_terminal = True

    def _kv_usage_percent_locked(self) -> float:
        pool=getattr(self,"_shared_cache_pool",None)
        used=(self._page_capacity-len(self._free_pages) if pool is None else
              pool.live_bits["page"].bit_count())
        return 100.0*used/self._page_capacity

    def _release_request_locked(self, request_id: str) -> None:
        record = self._requests.pop(request_id, None)
        if record is None:
            return
        if record.command.structured_output is not None:
            self._active_structured -= 1
            if not hasattr(self, "_released_constraints"):
                self._released_constraints = []
            self._released_constraints.append(request_id)
        index = self._request_order.index(request_id)
        self._request_order.pop(index)
        if record.state_slot is not None:
            self._release_shared(
                self._free_state_slots,
                self._shared_state_slots,
                record.state_slot,
            )
        for page in record.pages:
            self._release_shared(self._free_pages, self._shared_pages, page)
        for page in record.pending_cow_pages:
            self._release_shared(self._free_pages, self._shared_pages, page)
        record.pending_cow_pages = ()
        for page in record.mtp_pages:
            self._release_shared(self._free_mtp_pages, self._shared_mtp_pages, page)
        if self._request_order:
            if index < self._next_request_index:
                self._next_request_index -= 1
            self._next_request_index %= len(self._request_order)
        else:
            self._next_request_index = 0

    @staticmethod
    def _close_output(record: _ScheduledRequest) -> None:
        if not record.output_closed:
            record.output.finish()
            record.output_closed = True

    def _publish_worker_command(
        self,
        payload: bytes,
        profile: _StepFeedingProfile | None = None,
    ) -> None:
        """Publish one immutable command payload, then wake every local rank."""

        if not self._commands:
            return
        if len(payload) > self._shared_command_capacity:
            raise Qwen3_5TP2PP2ServingError(
                "worker command exceeds the shared command arena"
            )
        with self._command_send_lock:
            self._shared_command_view[: len(payload)] = payload
            notice = _COMMAND_NOTICE.pack(len(payload))
            for channel in self._commands:
                put_started_ns = time.monotonic_ns() if profile is not None else 0
                channel.send_bytes(notice)
                if profile is not None:
                    profile.record(
                        "command_queue_put_cpu_ms",
                        (time.monotonic_ns() - put_started_ns) / 1_000_000,
                    )

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            with self._condition:
                self._closed = True
                interrupt = getattr(self, "_continuation_interrupt", None)
                if interrupt is not None:
                    interrupt.set()
                for record in self._requests.values():
                    self._close_output(record)
                    self._finish_request_stats_locked(record)
                self._condition.notify_all()
            close_payload = _serialize_worker_command(_CloseCommand())
            with suppress(OSError, ValueError):
                self._publish_worker_command(close_payload)
            processes = tuple(p for p in self._processes if p.pid is not None)
            deadline = time.monotonic() + 10
            for process in processes:
                process.join(max(0.0, deadline - time.monotonic()))
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
            for resource_channel in (*self._commands, *self._results):
                with suppress(OSError, ValueError):
                    cast(Any, resource_channel).close()
            dispatcher = getattr(self, "_dispatcher", None)
            if dispatcher is not None and dispatcher is not threading.current_thread():
                dispatcher.join(timeout=5)
            interrupt = getattr(self, "_continuation_interrupt", None)
            if isinstance(interrupt, SharedPublicationInterrupt):
                interrupt.close()
            self._directory.cleanup()
            feeding_profile = getattr(self, "_feeding_profile", None)
            if feeding_profile is not None:
                feeding_profile.emit_once()
            heap_lifetime = getattr(self, "_heap_lifetime", None)
            if heap_lifetime is not None:
                heap_lifetime.close()


def _issue_qualification(
    *,
    model_id: str,
    admission: _Admission,
    max_sequence_tokens: int,
    graph_capacities: tuple[int, ...],
    rank_graph_evidence_ids: tuple[str, ...],
) -> Qwen3_5TP2PP2ServingQualification:
    if len(rank_graph_evidence_ids) != admission.parallel_plan.topology.world_size:
        raise ValueError("graph evidence must cover every topology rank")
    inventory_digest = admission.inventory.inventory_digest
    semantic_plan_digest = admission.parallel_plan.semantic_plan_digest
    topology_digest = admission.sealed.topology_digest
    partition_digest = admission.sealed.partition_digest
    graph_buckets = tuple(
        (captured_m, _logical_m_values(captured_m, graph_capacities))
        for captured_m in graph_capacities
    )
    graph_evidence_id = _graph_evidence_id(
        model_id=model_id,
        inventory_digest=inventory_digest,
        semantic_plan_digest=semantic_plan_digest,
        topology_digest=topology_digest,
        partition_digest=partition_digest,
        max_sequence_tokens=max_sequence_tokens,
        rank_graph_evidence_ids=rank_graph_evidence_ids,
        graph_buckets=graph_buckets,
    )
    return Qwen3_5TP2PP2ServingQualification(
        model_id=model_id,
        inventory_digest=inventory_digest,
        semantic_plan_digest=semantic_plan_digest,
        topology_digest=topology_digest,
        partition_digest=partition_digest,
        max_sequence_tokens=max_sequence_tokens,
        rank_graph_evidence_ids=rank_graph_evidence_ids,
        graph_evidence_id=graph_evidence_id,
        graph_buckets=graph_buckets,
    )


_TokenStream = TokenStream


class Qwen3_5TP2PP2PerformanceExecutor:
    """Request-multiplexed graph-only executor with token-wise publication."""

    def __init__(
        self,
        controller: _Controller,
        *,
        model_id: str,
        weight_variant: WeightVariantMeta,
        eos_token_ids: tuple[int, ...],
        qualification: Qwen3_5TP2PP2ServingQualification,
        sampling_defaults: SamplingParameters = GREEDY_SAMPLING,
        input_encoder: Any = None,
    ) -> None:
        self._controller = controller
        self._input_encoder = input_encoder
        self.model_id, self.weight_variant = model_id, weight_variant
        self.qualification = qualification
        self._eos_token_ids = eos_token_ids
        self._sampling_defaults = sampling_defaults
        self._state_lock = threading.Lock()
        self._closed = False
        self._poisoned = False
        self._active: dict[str, _TokenStream] = {}
        self._next_request = 0

    @property
    def max_sequence_tokens(self) -> int:
        """Return the qualified context capacity exposed to request admission."""

        return self._controller.max_sequence_tokens

    @property
    def vocab_size(self) -> int:
        return self._controller.vocab_size

    @property
    def supports_structured_output(self) -> bool:
        return getattr(self._controller, "constraint_graphs_qualified", False) is True

    @property
    def kv_capacity_tokens(self) -> int:
        """Shared logical token capacity; TP copies do not multiply the pool."""

        return self._controller._page_capacity * self._controller._block_size

    @property
    def kv_vram_capacity_tokens(self) -> int | None:
        return self._controller.kv_vram_capacity_tokens

    @property
    def rank_cache_capacities(self) -> tuple[Qwen3_5TP2PP2RankCacheCapacity, ...]:
        """Return the final, physically allocated cache profile for every rank."""

        return self._controller.rank_cache_capacities

    @property
    def ready(self) -> bool:
        with self._state_lock:
            return not self._closed and not self._poisoned and self._controller.ready

    @classmethod
    def start(
        cls,
        checkpoint: str | Path,
        *,
        checkpoint_weights: CheckpointWeightSource | None = None,
        model_id: str,
        weight_variant: WeightVariantMeta,
        devices: tuple[int, ...],
        eos_token_ids: tuple[int, ...],
        qualification: Qwen3_5TP2PP2ServingQualification | None = None,
        max_sequence_tokens: int | None = None,
        max_running_requests: int = 256,
        tp_size: int = 2,
        pp_size: int = 2,
        ep_size: int = 1,
        num_speculative_tokens: int = 0,
        max_speculative_tokens: int = 0,
        dynamic_max_speculative_tokens: int = 0,
        dynamic_speculation: bool = False,
        use_prefix_cache: bool = True,
        mamba_ssm_cache_dtype: str = "bfloat16",
        prefill_chunk_tokens: int | None = None,
        operator_config: OperatorConfig | None = None,
        draft_policy: DraftPolicyConfig | None = None,
        device_memory_budgets: dict[int, DeviceMemoryBudget] | None = None,
        input_encoder_reserves: dict[int, int] | None = None,
        input_encoder: Any = None,
        startup_timeout_seconds: float = 1800.0,
        worker_timeout_seconds: float = _WORKER_RESPONSE_TIMEOUT_SECONDS,
    ) -> Self:
        if any(type(device) is not int or device < 0 or type(size) is not int or size < 0
               for device, size in (input_encoder_reserves or {}).items()):
            raise ValueError("input encoder reserves require non-negative device indices and byte counts")
        if type(model_id) is not str or not model_id or model_id.strip() != model_id:
            raise ValueError("performance serving requires a canonical model id")
        if type(weight_variant) is not WeightVariantMeta or not weight_variant.is_base:
            raise ValueError("initial performance serving requires base weights")
        if weight_variant.base_model_id != model_id:
            raise ValueError("model and weight-variant identities differ")
        if (
            type(devices) is not tuple
            or len(devices) != tp_size * pp_size
            or any(type(device) is not int or device < 0 for device in devices)
            or len(set(devices)) != tp_size * pp_size
        ):
            raise ValueError("TP x PP requires one unique CUDA device per rank")
        if (
            type(eos_token_ids) is not tuple
            or any(type(token) is not int or token < 0 for token in eos_token_ids)
            or len(set(eos_token_ids)) != len(eos_token_ids)
        ):
            raise ValueError("EOS token ids must be an immutable unique tuple")
        if min(startup_timeout_seconds, worker_timeout_seconds) <= 0:
            raise ValueError("serving timeouts must be positive")
        if (
            min(tp_size, pp_size, ep_size) < 1
            or pp_size not in (1, 2)
            or tp_size % ep_size
        ):
            raise ValueError("performance serving topology is unsupported")

        if draft_policy is not None and type(draft_policy) is not DraftPolicyConfig:
            raise ValueError("draft_policy must be a parsed DraftPolicyConfig")
        if dynamic_speculation:
            diagnostic_event(_LOGGER, 'performance.dynamic_draft_policy_s',
                draft_policy or DraftPolicyConfig(),
            )
        if operator_config is not None and type(operator_config) is not OperatorConfig:
            raise ValueError("operator_config must be a parsed OperatorConfig")
        if operator_config is not None:
            diagnostic_event(_LOGGER, 'performance.operator_config_digest_s_value_s',
                operator_config.digest,
                operator_config,
            )
        sampling_defaults = _load_sampling_defaults(Path(checkpoint))
        admission = _admit(
            Path(checkpoint),
            devices,
            max_sequence_tokens,
            max_running_requests,
            tp_size=tp_size,
            pp_size=pp_size,
            ep_size=ep_size,
            num_speculative_tokens=num_speculative_tokens,
            max_speculative_tokens=max_speculative_tokens,
            dynamic_max_speculative_tokens=dynamic_max_speculative_tokens,
            dynamic_speculation=dynamic_speculation,
            use_prefix_cache=use_prefix_cache,
            mamba_ssm_cache_dtype=mamba_ssm_cache_dtype,
            prefill_chunk_tokens=prefill_chunk_tokens,
            checkpoint_weights=checkpoint_weights,
        )
        colocated_encoder = input_encoder if getattr(input_encoder, "colocated", False) else None
        admission = replace(
            admission,
            operator_config=operator_config or OperatorConfig(),
            operator_model_id=model_id,
            draft_policy=draft_policy or DraftPolicyConfig(),
            device_memory_budgets=dict(device_memory_budgets or {}),
            input_encoder_reserves=dict(input_encoder_reserves or {}),
            input_encoder_factory=colocated_encoder.factory if colocated_encoder is not None else None,
        )
        if weight_variant.base_revision != admission.inventory.inventory_digest:
            raise ValueError("weight variant revision differs from checkpoint admission")
        controller = _ProcessController(
            admission,
            startup_timeout_seconds=startup_timeout_seconds,
            worker_timeout_seconds=worker_timeout_seconds,
            encoder_group=colocated_encoder,
        )
        if colocated_encoder is not None:
            try:
                colocated_encoder.attach_main_workers(
                    controller._processes, controller.input_encoder_connections)
            except BaseException:
                colocated_encoder.close()
                controller.close()
                raise
        observed = _issue_qualification(
            model_id=model_id,
            admission=admission,
            max_sequence_tokens=controller.max_sequence_tokens,
            graph_capacities=controller._graph_capacities,
            rank_graph_evidence_ids=controller.rank_graph_evidence_ids,
        )
        if qualification is not None and qualification != observed:
            if colocated_encoder is not None:
                colocated_encoder.close()
            controller.close()
            raise Qwen3_5TP2PP2ServingError(
                "supplied qualification differs from this cold-start graph instance"
            )
        return cls(
            controller,
            model_id=model_id,
            weight_variant=weight_variant,
            eos_token_ids=eos_token_ids,
            qualification=observed,
            sampling_defaults=sampling_defaults,
            input_encoder=input_encoder,
        )

    @classmethod
    def _for_testing(
        cls,
        controller: _Controller,
        *,
        model_id: str = "test-model",
        eos_token_ids: tuple[int, ...] = (),
    ) -> Self:
        inventory_digest = "b2s256:" + "1" * 64
        semantic_plan_digest = "b2s256:" + "2" * 64
        topology_digest = "b2s256:" + "3" * 64
        partition_digest = "b2s256:" + "4" * 64
        rank_ids = controller.rank_graph_evidence_ids
        graph_capacities = getattr(
            controller, "_graph_capacities", _PHYSICAL_GRAPH_BUCKET_CAPACITIES
        )
        graph_buckets = tuple(
            (captured_m, _logical_m_values(captured_m, graph_capacities))
            for captured_m in graph_capacities
        )
        qualification = Qwen3_5TP2PP2ServingQualification(
            model_id=model_id,
            inventory_digest=inventory_digest,
            semantic_plan_digest=semantic_plan_digest,
            topology_digest=topology_digest,
            partition_digest=partition_digest,
            max_sequence_tokens=controller.max_sequence_tokens,
            rank_graph_evidence_ids=rank_ids,
            graph_evidence_id=_graph_evidence_id(
                model_id=model_id,
                inventory_digest=inventory_digest,
                semantic_plan_digest=semantic_plan_digest,
                topology_digest=topology_digest,
                partition_digest=partition_digest,
                max_sequence_tokens=controller.max_sequence_tokens,
                rank_graph_evidence_ids=rank_ids,
                graph_buckets=graph_buckets,
            ),
            graph_buckets=graph_buckets,
        )
        return cls(
            controller,
            model_id=model_id,
            weight_variant=WeightVariantMeta(model_id, qualification.inventory_digest),
            eos_token_ids=eos_token_ids,
            qualification=qualification,
        )

    def generate(
        self,
        input_ids: tuple[int, ...],
        max_tokens: int,
        metadata: object | None = None,
    ) -> Iterator[int]:
        if (
            type(input_ids) is not tuple
            or not input_ids
            or any(
                type(token) is not int or not 0 <= token < self._controller.vocab_size
                for token in input_ids
            )
        ):
            raise ValueError("input_ids must be a non-empty immutable tuple in true vocab")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if len(input_ids) + max_tokens - 1 > self._controller.max_sequence_tokens:
            raise ValueError("prompt plus decode exceeds captured context capacity")
        request_id = getattr(metadata, "request_id", None) if metadata is not None else None
        if request_id is None:
            with self._state_lock:
                request_id = f"sync:{self._next_request}"
                self._next_request += 1
        if type(request_id) is not str or not request_id or request_id.strip() != request_id:
            raise ValueError("generation metadata requires a canonical request id")

        sampling = getattr(metadata, "sampling", SamplingMeta())
        ignore_eos = getattr(metadata, "ignore_eos", False)
        if type(ignore_eos) is not bool:
            raise TypeError("performance ignore_eos must be a boolean")
        prepared = getattr(metadata, "prepared_input", None)
        command = _GenerateCommand(
            request_id,
            input_ids,
            max_tokens,
            (
                ()
                if ignore_eos
                and (
                    getattr(metadata, "structured_output", None) is None
                    or metadata.structured_output.kind.value == "identity"
                )
                else self._eos_token_ids
            ),
            resolve_sampling(self._sampling_defaults, sampling),
            *prepared_input_fields(
                input_ids,
                prepared,
            ),
            structured_output=(
                None
                if getattr(metadata, "structured_output", None) is None
                or (metadata.structured_output.kind.value == "identity"
                    and getattr(metadata, "reasoning_control", None) is None)
                else metadata.structured_output
            ),
            constraint_reasoning_end=getattr(metadata, "constraint_reasoning_end", ()),
            reasoning_control=getattr(metadata, "reasoning_control", None),
        )
        stream = _TokenStream(
            lambda source, error: self._release_stream(request_id, source, error),
            invalid_item_error=lambda: Qwen3_5TP2PP2ServingError(
                "token stream received an invalid item"
            ),
        )
        try:
            with self._state_lock:
                if self._closed:
                    raise Qwen3_5TP2PP2ServingError("performance executor is closed")
                if self._poisoned:
                    raise Qwen3_5TP2PP2ServingError("performance executor is poisoned")
                if not self._controller.ready:
                    raise Qwen3_5TP2PP2ServingError("performance topology is unavailable")
                if request_id in self._active:
                    raise Qwen3_5TP2PP2ServingError("request id is already active")
                self._active[request_id] = stream
            self._controller.submit(command, stream)
        except BaseException:
            with self._state_lock:
                self._active.pop(request_id, None)
            raise
        return stream

    def _release_stream(
        self,
        request_id: str,
        stream: _TokenStream,
        error: BaseException | None,
    ) -> None:
        fatal = error is not None and not isinstance(error, Qwen3_5TP2PP2Overloaded)
        with self._state_lock:
            if self._active.get(request_id) is stream:
                del self._active[request_id]
            if fatal and not self._closed:
                self._poisoned = True
        with suppress(Exception):
            self._controller.cancel(request_id)

    def cancel(self, request_id: str) -> None:
        if type(request_id) is not str or not request_id:
            raise ValueError("cancel requires a request id")
        with self._state_lock:
            stream = self._active.get(request_id)
        if stream is not None:
            self._controller.cancel(request_id)
            stream.stop()

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            active = tuple(self._active.items())
        for request_id, stream in active:
            with suppress(Exception):
                self._controller.cancel(request_id)
            stream.stop()
        try:
            close_encoder = getattr(self._input_encoder, "close", None)
            if close_encoder is not None:
                close_encoder()
        finally:
            self._controller.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()


__all__ = (
    "Qwen3_5TP2PP2Overloaded",
    "Qwen3_5TP2PP2PerformanceExecutor",
    "Qwen3_5TP2PP2ServingError",
    "Qwen3_5TP2PP2ServingQualification",
)
