"""Plain/packed weight graph serving over the shared scheduler and replay ABI.

Every startup checks its live graph and state resources. Independent numerical
and cross-M qualification is an explicit development/acceptance operation, not
an implicit production startup workload. Graph-debug remains unqualified.
Checkpoint loading, rank operators, sampling and HTTP semantics stay shared.
"""

from __future__ import annotations

from cubic_serve.executor.text_graph import GraphScratchPool

import gc
import json
import logging
import signal
import threading
import time
import traceback
from contextlib import ExitStack, nullcontext, suppress
from dataclasses import dataclass, replace
from datetime import timedelta
from math import lcm
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

from cubic_serve.ops.cuda.collective import all_reduce, barrier
from cubic_serve.contracts.execution import GroupKind, ParallelTopology
from cubic_serve.executor.draft import TransactionalDraftBinding
from cubic_serve.executor.readiness import execution_ready
from cubic_serve.executor.draft_graph import capture_draft_rounds, DraftPendingPool
from cubic_serve.executor.runner import configure_reference_math
from cubic_serve.generation.sampler import load_sampling_defaults, resolve_sampling
from cubic_serve.guardrail.compare import stable_digest
from cubic_serve.models.builtin import (
    BuiltinModelConfigError,
    builtin_text_config,
    model_draft_provider,
    parse_model_config_json,
    resolve_model_definition,
)
from cubic_serve.models.loader import LoadedModelDefinition
from cubic_serve.models.quantization import QuantizationConfig
from cubic_serve.models.quantized_checkpoint import CheckpointWeights
from cubic_serve.ops.strategies import OperatorConfig
from cubic_serve.quantization.registry import require_compute_mode
from cubic_serve.runtime.cache import (
    CacheArena,
    DeviceMemoryBudget,
    estimate_cache_arena_bytes,
    read_cuda_memory_snapshot,
)
from cubic_serve.runtime.cache.quantization import KVCacheStorageSpec
from cubic_serve.runtime.cache.shared_pool import SharedCacheStorage, SharedCacheField, SharedCacheOwnership
from cubic_serve.runtime.continuous_batch import (
    RESIDENT_STEP_CAPACITY,
    BatchCommand,
    BatchResult,
    CloseCommand,
    ContinueBatchCommand,
    GenerateCommand,
    owned_single_token_rows,
    prepared_input_fields,
    resident_batch_ownership,
    run_draft_continuation,
    run_grammar_draft_continuation,
    run_grammar_continuation,
    run_resident_continuation,
)
from cubic_serve.runtime.logging import (
    diagnostic_event,
    configure_logging,
    startup_graph_progress,
    startup_draft_portfolio,
    startup_graph_capture_started,
    startup_graph_captured,
    startup_graphs_ready,
    startup_sampling_ready,
    startup_sampling_started,
    startup_state_ready,
)
from cubic_serve.runtime.performance import (
    _BLOCK_SIZE,
    _COMMAND_NOTICE,
    CapturedDraftRank,
    Qwen3_5TP2PP2RankCacheCapacity,
    _CapturedGraphRank,
    _ConstraintSamplingArena,
    _constraint_prefix_masks_with_overlap,
    _cold_start_mtp_proposal_costs,
    _cold_start_target_costs,
    _cold_start_target_depth_costs,
    _deserialize_worker_command,
    _effective_max_sequence_tokens,
    _Failure,
    _LoadProgress,
    _median_cost_samples,
    _owner_state_capacity,
    _prepare_rank_greedy_sampling,
    _ProcessController,
    _qualify_draft_verification,
    _rank_weight_progress,
    _Ready,
    _request_grammar_pool,
    _warm_sampling,
    execute_draft_batch,
)
from cubic_serve.runtime.profiling import new_step_profile
from cubic_serve.runtime.quantized import load_quantized_rank_weights
from cubic_serve.runtime.token_chat import TokenGenerationMetadata
from cubic_serve.runtime.token_stream import TokenStream

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Plan:
    topology: ParallelTopology
    bindings: tuple[tuple[Any, ...], ...]

    def bindings_for_rank(self, rank: int) -> tuple[Any, ...]:
        return self.bindings[rank]


def _bootstrap_cache_arenas(schemas, device, rank):
    """Price graphs without a recurrent matrix for every inactive padding row."""
    schemas = tuple(schemas)
    pages, states = SharedCacheStorage.layouts(schemas)
    owners = max(spec.state_slot_capacity for spec, _ in schemas)
    horizon = max(spec.page_capacity for spec, _ in schemas)
    storage = SharedCacheStorage(
        schemas, budget_bytes=horizon * pages.entry_bytes + owners * (
            0 if states is None else states.entry_bytes),
        state_capacity=owners, device=device,
    )
    arenas = tuple(CacheArena(
        storage.arena_specs[index], layout, device, rank_id=rank,
        shared_storage=storage, shared_index=index,
    ) for index, (_, layout) in enumerate(schemas))
    return storage, arenas


def _text_graph_capacities(token_budget: int, max_speculative_tokens: int) -> tuple[int, ...]:
    """Permanent anchors within a token budget, independent of live clients."""
    if type(token_budget) is not int or token_budget <= 0:
        raise ValueError("CUDA graph token budget must be a positive integer")
    if type(max_speculative_tokens) is not int or max_speculative_tokens < 0:
        raise ValueError("draft capacity must be a nonnegative integer")
    capacities = tuple(1 << i for i in range(token_budget.bit_length()))
    if capacities[-1] < max_speculative_tokens + 1:
        raise ValueError("CUDA graph token budget cannot hold one complete draft review")
    return capacities


def _text_graph_owner_capacity(capacities: tuple[int, ...], max_speculative_tokens: int) -> int:
    """One owner cohort must fit every declared complete review depth."""
    roots = capacities[-1] // (max_speculative_tokens + 1)
    if roots < 1:
        raise ValueError("CUDA graph token budget cannot hold one complete draft review")
    # Compound heads are permanent power-of-two graph anchors as well. Do not
    # allocate an unusable fractional cohort or add shallow-only owner cohorts
    # when separating admission from the already established full-depth plan.
    return 1 << (roots.bit_length() - 1)


def _auto_text_graph_capacities(
    available_bytes: int, *, vocabulary_size: int, hidden_size: int,
    element_bytes: int, tp_size: int, bodies: int, max_speculative_tokens: int,
) -> tuple[int, ...]:
    """Bound startup graph rows from live headroom and tensor geometry.

    Allow a quarter of post-weight, post-reserve headroom for graphs. Per body,
    price four compute-type and two FP32 vocabulary planes, plus 32 compute-type
    hidden planes. This conservative sizing prior is not a resource receipt:
    complete portfolio calibration and live cache admission remain mandatory.
    The established 1024-row ceiling and permanent M1 anchor remain unchanged.
    """
    if type(available_bytes) is not int or available_bytes < 0 or any(
        type(n) is not int or n <= 0
        for n in (vocabulary_size, hidden_size, element_bytes, tp_size, bodies)
    ):
        raise ValueError("automatic CUDA graph sizing requires valid byte and tensor geometry")
    local_vocabulary = (vocabulary_size + tp_size - 1) // tp_size
    row_bytes = bodies * (
        local_vocabulary * (4 * element_bytes + 2 * 4) + hidden_size * 32 * element_bytes
    )
    minimum = 1 << max_speculative_tokens.bit_length()
    rows = max(minimum, min(1024, available_bytes // (4 * row_bytes)))
    return _text_graph_capacities(rows, max_speculative_tokens)


def _prepare_capture_inputs(stage, capacities, qualification):
    """One target/provider boundary between live binding and offline corpora."""
    if qualification is None or qualification.recording:
        from cubic_serve.runtime.quantized_qualification import eager_snapshots
        return eager_snapshots(stage, capacities)
    from cubic_serve.runtime.performance import _logical_m_values
    stage.bucket_logical_ms = _logical_m_values(stage.captured_m, capacities)
    stage.prepare_scratch(stage.resource)
    return None


def _log_memory_residency(
    *, rank, device, weights, arena, draft_arena, startup_free_bytes,
    final_free_bytes, reserve_bytes, page_bytes, draft_page_bytes,
    max_running_requests, maximum,
) -> None:
    """Explain the admitted pool from existing ledgers; no device query or copy."""
    storage = weights.storage_bytes
    shared=getattr(arena,"shared_storage",None)
    if shared is not None:
        backing=shared.backing.numel()
        state_entry=0 if shared.state_layout is None else shared.state_layout.entry_bytes
        _LOGGER.info(
            "serve memory rank=%d device=%s weights_device_bytes=%d weights_pinned_host_bytes=%d "
            "weights_pageable_host_bytes=%d shared_cache_backing_bytes=%d other_device_bytes=%d "
            "free_bytes=%d required_reserve_bytes=%d",
            rank,device,storage["device"],storage["pinned_host"],storage["pageable_host"],backing,
            startup_free_bytes-final_free_bytes-storage["device"]-backing,final_free_bytes,reserve_bytes)
        _LOGGER.info(
            "serve shared cache rank=%d pool_bytes=%d bytes_per_page=%d page_tokens=%d "
            "bytes_per_state_bundle=%d padding_bytes=%d max_running_requests=%d "
            "state_owner_slots=%d graph_max_M=%d K=%s V=%s rotation=%s "
            "capacity=KV-only-equivalent; live state and checkpoints borrow from this pool",
            rank,shared.budget_bytes,shared.page_layout.entry_bytes,arena.layout.block_size_tokens,
            state_entry,backing-shared.budget_bytes,max_running_requests,shared.state_capacity,maximum,
            arena.layout.storage.key.dtype.value,arena.layout.storage.value.dtype.value,
            arena.layout.storage.enable_rotation)
        return
    arenas = (arena,) if draft_arena is None else (arena, draft_arena)
    state_bytes = sum(t.numel() * t.element_size()
                      for item in arenas for t in item.states.values())
    cache_bytes = sum(item.backed_bytes for item in arenas)
    other_bytes = startup_free_bytes - final_free_bytes - storage["device"] - cache_bytes
    _LOGGER.info(
        "serve memory rank=%d device=%s weights_device_bytes=%d "
        "weights_pinned_host_bytes=%d weights_pageable_host_bytes=%d "
        "kv_page_backing_bytes=%d state_bytes=%d other_device_bytes=%d "
        "free_bytes=%d required_reserve_bytes=%d",
        rank, device, storage["device"], storage["pinned_host"], storage["pageable_host"],
        cache_bytes - state_bytes, state_bytes, other_bytes, final_free_bytes, reserve_bytes,
    )
    _LOGGER.info(
        "serve cache storage rank=%d K=%s V=%s rotation=%s page_tokens=%d "
        "target_bytes_per_token=%.3f draft_bytes_per_token=%.3f "
        "max_running_requests=%d graph_max_M=%d state_owner_slots=%d "
        "target_scratch_slots=%d draft_scratch_slots=%d",
        rank, arena.layout.storage.key.dtype.value, arena.layout.storage.value.dtype.value,
        arena.layout.storage.enable_rotation, arena.layout.block_size_tokens,
        page_bytes / arena.layout.block_size_tokens,
        draft_page_bytes / arena.layout.block_size_tokens,
        max_running_requests, maximum, arena.spec.state_slot_capacity,
        arena.spec.scratch_capacity * len(arena.spec.scratch_arena_ids),
        0 if draft_arena is None else
        draft_arena.spec.scratch_capacity * len(draft_arena.spec.scratch_arena_ids),
    )


def _worker(
    rank: int,
    init_method: str,
    admission: Any,
    commands: Any,
    shared_commands: Any,
    shared_capacity: int,
    results: Any,
    timeout: int,
    audit_statuses: Any,
    audit_barrier: Any,
    continuation_interrupt: Any = None,
) -> None:
    configure_logging()
    from cubic_serve.executor.resident_window import (
        prepare_resident_communication,
        prepare_resident_compactions,
    )

    prepare_resident_communication()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})
    # The controller already supplies one process barrier for this local
    # topology. Packed workers do not use it during startup qualification.
    # Reuse it for completion after sampling's host validity check and the
    # model stream synchronization; every rank must arrive before publishing
    # a result that lets the controller release the topology-wide lease.
    completion_barrier = audit_barrier
    stages: list[Any] = []
    startup_lifetime = ExitStack()
    sampling_lifetime = ExitStack()
    profile_sampling_lifetime = ExitStack()
    encoder_thread = None
    owned_encoder = None
    weights = None
    draft_provider = getattr(admission, "draft_provider", None)
    feeding_profile = None
    from cubic_serve.executor.vision import (
        finish_owned_input_encoder,
        prepare_owned_input_encoder,
        publish_owned_input_encoder,
        start_owned_input_encoder,
    )
    try:
        import os
        if not any(name in os.environ for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")):
            torch.set_num_threads(4)
        configure_reference_math()
        definition_source = getattr(admission, "model_definition_source", None)
        definition = resolve_model_definition(
            admission.config,
            definition=None if definition_source is None else definition_source.definition,
        )
        if (definition_source is not None
                and definition_source.execution_recipe(admission.config)
                != admission.model_program_recipe):
            raise ValueError("worker model execution recipe differs from controller admission")
        phase_started = time.perf_counter()
        topology = admission.parallel_plan.topology
        placement = topology.placements[rank]
        device = torch.device("cuda", placement.device_index)
        torch.cuda.set_device(device)
        memory_budget = admission.device_memory_budgets.get(placement.device_index)
        if memory_budget is None:
            memory_budget = DeviceMemoryBudget(read_cuda_memory_snapshot(device))
        dist.init_process_group(
            "nccl",
            init_method=init_method,
            rank=rank,
            world_size=len(topology.placements),
            device_id=device if len(topology.placements) > 1 else None,
            timeout=timedelta(seconds=timeout),
        )
        groups: dict[str, Any] = {}
        for stage_index in range(topology.pp_size):
            ranks = [p.rank for p in topology.placements if p.pipeline_stage == stage_index]
            group = dist.new_group(ranks)
            if rank in ranks:
                groups["sampling_tp"] = group
                if topology.tp_size > 1:
                    groups["tp"] = group
        for lane in range(topology.tp_size):
            ranks = [p.rank for p in topology.placements if p.tensor_rank == lane]
            if len(ranks) > 1:
                group = dist.new_group(ranks)
                if rank in ranks:
                    groups["pp"] = group
        if 1 < topology.ep_size < topology.tp_size:
            for item in topology.groups:
                if item.kind in (GroupKind.EXPERT, GroupKind.EXPERT_TENSOR):
                    group = dist.new_group(list(item.ranks))
                    if rank in item.ranks:
                        groups["ep" if item.kind is GroupKind.EXPERT else "etp"] = group
        diagnostic_event(_LOGGER, 'quantized_graph.worker_cuda_communication_initialized_rank_d_duration',
            rank,
            time.perf_counter() - phase_started,
        )
        from cubic_serve.runtime.quantized_qualification import execution_identity, StartupQualification
        from cubic_serve.runtime.preparation import StartupPreparation
        from cubic_serve.ops.cuda.startup import startup_weight_identity
        identity = execution_identity(admission)
        engineering = getattr(admission, "startup_qualification", False) or admission.startup_numerical_audit
        qualification = ((None if engineering else StartupPreparation(rank, device,
                              topology=(topology.tp_size, topology.pp_size, topology.ep_size)))
                         if admission.execution_mode == "cuda" else None)
        # A first bucket may load a cost model later reused without file reads.
        # Include those inputs as well as plans read by the temporary portfolio.
        from cubic_serve.ops.cuda.launch_cost import startup_plan_dependencies
        memory_dependency_scope = ExitStack()
        startup_lifetime.callback(memory_dependency_scope.close)
        memory_dependencies = memory_dependency_scope.enter_context(startup_plan_dependencies())
        # Numerical evidence must describe the production-selected operators.
        # Candidate timing belongs to a separate explicit qualification pass.
        startup_lifetime.enter_context(startup_weight_identity((
            identity.inventory.inventory_digest, identity.parallel_plan.weight_plan_digest,
            identity.sealed.topology_digest, identity.sealed.partition_digest, rank,
            admission.operator_config.digest, str(admission.mamba_ssm_cache_dtype),
        ), allow_calibration=(not admission.startup_numerical_audit and (
            admission.execution_mode != "cuda" or engineering))))
        phase_started = time.perf_counter()
        weight_loading_started = phase_started
        memory_budget.apply_allocator_limit()
        weights = load_quantized_rank_weights(
            admission.config, admission.checkpoint, topology, rank,
            owner_bindings=admission.parallel_plan.bindings_for_rank(rank),
            include_mtp=draft_provider is not None,
            progress=_rank_weight_progress(rank, results, unit="tensors"),
            _overlap_loading=True,
        )
        # Its communicator is already independent. Overlap CPU language reads,
        # while excluding loader CUDA work from visual capture and calibration.
        from cubic_serve.ops.cuda.startup import startup_cuda_lock
        with startup_cuda_lock(device):
            owned_encoder = prepare_owned_input_encoder(
                admission, topology=topology, rank=rank, device=device)
        input_encoder_admission = None
        # Engineering timing must exclude reader CPU/host traffic. Ordinary
        # startup restores plans without searching/timing candidates, so its
        # private constructor can consume ready operands while the reader runs.
        # Each published device tensor has a stream event; packed operands are
        # already materialized on CPU. The unconditional join below still
        # precedes every model forward, reference operation and graph capture.
        if qualification is None or qualification.engineering:
            weights.finish_loading()
        phase_started = time.perf_counter()
        if rank == 0:
            diagnostic_event(_LOGGER, 'quantized_graph.preparing_kernels_and_measuring_graph_memory')
        graph_factory = definition.text_graph
        if graph_factory is None:
            raise ValueError("model has no text graph construction binding")
        graph_definition = graph_factory(
            admission.config, topology, block_size=admission.block_size_tokens,
            recurrent_state_dtype=admission.mamba_ssm_cache_dtype,
            **({"definition": definition} if definition_source is not None else {}),
        )
        typed_stage = graph_definition.named_state_fields
        stage_type = graph_definition.stage_type
        options = dict(
            tp_group=groups.get("tp"),
            pp_group=groups.get("pp"),
            ep_group=groups.get("ep"),
            expert_tp_group=groups.get("etp"),
            page_size=admission.block_size_tokens,
            operator_config=admission.operator_config,
        )
        if getattr(admission, "cache_storage", None) is not None:
            options["cache_storage"] = admission.cache_storage
        # Duplicate names fail before construction: model options cannot replace
        # engine-owned collectives, operator policy or page ownership.
        options = dict(**options, **graph_definition.stage_options)
        try:
            probe = stage_type(admission.config, weights, topology, captured_m=1, **options)
            draft_probe = (None if draft_provider is None else draft_provider.build_stage(
                weights, topology, captured_m=1, **options,
            ))
        except BaseException:
            weights.finish_loading(cancel=True)
            raise
        # Construction above may consume private pending operands. No model
        # forward, reference check or capture is permitted before this join:
        # it proves final source identity and completes every loader stream.
        weights.finish_loading()
        weights_loaded_at = weights.loading_finished_at
        diagnostic_event(_LOGGER, 'quantized_graph.weight_loading_complete_rank_d_duration_s',
            rank,
            weights_loaded_at - weight_loading_started,
            weights.loading_wait_seconds,
        )
        if getattr(admission, "auto_graph_max_tokens", False):
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info(device)
            available = max(0, memory_budget.remaining_bytes(free)
                            - admission.input_encoder_reserves.get(placement.device_index, 0))
            text = builtin_text_config(admission.config)
            capacities = _auto_text_graph_capacities(
                available, vocabulary_size=text.vocabulary_size, hidden_size=text.hidden_size,
                element_bytes=probe.hidden.element_size(), tp_size=topology.tp_size,
                bodies=1 + int(draft_provider is not None),
                max_speculative_tokens=admission.max_speculative_tokens,
            )
            common_limit = torch.tensor([capacities[-1]], dtype=torch.int64, device=device)
            all_reduce(common_limit, op=dist.ReduceOp.MIN)
            admission.graph_capacities = _text_graph_capacities(
                int(common_limit.item()), admission.max_speculative_tokens)
            admission.graph_request_capacity = _text_graph_owner_capacity(
                admission.graph_capacities, admission.max_speculative_tokens)
            if rank == 0:
                _LOGGER.info("CUDA graph token limit: %d (auto)", admission.graph_capacities[-1])
        if engineering and admission.execution_mode == "cuda":
            qualification = StartupQualification.for_admission(admission, rank, device)
        numerical_witnesses: dict[str, Any] = {}
        reference_math_memo = None
        if admission.startup_numerical_audit:
            from cubic_serve.runtime.quantized_qualification import warm_reference_runtime

            warm_reference_runtime(device, probe.hidden.shape[-1])
            if typed_stage:
                from cubic_serve.guardrail.quantization import ReferenceMathMemo
                from cubic_serve.guardrail.quantized_graph.text import audit_original as audit_text
                from cubic_serve.runtime.text_qualification import state_audit_horizon

                reference_math_memo = ReferenceMathMemo()

                numerical_witnesses["typed_m1"] = audit_text(
                    probe, admission, groups, diagnose_independent_trajectory=True,
                    reference_math_memo=reference_math_memo,
                )
                if draft_probe is not None:
                    draft_steps = state_audit_horizon(
                        graph_definition.state_audit_steps, admission.max_sequence_tokens,
                        draft_probe.arena.layout.storage,
                    )
                    width = draft_provider.spec.feedback_width
                    feedback = (torch.arange(width, device=device).remainder(127).float()
                                .sub_(63).div_(32).to(torch.bfloat16).reshape(1, 1, width))
                    numerical_witnesses["draft_m1"] = audit_text(
                        draft_probe, admission, groups, diagnose_independent_trajectory=True,
                        reference_math_memo=reference_math_memo,
                        draft_provider=draft_provider,
                        feedback_rows=tuple(feedback.clone() for _ in range(draft_steps)),
                        column_completion_names=draft_provider.reference_columns,
                        feedback_recurrence=True,
                    )
                    if not numerical_witnesses["draft_m1"][
                        "cross_step_error_propagation_qualified"
                    ]:
                        raise RuntimeError(
                            "independent draft state propagation audit is incomplete")
                # Exercise the existing graph with input features and compact
                # three-axis positions, across two index compression groups.
                count = state_audit_horizon(
                    graph_definition.state_audit_steps, admission.max_sequence_tokens,
                    probe.arena.layout.storage,
                )
                text_config = builtin_text_config(admission.config)
                feature = (torch.arange(text_config.hidden_size).remainder(31)
                           .float().sub_(15).div_(16).to(torch.bfloat16))
                feature_bytes = feature.view(torch.uint8).numpy().tobytes()
                feature_rows = tuple(feature_bytes if 1 <= i < count - 2 else b""
                                     for i in range(count))
                tokens = tuple(admission.config.modality_tokens.image if row else
                               (17 + i * 7919) % text_config.vocabulary_size
                               for i, row in enumerate(feature_rows))
                axes = tuple((i // 2, i // 2, (i + 1) // 2) for i in range(count))
                numerical_witnesses["typed_visual_m1"] = audit_text(
                    probe, admission, groups, token_ids=tokens,
                    input_feature_rows=feature_rows, position_axes=axes,
                    diagnose_independent_trajectory=True,
                    reference_math_memo=reference_math_memo,
                )
                if not all(
                    numerical_witnesses[name]["cross_step_error_propagation_qualified"]
                    for name in ("typed_m1", "typed_visual_m1")
                ):
                    raise RuntimeError("independent typed state propagation audit is incomplete")
                diagnostic_event(_LOGGER, 'quantized_graph.operator_local_text_numerical_checks_complete_rank',
                    rank,
                )
        layout, spec, tables = probe.arena.layout, probe.arena.spec, probe.mrope_tables
        maximum = admission.graph_capacities[-1]
        spec = replace(
            spec,
            state_slot_capacity=_owner_state_capacity(
                # These owners exist only while pricing the graph portfolio.
                # Final owners borrow shared bytes; neither allocation is a
                # reservation proportional to the public request ceiling.
                admission.graph_request_capacity if draft_provider is not None else 1,
                admission.graph_capacities,
                max_speculative_tokens=admission.max_speculative_tokens,
                use_prefix_cache=admission.use_prefix_cache,
            ),
            scratch_capacity=maximum,
            page_capacity=1,
        )
        page_bytes = (
            estimate_cache_arena_bytes(layout, replace(spec, page_capacity=1025))
            - estimate_cache_arena_bytes(layout, spec)
        ) // 1024
        state_bytes = sum(t[0].numel() * t.element_size() for t in probe.arena.states.values())
        draft_layout, draft_spec, draft_page_bytes = None, None, 0
        if draft_probe is not None:
            draft_layout = draft_probe.arena.layout
            draft_spec = replace(
                draft_probe.arena.spec, state_slot_capacity=spec.state_slot_capacity,
                # A mixed transaction can retain M checkpoints while a compact
                # provider replay needs up to M-1 separate padding rows. Price
                # both live ranges before KV admission; the ordinary binder
                # still excludes every reserved checkpoint from padding.
                scratch_capacity=2 * maximum, page_capacity=1,
            )
            draft_page_bytes = (
                estimate_cache_arena_bytes(draft_layout, replace(draft_spec, page_capacity=1025))
                - estimate_cache_arena_bytes(draft_layout, draft_spec)
            ) // 1024
            state_bytes += sum(t[0].numel() * t.element_size()
                               for t in draft_probe.arena.states.values())
        # The resource cache binds construction inputs and native build sources;
        # it carries measured byte sizes, never qualification or live addresses.
        memory_plan_key = memory_plan_implementation = None
        if qualification is not None and not qualification.engineering:
            import os
            import sys
            import triton
            from cuda.bindings import driver
            from cubic_serve.contracts.implementation import implementation_sources
            from cubic_serve.ops.cuda.native import _check
            from cubic_serve.runtime.quantized_qualification import _bound_composition_plan

            memory_plan_implementation = implementation_sources(
                "cubic_serve.runtime.quantized_graph", type(probe).__module__,
                *((type(draft_probe).__module__,) if draft_probe is not None else ()),
            )
            native_manifest = Path(__file__).parents[1] / "_native" / "manifest.json"
            memory_plan_key = stable_digest((
                "complete-graph-memory-v1",
                identity.inventory.inventory_digest,
                identity.parallel_plan.weight_plan_digest,
                identity.parallel_plan.semantic_plan_digest,
                identity.sealed.topology_digest, identity.sealed.partition_digest,
                admission.operator_config.digest,
                repr((layout, spec, draft_layout, draft_spec)),
                admission.graph_capacities, admission.graph_request_capacity,
                admission.max_sequence_tokens, admission.max_speculative_tokens,
                admission.dynamic_speculation, admission.use_prefix_cache,
                tuple(getattr(admission, "reasoning_token_boundaries", ())),
                tuple(getattr(admission, "constraint_stop_token_ids", ())),
                _bound_composition_plan(probe, kernels_only=True),
                None if draft_probe is None else _bound_composition_plan(draft_probe, kernels_only=True),
                (sys.version, str(torch.__version__), str(torch.version.cuda), triton.__version__,
                 int(_check(driver.cuDriverGetVersion())[0]),
                 torch.cuda.memory.get_allocator_backend(),
                 os.environ.get("PYTORCH_ALLOC_CONF", ""),
                 os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
                 os.environ.get("CUDA_MODULE_LOADING", "")),
                native_manifest.read_text() if native_manifest.exists() else None,
            ))
        memory_dependency_scope.close()
        del draft_probe
        del probe
        gc.collect()
        torch.cuda.empty_cache()
        # Price the complete portfolio before giving memory to KV. M1 does
        # not represent larger expert workspaces, graph pools or row staging.
        # Upper context-width tables make the measurement conservative when
        # automatic context sizing subsequently chooses a smaller capacity.
        diagnostic_event(_LOGGER, 'quantized_graph.kernel_preparation_complete_rank_d_post_weight',
            rank,
            time.perf_counter() - weights_loaded_at,
            time.perf_counter() - weight_loading_started,
        )
        phase_started = time.perf_counter()
        page_size = layout.block_size_tokens
        wanted_pages = (admission.max_sequence_tokens + page_size - 1) // page_size
        # A page/state pool aliases potential byte grids. Its final backing
        # replaces both bootstrap classes, so use the measured portfolio path.
        stable_backing = False
        shared_storage = None
        memory_capacities = (1, *sorted((m for m in admission.graph_capacities if m != 1), reverse=True))
        runtime = SimpleNamespace(_groups={f"tp:{placement.pipeline_stage}": groups["sampling_tp"]})
        if stable_backing:
            free, total = torch.cuda.mem_get_info(device)
            input_reserve = admission.input_encoder_reserves.get(placement.device_index, 0)
            reserve = memory_budget.withheld_bytes + input_reserve
            combined_page_bytes = page_bytes + draft_page_bytes
            upper = min(total // combined_page_bytes,
                        admission.max_running_requests * wanted_pages + 2 * maximum)
            common = torch.tensor([upper], dtype=torch.int64, device=device)
            all_reduce(common, op=dist.ReduceOp.MIN)
            reserved_pages = int(common.item())
            resident_pages = (0 if draft_provider is not None else
                              (RESIDENT_STEP_CAPACITY + page_size - 1) // page_size)
            page_capacity = max(2 * maximum + 1, (1 + resident_pages) * maximum + 1)
            if reserved_pages < page_capacity:
                raise ValueError("insufficient memory for the qualified startup page horizon")
            max_context = min(admission.max_sequence_tokens,
                              (reserved_pages - 2 * maximum) * page_size)
            if not admission.auto_max_sequence_tokens and max_context != admission.max_sequence_tokens:
                raise ValueError("requested context exceeds the topology BF16 KV/state capacity")
            admission.max_sequence_tokens = max_context
            table_pages = (max_context + page_size - 1) // page_size
            # KV consumers execute on their owning rank. Collective activation
            # buffers have separate ownership; granting them peer access does
            # not require peer mappings for this rank-private cache storage.
            arena = CacheArena(replace(spec, page_capacity=page_capacity,
                                      reserved_page_capacity=reserved_pages),
                               layout, device, rank_id=rank)
            draft_arena = (None if draft_spec is None else CacheArena(
                replace(draft_spec, page_capacity=page_capacity,
                        reserved_page_capacity=reserved_pages),
                draft_layout, device, rank_id=rank))
            feedback_shape = (None if draft_provider is None else
                              (spec.state_slot_capacity + maximum, draft_provider.spec.feedback_width))
            target_hidden_slots = (None if feedback_shape is None else
                                   torch.zeros(feedback_shape, dtype=torch.bfloat16, device=device))
            overlap_plans = {}
            fixed_backing_identity = (arena.backing_identity,
                                      None if draft_arena is None else draft_arena.backing_identity)
            torch.cuda.reset_peak_memory_stats(device)
            diagnostic_event(_LOGGER, 'quantized_graph.fixed_address_kv_bootstrap_rank_d_admitted',
                rank,
                page_capacity,
                reserved_pages,
            )
        else:
            def measure_graph_memory():
                measurement_started = time.perf_counter()
                preparation_free, _ = torch.cuda.mem_get_info(device)
                diagnostic_event(_LOGGER, 'quantized_graph.shared_cache_measured_startup',
                    rank,
                )
                # Pricing captures ownerless bodies and binds resident windows
                # with qualify_resident=False. Their full journal is metadata,
                # not live KV. Final arenas own all request-horizon proofs.
                bootstrap_spec = replace(spec, page_capacity=maximum + 1,
                                         state_slot_capacity=1)
                draft_bootstrap_spec = (None if draft_spec is None else replace(
                    draft_spec, page_capacity=bootstrap_spec.page_capacity,
                    state_slot_capacity=bootstrap_spec.state_slot_capacity,
                ))
                bootstrap_schemas = (((bootstrap_spec, layout),) if draft_spec is None else (
                    (bootstrap_spec, layout), (draft_bootstrap_spec, draft_layout)))
                bootstrap_storage, bootstrap_arenas = _bootstrap_cache_arenas(
                    bootstrap_schemas, device, rank)
                bootstrap = bootstrap_arenas[0]
                draft_bootstrap = None if draft_spec is None else bootstrap_arenas[1]
                bootstrap_spec = bootstrap.spec
                draft_bootstrap_spec = None if draft_bootstrap is None else draft_bootstrap.spec
                bootstrap_backing_bytes = bootstrap_storage.backing.numel()
                _LOGGER.info(
                    "graph bootstrap rank=%d backing_bytes=%d owner_slots=%d "
                    "padding_states_per_arena=%d graph_max_M=%d",
                    rank, bootstrap_backing_bytes, bootstrap_spec.state_slot_capacity,
                    bootstrap_spec.state_scratch_capacity_per_arena, maximum,
                )
                bootstrap_state_rows = (bootstrap_spec.state_slot_capacity
                    + len(bootstrap_spec.scratch_arena_ids)
                    * bootstrap_spec.state_scratch_capacity_per_arena)
                feedback_shape = (bootstrap_state_rows, draft_provider.spec.feedback_width
                                  ) if draft_provider is not None else None
                profile_feedback = (None if feedback_shape is None else torch.zeros(
                    feedback_shape, dtype=torch.bfloat16, device=device,
                ))
                feedback_bytes = 0 if profile_feedback is None else (
                    profile_feedback.numel() * profile_feedback.element_size())
                runtime = SimpleNamespace(_groups={f"tp:{placement.pipeline_stage}": groups["sampling_tp"]})
                profiled_executors = []
                profile_targets, profile_providers = {}, {}
                profile_constraint_arena = None
                overlap_plans = {}
                transient_bytes = 0
                # Keep the permanent M1 boundary first, then calibrate the largest
                # shapes before smaller resident buckets consume their cold headroom.
                # Use the same lifetime order with and without optional auditing.
                memory_capacities = (
                    1, *sorted((m for m in admission.graph_capacities if m != 1), reverse=True),
                )
                target_scratch = GraphScratchPool(capacities=admission.graph_capacities)
                provider_scratch = target_scratch.sibling()
                for profile_index, capacity in enumerate(memory_capacities, start=1):
                    torch.cuda.reset_peak_memory_stats(device)
                    build_started = time.perf_counter()
                    stage = stage_type(
                        admission.config, weights, topology, captured_m=capacity,
                        table_pages=wanted_pages, arena=bootstrap, mrope_tables=tables, **options,
                        shared_scratch=target_scratch,
                    )
                    build_seconds = time.perf_counter() - build_started
                    capture_started = time.perf_counter()
                    stages.append(stage)
                    stage.capture()
                    if not typed_stage:
                        stage.prepare_decode_overlap()
                        overlap_plans[capacity] = stage.overlap_plan
                    profiled_executors.append(
                        _CapturedGraphRank(stage, stage.resource, stage, runtime, admission,
                                           profile_feedback)
                    )
                    if profile_constraint_arena is None:
                        profile_constraint_arena = _ConstraintSamplingArena(
                            profiled_executors[-1], maximum, qualification=qualification,
                            consume_logits=draft_provider is None,
                        )
                    _prepare_rank_greedy_sampling(
                        profiled_executors[-1], profile_sampling_lifetime, qualify_resident=False,
                        constraint_arena=profile_constraint_arena, qualification=qualification,
                    )
                    profile_targets[capacity] = profiled_executors[-1]
                    if draft_provider is not None:
                        draft_stage = draft_provider.build_stage(
                            weights, topology, captured_m=capacity, table_pages=wanted_pages,
                            arena=draft_bootstrap, mrope_tables=tables, **options,
                            shared_scratch=provider_scratch,
                        )
                        stages.append(draft_stage)
                        draft_stage.capture()
                        binding = TransactionalDraftBinding(_CapturedGraphRank(
                            draft_stage, draft_stage.resource, draft_stage, runtime, admission,
                            private_state_sources=True,
                        ))
                        profiled_executors.append(CapturedDraftRank(
                            draft_stage, draft_stage, runtime, admission,
                            request_page_capacity=bootstrap_spec.page_capacity,
                            target_hidden_slots=profile_feedback, binding=binding,
                        ))
                        _prepare_rank_greedy_sampling(
                            profiled_executors[-1], profile_sampling_lifetime, qualify_resident=False,
                            constraint_arena=profile_constraint_arena, qualification=qualification,
                        )
                        profile_providers[capacity] = profiled_executors[-1]
                    torch.cuda.synchronize(device)
                    transient_bytes = max(
                        transient_bytes,
                        torch.cuda.max_memory_allocated(device) - torch.cuda.memory_allocated(device),
                    )
                    diagnostic_event(_LOGGER, 'quantized_graph.graph_memory_measurement_rank_d_m_d',
                        rank,
                        capacity,
                        build_seconds,
                        time.perf_counter() - capture_started,
                    )
                    results.send(_LoadProgress(
                        rank, "graph_measurement", profile_index, len(memory_capacities), "buckets",
                    ))
                prepare_resident_compactions(profile_targets, profile_sampling_lifetime, qualification=qualification)
                profile_rounds = capture_draft_rounds(
                    profile_providers, profile_targets, dynamic_depth=admission.dynamic_speculation,
                    lifetime=profile_sampling_lifetime, qualification=qualification,
                    enforce_capacity=False, pricing_owner_capacity=spec.state_slot_capacity,
                    progress=startup_graph_progress(rank, "Calibrating MTP graph memory"))
                profiled_round_keys=frozenset(profile_rounds)
                extra_state_fields = (() if profile_feedback is None else (
                    SharedCacheField("feedback",profile_feedback.dtype,tuple(profile_feedback.shape[1:])),))
                pending_profile_bytes = 0
                if profile_providers:
                    pending_provider = profile_providers[min(profile_providers)]
                    pending_fields = DraftPendingPool.fields(pending_provider)
                    pending_profile_bytes = sum(f.nbytes for f in pending_fields) * spec.state_slot_capacity
                    extra_state_fields += pending_fields
                    del pending_provider
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                diagnostic_event(_LOGGER, 'quantized_graph.graph_memory_measurement_complete_rank_d_duration',
                    rank,
                    time.perf_counter() - measurement_started,
                )
                portfolio_free, _ = torch.cuda.mem_get_info(device)
                portfolio_allocated = torch.cuda.memory_allocated(device)
                allocator_slack = max(
                    0, torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
                )
                profile_sampling_lifetime.close()
                for stage in stages:
                    stage.close()
                profiled_executors.clear()
                profile_rounds.clear()
                profile_targets.clear()
                profile_providers.clear()
                profile_constraint_arena = None
                stages.clear()
                target_scratch.clear()
                provider_scratch.clear()
                del stage, bootstrap, profile_feedback, bootstrap_arenas, bootstrap_storage
                if draft_provider is not None:
                    del draft_stage, binding, draft_bootstrap
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
                free, total = torch.cuda.mem_get_info(device)
                released_active_bytes = max(
                    0, portfolio_allocated - torch.cuda.memory_allocated(device),
                )
                graph_bytes = max(
                    0, free - portfolio_free - bootstrap_backing_bytes
                    - feedback_bytes - pending_profile_bytes
                )
                diagnostic_event(_LOGGER, 'quantized_graph.graph_memory_budget_rank_d_resident_f',
                    rank,
                    graph_bytes / (1 << 20),
                    transient_bytes / (1 << 20),
                    allocator_slack / (1 << 20),
                    released_active_bytes / (1 << 20),
                    (free - portfolio_free) / (1 << 20),
                )
                return dict(
                    graph_bytes=graph_bytes, transient_bytes=transient_bytes,
                    allocator_slack=allocator_slack,
                    preparation_bytes=max(0, preparation_free - free),
                    extra_state_fields=tuple((field.name, str(field.dtype).removeprefix("torch."), field.shape)
                                             for field in extra_state_fields),
                    overlap_plans={m: (plan.parts, plan.estimated_linear, plan.collective_threads)
                                   for m, plan in overlap_plans.items()},
                    round_keys=tuple(sorted(profiled_round_keys)),
                )

            measurement = (measure_graph_memory() if memory_plan_key is None else
                qualification.memory_plan(memory_plan_key, measure_graph_memory,
                                          implementation=memory_plan_implementation,
                                          dependencies=memory_dependencies))
            graph_bytes = measurement["graph_bytes"]
            if memory_plan_key is not None and qualification.memory_plan_reused:
                # Cold profiling also materializes retained format preparations
                # and modules. They are absent on a warm restore until final
                # construction, so reserve those measured bytes before KV too.
                graph_bytes += measurement["preparation_bytes"]
            transient_bytes = measurement["transient_bytes"]
            allocator_slack = measurement["allocator_slack"]
            extra_state_fields = tuple(SharedCacheField(name, getattr(torch, dtype), shape)
                                       for name, dtype, shape in measurement["extra_state_fields"])
            from cubic_serve.executor.decode_overlap import DecodeOverlapPlan
            overlap_plans = {m: DecodeOverlapPlan(*values)
                             for m, values in measurement["overlap_plans"].items()}
            profiled_round_keys = frozenset(measurement["round_keys"])
            phase_started = time.perf_counter()
            # A saved size is not live admission: measure this process's free VRAM.
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info(device)
            # Retain the measured replacement margin until the complete final
            # portfolio closes. Removing it missed the live runtime reserve;
            # it is measured memory overhead, never a request-sized state slab.
            graph_bytes += allocator_slack
            reserve = memory_budget.withheld_bytes
            # Input encoders can execute in the frontend process while language
            # requests decode. Their unallocated staging capacity is additional to
            # the language runtime reserve, and follows physical device ownership.
            schemas = ((spec,layout),) if draft_spec is None else ((spec,layout),(draft_spec,draft_layout))
            page_layout,state_layout = SharedCacheStorage.layouts(schemas,extra_state_fields)
            encoder = None if owned_encoder is None else owned_encoder[0]
            auto_encoder = encoder is not None and getattr(encoder, "auto_memory", False)
            encoder_plan = ((encoder.working_set, encoder.max_input_rows,
                             encoder.resources.staging_bytes) if auto_encoder else
                            getattr(admission, "input_encoder_working_sets", {}).get(
                                placement.device_index))
            media_available = SharedCacheStorage.budget_from_available(
                min(free, memory_budget.startup.free_bytes) - reserve -
                graph_bytes - transient_bytes, schemas, extra_state_fields)
            media_owner_states = (1 + admission.max_speculative_tokens +
                                  int(admission.use_prefix_cache))
            if encoder_plan is not None:
                plan, minimum_rows, staging_bytes = encoder_plan
            merge_extent = torch.tensor(0 if encoder_plan is None else plan.merge_extent,
                                        dtype=torch.int64, device=device)
            all_reduce(merge_extent, op=dist.ReduceOp.MAX)
            media_merge_extent = int(merge_extent.item())

            def admit_media(available_bytes):
                nonlocal input_encoder_admission
                encoder_rows = (1 << 62)
                owner_bytes = (0 if state_layout is None else
                               media_owner_states * state_layout.entry_bytes)
                if encoder_plan is not None:
                    encoder_rows = plan.fit(
                        available_bytes, minimum_rows=minimum_rows, staging_bytes=staging_bytes,
                        cache_page_bytes=page_layout.entry_bytes, cache_page_tokens=page_size,
                        additional_tokens=maximum, owner_bytes=owner_bytes,
                        working_alignment_bytes=page_layout.entry_bytes)
                elif media_merge_extent:
                    # Stages without an encoder still need KV for its features.
                    encoder_rows = (((available_bytes - owner_bytes) // page_layout.entry_bytes
                                     * page_size - maximum) * media_merge_extent)
                agreed_rows = torch.tensor(encoder_rows, dtype=torch.int64, device=device)
                all_reduce(agreed_rows, op=dist.ReduceOp.MIN)
                if encoder_plan is not None:
                    rows = min(int(agreed_rows.item()), plan.declared_rows)
                    rows = rows // plan.merge_extent * plan.merge_extent
                    if rows < minimum_rows:
                        raise ValueError("insufficient topology cache for the minimum media input")
                    if auto_encoder:
                        rows = min(rows, encoder.max_dynamic_input_rows)
                        encoder.max_dynamic_input_rows = rows
                        encoder.peak_input_bytes = max(staging_bytes, plan.transient_bytes(rows))
                        encoder.resources.total_bytes = (encoder.resources.resident_bytes +
                                                         encoder.peak_input_bytes)
                    peak = max(staging_bytes, plan.transient_bytes(rows))
                    admission.input_encoder_reserves[placement.device_index] = peak
                    input_encoder_admission = (placement.device_index, rows, peak)

            admit_media(media_available)
            reserve += admission.input_encoder_reserves.get(placement.device_index, 0)
            pool_budget = SharedCacheStorage.budget_from_available(
                min(free,memory_budget.startup.free_bytes)-reserve-graph_bytes-transient_bytes,
                schemas,extra_state_fields)
            combined_page_bytes = page_layout.entry_bytes
            state_bytes = 0 if state_layout is None else state_layout.entry_bytes
            common = torch.empty(2,dtype=torch.int64,device=device)
            allocation_correction = 0
            while True:
                if pool_budget < combined_page_bytes or state_bytes > pool_budget:
                    raise ValueError("insufficient shared KV/state byte capacity")
                common[0] = pool_budget // combined_page_bytes
                common[1] = pool_budget // (state_bytes or combined_page_bytes)
                all_reduce(common,op=dist.ReduceOp.MIN)
                page_capacity, state_capacity = map(int,common.tolist())
                # Physical graph rows include padding and tokens from the same
                # owner. They are not a minimum number of live KV owners.
                # Private graph pages were already excluded from pool_budget.
                if page_capacity <= 0 or state_capacity <= 0:
                    raise ValueError("insufficient shared cache capacity for graph ownership")
                shared_storage = SharedCacheStorage(
                    schemas,budget_bytes=pool_budget,device=device,state_capacity=state_capacity,
                    extra_state_fields=extra_state_fields)
                arena = CacheArena(shared_storage.arena_specs[0],layout,device,rank_id=rank,
                                   shared_storage=shared_storage,shared_index=0)
                draft_arena = (None if draft_spec is None else CacheArena(
                    shared_storage.arena_specs[1],draft_layout,device,rank_id=rank,
                    shared_storage=shared_storage,shared_index=1))
                target_hidden_slots = shared_storage.extra_states.get("feedback")
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                arena_free,_ = torch.cuda.mem_get_info(device)
                deficit = max(0,reserve+graph_bytes+transient_bytes-arena_free)
                rounding = max(0,free-arena_free-shared_storage.backing.numel())
                common[0] = deficit + rounding if deficit else 0
                all_reduce(common[:1],op=dist.ReduceOp.MAX)
                correction = int(common[0].item())
                if not correction:
                    break
                # Reconcile media and KV against actual allocator plateaus.
                # Every correction decreases their total admitted byte envelope.
                arena=draft_arena=target_hidden_slots=shared_storage=None
                gc.collect();torch.cuda.empty_cache()
                allocation_correction += (correction+15)//16*16
                admit_media(media_available - allocation_correction)
                reserve = (memory_budget.withheld_bytes +
                           admission.input_encoder_reserves.get(placement.device_index, 0))
                pool_budget = SharedCacheStorage.budget_from_available(
                    min(free, memory_budget.startup.free_bytes) - reserve - graph_bytes -
                    transient_bytes, schemas, extra_state_fields) - allocation_correction
            spec=arena.spec
            if draft_arena is not None:draft_spec=draft_arena.spec
            # Context must fit one live request plus its actual checkpoint/COW
            # transaction, not just the KV-only equivalent of the shared bytes.
            minimum_states=1+admission.max_speculative_tokens+int(admission.use_prefix_cache)
            minimum_pages=(1+admission.max_speculative_tokens+page_size-1)//page_size
            if not shared_storage.allocator.cold_owner_capacity(
                    state_slots=minimum_states, pages=minimum_pages):
                raise ValueError("insufficient shared cache capacity for one request")
            common[0] = max(0,pool_budget-minimum_states*state_bytes)//combined_page_bytes
            all_reduce(common[:1],op=dist.ReduceOp.MIN)
            max_context = _effective_max_sequence_tokens(
                admission.max_sequence_tokens,int(common[0].item()),maximum,
                block_size=page_size, scratch_pages=0)
            if input_encoder_admission is not None:
                image_tokens = ((input_encoder_admission[1] + plan.merge_extent - 1)
                                // plan.merge_extent)
                if image_tokens + maximum > int(common[0].item()) * page_size:
                    raise ValueError("admitted media exceeds its physical language context")
            if not admission.auto_max_sequence_tokens and max_context != admission.max_sequence_tokens:
                raise ValueError("requested context exceeds shared KV/state capacity")
            admission.max_sequence_tokens=max_context
            table_pages=(max_context+page_size-1)//page_size
        from cubic_serve.runtime.quantized_qualification import (
            _bound_composition_plan,
            audit_capture,
            audit_state,
            execution_identity,
            portfolio_resources,
        )

        diagnostic_event(_LOGGER, 'quantized_graph.cache_allocation_complete_rank_d_duration_s',
            rank,
            time.perf_counter() - phase_started,
        )
        phase_started = time.perf_counter()
        startup_graph_capture_started(rank)
        identity.max_sequence_tokens = admission.max_sequence_tokens
        execution_witnesses: dict[str, str] = {}
        executors = {}
        draft_executors = {}
        constraint_arena = None
        target_scratch = GraphScratchPool(capacities=admission.graph_capacities)
        provider_scratch = target_scratch.sibling()
        if admission.startup_numerical_audit and not typed_stage:
            from cubic_serve.guardrail.quantized_graph.collective import prepare as observe_bindings
        else:
            observe_bindings = nullcontext
        def capture_bucket(capacity: int) -> Any:
            nonlocal constraint_arena
            if capacity in executors:
                return executors[capacity]
            build_started = time.perf_counter()
            with observe_bindings():
                stage = stage_type(
                    admission.config,
                    weights,
                    topology,
                    captured_m=capacity,
                    table_pages=table_pages,
                    arena=arena,
                    mrope_tables=tables,
                    **options,
                    shared_scratch=target_scratch,
                )
            build_seconds = time.perf_counter() - build_started
            eager_started = time.perf_counter()
            snapshots = _prepare_capture_inputs(stage, admission.graph_capacities, qualification)
            eager_seconds = time.perf_counter() - eager_started
            capture_started = time.perf_counter()
            stage.capture()
            if not typed_stage:
                if stable_backing:
                    stage.prepare_decode_overlap()
                else:
                    stage.prepare_decode_overlap(retained_plan=overlap_plans[capacity])
            stages.append(stage)
            capture_seconds = time.perf_counter() - capture_started
            audit_started = time.perf_counter()
            if qualification is None:
                execution_witnesses[str(capacity)] = audit_capture(stage, identity, snapshots)
            elif not qualification.engineering:
                execution_witnesses[str(capacity)] = qualification.capture(stage)
            else:
                execution_witnesses[str(capacity)] = qualification.run(
                    "target-capture", (capacity, _bound_composition_plan(stage)),
                    lambda snapshots=snapshots: audit_capture(stage, identity, snapshots))
            del snapshots
            executors[capacity] = _CapturedGraphRank(
                stage, stage.resource, stage, runtime, admission, target_hidden_slots,
            )
            if constraint_arena is None:
                constraint_arena = _ConstraintSamplingArena(executors[capacity], maximum,
                    qualification=qualification, consume_logits=draft_provider is None)
            _prepare_rank_greedy_sampling(executors[capacity], sampling_lifetime,
                                          qualify_resident=not stable_backing,
                                          constraint_arena=constraint_arena,
                                          qualification=qualification)
            if draft_provider is not None:
                draft_stage = draft_provider.build_stage(
                    weights, topology, captured_m=capacity, table_pages=table_pages,
                    arena=draft_arena, mrope_tables=tables, **options,
                    shared_scratch=provider_scratch,
                )
                draft_snapshots = _prepare_capture_inputs(
                    draft_stage, admission.graph_capacities, qualification)
                draft_stage.capture()
                stages.append(draft_stage)
                if qualification is None:
                    execution_witnesses[f"draft:{capacity}"] = audit_capture(draft_stage, identity, draft_snapshots)
                elif not qualification.engineering:
                    execution_witnesses[f"draft:{capacity}"] = qualification.capture(draft_stage)
                else:
                    execution_witnesses[f"draft:{capacity}"] = qualification.run(
                        "provider-capture", (capacity, _bound_composition_plan(draft_stage)),
                        lambda: audit_capture(draft_stage, identity, draft_snapshots))
                del draft_snapshots
                binding = TransactionalDraftBinding(_CapturedGraphRank(
                    draft_stage, draft_stage.resource, draft_stage, runtime, admission,
                    private_state_sources=True,
                ))
                draft_executors[capacity] = CapturedDraftRank(
                    draft_stage, draft_stage, runtime, admission,
                    request_page_capacity=page_capacity,
                    target_hidden_slots=target_hidden_slots, binding=binding,
                )
                _prepare_rank_greedy_sampling(draft_executors[capacity], sampling_lifetime,
                                              qualify_resident=not stable_backing,
                                              constraint_arena=constraint_arena,
                                              qualification=qualification)
            startup_graph_captured(
                admission.execution_mode, rank, capacity, build_seconds, eager_seconds,
                capture_seconds, time.perf_counter() - audit_started,
            )
            completion = None if typed_stage else stage.route_completion
            if completion is not None and completion.predicted_us is not None:
                diagnostic_event(_LOGGER, 'quantized_graph.s_captured_rank_d_m_d_route',
                    admission.execution_mode,
                    rank,
                    capacity,
                    completion.selected,
                    *completion.predicted_us,
                )
            results.send(_LoadProgress(
                rank, "graph_capture", len(executors), len(admission.graph_capacities), "buckets",
                capacity=capacity,
            ))
            return executors[capacity]

        deferred_text_audit = admission.startup_numerical_audit and typed_stage
        for capacity in ((1,) if deferred_text_audit else memory_capacities):
            capture_bucket(capacity)
        phase_started = time.perf_counter()
        if admission.startup_numerical_audit and not typed_stage:
            from cubic_serve.guardrail.quantized_graph import audit_original
            from cubic_serve.guardrail.quantized_graph.cross_m import audit_cross_m

            for capacity in sorted({1, min(8, maximum)}):
                numerical_witnesses[str(capacity)] = audit_original(
                    executors[capacity].body, admission, groups,
                )
                diagnostic_event(_LOGGER, 'quantized_graph.independent_numerical_audit_passed_rank_d_m',
                    rank,
                    capacity,
                )
            numerical_witnesses["cross_m"] = audit_cross_m(
                executors, admission, groups,
                completed_witness=lambda completed, total: results.send(_LoadProgress(
                    rank, "numerical_audit", completed, total, "witnesses")),
            )
            diagnostic_event(_LOGGER, 'quantized_graph.independent_numerical_qualification_complete_rank_d_duration',
                rank,
                time.perf_counter() - phase_started,
            )
        phase_started = time.perf_counter()
        if admission.startup_numerical_audit and typed_stage:
            from cubic_serve.guardrail.quantized_graph.text_cross_m import (
                audit_cross_m as audit_text_cross_m,
            )

            numerical_witnesses["typed_cross_m"] = audit_text_cross_m(
                executors, admission, groups, numerical_witnesses["typed_m1"],
                capacities=admission.graph_capacities, capture_bucket=capture_bucket,
                reference_math_memo=reference_math_memo,
                visual_report=numerical_witnesses["typed_visual_m1"],
                visual_tokens=tokens, input_feature_rows=feature_rows, position_axes=axes,
                completed_witness=lambda completed, total: results.send(_LoadProgress(
                    rank, "target_numerical_audit", completed, total, "witnesses")),
            )
            if draft_executors:
                # A recurrent proposal trajectory and a fixed-feedback cross-M
                # trace are different witnesses. Neither target capture identity
                # nor a draft M1 receipt qualifies the provider's larger buckets.
                fixed_feedback = tuple(feedback.clone() for _ in range(draft_steps))
                numerical_witnesses["draft_fixed_feedback_m1"] = audit_text(
                    draft_executors[1].body, admission, groups,
                    diagnose_independent_trajectory=True, draft_provider=draft_provider,
                    reference_math_memo=reference_math_memo,
                    feedback_rows=fixed_feedback,
                    column_completion_names=draft_provider.reference_columns,
                )
                numerical_witnesses["draft_cross_m"] = audit_text_cross_m(
                    draft_executors, admission, groups,
                    numerical_witnesses["draft_fixed_feedback_m1"],
                    draft_provider=draft_provider, feedback_rows=fixed_feedback,
                    reference_math_memo=reference_math_memo,
                    completed_witness=lambda completed, total: results.send(_LoadProgress(
                        rank, "provider_numerical_audit", completed, total, "witnesses")),
                )
            reference_math_memo.clear()
            reference_math_memo = None
            diagnostic_event(_LOGGER, 'quantized_graph.typed_cross_m_identity_checks_complete_rank',
                rank,
            )
        retained_draft_rounds = None
        if stable_backing:
            retained_draft_rounds = capture_draft_rounds(
                draft_executors, executors, dynamic_depth=admission.dynamic_speculation,
                lifetime=sampling_lifetime, qualification=qualification)
            torch.cuda.synchronize(device)
        # Engineering witnesses need the complete graph/state portfolio, not
        # the unused future serving KV pool. Retain the original runtime reserve
        # measurement; temporary witness images must not reduce final capacity.
        defer_page_admission = bool(stable_backing and qualification is not None
                                    and qualification.engineering)
        admission_transient_bytes = (max(0, torch.cuda.max_memory_allocated(device)
                                         - torch.cuda.memory_allocated(device))
                                     if stable_backing else 0)
        def finalize_page_admission():
            nonlocal page_capacity, max_context
            admission_started = time.perf_counter()
            # Price the consensus buffer's real allocator backing before KV.
            # Even one int64 can require a new device allocation segment.
            common = torch.empty(1, dtype=torch.int64, device=device)
            torch.cuda.empty_cache()
            measured_free, _ = torch.cuda.mem_get_info(device)
            transient_bytes = admission_transient_bytes
            available = max(0, memory_budget.remaining_bytes(measured_free)
                            - admission.input_encoder_reserves.get(placement.device_index, 0)
                            - transient_bytes)

            def additional_backing(pages):
                return (arena.additional_page_backing_bytes(pages)
                        + (0 if draft_arena is None else draft_arena.additional_page_backing_bytes(pages)))

            lower, upper = page_capacity, reserved_pages
            while lower < upper:
                middle = (lower + upper + 1) // 2
                if additional_backing(middle) <= available:
                    lower = middle
                else:
                    upper = middle - 1
            common.fill_(lower)
            all_reduce(common, op=dist.ReduceOp.MIN)
            page_capacity = int(common.item())
            if page_capacity <= 2 * maximum or additional_backing(page_capacity) > available:
                raise ValueError("insufficient memory after the qualified graph portfolio")
            expected_addition = additional_backing(page_capacity)
            original_page_capacity = arena.spec.page_capacity
            arena.finalize_page_capacity(page_capacity)
            if draft_arena is not None:
                draft_arena.finalize_page_capacity(page_capacity)
            if fixed_backing_identity != (arena.backing_identity,
                    None if draft_arena is None else draft_arena.backing_identity):
                raise RuntimeError("fixed KV addresses changed during page admission")
            admitted_context = _effective_max_sequence_tokens(
                admission.max_sequence_tokens, page_capacity, maximum, block_size=page_size)
            if not admission.auto_max_sequence_tokens and admitted_context != admission.max_sequence_tokens:
                raise ValueError("requested context exceeds the topology BF16 KV/state capacity")
            admission.max_sequence_tokens = max_context = admitted_context
            for executor in executors.values():
                executor.max_sequence_tokens = max_context
            for executor in draft_executors.values():
                executor._request_page_capacity = page_capacity
                executor._draft_binding.executor.max_sequence_tokens = max_context
            execution_witnesses["physical_admission"] = stable_digest({
                "pages": page_capacity, "reserved_pages": reserved_pages,
                "max_context": max_context, "added_bytes": expected_addition,
                "fixed_addresses": True,
            })
            diagnostic_event(_LOGGER, 'quantized_graph.fixed_address_kv_admission_rank_d_pages',
                rank,
                page_capacity,
                expected_addition / (1 << 20),
                transient_bytes / (1 << 20),
                max_context,
                time.perf_counter() - admission_started,
            )
            if defer_page_admission:
                # The model/state proofs used the fully backed bootstrap and
                # scratch ranges. Prove the newly admitted, previously unmapped
                # pages once at their own zero-initialization boundary.
                from cubic_serve.runtime.quantized_qualification import _storage_is_zero

                dirty = []
                for role, cache_arena in (("target", arena), ("provider", draft_arena)):
                    if cache_arena is None:
                        continue
                    for name, field in (("key", cache_arena.key),
                                        ("value", cache_arena.value),
                                        ("written", cache_arena.written),
                                        *cache_arena.page_fields.items()):
                        added = field[original_page_capacity:page_capacity]
                        if not _storage_is_zero(added):
                            dirty.append((role, name))
                failures = [None] * dist.get_world_size()
                dist.all_gather_object(failures, dirty)
                if any(failures):
                    raise ValueError("newly admitted KV pages changed their zero boundary")
                execution_witnesses["physical_extension_zero"] = stable_digest({
                    "range": (original_page_capacity, page_capacity),
                    "fields": tuple((role, tuple(cache_arena.page_fields))
                                    for role, cache_arena in (("target", arena),
                                                             ("provider", draft_arena))
                                    if cache_arena is not None),
                    "exact_storage_bits": True,
                })
        if stable_backing:
            if not defer_page_admission:
                finalize_page_admission()
            for executor in executors.values():
                if executor._target_hidden_slots is None:
                    for window in (executor._resident_window, executor._resident_stochastic_window):
                        if qualification is None:
                            window.qualify()
                        else:
                            qualification.window(window)
        identity.max_sequence_tokens = admission.max_sequence_tokens
        startup_graphs_ready(rank)
        phase_started = time.perf_counter()
        if qualification is not None and not qualification.engineering:
            qualification.bind_portfolio(executors)
        state_witness = (audit_state(executors, identity) if qualification is None else
                         qualification.state(executors) if not qualification.engineering else
                         qualification.run("target-state", tuple(
                             (m, _bound_composition_plan(e.body)) for m, e in executors.items()),
                             lambda: audit_state(executors, identity)))
        if draft_executors:
            draft_state_executors = {
                capacity: executor._draft_binding.executor
                for capacity, executor in draft_executors.items()
            }
            draft_state_witness = (audit_state(draft_state_executors, identity) if qualification is None else
                qualification.state(draft_state_executors) if not qualification.engineering else
                qualification.run("provider-state", tuple(
                    (m, _bound_composition_plan(e.body)) for m, e in draft_state_executors.items()),
                    lambda: audit_state(draft_state_executors, identity)))
            state_witness = stable_digest({"target": state_witness, "draft": draft_state_witness})
        startup_state_ready(rank, time.perf_counter() - phase_started)
        phase_started = time.perf_counter()
        startup_sampling_started(rank)
        def warm_sampling(portfolio):
            return _warm_sampling(portfolio, price_sampling_through=(admission.graph_request_capacity
                                  if admission.dynamic_speculation else 0),
                                  full_bucket_only=qualification is not None and not qualification.engineering)
        constraint_sampling_cost_samples = (warm_sampling(executors) if qualification is None else
            qualification.costs("sampling-costs", executors, warm_sampling,
                                entry=admission.graph_request_capacity if admission.dynamic_speculation else 0))
        draft_rounds = {}
        if draft_executors:
            if qualification is None:
                _qualify_draft_verification(draft_executors, executors)
            else:
                qualification.draft_verifier(draft_executors, executors)
            draft_rounds = capture_draft_rounds(
                draft_executors, executors, dynamic_depth=admission.dynamic_speculation,
                lifetime=sampling_lifetime, qualify_prompts=True,
                retained_rounds=retained_draft_rounds, qualification=qualification,
                progress=startup_graph_progress(rank, "Preparing MTP graphs"))
            if not stable_backing and not frozenset(draft_rounds)<=profiled_round_keys:
                raise ValueError("final speculative portfolio contains unpriced resources")
            from cubic_serve.runtime.quantized_qualification import qualify_draft_windows
            execution_witnesses["draft_window"] = (qualify_draft_windows if qualification is None
                                                     else qualification.draft_rounds)(
                draft_rounds, draft_executors, executors, target_hidden_slots, admission,
                **({"draft_provider": draft_provider, "groups": groups}
                   if qualification is not None else {}),
                rank=rank, sampler_rank=(topology.pp_size - 1) * topology.tp_size,
                progress=lambda completed, total: results.send(_LoadProgress(
                    rank, "mtp_composition", completed, total, "witnesses")),
            )
            startup_draft_portfolio(rank, admission.num_speculative_tokens, tuple(draft_rounds))
        target_cost_samples = (
            (_cold_start_target_costs(executors) if qualification is None else
             qualification.costs("target-costs", executors, _cold_start_target_costs))
            if admission.dynamic_speculation else ()
        )
        target_depth_samples = (
            (_cold_start_target_depth_costs(executors, admission.dynamic_max_speculative_tokens + 1)
             if qualification is None else qualification.costs("target-depth-costs", executors,
                lambda portfolio: _cold_start_target_depth_costs(portfolio,
                    admission.dynamic_max_speculative_tokens + 1),
                entry=admission.dynamic_max_speculative_tokens + 1))
            if admission.dynamic_speculation else ()
        )
        proposal_cost_samples = (
            (_cold_start_mtp_proposal_costs(draft_executors) if qualification is None else
             qualification.costs("provider-costs", draft_executors, _cold_start_mtp_proposal_costs))
            if admission.dynamic_speculation else ()
        )
        startup_sampling_ready(rank, time.perf_counter() - phase_started)
        feeding_profile = new_step_profile(
            f"worker-{rank}", schema="cubic.text-graph.feeding-profile.v1", logger=_LOGGER,
        )
        if feeding_profile is not None:
            for executor in executors.values():
                executor.enable_feeding_profile(feeding_profile)
            for executor in draft_executors.values():
                executor.enable_feeding_profile(feeding_profile)
        prepare_resident_compactions(executors, sampling_lifetime, qualification=qualification)
        if defer_page_admission:
            finalize_page_admission()
            identity.max_sequence_tokens = admission.max_sequence_tokens
        torch.cuda.synchronize(device)
        barrier()
        # Match the profiled memory boundary: cold capture temporaries are not
        # runtime residency, even when the allocator still caches their blocks.
        cold_free, _ = torch.cuda.mem_get_info(device)
        torch.cuda.empty_cache()
        final_free, _ = torch.cuda.mem_get_info(device)
        diagnostic_event(_LOGGER, 'quantized_graph.graph_memory_final_rank_d_free_f',
            rank,
            final_free / (1 << 20),
            reserve / (1 << 20),
            (final_free - cold_free) / (1 << 20),
            memory_budget.withheld_bytes / (1 << 20),
            admission.input_encoder_reserves.get(placement.device_index, 0) / (1 << 20),
            (final_free - reserve) / (1 << 20),
        )
        if final_free < reserve:
            raise ValueError(
                "final graph portfolio exceeded its measured memory budget: "
                f"rank={rank} free={final_free / (1 << 20):.1f} MiB "
                f"required_reserve={reserve / (1 << 20):.1f} MiB"
            )
        _log_memory_residency(
            rank=rank, device=device, weights=weights, arena=arena, draft_arena=draft_arena,
            startup_free_bytes=memory_budget.startup.free_bytes,
            final_free_bytes=final_free, reserve_bytes=reserve,
            page_bytes=page_bytes, draft_page_bytes=draft_page_bytes,
            max_running_requests=admission.max_running_requests, maximum=maximum,
        )
        cache = Qwen3_5TP2PP2RankCacheCapacity(
            rank,
            str(device),
            page_capacity,
            spec.state_slot_capacity,
            maximum,
            combined_page_bytes,
            state_bytes,
            shared_storage.backing.numel() if shared_storage is not None else (
                arena.backed_bytes + (0 if draft_arena is None else draft_arena.backed_bytes)
                + (0 if target_hidden_slots is None else
                   target_hidden_slots.numel() * target_hidden_slots.element_size())),
            0,
            reserve,
            free,
            free,
            final_free,
            total,
            shared_pool_bytes=None if shared_storage is None else shared_storage.budget_bytes,
            vram_page_capacity_estimate=(
                page_capacity
                + max(0, min(final_free, memory_budget.startup.free_bytes) - reserve)
                // combined_page_bytes
            ),
        )
        publish_owned_input_encoder(owned_encoder, admission)
        encoder_thread = start_owned_input_encoder(
            owned_encoder, getattr(admission, "input_encoder_connection", None),
            lambda failure: results.send(_Failure(rank, None, failure)))
        from cubic_serve.runtime.heap_lifetime import frozen_startup_heap

        sampling_lifetime.enter_context(frozen_startup_heap())
        from cubic_serve.executor.resident_window import (
            resident_compaction_capacities,
            resident_input_program_capacities,
            resident_publication_names,
            resident_segment_program_capacities,
        )

        diagnostic_event(_LOGGER, 'quantized_graph.post_weight_startup_complete_rank_d_duration',
            rank,
            time.perf_counter() - weights_loaded_at,
        )
        if qualification is not None:
            qualification.finish()
        startup_lifetime.close()
        results.send(
            _Ready(
                rank,
                builtin_text_config(admission.config).vocabulary_size,
                max_context,
                maximum,
                stable_digest({"execution": execution_witnesses, "numerical": numerical_witnesses}),
                state_witness,
                cache,
                target_cost_samples,
                proposal_cost_samples,
                target_depth_samples=target_depth_samples,
                constraint_sampling_cost_samples=constraint_sampling_cost_samples,
                input_encoder_admission=input_encoder_admission,
                compound_cost_samples=(
                    tuple((rows, round_.depth + 1, greedy, _median_cost_samples(values))
                          for (_review, _depth), round_ in sorted(draft_rounds.items())
                          for (rows, greedy), values in sorted(round_._startup_round_costs.items()))
                    if draft_executors else ()
                ),
                replay_resources=portfolio_resources(executors, draft_executors),
                resident_publications=resident_publication_names(executors),
                resident_input_program_capacities=resident_input_program_capacities(executors),
                resident_stochastic_input_program_capacities=(
                    resident_input_program_capacities(executors, stochastic=True)
                ),
                resident_compaction_capacities=resident_compaction_capacities(executors),
                constraint_graphs_qualified=all(
                    item._sampling.masked_plans
                    and all(execution_ready(plan) for plan in item._sampling.masked_plans)
                    for item in (*executors.values(), *draft_executors.values())
                ),
                constraint_compound_cost_samples=tuple(
                    (rows, round_.depth + 1, greedy, processing, _median_cost_samples(values))
                    for (_review, _depth), round_ in sorted(draft_rounds.items())
                    for (rows, greedy, processing), values in sorted(
                        getattr(round_, "_startup_constraint_round_costs", {}).items())
                ),
                resident_segment_program_capacities=resident_segment_program_capacities(executors),
                resident_draft_capacities=tuple(
                    capacity for capacity, executor in executors.items()
                    if execution_ready(getattr(getattr(executor, "_pipelined_draft_round", None),
                                               "_resident_window", None))
                ),
                grammar_draft_continuation_capacities=tuple(
                    capacity for capacity, executor in executors.items()
                    if execution_ready(getattr(executor, "_pipelined_draft_round", None),
                                       "grammar_continuation_qualified")
                ),
                grammar_continuation_capacities=tuple(
                    capacity for capacity, executor in executors.items()
                    if executor._input_update_graph is not None
                    and executor._sampling.masked_plans
                    and all(execution_ready(plan) for plan in executor._sampling.masked_plans)
                ),
            )
        )
        view = memoryview(shared_commands).cast("B")
        grammar_pool = None
        sampler_rank = (topology.pp_size - 1) * topology.tp_size
        resident: _CapturedGraphRank | None = None
        resident_id = -1
        resident_requests: tuple[str, ...] = ()
        offsets: tuple[int, ...] = ()
        page_counts: tuple[int, ...] = ()
        resident_draft_template = None
        resident_grammar_bindings = ()
        cache_ownership = None if shared_storage is None else SharedCacheOwnership(shared_storage.allocator)
        while True:
            notice = commands.recv_bytes()
            if len(notice) != _COMMAND_NOTICE.size:
                raise ValueError("invalid graph command notice")
            (size,) = _COMMAND_NOTICE.unpack(notice)
            if not 0 < size <= shared_capacity:
                raise ValueError("graph command exceeds retained command storage")
            payload=bytes(view[:size])
            witnessed=cache_ownership is not None and payload.startswith(SharedCacheOwnership._tag)
            if witnessed:payload=cache_ownership.unpack(payload)
            command = _deserialize_worker_command(payload)
            if isinstance(command, CloseCommand):
                break
            if cache_ownership is not None:
                if not witnessed:raise ValueError("shared cache command lacks ownership")
                cache_ownership.validate_command(command)
            if isinstance(command, ContinueBatchCommand):
                if resident is None:
                    raise ValueError("steady continuation has no resident batch")
                if command.draft is not None:
                    if resident_draft_template is not None:
                        resident_draft_template = run_grammar_draft_continuation(
                            command, resident, template=resident_draft_template,
                            execute=lambda next_command: execute_draft_batch(
                                next_command, resident, draft_executors, target_hidden_slots,
                                admission, admission.graph_capacities, rank=rank,
                                sampler_rank=sampler_rank, grammar_pool=grammar_pool,
                            ),
                            previous_batch_id=resident_id, request_ids=resident_requests,
                            rank=rank, sampler_rank=sampler_rank,
                            completion_statuses=audit_statuses,
                            completion_barrier=completion_barrier, timeout=timeout,
                            interrupt=continuation_interrupt, publish=results.send,
                            grammar_pool=grammar_pool,
                        )
                        resident_id = command.batch_id
                        continue
                    run_draft_continuation(
                        command, resident, previous_batch_id=resident_id,
                        request_ids=resident_requests, rank=rank, sampler_rank=sampler_rank,
                        completion_barrier=completion_barrier, timeout=timeout,
                        interrupt=continuation_interrupt, publish=results.send,
                        grammar_pool=grammar_pool,
                    )
                    resident_id = command.batch_id
                    continue
                if resident_grammar_bindings:
                    offsets, page_counts, resident_grammar_bindings = run_grammar_continuation(
                        command, resident, bindings=resident_grammar_bindings,
                        previous_batch_id=resident_id, request_ids=resident_requests,
                        offsets=offsets, page_counts=page_counts, block_size=page_size,
                        rank=rank, sampler_rank=sampler_rank,
                        completion_statuses=audit_statuses, completion_barrier=completion_barrier,
                        timeout=timeout, interrupt=continuation_interrupt,
                        publish=results.send, grammar_pool=grammar_pool,
                        prepare_masks=_constraint_prefix_masks_with_overlap,
                    )
                    resident_id = command.batch_id
                    continue
                offsets, page_counts = run_resident_continuation(
                    command, resident, previous_batch_id=resident_id,
                    request_ids=resident_requests, offsets=offsets, page_counts=page_counts,
                    block_size=page_size, rank=rank, sampler_rank=sampler_rank,
                    completion_statuses=audit_statuses, completion_barrier=completion_barrier,
                    timeout=timeout, interrupt=continuation_interrupt, publish=results.send,
                )
                resident_id = command.batch_id
                resident_requests = command.request_ids
                continue
            if not isinstance(command, BatchCommand):
                raise ValueError("unsupported packed graph command")
            executor = executors[command.captured_m]
            if command.constraints or command.released_constraints:
                grammar_pool = _request_grammar_pool(
                    grammar_pool, command, admission, executor.vocab_size
                )
            # One controller dispatch owns all ranks until every output/state
            # consumer completes. No subsequent fill overlaps this transaction.
            executor.slot.select_decode_overlap(owned_single_token_rows(command.rows))
            if draft_executors or command.constraints:
                round_started_ns = time.monotonic_ns()
                result = execute_draft_batch(
                    command,
                    executor,
                    draft_executors,
                    target_hidden_slots,
                    admission,
                    admission.graph_capacities,
                    rank=rank,
                    sampler_rank=sampler_rank,
                    mtp_feeding_profile=feeding_profile,
                    grammar_pool=grammar_pool,
                )
            else:
                executor.replay_batch(command.rows)
                tokens = executor.sample_tokens(command.rows)
                result = BatchResult(
                    rank, command.batch_id, tuple(r.request_id for r in command.rows), tokens
                )
            executor.slot.synchronize()
            if draft_executors:
                # Verification, state repair and draft catch-up also enqueue
                # work on the rank's feeding stream. Durable consumption must
                # finish before the controller can release the joint lease.
                torch.cuda.current_stream(device).synchronize()
            ownership = (
                resident_batch_ownership(executor, command.rows, block_size=page_size)
                if not command.constraints or all(row.is_decode for row in command.rows)
                else None
            )
            completion_barrier.wait(timeout)
            if draft_executors:
                result = replace(result, step_ms=(time.monotonic_ns() - round_started_ns) / 1e6)
            # Residency depends on this completed transaction, not whether a
            # draft provider exists. Terminal K0 is ordinary owned decode.
            resident = executor if ownership is not None else None
            resident_id = command.batch_id if resident is not None else -1
            resident_requests, offsets, page_counts = ownership or ((), (), ())
            resident_grammar_bindings = command.constraints if ownership is not None else ()
            resident_draft_template = None
            if result.resident_draft_depth:
                resident = executor
                resident_id = command.batch_id
                resident_requests = tuple(dict.fromkeys(result.request_ids))
            if (result.pipelined_draft_depth
                    and execution_ready(getattr(executor, "_pipelined_draft_round", None),
                                        "grammar_continuation_qualified")):
                resident = executor
                resident_id = command.batch_id
                resident_requests = tuple(dict.fromkeys(result.request_ids))
                resident_draft_template = command
            if rank == sampler_rank:
                results.send(result)
    except BaseException:
        with suppress(OSError, EOFError):
            results.send(_Failure(rank, None, traceback.format_exc()))
        raise
    finally:
        startup_lifetime.close()
        if weights is not None:
            weights.finish_loading(cancel=True)
        if feeding_profile is not None:
            feeding_profile.emit_once()
        finish_owned_input_encoder(encoder_thread)
        profile_sampling_lifetime.close()
        sampling_lifetime.close()
        for stage in stages:
            stage.close()
        if dist.is_initialized():
            dist.destroy_process_group()


class TextGraphExecutor:
    """Shared model graph executor with plain or format-owned weight bindings."""

    def __init__(
        self,
        model_root: str,
        *,
        devices: tuple[int, ...],
        tp: int,
        pp: int,
        ep: int,
        max_requests: int,
        max_context: int | None,
        prefill_chunk: int | None,
        cuda_graph_max_tokens: int | str = "auto",
        operator_config: OperatorConfig | None,
        eos_token_ids: tuple[int, ...],
        use_prefix_cache: bool = True,
        execution_mode: str = "debug",
        compute_dtype: str = "bf16",
        mamba_ssm_cache_dtype: str = "bfloat16",
        cache_storage: KVCacheStorageSpec | None = None,
        startup_numerical_audit: bool = False,
        startup_qualification: bool = False,
        input_encoder_reserves: dict[int, int] | None = None,
        device_memory_budgets: dict[int, DeviceMemoryBudget] | None = None,
        input_encoder: Any = None,
        checkpoint: CheckpointWeights | None = None,
        num_speculative_tokens: int = 0,
        max_speculative_tokens: int = 0,
        dynamic_speculation: bool = False,
        draft_policy: Any = None,
        model_definition_source: LoadedModelDefinition | None = None,
        reasoning_token_boundaries: tuple[tuple[int, ...], ...] = (),
    ) -> None:
        self._input_encoder = input_encoder
        colocated_encoder = input_encoder if getattr(input_encoder, "colocated", False) else None
        encoder_reserves = dict(input_encoder_reserves or {})
        memory_budgets = dict(device_memory_budgets or {})
        if any(type(device) is not int or device < 0 or not isinstance(budget, DeviceMemoryBudget)
               or budget.startup.device != f"cuda:{device}"
               for device, budget in memory_budgets.items()):
            raise ValueError("GPU budgets must match their physical CUDA device")
        if any(type(device) is not int or device < 0 or type(size) is not int or size < 0
               for device, size in encoder_reserves.items()):
            raise ValueError(
                "input encoder reserves require non-negative device indices and byte counts")
        if execution_mode not in ("debug", "cuda"):
            raise ValueError("text graph execution requires debug or cuda")
        if startup_numerical_audit and execution_mode != "cuda":
            raise ValueError("startup numerical audit requires cuda mode")
        if execution_mode == "cuda" and pp != 1:
            raise ValueError("independent text performance qualification requires PP1")
        if type(max_requests) is not int or max_requests <= 0:
            raise ValueError("max_requests must be positive")
        if prefill_chunk is not None and (type(prefill_chunk) is not int or prefill_chunk <= 0):
            raise ValueError("prefill_chunk must be positive")
        # Validate even before model/checkpoint I/O. This budget bounds physical
        # rows, including padding, rather than rounding above an administrator's
        # limit to accommodate a particular request count.
        auto_graph_max_tokens = cuda_graph_max_tokens == "auto"
        graph_token_limit = 1024 if auto_graph_max_tokens else cuda_graph_max_tokens
        _text_graph_capacities(graph_token_limit, max_speculative_tokens)
        root = Path(model_root)
        if model_definition_source is not None:
            model_definition_source.verify_source()
        selected_definition = (None if model_definition_source is None
                               else model_definition_source.definition)
        encoder_namespace = getattr(colocated_encoder, "model_definition_namespace", None)
        if encoder_namespace is not None and (
            model_definition_source is None
            or encoder_namespace != model_definition_source.module_name
        ):
            raise ValueError("colocated input encoder belongs to a different model definition")
        document = (root / "config.json").read_bytes()
        quantized = QuantizationConfig.from_model_config(json.loads(document)) is not None
        config = parse_model_config_json(
            document, compute_dtype=compute_dtype if quantized else None,
            definition=selected_definition,
        )
        if (
            type(num_speculative_tokens) is not int or num_speculative_tokens < 0
            or type(max_speculative_tokens) is not int or max_speculative_tokens < 0
        ):
            raise ValueError("draft depth must be a nonnegative integer")
        max_speculative_tokens = max_speculative_tokens or num_speculative_tokens
        if num_speculative_tokens > max_speculative_tokens:
            raise ValueError("initial draft depth exceeds its admitted capacity")
        draft_provider = (model_draft_provider(config, method="mtp",
                                              max_depth=max_speculative_tokens,
                                              definition=selected_definition)
                          if num_speculative_tokens else None)
        self.config = config
        if cache_storage is not None:
            if not isinstance(cache_storage, KVCacheStorageSpec):
                raise TypeError("cache_storage requires independent typed K/V storage specs")
            dtype = getattr(torch, builtin_text_config(config).dtype)
            cache_storage = cache_storage.resolve(dtype)
            default_storage = KVCacheStorageSpec().resolve(dtype)
            # The original native path and startup identities remain unchanged
            # when the caller explicitly requests its existing storage policy.
            if cache_storage == default_storage:
                cache_storage = None
            elif any(spec.dtype != native.dtype and not spec.dtype.quantized
                     for spec, native in ((cache_storage.key, default_storage.key),
                                          (cache_storage.value, default_storage.value))):
                raise ValueError("plain KV storage must match the model compute dtype")
        if not quantized:
            requested_dtype = {"bf16": "bfloat16", "fp16": "float16"}.get(
                compute_dtype, compute_dtype
            )
            if requested_dtype != builtin_text_config(config).dtype:
                raise ValueError("plain text graph execution must preserve the checkpoint dtype")
        operator_config = operator_config or OperatorConfig()
        if mamba_ssm_cache_dtype not in ("auto", "float32", "bfloat16"):
            raise ValueError("mamba_ssm_cache_dtype must be auto, float32, or bfloat16")
        if execution_mode == "cuda":
            required = ("auto", "auto", "fp32")
            actual = (
                operator_config.attention_dtype,
                operator_config.activation_dtype,
                operator_config.softmax_dtype,
            )
            if actual != required:
                raise ValueError(
                    "independent text performance qualification requires "
                    f"attention={required[0]}, activation={required[1]}, softmax={required[2]}"
                )
        topology = ParallelTopology.single_node(
            topology_id=f"text.{execution_mode}.tp{tp}.pp{pp}.ep{ep}",
            device_indices=devices,
            tp_size=tp,
            pp_size=pp,
            ep_size=ep,
        )
        try:
            definition = resolve_model_definition(config, definition=selected_definition)
        except BuiltinModelConfigError as error:
            raise ValueError("this model has no text graph implementation") from error
        if definition.text_graph is None:
            raise ValueError("model has no text graph construction binding")
        graph_definition = definition.text_graph(
            config, topology, block_size=lcm(
                _BLOCK_SIZE, 1 if cache_storage is None else cache_storage.column_group_alignment),
            recurrent_state_dtype=mamba_ssm_cache_dtype,
            **({"definition": definition} if model_definition_source is not None else {}),
        )
        typed = graph_definition.named_state_fields
        mamba_ssm_cache_dtype = graph_definition.recurrent_state_dtype
        logical = definition.logical_weights(config)
        bindings = tuple(
            definition.owner_slices(
                config, topology, r, logical_weights=logical,
                include_mtp=draft_provider is not None)
            for r in range(len(devices))
        )
        if checkpoint is None:
            from cubic_serve.models.builtin import admit_model_checkpoint

            checkpoint = admit_model_checkpoint(
                root, model_config=config, definition=definition, require_quantization=quantized,
            )
        else:
            checkpoint.validate_for(root, logical)
        if operator_config.compute_mode != "wna16":
            if checkpoint.quantization is None:
                raise ValueError("non-default packed compute requires a packed checkpoint")
            # Physical headers, not exporter-wide rules, declare the actual
            # encodings. Check active linear consumers here; the common MoE
            # binder separately checks its own routed gate/up/down formats.
            # A format used only by an LM head need not implement routed MoE.
            active_weights = {binding.tensor_name for rank_bindings in bindings
                              for binding in rank_bindings}
            encodings = {weight.encoding for weight in checkpoint.weights
                         if weight.encoding is not None
                         and weight.logical.name in active_weights
                         and weight.logical.operation == "linear"}
            for encoding in encodings:
                require_compute_mode(encoding, "linear", operator_config.compute_mode)
        plan = _Plan(topology, bindings)
        # Request admission and the resident graph envelope are separate limits.
        # The existing fair scheduler drains larger client populations through
        # complete transactions in these sealed buckets; it never captures a
        # graph at runtime or falls back to eager execution.
        capacities = _text_graph_capacities(graph_token_limit, max_speculative_tokens)
        graph_request_capacity = _text_graph_owner_capacity(capacities, max_speculative_tokens)
        from cubic_serve.runtime.speculative import DraftPolicyConfig
        block_size = graph_definition.block_size_tokens

        admission = SimpleNamespace(
            model_root=Path(model_root),
            input_encoder_factory=(
                colocated_encoder.factory if colocated_encoder is not None else None),
            device_memory_budgets=memory_budgets,
            input_encoder_reserves=encoder_reserves,
            execution_mode=execution_mode,
            startup_numerical_audit=startup_numerical_audit,
            startup_qualification=startup_qualification,
            mamba_ssm_cache_dtype=mamba_ssm_cache_dtype,
            config=config,
            block_size_tokens=block_size,
            checkpoint=checkpoint.payload_view(),
            parallel_plan=plan,
            max_running_requests=max_requests,
            max_sequence_tokens=min(
                max_context or builtin_text_config(config).max_position_embeddings,
                builtin_text_config(config).max_position_embeddings,
            ),
            auto_max_sequence_tokens=max_context is None,
            graph_capacities=capacities,
            auto_graph_max_tokens=auto_graph_max_tokens,
            graph_request_capacity=graph_request_capacity,
            draft_provider=draft_provider,
            num_speculative_tokens=num_speculative_tokens,
            max_speculative_tokens=max_speculative_tokens,
            dynamic_speculation=dynamic_speculation,
            draft_policy=draft_policy or DraftPolicyConfig(),
            dynamic_max_speculative_tokens=max_speculative_tokens,
            use_prefix_cache=use_prefix_cache,
            prefill_chunk_tokens=prefill_chunk,
            operator_config=operator_config,
            reasoning_token_boundaries=reasoning_token_boundaries,
            constraint_stop_token_ids=eos_token_ids,
        )
        if cache_storage is not None:
            admission.cache_storage = cache_storage
        admission.input_encoder_working_sets = getattr(
            input_encoder, "working_sets_by_device", {})
        if (input_encoder is not None and getattr(input_encoder, "auto_memory", False)
                and not admission.input_encoder_working_sets):
            admission.input_encoder_working_sets = {
                input_encoder.weights.device.index: (
                    input_encoder.working_set, input_encoder.max_input_rows,
                    input_encoder.resources.staging_bytes)}
        if model_definition_source is not None:
            admission.model_definition_source = model_definition_source
            admission.model_program_recipe = model_definition_source.execution_recipe(config)
            if colocated_encoder is not None:
                admission.input_encoder_model_config = colocated_encoder.config
                admission.input_encoder_program_recipe = model_definition_source.execution_recipe(
                    colocated_encoder.config,
                )
        diagnostic_event(_LOGGER, 'quantized_graph.text_precision_activation_s_attention_s_softmax',
            operator_config.activation_dtype,
            operator_config.attention_dtype,
            operator_config.softmax_dtype,
            mamba_ssm_cache_dtype,
        )
        diagnostic_event(_LOGGER, 'quantized_graph.initializing_execution_workers_and_cuda_communication')
        self._controller = _ProcessController(
            admission,
            # This is a rank-local inactivity lease, renewed only by validated
            # progress. Audit-runtime objectives never impose a total cutoff.
            startup_timeout_seconds=1200,
            worker_target=_worker,
            graph_capacities=capacities,
            encoder_group=colocated_encoder,
        )
        if (input_encoder is not None and colocated_encoder is None
                and self._controller.input_encoder_admissions):
            input_encoder.limit_admission(self._controller.input_encoder_admissions)
        if colocated_encoder is not None:
            try:
                colocated_encoder.attach_main_workers(
                    self._controller._processes, self._controller.input_encoder_connections)
            except BaseException:
                colocated_encoder.close()
                self._controller.close()
                raise
        self.max_sequence_tokens = self._controller.max_sequence_tokens
        self.semantic_digest = checkpoint.semantic_digest
        if typed:
            self.semantic_digest = stable_digest({
                "checkpoint": checkpoint.semantic_digest,
                "operators": operator_config.digest,
                "recurrent_state_dtype": mamba_ssm_cache_dtype,
            })
        elif operator_config.compute_mode != "wna16":
            import hashlib

            self.semantic_digest = hashlib.blake2s(
                (checkpoint.semantic_digest + operator_config.digest).encode()
            ).hexdigest()
        if cache_storage is not None:
            self.semantic_digest = stable_digest({
                "execution": self.semantic_digest, "cache_storage": cache_storage.identity,
            })
        if model_definition_source is not None:
            self.semantic_digest = model_definition_source.program_identity(
                self.semantic_digest, admission.model_program_recipe,
            )
        self._defaults = load_sampling_defaults(root)
        self._eos = eos_token_ids
        self._lock = threading.Lock()
        self._streams: dict[str, TokenStream] = {}

    @property
    def kv_capacity_shared(self) -> bool:
        return getattr(self._controller,"_shared_cache_pool",None) is not None

    @property
    def kv_capacity_tokens(self) -> int:
        """Shared logical token capacity; TP copies do not multiply the pool."""

        return self._controller._page_capacity * self._controller._block_size

    @property
    def kv_vram_capacity_tokens(self) -> int | None:
        return self._controller.kv_vram_capacity_tokens

    @property
    def ready(self) -> bool:
        return self._controller.ready

    @property
    def vocab_size(self) -> int:
        return self._controller.vocab_size

    @property
    def supports_structured_output(self) -> bool:
        return getattr(self._controller, "constraint_graphs_qualified", False) is True

    def generate(
        self, input_ids: tuple[int, ...], max_tokens: int, metadata: TokenGenerationMetadata
    ) -> TokenStream:
        if (
            not self.ready
            or not input_ids
            or max_tokens <= 0
            or len(input_ids) + max_tokens > self.max_sequence_tokens
            or any(
                type(t) is not int or not 0 <= t < self._controller.vocab_size for t in input_ids
            )
        ):
            raise ValueError("request exceeds captured graph token/context admission")
        stream = TokenStream(lambda *_: self.cancel(metadata.request_id))
        command = GenerateCommand(
            metadata.request_id,
            input_ids,
            max_tokens,
            (
                ()
                if metadata.ignore_eos and metadata.structured_output.kind.value == "identity"
                else self._eos
            ),
            resolve_sampling(self._defaults, metadata.sampling),
            *prepared_input_fields(
                input_ids, metadata.prepared_input,
                hidden_size=builtin_text_config(self.config).hidden_size,
                max_position=builtin_text_config(self.config).max_position_embeddings,
                max_tokens=max_tokens,
            ),
            structured_output=(
                None
                if metadata.structured_output.kind.value == "identity"
                and metadata.reasoning_control is None
                else metadata.structured_output
            ),
            constraint_reasoning_end=metadata.constraint_reasoning_end,
            reasoning_control=metadata.reasoning_control,
        )
        with self._lock:
            if metadata.request_id in self._streams:
                raise ValueError("duplicate request id")
            self._streams[metadata.request_id] = stream
        try:
            self._controller.submit(command, stream)
        except BaseException:
            with self._lock:
                self._streams.pop(metadata.request_id, None)
            raise
        return stream

    def cancel(self, request_id: str) -> None:
        self._controller.cancel(request_id)
        with self._lock:
            self._streams.pop(request_id, None)

    def close(self) -> None:
        try:
            close_encoder = getattr(self._input_encoder, "close", None)
            if close_encoder is not None:
                close_encoder()
        finally:
            self._controller.close()


# Keep the explicit debug API compatible; its default never claims performance.
QuantizedGraphExecutor = TextGraphExecutor
QuantizedGraphDebugExecutor = TextGraphExecutor
