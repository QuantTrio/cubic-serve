"""Audit the real Qwen3.8-27B TP2 x PP2 decode graph on four GPUs.

The qualification boundary is deliberately strict: a captured bucket must be
bit-identical to its same-body eager launch, and two padding poisons must leave
every active output and rank-local state bit-identical.  The canonical eager
reference is reported separately; a non-exact result is never accepted without
the missing leaf-level reduction witnesses.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor
from torch.multiprocessing.spawn import spawn

from cubic_serve.contracts.execution import ParallelTopology, RankReplayResources
from cubic_serve.executor.distributed_reference import (
    Qwen3_5DistributedCache,
    Qwen3_5DistributedReferenceModel,
    seal_qwen3_5_distributed_reference,
)
from cubic_serve.executor.parallel import initialize_eager_nccl
from cubic_serve.executor.runner import ReferenceExecutionContext, configure_reference_math
from cubic_serve.models.checkpoint import CheckpointInventory, inspect_checkpoint
from cubic_serve.models.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5ParallelWeightPlan,
    Qwen3_5WeightPlan,
    load_qwen3_5_rank_weights,
    parse_qwen3_5_config_json,
    plan_qwen3_5_checkpoint_weights,
    plan_qwen3_5_tp2_pp2_weights,
)
from cubic_serve.models.qwen3_5.weights import plan_qwen3_5_dense_weights
from cubic_serve.runtime.model_state import (
    CONV_STATE_ID,
    RECURRENT_STATE_ID,
)

_CHECKPOINT = Path("/pubdata/llm_models/Qwen/Qwen3.8-27B")
_CAPTURED_MS = (128, 256, 512)
_CONTEXT = 128
_BLOCK = 16
_CANONICAL_TOKEN = 17
_GIB = 1 << 30
type Snapshot = dict[str, Tensor]


@dataclass(frozen=True, slots=True)
class _PerformanceAPI:
    body: type[Any]
    slot: type[Any]
    error: type[BaseException]


@dataclass(frozen=True, slots=True)
class _Admission:
    config: Qwen3_5Config
    inventory: CheckpointInventory
    weight_plan: Qwen3_5WeightPlan
    parallel_plan: Qwen3_5ParallelWeightPlan
    sealed: Any


def _emit(payload: object) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)


def _csv_ints(value: str, *, allowed: tuple[int, ...]) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("values must be comma-separated integers") from error
    if not result or len(set(result)) != len(result) or any(item not in allowed for item in result):
        raise argparse.ArgumentTypeError(f"values must be unique members of {allowed}")
    return result


def _load_performance_api(
    module_name: str = "cubic_serve.executor.model_runner",
) -> _PerformanceAPI:
    try:
        module = importlib.import_module(module_name)
        body = module.Qwen3_5TP2PP2DecodeBody
        slot = module.Qwen3_5TP2PP2GraphSlot
        error = module.Qwen3_5ParallelPerformanceError
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(f"TP2 x PP2 performance API is not ready: {exc}") from exc
    if not callable(getattr(body, "build", None)) or not callable(getattr(slot, "capture", None)):
        raise RuntimeError("TP2 x PP2 performance API is incomplete")
    return _PerformanceAPI(body, slot, error)


def _topology(
    devices: tuple[int, ...], tp_size: int = 2, pp_size: int = 2
) -> ParallelTopology:
    return ParallelTopology.single_node(
        topology_id=f"qwen3.8-27b:audit:tp{tp_size}:pp{pp_size}",
        device_indices=devices,
        tp_size=tp_size,
        pp_size=pp_size,
    )


def _admit(
    checkpoint: Path,
    devices: tuple[int, ...],
    tp_size: int = 2,
    pp_size: int = 2,
) -> _Admission:
    config = parse_qwen3_5_config_json((checkpoint / "config.json").read_bytes())
    inventory = inspect_checkpoint(checkpoint)
    weight_plan = plan_qwen3_5_checkpoint_weights(config, inventory)
    topology = _topology(devices, tp_size, pp_size)
    parallel_plan = (
        plan_qwen3_5_tp2_pp2_weights(config, weight_plan, topology)
        if (tp_size, pp_size) == (2, 2)
        else plan_qwen3_5_dense_weights(config, weight_plan, topology)
    )
    return _Admission(
        config,
        inventory,
        weight_plan,
        parallel_plan,
        seal_qwen3_5_distributed_reference(config, parallel_plan),
    )


def _resource(rank: int, captured_m: int, purpose: str = "audit") -> RankReplayResources:
    stem = f"tp2pp2-{purpose}-m{captured_m}-rank{rank}"
    return RankReplayResources(
        rank,
        stem + "-graph",
        stem + "-stream",
        stem + "-buffers",
        stem + "-workspace",
        stem + "-scratch",
    )


def _build_body(
    api: _PerformanceAPI,
    admission: _Admission,
    weights: Any,
    runtime: Any,
    captured_m: int,
    table_pages: int,
    purpose: str = "audit",
    *,
    recurrent_state_dtype: torch.dtype = torch.float32,
) -> tuple[Any, RankReplayResources]:
    placement = admission.parallel_plan.topology.placements[weights.rank]
    resource = _resource(weights.rank, captured_m, purpose)
    body = api.body.build(
        config=admission.config,
        parallel_plan=admission.parallel_plan,
        weights=weights,
        tp_group=runtime._groups[f"tp:{placement.pipeline_stage}"],
        pp_group=(
            None
            if admission.parallel_plan.topology.pp_size == 1
            else runtime._groups[f"pp:{placement.tensor_rank}"]
        ),
        captured_m=captured_m,
        table_pages=table_pages,
        page_capacity=table_pages + 2 * captured_m,
        state_capacity=captured_m,
        scratch_arena_id=resource.scratch_arena_id,
        recurrent_state_dtype=recurrent_state_dtype,
    )
    return body, resource


def _prepare(
    body: Any,
    resource: RankReplayResources,
    *,
    logical_m: int,
    context_length: int,
    poison: int | None,
    empty_context: bool = False,
    active_layout: str = "interleaved",
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Install deterministic owner state and lease-private padding state."""

    captured_m = cast(int, body.captured_m)
    if active_layout not in ("interleaved", "prefix"):
        raise ValueError("active layout must be interleaved or prefix")
    if not 0 < logical_m <= captured_m:
        raise ValueError("logical M must fit captured M")
    if active_layout == "interleaved" and logical_m not in (
        captured_m // 2,
        captured_m,
    ):
        raise ValueError("interleaved audit requires half or full captured M")
    if context_length <= 0 or context_length % _BLOCK:
        raise ValueError("context length must be a positive multiple of 16")
    if empty_context and context_length != _BLOCK:
        raise ValueError("canonical empty-context lane uses one physical page")
    pages = context_length // _BLOCK
    active = (
        tuple(range(logical_m))
        if active_layout == "prefix"
        else (
            tuple(range(captured_m))
            if logical_m == captured_m
            else tuple(range(0, captured_m, 2))
        )
    )
    padding = tuple(row for row in range(captured_m) if row not in active)
    if padding and poison not in (0, 1):
        raise ValueError("a lower logical M requires poison 0 or 1")
    if not padding and poison is not None:
        raise ValueError("a full bucket has no padding poison")

    body.prepare_scratch(resource)
    arena, buffers = body.arena, body.buffers
    owner_pages = logical_m * pages
    arena.key[:owner_pages].zero_()
    arena.value[:owner_pages].zero_()
    arena.written[:owner_pages].zero_()
    for state in arena.states.values():
        state[:logical_m].zero_()

    if not empty_context:
        values = (
            torch.arange(owner_pages, dtype=torch.float32, device=arena.device)
            .remainder_(31)
            .add_(1)
            .div_(128)
            .to(torch.bfloat16)
        )
        arena.key[:owner_pages].copy_(values.view(-1, 1, 1, 1, 1))
        arena.value[:owner_pages].copy_(values.neg().view(-1, 1, 1, 1, 1))
        arena.written[:owner_pages].fill_(True)
        conv = arena.states[CONV_STATE_ID][:logical_m]
        recurrent = arena.states[RECURRENT_STATE_ID][:logical_m]
        slots = torch.arange(logical_m, dtype=torch.int64, device=arena.device)
        conv_values = (
            slots.remainder(13).add(1).to(conv.dtype).div_(256).view(
                logical_m, *(1 for _ in conv.shape[1:])
            )
        )
        recurrent_values = (
            slots.remainder(11).add(1).neg().to(recurrent.dtype).div_(1024).view(
                logical_m, *(1 for _ in recurrent.shape[1:])
            )
        )
        conv.copy_(conv_values.expand_as(conv))
        recurrent.copy_(recurrent_values.expand_as(recurrent))

    global_tokens = torch.zeros(captured_m, dtype=torch.int64, device=arena.device)
    state_layers = buffers.conv_slots.shape[1]
    local_layers = torch.arange(state_layers, dtype=torch.int32, device=arena.device)
    for logical, physical in enumerate(active):
        first = logical * pages
        page_ids = torch.arange(first, first + pages, dtype=torch.int32, device=arena.device)
        buffers.position_ids[:, physical].fill_(0 if empty_context else context_length - 1)
        buffers.page_source[physical].fill_(first + pages - 1)
        buffers.page_destination[physical].fill_(first + pages - 1)
        buffers.token_offsets[physical].fill_(0 if empty_context else (context_length - 1) % _BLOCK)
        buffers.page_table[physical].copy_(page_ids)
        buffers.sequence_lengths[physical].fill_(1 if empty_context else context_length)
        buffers.segment_lengths[physical].fill_(1)
        aggregate = logical * state_layers + local_layers
        buffers.conv_slots[:, :, physical].copy_(aggregate.expand(2, -1))
        buffers.recurrent_slots[:, :, physical].copy_(aggregate.expand(2, -1))
        global_tokens[physical] = _CANONICAL_TOKEN if empty_context else 17 + logical % 97

    if padding:
        assert poison is not None
        page_start = arena.spec.page_capacity
        state_start = arena.spec.state_slot_capacity
        base = (37.0, -73.0)[poison]
        for ordinal, physical in enumerate(padding):
            page, state_slot = page_start + ordinal, state_start + ordinal
            buffers.position_ids[:, physical].fill_(31 + 32 * poison)
            buffers.page_source[physical].fill_(page)
            buffers.page_destination[physical].fill_(page)
            buffers.token_offsets[physical].zero_()
            buffers.page_table[physical].fill_(page)
            buffers.sequence_lengths[physical].fill_(1)
            buffers.segment_lengths[physical].zero_()
            aggregate = state_slot * state_layers + local_layers
            buffers.conv_slots[:, :, physical].copy_(aggregate.expand(2, -1))
            buffers.recurrent_slots[:, :, physical].copy_(aggregate.expand(2, -1))
            global_tokens[physical] = 1001 + 997 * poison + ordinal
            arena.key[page].fill_(base)
            arena.value[page].fill_(-base)
            arena.written[page].fill_(True)
        state_stop = state_start + len(padding)
        arena.states[CONV_STATE_ID][state_start:state_stop].fill_(base / 4)
        arena.states[RECURRENT_STATE_ID][state_start:state_stop].fill_(base / 16)

    if body.stage == 0:
        body.fill_token_ids(global_tokens.contiguous())
    if active:
        active_rows = torch.tensor(active, dtype=torch.int64, device=arena.device)
        expected_pages = torch.arange(
            logical_m * pages, dtype=torch.int32, device=arena.device
        ).view(logical_m, pages)
        if not torch.equal(
            buffers.page_table.index_select(0, active_rows), expected_pages
        ):
            raise RuntimeError("active row does not bind distinct context pages")
    if padding:
        padding_rows = torch.tensor(padding, dtype=torch.int64, device=arena.device)
        padding_pages = buffers.page_table.index_select(0, padding_rows)
        if not torch.equal(
            padding_pages, padding_pages[:, :1].expand_as(padding_pages)
        ):
            raise RuntimeError("padding row does not bind one private scratch page")
    torch.cuda.synchronize(arena.device)
    return active, padding


def _views(body: Any, active: tuple[int, ...], pages: int) -> Snapshot:
    logical_m = len(active)
    rows = torch.tensor(active, dtype=torch.int64, device=body.hidden.device)
    values: Snapshot = {
        "hidden": body.hidden.index_select(0, rows),
        "kv_key": body.arena.key[: logical_m * pages],
        "kv_value": body.arena.value[: logical_m * pages],
        "kv_written": body.arena.written[: logical_m * pages],
        "conv": body.arena.states[CONV_STATE_ID][:logical_m],
        "recurrent": body.arena.states[RECURRENT_STATE_ID][:logical_m],
        "guard": body.buffers.guard,
    }
    if body.logits is not None:
        values["logits"] = body.logits.index_select(0, rows)
    return values


def _snapshot(body: Any, active: tuple[int, ...], pages: int) -> Snapshot:
    # C512 rank-local recurrent state is too large for a second GPU image.  The
    # oracle image lives on host; comparison copies one candidate field at a
    # time, preserving exact bits without inflating candidate device memory.
    return {
        name: tensor.detach().to(device="cpu")
        for name, tensor in _views(body, active, pages).items()
    }


def _bit_equal(left: Tensor, right: Tensor) -> bool:
    return (
        left.shape == right.shape
        and left.dtype is right.dtype
        and torch.equal(
            left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)
        )
    )


def _finite(tensor: Tensor, chunk: int = 8 << 20) -> bool:
    if not tensor.is_floating_point():
        return True
    flat = tensor.reshape(-1)
    return all(
        bool(torch.isfinite(flat[start : start + chunk]).all().item())
        for start in range(0, flat.numel(), chunk)
    )


def _first_difference(left: Tensor, right: Tensor, chunk: int = 8 << 20) -> dict[str, object]:
    if left.shape != right.shape or left.dtype is not right.dtype:
        return {
            "expected_shape": list(left.shape),
            "actual_shape": list(right.shape),
            "expected_dtype": str(left.dtype),
            "actual_dtype": str(right.dtype),
        }
    expected, actual = left.reshape(-1), right.reshape(-1)
    width = left.element_size()
    for start in range(0, expected.numel(), chunk):
        stop = min(expected.numel(), start + chunk)
        lhs = expected[start:stop].contiguous().view(torch.uint8).reshape(-1, width)
        rhs = actual[start:stop].contiguous().view(torch.uint8).reshape(-1, width)
        rows = torch.any(lhs != rhs, dim=1)
        if bool(rows.any().item()):
            flat = start + int(torch.nonzero(rows, as_tuple=False)[0].item())
            index = tuple(int(v) for v in torch.unravel_index(torch.tensor(flat), left.shape))
            return {
                "index": index,
                "expected": expected[flat].item(),
                "actual": actual[flat].item(),
            }
    return {"error": "digest mismatch without a located element"}


def _compare(expected: Snapshot, actual: Snapshot) -> dict[str, object]:
    if set(expected) != set(actual):
        return {"passed": False, "error": "snapshot fields differ"}
    fields: dict[str, object] = {}
    passed = True
    for name in expected:
        finite = _finite(actual[name])
        candidate = actual[name].detach().to(device=expected[name].device)
        exact = _bit_equal(expected[name], candidate)
        item: dict[str, object] = {"bit_exact": exact, "finite": finite}
        if not exact:
            item["first_difference"] = _first_difference(expected[name], candidate)
        fields[name] = item
        passed = passed and exact and finite
        del candidate
    return {"passed": passed, "fields": fields}


def _padding_outputs(body: Any, padding: tuple[int, ...]) -> Snapshot:
    rows = torch.tensor(padding, dtype=torch.int64, device=body.hidden.device)
    values = {"hidden": body.hidden.index_select(0, rows).to(device="cpu")}
    if body.logits is not None:
        values["logits"] = body.logits.index_select(0, rows).to(device="cpu")
    return values


def _audit_bucket(
    api: _PerformanceAPI,
    admission: _Admission,
    weights: Any,
    runtime: Any,
    captured_m: int,
    context_length: int,
) -> dict[str, object]:
    pages = context_length // _BLOCK
    body, resource = _build_body(api, admission, weights, runtime, captured_m, pages)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    started = time.monotonic()
    slot = api.slot.capture(body, resource)

    active, _ = _prepare(
        body, resource, logical_m=captured_m,
        context_length=context_length, poison=None,
    )
    slot.replay()
    slot.synchronize()
    torch.distributed.barrier()
    candidate_peak = torch.cuda.max_memory_allocated(runtime.device)
    graph = _snapshot(body, active, pages)
    host_snapshot_bytes = sum(tensor.numel() * tensor.element_size() for tensor in graph.values())
    active, _ = _prepare(
        body, resource, logical_m=captured_m,
        context_length=context_length, poison=None,
    )
    torch.distributed.barrier()
    body.launch()
    torch.cuda.synchronize(runtime.device)
    torch.distributed.barrier()
    candidate_peak = max(candidate_peak, torch.cuda.max_memory_allocated(runtime.device))
    same_m = _compare(graph, _views(body, active, pages))
    del graph

    lower_m = captured_m // 2
    active, padding = _prepare(
        body, resource, logical_m=lower_m,
        context_length=context_length, poison=0,
    )
    slot.replay()
    slot.synchronize()
    torch.distributed.barrier()
    candidate_peak = max(candidate_peak, torch.cuda.max_memory_allocated(runtime.device))
    first = _snapshot(body, active, pages)
    host_snapshot_bytes = max(
        host_snapshot_bytes,
        sum(tensor.numel() * tensor.element_size() for tensor in first.values()),
    )
    first_padding = _padding_outputs(body, padding)
    active, padding = _prepare(
        body, resource, logical_m=lower_m,
        context_length=context_length, poison=1,
    )
    slot.replay()
    slot.synchronize()
    torch.distributed.barrier()
    candidate_peak = max(candidate_peak, torch.cuda.max_memory_allocated(runtime.device))
    poison = _compare(first, _views(body, active, pages))
    second_padding = _padding_outputs(body, padding)
    padding_changed = {
        name: all(
            not _bit_equal(first_padding[name][row], second_padding[name][row])
            for row in range(len(padding))
        )
        for name in first_padding
    }
    poison["padding_outputs_changed"] = padding_changed
    poison["all_padding_outputs_changed"] = all(padding_changed.values())
    poison["passed"] = bool(poison["passed"]) and all(padding_changed.values())
    passed = bool(same_m["passed"]) and bool(poison["passed"])
    report = {
        "captured_m": captured_m,
        "bucket_logical_ms": [lower_m, captured_m],
        "status": "pass" if passed else "fail",
        "fallback": False,
        "context_tokens": context_length,
        "active_pages_per_row": pages,
        "padding_pages_per_row": 1,
        "graph_vs_same_body_eager": same_m,
        "lower_m_dual_padding_poison": poison,
        "seconds": round(time.monotonic() - started, 3),
        "candidate_peak_allocated_gib": round(candidate_peak / _GIB, 3),
        "harness_peak_allocated_gib": round(
            torch.cuda.max_memory_allocated(runtime.device) / _GIB, 3
        ),
        "host_exact_snapshot_gib_max": round(host_snapshot_bytes / _GIB, 3),
    }
    del first, first_padding, second_padding, slot, body
    gc.collect()
    torch.cuda.synchronize(runtime.device)
    torch.cuda.empty_cache()
    return report


def _canonical_expected(
    body: Any, cache: Qwen3_5DistributedCache, logits: Tensor | None
) -> Snapshot:
    expected: Snapshot = {
        "kv_key": torch.zeros_like(body.arena.key[:1]),
        "kv_value": torch.zeros_like(body.arena.value[:1]),
        "kv_written": torch.zeros_like(body.arena.written[:1]),
        "conv": torch.zeros_like(body.arena.states[CONV_STATE_ID][:1]),
        "recurrent": torch.zeros_like(body.arena.states[RECURRENT_STATE_ID][:1]),
    }
    full_local = {layer: offset for offset, layer in enumerate(body.full_layer_ids)}
    gdn_local = {layer: offset for offset, layer in enumerate(body.gdn_layer_ids)}
    for layer in body.layer_ids:
        source = cache.layers[layer - cache.layer_start]
        if layer in full_local:
            assert source.key is not None and source.value is not None
            local = full_local[layer]
            expected["kv_key"][0, local, 0].copy_(source.key[0, :, 0])
            expected["kv_value"][0, local, 0].copy_(source.value[0, :, 0])
            expected["kv_written"][0, local, 0] = True
        else:
            assert source.convolution is not None and source.recurrent is not None
            local = gdn_local[layer]
            expected["conv"][0, local].copy_(source.convolution[0])
            expected["recurrent"][0, local].copy_(source.recurrent[0])
    if body.logits is not None:
        assert logits is not None and body.vocab_span is not None
        expected["logits"] = logits[
            0, 0, body.vocab_span.start : body.vocab_span.stop
        ].unsqueeze(0)
    return expected


def _canonical_compare(
    api: _PerformanceAPI,
    admission: _Admission,
    weights: Any,
    runtime: Any,
) -> dict[str, object]:
    reference = Qwen3_5DistributedReferenceModel(
        config=admission.config,
        weights=weights,
        parallel_plan=admission.parallel_plan,
        runtime=runtime,
    )
    with ReferenceExecutionContext():
        step = reference.forward(
            torch.tensor([[_CANONICAL_TOKEN]], dtype=torch.int64, device=runtime.device),
            epoch_id="tp2pp2-performance-canonical-m1",
        )
    body, resource = _build_body(api, admission, weights, runtime, 2, 1, "canonical")
    active, _ = _prepare(
        body, resource, logical_m=1,
        context_length=_BLOCK, poison=0, empty_context=True,
    )
    torch.distributed.barrier()
    body.launch()
    torch.cuda.synchronize(runtime.device)
    torch.distributed.barrier()
    actual_all = _views(body, active, 1)
    expected = _canonical_expected(body, step.cache, step.logits)
    actual = {name: actual_all[name] for name in expected}
    comparison = _compare(expected, actual)
    exact = bool(comparison["passed"])
    result: dict[str, object] = {
        "status": "exact" if exact else "reference_unqualified",
        "canonical_difference_class": "exact" if exact else "reduction_order",
        "reduction_order_qualified": exact,
        "comparison": comparison,
        "blockers": [] if exact else [
            {
                "code": "missing_row_parallel_leaf_fp64_enclosures",
                "required": ["shape", "dtype", "tp_split", "operands", "reduction_indices"],
            },
            {
                "code": "missing_leaf_trace_binding",
                "required": ["op_signature_digest", "candidate_launch_digest", "witness_digest"],
            },
        ],
    }
    del expected, body, reference, step
    gc.collect()
    torch.cuda.synchronize(runtime.device)
    torch.cuda.empty_cache()
    return result


def _worker(
    rank: int,
    init_method: str,
    report_directory: str,
    admission: _Admission,
    captured_ms: tuple[int, ...],
    context_length: int,
    run_canonical: bool,
) -> None:
    report_path = Path(report_directory) / f"rank-{rank}.json"
    try:
        api = _load_performance_api()
        configure_reference_math()
        with initialize_eager_nccl(
            admission.sealed, rank=rank, init_method=init_method, timeout_seconds=1800
        ) as runtime:
            weights = load_qwen3_5_rank_weights(
                inventory=admission.inventory,
                weight_plan=admission.weight_plan,
                parallel_plan=admission.parallel_plan,
                rank=rank,
                device=runtime.device,
            )
            resident_gib = weights.resident_bytes / _GIB
            buckets = [
                _audit_bucket(api, admission, weights, runtime, captured_m, context_length)
                for captured_m in captured_ms
            ]
            canonical = (
                _canonical_compare(api, admission, weights, runtime)
                if run_canonical
                else {
                    "status": "not_run",
                    "reduction_order_qualified": False,
                    "blockers": [{"code": "canonical_reference_not_requested"}],
                }
            )
            placement = admission.parallel_plan.topology.placements[rank]
            report = {
                "rank": rank,
                "device": placement.device_index,
                "stage": placement.pipeline_stage,
                "tensor_rank": placement.tensor_rank,
                "resident_gib": round(resident_gib, 3),
                "fallback": False,
                "canonical_reference": canonical,
                "buckets": buckets,
            }
        report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    except BaseException:
        report_path.write_text(
            json.dumps({"rank": rank, "fallback": False, "error": traceback.format_exc()}),
            encoding="utf-8",
        )
        raise


def _spawn_with_timeout(target: Any, args: tuple[Any, ...], *, timeout: float) -> None:
    context = cast(Any, spawn)(target, args=args, nprocs=4, join=False)
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"TP2 x PP2 run exceeded {timeout:.0f}s")
            if context.join(timeout=remaining, grace_period=10):
                return
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()


def _read_reports(directory: Path) -> list[dict[str, object]]:
    reports = []
    for rank in range(4):
        path = directory / f"rank-{rank}.json"
        reports.append(
            json.loads(path.read_text("utf-8"))
            if path.is_file()
            else {"rank": rank, "fallback": False, "error": "worker produced no report"}
        )
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=_CHECKPOINT)
    parser.add_argument(
        "--devices", type=lambda value: _csv_ints(value, allowed=tuple(range(64))),
        default=(0, 1, 2, 3),
    )
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--pp-size", type=int, default=2)
    parser.add_argument(
        "--captured-ms", type=lambda value: _csv_ints(value, allowed=_CAPTURED_MS),
        default=_CAPTURED_MS,
    )
    parser.add_argument("--context-length", type=int, default=_CONTEXT)
    parser.add_argument("--skip-canonical", action="store_true")
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    if (args.tp_size, args.pp_size) not in ((2, 2), (4, 1)):
        parser.error("four-rank audit supports only TP2 x PP2 or TP4 x PP1")
    if len(args.devices) != 4 or len(set(args.devices)) != 4:
        parser.error("four-rank audit requires four unique CUDA devices")
    if args.context_length <= 0 or args.context_length % _BLOCK:
        parser.error("context-length must be a positive multiple of 16")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    if not (args.checkpoint / "model.safetensors.index.json").is_file():
        _emit({"verdict": "error", "fallback": False, "error": "checkpoint unavailable"})
        return 1
    try:
        _load_performance_api()
    except RuntimeError as error:
        _emit({"verdict": "not_ready", "fallback": False, "error": str(error)})
        return 3
    if not torch.distributed.is_nccl_available() or torch.cuda.device_count() <= max(args.devices):
        _emit(
            {
                "verdict": "resource_rejected",
                "fallback": False,
                "error": "four requested NCCL CUDA devices are unavailable",
            }
        )
        return 4

    started = time.monotonic()
    try:
        admission = _admit(args.checkpoint, args.devices, args.tp_size, args.pp_size)
        with tempfile.TemporaryDirectory(prefix="cubic-qwen-tp2-pp2-audit-") as raw:
            directory = Path(raw)
            try:
                _spawn_with_timeout(
                    _worker,
                    (directory.joinpath("store").resolve().as_uri(), str(directory), admission,
                     args.captured_ms, args.context_length, not args.skip_canonical),
                    timeout=args.timeout,
                )
            except BaseException as error:
                reports = _read_reports(directory)
                _emit({
                    "verdict": "error", "fallback": False,
                    "error": f"{type(error).__name__}: {error}", "ranks": reports,
                })
                return 1
            reports = _read_reports(directory)
    except BaseException as error:
        _emit({"verdict": "error", "fallback": False, "error": f"{type(error).__name__}: {error}"})
        return 1

    graph_passed = all(
        all(
            cast(dict[str, object], bucket).get("status") == "pass"
            for bucket in cast(list[object], report.get("buckets", []))
        )
        for report in reports
    )
    canonical_exact = all(
        cast(dict[str, object], report.get("canonical_reference", {})).get("status") == "exact"
        for report in reports
    )
    verdict = "pass" if graph_passed and canonical_exact else (
        "reference_unqualified" if graph_passed else "fail"
    )
    _emit({
        "verdict": verdict,
        "fallback": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "topology": {
            "tp": args.tp_size, "pp": args.pp_size, "ep": 1,
            "devices": args.devices,
        },
        "captured_ms": args.captured_ms,
        "context_tokens": args.context_length,
        "graph_guardrail_passed": graph_passed,
        "canonical_reference_qualified": canonical_exact,
        "seconds": round(time.monotonic() - started, 3),
        "ranks": reports,
    })
    return 0 if verdict == "pass" else 2


if __name__ == "__main__":
    sys.exit(main())
