"""Raw Qwen3.8 decode CUDA Graph throughput; not serving or audited evidence.

Bucket selection follows the multi-size capture-table shape used by current
vLLM/SGLang runners: a logical M selects a sealed captured-M lane.  Every lane
declares and exercises two logical sizes; there is no singleton/exact-M lane and
no eager fallback.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from cubic_serve.contracts.execution import RankReplayResources
from cubic_serve.executor.model_runner import Qwen3_5DecodeGraphSlot
from cubic_serve.models.qwen3_5.model import load_qwen3_5_reference
from cubic_serve.ops.cuda.position import build_bf16_mrope_tables
from cubic_serve.runtime.model_state import DecodeState, build_decode_state_schema

_DEFAULT_CHECKPOINT = Path("/pubdata/llm_models/Qwen/Qwen3.8-27B")
_LOGICAL_MS = (1, 4, 8, 16, 32, 64, 128, 256, 512)
_GIB = 1 << 30


@dataclass(frozen=True, slots=True)
class _Bucket:
    captured_m: int
    logical_ms: tuple[int, int]


def _buckets() -> tuple[_Bucket, ...]:
    buckets = tuple(
        _Bucket(capacity, (capacity // 2, capacity))
        for capacity in (2, 4, 8, 16, 32, 64, 128, 256, 512)
    )
    if any(len(set(bucket.logical_ms)) != 2 for bucket in buckets):
        raise RuntimeError("singleton graph buckets are forbidden")
    return buckets


def _select_bucket(logical_m: int) -> _Bucket:
    captured_m = 2 if logical_m == 1 else logical_m
    found = tuple(bucket for bucket in _buckets() if bucket.captured_m == captured_m)
    if len(found) != 1:
        raise ValueError(f"logical M={logical_m} has no unique multi-M graph bucket")
    return found[0]


def _parse_logical_ms(value: str) -> tuple[int, ...]:
    try:
        requested = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("logical Ms must be comma-separated integers") from error
    if not requested or len(set(requested)) != len(requested) or any(
        item not in _LOGICAL_MS for item in requested
    ):
        raise argparse.ArgumentTypeError(f"logical Ms must be unique members of {_LOGICAL_MS}")
    return requested


def _arena_bytes(schema: object, captured_m: int) -> int:
    arena_spec = schema.arena_spec  # type: ignore[attr-defined]
    layout = schema.kv_layout  # type: ignore[attr-defined]
    del captured_m
    scratch = len(arena_spec.scratch_arena_ids) * arena_spec.scratch_capacity
    page = math.prod(layout.page_shape)
    per_page = 2 * page * 2 + len(layout.layer_ids) * layout.block_size_tokens
    per_state = sum(
        math.prod(spec.shape) * (2 if spec.dtype is torch.bfloat16 else 4)
        for spec in arena_spec.state_specs
    )
    return (arena_spec.page_capacity + scratch) * per_page + (
        arena_spec.state_slot_capacity + scratch
    ) * per_state


def _prepare_rows(
    slot: Qwen3_5DecodeGraphSlot, logical_m: int, context_length: int
) -> tuple[int, ...]:
    body, capacity = slot.body, slot.body.buffers.token_ids.shape[0]
    if logical_m not in (capacity // 2, capacity):
        raise RuntimeError("benchmark attempted a logical M outside its multi-M bucket")
    body.prepare_capture_scratch(slot.resources[0])
    active = tuple(range(capacity)) if logical_m == capacity else tuple(range(0, capacity, 2))
    arena, buffers = body.arena, body.buffers
    pages = context_length // arena.layout.block_size_tokens
    page_values = torch.arange(
        1, arena.spec.page_capacity + 1, dtype=torch.float32, device=arena.device
    ).remainder_(31).add_(1).div_(128).to(torch.bfloat16)
    arena.key[: arena.spec.page_capacity].copy_(
        page_values.view(-1, 1, 1, 1, 1)
    )
    arena.value[: arena.spec.page_capacity].copy_(
        page_values.neg().view(-1, 1, 1, 1, 1)
    )
    arena.written[: arena.spec.page_capacity].fill_(True)
    for state in arena.states.values():
        state[: arena.spec.state_slot_capacity].zero_()
    buffers.token_ids.fill_(body.model.config.graph.text.vocabulary_size - 1)
    buffers.position_ids.fill_(body.model.config.graph.text.max_position_embeddings - 1)
    layers = len(body.gdn_layers)
    for owner, row in enumerate(active):
        first_page = owner * pages
        buffers.token_ids[row].fill_(17 + owner % 97)
        buffers.position_ids[:, row].fill_(context_length - 1)
        buffers.page_source[row].fill_(first_page + pages - 1)
        buffers.page_destination[row].fill_(first_page + pages - 1)
        buffers.token_offsets[row].fill_(
            (context_length - 1) % arena.layout.block_size_tokens
        )
        buffers.sequence_lengths[row].fill_(context_length)
        for offset, target in enumerate(buffers.page_table[row]):
            target.fill_(first_page + offset)
        for local in range(layers):
            state_index = owner * layers + local
            buffers.conv_slots[:, local, row].fill_(state_index)
            buffers.recurrent_slots[:, local, row].fill_(state_index)
    for row in active:
        if (
            int(torch.unique(buffers.page_table[row]).numel()) != pages
            or not bool(arena.written[buffers.page_table[row].long()].all().item())
            or int(buffers.sequence_lengths[row].item()) != context_length
        ):
            raise RuntimeError("active row does not bind a complete distinct-page context")
    padding = tuple(row for row in range(capacity) if row not in active)
    page_boundary = arena.spec.page_capacity
    if any(
        int(buffers.sequence_lengths[row].item()) != 1
        or int(buffers.page_table[row, 0].item()) < page_boundary
        for row in padding
    ):
        raise RuntimeError("padding row escaped private context-1 scratch")
    slot.stream.wait_stream(torch.cuda.current_stream(body.arena.device))
    return active


def _measure(
    slot: Qwen3_5DecodeGraphSlot,
    logical_m: int,
    *,
    context_length: int,
    warmup: int,
    iterations: int,
    rounds: int,
) -> dict[str, object]:
    active = _prepare_rows(slot, logical_m, context_length)
    for _ in range(warmup):
        slot.replay()
    slot.synchronize()
    samples = []
    for _ in range(rounds):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(slot.body.arena.device), torch.cuda.stream(slot.stream):
            begin.record()
            for _ in range(iterations):
                slot.graph.replay()
            end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) / iterations)
    slot.synchronize()
    if not bool(torch.isfinite(slot.body.buffers.logits[list(active)]).all().item()):
        raise RuntimeError("graph benchmark produced non-finite active logits")
    latency = statistics.median(samples)
    return {
        "logical_m": logical_m,
        "captured_m": slot.body.buffers.token_ids.shape[0],
        "bucket_logical_ms": (slot.body.buffers.token_ids.shape[0] // 2,
                              slot.body.buffers.token_ids.shape[0]),
        "context_length": context_length,
        "kv_pages_per_active_row": context_length // slot.body.arena.layout.block_size_tokens,
        "latency_ms": round(latency, 4),
        "decode_tokens_per_second": round(1000.0 * logical_m / latency, 2),
        "round_samples_ms": tuple(round(value, 4) for value in samples),
    }


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=_DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--logical-ms", type=_parse_logical_ms, default=_LOGICAL_MS)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--reserve-gib", type=float, default=2.0)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.rounds) <= 0 or args.reserve_gib < 0:
        parser.error("warmup/iterations/rounds must be positive and reserve-gib nonnegative")
    if args.context_length <= 0 or args.context_length % 16:
        parser.error("context-length must be a positive multiple of the 16-token KV block")
    if not args.checkpoint.is_dir():
        parser.error(f"checkpoint directory is unavailable: {args.checkpoint}")

    started = time.monotonic()
    model = load_qwen3_5_reference(args.checkpoint, device=args.device)
    device, text = model.weights.device, model.config.graph.text
    if args.context_length > text.max_position_embeddings:
        parser.error("context-length exceeds the model position limit")
    context_pages = args.context_length // 16
    rotary = int(text.full_attention.head_dim * text.mrope.partial_rotary_factor)
    mrope = build_bf16_mrope_tables(
        max_position=text.max_position_embeddings,
        device=device,
        rotary_dim=rotary,
        section=text.mrope.section,
        rope_theta=text.mrope.rope_theta,
    )
    torch.cuda.synchronize(device)
    _emit({
        "kind": "environment",
        "audited": False,
        "serving": False,
        "device": torch.cuda.get_device_name(device),
        "checkpoint": str(args.checkpoint.resolve()),
        "model_load_seconds": round(time.monotonic() - started, 3),
        "logical_ms": args.logical_ms,
        "context_length": args.context_length,
        "kv_pages_per_active_row": context_pages,
    })

    by_bucket: dict[int, list[int]] = {}
    for logical_m in args.logical_ms:
        bucket = _select_bucket(logical_m)
        by_bucket.setdefault(bucket.captured_m, []).append(logical_m)

    for captured_m, requested in sorted(by_bucket.items()):
        bucket = _select_bucket(requested[0])
        scratch_id = f"bench-scratch-m{captured_m}"
        schema = build_decode_state_schema(
            model.config,
            arena_id=f"bench-arena-m{captured_m}",
            page_capacity=captured_m * context_pages,
            state_capacity=captured_m,
            scratch_capacity=captured_m,
            scratch_arena_ids=(scratch_id,),
            max_pages=context_pages,
            block_size=16,
        )
        required = _arena_bytes(schema, captured_m)
        free, _total = torch.cuda.mem_get_info(device)
        if required + int(args.reserve_gib * _GIB) > free:
            for logical_m in requested:
                _emit({
                    "kind": "result",
                    "status": "resource_rejected",
                    "logical_m": logical_m,
                    "captured_m": captured_m,
                    "bucket_logical_ms": bucket.logical_ms,
                    "required_arena_gib": round(required / _GIB, 3),
                    "free_gib": round(free / _GIB, 3),
                    "context_length": args.context_length,
                    "kv_pages_per_active_row": context_pages,
                    "fallback": False,
                })
            continue
        state = slot = None
        try:
            torch.cuda.reset_peak_memory_stats(device)
            state = DecodeState(schema, device)
            resource = RankReplayResources(
                0,
                f"bench-graph-m{captured_m}",
                f"bench-stream-m{captured_m}",
                f"bench-buffers-m{captured_m}",
                f"bench-workspace-m{captured_m}",
                scratch_id,
            )
            capture_started = time.monotonic()
            slot = Qwen3_5DecodeGraphSlot.capture(
                model=model,
                arena=state.arena,
                resource=resource,
                captured_m=captured_m,
                table_pages=context_pages,
                mrope_tables=mrope,
            )
            # Exercise both declared logical sizes on this exact graph instance.
            for logical_m in bucket.logical_ms:
                _measure(
                    slot,
                    logical_m,
                    context_length=args.context_length,
                    warmup=1,
                    iterations=1,
                    rounds=1,
                )
            for logical_m in requested:
                result = _measure(
                    slot,
                    logical_m,
                    context_length=args.context_length,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    rounds=args.rounds,
                )
                result.update({
                    "kind": "result",
                    "status": "measured",
                    "capture_seconds": round(time.monotonic() - capture_started, 3),
                    "peak_allocated_gib": round(
                        torch.cuda.max_memory_allocated(device) / _GIB, 3
                    ),
                    "fallback": False,
                })
                _emit(result)
        except (RuntimeError, torch.OutOfMemoryError) as error:
            for logical_m in requested:
                _emit({
                    "kind": "result",
                    "status": "capture_rejected",
                    "logical_m": logical_m,
                    "captured_m": captured_m,
                    "bucket_logical_ms": bucket.logical_ms,
                    "context_length": args.context_length,
                    "kv_pages_per_active_row": context_pages,
                    "error": f"{type(error).__name__}: {error}",
                    "fallback": False,
                })
        finally:
            del slot, state
            gc.collect()
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
