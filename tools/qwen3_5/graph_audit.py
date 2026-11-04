"""Real Qwen3.8-27B CUDA Graph wiring and private-padding audit.

Each captured bucket is compared only with the same-M body. Its lower logical
M is replayed with two private-scratch poisons; cross-M equality is not an oracle.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

from cubic_serve.contracts.execution import RankReplayResources
from cubic_serve.executor.model_runner import (
    _CANARY,
    Qwen3_5DecodeGraphSlot,
    _DecodeBody,
)
from cubic_serve.models.qwen3_5.model import Qwen3_5ReferenceModel, load_qwen3_5_reference
from cubic_serve.ops.cuda.position import MRoPETableEvidence, build_bf16_mrope_tables
from cubic_serve.runtime.model_state import (
    CONV_STATE_ID,
    RECURRENT_STATE_ID,
    DecodeState,
    build_decode_state_schema,
)

_CHECKPOINT = Path("/pubdata/llm_models/Qwen/Qwen3.8-27B")
_CAPTURED_MS = (2, 4, 8, 16, 32, 64)
_CONTEXT, _PAGES = 128, 8
type _Snapshot = dict[str, Tensor]
type _MRoPETables = tuple[Tensor, Tensor, MRoPETableEvidence]


def _captured_ms(value: str) -> tuple[int, ...]:
    try:
        selected = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("captured Ms must be comma-separated integers") from error
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(item not in _CAPTURED_MS for item in selected)
    ):
        raise argparse.ArgumentTypeError(f"captured Ms must be unique members of {_CAPTURED_MS}")
    return selected


def _emit(payload: object) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)


def _bytes(tensor: Tensor) -> Tensor:
    return tensor.contiguous().view(torch.uint8).reshape(-1)


def _equal(left: Tensor, right: Tensor) -> bool:
    if left.device != right.device:
        right = right.to(left.device)
    return (
        left.shape == right.shape
        and left.dtype is right.dtype
        and torch.equal(_bytes(left), _bytes(right))
    )


def _compare(expected: _Snapshot, actual: _Snapshot) -> dict[str, bool]:
    return {name: _equal(value, actual[name]) for name, value in expected.items()}


def _finite(snapshot: _Snapshot) -> bool:
    return all(
        bool(torch.isfinite(value).all().item())
        for value in snapshot.values()
        if value.is_floating_point()
    )


def _view(body: _DecodeBody, active: tuple[int, ...]) -> _Snapshot:
    logical_m = len(active)
    rows = torch.tensor(active, dtype=torch.int64, device=body.arena.device)
    return {
        "logits": body.buffers.logits[rows],
        "kv_key": body.arena.key[: logical_m * _PAGES],
        "kv_value": body.arena.value[: logical_m * _PAGES],
        "kv_written": body.arena.written[: logical_m * _PAGES],
        "conv": body.arena.states[CONV_STATE_ID][:logical_m],
        "recurrent": body.arena.states[RECURRENT_STATE_ID][:logical_m],
        "guard": body.buffers.guard,
    }


def _snapshot(body: _DecodeBody, active: tuple[int, ...]) -> _Snapshot:
    return {name: value.to("cpu") for name, value in _view(body, active).items()}


def _prepare(
    body: _DecodeBody, logical_m: int, poison: int | None
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    buffers, arena = body.buffers, body.arena
    captured_m = buffers.token_ids.shape[0]
    if logical_m == captured_m:
        active = tuple(range(captured_m))
    elif logical_m * 2 == captured_m:
        active = tuple(range(0, captured_m, 2))
    else:
        raise ValueError("logical M must be captured M or its lower half")
    padding = tuple(row for row in range(captured_m) if row not in active)
    if (poison is None) != (not padding):
        raise ValueError("full-M has no poison and lower-M requires one")

    arena.key.zero_()
    arena.value.zero_()
    arena.written.zero_()
    for state in arena.states.values():
        state.zero_()
    buffers.token_ids.zero_()
    buffers.position_ids.zero_()
    buffers.guard.copy_(torch.tensor(_CANARY, dtype=torch.int64, device=arena.device))

    owner_pages = logical_m * _PAGES
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

    scratch_page = arena.spec.page_capacity
    scratch_state = arena.spec.state_slot_capacity
    page_rows: dict[int, tuple[int, ...]] = {}
    state_rows: dict[int, int] = {}
    for logical, physical in enumerate(active):
        page_rows[physical] = tuple(range(logical * _PAGES, (logical + 1) * _PAGES))
        state_rows[physical] = logical
        buffers.token_ids[physical].fill_(17 + logical % 97)
        buffers.position_ids[:, physical].fill_(_CONTEXT - 1)
    if padding:
        assert poison is not None
        for index, physical in enumerate(padding):
            page_rows[physical] = (scratch_page + index,) * _PAGES
            state_rows[physical] = scratch_state + index
            buffers.token_ids[physical].fill_(1001 + 997 * poison + index)
            buffers.position_ids[:, physical].fill_(31 + 32 * poison)

    layers = len(body.gdn_layers)
    token_offset = (_CONTEXT - 1) % arena.layout.block_size_tokens
    for physical in range(captured_m):
        pages, state_slot = page_rows[physical], state_rows[physical]
        buffers.page_source[physical].fill_(pages[-1])
        buffers.page_destination[physical].fill_(pages[-1])
        buffers.token_offsets[physical].fill_(token_offset if physical in active else 0)
        buffers.sequence_lengths[physical].fill_(_CONTEXT if physical in active else 1)
        buffers.page_table[physical].copy_(
            torch.tensor(pages, dtype=torch.int32, device=arena.device)
        )
        for local in range(layers):
            aggregate = state_slot * layers + local
            buffers.conv_slots[:, local, physical].fill_(aggregate)
            buffers.recurrent_slots[:, local, physical].fill_(aggregate)

    if padding:
        assert poison is not None
        base = (37.0, -73.0)[poison]
        page_slice = slice(scratch_page, scratch_page + len(padding))
        state_slice = slice(scratch_state, scratch_state + len(padding))
        arena.key[page_slice].fill_(base)
        arena.value[page_slice].fill_(-base)
        arena.written[page_slice].fill_(True)
        arena.states[CONV_STATE_ID][state_slice].fill_(base / 4)
        arena.states[RECURRENT_STATE_ID][state_slice].fill_(base / 16)

    tables = buffers.page_table.long()
    if any(int(torch.unique(tables[row]).numel()) != _PAGES for row in active):
        raise RuntimeError("every active row must bind eight distinct context pages")
    if any(int(torch.unique(tables[row]).numel()) != 1 for row in padding):
        raise RuntimeError("every padding row must bind one private scratch page")
    if not bool(arena.written[tables].all().item()):
        raise RuntimeError("every active context and padding scratch page must be written")
    torch.cuda.synchronize(arena.device)
    return active, padding


@torch.inference_mode()
def _audit_bucket(
    model: Qwen3_5ReferenceModel, captured_m: int, tables: _MRoPETables
) -> dict[str, object]:
    device = model.weights.device
    schema = build_decode_state_schema(
        model.config,
        arena_id=f"audit-m{captured_m}",
        page_capacity=captured_m * _PAGES,
        state_capacity=captured_m,
        scratch_capacity=captured_m,
        scratch_arena_ids=(f"audit-scratch-m{captured_m}",),
        max_pages=_PAGES,
    )
    torch.cuda.reset_peak_memory_stats(device)
    state = DecodeState(schema, device)
    started = time.monotonic()
    slot = Qwen3_5DecodeGraphSlot.capture(
        model=model,
        arena=state.arena,
        resource=RankReplayResources(
            0,
            f"audit-graph-m{captured_m}",
            f"audit-stream-m{captured_m}",
            f"audit-buffers-m{captured_m}",
            f"audit-workspace-m{captured_m}",
            f"audit-scratch-m{captured_m}",
        ),
        captured_m=captured_m,
        table_pages=_PAGES,
        mrope_tables=tables,
    )
    slot.synchronize()
    body = slot.body

    active, _ = _prepare(body, captured_m, None)
    slot.replay()
    slot.synchronize()
    graph = _snapshot(body, active)
    graph_finite = _finite(graph)
    active, _ = _prepare(body, captured_m, None)
    body.launch()
    torch.cuda.synchronize(device)
    direct = _view(body, active)
    same_m = _compare(graph, direct)
    same_m_passed = all(same_m.values()) and graph_finite
    del graph, direct

    lower_m = captured_m // 2
    active, padding = _prepare(body, lower_m, 0)
    slot.replay()
    slot.synchronize()
    first = _snapshot(body, active)
    first_padding = body.buffers.logits[list(padding)].clone()
    first_finite = _finite(first)
    active, padding = _prepare(body, lower_m, 1)
    slot.replay()
    slot.synchronize()
    second = _view(body, active)
    poison_exact = _compare(first, second)
    padding_changed = all(
        not _equal(first_padding[row], body.buffers.logits[physical])
        for row, physical in enumerate(padding)
    )
    poison_passed = all(poison_exact.values()) and first_finite and padding_changed
    passed = same_m_passed and poison_passed
    return {
        "captured_m": captured_m,
        "logical_ms": [lower_m, captured_m],
        "status": "pass" if passed else "fail",
        "fallback": False,
        "capture_and_audit_seconds": round(time.monotonic() - started, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device) / (1 << 30), 3),
        "graph_vs_same_m": {
            "passed": same_m_passed,
            "finite": graph_finite,
            "exact": same_m,
        },
        "lower_m_dual_padding_poison": {
            "passed": poison_passed,
            "active_finite": first_finite,
            "active_exact": poison_exact,
            "all_padding_outputs_changed": padding_changed,
            "padding_pages_per_row": 1,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--captured-ms", type=_captured_ms, default=_CAPTURED_MS)
    args = parser.parse_args()
    if not args.checkpoint.is_dir():
        _emit({"verdict": "error", "error": f"checkpoint unavailable: {args.checkpoint}"})
        return 1
    started = time.monotonic()
    try:
        model = load_qwen3_5_reference(args.checkpoint, device=args.device)
        text, device = model.config.graph.text, model.weights.device
        rotary = int(text.full_attention.head_dim * text.mrope.partial_rotary_factor)
        tables = build_bf16_mrope_tables(
            max_position=text.max_position_embeddings,
            device=device,
            rotary_dim=rotary,
            section=text.mrope.section,
            rope_theta=text.mrope.rope_theta,
        )
        buckets: list[dict[str, object]] = []
        for captured_m in args.captured_ms:
            try:
                buckets.append(_audit_bucket(model, captured_m, tables))
            except torch.OutOfMemoryError as error:
                buckets.append(
                    {
                        "captured_m": captured_m,
                        "logical_ms": [captured_m // 2, captured_m],
                        "status": "resource_rejected",
                        "fallback": False,
                        "error": f"{type(error).__name__}: {str(error).splitlines()[0]}",
                    }
                )
            except Exception as error:
                buckets.append(
                    {
                        "captured_m": captured_m,
                        "logical_ms": [captured_m // 2, captured_m],
                        "status": "error",
                        "fallback": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            finally:
                gc.collect()
                torch.cuda.empty_cache()
        passed = all(bucket["status"] == "pass" for bucket in buckets)
        result = {
            "verdict": "pass" if passed else "fail",
            "checkpoint": str(args.checkpoint.resolve()),
            "device": torch.cuda.get_device_name(device),
            "context_tokens": _CONTEXT,
            "active_pages_per_row": _PAGES,
            "padding_pages_per_row": 1,
            "cross_m_exact_oracle": False,
            "total_seconds": round(time.monotonic() - started, 3),
            "buckets": buckets,
        }
    except Exception as error:
        _emit({"verdict": "error", "error": f"{type(error).__name__}: {error}"})
        return 1
    _emit(result)
    return 0 if result["verdict"] == "pass" else 2


if __name__ == "__main__":
    sys.exit(main())
