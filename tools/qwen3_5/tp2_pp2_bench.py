"""Measure real four-GPU Qwen3.8-27B graph decode throughput.

Each result is one global logical batch, so aggregate tokens/s is ``M /`` the
slowest-rank wall latency.  The tool never substitutes eager execution and
never prints a measurement for a rejected capture.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, cast

import torch
from tp2_pp2_audit import (
    _Admission,
    _admit,
    _build_body,
    _csv_ints,
    _emit,
    _finite,
    _load_performance_api,
    _prepare,
    _read_reports,
    _spawn_with_timeout,
)

from cubic_serve.executor.parallel import initialize_eager_nccl
from cubic_serve.models.qwen3_5 import load_qwen3_5_rank_weights

_CHECKPOINT = Path("/pubdata/llm_models/Qwen/Qwen3.8-27B")
_LOGICAL_MS = tuple(range(1, 129))
_CAPTURED_MS = (1, 2, 4, 8, 16, 32, 64, 128)
_CONTEXT = 128
_BLOCK = 16
_GIB = 1 << 30


def _aggregate_tokens_per_second(logical_m: int, latency_ms: float) -> float:
    if logical_m <= 0 or latency_ms <= 0:
        raise ValueError("logical M and latency must be positive")
    return 1000.0 * logical_m / latency_ms


def _captured_m(logical_m: int) -> int:
    return next(size for size in _CAPTURED_MS if logical_m <= size)


def _measure(
    slot: Any,
    body: Any,
    resource: Any,
    logical_m: int,
    *,
    context_length: int,
    warmup: int,
    iterations: int,
    rounds: int,
) -> dict[str, object]:
    _prepare(
        body,
        resource,
        logical_m=logical_m,
        context_length=context_length,
        poison=None if logical_m == body.captured_m else 0,
        active_layout="prefix",
    )
    for _ in range(warmup):
        slot.replay()
    slot.synchronize()
    torch.distributed.barrier()
    samples: list[float] = []
    for _ in range(rounds):
        torch.distributed.barrier()
        started = time.perf_counter()
        for _ in range(iterations):
            slot.replay()
        slot.synchronize()
        elapsed_ms = 1000.0 * (time.perf_counter() - started) / iterations
        critical = torch.tensor(elapsed_ms, dtype=torch.float64, device=body.hidden.device)
        torch.distributed.all_reduce(critical, op=torch.distributed.ReduceOp.MAX)
        samples.append(float(critical.item()))
    output = body.logits if body.logits is not None else body.hidden
    if not _finite(output):
        raise RuntimeError("benchmark produced a non-finite rank-local output")
    latency = statistics.median(samples)
    return {
        "status": "measured",
        "logical_m": logical_m,
        "captured_m": body.captured_m,
        "bucket_logical_ms": list(body.bucket_logical_ms),
        "context_tokens": context_length,
        "active_pages_per_row": context_length // _BLOCK,
        "latency_ms": round(latency, 4),
        "aggregate_decode_tokens_per_second": round(
            _aggregate_tokens_per_second(logical_m, latency), 2
        ),
        "critical_path_round_samples_ms": [round(value, 4) for value in samples],
        "fallback": False,
    }


def _worker(
    rank: int,
    init_method: str,
    report_directory: str,
    admission: _Admission,
    logical_ms: tuple[int, ...],
    context_length: int,
    warmup: int,
    iterations: int,
    rounds: int,
) -> None:
    report_path = Path(report_directory) / f"rank-{rank}.json"
    try:
        print(f"rank={rank} stage=nccl-start", file=sys.stderr, flush=True)
        api = _load_performance_api()
        with initialize_eager_nccl(
            admission.sealed, rank=rank, init_method=init_method, timeout_seconds=1800
        ) as runtime:
            print(f"rank={rank} stage=weights-start", file=sys.stderr, flush=True)
            weights = load_qwen3_5_rank_weights(
                inventory=admission.inventory,
                weight_plan=admission.weight_plan,
                parallel_plan=admission.parallel_plan,
                rank=rank,
                device=runtime.device,
            )
            print(f"rank={rank} stage=weights-ready", file=sys.stderr, flush=True)
            measured: dict[int, dict[str, object]] = {}
            for captured_m in sorted({_captured_m(value) for value in logical_ms}):
                print(
                    f"rank={rank} stage=build-c{captured_m}", file=sys.stderr, flush=True
                )
                torch.cuda.reset_peak_memory_stats(runtime.device)
                body, resource = _build_body(
                    api, admission, weights, runtime, captured_m,
                    context_length // _BLOCK, "bench",
                )
                capture_started = time.monotonic()
                slot = api.slot.capture(body, resource)
                print(
                    f"rank={rank} stage=capture-ready-c{captured_m}",
                    file=sys.stderr,
                    flush=True,
                )
                # Prove every non-C1 graph covers a lower logical M.
                lower_m = 1 if captured_m == 1 else captured_m // 2
                _prepare(
                    body, resource, logical_m=lower_m,
                    context_length=context_length,
                    poison=None if captured_m == 1 else 0,
                    active_layout="prefix",
                )
                print(
                    f"rank={rank} stage=proof-prepared-c{captured_m}",
                    file=sys.stderr,
                    flush=True,
                )
                slot.replay()
                print(
                    f"rank={rank} stage=proof-enqueued-c{captured_m}",
                    file=sys.stderr,
                    flush=True,
                )
                slot.synchronize()
                print(
                    f"rank={rank} stage=proof-synchronized-c{captured_m}",
                    file=sys.stderr,
                    flush=True,
                )
                torch.distributed.barrier()
                print(
                    f"rank={rank} stage=proof-ready-c{captured_m}",
                    file=sys.stderr,
                    flush=True,
                )
                capture_seconds = round(time.monotonic() - capture_started, 3)
                for logical_m in (
                    value for value in logical_ms if _captured_m(value) == captured_m
                ):
                    result = _measure(
                        slot, body, resource, logical_m,
                        context_length=context_length, warmup=warmup,
                        iterations=iterations, rounds=rounds,
                    )
                    allocated = torch.cuda.memory_allocated(runtime.device)
                    result.update(
                        {
                            "capture_seconds": capture_seconds,
                            "memory": {
                                "allocated_gib": round(allocated / _GIB, 3),
                                "peak_allocated_gib": round(
                                    torch.cuda.max_memory_allocated(runtime.device) / _GIB, 3
                                ),
                                "reserved_gib": round(
                                    torch.cuda.memory_reserved(runtime.device) / _GIB, 3
                                ),
                                "peak_reserved_gib": round(
                                    torch.cuda.max_memory_reserved(runtime.device) / _GIB, 3
                                ),
                            },
                        }
                    )
                    measured[logical_m] = result
                del slot, body
                gc.collect()
                torch.cuda.synchronize(runtime.device)
                torch.cuda.empty_cache()
            results = [measured[value] for value in logical_ms]
            placement = admission.parallel_plan.topology.placements[rank]
            report = {
                "rank": rank,
                "device": placement.device_index,
                "stage": placement.pipeline_stage,
                "tensor_rank": placement.tensor_rank,
                "resident_weight_gib": round(weights.resident_bytes / _GIB, 3),
                "fallback": False,
                "results": results,
            }
        report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    except BaseException:
        report_path.write_text(
            json.dumps({"rank": rank, "fallback": False, "error": traceback.format_exc()}),
            encoding="utf-8",
        )
        raise


def _aggregate(
    reports: list[dict[str, object]], logical_ms: tuple[int, ...]
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for index, logical_m in enumerate(logical_ms):
        rank_results = [
            cast(list[dict[str, object]], report["results"])[index]
            for report in reports
        ]
        if any(item.get("status") != "measured" for item in rank_results):
            results.append({"logical_m": logical_m, "status": "rejected", "fallback": False})
            continue
        latency = max(cast(float, item["latency_ms"]) for item in rank_results)
        results.append(
            {
                "status": "measured",
                "logical_m": logical_m,
                "captured_m": cast(int, rank_results[0]["captured_m"]),
                "bucket_logical_ms": cast(
                    list[int], rank_results[0]["bucket_logical_ms"]
                ),
                "context_tokens": cast(int, rank_results[0]["context_tokens"]),
                "latency_ms": round(latency, 4),
                "aggregate_decode_tokens_per_second": round(
                    _aggregate_tokens_per_second(logical_m, latency), 2
                ),
                "capture_seconds_max": max(
                    cast(float, item["capture_seconds"]) for item in rank_results
                ),
                "rank_memory": [
                    {"rank": report["rank"], **cast(dict[str, object], item["memory"])}
                    for report, item in zip(reports, rank_results, strict=True)
                ],
                "fallback": False,
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=_CHECKPOINT)
    parser.add_argument(
        "--devices", type=lambda value: _csv_ints(value, allowed=tuple(range(64))),
        default=(0, 1, 2, 3),
    )
    parser.add_argument(
        "--logical-ms", type=lambda value: _csv_ints(value, allowed=_LOGICAL_MS),
        default=(1, 4, 8, 16, 32, 64, 128),
    )
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--pp-size", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=_CONTEXT)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    if (args.tp_size, args.pp_size) not in ((2, 2), (4, 1)):
        parser.error("four-rank benchmark supports TP2 x PP2 or TP4 x PP1")
    if len(args.devices) != 4 or len(set(args.devices)) != 4:
        parser.error("four-rank benchmark requires four unique CUDA devices")
    if args.context_length <= 0 or args.context_length % _BLOCK:
        parser.error("context-length must be a positive multiple of 16")
    if min(args.warmup, args.iterations, args.rounds, args.timeout) <= 0:
        parser.error("warmup, iterations, rounds and timeout must be positive")
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
        admission = _admit(
            args.checkpoint, args.devices, args.tp_size, args.pp_size
        )
        with tempfile.TemporaryDirectory(prefix="cubic-qwen-tp2-pp2-bench-") as raw:
            directory = Path(raw)
            try:
                _spawn_with_timeout(
                    _worker,
                    (
                        directory.joinpath("store").resolve().as_uri(), str(directory),
                        admission, args.logical_ms, args.context_length, args.warmup,
                        args.iterations, args.rounds,
                    ),
                    timeout=args.timeout,
                )
            except BaseException as error:
                _emit(
                    {
                        "verdict": "error", "fallback": False,
                        "error": f"{type(error).__name__}: {error}",
                        "ranks": _read_reports(directory),
                    }
                )
                return 1
            reports = _read_reports(directory)
    except BaseException as error:
        _emit({"verdict": "error", "fallback": False, "error": f"{type(error).__name__}: {error}"})
        return 1

    if any("error" in report for report in reports):
        _emit({"verdict": "error", "fallback": False, "ranks": reports})
        return 1
    results = _aggregate(reports, args.logical_ms)
    measured = all(result["status"] == "measured" for result in results)
    _emit(
        {
            "verdict": "measured" if measured else "resource_rejected",
            "mode": "raw_compute",
            "audited": False,
            "serving": False,
            "fallback": False,
            "checkpoint": str(args.checkpoint.resolve()),
            "topology": {
                "tp": args.tp_size,
                "pp": args.pp_size,
                "ep": 1,
                "devices": args.devices,
            },
            "measurement": {
                "warmup": args.warmup,
                "iterations_per_round": args.iterations,
                "rounds": args.rounds,
                "latency_aggregation": "median_of_round_max_rank_completion",
                "throughput": "global_logical_m_per_critical_path_second",
            },
            "seconds": round(time.monotonic() - started, 3),
            "results": results,
            "ranks": [
                {key: report[key] for key in (
                    "rank", "device", "stage", "tensor_rank", "resident_weight_gib"
                )}
                for report in reports
            ],
        }
    )
    return 0 if measured else 2


if __name__ == "__main__":
    sys.exit(main())
