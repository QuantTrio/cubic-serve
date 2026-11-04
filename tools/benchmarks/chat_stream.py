#!/usr/bin/env python3
"""Canonical public streaming Chat Completions serving benchmark.

The metric semantics are triangulated from these pinned upstream sources:

* llama.cpp 0b5be7e4a25862bc2777d0c47eae18788a8c963a,
  ``scripts/server-bench.py``
* vLLM 5e71a11eb2b595ede15ecc39c7dddca38e03deb5,
  ``vllm/benchmarks/serve.py`` and
  ``vllm/benchmarks/lib/endpoint_request_func.py``
* SGLang e51a3ae65e3401b21de860c90a64102133d2d6a6,
  ``python/sglang/benchmark/serving.py``

Only the public OpenAI-compatible streaming endpoint is measured. Internal
executors, raw graph replay, non-streaming requests, and offline batches are not
accepted as serving throughput.

A run requesting at least 30,000 output tokens also fails unless its exact SSE
timeline retains at least 70% of the first 2,048-token rate in the final 2,048.

Device telemetry was separately checked against llama.cpp
``c841aeeb8bb2fe417038dadfa9b007cf1a9ef950`` (the serving benchmark has no
device sampler), vLLM ``73fac1e1597f8f7b4029956982e8653f0cf0f067`` (NVML status
queries avoid a CUDA context), and SGLang
``ba433bb4628f44d5031e910ad32194ad6c145ba8`` (portable ``nvidia-smi`` query and
failure handling). It remains diagnostic evidence over the same public-request
window, not another throughput metric.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import random
import secrets
import statistics
import subprocess
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Protocol

import httpx

CONCURRENCY_LEVELS = (1, 4, 8, 16, 32, 64, 128, 256)
UPSTREAM_REFERENCES = (
    {
        "project": "llama.cpp",
        "commit": "0b5be7e4a25862bc2777d0c47eae18788a8c963a",
        "paths": ["scripts/server-bench.py"],
    },
    {
        "project": "vLLM",
        "commit": "5e71a11eb2b595ede15ecc39c7dddca38e03deb5",
        "paths": [
            "vllm/benchmarks/serve.py",
            "vllm/benchmarks/lib/endpoint_request_func.py",
        ],
    },
    {
        "project": "SGLang",
        "commit": "e51a3ae65e3401b21de860c90a64102133d2d6a6",
        "paths": ["python/sglang/benchmark/serving.py"],
    },
)
DEVICE_TELEMETRY_REFERENCES = (
    {
        "project": "llama.cpp",
        "commit": "c841aeeb8bb2fe417038dadfa9b007cf1a9ef950",
        "paths": ["scripts/server-bench.py"],
        "finding": "public serving timing is present; device telemetry is absent",
    },
    {
        "project": "vLLM",
        "commit": "73fac1e1597f8f7b4029956982e8653f0cf0f067",
        "paths": ["vllm/benchmarks/serve.py", "vllm/utils/import_utils.py"],
        "finding": "public request window plus CUDA-context-free NVML status queries",
    },
    {
        "project": "SGLang",
        "commit": "ba433bb4628f44d5031e910ad32194ad6c145ba8",
        "paths": [
            "python/sglang/benchmark/serving.py",
            "python/sglang/srt/utils/common.py",
        ],
        "finding": "public request window plus guarded nvidia-smi device queries",
    },
)

_NVIDIA_SMI_FIELDS = (
    "index",
    "uuid",
    "name",
    "power.draw",
    "utilization.gpu",
    "utilization.memory",
    "clocks.current.sm",
    "clocks.current.memory",
)
_LONG_GENERATION_TOKENS = 30_000
_LONG_GENERATION_EDGE_TOKENS = 2_048
_MIN_LONG_GENERATION_TAIL_RATIO = 0.70


@dataclass(frozen=True, slots=True)
class RequestSpec:
    ordinal: int
    scheduled_offset_s: float
    prompt: str
    max_output_tokens: int
    kind: str = "regular"
    sampling_seed: int | None = None
    image_data_urls: tuple[str, ...] = ()
    media_data_urls: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class RequestResult:
    ordinal: int
    scheduled_offset_s: float
    started_at: float
    ended_at: float
    output_arrivals: tuple[float, ...]
    input_tokens: int | None
    output_tokens: int | None
    content_characters: int
    status_code: int | None
    saw_done: bool
    error: str | None
    kind: str = "regular"
    sampling_seed: int | None = None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass(frozen=True, slots=True)
class DeviceSample:
    observed_at: float
    index: int
    uuid: str
    name: str
    power_w: float | None
    gpu_utilization_percent: float | None
    memory_activity_percent: float | None
    sm_clock_mhz: float | None
    memory_clock_mhz: float | None


@dataclass(frozen=True, slots=True)
class DeviceSamplingWindow:
    samples: tuple[DeviceSample, ...]
    query_count: int
    failed_query_count: int
    errors: tuple[str, ...]


class DeviceQuery(Protocol):
    def __call__(self, device_indices: tuple[int, ...] | None) -> tuple[DeviceSample, ...]: ...


def _optional_float(value: str) -> float | None:
    stripped = value.strip()
    if not stripped or stripped.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return None
    try:
        parsed = float(stripped)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def parse_nvidia_smi_csv(output: str, *, observed_at: float) -> tuple[DeviceSample, ...]:
    """Parse one portable ``nvidia-smi --query-gpu`` snapshot."""

    samples: list[DeviceSample] = []
    for row in csv.reader(StringIO(output)):
        if not row or all(not item.strip() for item in row):
            continue
        if len(row) != len(_NVIDIA_SMI_FIELDS):
            raise ValueError(
                f"nvidia-smi returned {len(row)} columns; expected {len(_NVIDIA_SMI_FIELDS)}"
            )
        try:
            index = int(row[0].strip())
        except ValueError as exc:
            raise ValueError(f"nvidia-smi returned invalid GPU index {row[0]!r}") from exc
        samples.append(
            DeviceSample(
                observed_at=observed_at,
                index=index,
                uuid=row[1].strip(),
                name=row[2].strip(),
                power_w=_optional_float(row[3]),
                gpu_utilization_percent=_optional_float(row[4]),
                memory_activity_percent=_optional_float(row[5]),
                sm_clock_mhz=_optional_float(row[6]),
                memory_clock_mhz=_optional_float(row[7]),
            )
        )
    if not samples:
        raise ValueError("nvidia-smi returned no GPU rows")
    return tuple(samples)


def query_nvidia_smi(device_indices: tuple[int, ...] | None) -> tuple[DeviceSample, ...]:
    command = ["nvidia-smi"]
    if device_indices is not None:
        command.append("--id=" + ",".join(str(index) for index in device_indices))
    command.extend(
        [
            "--query-gpu=" + ",".join(_NVIDIA_SMI_FIELDS),
            "--format=csv,noheader,nounits",
        ]
    )
    started_at = time.perf_counter()
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    observed_at = (started_at + time.perf_counter()) / 2
    return parse_nvidia_smi_csv(completed.stdout, observed_at=observed_at)


class DeviceSampler:
    """Best-effort device telemetry that never changes request outcomes."""

    def __init__(
        self,
        *,
        device_indices: tuple[int, ...] | None,
        interval_s: float,
        query: DeviceQuery = query_nvidia_smi,
    ) -> None:
        self.device_indices = device_indices
        self.interval_s = interval_s
        self._query = query

    async def _sample_once(self) -> tuple[DeviceSample, ...]:
        return await asyncio.to_thread(self._query, self.device_indices)

    async def collect_for(self, duration_s: float) -> DeviceSamplingWindow:
        stop_at = time.perf_counter() + duration_s
        samples: list[DeviceSample] = []
        errors: list[str] = []
        query_count = 0
        while True:
            query_count += 1
            try:
                samples.extend(await self._sample_once())
            except Exception as exc:  # telemetry is deliberately non-fatal
                errors.append(f"{type(exc).__name__}: {exc}")
            delay = min(self.interval_s, stop_at - time.perf_counter())
            if delay <= 0:
                break
            await asyncio.sleep(delay)
        return DeviceSamplingWindow(
            samples=tuple(samples),
            query_count=query_count,
            failed_query_count=len(errors),
            errors=tuple(dict.fromkeys(errors)),
        )

    async def collect_until(self, stop: asyncio.Event) -> DeviceSamplingWindow:
        samples: list[DeviceSample] = []
        errors: list[str] = []
        query_count = 0
        while not stop.is_set():
            query_count += 1
            try:
                samples.extend(await self._sample_once())
            except Exception as exc:  # telemetry is deliberately non-fatal
                errors.append(f"{type(exc).__name__}: {exc}")
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=self.interval_s)
        return DeviceSamplingWindow(
            samples=tuple(samples),
            query_count=query_count,
            failed_query_count=len(errors),
            errors=tuple(dict.fromkeys(errors)),
        )


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(values: Sequence[float], *, scale: float = 1.0) -> dict[str, float | None]:
    scaled = [value * scale for value in values]
    return {
        "mean": statistics.fmean(scaled) if scaled else None,
        "p50": _percentile(scaled, 50),
        "p90": _percentile(scaled, 90),
        "p99": _percentile(scaled, 99),
        "max": max(scaled) if scaled else None,
    }


def _device_distribution(values: Sequence[float | None]) -> dict[str, float | None]:
    available = [value for value in values if value is not None]
    return {
        "mean": statistics.fmean(available) if available else None,
        "p95": _percentile(available, 95),
        "peak": max(available) if available else None,
    }


def _device_window_summary(samples: Sequence[DeviceSample]) -> dict[str, object]:
    return {
        "sample_count": len(samples),
        "power_w": _device_distribution([sample.power_w for sample in samples]),
        "gpu_utilization_percent": _device_distribution(
            [sample.gpu_utilization_percent for sample in samples]
        ),
        "memory_activity_percent": _device_distribution(
            [sample.memory_activity_percent for sample in samples]
        ),
        "sm_clock_mhz": _device_distribution([sample.sm_clock_mhz for sample in samples]),
        "memory_clock_mhz": _device_distribution([sample.memory_clock_mhz for sample in samples]),
    }


def _mean_metric(window: dict[str, object], name: str) -> float | None:
    metric = window.get(name)
    if not isinstance(metric, dict):
        return None
    value = metric.get("mean")
    return value if isinstance(value, float | int) else None


def _mean_delta(baseline: dict[str, object], request: dict[str, object], name: str) -> float | None:
    idle = _mean_metric(baseline, name)
    active = _mean_metric(request, name)
    return None if idle is None or active is None else active - idle


def summarize_device_sampling(
    *,
    baseline: DeviceSamplingWindow,
    request: DeviceSamplingWindow,
    origin: float,
    ended_at: float,
    interval_s: float,
    requested_device_indices: tuple[int, ...] | None,
) -> dict[str, object]:
    request_samples = tuple(
        sample for sample in request.samples if origin <= sample.observed_at <= ended_at
    )
    identities = {
        sample.index: (sample.uuid, sample.name) for sample in (*baseline.samples, *request_samples)
    }
    devices: list[dict[str, object]] = []
    for index, (uuid, name) in sorted(identities.items()):
        baseline_summary = _device_window_summary(
            [sample for sample in baseline.samples if sample.index == index]
        )
        request_summary = _device_window_summary(
            [sample for sample in request_samples if sample.index == index]
        )

        devices.append(
            {
                "index": index,
                "uuid": uuid,
                "name": name,
                "idle_baseline": baseline_summary,
                "request_window": request_summary,
                "mean_delta_from_idle": {
                    "power_w": _mean_delta(baseline_summary, request_summary, "power_w"),
                    "gpu_utilization_percentage_points": _mean_delta(
                        baseline_summary, request_summary, "gpu_utilization_percent"
                    ),
                    "memory_activity_percentage_points": _mean_delta(
                        baseline_summary, request_summary, "memory_activity_percent"
                    ),
                    "sm_clock_mhz": _mean_delta(baseline_summary, request_summary, "sm_clock_mhz"),
                    "memory_clock_mhz": _mean_delta(
                        baseline_summary, request_summary, "memory_clock_mhz"
                    ),
                },
            }
        )
    errors = tuple(dict.fromkeys((*baseline.errors, *request.errors)))
    if not request_samples:
        status = "unavailable"
    elif not baseline.samples or baseline.failed_query_count or request.failed_query_count:
        status = "partial"
    else:
        status = "ok"
    return {
        "status": status,
        "backend": "nvidia-smi",
        "sampling_interval_s": interval_s,
        "requested_device_indices": (
            "all" if requested_device_indices is None else list(requested_device_indices)
        ),
        "memory_activity_semantics": (
            "NVML utilization.memory: percentage of the sample period with global device "
            "memory reads or writes; it is an activity signal, not achieved bytes/second"
        ),
        "idle_baseline": {
            "query_count": baseline.query_count,
            "failed_query_count": baseline.failed_query_count,
        },
        "request_window": {
            "duration_s": max(ended_at - origin, 0.0),
            "query_count": request.query_count,
            "failed_query_count": request.failed_query_count,
            "in_window_sample_count": len(request_samples),
        },
        "errors": list(errors),
        "devices": devices,
        # Retain the existing observations so ramp-up, full concurrency and
        # retirement can be attributed against this exact SSE timeline. No
        # additional device queries or work enter the request interval.
        "samples": [dict(asdict(sample), offset_s=sample.observed_at - origin)
                    for sample in request_samples],
    }


def arrival_offsets(
    count: int,
    *,
    pattern: str,
    request_rate: float | None,
    seed: int,
) -> tuple[float, ...]:
    """Return deterministic cumulative request offsets for one benchmark level."""

    if count < 1:
        raise ValueError("request count must be positive")
    if pattern == "burst":
        return (0.0,) * count
    if request_rate is None or not math.isfinite(request_rate) or request_rate <= 0:
        raise ValueError("fixed and poisson arrivals require a finite positive request rate")
    if pattern == "fixed":
        return tuple(index / request_rate for index in range(count))
    if pattern == "poisson":
        generator = random.Random(seed)
        offsets = [0.0]
        for _ in range(1, count):
            offsets.append(offsets[-1] + generator.expovariate(request_rate))
        return tuple(offsets)
    raise ValueError(f"unsupported arrival pattern: {pattern!r}")


async def iter_sse_data(response: httpx.Response) -> AsyncIterator[str]:
    """Parse SSE framing without assuming HTTP chunks align with events."""

    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if field != "data":
            continue
        if separator and value.startswith(" "):
            value = value[1:]
        data_lines.append(value)
    if data_lines:
        yield "\n".join(data_lines)


def _token_bearing_delta(delta: object) -> tuple[bool, int]:
    if not isinstance(delta, dict):
        return False, 0
    characters = 0
    for key in ("reasoning_content", "content"):
        value = delta.get(key)
        if isinstance(value, str):
            characters += len(value)
    tool_calls = delta.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            for key in ("name", "arguments"):
                value = function.get(key)
                if isinstance(value, str):
                    characters += len(value)
    return characters > 0, characters


def _natural(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


async def stream_request(
    client: httpx.AsyncClient,
    *,
    api_url: str,
    model: str,
    spec: RequestSpec,
    reasoning_effort: str | None,
) -> RequestResult:
    payload: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": spec.prompt}],
        "max_completion_tokens": spec.max_output_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }
    if spec.image_data_urls or spec.media_data_urls:
        payload["messages"] = [{"role": "user", "content": [
            {"type": "text", "text": spec.prompt},
            *({"type": "image_url", "image_url": {"url": url}}
              for url in spec.image_data_urls),
            *({"type": kind + "_url", kind + "_url": {"url": url}}
              for kind, url in spec.media_data_urls),
        ]}]
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    if spec.sampling_seed is not None:
        payload["seed"] = spec.sampling_seed

    started_at = time.perf_counter()
    ended_at = started_at
    arrivals: list[float] = []
    continuous_count = 0
    saw_continuous = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    content_characters = 0
    status_code: int | None = None
    saw_done = False
    error: str | None = None
    try:
        async with client.stream("POST", api_url, json=payload) as response:
            status_code = response.status_code
            if response.status_code != 200:
                body = (await response.aread()).decode("utf-8", errors="replace")[:4096]
                error = f"HTTP {response.status_code}: {body}"
            else:
                content_type = response.headers.get("content-type", "")
                if "text/event-stream" not in content_type.lower():
                    error = f"expected text/event-stream, got {content_type!r}"
                else:
                    async for data_text in iter_sse_data(response):
                        ended_at = time.perf_counter()
                        if data_text == "[DONE]":
                            saw_done = True
                            continue
                        try:
                            event = json.loads(data_text)
                        except json.JSONDecodeError as exc:
                            error = f"invalid SSE JSON: {exc.msg}"
                            break
                        if not isinstance(event, dict):
                            error = "SSE data must be a JSON object"
                            break
                        if isinstance(event.get("error"), dict):
                            error = "stream error: " + json.dumps(
                                event["error"], ensure_ascii=False
                            )
                            break
                        choices = event.get("choices")
                        usage = event.get("usage")
                        if isinstance(usage, dict):
                            candidate_input = _natural(usage.get("prompt_tokens"))
                            candidate_output = _natural(usage.get("completion_tokens"))
                            if candidate_input is not None:
                                input_tokens = candidate_input
                            if candidate_output is not None:
                                output_tokens = candidate_output
                                if choices:
                                    if candidate_output < continuous_count:
                                        raise ValueError("continuous token count regressed")
                                    arrivals.extend([ended_at] * (candidate_output - continuous_count))
                                    continuous_count = candidate_output
                                    saw_continuous = True
                        choices = event.get("choices")
                        if not isinstance(choices, list) or not choices:
                            continue
                        choice = choices[0]
                        if not isinstance(choice, dict):
                            continue
                        has_output, characters = _token_bearing_delta(choice.get("delta"))
                        if has_output:
                            if not saw_continuous:
                                arrivals.append(ended_at)
                            content_characters += characters
    except Exception as exc:  # the error is part of the benchmark artifact
        error = f"{type(exc).__name__}: {exc}"
    ended_at = time.perf_counter()

    if error is None and not saw_done:
        error = "stream ended without data: [DONE]"
    if error is None and output_tokens is None:
        error = "stream usage omitted completion_tokens"
    if error is None and input_tokens is None:
        error = "stream usage omitted prompt_tokens"
    if error is None and output_tokens == 0:
        error = "request produced zero output tokens"
    if error is None and not arrivals:
        error = "stream produced no token-bearing delta for TTFT"
    return RequestResult(
        ordinal=spec.ordinal,
        scheduled_offset_s=spec.scheduled_offset_s,
        started_at=started_at,
        ended_at=ended_at,
        output_arrivals=tuple(arrivals),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        content_characters=content_characters,
        status_code=status_code,
        saw_done=saw_done,
        error=error,
        kind=spec.kind,
        sampling_seed=spec.sampling_seed,
    )


def _request_metrics(result: RequestResult, origin: float) -> dict[str, object]:
    ttft = result.output_arrivals[0] - result.started_at if result.output_arrivals else None
    decode_span = (
        result.output_arrivals[-1] - result.output_arrivals[0]
        if len(result.output_arrivals) > 1
        else None
    )
    tpot = (
        decode_span / (result.output_tokens - 1)
        if decode_span is not None and result.output_tokens is not None and result.output_tokens > 1
        else None
    )
    decode_rate = None if tpot is None or tpot == 0 else 1.0 / tpot
    return {
        "ordinal": result.ordinal,
        "kind": result.kind,
        "sampling_seed": result.sampling_seed,
        "scheduled_offset_s": result.scheduled_offset_s,
        "actual_start_offset_s": result.started_at - origin,
        "arrival_lag_ms": (result.started_at - origin - result.scheduled_offset_s) * 1000,
        "latency_ms": (result.ended_at - result.started_at) * 1000,
        "ttft_ms": None if ttft is None else ttft * 1000,
        "tpot_ms": None if tpot is None else tpot * 1000,
        "decode_tokens_per_second": decode_rate,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "timed_output_tokens_or_legacy_deltas": len(result.output_arrivals),
        "token_timeline_exact": result.output_tokens == len(result.output_arrivals),
        "max_output_gap_ms": 1000 * max(
            (b - a for a, b in zip(result.output_arrivals, result.output_arrivals[1:])),
            default=0.0,
        ),
        "output_arrival_offsets_s": [t - origin for t in result.output_arrivals],
        "content_characters": result.content_characters,
        "status_code": result.status_code,
        "saw_done": result.saw_done,
        "error": result.error,
    }


def _time_windows(
    results: Sequence[RequestResult],
    *,
    origin: float,
    ended_at: float,
    window_s: float,
) -> list[dict[str, object]]:
    duration = max(ended_at - origin, 0.0)
    count = max(1, math.ceil(duration / window_s))
    windows: list[dict[str, object]] = []
    for index in range(count):
        start = origin + index * window_s
        stop = min(start + window_s, ended_at)
        width = max(stop - start, 0.0)
        tokens = sum(
            (result.output_tokens or 0) / len(result.output_arrivals)
            for result in results
            if result.success and result.output_arrivals
            for arrival in result.output_arrivals
            if start <= arrival < stop
        )
        active_seconds = sum(
            max(0.0, min(result.ended_at, stop) - max(result.started_at, start))
            for result in results
        )
        windows.append(
            {
                "start_s": index * window_s,
                "duration_s": width,
                "output_tokens": tokens,
                "output_tokens_per_second": tokens / width if width > 0 else None,
                "temporal_accounting_exact": all(
                    result.output_tokens == len(result.output_arrivals)
                    for result in results
                    if result.success
                    and any(start <= arrival < stop for arrival in result.output_arrivals)
                ),
                "mean_active_requests": active_seconds / width if width > 0 else None,
                "started_requests": sum(start <= result.started_at < stop for result in results),
                "completed_requests": sum(start <= result.ended_at < stop for result in results),
            }
        )
    return windows


def _max_active_requests(results: Sequence[RequestResult]) -> int:
    active = maximum = 0
    boundaries = sorted(
        [(result.started_at, 1) for result in results]
        + [(result.ended_at, -1) for result in results],
        key=lambda item: (item[0], item[1]),
    )
    for _, change in boundaries:
        active += change
        maximum = max(maximum, active)
    return maximum


def long_generation_guardrail(
    results: Sequence[RequestResult],
    *,
    minimum_tokens: int,
    edge_tokens: int,
    minimum_tail_ratio: float,
) -> dict[str, object]:
    """Reject a severe late-decode slowdown on the same public SSE timeline."""

    rows: list[dict[str, object]] = []
    for result in results:
        if result.kind != "regular":
            continue
        arrivals = result.output_arrivals
        exact = result.output_tokens == len(arrivals)
        enough = exact and len(arrivals) >= minimum_tokens and len(arrivals) >= 2 * edge_tokens
        head = tail = ratio = None
        if enough:
            head_span = arrivals[edge_tokens - 1] - arrivals[0]
            tail_span = arrivals[-1] - arrivals[-edge_tokens]
            head = (edge_tokens - 1) / head_span if head_span > 0 else None
            tail = (edge_tokens - 1) / tail_span if tail_span > 0 else None
            ratio = tail / head if head and tail is not None else None
        passed = ratio is not None and ratio >= minimum_tail_ratio
        rows.append(
            {
                "ordinal": result.ordinal,
                "output_tokens": result.output_tokens,
                "timeline_exact": exact,
                "head_tokens_per_second": head,
                "tail_tokens_per_second": tail,
                "tail_to_head_ratio": ratio,
                "passed": passed,
            }
        )
    return {
        "enabled": True,
        "minimum_output_tokens": minimum_tokens,
        "edge_tokens": edge_tokens,
        "minimum_tail_to_head_ratio": minimum_tail_ratio,
        "passed": bool(rows) and all(row["passed"] is True for row in rows),
        "requests": rows,
    }


def summarize_level(
    results: Sequence[RequestResult],
    *,
    concurrency: int,
    origin: float,
    ended_at: float,
    window_s: float,
    long_generation: bool = False,
) -> dict[str, object]:
    successful = [result for result in results if result.success]
    duration = ended_at - origin
    request_rows = [_request_metrics(result, origin) for result in results]
    ttfts = [
        (result.output_arrivals[0] - result.started_at) * 1000
        for result in successful
        if result.output_arrivals
    ]
    decode_spans = [
        (result.output_arrivals[-1] - result.output_arrivals[0], result.output_tokens)
        for result in successful
        if len(result.output_arrivals) > 1
        and result.output_tokens is not None
        and result.output_tokens > 1
    ]
    tpots = [span / (tokens - 1) * 1000 for span, tokens in decode_spans]
    decode_rates = [1000 / tpot for tpot in tpots if tpot > 0]
    arrival_lags = [
        (result.started_at - origin - result.scheduled_offset_s) * 1000 for result in results
    ]
    total_output = sum(result.output_tokens or 0 for result in successful)
    guardrail = (
        long_generation_guardrail(
            results,
            minimum_tokens=_LONG_GENERATION_TOKENS,
            edge_tokens=_LONG_GENERATION_EDGE_TOKENS,
            minimum_tail_ratio=_MIN_LONG_GENERATION_TAIL_RATIO,
        )
        if long_generation
        else {"enabled": False}
    )
    canonical_valid = len(successful) == len(results) and guardrail.get("passed") is not False
    return {
        "concurrency": concurrency,
        "requested": len(results),
        "completed": len(successful),
        "failed": len(results) - len(successful),
        "canonical_valid": canonical_valid,
        "max_active_requests": _max_active_requests(results),
        "duration_s": duration,
        "total_input_tokens": sum(result.input_tokens or 0 for result in successful),
        "total_output_tokens": total_output,
        "token_timeline_exact_requests": sum(
            result.output_tokens == len(result.output_arrivals) for result in successful
        ),
        "window_timeline_exact": all(
            result.output_tokens == len(result.output_arrivals) for result in successful
        ),
        "request_throughput_per_second": len(successful) / duration if duration > 0 else None,
        # This includes request ramp-up, prefill, and retirement tails.  Keep it
        # only as an explicitly named makespan diagnostic; decode throughput is
        # derived from the exact SSE arrival timeline below.
        "batch_makespan_output_tokens_per_second_diagnostic": (
            total_output / duration if duration > 0 else None
        ),
        "ttft_ms": _distribution(ttfts),
        "tpot_ms": _distribution(tpots),
        "per_request_decode_tokens_per_second": _distribution(decode_rates),
        "arrival_lag_ms": _distribution(arrival_lags),
        "long_generation_guardrail": guardrail,
        "windows": _time_windows(results, origin=origin, ended_at=ended_at, window_s=window_s),
        "requests": request_rows,
        "errors": [
            {"ordinal": result.ordinal, "status_code": result.status_code, "error": result.error}
            for result in results
            if not result.success
        ],
    }


async def run_level(
    client: httpx.AsyncClient,
    *,
    api_url: str,
    model: str,
    prompt: str,
    max_output_tokens: int,
    concurrency: int,
    request_multiplier: int,
    arrival_pattern: str,
    request_rate: float | None,
    seed: int,
    reasoning_effort: str | None,
    window_s: float,
    staggered_prefill_prompt: str | None = None,
    staggered_prefill_at_s: float | None = None,
    staggered_prefill_output_tokens: int = 1,
    device_sampler: DeviceSampler | None = None,
    device_idle_s: float = 1.0,
    long_generation: bool = False,
    prompt_namespace: str | None = None,
    image_data_urls: tuple[str, ...] = (),
    media_data_urls: tuple[tuple[str, str], ...] = (),
) -> dict[str, object]:
    request_count = concurrency * request_multiplier
    offsets = arrival_offsets(
        request_count, pattern=arrival_pattern, request_rate=request_rate, seed=seed
    )
    specs = [
        RequestSpec(
            index,
            offset,
            (
                prompt
                if prompt_namespace is None
                else f"{index:08x}{prompt_namespace}\n{prompt}"
            ),
            max_output_tokens,
            sampling_seed=(seed + index) % 2**63,
            image_data_urls=image_data_urls,
            media_data_urls=media_data_urls,
        )
        for index, offset in enumerate(offsets)
    ]
    if staggered_prefill_prompt is not None:
        if staggered_prefill_at_s is None:
            raise ValueError("staggered prefill requires its arrival offset")
        specs.append(
            RequestSpec(
                request_count,
                staggered_prefill_at_s,
                (
                    staggered_prefill_prompt
                    if prompt_namespace is None
                    else f"{request_count:08x}{prompt_namespace}\n{staggered_prefill_prompt}"
                ),
                staggered_prefill_output_tokens,
                "staggered_prefill",
                sampling_seed=(seed + request_count) % 2**63,
            )
        )
    # The injected prefill arrives in addition to C already-running decodes.
    semaphore = asyncio.Semaphore(concurrency + (staggered_prefill_prompt is not None))
    baseline = (
        await device_sampler.collect_for(device_idle_s) if device_sampler is not None else None
    )
    origin = time.perf_counter()
    telemetry_stop = asyncio.Event()
    telemetry_task = (
        asyncio.create_task(device_sampler.collect_until(telemetry_stop))
        if device_sampler is not None
        else None
    )

    async def scheduled(spec: RequestSpec) -> RequestResult:
        delay = origin + spec.scheduled_offset_s - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        async with semaphore:
            return await stream_request(
                client,
                api_url=api_url,
                model=model,
                spec=spec,
                reasoning_effort=reasoning_effort,
            )

    telemetry: DeviceSamplingWindow | None = None
    try:
        results = await asyncio.gather(*(scheduled(spec) for spec in specs))
        ended_at = time.perf_counter()
    finally:
        telemetry_stop.set()
        if telemetry_task is not None:
            telemetry = await telemetry_task
    summary = summarize_level(
        results,
        concurrency=concurrency,
        origin=origin,
        ended_at=ended_at,
        window_s=window_s,
        long_generation=long_generation,
    )
    if device_sampler is not None and baseline is not None and telemetry is not None:
        summary["device_telemetry"] = summarize_device_sampling(
            baseline=baseline,
            request=telemetry,
            origin=origin,
            ended_at=ended_at,
            interval_s=device_sampler.interval_s,
            requested_device_indices=device_sampler.device_indices,
        )
    return summary


async def discover_model(client: httpx.AsyncClient, base_url: str) -> str:
    response = await client.get(base_url.rstrip("/") + "/v1/models")
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or not data:
        raise ValueError("/v1/models returned no models")
    first = data[0]
    model = first.get("id") if isinstance(first, dict) else None
    if not isinstance(model, str) or not model:
        raise ValueError("/v1/models first entry has no string id")
    return model


def _parse_concurrency(value: str) -> tuple[int, ...]:
    try:
        levels = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("concurrency must be comma-separated integers") from exc
    if not levels or any(level < 1 for level in levels) or len(set(levels)) != len(levels):
        raise argparse.ArgumentTypeError("concurrency levels must be unique positive integers")
    return levels


def _parse_gpu_devices(value: str) -> tuple[int, ...] | None:
    if value.strip().lower() == "all":
        return None
    try:
        indices = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "GPU devices must be 'all' or comma-separated integer indices"
        ) from exc
    if not indices or any(index < 0 for index in indices) or len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("GPU device indices must be unique non-negative integers")
    return indices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30015")
    parser.add_argument("--model", help="model id; defaults to the first /v1/models entry")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", default="请准确解释为什么天空看起来是蓝色的。")
    prompt_group.add_argument("--prompt-file", type=Path)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument(
        "--concurrency",
        type=_parse_concurrency,
        default=CONCURRENCY_LEVELS,
        help="comma-separated levels (default: 1,4,8,16,32,64,128,256)",
    )
    parser.add_argument("--request-multiplier", type=int, default=1)
    parser.add_argument(
        "--prefix-cache-state",
        choices=("isolated", "shared"),
        default="isolated",
        help=(
            "isolate each request from prior prefix-cache state (default), or deliberately "
            "share the exact prompt"
        ),
    )
    parser.add_argument("--arrival-pattern", choices=("burst", "fixed", "poisson"), default="burst")
    parser.add_argument(
        "--prompt-namespace",
        help=(
            "explicit isolated-prefix namespace for paired runs; compare with independent "
            "or reset prefix caches (default: a fresh random namespace per run)"
        ),
    )
    parser.add_argument("--request-rate", type=float)
    parser.add_argument("--image-file", type=Path, action="append", default=[],
                        help="Attach an image to each regular public streaming request; repeatable")
    parser.add_argument("--video-file", type=Path, action="append", default=[],
                        help="Attach a video to each regular public streaming request; repeatable")
    parser.add_argument("--audio-file", type=Path, action="append", default=[],
                        help="Attach input audio on models that support it; repeatable")
    parser.add_argument(
        "--seed", type=int, default=1,
        help="Base seed for arrival timing and per-request model sampling",
    )
    parser.add_argument(
        "--staggered-prefill-prompt-file",
        type=Path,
        help="inject this long prefill while the regular decode streams are active",
    )
    parser.add_argument("--staggered-prefill-at-seconds", type=float)
    parser.add_argument("--staggered-prefill-output-tokens", type=int, default=1)
    parser.add_argument(
        "--reasoning-effort",
        choices=("model-default", "none", "minimal", "low", "medium", "high", "xhigh", "max"),
        default="xhigh",
    )
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=6 * 60 * 60)
    parser.add_argument(
        "--gpu-devices",
        type=_parse_gpu_devices,
        default=None,
        help="physical GPU indices to sample, comma-separated; default: all reported GPUs",
    )
    parser.add_argument("--device-sample-interval-seconds", type=float, default=0.25)
    parser.add_argument("--device-idle-baseline-seconds", type=float, default=1.0)
    parser.add_argument("--continue-after-errors", action="store_true")
    parser.add_argument(
        "--output", type=Path, help="write the JSON artifact here; stdout otherwise"
    )
    return parser


async def run(args: argparse.Namespace) -> dict[str, object]:
    import base64
    import hashlib
    import mimetypes

    image_data_urls, image_records = [], []
    media_data_urls, media_records = [], []
    for kind in ("video", "audio"):
        for path in getattr(args, kind + "_file"):
            raw = path.read_bytes()
            mime = mimetypes.guess_type(path.name)[0]
            if mime is None or not mime.startswith(kind + "/"):
                raise ValueError(f"--{kind}-file must have a recognized {kind} MIME type")
            media_data_urls.append((kind, f"data:{mime};base64,{base64.b64encode(raw).decode()}"))
            media_records.append({"modality": kind, "path": str(path), "mime_type": mime,
                                  "bytes": len(raw),
                                  "content_digest": "b2s256:" + hashlib.blake2s(raw).hexdigest()})
    for path in args.image_file:
        raw = path.read_bytes()
        mime, _ = mimetypes.guess_type(path.name)
        if mime is None or not mime.startswith("image/"):
            raise ValueError("--image-file must have a recognized image MIME type")
        image_data_urls.append(f"data:{mime};base64,{base64.b64encode(raw).decode()}")
        image_records.append({"path": str(path), "mime_type": mime,
                              "content_digest": "b2s256:" + hashlib.blake2s(raw).hexdigest()})
    if args.prompt_namespace is not None and (
        not args.prompt_namespace or args.prefix_cache_state != "isolated"
    ):
        raise ValueError("--prompt-namespace requires a nonempty value and isolated prefix state")
    if args.max_output_tokens < 1:
        raise ValueError("--max-output-tokens must be positive")
    if args.request_multiplier < 1:
        raise ValueError("--request-multiplier must be positive")
    if not math.isfinite(args.window_seconds) or args.window_seconds <= 0:
        raise ValueError("--window-seconds must be finite and positive")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be finite and positive")
    if (
        not math.isfinite(args.device_sample_interval_seconds)
        or args.device_sample_interval_seconds <= 0
    ):
        raise ValueError("--device-sample-interval-seconds must be finite and positive")
    if (
        not math.isfinite(args.device_idle_baseline_seconds)
        or args.device_idle_baseline_seconds <= 0
    ):
        raise ValueError("--device-idle-baseline-seconds must be finite and positive")
    if (args.staggered_prefill_prompt_file is None) != (args.staggered_prefill_at_seconds is None):
        raise ValueError(
            "--staggered-prefill-prompt-file and --staggered-prefill-at-seconds "
            "must be provided together"
        )
    if args.staggered_prefill_at_seconds is not None and (
        not math.isfinite(args.staggered_prefill_at_seconds)
        or args.staggered_prefill_at_seconds < 0
    ):
        raise ValueError("--staggered-prefill-at-seconds must be finite and non-negative")
    if args.staggered_prefill_output_tokens < 1:
        raise ValueError("--staggered-prefill-output-tokens must be positive")
    prompt = (
        args.prompt_file.read_text(encoding="utf-8")
        if args.prompt_file is not None
        else args.prompt
    )
    if not prompt:
        raise ValueError("prompt must not be empty")
    staggered_prompt = (
        args.staggered_prefill_prompt_file.read_text(encoding="utf-8")
        if args.staggered_prefill_prompt_file is not None
        else None
    )
    if staggered_prompt == "":
        raise ValueError("staggered prefill prompt must not be empty")
    base_url = args.base_url.rstrip("/")
    api_url = base_url + "/v1/chat/completions"
    api_key = os.environ.get("OPENAI_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    max_connections = max(args.concurrency) + (staggered_prompt is not None)
    limits = httpx.Limits(
        max_connections=max_connections, max_keepalive_connections=max_connections
    )
    timeout = httpx.Timeout(args.timeout_seconds)
    device_sampler = DeviceSampler(
        device_indices=args.gpu_devices,
        interval_s=args.device_sample_interval_seconds,
    )
    levels: list[dict[str, object]] = []
    prompt_namespace = args.prompt_namespace
    if prompt_namespace is None and args.prefix_cache_state == "isolated":
        prompt_namespace = secrets.token_hex(8)
    async with httpx.AsyncClient(headers=headers, limits=limits, timeout=timeout) as client:
        model = args.model or await discover_model(client, base_url)
        for index, concurrency in enumerate(args.concurrency):
            level = await run_level(
                client,
                api_url=api_url,
                model=model,
                prompt=prompt,
                image_data_urls=tuple(image_data_urls),
                media_data_urls=tuple(media_data_urls),
                max_output_tokens=args.max_output_tokens,
                concurrency=concurrency,
                request_multiplier=args.request_multiplier,
                arrival_pattern=args.arrival_pattern,
                request_rate=args.request_rate,
                seed=args.seed + index,
                reasoning_effort=(
                    None if args.reasoning_effort == "model-default" else args.reasoning_effort
                ),
                window_s=args.window_seconds,
                staggered_prefill_prompt=staggered_prompt,
                staggered_prefill_at_s=args.staggered_prefill_at_seconds,
                staggered_prefill_output_tokens=args.staggered_prefill_output_tokens,
                device_sampler=device_sampler,
                device_idle_s=args.device_idle_baseline_seconds,
                long_generation=args.max_output_tokens >= _LONG_GENERATION_TOKENS,
                prompt_namespace=(
                    None
                    if prompt_namespace is None
                    else f"{index:08x}{prompt_namespace}"
                ),
            )
            levels.append(level)
            if level["failed"] and not args.continue_after_errors:
                break
    return {
        "schema": "cubic.chat-stream-benchmark.v1",
        "metric_scope": "public_openai_streaming_chat_completions_only",
        "token_timeline_policy": (
            "continuous cumulative usage assigns generated tokens to their public SSE arrival; "
            "without continuous usage, delta timing is a legacy estimate weighted to final usage"
        ),
        "upstream_references": UPSTREAM_REFERENCES,
        "device_telemetry_references": DEVICE_TELEMETRY_REFERENCES,
        "endpoint": api_url,
        "model": model,
        "request": {
            "prompt_characters": len(prompt),
            "prefix_cache_state": args.prefix_cache_state,
            "prompt_namespace": prompt_namespace,
            "max_output_tokens": args.max_output_tokens,
            "reasoning_effort": args.reasoning_effort,
            "sampling_seed_policy": "(CLI seed + level index + request ordinal) modulo 2**63",
            "sampling_seed_base": args.seed,
            "images": image_records,
            "media": media_records,
            "long_generation_guardrail": {
                "enabled": args.max_output_tokens >= _LONG_GENERATION_TOKENS,
                "minimum_output_tokens": _LONG_GENERATION_TOKENS,
                "edge_tokens": _LONG_GENERATION_EDGE_TOKENS,
                "minimum_tail_to_head_ratio": _MIN_LONG_GENERATION_TAIL_RATIO,
            },
        },
        "arrival": {
            "pattern": args.arrival_pattern,
            "request_rate": args.request_rate,
            "seed": args.seed,
            "staggered_prefill": (
                None
                if staggered_prompt is None
                else {
                    "at_seconds": args.staggered_prefill_at_seconds,
                    "prompt_characters": len(staggered_prompt),
                    "max_output_tokens": args.staggered_prefill_output_tokens,
                    "meaning": "in addition to C regular decode streams",
                }
            ),
        },
        "requested_concurrency_levels": list(args.concurrency),
        "request_multiplier": args.request_multiplier,
        "window_seconds": args.window_seconds,
        "device_sampling": {
            "backend": "nvidia-smi",
            "requested_device_indices": (
                "all" if args.gpu_devices is None else list(args.gpu_devices)
            ),
            "sample_interval_seconds": args.device_sample_interval_seconds,
            "idle_baseline_seconds": args.device_idle_baseline_seconds,
            "failure_policy": "record N/A without changing public request results",
        },
        "levels": levels,
        "stopped_at_first_error": bool(
            levels and levels[-1]["failed"] and not args.continue_after_errors
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        artifact = asyncio.run(run(args))
    except (OSError, ValueError, httpx.HTTPError) as exc:
        raise SystemExit(f"benchmark configuration/startup failed: {exc}") from exc
    output = json.dumps(artifact, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(output, end="")
    else:
        args.output.write_text(output, encoding="utf-8")
    levels = artifact.get("levels")
    if not isinstance(levels, list):  # pragma: no cover - run() owns this schema
        raise RuntimeError("benchmark artifact levels must be a list")
    valid = all(
        isinstance(level, dict) and level.get("canonical_valid") is True for level in levels
    )
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
