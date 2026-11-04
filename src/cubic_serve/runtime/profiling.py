"""Bounded opt-in timing for production execution paths."""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque

import torch

PROFILE_ENV = "CUBIC_SERVE_FEEDING_PROFILE"
PROFILE_STEPS_ENV = "CUBIC_SERVE_FEEDING_PROFILE_STEPS"
PROFILE_SAMPLE_LIMIT = 4096
PROFILE_DEFAULT_STEPS = 100


class StepProfile:
    """Collect a bounded timing window without affecting disabled execution."""

    def __init__(
        self,
        role: str,
        *,
        anchor_metric: str,
        sample_steps: int,
        schema: str,
        logger: logging.Logger,
    ) -> None:
        self.role = role
        self.active = True
        self.anchor_metric = anchor_metric
        self.sample_steps = sample_steps
        self.schema = schema
        self._logger = logger
        self._samples: dict[str, deque[float]] = {}
        self._values: dict[str, deque[float]] = {}
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()
        self._emitted = False

    def record(self, metric: str, milliseconds: float) -> None:
        if not self.active:
            return
        if milliseconds < 0:
            raise ValueError("profile duration cannot be negative")
        with self._lock:
            samples = self._samples.setdefault(
                metric, deque(maxlen=PROFILE_SAMPLE_LIMIT)
            )
            samples.append(milliseconds)
            self._counts[metric] = self._counts.get(metric, 0) + 1
            complete = (
                metric == self.anchor_metric
                and self._counts[metric] >= self.sample_steps
            )
            if complete:
                self.active = False
        if complete:
            self.emit_once()

    def record_value(self, metric: str, value: float) -> None:
        """Record a bounded non-duration value in the active timing window."""

        if not self.active:
            return
        with self._lock:
            values = self._values.setdefault(
                metric, deque(maxlen=PROFILE_SAMPLE_LIMIT)
            )
            values.append(value)

    @staticmethod
    def _percentile(samples: tuple[float, ...], percentile: float) -> float:
        ordered = sorted(samples)
        index = max(
            0,
            min(
                len(ordered) - 1,
                int(percentile * len(ordered) + 0.999999) - 1,
            ),
        )
        return ordered[index]

    def summary(self) -> dict[str, object]:
        with self._lock:
            metrics = {
                metric: {
                    "count": self._counts[metric],
                    "retained": len(samples),
                    "sum_ms": round(sum(samples), 4),
                    "p50_ms": round(self._percentile(tuple(samples), 0.50), 4),
                    "p95_ms": round(self._percentile(tuple(samples), 0.95), 4),
                }
                for metric, samples in sorted(self._samples.items())
                if samples
            }
            values = {
                metric: {
                    "retained": len(samples),
                    "sum": round(sum(samples), 4),
                    "p50": round(self._percentile(tuple(samples), 0.50), 4),
                    "p95": round(self._percentile(tuple(samples), 0.95), 4),
                }
                for metric, samples in sorted(self._values.items())
                if samples
            }
        return {
            "schema": self.schema,
            "role": self.role,
            "sample_limit": PROFILE_SAMPLE_LIMIT,
            "sample_steps": self.sample_steps,
            "metrics": metrics,
            "values": values,
        }

    def emit_once(self) -> None:
        with self._lock:
            if self._emitted:
                return
            self._emitted = True
        self._logger.warning("feeding profile %s", json.dumps(self.summary(), sort_keys=True))


def new_step_profile(
    role: str,
    *,
    schema: str,
    logger: logging.Logger,
) -> StepProfile | None:
    enabled = os.environ.get(PROFILE_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        return None
    raw_steps = os.environ.get(PROFILE_STEPS_ENV, str(PROFILE_DEFAULT_STEPS))
    try:
        sample_steps = int(raw_steps)
    except ValueError as error:
        raise ValueError(f"{PROFILE_STEPS_ENV} must be a positive integer") from error
    if sample_steps <= 0:
        raise ValueError(f"{PROFILE_STEPS_ENV} must be a positive integer")
    anchor = "result_commit_cpu_ms" if role == "controller" else "validity_host_sync_cpu_ms"
    return StepProfile(
        role,
        anchor_metric=anchor,
        sample_steps=sample_steps,
        schema=schema,
        logger=logger,
    )


class CudaStepProfiler:
    """Preallocated CUDA events feeding one process-local aggregate."""

    def __init__(
        self,
        aggregate: StepProfile,
        device: torch.device,
        metrics: tuple[str, ...],
    ) -> None:
        self._aggregate = aggregate
        with torch.cuda.device(device):
            self._events = {
                metric: (
                    torch.cuda.Event(enable_timing=True),  # type: ignore[no-untyped-call]
                    torch.cuda.Event(enable_timing=True),  # type: ignore[no-untyped-call]
                )
                for metric in metrics
            }

    @property
    def active(self) -> bool:
        return self._aggregate.active

    def begin_cuda(self, metric: str) -> None:
        if self.active:
            self._events[metric][0].record()

    def end_cuda(self, metric: str) -> None:
        if self.active:
            self._events[metric][1].record()

    def flush_cuda(self, metrics: tuple[str, ...]) -> None:
        if not self.active:
            return
        for metric in metrics:
            start, end = self._events[metric]
            # Deferred sampling validity deliberately leaves the stream in
            # flight. Profiling must neither read an incomplete event nor add
            # a synchronization to that execution path. Keep missing timing
            # samples visible rather than reporting them as zero milliseconds.
            if not end.query():
                self._aggregate.record_value(f"{metric}_incomplete_samples", 1)
                continue
            self._aggregate.record(metric, start.elapsed_time(end))

    def record_cpu(self, metric: str, milliseconds: float) -> None:
        self._aggregate.record(metric, milliseconds)
