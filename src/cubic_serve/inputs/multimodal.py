"""Engine-owned budgets and admission for input encoders of any modality."""

from __future__ import annotations

import asyncio
import math
import threading
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from cubic_serve.inputs.chat import ChatProcessorError


class EncoderAdmissionError(ChatProcessorError):
    """A media item cannot fit the configured encoder resource budget."""


@dataclass(frozen=True, slots=True)
class MultimodalInputPolicy:
    """Resource controls; model processors still own sampling and semantics."""

    encoder_memory_bytes: int = 3 << 29
    preprocess_workers: int = 2
    feature_cache_bytes: int = 256 << 20
    video_frames_per_second: float = 2.0
    video_max_frames: int = 768

    def __post_init__(self):
        if type(self.encoder_memory_bytes) is not int or self.encoder_memory_bytes <= 0:
            raise ValueError("multimodal memory budget must be positive integer bytes")
        if type(self.preprocess_workers) is not int or self.preprocess_workers <= 0:
            raise ValueError("multimodal preprocess workers must be a positive integer")
        if type(self.feature_cache_bytes) is not int or self.feature_cache_bytes < 0:
            raise ValueError("multimodal feature cache bytes must be a non-negative integer")
        if (isinstance(self.video_frames_per_second, bool) or
                not math.isfinite(self.video_frames_per_second) or
                self.video_frames_per_second <= 0):
            raise ValueError("video frames per second must be finite and positive")
        if type(self.video_max_frames) is not int or self.video_max_frames < 4:
            raise ValueError("video max frames must be an integer of at least four")


class EncoderMemoryBudget:
    """One device's shared resident budget and exclusive staging ownership.

    Initialize after loading encoder weights. Weights are already reflected in
    free memory; graphs and temporary input/output storage share this budget.
    Feature caches and live request outputs stay on the host. Callers must finish
    device work and durable host copies before releasing the staging lease.
    """

    def __init__(self, policy: MultimodalInputPolicy):
        self.policy = policy
        self.total_bytes: int | None = None
        self.resident_bytes = 0
        self.staging_bytes = 0
        self._lock = threading.Lock()

    def initialize(self, free_bytes: int):
        if self.total_bytes is None:
            if type(free_bytes) is not int or free_bytes <= 0:
                raise RuntimeError("no GPU memory remains for input encoders")
            self.total_bytes = min(free_bytes, self.policy.encoder_memory_bytes)

    def capture_capacity(self, *, linear_bytes: int, quadratic_bytes: int,
                         staging_bytes_per_row: int, minimum: int, declared: int,
                         measured_capacity: int = 0, measured_resident_bytes: int = 0) -> int:
        if self.total_bytes is None:
            raise RuntimeError("encoder memory budget has not been initialized")
        if min(linear_bytes, quadratic_bytes, staging_bytes_per_row) < 0:
            raise ValueError("encoder memory costs must be non-negative")
        if minimum <= 0 or minimum & (minimum - 1) or declared < minimum:
            raise ValueError("encoder capacity requires a positive power-of-two minimum")
        if measured_capacity < 0 or measured_resident_bytes < 0:
            raise ValueError("encoder capture measurements must be non-negative")
        if measured_resident_bytes and not measured_capacity:
            raise ValueError("encoder capture residency requires its measured capacity")
        if measured_capacity:
            # Measure the complete retained portfolio, excluding encoder weights.
            # Keep the geometry-owned quadratic term and fit the remaining
            # per-row cost. This includes sealed operands and graph-pool storage
            # that a tensor inventory alone cannot price before capture.
            remaining = max(0, measured_resident_bytes -
                            quadratic_bytes * measured_capacity**2)
            linear_bytes = max(linear_bytes, (remaining + measured_capacity - 1) //
                               measured_capacity)
        capacity = minimum
        admitted = 0
        while capacity <= declared:
            cost = (self.resident_bytes + linear_bytes * capacity +
                    quadratic_bytes * capacity**2 +
                    max(self.staging_bytes, staging_bytes_per_row * capacity))
            if cost > self.total_bytes:
                break
            admitted = capacity
            capacity *= 2
        if not admitted:
            raise RuntimeError("encoder memory budget cannot admit its minimum graph")
        return admitted

    def reserve(self, *, resident_bytes: int, staging_bytes: int):
        if min(resident_bytes, staging_bytes) < 0:
            raise ValueError("encoder memory reservations must be non-negative")
        resident = self.resident_bytes + resident_bytes
        staging = max(self.staging_bytes, staging_bytes)
        if self.total_bytes is None or resident + staging > self.total_bytes:
            raise RuntimeError("encoder graphs and staging exceed the admitted memory budget")
        self.resident_bytes, self.staging_bytes = resident, staging

    @contextmanager
    def lease(self, transient_bytes: int):
        if not 0 <= transient_bytes <= self.staging_bytes:
            raise ValueError("encoder input/output exceeds its reserved staging capacity")
        with self._lock:
            yield

    @contextmanager
    def lease_dynamic(self, transient_bytes: int):
        """Serialize a non-graph encoder item within the same resident budget."""
        if transient_bytes < 0 or self.total_bytes is None:
            raise ValueError("encoder transient size requires an initialized budget")
        if self.resident_bytes + max(self.staging_bytes, transient_bytes) > self.total_bytes:
            raise EncoderAdmissionError(
                "encoder item exceeds multimodal memory budget: "
                f"required_bytes={self.resident_bytes + max(self.staging_bytes, transient_bytes)} "
                f"budget_bytes={self.total_bytes}",
                public_message="media input is too large",
            )
        with self._lock:
            yield


def bounded_encoder_groups(lengths: Sequence[int], capacity: int, *,
                           allow_oversize: bool = False) -> tuple[tuple[int, int], ...]:
    """Pack declared independent segments in order, never split a segment.

    Only a model adapter can establish that segments are independent. Resource
    policy cannot infer independence from a video frame or audio sample count.
    """

    if capacity <= 0:
        raise ValueError("admitted encoder graph capacity must be positive")
    for length in lengths:
        if type(length) is not int or length <= 0 or (length > capacity and not allow_oversize):
            raise ValueError(
                "an independent encoder segment exceeds admitted graph capacity: "
                f"required_rows={length} capacity_rows={capacity}")
    groups = []
    start = rows = 0
    for index, length in enumerate(lengths):
        if rows + length > capacity:
            if rows:
                groups.append((start, index))
            start, rows = index, 0
        rows += length
        if rows > capacity:
            groups.append((start, index + 1))
            start, rows = index + 1, 0
    if rows:
        groups.append((start, len(lengths)))
    return tuple(groups)


class MultimodalPreprocessingGate:
    """Bound actual worker lifetimes, including work after HTTP cancellation."""

    def __init__(self, workers: int):
        if type(workers) is not int or workers <= 0:
            raise ValueError("multimodal preprocess workers must be a positive integer")
        self._slots = asyncio.Semaphore(workers)
        self._tasks: set[asyncio.Task] = set()

    async def run(self, function: Callable, /, *args, **kwargs):
        await self._slots.acquire()
        try:
            task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        except BaseException:
            self._slots.release()
            raise
        self._tasks.add(task)

        def finished(done):
            self._tasks.discard(done)
            self._slots.release()
            if not done.cancelled():
                done.exception()  # Also observe failures from an abandoned request.

        task.add_done_callback(finished)
        return await asyncio.shield(task)
