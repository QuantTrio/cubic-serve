"""Startup-only costs for kernels competing with a captured collective.

No hardware/model names, manual tuning tables or runtime graph capture.
Compute calibration belongs to the concrete operator/layout; transport calibration
belongs to its actual process group and payload. Neither changes reduction math.

References: vLLM 89af4f1 batch_invariant.py/ubatching.py (fixed K/dependencies),
SGLang ba433bb flashinfer_autotune.py (shape tuning must not change reduction),
llama.cpp 458681e ggml-cuda.cu/allreduce.cu (separate transport/compute ownership).
Graph resource introspection uses the CUDA 13.0 driver API, not profiler counters.
"""

from __future__ import annotations

import ctypes as ct
import fcntl
import hashlib
import inspect
import json
import math
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any, ClassVar, TypeVar

import torch
import torch.distributed as dist

from cubic_serve.ops.cuda.cache_identity import cuda_libraries, gpu_model, module_kernels, plan_path
from cubic_serve.ops.cuda.startup import weight_bound_calibration_key
from cubic_serve.ops.strategies import (
    LaunchParameters,
    LinearParameters,
    collective_parameters,
    launch_parameters,
    parameter_overrides,
)


def fit_nonnegative_cost(
    features: Sequence[Sequence[float]], times: Sequence[float],
) -> tuple[float, ...]:
    """Fit a small physical cost basis; holdouts must be checked by its owner.

    Relative residuals protect latency-sized observations. Bounded IRLS and
    active-set enumeration retain the existing Marlin fitting contract.
    """
    x = torch.tensor(features, dtype=torch.float64)
    y = torch.tensor(times, dtype=torch.float64)
    if (x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.numel()
            or not y.numel() or not 1 <= x.shape[1] <= 8
            or not torch.isfinite(x).all() or not torch.isfinite(y).all()
            or (x < 0).any() or (y <= 0).any()):
        raise ValueError("cost fit requires finite nonnegative features and positive times")
    columns_count = x.shape[1]
    x = x / y[:, None]
    y = torch.ones_like(y)
    best, weights = None, torch.ones_like(y)
    for _ in range(6):
        solutions = []
        for mask in range(1, 1 << columns_count):
            columns = [i for i in range(columns_count) if mask & (1 << i)]
            # Geometry/domain features can be dependent, including within an
            # active subset. Use the CPU rank-revealing SVD driver so an
            # identical calibration does not change its extrapolation plan.
            coefficients = torch.linalg.lstsq(
                x[:, columns] * weights[:, None], y * weights, driver="gelsd"
            ).solution
            if (coefficients < 0).any():
                continue
            beta = torch.zeros(columns_count, dtype=torch.float64)
            beta[columns] = coefficients
            residual = float((((x @ beta - y) * weights) ** 2).sum())
            solutions.append((residual, beta))
        if not solutions:
            break
        beta = min(solutions, key=lambda item: item[0])[1]
        errors = (x @ beta - y).abs()
        residual = float(errors.max())
        if best is None or residual < best[0]:
            best = residual, tuple(float(value) for value in beta)
        weights = (weights * (errors + 1e-3)).sqrt()
        weights /= weights.max()
    if best is None:
        raise ValueError("cost fit has no nonnegative solution")
    return best[1]


def prediction_interval(prediction: float, relative_error: float) -> tuple[float, float]:
    """Invert |prediction/observed - 1|, in the caller's time unit.

    These are empirical held-out bounds, not a statistical confidence interval
    or permission to extrapolate outside the owning model's qualified domain.
    Error >= 1 cannot provide a finite upper bound and cannot certify a gain.
    """
    if (not math.isfinite(prediction) or prediction < 0
            or not math.isfinite(relative_error) or relative_error < 0):
        raise ValueError("cost intervals require finite nonnegative prediction and error")
    return (prediction / (1 + relative_error),
            prediction / (1 - relative_error) if relative_error < 1 else math.inf)


def predicted_gain(
    candidate: tuple[float, float], baseline: tuple[float, float],
    min_gain_percent: float = 0,
) -> bool:
    """Compare complete-boundary intervals, with no assumed error correlation."""
    if not 0 <= min_gain_percent < 100:
        raise ValueError("cost gain must be a percentage below 100")
    return candidate[1] < baseline[0] * (1 - min_gain_percent / 100)


@dataclass(frozen=True)
class KernelResources:
    blocks: int
    threads: int
    registers: int
    shared: int


def scratch_ranges_disjoint(
    left: Sequence[torch.Tensor], right: Sequence[torch.Tensor],
) -> bool:
    """Check declared contiguous scratch footprints without a Cartesian scan.

    Preserve the original endpoint predicate, including empty views and
    overlapping/duplicate views within either lane. After sorting by start,
    an interval ending before the other lane's current start cannot overlap
    any of that lane's remaining intervals. Only cross-lane aliasing matters.
    """
    if not left or not right:
        return True

    def ranges(tensors):
        result = set()
        for tensor in tensors:
            start = tensor.data_ptr()
            result.add((start, start + tensor.numel() * tensor.element_size()))
        return sorted(result)

    a, b = ranges(left), ranges(right)
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i][1] <= b[j][0]:
            i += 1
        elif b[j][1] <= a[i][0]:
            j += 1
        else:
            return False
    return True


@dataclass(frozen=True)
class DeviceResources:
    sms: int
    threads: int
    registers: int
    shared: int


@dataclass(frozen=True)
class WorkCost:
    """Portable roofline coefficients in milliseconds, operations and bytes.

    Calibration is device/dtype local, never a shipped SKU or model table.
    This lower-bound model must be checked against the actual execution path.
    """

    startup_ms: float
    operations_per_ms: float
    bytes_per_ms: float

    def __post_init__(self) -> None:
        if (not math.isfinite(self.startup_ms) or self.startup_ms < 0
                or any(not math.isfinite(v) or v <= 0
                       for v in (self.operations_per_ms, self.bytes_per_ms))):
            raise ValueError("work-cost coefficients must be finite and rates positive")

    def predict(
        self, operations: float, byte_count: float, *, supply: float = 1.0, launches: int = 1,
    ) -> float:
        if (not all(math.isfinite(v) for v in (operations, byte_count, supply))
                or type(launches) is not int or launches < 0
                or min(operations, byte_count) < 0 or not 0 < supply <= 1):
            raise ValueError("work and resident supply must be nonnegative and bounded")
        return launches * self.startup_ms + max(
            operations / self.operations_per_ms, byte_count / self.bytes_per_ms,
        ) / supply

    def branch_costs(
        self, first: tuple[float, float, int], second: tuple[float, float, int], *,
        fork_join_launches: int,
    ) -> tuple[float, float]:
        """Serial critical path versus overlap with aggregate resource demand.

        Two branches cannot both consume the full device compute or bandwidth
        rate. Price their summed work as well as each branch's critical path.
        Dependency legality remains a separate prerequisite owned by the caller.
        """
        if type(fork_join_launches) is not int or fork_join_launches < 0:
            raise ValueError("fork/join launch count must be a nonnegative integer")
        a = self.predict(first[0], first[1], launches=first[2])
        b = self.predict(second[0], second[1], launches=second[2])
        shared_floor = self.predict(first[0] + second[0], first[1] + second[1], launches=0)
        return a + b, max(a, b, shared_floor) + fork_join_launches * self.startup_ms


@lru_cache(maxsize=16)
def device_work_cost(device: torch.device, dtype: torch.dtype) -> WorkCost:
    from cubic_serve.ops.cuda.startup import startup_cuda_lock

    with startup_cuda_lock(device):
        path = _launch_calibration_path(("device-work", dtype, collective_parameters()), (), device)
        raw = _read_startup_calibration(path, "cubic.device-work-cost.v1")
        if raw is not None:
            try:
                return WorkCost(**raw["cost"])
            except (TypeError, KeyError, ValueError):
                pass
        value = _measure_device_work_cost(device, dtype)
        _write_startup_calibration(path, "cubic.device-work-cost.v1", {"cost": asdict(value)})
        return value


def _measure_device_work_cost(device: torch.device, dtype: torch.dtype) -> WorkCost:
    """Bounded cold probes; retain coefficients only, no GPU buffers or graphs."""
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("work-cost calibration belongs before capture")
        hardware = torch.cuda.get_device_properties(device)
        sms = hardware.multi_processor_count
        side = 128 * math.ceil(math.sqrt(sms))
        a = torch.full((side, side), 0.125, device=device, dtype=dtype)
        b, c = torch.empty_like(a), torch.empty_like(a)
        b.fill_(0.125)
        # Repeated probes must not mistake an L2-resident copy for HBM rate.
        count = max(a.numel(), math.ceil(4 * hardware.L2_cache_size / a.element_size()))
        source = torch.full((count,), 0.125, device=device, dtype=dtype)
        target = torch.empty_like(source)

        def measure(fn: Callable[[], Any]) -> float:
            fn()
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            try:
                repeats = collective_parameters().route_completion_calibration_replays
                with torch.cuda.graph(graph):
                    for _ in range(repeats):
                        fn()
                return _time_graph(graph, launch_parameters().calibration_samples) / repeats
            finally:
                graph.reset()

        startup = measure(lambda: torch.mul(a[:1, :1], 1.0, out=c[:1, :1]))
        compute = measure(lambda: torch.mm(a, b, out=c))
        traffic = measure(lambda: torch.mul(source, 1.0, out=target))
        return WorkCost(
            startup, 2 * side**3 / max(compute - startup, compute / 2),
            2 * source.numel() * source.element_size() / max(traffic - startup, traffic / 2),
        )


@dataclass(frozen=True)
class TransferCost:
    """Cold-measured startup and inverse bandwidth, in microseconds/bytes.

    Rates belong to an actual process group and operation, not a GPU name.
    A failed fit is unusable evidence; it must not invent a negative cost.
    """

    startup_us: float
    us_per_byte: float

    def __post_init__(self) -> None:
        if any(not math.isfinite(x) or x < 0 for x in (self.startup_us, self.us_per_byte)):
            raise ValueError("transfer costs must be finite and nonnegative")

    def predict(self, byte_count: float) -> float:
        if not math.isfinite(byte_count) or byte_count < 0:
            raise ValueError("transfer size must be finite and nonnegative")
        return self.startup_us + self.us_per_byte * byte_count

    @classmethod
    def fit(cls, points: Sequence[tuple[float, float]]) -> TransferCost:
        if len(points) != 2:
            raise ValueError("transfer fit requires two distinct payload measurements")
        (x0, y0), (x1, y1) = sorted(points)
        if any(not math.isfinite(x) or x <= 0 for p in points for x in p) or x0 == x1:
            raise ValueError("transfer fit requires positive finite distinct measurements")
        slope = (y1 - y0) / (x1 - x0)
        if slope <= 0:
            raise ValueError("transfer fit has no measurable positive bandwidth slope")
        return cls(max(0.0, y0 - slope * x0), slope)


@dataclass(frozen=True)
class OrderedCollectiveCost:
    """Measured payload intervals for a real group's ordered completion.

    Bandwidth may improve or deteriorate as payload grows. Each positive
    affine fit owns its measured interval; interpolate between their inner
    anchors instead of extrapolating either regime over the other one.
    Held-out payloads validate the intervals and their transition before dispatch.
    Outside the measured envelope the existing analytical rule remains owner.
    """

    gather: tuple[TransferCost, TransferCost]
    partition: tuple[TransferCost, TransferCost]
    minimum_bytes: int
    maximum_bytes: int
    error_fraction: float
    transition_bytes: tuple[int, int]
    interval_error_fractions: tuple[tuple[float, float, float], ...]
    host: tuple[TransferCost, TransferCost] | None = None
    host_error_fraction: float = 0.0
    host_partition: tuple[TransferCost, TransferCost] | None = None
    host_partition_error_fraction: float = 0.0
    legacy_error_fractions: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if (not 0 < self.minimum_bytes < self.maximum_bytes
                or len(self.transition_bytes) != 2
                or not self.minimum_bytes < self.transition_bytes[0]
                < self.transition_bytes[1] < self.maximum_bytes
                or len(self.interval_error_fractions) != 4
                or any(len(errors) != 3 or any(not math.isfinite(e) or e < 0 for e in errors)
                       for errors in self.interval_error_fractions)
                or len(self.gather) != 2 or len(self.partition) != 2
                or not math.isfinite(self.error_fraction) or self.error_fraction < 0
                or (self.host is not None and len(self.host) != 2)
                or not math.isfinite(self.host_error_fraction) or self.host_error_fraction < 0
                or (self.host_partition is not None and len(self.host_partition) != 2)
                or not math.isfinite(self.host_partition_error_fraction)
                or self.host_partition_error_fraction < 0
                or (self.legacy_error_fractions is not None
                    and (len(self.legacy_error_fractions) != 2
                         or any(not math.isfinite(e) or e < 0
                                for e in self.legacy_error_fractions)))):
            raise ValueError("transfer fit requires finite costs and a bounded payload interval")

    def predict(self, payload_bytes: int) -> tuple[float, float]:
        return (self._predict(self.gather, payload_bytes, self.transition_bytes),
                self._predict(self.partition, payload_bytes, self.transition_bytes))

    @staticmethod
    def _predict(fits: tuple[TransferCost, TransferCost], payload: int,
                 transition: tuple[int, int]) -> float:
        lower, upper = transition
        if payload <= lower:
            return fits[0].predict(payload)
        if payload >= upper:
            return fits[1].predict(payload)
        fraction = (payload - lower) / (upper - lower)
        return ((1 - fraction) * fits[0].predict(lower)
                + fraction * fits[1].predict(upper))

    def _error(self, transport: int, payload: int) -> float:
        # Neither another transport nor a disjoint payload interval can
        # invalidate the actual candidate-versus-baseline comparison.
        lower, upper = self.transition_bytes
        interval = 0 if payload <= lower else 2 if payload >= upper else 1
        return self.interval_error_fractions[transport][interval]

    def select(self, payload_bytes: int, *, fallback: bool,
               min_gain_percent: int, max_error_percent: int) -> bool:
        if (payload_bytes <= 0 or not 0 <= min_gain_percent < 100
                or not 0 <= max_error_percent < 100):
            raise ValueError("transfer selection requires positive payload and bounded margins")
        if (not self.minimum_bytes <= payload_bytes <= self.maximum_bytes
                or max(self._error(0, payload_bytes), self._error(1, payload_bytes))
                * 100 > max_error_percent):
            return fallback
        gather, partition = self.predict(payload_bytes)
        gather_bounds = prediction_interval(gather, self._error(0, payload_bytes))
        partition_bounds = prediction_interval(partition, self._error(1, payload_bytes))
        if predicted_gain(partition_bounds, gather_bounds, min_gain_percent):
            return True
        if predicted_gain(gather_bounds, partition_bounds, min_gain_percent):
            return False
        return fallback

    def select_host(self, payload_bytes: int, *, partitioned: bool,
                    min_gain_percent: int, max_error_percent: int) -> bool:
        """Admit a third transport only inside its independently checked fit."""
        if not 0 <= min_gain_percent < 100 or not 0 <= max_error_percent < 100:
            raise ValueError("transport margins must be percentages below 100")
        if (self.host is None or not self.minimum_bytes <= payload_bytes <= self.maximum_bytes
                or max(self._error(int(partitioned), payload_bytes), self._error(2, payload_bytes))
                * 100 > max_error_percent):
            return False
        baseline = self.predict(payload_bytes)[int(partitioned)]
        predicted = self._predict(self.host, payload_bytes, self.transition_bytes)
        return predicted_gain(
            prediction_interval(predicted, self._error(2, payload_bytes)),
            prediction_interval(baseline, self._error(int(partitioned), payload_bytes)),
            min_gain_percent,
        )

    def select_host_partition(self, payload_bytes: int, *, partitioned: bool,
                              host_selected: bool, min_gain_percent: int,
                              max_error_percent: int) -> bool:
        """Price the extra publication phase against saved rank-replicated reads."""
        if not 0 <= min_gain_percent < 100 or not 0 <= max_error_percent < 100:
            raise ValueError("transport margins must be percentages below 100")
        baseline_error = self._error(2 if host_selected else int(partitioned), payload_bytes)
        if (self.host_partition is None
                or not self.minimum_bytes <= payload_bytes <= self.maximum_bytes
                or max(baseline_error, self._error(3, payload_bytes))
                * 100 > max_error_percent):
            return False
        baseline = self.predict(payload_bytes)[int(partitioned)]
        if host_selected:
            if self.host is None or self._error(2, payload_bytes) * 100 > max_error_percent:
                return False
            baseline = self._predict(self.host, payload_bytes, self.transition_bytes)
        predicted = self._predict(self.host_partition, payload_bytes, self.transition_bytes)
        return predicted_gain(
            prediction_interval(predicted, self._error(3, payload_bytes)),
            prediction_interval(baseline, baseline_error), min_gain_percent,
        )

    @classmethod
    def fit(cls, anchors: Sequence[tuple[int, tuple[float, ...]]],
            holdouts: Sequence[tuple[int, tuple[float, ...]]]) -> OrderedCollectiveCost:
        if len(anchors) != 4 or not holdouts:
            raise ValueError("transfer fit requires four anchors and independent holdouts")
        points = sorted(anchors)
        if len({p[0] for p in points}) != 4:
            raise ValueError("transfer fit requires distinct anchor payloads")
        transition = (points[1][0], points[2][0])
        fits = tuple(tuple(TransferCost.fit([(b, t[i]) for b, t in pair])
                           for pair in (points[:2], points[2:])) for i in range(2))
        errors = [0.0, 0.0]
        for payload, times in (*points, *holdouts):
            if (not points[0][0] <= payload <= points[-1][0]
                    or any(not math.isfinite(t) or t <= 0 for t in times)):
                raise ValueError("transfer fit holdout must be finite and inside anchor range")
            for i in range(2):
                errors[i] = max(errors[i], abs(
                    cls._predict(fits[i], payload, transition) / times[i] - 1))
        host, host_error = None, 0.0
        if all(len(times) >= 3 for _, times in (*points, *holdouts)):
            try:
                host = tuple(TransferCost.fit([(b, t[2]) for b, t in pair])
                             for pair in (points[:2], points[2:]))
                host_error = max(abs(cls._predict(host, b, transition) / t[2] - 1)
                                 for b, t in (*points, *holdouts))
            except ValueError:
                # Unsupported/noisy host timing must not invalidate the two
                # existing transports or fabricate a bandwidth coefficient.
                host = None
        host_partition, partition_error = None, 0.0
        if all(len(times) >= 4 for _, times in (*points, *holdouts)):
            try:
                host_partition = tuple(TransferCost.fit([(b, t[3]) for b, t in pair])
                                       for pair in (points[:2], points[2:]))
                partition_error = max(abs(cls._predict(host_partition, b, transition) / t[3] - 1)
                                      for b, t in (*points, *holdouts))
            except ValueError:
                host_partition = None
        interval_errors = []
        for index, curves in enumerate((*fits, host, host_partition)):
            local_errors = []
            for interval in range(3):
                left, right = (points[interval][0], points[interval + 1][0])
                witnesses = [(b, t) for b, t in holdouts if left < b < right]
                # An interval without an independent witness is unqualified.
                error = 1.0
                if curves is not None and witnesses:
                    error = max(abs(cls._predict(curves, b, transition) / t[index] - 1)
                                for b, t in (*points[interval:interval + 2], *witnesses))
                local_errors.append(error)
            interval_errors.append(tuple(local_errors))
        return cls(fits[0], fits[1], points[0][0], points[-1][0], max(errors), transition,
                   tuple(interval_errors),
                   host, host_error, host_partition, partition_error,
                   (errors[0], errors[1]))


@dataclass(frozen=True)
class RouteCompletionCost:
    """Price transport around the unchanged ordered rank-sum/top-k fold.

    The first three fits include completion kernels. The fourth measures only
    padding copies. Calibrate for the actual group, dtype, K and H; M is then
    a formula input, never a table of measured batch choices. A held-out payload
    checks extrapolation before the rule can choose a different transport.
    """

    ranks: int
    gather: TransferCost
    partition: TransferCost
    folded: TransferCost
    padding_copy: TransferCost
    error_fraction: float
    gather_error_fraction: float = 0.0
    padding_error_fraction: float = 0.0

    def __post_init__(self) -> None:
        errors = self.error_fraction, self.gather_error_fraction, self.padding_error_fraction
        if self.ranks < 1 or any(not math.isfinite(x) or x < 0 for x in errors):
            raise ValueError("route completion requires a valid group and measured fit error")

    def predict(
        self, rows: int, top_k: int, width: int, element_bytes: int, *,
        partitioned: bool, index_bytes: int = 4,
    ) -> tuple[float, float]:
        if min(rows, top_k, width, element_bytes, index_bytes) < 1:
            raise ValueError("route completion requires positive tensor extents")
        r = self.ranks
        if r == 1:
            return 0.0, 0.0
        payload = rows * top_k * width * element_bytes
        old = (self.partition.predict(2 * (r - 1) * payload / r) if partitioned
               else self.gather.predict((r - 1) * payload))
        padded = ((rows + r - 1) // r) * r
        # Exchange full routes, but gather only their already-rounded fold.
        new = self.folded.predict((r - 1) / r * padded * width * element_bytes * (top_k + 1))
        if padded != rows:
            # Three input copies (routes/IDs/weights) and one output trim.
            copied = 2 * rows * (
                top_k * (width * element_bytes + index_bytes + element_bytes)
                + width * element_bytes
            )
            new += self.padding_copy.predict(copied)
        return old, new

    def beneficial(
        self, rows: int, top_k: int, width: int, element_bytes: int, *,
        partitioned: bool, min_gain_percent: int, max_error_percent: int,
        index_bytes: int = 4,
    ) -> bool:
        if not 0 <= min_gain_percent < 100 or not 0 <= max_error_percent < 100:
            raise ValueError("route completion margins must lie in [0,100)")
        old, new = self.predict(rows, top_k, width, element_bytes,
                                partitioned=partitioned, index_bytes=index_bytes)
        # A padding-copy fit is irrelevant when no copy executes. Likewise the
        # small-payload gather fit does not govern the partitioned baseline.
        error = max(self.error_fraction,
                    0.0 if partitioned else self.gather_error_fraction,
                    self.padding_error_fraction if rows % self.ranks else 0.0)
        baseline_error = self.error_fraction if partitioned else self.gather_error_fraction
        candidate_error = max(self.error_fraction,
                              self.padding_error_fraction if rows % self.ranks else 0.0)
        return (error * 100 <= max_error_percent and predicted_gain(
            prediction_interval(new, candidate_error),
            prediction_interval(old, baseline_error), min_gain_percent,
        ))


@dataclass(frozen=True)
class MatmulGeometry:
    block_m: int
    block_n: int
    block_k: int
    warps: int
    stages: int


def matmul_geometries(
    device: DeviceResources,
    *,
    rows: int,
    columns: tuple[int, ...],
    reduction: int,
    element_bytes: int,
    baseline: MatmulGeometry,
    groups: int = 1,
    communication: Sequence[KernelResources] = (),
    locked: frozenset[str] = frozenset(),
    limits: LaunchParameters | None = None,
    work_cost: WorkCost | None = None,
) -> tuple[MatmulGeometry, ...]:
    """Bound a portable startup search by traffic, padding and resident warps.

    This proxy ranks compilation candidates, not serving latency. Actual compiled
    resource use and measured operator/transport costs make the final selection.
    K is the operator's reduction contract and is never a geometry search axis.
    Routed callers supply rank-local groups and their estimated rows per group;
    grouped projections supply their separate N extents, not one invented GEMM.
    """
    if (min(rows, reduction, groups, element_bytes, device.sms, device.threads,
            device.registers, device.shared) <= 0 or not columns or min(columns) <= 0):
        raise ValueError("launch geometry requires positive device and tensor extents")
    limits = limits or launch_parameters()

    def powers(bound: int) -> tuple[int, ...]:
        return tuple(1 << n for n in range(4, max(16, bound).bit_length()))

    def choices(name: str, values: tuple[int, ...]) -> tuple[int, ...]:
        return (getattr(baseline, name),) if name in locked else values

    estimated_resources: dict[MatmulGeometry, KernelResources] = {}

    def costs(point: MatmulGeometry) -> tuple[float, float] | None:
        bm, bn, bk = point.block_m, point.block_n, point.block_k
        threads = 32 * point.warps  # CUDA warp width, independent of SM generation.
        shared = (bm + bn) * bk * element_bytes * point.stages
        registers_per_thread = (
            math.ceil(bm * bn / threads)
            + math.ceil((bm + bn) * bk * element_bytes / (4 * threads))
            + limits.register_reserve
        )
        residents = min(device.threads // threads,
                        device.registers // (registers_per_thread * threads),
                        device.shared // max(1, shared))
        if residents < 1:
            return None
        row_tiles = math.ceil(rows / bm)
        column_tiles = sum(math.ceil(n / bn) for n in columns)
        blocks = groups * row_tiles * column_tiles
        estimated_resources[point] = KernelResources(
            blocks, threads, registers_per_thread * threads, shared,
        )
        # Account separately for an underfilled device and resident warp supply.
        supply = max(1.0, device.sms / blocks)
        # Resident warp count is a feasibility bound, not a measured MMA rate.
        # Maximizing it rewards extra warps and shallow pipelines even when
        # their matrix layout is redundant or their loads cannot overlap.
        # Price actual rates and collective coexistence after compilation.
        work = blocks * bm * bn * reduction * supply
        traffic = blocks * (bm + bn) * reduction * element_bytes * supply
        return work, traffic

    anchor = costs(baseline)
    candidates = []
    for bm in choices("block_m", powers(min(limits.max_block_m, max(16, rows)))):
        for bn in choices("block_n", powers(min(limits.max_block_n, max(columns)))):
            for warps in choices("warps", tuple(w for w in (4, 8) if w <= limits.max_warps)):
                for stages in choices("stages", tuple(range(1, limits.max_stages + 1))):
                    point = MatmulGeometry(bm, bn, baseline.block_k, warps, stages)
                    value = costs(point)
                    if value is not None and point != baseline:
                        candidates.append((point, value))
    if not candidates or limits.max_candidates == 1:
        return (baseline,)
    scale = anchor or tuple(min(v[i] for _, v in candidates) for i in range(2))
    candidates.sort(key=lambda item: (
        (work_cost.predict(2 * item[1][0], item[1][1], launches=0)
         if work_cost is not None else max(item[1][i] / scale[i] for i in range(2))),
        sum(item[1][i] / scale[i] for i in range(2)),
        abs(item[0].warps - baseline.warps), abs(item[0].stages - baseline.stages),
        item[0].warps, -item[0].stages, item[0].block_m, item[0].block_n,
    ))
    selected = [baseline]
    if communication and baseline in estimated_resources:
        # Preserve SM headroom for nodes excluded by the anchor. Grid size is
        # known exactly; estimated register/shared counts cannot prove that a
        # smaller pipeline really co-resides after compiler layout conversion.
        anchor_resources = estimated_resources[baseline]
        excluded = tuple(node for node in communication
                         if resource_exclusion(anchor_resources, (node,), device) > 0)
        if excluded:
            def headroom(point: MatmulGeometry) -> float:
                return resource_exclusion(estimated_resources[point], excluded, device,
                                          allow_co_residency=False)

            fit, _ = min(candidates, key=lambda item: headroom(item[0]))
            if headroom(fit) < headroom(baseline):
                selected.append(fit)
    # Equal arithmetic/traffic proxies do not imply equal compiled pipelines.
    # In particular, staging reconstructed weights can materialize a wider
    # intermediate and add shared-memory traffic absent from this cheap model.
    # Preserve one minimal-storage pipeline at the *same* placement, so cold
    # calibration can distinguish that effect from M/N padding and reuse.
    # This is a bounded identifiability probe, not a prediction that shallow
    # pipelines are faster. Explicit stage overrides remain binding.
    pipelines = [p for p, _ in candidates
                 if (p.block_m, p.block_n, p.warps) ==
                 (baseline.block_m, baseline.block_n, baseline.warps)
                 and p.stages != baseline.stages]
    if pipelines and len(selected) < limits.max_candidates and "stages" not in locked:
        compact = min(pipelines, key=lambda p: estimated_resources[p].shared)
        if compact not in selected:
            selected.append(compact)
    # Spend the remaining budget on distinct output placements.
    placements = {(p.block_m, p.block_n) for p in selected}
    vectors = {costs(p) for p in selected}
    for point, value in candidates:
        if len(selected) >= limits.max_candidates:
            break
        placement = point.block_m, point.block_n
        if placement not in placements and value not in vectors:
            selected.append(point)
            placements.add(placement)
            vectors.add(value)
            if len(selected) == limits.max_candidates:
                break
    for point, _ in candidates:
        if len(selected) >= limits.max_candidates:
            break
        if point not in selected:
            selected.append(point)
    return tuple(selected)


def linear_geometries(
    device: DeviceResources, *, rows: int, columns: tuple[int, ...], reduction: int,
    element_bytes: int, baseline: MatmulGeometry, parameters: LinearParameters,
    communication: Sequence[KernelResources] = (),
    partitions: int = 1,
    work_cost: WorkCost | None = None,
) -> tuple[MatmulGeometry, ...]:
    """Share projection override semantics across native and packed engines."""
    explicit = frozenset().union(*(parameter_overrides("linear", n=n, k=reduction)
                                   for n in columns))
    locked = frozenset(name for name, key in (
        ("block_m", "fixed_block_m"), ("block_n", "fixed_block_n"),
        ("warps", "fixed_warps"), ("stages", "fixed_stages"),
    ) if key in explicit)
    limits = launch_parameters()
    for key in ("fixed_max_block_m", "overlap_block_m"):
        if key in explicit:
            limits = replace(limits, max_block_m=min(limits.max_block_m, getattr(parameters, key)))
    if "overlap_block_n" in explicit:
        limits = replace(limits, max_block_n=min(limits.max_block_n, parameters.overlap_block_n))
    if partitions < 1 or reduction % partitions:
        raise ValueError("linear partitions must divide the fixed reduction extent")
    return matmul_geometries(device, rows=rows, columns=columns, reduction=reduction // partitions,
                            groups=partitions,
                            element_bytes=element_bytes, baseline=baseline,
                            locked=locked, limits=limits, communication=communication,
                            work_cost=work_cost)


def overlapping_cost(
    compute_ms: float,
    compute: KernelResources,
    communication_ms: float,
    communication: Sequence[KernelResources],
    device: DeviceResources,
) -> float:
    """Estimate between ideal overlap and serial execution, never an SLA.

Resource exclusion is necessary evidence; resource compatibility alone does not
guarantee CUDA co-scheduling. Whole-graph qualification checks the prediction.
Keep each communication node's footprint coherent rather than combining maxima
from different kernels into an impossible synthetic block.
"""
    blocked = resource_exclusion(compute, communication, device)
    if not communication:
        return compute_ms
    return max(compute_ms, communication_ms) + blocked * min(compute_ms, communication_ms)


def resource_exclusion(
    compute: KernelResources, communication: Sequence[KernelResources], device: DeviceResources,
    *, allow_co_residency: bool = True,
) -> float:
    """Fraction of collective SM demand excluded by this coherent CTA footprint."""
    free_sms = max(0, device.sms - compute.blocks)
    blocked = 0.0
    for kernel in communication:
        fits = allow_co_residency and (
            compute.threads + kernel.threads <= device.threads
            and compute.registers + kernel.registers <= device.registers
            and compute.shared + kernel.shared <= device.shared
        )
        if not fits:
            demand = min(device.sms, kernel.blocks)
            blocked = max(blocked, max(0, demand - free_sms) / max(1, demand))
    return blocked


class _KernelNodeParams(ct.Structure):
    # CUDA_KERNEL_NODE_PARAMS_v2, CUDA 13.0 cuda.h.
    _fields_: ClassVar[list[tuple[str, Any]]] = [
        ("func", ct.c_void_p),
        *((name, ct.c_uint) for name in ("gx", "gy", "gz", "bx", "by", "bz", "shared")),
        ("parameters", ct.c_void_p), ("extra", ct.c_void_p),
        ("kernel", ct.c_void_p), ("context", ct.c_void_p),
    ]


def _driver() -> Any:
    driver = ct.CDLL("libcuda.so.1")
    driver.cuGraphGetNodes.argtypes = [
        ct.c_void_p, ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_size_t),
    ]
    driver.cuGraphNodeGetType.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int)]
    driver.cuGraphChildGraphNodeGetGraph.argtypes = [ct.c_void_p, ct.POINTER(ct.c_void_p)]
    driver.cuGraphKernelNodeGetParams_v2.argtypes = [ct.c_void_p, ct.POINTER(_KernelNodeParams)]
    driver.cuFuncGetAttribute.argtypes = [ct.POINTER(ct.c_int), ct.c_int, ct.c_void_p]
    driver.cuKernelGetFunction.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p]
    driver.cuDeviceGetAttribute.argtypes = [ct.POINTER(ct.c_int), ct.c_int, ct.c_int]
    return driver


def _check(status: int) -> None:
    if status:
        raise RuntimeError(f"CUDA launch-cost introspection failed: driver status {status}")


def graph_kernel_resources(graph: torch.cuda.CUDAGraph) -> tuple[KernelResources, ...]:
    """Read live kernel nodes before disposing this cold-start-only graph."""
    driver = _driver()
    result: list[KernelResources] = []

    def visit(handle: int) -> None:
        count = ct.c_size_t()
        _check(driver.cuGraphGetNodes(handle, None, ct.byref(count)))
        nodes = (ct.c_void_p * count.value)()
        _check(driver.cuGraphGetNodes(handle, nodes, ct.byref(count)))
        for node in nodes:
            kind = ct.c_int()
            _check(driver.cuGraphNodeGetType(node, ct.byref(kind)))
            if kind.value == 4:  # CU_GRAPH_NODE_TYPE_GRAPH
                child = ct.c_void_p()
                _check(driver.cuGraphChildGraphNodeGetGraph(node, ct.byref(child)))
                visit(child.value)
            elif kind.value == 0:  # CU_GRAPH_NODE_TYPE_KERNEL
                params = _KernelNodeParams()
                _check(driver.cuGraphKernelNodeGetParams_v2(node, ct.byref(params)))
                function = ct.c_void_p(params.func)
                if not function.value:
                    _check(driver.cuKernelGetFunction(ct.byref(function), params.kernel))
                registers, shared = ct.c_int(), ct.c_int()
                _check(driver.cuFuncGetAttribute(ct.byref(registers), 4, function))
                _check(driver.cuFuncGetAttribute(ct.byref(shared), 1, function))
                threads = params.bx * params.by * params.bz
                result.append(KernelResources(
                    params.gx * params.gy * params.gz, threads,
                    registers.value * threads, shared.value + params.shared,
                ))

    visit(graph.raw_cuda_graph())
    return tuple(result)


def device_resources(device: torch.device) -> DeviceResources:
    properties = torch.cuda.get_device_properties(device)
    registers = ct.c_int()
    # MAX_REGISTERS_PER_MULTIPROCESSOR, not Triton's per-block register limit.
    _check(_driver().cuDeviceGetAttribute(ct.byref(registers), 82, device.index))
    return DeviceResources(
        properties.multi_processor_count, properties.max_threads_per_multi_processor,
        registers.value, properties.shared_memory_per_multiprocessor,
    )


def _time_graph(
    graph: torch.cuda.CUDAGraph, samples: int, prepare: Callable[[], None] | None = None,
) -> float:
    if prepare is not None:
        prepare()
    graph.replay()
    times = []
    for _ in range(samples):
        if prepare is not None:
            prepare()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return median(times)


_Choice = TypeVar("_Choice")


def _equal_calibration_output(
    actual: torch.Tensor, expected: torch.Tensor, *, max_elements: int = 8 << 20,
) -> bool:
    """Preserve torch.equal semantics without a full-state comparison temporary.

    Narrowing uses views, including for strided operands. Split again if one
    slice still exceeds the bound. This is cold validation after launch finishes;
    it neither samples state nor changes the operator's comparison contract.
    """
    if type(max_elements) is not int or max_elements <= 0:
        raise ValueError("calibration comparison requires a positive element bound")
    if actual.shape != expected.shape:
        return False
    if actual.numel() <= max_elements:
        return torch.equal(actual, expected)
    axis = max(range(actual.ndim), key=lambda i: actual.shape[i])
    step = max(1, max_elements // (actual.numel() // actual.shape[axis]))
    for start in range(0, actual.shape[axis], step):
        length = min(step, actual.shape[axis] - start)
        if not _equal_calibration_output(
            actual.narrow(axis, start, length), expected.narrow(axis, start, length),
            max_elements=max_elements,
        ):
            return False
    return True


@lru_cache(maxsize=1)
def _launch_calibration_sources() -> str:
    # Vendor CUDA kernels used by the generic device-work probe, not Python
    # launch glue or an installed-package/project revision.
    from cubic_serve.ops.cuda.compiled_artifact import _mapped_cublas_hashes
    return _mapped_cublas_hashes()


@lru_cache(maxsize=32)
def _launch_calibration_device(device: torch.device) -> tuple[Any, ...]:
    return gpu_model(device)


def _launch_calibration_path(key: Any, candidates: Sequence[Any], device: torch.device, *, kernels=None):
    def encode(value: Any) -> Any:
        from triton.language import dtype
        if isinstance(value, torch.device):
            return value.type
        if isinstance(value, torch.dtype):
            return str(value)
        if isinstance(value, dtype):
            return ("triton.dtype", value.name)
        if is_dataclass(value) and not isinstance(value, type):
            return (type(value).__module__, type(value).__qualname__, asdict(value))
        raise TypeError("launch fixture has no portable identity")
    try:
        identity = (key, tuple(candidates), asdict(launch_parameters()))
        entry = json.loads(json.dumps(identity, sort_keys=True, default=encode, allow_nan=False))
        images = _launch_calibration_sources() if kernels is None else kernels
        return plan_path("launch", images, device, entry)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


_PLAN_READS: ContextVar[dict[str, str | None] | None] = ContextVar("startup_plan_reads", default=None)


def _plan_file_digest(path: Path) -> str | None:
    try:
        return hashlib.blake2s(path.read_bytes()).hexdigest()
    except OSError:
        return None


@contextmanager
def startup_plan_dependencies():
    """Record only plans consulted while constructing a resource portfolio.

    Missing entries matter too: installing a measured choice later invalidates
    the resource estimate made with its default. No tensor or graph is retained.
    """
    reads: dict[str, str | None] = {}
    token = _PLAN_READS.set(reads)
    try:
        yield reads
    finally:
        _PLAN_READS.reset(token)


def _read_startup_calibration(path: Path | None, schema: str) -> dict[str, Any] | None:
    if path is None:
        return None
    if (reads := _PLAN_READS.get()) is not None:
        reads[str(path.resolve())] = _plan_file_digest(path)
    try:
        envelope = json.loads(path.read_text())
        body = envelope["body"]
        digest = hashlib.blake2s(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if (envelope["digest"] == digest and body["key"] == path.stem
                and body["schema"] == schema):
            from cubic_serve.ops.cuda.startup import calibration_allowed
            if body.get("origin") == "production_default" and calibration_allowed():
                return None
            return body
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None


def _read_launch_calibration(path: Path | None, count: int) -> int | None:
    body = _read_startup_calibration(path, "cubic.launch-calibration.v1")
    if body is None:
        return None
    try:
        chosen, scores = body["chosen"], body["scores"]
        if (body["fixture_equivalence"] is True
                and type(chosen) is int and 0 <= chosen < count and len(scores) == count
                and all(value is None or (type(value) in (int, float)
                                          and math.isfinite(value) and value >= 0) for value in scores)
                and scores[chosen] is not None
                and chosen == min(range(count), key=lambda i: math.inf if scores[i] is None else scores[i])):
            return chosen
    except (OSError, ValueError, TypeError, KeyError):
        pass
    if body.get("basis") == "retained_default" and body.get("chosen") == 0:
        return 0
    return None


def _write_startup_calibration(path: Path | None, schema: str, values: dict[str, Any]) -> None:
    if path is None:
        return
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        body = dict(values, schema=schema, key=path.stem)
        digest = hashlib.blake2s(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as out:
            temporary = Path(out.name)
            json.dump(dict(body=body, digest=digest), out, allow_nan=False)
        temporary.replace(path)
        if (reads := _PLAN_READS.get()) is not None:
            reads[str(path.resolve())] = _plan_file_digest(path)
    except (OSError, TypeError, ValueError):
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


@contextmanager
def local_plan_lock(path: Path | None):
    """Serialize local-kernel selection, never a distributed calibration.

    As with the native compiler cache, a cold consumer locks and rechecks the
    entry before doing work. A complete model/GPU-shared plan then has one
    publisher; another rank cannot select a different plan for the same entry.
    """
    if path is None:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _write_launch_calibration(path: Path | None, chosen: int, scores: list[float]) -> None:
    _write_startup_calibration(path, "cubic.launch-calibration.v1", dict(
        chosen=chosen, scores=[value if math.isfinite(value) else None for value in scores],
        fixture_equivalence=True))


@lru_cache(maxsize=32)
def startup_operator_choices(device: torch.device) -> dict[tuple[Any, ...], int]:
    """Retain geometry choices across bodies when no communication scope exists."""
    return {}


def select_launch(
    candidates: Sequence[_Choice],
    *,
    device: torch.device,
    bind: Callable[[_Choice], None],
    launch: Callable[[], None],
    snapshot: Callable[[], tuple[torch.Tensor, ...]],
    prepare: Callable[[], None] | None = None,
    score: Callable[[float, tuple[KernelResources, ...]], float] | None = None,
    cache_key: Any = None,
    score_key: Any = None,
    kernels: Any = None,
) -> _Choice:
    """Cold-only operator calibration with caller-owned immutable test operands.

    The operator owns binding, state reset and numerical boundaries. This shared
    layer owns capture lifetime, resource inspection and bounded measurement.
    Nothing here observes or changes a live request or captures after readiness.
    """
    from triton.runtime.errors import OutOfResources

    from cubic_serve.ops.cuda.startup import startup_cuda_lock

    with torch.cuda.device(device), startup_cuda_lock(device):
        if not candidates or torch.cuda.is_current_stream_capturing():
            raise ValueError("launch selection requires cold nonempty candidates")
        # Only callers declaring an exhaustive, source-independent synthetic
        # fixture key opt in. A custom score must declare its dependencies too.
        if kernels is None:
            owner = inspect.getmodule(bind)
            kernels = None if owner is None else module_kernels(owner)
        path = (_launch_calibration_path((cache_key, score_key), candidates, device, kernels=kernels)
                if cache_key is not None and (score is None or score_key is not None) else None)
        cached = _read_launch_calibration(path, len(candidates))
        if cached is not None:
            bind(candidates[cached])
            if prepare is not None:
                prepare()
            return candidates[cached]
        with local_plan_lock(path):
            cached = _read_launch_calibration(path, len(candidates))
            if cached is not None:
                bind(candidates[cached])
                if prepare is not None:
                    prepare()
                return candidates[cached]
            from cubic_serve.ops.cuda.startup import calibration_allowed
            if not calibration_allowed():
                bind(candidates[0])
                if prepare is not None:
                    prepare()
                _write_startup_calibration(path, "cubic.launch-calibration.v1",
                                           {"basis": "retained_default", "chosen": 0,
                                            "origin": "production_default"})
                return candidates[0]
            from cubic_serve.ops.cuda.startup import require_calibration_permission
            require_calibration_permission("launch selection")
            reference = None
            measured = []
            for point in candidates:
                try:
                    bind(point)
                except OutOfResources:
                    # Geometry uses a cheap resource estimate. A format's decoder
                    # and compiler layout may exceed it; discard only this cold
                    # candidate, never a numerical failure or a serving replay.
                    measured.append(float("inf"))
                    continue
                if prepare is not None:
                    prepare()
                launch()
                torch.cuda.synchronize(device)
                outputs = snapshot()
                if reference is None:
                    reference = tuple(t.clone() for t in outputs)
                elif len(outputs) != len(reference) or any(
                    not _equal_calibration_output(t, r)
                    for t, r in zip(outputs, reference, strict=True)
                ):
                    raise RuntimeError("launch geometry changed calibrated output/state bits")
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                try:
                    with torch.cuda.graph(graph):
                        launch()
                    resources = graph_kernel_resources(graph)
                    graph.instantiate()
                    elapsed = _time_graph(graph, launch_parameters().calibration_samples, prepare)
                    measured.append(score(elapsed, resources) if score else elapsed)
                finally:
                    graph.reset()
            chosen = min(range(len(measured)), key=measured.__getitem__)
            if not math.isfinite(measured[chosen]):
                raise RuntimeError("no resource-qualified launch candidate")
            bind(candidates[chosen])
            if prepare is not None:
                prepare()
            _write_launch_calibration(path, chosen, measured)
            return candidates[chosen]


@dataclass(frozen=True)
class LinearTile:
    kernel: Any
    grid: tuple[int, int, int]
    block_m: int
    block_n: int
    geometry: MatmulGeometry | None = None


@dataclass
class LinearLaunchCost:
    device: DeviceResources
    communication_ms: float
    communication: tuple[KernelResources, ...]
    choices: dict[tuple[Any, ...], int] = field(default_factory=dict)
    operator_choices: dict[tuple[Any, ...], int] = field(default_factory=dict)

    @classmethod
    def calibrate(
        cls, template: torch.Tensor, launch: Callable[[torch.Tensor], None],
        *, group: Any, samples: int,
    ) -> LinearLaunchCost:
        """Measure the actual payload/transport with private input, before readiness."""
        owner = inspect.getmodule(launch)
        images = (cuda_libraries(("libnccl",)),
                  () if owner is None else module_kernels(owner))
        path = _launch_calibration_path(
            ("collective", tuple(template.shape), tuple(template.stride()), str(template.dtype),
             dist.get_world_size(group), dist.get_rank(group), samples), (), template.device,
            kernels=images)
        raw = _read_startup_calibration(path, "cubic.collective-cost.v1")
        retained = None
        if raw is not None:
            try:
                milliseconds = raw["communication_ms"]
                if type(milliseconds) in (float, int) and math.isfinite(milliseconds) and milliseconds > 0:
                    retained = cls(device_resources(template.device), milliseconds,
                                   tuple(KernelResources(**item) for item in raw["resources"]))
            except (TypeError, KeyError, ValueError):
                pass
        agreement = torch.tensor([int(retained is not None)], dtype=torch.int32,
                                 device=template.device)
        dist.all_reduce(agreement, op=dist.ReduceOp.MIN, group=group)
        if bool(agreement.item()):
            return retained
        value = torch.zeros_like(template)
        launch(value)
        torch.cuda.synchronize(template.device)
        dist.barrier(group=group)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        try:
            with torch.cuda.graph(graph):
                launch(value)
            resources = graph_kernel_resources(graph)
            if not resources:
                raise RuntimeError("collective calibration captured no kernel resources")
            graph.instantiate()
            elapsed = torch.tensor([_time_graph(graph, samples)], device=template.device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group)
            measured = float(elapsed.item())
            _write_startup_calibration(path, "cubic.collective-cost.v1",
                {"communication_ms": measured, "resources": [asdict(item) for item in resources]})
            return cls(device_resources(template.device), measured, resources)
        finally:
            graph.reset()

    def select(
        self, tiles: Sequence[LinearTile], x: torch.Tensor,
        weights: tuple[torch.Tensor, ...], *, grouped: bool, masked: bool = True,
        replays: int = 1,
        partitions: int = 1,
    ) -> LinearTile:
        """Bounded compute-rate calibration; the joint cost formula selects the tile.

Use private finite operands, preserve the input RNG and compare candidate output
before measuring. This local identity check is not the full-model eager oracle.
No shape-specific rates or measured device constants ship with the engine.
"""
        if type(replays) is not int or replays < 1:
            raise ValueError("linear calibration replays must be a positive integer")
        if len(tiles) == 1:
            return tiles[0]
        if partitions < 1 or (partitions > 1 and masked and not grouped):
            raise ValueError("fixed partial calibration requires grouped or unmasked linear")
        images = tuple(hashlib.blake2s(t.kernel.asm["cubin"]).hexdigest() for t in tiles)
        key = (images, tuple(x.shape), tuple(x.stride()), x.dtype, replays, partitions)
        if key in self.choices:
            return tiles[self.choices[key]]
        outputs, arguments, merge_arguments = (), (), ()
        merge, merge_grid = None, (1, 1, 1)
        merge_kernel = None
        merge_block = 256
        merge_options = {"num_warps": 4, "enable_fp_fusion": False}
        if partitions > 1:
            if grouped:
                from cubic_serve.ops.cuda.linear.grouped import _grouped_partition_merge
                merge_kernel = _grouped_partition_merge
            else:
                from cubic_serve.quantization.common.kernels import _merge_split_k
                merge_kernel = _merge_split_k
        # Actual candidate cubins and the lazy merge's JIT/launch contract own
        # these prices. Unrelated native files beside this Python module do not.
        kernel_dependencies = (tuple(sorted(set(images))), () if merge_kernel is None else (
            merge_kernel.cache_key, merge_block, tuple(sorted(merge_options.items())),
        ))

        def prepare_fixture():
            nonlocal outputs, arguments, merge, merge_arguments, merge_grid
            if outputs:
                return
            generator = torch.Generator(device=x.device).manual_seed(0)
            operand = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
            outputs = tuple(torch.empty((x.shape[0], w.shape[0]), device=x.device, dtype=x.dtype)
                            for w in weights)
            active = torch.tensor([x.shape[0]], dtype=torch.int32, device=x.device)
            arguments = (operand, weights, outputs, active) if grouped else (
                operand, weights[0], outputs[0], active
            )
            if not masked:
                if grouped:
                    raise ValueError("unmasked grouped linear calibration is unsupported")
                arguments = arguments[:-1]
            merge = None
            if partitions > 1:
                import triton  # type: ignore[import-untyped]

                if grouped:
                    ns = tuple(w.shape[0] for w in weights)
                    partials = torch.empty((partitions * sum(out.numel() for out in outputs),),
                                           device=x.device, dtype=torch.float32)
                    arguments = (operand, weights, partials, active)
                    merge_arguments = (partials, outputs, active)
                    merge_grid = (sum(triton.cdiv(out.numel(), merge_block) for out in outputs), 1, 1)
                    merge = merge_kernel.warmup(
                        *merge_arguments, ns, x.shape[0], partitions, merge_block,
                        grid=merge_grid, **merge_options,
                    )
                else:
                    partials = torch.empty((partitions, *outputs[0].shape),
                                           device=x.device, dtype=torch.float32)
                    arguments = (operand, weights[0], partials)
                    merge_arguments = (partials, outputs[0])
                    merge_grid = (triton.cdiv(outputs[0].numel(), merge_block), 1, 1)
                    merge = merge_kernel.warmup(
                        *merge_arguments, outputs[0].numel(), partitions, merge_block,
                        grid=merge_grid, **merge_options,
                    )
                merge._init_handles()
        eligible = [i for i, tile in enumerate(tiles) if not tile.kernel.n_spills]
        if not eligible:
            raise RuntimeError("no spill-free linear cost candidate")
        selected = tiles[eligible[0]]

        def bind(index: int) -> None:
            nonlocal selected
            selected = tiles[index]

        def launch() -> None:
            prepare_fixture()
            stream = int(torch.cuda.current_stream(x.device).cuda_stream)
            # Repeated nodes amortize submission/event gaps for microsecond
            # kernels. This is cold calibration only, never serving work.
            for _ in range(replays):
                selected.kernel[selected.grid](*arguments, stream=stream)
                if merge is not None:
                    merge[merge_grid](*merge_arguments, stream=stream)

        def score(elapsed: float, resources: tuple[KernelResources, ...]) -> float:
            stride = 2 if merge is not None else 1
            if len(resources) != stride * replays or any(
                r != resources[i % stride] for i, r in enumerate(resources)
            ):
                raise RuntimeError("linear calibration requires one coherent kernel footprint")
            return max(overlapping_cost(
                elapsed / replays, node, self.communication_ms,
                self.communication, self.device,
            ) for node in resources[:stride])

        chosen = select_launch(
            eligible, device=x.device, bind=bind, launch=launch,
            snapshot=lambda: outputs, score=score,
            cache_key=weight_bound_calibration_key(
                ("linear", key, grouped, masked,
                 tuple((tuple(w.shape), tuple(w.stride()), str(w.dtype)) for w in weights))),
            score_key=(self.device, self.communication_ms, self.communication),
            kernels=kernel_dependencies,
        )
        self.choices[key] = chosen
        return tiles[chosen]


_LINEAR_COST: ContextVar[LinearLaunchCost | None] = ContextVar("linear_launch_cost", default=None)


def current_linear_cost() -> LinearLaunchCost | None:
    return _LINEAR_COST.get()


@contextmanager
def linear_cost_scope(cost: LinearLaunchCost) -> Iterator[None]:
    token = _LINEAR_COST.set(cost)
    try:
        yield
    finally:
        _LINEAR_COST.reset(token)
