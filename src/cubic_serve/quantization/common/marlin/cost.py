"""Cold native calibration and a held-out, resource/working-set cost regression.

Calibration shapes depend on device supply and matrix geometry, never the requested
logical M. Coefficients include the actual decoder, register allocation and pipeline.
No checkpoint tensors, live inputs, graph buffers or serving RNG enter a probe.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from functools import cache, lru_cache, partial
from pathlib import Path

import torch

from cubic_serve.ops.cuda.cache_identity import gpu_model, native_headers, plan_path
from cubic_serve.ops.cuda.launch_cost import (
    _launch_calibration_path,
    _read_startup_calibration,
    _time_graph,
    _write_startup_calibration,
    fit_nonnegative_cost,
    local_plan_lock,
    predicted_gain,
    prediction_interval,
)
from cubic_serve.ops.cuda.native import CudaScalar
from cubic_serve.ops.strategies import (
    launch_parameters,
    operator_strategy_scope,
    parameter_overrides,
)
from cubic_serve.quantization.common.binding import PackedOperatorBinding, _PackedLaunch
from cubic_serve.quantization.common.marlin.compiler import (
    KernelResources,
    bind_launch,
    compile_module,
    kernel_resources,
)
from cubic_serve.quantization.common.marlin.parameters import (
    MarlinParameters,
    pipeline_compatible,
    shared_bytes,
)
from cubic_serve.quantization.common.marlin.repack import _merge_dense_partials
from cubic_serve.runtime.logging import calibration_k_plan


def fixed_k_partitions(*, n, k, group, block_n, block_k, sms, supply_factor, max_partitions):
    """Bound sparse-row supply without paying for a worse final SM work wave.

    Partition ownership is independent of M. Bound candidates by the configured
    supply target, then price tail K work and fractional SM coverage. Coverage
    is a cost, not a hard gate: a nearly full wave can beat extra short slices.
    Ties use fewer partials to protect high-row merge bandwidth.
    """
    quantum = math.lcm(group, block_k)
    if n < block_n or n % block_n or k % quantum or sms < 1 or max_partitions < 1:
        raise ValueError("fixed K partitions require aligned geometry and positive resources")
    units, tiles = k // quantum, n // block_n
    legal = [p for p in range(1, min(units, max_partitions) + 1) if units % p == 0]
    upper = min((p for p in legal if tiles * p >= supply_factor * sms), default=legal[-1])
    return min(
        (p for p in legal if p <= upper),
        key=lambda p: (
            math.ceil(tiles * p / sms) * (units // p) / min(1.0, tiles * p / sms),
            p,
        ),
    )


@dataclass(frozen=True)
class ProbeFormat:
    """Format-owned finite arithmetic fixture; no format-name dispatch."""

    bits: int
    activation_dtype: torch.dtype
    output_dtype: torch.dtype
    scale_dtype: torch.dtype
    weight_word: int
    scale_value: float
    expected_weight: float
    global_scale: float | None = None
    activation_scale: float | None = None
    zero_word: int | None = None
    group_metadata: tuple[float, ...] = ()
    column_scale: bool = False
    fixed_partitions: int = 0
    dense_linear: bool = False
    storage_bits: int = 0
    group_storage_bytes: float = 0
    metadata_packer: Callable[[torch.Tensor], torch.Tensor] | None = None
    min_block_m: int = 16
    single_subtile: bool = False
    native_dense_reduce: bool = False


@dataclass(frozen=True)
class NativeCost:
    coefficients: tuple[float, float, float]
    error: float
    sms: int
    l2_bytes: int
    block_m: int
    block_n: int
    n: int
    k: int
    weight_bytes: float
    output_bytes: int
    input_bytes: int = 2
    fixed_partitions: int = 0
    routed_partials: bool = False
    # Complete cold calibration and held-out measurements. No live routes,
    # checkpoint values, model names or public concurrency enter this evidence.
    samples: tuple[tuple[int, int, float, bool], ...] = ()

    def features(self, active: float, rows: float, row_tiles: float) -> tuple[float, ...]:
        # Residency controls legal launch supply. CTAs still share SM issue
        # units: occupancy must not divide their instruction work a second time.
        supply = row_tiles * (self.n // self.block_n) * max(1, self.fixed_partitions) / self.sms
        # Independent fixed partitions do not spin-wait on a predecessor CTA.
        # Their partly occupied final wave costs proportional issue work.
        waves = max(0.0, (supply if self.fixed_partitions else math.ceil(supply)) - 1)
        footprint = active * self.weight_bytes + rows * (
            self.k * self.input_bytes + self.n * self.output_bytes
        )
        # Fixed partials add traffic even when weights fit L2; their merge is
        # real work and grows with output rows. Keep the whole-K model unchanged.
        partial_bytes = (
            self.n * (
                4 * self.fixed_partitions * (
                    (row_tiles * self.block_m if self.routed_partials else rows) + rows
                ) + rows * self.output_bytes
            )
            if self.fixed_partitions
            else 0
        )
        return 1.0, float(waves), (max(0.0, footprint - self.l2_bytes) + partial_bytes) / 1e8

    def predict(self, active: float, rows: float, row_tiles: float) -> float:
        return sum(
            a * b
            for a, b in zip(self.coefficients, self.features(active, rows, row_tiles), strict=True)
        )

    def relative_interval(
        self, baseline: NativeCost, active: float, rows: float, row_tiles: float,
    ) -> tuple[tuple[float, float], float] | None:
        """Validate a ratio on the complete matching cold/held-out corpus.

        Shared absolute modelling error is not two independent errors. The
        ratio has its own observed-relative residual, including every holdout.
        Different geometry, incomplete evidence and extrapolation retain the
        independent absolute-bound policy.
        """
        geometry = ("sms", "block_m", "block_n", "n", "k", "weight_bytes",
                    "output_bytes", "input_bytes", "fixed_partitions", "routed_partials")
        if (not isinstance(baseline, NativeCost) or not self.samples
                or not _valid_samples(self.samples) or not _valid_samples(baseline.samples)
                or any(getattr(self, key) != getattr(baseline, key) for key in geometry)):
            return None
        original = {(a, r, held_out): time for a, r, time, held_out in baseline.samples}
        observed = {(a, r, held_out): time for a, r, time, held_out in self.samples}
        if (observed.keys() != original.keys() or not original
                or {held_out for _a, _r, held_out in original} != {False, True}):
            return None
        domain = tuple((a, a * r, a * math.ceil(r / self.block_m))
                       for a, r, _held_out in original)
        if any(not min(point[i] for point in domain) <= value <= max(point[i] for point in domain)
               for i, value in enumerate((active, rows, row_tiles))):
            return None
        if (active <= 0
                or not min(r for _a, r, _held in original) <= rows / active <= max(
                    r for _a, r, _held in original)
                or row_tiles / active > max(math.ceil(r / self.block_m)
                                            for _a, r, _held in original)):
            return None
        error = 0.0
        for key, measured in observed.items():
            a, r, _held_out = key
            operands = a, a * r, a * math.ceil(r / self.block_m)
            prior = baseline.predict(*operands)
            candidate = self.predict(*operands)
            if min(prior, candidate, measured, original[key]) <= 0:
                return None
            ratio = candidate / prior
            actual_ratio = measured / original[key]
            residual = abs(ratio / actual_ratio - 1.0)
            if not math.isfinite(residual):
                return None
            error = max(error, residual)
        denominator = baseline.predict(active, rows, row_tiles)
        if denominator <= 0 or error >= 1:
            return None
        ratio = self.predict(active, rows, row_tiles) / denominator
        return prediction_interval(ratio, error), error

    @staticmethod
    def fit(features: list[tuple[float, ...]], times: list[float]) -> tuple[float, float, float]:
        """Three-term physical model, using the shared bounded fitter."""
        launch, waves, traffic = fit_nonnegative_cost(features, times)
        return launch, waves, traffic


def _calibration_identity(value):
    """Describe arithmetic fixtures exactly; unknown callbacks remain uncached."""
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, tuple | list):
        return [_calibration_identity(v) for v in value]
    if isinstance(value, dict):
        return {k: _calibration_identity(v) for k, v in value.items()}
    if isinstance(value, partial):
        return [_calibration_identity(value.func), _calibration_identity(value.args),
                _calibration_identity(value.keywords)]
    if inspect.isfunction(value) and not value.__closure__:
        path = inspect.getsourcefile(value)
        if path is not None:
            # Fixture callbacks describe entry data, not executable GPU code.
            # CUDA source/images below own invalidation of the actual kernel.
            return [value.__module__, value.__qualname__]
    if hasattr(value, "__dataclass_fields__"):
        return {f.name: _calibration_identity(getattr(value, f.name)) for f in fields(value)
                if not (isinstance(value, ProbeFormat) and f.name == "native_dense_reduce"
                        and not value.native_dense_reduce)}
    raise TypeError("calibration fixture has no portable identity")


@cache
def _calibration_implementation_identity():
    return _merge_dense_partials.cache_key


def _reduction_identity(spec):
    if spec.native_dense_reduce:
        from .a8_partial_reduce import source_identity
        return source_identity()
    return _calibration_implementation_identity()


def _calibration_cache_path(module, params, spec, model, experts, group, samples):
    try:
        kernels = (module.cubin_hash, _reduction_identity(spec))
        entry = _calibration_identity((params, spec, model, experts, group, samples))
        return plan_path("marlin", kernels, module.device, entry)
    except (OSError, TypeError, ValueError, AttributeError):
        return None


@lru_cache(maxsize=32)
def _calibration_device_identity(device):
    return gpu_model(device)


def _read_calibration(path, model):
    if path is None:
        return None
    try:
        record = json.loads(path.read_text())
        data = record["cost"]
        digest = hashlib.blake2s(
            json.dumps(data, sort_keys=True, allow_nan=False).encode(),
        ).hexdigest()
        if (set(record) != {"schema", "key", "digest", "cost"}
                or record["schema"] != 2 or record["key"] != path.stem
                or record["digest"] != digest
                or set(data) != {"coefficients", "error", "l2_bytes", "samples"}
                or len(data["coefficients"]) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                       for v in (*data["coefficients"], data["error"]))
                or type(data["l2_bytes"]) is not int or not 0 < data["l2_bytes"] <= model.l2_bytes
                or not _valid_samples(data["samples"])):
            return None
        return replace(model, coefficients=tuple(data["coefficients"]),
                       error=data["error"], l2_bytes=data["l2_bytes"],
                       samples=tuple(tuple(row) for row in data["samples"]))
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _valid_samples(samples):
    # The current bounded protocol has at most 11 anchors plus five holdouts.
    return (
        isinstance(samples, tuple | list) and len(samples) <= 16
        and all(isinstance(row, tuple | list) and len(row) == 4
                and all(type(value) is int and value > 0 for value in row[:2])
                and type(row[2]) in (float, int) and math.isfinite(row[2]) and row[2] > 0
                and type(row[3]) is bool for row in samples)
        and len({(row[0], row[1], row[3]) for row in samples}) == len(samples)
    )


def _write_calibration(path, model):
    if path is None:
        return
    data = dict(coefficients=model.coefficients, error=model.error,
                l2_bytes=model.l2_bytes, samples=model.samples)
    try:
        digest = hashlib.blake2s(
            json.dumps(data, sort_keys=True, allow_nan=False).encode(),
        ).hexdigest()
        record = dict(schema=2, key=path.stem, digest=digest, cost=data)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                json.dump(record, stream, sort_keys=True, allow_nan=False)
                stream.close()
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
    except OSError:
        pass


@cache
def _plan_compiler_identity():
    baseline = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.0")) / "bin/nvcc"
    nvcc = str(baseline) if baseline.is_file() else shutil.which("nvcc")
    if nvcc is None:
        raise OSError("native compiler unavailable")
    version = subprocess.run(
        [nvcc, "--version"], check=True, capture_output=True, text=True, timeout=10
    ).stdout
    return version, native_headers(Path(__file__).with_name("native"))


def _plan_path(kind, params, spec, device, points, geometry):
    # Reuse the common source/device/environment identity and atomic envelope.
    # No candidate compilation, GPU timing, or cost prediction is needed here.
    try:
        key = ("cubic.marlin-plan.v1", kind,
               _calibration_identity((params, spec, geometry)))
        kernels = (tuple(source for _, source in points), _plan_compiler_identity(),
                   _reduction_identity(spec))
        return _launch_calibration_path(key, tuple(point for point, _ in points), device,
                                        kernels=kernels)
    except (AttributeError, OSError, TypeError, ValueError, subprocess.SubprocessError):
        return None


def _read_plan(path, count, replace_production_default=False):
    body = _read_startup_calibration(path, "cubic.marlin-plan.v1")
    if body is not None:
        if replace_production_default and body.get("origin") == "production_default":
            return None
        chosen = body.get("chosen")
        if (body.get("complete") is True
                and body.get("basis") in ("timing_policy", "retained_default")
                and type(chosen) is int and 0 <= chosen < count):
            return chosen
    return None


def _write_plan(path, chosen, *, basis="timing_policy", origin=None):
    _write_startup_calibration(path, "cubic.marlin-plan.v1",
                               dict(chosen=chosen, complete=True, basis=basis, origin=origin))


def capacity_k_partitions(
    params, *, partitions, rows, n, k, group, spec, source, device,
):
    """Remove excess dense K partials once physical rows supply the device.

    The sparse-row plan remains the upper bound, including all explicit limits.
    Each startup graph capacity seals its own reduction; no live request changes
    it. Preserve the single-row/tile plan and the format's legal K boundaries.
    A complete warm plan returns before resource pricing, without GPU timings.
    """
    quantum = math.lcm(group, params.marlin_block_k)
    if (min(partitions, rows, n, k, group) < 1
            or n % params.marlin_block_n or k % (partitions * quantum)):
        raise ValueError("dense capacity partitions require a legal original reduction")
    if partitions == 1 or rows <= params.marlin_block_m:
        return partitions
    path = _plan_path(
        "dense-capacity-partitions.v1", params, spec, device,
        ((partitions, source(partitions)),),
        (rows, n, k, group, partitions),
    )
    cached = _read_plan(path, partitions)
    if cached is not None and k % ((cached + 1) * quantum) == 0:
        return cached + 1
    with local_plan_lock(path):
        cached = _read_plan(path, partitions)
        if cached is not None and k % ((cached + 1) * quantum) == 0:
            return cached + 1
        row_tiles = math.ceil(rows / params.marlin_block_m)
        tiles = row_tiles * (n // params.marlin_block_n)
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        required = math.ceil(sms / tiles)
        units = k // quantum
        chosen = next(
            (p for p in range(max(1, required), partitions + 1) if units % p == 0),
            partitions,
        )
        _write_plan(path, chosen - 1, basis="retained_default", origin="resource_default")
        return chosen


@lru_cache(maxsize=256)
def calibrate(
    source: str,
    p: MarlinParameters,
    spec: ProbeFormat,
    device: torch.device,
    n: int,
    k: int,
    experts: int,
    group: int,
    samples: int,
) -> NativeCost:
    from cubic_serve.ops.cuda.startup import startup_cuda_lock

    with startup_cuda_lock(device):
        return _calibrate(source, p, spec, device, n, k, experts, group, samples)


def _calibrate(
    source: str,
    p: MarlinParameters,
    spec: ProbeFormat,
    device: torch.device,
    n: int,
    k: int,
    experts: int,
    group: int,
    samples: int,
) -> NativeCost:
    """Fit independent cold shapes, then validate shapes excluded from the fit."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("native cost calibration must precede graph capture")
    hw = torch.cuda.get_device_properties(device)
    bm, bn = p.marlin_block_m, p.marlin_block_n
    model = NativeCost(
        (0.0, 0.0, 0.0),
        0.0,
        hw.multi_processor_count,
        hw.L2_cache_size,
        bm,
        bn,
        n,
        k,
        n * k * (spec.storage_bits or spec.bits) // 8
        + n
        * (k // group)
        * (
            spec.group_storage_bytes or 4 * len(spec.group_metadata)
            if spec.group_metadata
            else torch.empty((), dtype=spec.scale_dtype).element_size()
        ),
        torch.empty((), dtype=spec.output_dtype).element_size(),
        input_bytes=torch.empty((), dtype=spec.activation_dtype).element_size(),
        fixed_partitions=spec.fixed_partitions,
        routed_partials=not spec.dense_linear,
    )
    shared = shared_bytes(
        p,
        bits=spec.storage_bits or spec.bits,
        group=group,
        metadata_planes=len(spec.group_metadata),
        row_metadata=not spec.dense_linear,
        zero_points=not spec.dense_linear or spec.zero_word is not None,
    )
    # An exact implementation/fixture/device receipt retains the original fit
    # and holdout error. It is tuning evidence, never numerical qualification
    # of the model or this process's graph/state resources.
    with torch.cuda.device(device):
        torch.cuda.init()
        module, entry = compile_module(source, device)
    cache_path = _calibration_cache_path(module, p, spec, model, experts, group, samples)
    cached = _read_calibration(cache_path, model)
    if cached is not None:
        return cached
    from cubic_serve.ops.cuda.startup import require_calibration_permission
    require_calibration_permission("Marlin cost calibration")
    if spec.activation_dtype == torch.int8:
        bits = spec.storage_bits or spec.bits
        code = (1 << (bits - 1)) + 1 if bits < 8 else 1
        pattern = [
            sum(((code >> ((32 * word + bit) % bits)) & 1) << bit for bit in range(32))
            for word in range(bits)
        ]
        pattern = [v if v < 2**31 else v - 2**32 for v in pattern]
        weight = (
            torch.tensor(pattern, device=device, dtype=torch.int32)
            .repeat(experts * n * k // 32)
            .reshape(experts, k // 16, n * bits // 2)
        )
    elif spec.storage_bits and spec.storage_bits != spec.bits:
        bits = spec.storage_bits
        code = (
            (spec.weight_word & ((1 << spec.bits) - 1)) - (1 << (spec.bits - 1)) + (1 << (bits - 1))
        )
        pattern = [
            sum(((code >> ((32 * w + j) % bits)) & 1) << j for j in range(32)) for w in range(bits)
        ]
        pattern = [v if v < 2**31 else v - 2**32 for v in pattern]
        weight = (
            torch.tensor(pattern, device=device, dtype=torch.int32)
            .repeat(experts * n * k // 32)
            .reshape(experts, k // 16, n * bits // 2)
        )
    else:
        weight = torch.full(
            (experts, k // 16, n * spec.bits // 2),
            spec.weight_word,
            dtype=torch.int32,
            device=device,
        )
    scales = torch.full(
        (experts, k // group, n), spec.scale_value, dtype=spec.scale_dtype, device=device
    )
    correction = (
        None
        if spec.global_scale is None
        else torch.full((experts,), spec.global_scale, device=device)
    )
    if spec.group_metadata:
        if spec.global_scale is not None:
            raise ValueError("group metadata and scalar correction share one native operand")
        correction = (
            torch.tensor(spec.group_metadata, device=device, dtype=torch.float32)[
                None, :, None, None
            ]
            .expand(experts, -1, k // group, n)
            .contiguous()
        )
    if spec.metadata_packer is not None:
        correction = spec.metadata_packer(correction)
    zeros = (
        None
        if spec.zero_word is None
        else torch.full(
            (experts, k // group, n // (32 // spec.bits)), spec.zero_word, dtype=torch.int32, device=device
        )
    )
    @cache
    def measure(active: int, rows: int) -> float:
        r, padded = active * rows, math.ceil(rows / bm) * bm
        activation = 1 if spec.activation_dtype == torch.int8 else 0.125
        q = torch.full((r, k), activation, dtype=spec.activation_dtype, device=device)
        a_scales = (
            torch.full((r,), spec.activation_scale, dtype=torch.float32, device=device)
            if spec.activation_scale is not None else None
        )
        out = torch.full((r, n), float("nan"), dtype=spec.output_dtype, device=device)
        ids = (
            torch.arange(padded, device=device)[None, :]
            + torch.arange(active, device=device)[:, None] * rows
        ).to(torch.int32)
        ids[:, rows:] = r
        owners = torch.arange(active, device=device, dtype=torch.int32).repeat_interleave(
            padded // bm
        )
        total = torch.tensor([ids.numel()], dtype=torch.int32, device=device)
        routing = torch.ones(r, device=device)
        physical_rows = math.ceil(r / bm) * bm if spec.dense_linear else active * padded
        tiles = physical_rows // bm * (n // bn)
        partials = (
            torch.empty(tiles * spec.fixed_partitions * bm * bn, device=device, dtype=torch.float32)
            if spec.fixed_partitions
            else None
        )
        args = (
            q,
            weight,
            out,
            partials,
            None,
            a_scales,
            scales,
            correction,
            zeros,
            None,
            ids,
            owners,
            total,
            routing,
            CudaScalar("i32", 1),
            CudaScalar("bool", False),
            CudaScalar("i32", k // group),
            CudaScalar("i32", r),
            CudaScalar("i32", n),
            CudaScalar("i32", k),
            None,
            CudaScalar("bool", False),
            CudaScalar("bool", False),
            CudaScalar("bool", True),
            CudaScalar("i32", spec.fixed_partitions),
        )
        if spec.fixed_partitions:
            assert partials is not None
            multiply = module.bind(
                entry,
                args,
                operator_id="marlin.cost.partial.probe",
                grid=(tiles * spec.fixed_partitions, 1, 1),
                block=(p.marlin_warps * 32, 1, 1),
                shared_bytes=shared,
            )
            if spec.native_dense_reduce:
                from .a8_partial_reduce import bind as bind_partial_reduce
                if not spec.dense_linear or spec.activation_dtype != torch.int8:
                    raise ValueError("native dense reducer requires the A8 fragment contract")
                finish = bind_partial_reduce(partials, out, m=r, n=n, block_m=bm,
                                             block_n=bn, partitions=spec.fixed_partitions)
            else:
                finish = _PackedLaunch.compile(
                    _merge_dense_partials,
                    (partials if spec.dense_linear else (partials, ids, total), out),
                    (r, n, bm, bn, spec.fixed_partitions, 256, (), (),
                     not spec.dense_linear, spec.activation_dtype == torch.int8),
                    (math.ceil((r if spec.dense_linear else physical_rows) * n / 256),),
                    {"num_warps": 4, "enable_fp_fusion": False},
                )
            op = PackedOperatorBinding(out, (multiply, finish), scratch=(partials,))
        else:
            op = bind_launch(
                module,
                entry,
                args,
                operator_id="marlin.cost.probe",
                params=p,
                output_tiles=owners.numel() * (n // bn),
                shared_bytes=shared,
            )
        op.launch()
        torch.cuda.synchronize(device)
        expected = torch.tensor(
            activation * (spec.activation_scale or 1.0) * k * spec.expected_weight,
            dtype=out.dtype, device=device,
        )
        if not bool((out == expected).all()):
            raise RuntimeError(
                "native cost fixture violated its format arithmetic: "
                f"M={r} N={n} K={k} group={group} carrier={spec.bits} "
                f"tile=({bm},{bn},{p.marlin_block_k}) partitions={spec.fixed_partitions} "
                f"expected={float(expected):.6g} "
                f"range=({float(out.min()):.6g},{float(out.max()):.6g})"
            )
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                for _ in range(p.marlin_calibration_replays):
                    op.launch()
            return _time_graph(graph, samples) / p.marlin_calibration_replays
        finally:
            graph.reset()

    saturation = max(
        1, math.ceil(model.sms / (experts * (n // bn) * max(1, spec.fixed_partitions)))
    )
    # Effective metadata bytes may be fractional per pipeline quantum;
    # calibration populations must still count whole experts.
    cache_groups = max(1, min(experts, int(hw.L2_cache_size // model.weight_bytes)))
    anchors = tuple(
        dict.fromkeys(
            (
                (1, 1),
                (min(2, experts), 1),
                (max(1, cache_groups // 2), 1),
                (cache_groups, 1),
                (experts, 1),
                (experts, (saturation - 1) * bm + 1),
                (experts, saturation * bm + 1),
                (experts, (saturation + 1) * bm + 1),
            )
        )
    )
    if spec.dense_linear or spec.fixed_partitions:
        # Include both ends of a row tile. Sampling only q*BM+1 makes row
        # traffic affine in CTA waves, so the two coefficients are unidentifiable
        # and arbitrary fits fail at partially occupied tiles.
        anchors = tuple(
            dict.fromkeys(
                (*anchors, (1, bm), (1, 4 * saturation * bm + 1), (1, 8 * saturation * bm + 1))
            )
        )
    holdout = tuple(
        dict.fromkeys(
            (
                (min(3, experts), 2),
                (max(1, experts // 2), 1),
                (experts, bm // 2),
                (experts, saturation * bm + bm // 2),
            )
        )
    )
    if spec.dense_linear or spec.fixed_partitions:
        holdout = tuple(dict.fromkeys((*holdout, (1, 3 * saturation * bm + bm // 2))))
    times = [measure(a, r) for a, r in anchors]
    # Fit usable cache capacity: associativity and competing metadata mean
    # the complete advertised L2 is not necessarily available to this layout.
    # The bounded CPU fit uses only calibration points, never held-out M.
    fitted = []
    fractions = (
        (1.0,)
        if experts == 1
        else tuple(1 - i * (1 - p.marlin_cache_fraction_min_percent / 100) / 8 for i in range(9))
    )
    for fraction in fractions:
        candidate = replace(model, l2_bytes=int(hw.L2_cache_size * fraction))
        features = [candidate.features(a, a * r, a * math.ceil(r / bm)) for a, r in anchors]
        candidate = replace(candidate, coefficients=NativeCost.fit(features, times))
        residual = max(
            abs(candidate.predict(a, a * r, a * math.ceil(r / bm)) / t - 1)
            for (a, r), t in zip(anchors, times, strict=True)
        )
        fitted.append((residual, candidate))
    model = min(fitted, key=lambda item: item[0])[1]
    errors = [
        abs(model.predict(a, a * r, a * math.ceil(r / bm)) / measure(a, r) - 1) for a, r in holdout
    ]
    errors += [
        abs(model.predict(a, a * r, a * math.ceil(r / bm)) / t - 1)
        for (a, r), t in zip(anchors, times, strict=True)
    ]
    model = replace(model, error=max(errors), samples=tuple(
        (a, r, measure(a, r), held_out)
        for held_out, population in ((False, anchors), (True, holdout))
        for a, r in population
    ))
    _write_calibration(cache_path, model)
    return model


def expected_row_tiles(mean: float, block_m: int) -> tuple[float, float]:
    """Poisson occupancy prior; does not read routes or change expert ownership."""
    if mean <= 0:
        return 0.0, 0.0
    if mean > 500:
        return 1.0, mean / block_m + (block_m - 1) / (2 * block_m)
    probability, tiles = math.exp(-mean), 0.0
    for count in range(1, math.ceil(mean + 10 * math.sqrt(mean + 1) + block_m)):
        probability *= mean / count
        tiles += probability * math.ceil(count / block_m)
    return -math.expm1(-mean), tiles


def plan_dense_partitions(
    params: MarlinParameters,
    *,
    operators,
    source: Callable[[int], str],
    spec: ProbeFormat,
    device: torch.device,
    n: int,
    k: int,
    group: int,
    partition_group: int,
    max_partitions: int = 0,
    experts: int = 1,
    routing_top_k: int = 1,
) -> int:
    """Seal one geometry/resource-derived reduction plan independently of M.

    All divisors preserve the caller's decoding granularity and pipeline tiles. Resource
    crossings bracket a bounded calibration set; there is no default numeric
    partition cap. The existing measured model includes decoder/MMA issue work,
    cache/weight traffic and partial-write/merge traffic. Minimax relative cost
    across sparse, complete-tile and SM-covering rows avoids a C1-only choice.
    """
    with operator_strategy_scope(
        operators.config, device=operators.device_identity, model=operators.model_identity
    ):
        samples = launch_parameters().calibration_samples
    quantum = math.lcm(partition_group, params.marlin_block_k)
    if (
        min(n, k, group, partition_group) <= 0
        or k % quantum
        or n % params.marlin_block_n
        or max_partitions < 0 or min(experts, routing_top_k) <= 0
    ):
        raise ValueError("dense partition plan requires aligned geometry and a nonnegative cap")
    units = k // quantum
    active = 1 if spec.dense_linear else min(experts, routing_top_k)
    tiles = active * (n // params.marlin_block_n)
    hw = torch.cuda.get_device_properties(device)
    legal = [
        p
        for p in range(1, units + 1)
        if units % p == 0 and (not max_partitions or p <= max_partitions)
    ]
    shared = shared_bytes(
        params,
        bits=spec.storage_bits or spec.bits,
        group=group,
        metadata_planes=len(spec.group_metadata),
        row_metadata=not spec.dense_linear,
        zero_points=spec.zero_word is not None,
    )
    # This is a resource upper bound for candidate enumeration, not an occupancy
    # claim. Actual compiled registers, scheduling and bandwidth enter calibration.
    resident = min(
        hw.max_threads_per_multi_processor // (params.marlin_warps * 32),
        hw.shared_memory_per_multiprocessor // max(1, shared),
    )
    if resident < 1:
        raise ValueError("dense partition plan has no legal resource capacity")
    candidates = {1}
    for supply in (hw.multi_processor_count, hw.multi_processor_count * resident):
        target = supply / tiles
        candidates.add(max((p for p in legal if p <= target), default=legal[0]))
        candidates.add(min((p for p in legal if p >= target), default=legal[-1]))
    candidates = {p for p in candidates if p in legal}
    # Reuse the common bounded calibration budget. Rank excess crossings by
    # their distance to one SM wave, never by a device/model/row-count table.
    ordered = [
        1,
        *sorted(
            candidates - {1}, key=lambda p: (abs(math.log(tiles * p / hw.multi_processor_count)), p)
        ),
    ]
    ordered = ordered[: params.marlin_max_candidates]
    if len(ordered) == 1:
        return 1
    return _select_dense_partition_model(
        tuple((p, source(p)) for p in ordered), params, spec, device, n, k, group, samples,
        experts, routing_top_k,
    )


@cache
def _select_dense_partition_model(
    points, params, spec, device, n, k, group, samples, experts=1, routing_top_k=1,
):
    path = _plan_path("partitions", params, spec, device, points,
                      (n, k, group, samples, experts, routing_top_k))
    cached = _read_plan(path, len(points))
    if cached is not None:
        return points[cached][0]
    with local_plan_lock(path):
        cached = _read_plan(path, len(points))
        if cached is not None:
            return points[cached][0]
        from cubic_serve.ops.cuda.startup import calibration_allowed
        if not calibration_allowed():
            hw = torch.cuda.get_device_properties(device)
            active = 1 if spec.dense_linear else min(experts, routing_top_k)
            predicted = fixed_k_partitions(n=n * active, k=k, group=group,
                block_n=params.marlin_block_n, block_k=params.marlin_block_k,
                sms=hw.multi_processor_count, supply_factor=1.,
                max_partitions=max(point[0] for point in points))
            chosen = min(range(len(points)), key=lambda i: abs(points[i][0] - predicted))
            _write_plan(path, chosen, basis="retained_default", origin="production_default")
            return points[chosen][0]
        from cubic_serve.ops.cuda.startup import require_calibration_permission
        require_calibration_permission("Marlin partition plan")
        # A fit is immutable for a source/resource ABI. Graph buckets cannot change
        # their reduction plan because another calibration evicted an intermediate.
        hw = torch.cuda.get_device_properties(device)
        tiles = n // params.marlin_block_n
        models = {}
        for partitions, source in points:
            model = calibrate(
                source,
                params,
                replace(spec, fixed_partitions=(
                    partitions if spec.dense_linear or partitions > 1 else 0
                )),
                device,
                n,
                k,
                experts,
                group,
                samples,
            )
            if 0 <= model.error < 1 and model.error <= params.marlin_max_error_percent / 100:
                models[partitions] = model
        if not models:
            default = next((i for i, point in enumerate(points) if point[0] == 1), None)
            if default is not None:
                _write_plan(path, default, basis="retained_default")
            return 1
        required_tiles = max(1, math.ceil(hw.multi_processor_count / tiles))
        if spec.dense_linear:
            covering_rows = required_tiles * params.marlin_block_m
        elif required_tiles <= experts:
            # Each routed token contributes top_k expert rows. While experts remain
            # unsaturated, each occupied expert already contributes one row tile.
            covering_rows = math.ceil(required_tiles / routing_top_k)
        else:
            # Balanced occupancy has exhausted the expert axis. Supply the first
            # token beyond the preceding full per-expert tile wave.
            preceding_tiles = math.ceil(required_tiles / experts) - 1
            covering_rows = preceding_tiles * params.marlin_block_m * experts // routing_top_k + 1
        rows = sorted({1, params.marlin_block_m, covering_rows})
        def workload(m: int) -> tuple[int, int, int]:
            if spec.dense_linear:
                return 1, m, math.ceil(m / params.marlin_block_m)
            routed = m * routing_top_k
            active = min(experts, routed)
            return active, routed, active * math.ceil(routed / active / params.marlin_block_m)

        costs = {partitions: [model.predict(*workload(m)) for m in rows]
                 for partitions, model in models.items()}
        lower = [
            min(prediction_interval(costs[p][i], model.error)[0] for p, model in models.items())
            for i in range(len(rows))
        ]
        selected = min(
            models,
            key=lambda p: (
                max(
                    prediction_interval(value, models[p].error)[1] / floor
                    for value, floor in zip(costs[p], lower, strict=True)
                ),
                p,
            ),
        )
        _write_plan(path, next(i for i, point in enumerate(points) if point[0] == selected))
        calibration_k_plan(
            logging.getLogger(__name__), spec.dense_linear,
            n,
            k,
            spec.storage_bits or spec.bits,
            hw.multi_processor_count,
            selected,
            rows,
            costs, {p: m.error * 100 for p, m in models.items()},
        )
        return selected


def _memory_reuse_domain(spec: ProbeFormat) -> bool:
    # Full-byte symmetric weights have no low-bit unpack/decode arithmetic.
    # Fixed partitions and extra metadata need their own measured cost model.
    return (spec.dense_linear and not spec.fixed_partitions
            and spec.activation_dtype == torch.int8 and spec.bits == 8
            and (spec.storage_bits or spec.bits) == 8
            and spec.output_dtype in (torch.bfloat16, torch.float16)
            and spec.zero_word is None and not spec.group_metadata)


def memory_reuse_default(
    params, points: list[tuple[MarlinParameters, KernelResources]], *,
    m, n, k, weight_bytes, sms,
):
    """Conservative bandwidth-work ordering, not a fitted GEMM latency model.

    L2 hits still transport packed operands to each row tile; cache capacity
    alone cannot make repeated weight reads free. Only row reuse changes:
    retain N/K tiles and warp geometry, no additional
    row padding or spills, no loss of resident warps or pipeline row storage,
    and enough independent work for a complete resident wave. Resources are
    supplied by the actual compiled kernel, never an architecture/SKU table.
    """
    if m < 2 * params.marlin_block_m:
        return params
    baseline = next((r for p, r in points if p == params), None)
    if baseline is None or baseline.resident < 1:
        return params

    def work(p, r):
        row_tiles = math.ceil(m / p.marlin_block_m)
        padded = row_tiles * p.marlin_block_m
        tiles = row_tiles * (n // p.marlin_block_n)
        supply = tiles / (sms * r.resident)
        traffic = (weight_bytes * row_tiles
                   + (n // p.marlin_block_n) * padded * k + padded * n * 2)
        return (traffic * math.ceil(supply) / supply, padded, tiles,
                r.resident * p.marlin_block_m * p.marlin_stages)

    best, base = params, work(params, baseline)
    best_score = base[0]
    for p, resource in points:
        if ((p.marlin_block_n, p.marlin_block_k, p.marlin_warps)
                != (params.marlin_block_n, params.marlin_block_k, params.marlin_warps)
                or resource.resident < 1 or resource.local_bytes > baseline.local_bytes
                or resource.resident * p.marlin_warps < baseline.resident * params.marlin_warps
                or m < p.marlin_block_m):
            continue
        score, padded, tiles, pipeline = work(p, resource)
        if (padded > base[1] or tiles < sms * resource.resident or pipeline < base[3]):
            continue
        if score < best_score * (1 - params.marlin_min_gain_percent / 100):
            best, best_score = p, score
    return best


def _native_memory_reuse_default(params, portfolio, *, spec, source, device, n, k, group, rows):
    if not _memory_reuse_domain(spec) or rows < 2 * params.marlin_block_m:
        return params
    hw = torch.cuda.get_device_properties(device)
    weight_bytes = (n * k + n * (k // group)
                    * torch.empty((), dtype=spec.scale_dtype).element_size())
    shared = shared_bytes(params, bits=8, group=group,
                          row_metadata=False, zero_points=False)
    module, entry = compile_module(source(params), device)
    baseline = kernel_resources(module, entry, params, shared)
    if baseline.resident < 1:
        return params
    points = [(params, baseline)]
    padded = math.ceil(rows / params.marlin_block_m) * params.marlin_block_m
    for p in portfolio:
        if ((p.marlin_block_n, p.marlin_block_k, p.marlin_warps)
                != (params.marlin_block_n, params.marlin_block_k, params.marlin_warps)
                or p.marlin_block_m <= params.marlin_block_m
                or math.ceil(rows / p.marlin_block_m) * p.marlin_block_m > padded):
            continue
        shared = shared_bytes(p, bits=8, group=group,
                              row_metadata=False, zero_points=False)
        # A candidate cannot retain baseline residency if even its dynamic
        # storage exceeds the per-SM budget. Avoid loading such a binary.
        if shared * baseline.resident > hw.shared_memory_per_multiprocessor:
            continue
        module, entry = compile_module(source(p), device)
        points.append((p, kernel_resources(module, entry, p, shared)))
    return memory_reuse_default(params, points, m=rows, n=n, k=k,
                                weight_bytes=weight_bytes, sms=hw.multi_processor_count)


def select(
    params,
    *,
    operators,
    family: str,
    scope: dict[str, int],
    source: Callable,
    spec: ProbeFormat,
    device: torch.device,
    n: int,
    k: int,
    experts: int,
    group: int,
    rows: float,
    routed: bool,
    alternates: tuple[MarlinParameters, ...] = (),
):
    """Score predicted work at requested M; never time that M to choose its tile."""
    if not params.marlin_auto_geometry:
        return params
    with operator_strategy_scope(
        operators.config, device=operators.device_identity, model=operators.model_identity
    ):
        locked = parameter_overrides(family, **scope)
        limits = launch_parameters()
    candidates = [params, *alternates]
    candidates += [
        replace(params, marlin_block_m=m)
        for m in (8, 16, 32, 64)
        if m >= spec.min_block_m
        and (
            m != params.marlin_block_m
            if spec.fixed_partitions
            else m == 2 * params.marlin_block_m
            or (m == 8 and m < params.marlin_block_m)
        )
    ]
    if params.marlin_warps == 4:
        candidates.append(replace(params, marlin_warps=8, marlin_block_n=params.marlin_block_n * 2))
    candidates += [replace(params, marlin_stages=s) for s in (2, 3, 4) if s != params.marlin_stages]
    hw = torch.cuda.get_device_properties(device)
    candidates = [
        p
        for p in candidates
        if pipeline_compatible(
            p, group=p.marlin_block_k if spec.column_scale else group,
            single_subtile=spec.single_subtile,
        )
        and p.marlin_block_m <= limits.max_block_m
        and p.marlin_warps <= limits.max_warps
        and n % p.marlin_block_n == 0
        and p.marlin_stages <= limits.max_stages
        and not any(
            getattr(p, name) != getattr(params, name)
            for name in locked
            if name.startswith("marlin_")
        )
        and shared_bytes(
            p,
            bits=spec.storage_bits or spec.bits,
            group=group,
            metadata_planes=len(spec.group_metadata),
        )
        <= hw.shared_memory_per_block_optin
    ]
    portfolio = candidates[: params.marlin_max_candidates]
    path = _plan_path(
        "geometry.row-reuse.v2" if _memory_reuse_domain(spec) else "geometry",
        params, spec, device,
        tuple((point, source(point)) for point in portfolio),
        (family, scope, n, k, experts, group, rows, routed, tuple(sorted(locked)), limits),
    )
    cached = _read_plan(path, len(portfolio), _memory_reuse_domain(spec))
    if cached is not None:
        return portfolio[cached]
    with local_plan_lock(path):
        cached = _read_plan(path, len(portfolio), _memory_reuse_domain(spec))
        if cached is not None:
            return portfolio[cached]
        from cubic_serve.ops.cuda.startup import calibration_allowed
        if not calibration_allowed():
            selected = _native_memory_reuse_default(
                params, portfolio, spec=spec, source=source, device=device,
                n=n, k=k, group=group, rows=rows,
            )
            if selected in portfolio:
                _write_plan(path, portfolio.index(selected), basis="retained_default",
                            origin="resource_default" if _memory_reuse_domain(spec)
                            else "production_default")
            return selected
        from cubic_serve.ops.cuda.startup import require_calibration_permission
        require_calibration_permission("Marlin geometry plan")
        scored = []
        for p in candidates[: params.marlin_max_candidates]:
            cost = calibrate(
                source(p), p, spec, device, n, k, experts, group, limits.calibration_samples
            )
            if (cost.error > params.marlin_max_error_percent / 100
                    and not getattr(cost, "samples", ())):
                continue
            estimates = []
            workloads = []
            # Dense launches execute their complete bound physical
            # M. Smaller graph capacities select their own tile; comparing a tile
            # against M/2 here wrongly prevents a profitable capacity transition.
            # Routed launches instead consume variable active rows within the bound.
            populations = (
                (rows,)
                if spec.dense_linear or spec.fixed_partitions
                else (rows, rows / 2 if routed else max(1.0, rows / 2))
            )
            for population in populations:
                if routed:
                    activity, tiles = expected_row_tiles(population, p.marlin_block_m)
                    active, logical_rows = experts * activity, experts * population
                    row_tiles = experts * tiles
                else:
                    active, logical_rows = 1.0, population
                    row_tiles = math.ceil(population / p.marlin_block_m)
                workloads.append((active, logical_rows, row_tiles))
                estimates.append(cost.predict(active, logical_rows, row_tiles))
            scored.append((estimates, cost.error, p, cost, workloads))
        baseline = next((item for item in scored if item[2] == params), None)
        if baseline is None or baseline[1] > params.marlin_max_error_percent / 100:
            if params in portfolio:
                _write_plan(path, portfolio.index(params), basis="retained_default")
            return params
        qualified = []
        for candidate in scored:
            if candidate[2] == params:
                continue
            if candidate[1] <= params.marlin_max_error_percent / 100:
                intervals = tuple(
                    (prediction_interval(value, candidate[1]),
                     prediction_interval(original, baseline[1]))
                    for value, original in zip(candidate[0], baseline[0], strict=True)
                )
                if all(predicted_gain(value, original, params.marlin_min_gain_percent)
                       for value, original in intervals):
                    qualified.append((max(value[1] / original[0] for value, original in intervals),
                                      candidate[2]))
            # A relative model is a separate validated witness. It uses the same
            # full corpus, includes independent holdouts, keeps the configured
            # maximum modelling error and requires the same minimum proven gain.
            if isinstance(candidate[3], NativeCost) and isinstance(baseline[3], NativeCost):
                relative = tuple(candidate[3].relative_interval(baseline[3], *work)
                                 for work in candidate[4])
                witnessed = tuple(item for item in relative if item is not None)
                if relative and len(witnessed) == len(relative) and all(
                    item[1] <= params.marlin_max_error_percent / 100
                    and item[0][1] < 1 - params.marlin_min_gain_percent / 100
                    for item in witnessed
                ):
                    qualified.append((max(item[0][1] for item in witnessed), candidate[2]))
        if qualified:
            best = min(qualified, key=lambda item: item[0])[1]
            _write_plan(path, portfolio.index(best))
            return best
        _write_plan(path, portfolio.index(params))
        return params
