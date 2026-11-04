"""Cold, format-independent selection of the packed row contraction.

Formats supply encoded fixtures and decoder arithmetic. Models do not register
names or eligible layers. The fitted rule uses tensor geometry and SM supply;
it keeps a single K partition plan across the row population. Empirical bounds
include neighboring N and held-out M for each activation-storage domain; they
never qualify a serving graph.
"""
from __future__ import annotations

import logging
import math
import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any

import torch

from cubic_serve.ops.cuda.launch_cost import (
    _launch_calibration_path,
    _read_startup_calibration,
    _write_startup_calibration,
    local_plan_lock,
    fit_nonnegative_cost,
    prediction_interval,
)
from cubic_serve.ops.strategies import operator_strategy_scope, parameter_overrides
from cubic_serve.quantization.common.small_rows import SmallRowsWeight, bind

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Geometry:
    bits: int
    group: int
    raw_bytes: int
    elements: int


@dataclass(frozen=True)
class Cost:
    native: tuple[float, ...]
    baseline: tuple[float, ...]
    native_error: float
    baseline_error: float
    partitions: int
    warps: int
    geometry: Geometry
    native_domain_errors: tuple[tuple[str, float], ...] = ()


_costs: dict[tuple[Any, ...], Cost | None] = {}


def _recipe(raw):
    recipe = None if raw is None else raw.get("recipe")
    if (not isinstance(recipe, (list, tuple)) or len(recipe) != 3
            or recipe[0] not in ("baseline", "simt", "mma")
            or any(type(value) is not int for value in recipe[1:])
            or recipe[0] == "simt" and (recipe[1] not in (4, 8, 16) or recipe[2] not in (1, 2, 4))
            or recipe[0] != "simt" and tuple(recipe[1:]) != (0, 0)):
        return None
    return tuple(recipe)


@lru_cache(maxsize=32)
def _recipe_kernels(linear):
    """The competing native bodies and the format's decoder, not Python glue."""
    from cubic_serve.ops.cuda.cache_identity import module_kernels, native_headers
    from cubic_serve.quantization.common import small_rows, small_mma
    from cubic_serve.quantization.common.marlin import compiler, repack
    from types import ModuleType

    adapter = getattr(linear, "row_contraction", None)
    module = adapter if isinstance(adapter, ModuleType) else inspect.getmodule(
        getattr(adapter, "prepare", None))
    root = Path(compiler.__file__).with_name("native")
    headers = native_headers(root)
    return (module_kernels(small_rows), module_kernels(small_mma),
            () if module is None else module_kernels(module),
            headers, module_kernels(repack))


def _read_cost(path):
    raw = _read_startup_calibration(path, "cubic.packed-row-cost.v1")
    if raw is None:
        return None
    try:
        value = raw["cost"]
        geometry = Geometry(**value["geometry"])
        cost = Cost(tuple(value["native"]), tuple(value["baseline"]),
                    value["native_error"], value["baseline_error"],
                    value["partitions"], value["warps"], geometry,
                    tuple(tuple(item) for item in value["native_domain_errors"]))
        numbers = (*cost.native, *cost.baseline, cost.native_error, cost.baseline_error,
                   *(error for _, error in cost.native_domain_errors))
        if (len(cost.native) == 5 and len(cost.baseline) == 3
                and all(type(x) in (float, int) and math.isfinite(x) and x >= 0 for x in numbers)
                and type(cost.partitions) is int and cost.partitions in (1, 2, 4)
                and type(cost.warps) is int and cost.warps in (4, 8, 16)
                and all(type(v) is int and v > 0 for v in asdict(geometry).values())
                and len({name for name, _ in cost.native_domain_errors}) == len(cost.native_domain_errors)
                and all(name in ("full", "tiled", "external") for name, _ in cost.native_domain_errors)):
            return cost
    except (KeyError, TypeError, ValueError):
        pass
    return None

# A numerical observer reconstructs a single logical row using the candidate's
# sealed arithmetic choice. It must not price that row again as a C1 workload.
# The scope exists only during startup binding; replay never reads it.
_audit_plan: ContextVar[dict[Any, tuple[Any, ...]] | None] = ContextVar(
    "packed_row_audit_plan", default=None,
)


def _plan_key(operators, x, weights, params):
    first = weights[0]
    return (operators.config.compute_mode, x.dtype, first.encoding, first.shape,
            first.partitions, len(weights), params)


def row_arithmetic_plan(operators, capacity: int) -> dict[Any, tuple[Any, ...]]:
    if type(capacity) is not int or capacity < 1:
        raise ValueError("row arithmetic plan requires a positive physical capacity")
    return {(key, Fraction(rows, capacity)): recipe
            for (key, rows), recipe in getattr(operators, "_packed_row_plans", {}).items()}


@contextmanager
def row_arithmetic_scope(plan):
    token = _audit_plan.set(dict(plan))
    try:
        yield
    finally:
        _audit_plan.reset(token)


def audited_row_recipe(key: Any, rows: int) -> tuple[Any, ...] | None:
    """Read a format's sealed reduction recipe during an explicit audit only."""
    plan = _audit_plan.get()
    if plan is None:
        return None
    identity = key, Fraction(rows, 1)
    if identity not in plan:
        raise ValueError("candidate row arithmetic occurrence is absent from its audit plan")
    return plan[identity]


def record_row_recipe(operators: Any, key: Any, rows: int,
                      recipe: tuple[Any, ...]) -> None:
    """Retain format-owned arithmetic in the existing cross-M plan inventory."""
    plans = getattr(operators, "_packed_row_plans", None)
    if plans is None:
        plans = operators._packed_row_plans = {}
    occurrence = key, rows
    if occurrence in plans and plans[occurrence] != recipe:
        raise RuntimeError("startup row arithmetic choice changed within one owner")
    plans[occurrence] = recipe


def _tile(k: int, group: int) -> int:
    quantum = math.lcm(128, group)
    return max((t for t in range(quantum, min(1024, group * 32) + 1, quantum)
                if k % t == 0), default=quantum)


def _staging(g: Geometry, tile: int, rows: int, members: int) -> int:
    groups = tile // g.group
    return members * (2 * (tile * g.bits // 8 + 16 * math.ceil(groups * g.raw_bytes / 16))
                      + groups * 8 + rows * 4)


def _features(n: int, k: int, m: int, members: int, g: Geometry,
              sms: int, partitions: int, warps: int) -> tuple[float, ...]:
    tile = _tile(k, g.group)
    staging = warps * _staging(g, tile, m, members)
    full = 2 * k * m + staging <= 48 * 1024
    tiled = not full and partitions == 1 and 4 * tile * m + staging <= 48 * 1024
    work = max(1.0, math.ceil(n / (warps // partitions)) / sms) * math.ceil(k / tile / partitions)
    return (1.0, work, work * m, work * float(tiled), work * m * float(not full and not tiled))


def _reuse_domain(features: tuple[float, ...]) -> str:
    return "tiled" if features[3] else "external" if features[4] else "full"


def _baseline_features(n: int, k: int, m: int, sms: int) -> tuple[float, ...]:
    return (1.0, math.ceil(n / 64) / sms * (k / 128), m * n / 1e6)


def _predict(coefficients: tuple[float, ...], features: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(coefficients, features, strict=True))


def _measure(binding: Any, outputs: tuple[torch.Tensor, ...]) -> float:
    binding.launch()
    torch.cuda.synchronize(outputs[0].device)
    if any(not bool((y == 0).all()) for y in outputs):
        raise RuntimeError("packed row cost fixture violated its zero-weight contract")
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            for _ in range(8):
                binding.launch()
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))  # type: ignore[no-untyped-call]
                  for _ in range(7)]
        for start, end in events:
            start.record()
            graph.replay()
            end.record()
        torch.cuda.synchronize(outputs[0].device)
        return float(median(start.elapsed_time(end) * 1000 / 8 for start, end in events))
    finally:
        graph.reset()


def _fit(operators: Any, x: torch.Tensor, weights: tuple[Any, ...], params: Any,
         adapter: Any, baseline: Any) -> Cost | None:
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators

    first = weights[0]
    n, k = first.shape
    group = first.encoding.group_in
    tile = _tile(k, group)
    choices = tuple(p for p in (1, 2, 4) if k // tile >= p)
    n1 = max(64, n // 128 * 64)
    n2 = max(64, 3 * n // 256 * 64)
    # Preserve every format-owned output partition in neighboring fixtures.
    if n1 < 16 * len(first.partitions or (n,)):
        return None
    # Storage transitions can make the work/row/domain columns dependent at
    # M1/M2/M4. One neighboring-N M3 observation distinguishes those terms;
    # original-N M3 and the other neighboring N remain independent holdouts.
    train = tuple((nn, m) for nn in sorted({n1, n}) for m in (1, 2, 3, 4)
                  if m != 3 or nn != n)
    holdout = tuple(dict.fromkeys(((n, 3), (n2, 1), (n2, 3))))
    names: tuple[str | int, ...] = ("baseline", *choices)
    times: dict[str | int, list[float]] = {name: [] for name in names}
    own = PackedLinearOperators(config=operators.config, device=operators.device_identity,
                               model=operators.model_identity)
    probes: dict[int, tuple[tuple[Any, ...], tuple[SmallRowsWeight, ...]]] = {}
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    geometry: Geometry | None = None
    warps = 0
    try:
        for nn, m in (*train, *holdout):
            if nn not in probes:
                matrices = tuple(adapter.probe(w, nn, x.device) for w in weights)
                prepared = tuple(adapter.prepare(own, w, x.device) for w in matrices)
                probes[nn] = matrices, prepared
            matrices, prepared = probes[nn]
            if geometry is None:
                descriptor = prepared[0]
                geometry = Geometry(descriptor.bits, descriptor.group,
                                    int(dict(descriptor.constants)["FORMAT_RAW_BYTES"]),
                                    descriptor.elements)
                warps = next((w for w in (16, 8, 4)
                              if w * _staging(geometry, tile, 4, len(weights)) <= 48 * 1024), 0)
                if not warps:
                    return None
            inp = torch.full((m, k), 1 / 128, dtype=x.dtype, device=x.device)
            targets = tuple(torch.empty((m, nn), dtype=x.dtype, device=x.device) for _ in weights)
            old = baseline(own, inp, matrices, targets, params)
            if old is None:
                return None
            for name in times:
                candidate = old if name == "baseline" else bind(
                    inp, prepared, targets, warps=warps, partitions=int(name))
                times[name].append(_measure(candidate, targets))
        assert geometry is not None
        fits: dict[str | int, tuple[tuple[float, ...], float]] = {}
        for name, values in times.items():
            def features(nn: int, m: int, name: str | int = name) -> tuple[float, ...]:
                assert geometry is not None
                return (_baseline_features(nn, k, m, sms) if name == "baseline" else
                        _features(nn, k, m, len(weights), geometry, sms, int(name), warps))
            coefficients = fit_nonnegative_cost([features(nn, m) for nn, m in train],
                                                values[:len(train)])
            error = max(abs(_predict(coefficients, features(nn, m)) / value - 1)
                        for (nn, m), value in zip((*train, *holdout), values, strict=True))
            fits[name] = coefficients, error
        predictions = {p: tuple(_predict(fits[p][0], _features(
            n, k, m, len(weights), geometry, sms, p, warps)) for m in (1, 2, 4)) for p in choices}
        partition = min(choices, key=lambda p: (max(
            predictions[p][i] / min(predictions[q][i] for q in choices) for i in range(3)), p))
        native, ne = fits[partition]
        old, oe = fits["baseline"]
        domain_errors: dict[str, float] = {}
        validation_domains: set[str] = set()
        for (nn, m), value in zip((*train, *holdout), times[partition], strict=True):
            point_features = _features(nn, k, m, len(weights), geometry, sms, partition, warps)
            domain = _reuse_domain(point_features)
            error = abs(_predict(native, point_features) / value - 1)
            domain_errors[domain] = max(domain_errors.get(domain, 0.0), error)
            if (nn, m) in holdout and (nn, m) not in train:
                validation_domains.add(domain)
        validated_errors = tuple(sorted((domain, error) for domain, error in domain_errors.items()
                                         if domain in validation_domains))
        _LOGGER.info("packed row cost n=%d k=%d members=%d bits=%d group=%d "
                     "partitions=%d warps=%d native_error=%.6f baseline_error=%.6f "
                     "validated_storage_errors=%s",
                     n, k, len(weights), geometry.bits, geometry.group, partition, warps,
                     ne, oe, validated_errors)
        return Cost(native, old, ne, oe, partition, warps, geometry, validated_errors)
    finally:
        probes.clear()
        own.prepared_weights.clear()
        own.materialized_weights.clear()


def _choose_simt(operators: Any, x: torch.Tensor, weights: tuple[Any, ...],
           outputs: tuple[torch.Tensor, ...], params: Any, linear: Any) -> Any:
    """Return a startup binding or decline; never changes a requested compute mode."""
    if (operators.config.compute_mode != "wna16" or not getattr(params, "marlin_enabled", False)
            or x.dtype not in (torch.bfloat16, torch.float16) or not 1 <= x.shape[0] <= 4
            or x.data_ptr() % 16 or len(weights) not in (1, 2)):
        return None
    adapter = getattr(linear, "row_contraction", None)
    if adapter is None:
        return None
    first = weights[0]
    n, k = first.shape
    enc = first.encoding
    group = enc.group_in
    if (enc.group_out != 1 or group < 32 or k % 128 or k % group or n < 64
            or k % _tile(k, group) or any(w.shape != first.shape or w.encoding != enc
                   or w.partitions != first.partitions for w in weights)
            or any(y.dtype != x.dtype for y in outputs)
            or not all(adapter.supported(w) for w in weights)):
        return None
    with operator_strategy_scope(operators.config, device=operators.device_identity,
                                 model=operators.model_identity):
        from cubic_serve.quantization.registry import get_extension
        family = get_extension(enc.format).family("linear")
        if any(parameter_overrides(family, n=w.shape[0], k=k) for w in weights):
            return None
    def baseline(own: Any, inp: torch.Tensor, matrices: tuple[Any, ...],
                 targets: tuple[torch.Tensor, ...], parameters: Any) -> Any:
        return (linear.bind_native(own, inp, matrices[0], targets[0], parameters)
                if len(matrices) == 1 else
                linear.bind_grouped(own, inp, matrices, targets, parameters))
    key = (x.device, x.dtype, enc, first.shape, first.partitions, len(weights), params)
    if key not in _costs:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("packed row selection must precede graph capture")
        from cubic_serve.ops.cuda.startup import startup_cuda_lock
        with startup_cuda_lock(x.device):
            # External adapters keep live calibration unless their own source
            # binding is declared. Builtin format/fixture sources are in the key.
            builtin = all(getattr(call, "__module__", "").startswith("cubic_serve.")
                          for call in (adapter.prepare, adapter.probe))
            path = (_launch_calibration_path(("packed-row", key), (), x.device,
                                            kernels=_recipe_kernels(linear))
                    if builtin else None)
            cost = _read_cost(path)
            if cost is None:
                cost = _fit(operators, x, weights, params, adapter, baseline)
                if cost is not None:
                    _write_startup_calibration(path, "cubic.packed-row-cost.v1", {"cost": asdict(cost)})
            _costs[key] = cost
    cost = _costs[key]
    if cost is None:
        return None
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    features = _features(n, k, x.shape[0], len(weights), cost.geometry,
                         sms, cost.partitions, cost.warps)
    # A different input-storage boundary cannot invalidate a validated domain,
    # nor can a domain without an independent holdout authorize this binding.
    native_error = dict(cost.native_domain_errors).get(_reuse_domain(features), math.inf)
    if max(native_error, cost.baseline_error) > .15:
        return None
    native = _predict(cost.native, features)
    old = _predict(cost.baseline, _baseline_features(n, k, x.shape[0], sms))
    if prediction_interval(native, native_error)[1] * 1.02 >= prediction_interval(
            old, cost.baseline_error)[0]:
        return None
    return bind(x, tuple(adapter.prepare(operators, w, x.device) for w in weights), outputs,
                warps=cost.warps, partitions=cost.partitions)


def _choose_and_record(operators: Any, x: torch.Tensor, weights: tuple[Any, ...],
           outputs: tuple[torch.Tensor, ...], params: Any, linear: Any) -> Any:
    key = _plan_key(operators, x, weights, params)
    audit_plan = _audit_plan.get()
    if audit_plan is not None:
        # The observer represents one logical token. Relative row extent also
        # distinguishes projections over its streams or other expanded fields.
        identity = key, Fraction(x.shape[0], 1)
        if identity not in audit_plan:
            raise ValueError("candidate row arithmetic occurrence is absent from its audit plan")
        kind, warps, partitions = audit_plan[identity]
        if kind == "baseline":
            return None
        if kind in ("simt", "mma") and operators.config.compute_mode != "wna16":
            raise ValueError("selected weight-only row arithmetic requires explicit WNA16")
        adapter = getattr(linear, "row_contraction", None)
        if adapter is None:
            raise ValueError("candidate row arithmetic adapter is unavailable")
        if kind == "simt":
            return bind(x, tuple(adapter.prepare(operators, w, x.device) for w in weights),
                        outputs, warps=warps, partitions=partitions)
        if kind == "mma":
            from cubic_serve.quantization.common.small_mma import bind as bind_mma
            prepared = tuple(adapter.prepare_mma(operators, w, x.device) for w in weights)
            if any(w is None for w in prepared):
                raise ValueError("candidate MMA arithmetic adapter is unavailable")
            return bind_mma(x, prepared, outputs)
        raise ValueError("candidate row arithmetic recipe is unknown")
    result = _choose_simt(operators, x, weights, outputs, params, linear)
    if result is not None:
        first = weights[0]
        cost = _costs[x.device, x.dtype, first.encoding, first.shape,
                      first.partitions, len(weights), params]
        assert cost is not None
        recipe = "simt", cost.warps, cost.partitions
    else:
        from cubic_serve.quantization.common.small_mma_cost import choose as choose_mma
        result = choose_mma(operators, x, weights, outputs, params, linear)
        recipe = ("mma", 0, 0) if result is not None else ("baseline", 0, 0)
    plans = getattr(operators, "_packed_row_plans", None)
    if plans is None:
        plans = operators._packed_row_plans = {}
    occurrence = key, x.shape[0]
    if occurrence in plans and plans[occurrence] != recipe:
        raise RuntimeError("startup row arithmetic choice changed within one owner")
    plans[occurrence] = recipe
    return result


def choose(operators: Any, x: torch.Tensor, weights: tuple[Any, ...],
           outputs: tuple[torch.Tensor, ...], params: Any, linear: Any) -> Any:
    """Read the final row recipe; only explicit tuning may fit candidates."""
    if _audit_plan.get() is not None:
        return _choose_and_record(operators, x, weights, outputs, params, linear)
    from cubic_serve.ops.cuda.startup import calibration_allowed
    from cubic_serve.quantization.common import small_mma

    key = _plan_key(operators, x, weights, params)
    adapter = getattr(linear, "row_contraction", None)
    first = weights[0]
    # Recipe reuse cannot bypass the operand eligibility checks that the cold
    # selectors perform. Ineligible occurrences retain their baseline arithmetic
    # and must not overwrite the recipe of an eligible, aligned occurrence.
    compatible = (adapter is not None and operators.config.compute_mode == "wna16"
        and getattr(params, "marlin_enabled", False)
        and x.dtype in (torch.bfloat16, torch.float16) and 1 <= x.shape[0] <= 4
        and x.data_ptr() % 16 == 0 and len(weights) in (1, 2)
        and all(w.shape == first.shape and w.encoding == first.encoding
                and w.partitions == first.partitions and adapter.supported(w) for w in weights)
        and all(y.dtype == x.dtype for y in outputs))
    occurrence = key, x.shape[0]
    retained = getattr(operators, "_packed_row_plans", {})
    if compatible and occurrence in retained:
        # Other processes may publish a newer portable choice. Once an owner
        # has bound its arithmetic, that external update cannot change it.
        path, raw = None, {"recipe": retained[occurrence]}
    elif compatible:
        images = _recipe_kernels(linear)
        path = _launch_calibration_path(("row-recipe", key, x.shape[0]), (), x.device, kernels=images)
        raw = _read_startup_calibration(path, "cubic.row-recipe.v1")
    else:
        path, raw = None, {"recipe": ("baseline", 0, 0)}
    recipe = _recipe(raw)
    if recipe is None:
        with local_plan_lock(path):
            recipe = _recipe(_read_startup_calibration(path, "cubic.row-recipe.v1"))
            if recipe is None:
                if calibration_allowed():
                    result = _choose_and_record(operators, x, weights, outputs, params, linear)
                    recipe = operators._packed_row_plans[key, x.shape[0]]
                    _write_startup_calibration(path, "cubic.row-recipe.v1", {"recipe": recipe})
                    return result
                recipe = ("baseline", 0, 0)
                _write_startup_calibration(path, "cubic.row-recipe.v1",
                    {"recipe": recipe, "origin": "production_default"})
    recipe = tuple(recipe)
    plans = getattr(operators, "_packed_row_plans", None)
    if plans is None:
        plans = operators._packed_row_plans = {}
    if occurrence in plans and plans[occurrence] != recipe:
        raise RuntimeError("startup row arithmetic choice changed within one owner")
    plans[occurrence] = recipe
    if recipe[0] == "baseline":
        return None
    if adapter is None:
        raise ValueError("selected row recipe has no format adapter")
    if recipe[0] == "simt":
        return bind(x, tuple(adapter.prepare(operators, w, x.device) for w in weights), outputs,
                    warps=recipe[1], partitions=recipe[2])
    prepared = tuple(adapter.prepare_mma(operators, w, x.device) for w in weights)
    if any(value is None for value in prepared):
        raise ValueError("selected MMA recipe has no compatible format binding")
    return small_mma.bind(x, prepared, outputs)
