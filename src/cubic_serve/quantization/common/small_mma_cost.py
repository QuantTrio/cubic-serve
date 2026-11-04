"""Cold geometry-only pricing for the shared transposed-MMA candidate.

This rule qualifies only its measured M1..4 domain. Larger MMA bindings remain
operator diagnostics until independently priced and numerically qualified.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any

import torch

from cubic_serve.ops.cuda.launch_cost import fit_nonnegative_cost, prediction_interval
from cubic_serve.quantization.common.small_mma import bind, shared_bytes, tile_k


@dataclass(frozen=True)
class Cost:
    candidate: tuple[float, ...]
    baseline: tuple[float, ...]
    candidate_error: float
    baseline_error: float


_costs: dict[tuple[Any, ...], Cost | None] = {}
_LOGGER = logging.getLogger(__name__)


def _features(n: int, k: int, m: int, members: int, sms: int, group: int) -> tuple[float, ...]:
    work = math.ceil(n/16)*members/sms*(k/tile_k(k, group))
    return 1.0, work, work*m


def _fit(operators, x, weights, params, adapter, linear):
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
    from cubic_serve.quantization.common.small_rows_cost import _measure, _predict

    first = weights[0]
    n, k = first.shape
    n1, n2 = max(64, n//128*64), max(64, 3*n//256*64)
    if n1 < 16*len(first.partitions or (n,)):
        return None
    train = tuple((nn, m) for nn in sorted({n1, n}) for m in (1, 2, 4))
    holdout = tuple(dict.fromkeys(((n, 3), (n2, 1), (n2, 3))))
    own = PackedLinearOperators(config=operators.config, device=operators.device_identity,
                               model=operators.model_identity)
    probes, values = {}, [[], []]
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    try:
        for nn, m in (*train, *holdout):
            if nn not in probes:
                matrices = tuple(adapter.probe(w, nn, x.device) for w in weights)
                prepared = tuple(adapter.prepare_mma(own, w, x.device) for w in matrices)
                if any(w is None for w in prepared):
                    return None
                probes[nn] = matrices, prepared
            matrices, prepared = probes[nn]
            inp = torch.full((m, k), 1/128, dtype=x.dtype, device=x.device)
            targets = tuple(torch.empty((m, nn), dtype=x.dtype, device=x.device) for _ in weights)
            baseline = (linear.bind_native(own, inp, matrices[0], targets[0], params)
                        if len(weights) == 1 else
                        linear.bind_grouped(own, inp, matrices, targets, params))
            if baseline is None:
                return None
            candidate = bind(inp, prepared, targets)
            for index, binding in enumerate((candidate, baseline)):
                values[index].append(_measure(binding, targets))
        coefficients, errors = [], []
        features = lambda nn, m: _features(nn, k, m, len(weights), sms, first.encoding.group_in)
        for times in values:
            fit = fit_nonnegative_cost([features(nn, m) for nn, m in train], times[:len(train)])
            error = max(abs(_predict(fit, features(nn, m))/value - 1)
                        for (nn, m), value in zip((*train, *holdout), times, strict=True))
            coefficients.append(fit)
            errors.append(error)
        _LOGGER.info("packed MMA cost n=%d k=%d members=%d bits=%d group=%d "
                     "candidate_error=%.6f baseline_error=%.6f",
                     n, k, len(weights), first.encoding.bits, first.encoding.group_in,
                     *errors)
        return Cost(*coefficients, *errors)
    finally:
        probes.clear()
        own.prepared_weights.clear()
        own.materialized_weights.clear()


def choose(operators, x, weights, outputs, params, linear):
    """Called after the existing SIMT selector; never bypasses format/mode checks."""
    from cubic_serve.quantization.common.small_rows_cost import _predict

    adapter = getattr(linear, "row_contraction", None)
    if adapter is None or not hasattr(adapter, "prepare_mma"):
        return None
    first = weights[0]
    n, k = first.shape
    enc = first.encoding
    group = min(k, enc.group_in)
    if (operators.config.compute_mode != "wna16" or not params.marlin_enabled
            or x.dtype not in (torch.bfloat16, torch.float16) or not 1 <= x.shape[0] <= 4
            or len(weights) not in (1, 2) or not 2 <= enc.bits <= 4
            or n < 64 or n%64 or group < 32 or group%32
            or k%128 or k%group or x.data_ptr()%16
            or any(w.shape != first.shape or w.encoding != enc
                   or w.partitions != first.partitions or not adapter.supported(w) for w in weights)):
        return None
    # Conservative transport bound shared by all current format adapters;
    # actual packed width is supplied by the format at binding.
    width = 8
    if shared_bytes(rows=4, bits=enc.bits, group=group,
                    raw_bytes=width, tile_k=tile_k(k, group)) > 48*1024:
        return None
    # Explicit projection overrides retain their named operator plan.
    from cubic_serve.ops.strategies import operator_strategy_scope, parameter_overrides
    from cubic_serve.quantization.registry import get_extension
    with operator_strategy_scope(operators.config, device=operators.device_identity,
                                 model=operators.model_identity):
        family = get_extension(enc.format).family("linear")
        if any(parameter_overrides(family, n=w.shape[0], k=k) for w in weights):
            return None
    key = x.device, x.dtype, enc, first.shape, first.partitions, len(weights), params
    if key not in _costs:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("packed MMA selection must precede capture")
        _costs[key] = _fit(operators, x, weights, params, adapter, linear)
    cost = _costs[key]
    if cost is None or max(cost.candidate_error, cost.baseline_error) > .15:
        return None
    features = _features(n, k, x.shape[0], len(weights),
                         torch.cuda.get_device_properties(x.device).multi_processor_count, group)
    candidate = _predict(cost.candidate, features)
    baseline = _predict(cost.baseline, features)
    if prediction_interval(candidate, cost.candidate_error)[1]*1.02 >= prediction_interval(
            baseline, cost.baseline_error)[0]:
        return None
    prepared = tuple(adapter.prepare_mma(operators, w, x.device) for w in weights)
    if any(w is None for w in prepared):
        return None
    return bind(x, prepared, outputs)
