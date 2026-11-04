"""Bounded independent arithmetic checks for native quantized eager operators.

These diagnostics do not issue artifact, CUDA Graph, or serving-performance
seals. FP64 anchors and operand-derived FP32/RNE bounds replace blanket atol.
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from dataclasses import dataclass
from functools import wraps
from typing import Any, cast

import torch
from torch import Tensor

from cubic_serve.executor.runner import require_reference_execution
from cubic_serve.models.quantization import ActivationFormat
from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.ops.eager.quantization import (
    QuantizedReferenceOperators,
    activation_from_payload,
    decode_matrix,
)
from cubic_serve.ops.strategies import compute_mode as _reference_compute_mode


@dataclass(frozen=True)
class _OperatorEvent:
    kind: str
    operands: tuple[Tensor, ...]
    weight: Tensor | QuantizedMatrix | None
    options: dict[str, Any]
    output: Tensor


def _same_storage_values(actual: Tensor, expected: Tensor) -> bool:
    """Compare typed logical values bytewise, including signed zero."""
    return (
        actual.dtype == expected.dtype
        and actual.shape == expected.shape
        and actual.device == expected.device
        and torch.equal(
            actual.contiguous().reshape(-1).view(torch.uint8),
            expected.contiguous().reshape(-1).view(torch.uint8),
        )
    )


class QuantizedOperatorObserver:
    """Retain stream-ordered snapshots only; diagnostics run after forward."""

    def __init__(self, operators: Any):
        self.operators = operators
        self.events: list[_OperatorEvent] = []

    def linear(self, inputs: Tensor, weight: Tensor | QuantizedMatrix) -> Tensor:
        output = self.operators.linear(inputs, weight)
        self.events.append(_OperatorEvent("linear", (inputs.clone(),), weight, {}, output.clone()))
        return cast(Tensor, output)

    def embedding(
        self, ids: Tensor, weight: Tensor | QuantizedMatrix, *, dtype: torch.dtype
    ) -> Tensor:
        output = self.operators.embedding(ids, weight, dtype=dtype)
        self.events.append(
            _OperatorEvent("embedding", (ids.clone(),), weight, {"dtype": dtype}, output.clone())
        )
        return cast(Tensor, output)

    def full_attention(
        self, query: Tensor, key: Tensor, value: Tensor, **options: Tensor
    ) -> Tensor:
        output = self.operators.full_attention(query, key, value, **options)
        self.events.append(
            _OperatorEvent(
                "full_attention",
                tuple(t.clone() for t in (query, key, value)),
                None,
                {k: v.clone() for k, v in options.items()},
                output.clone(),
            )
        )
        return cast(Tensor, output)


class QuantizedCausalReplay:
    """Witness every producing boundary after the independent full forward.

    Candidate inputs must equal the independently produced replay inputs. Only
    a witnessed linear reduction may then resume from the observed rounded
    output. The independent, unrestricted reference forward remains a separate
    required diagnostic; this replay cannot replace it or issue a graph seal.
    """

    def __init__(self, events: list[_OperatorEvent], reference: Any):
        self.events, self.reference = events, reference
        self.position = 0
        self.reports: list[dict[str, float | int | str]] = []
        self._proved: dict[int, list[tuple[_OperatorEvent, dict[str, float | int | str]]]] = {}

    def _event(
        self, kind: str, operands: tuple[Tensor, ...], weight: Any, options: dict[str, Any]
    ) -> _OperatorEvent:
        if self.position >= len(self.events):
            raise RuntimeError("causal replay emitted an extra operator event")
        event = self.events[self.position]
        self.position += 1
        if (
            event.kind != kind
            or event.weight is not weight
            or event.options.keys() != options.keys()
        ):
            raise RuntimeError(f"causal replay event schedule differs at {self.position}: {kind}")
        if len(operands) != len(event.operands):
            raise RuntimeError("causal replay operand schedule differs")
        for actual, expected in zip(operands, event.operands, strict=True):
            if not _same_storage_values(actual, expected):
                raise AssertionError(f"unwitnessed producing boundary at {self.position}: {kind}")
        for key, expected in event.options.items():
            actual = options[key]
            same = (
                _same_storage_values(actual, expected)
                if isinstance(expected, Tensor)
                else actual == expected
            )
            if not same:
                raise AssertionError("causal replay operator metadata differs")
        return event

    def linear(self, inputs: Tensor, weight: Tensor | QuantizedMatrix) -> Tensor:
        event = self._event("linear", (inputs,), weight, {})
        report = None
        for previous, proof in self._proved.get(id(weight), ()):
            if _same_storage_values(inputs, previous.operands[0]) and _same_storage_values(
                event.output, previous.output
            ):
                # Repeated expert calls with identical immutable weights,
                # operands and rounded outputs reuse the complete-K proof.
                # No approximate fingerprint or partial-column comparison.
                report = {**proof, "reused_from_event": proof["event"]}
                break
        if report is None:
            report = check_linear_enclosure(inputs, weight, event.output)
            report["event"] = self.position
            self._proved.setdefault(id(weight), []).append((event, report.copy()))
        report["event"] = self.position
        self.reports.append(report)
        return event.output.clone()

    def embedding(
        self, ids: Tensor, weight: Tensor | QuantizedMatrix, *, dtype: torch.dtype
    ) -> Tensor:
        event = self._event("embedding", (ids,), weight, {"dtype": dtype})
        expected = self.reference.embedding(ids, weight, dtype=dtype)
        if not _same_storage_values(expected, event.output):
            raise AssertionError("native embedding differs from independent decoded rows")
        return cast(Tensor, expected)

    def full_attention(
        self, query: Tensor, key: Tensor, value: Tensor, **options: Tensor
    ) -> Tensor:
        event = self._event("full_attention", (query, key, value), None, options)
        expected = self.reference.full_attention(query, key, value, **options)
        if not _same_storage_values(expected, event.output):
            raise AssertionError("attention arithmetic changed during quantization adaptation")
        return cast(Tensor, expected)

    def finish(self) -> None:
        if self.position != len(self.events):
            raise RuntimeError("causal replay did not consume the complete forward schedule")


def check_linear_enclosure(
    inputs: Tensor,
    weight: Tensor | QuantizedMatrix,
    observed: Tensor,
    *,
    output_rows: int | None = None,
    fp8_token_scales: Tensor | None = None,
    reference_device: torch.device | str | None = None,
    compute_mode: str | None = None,
    int8_token_scales: Tensor | None = None,
) -> dict[str, float | int | str]:
    """Check all outputs against a bounded-workspace FP64 operand enclosure.

    Independent Torch decoding and FP64 dots follow the operand device by
    default. An explicit CPU device retains the previous oracle for paired
    verification. This executes only after the observed forward has completed;
    it never materializes a complete decoded weight or calls a candidate kernel.
    """
    if int8_token_scales is None and (
        inputs.dtype not in (torch.float16, torch.bfloat16) or observed.dtype is not inputs.dtype
    ):
        raise ValueError("linear enclosure requires one explicit FP16/BF16 boundary")
    device = inputs.device if reference_device is None else torch.device(reference_device)
    if compute_mode is None:
        from cubic_serve.ops.strategies import compute_mode as scoped_mode
        compute_mode = scoped_mode()
    if (compute_mode == "wna8" and isinstance(weight, QuantizedMatrix)
            and int8_token_scales is None):
        from cubic_serve.quantization.registry import select_compute_mode

        compute_mode = select_compute_mode(
            weight.encoding, "linear", compute_mode,
            m=inputs.reshape(-1, inputs.shape[-1]).shape[0],
            n=weight.shape[0], k=weight.shape[1],
            dtype=inputs.dtype, device=inputs.device,
        )
    x = inputs.detach().to(device).reshape(-1, inputs.shape[-1])
    if int8_token_scales is not None:
        if (
            compute_mode != "wna8" or inputs.dtype != torch.int8
            or observed.dtype not in (torch.float16, torch.bfloat16)
            or int8_token_scales.dtype != torch.float32
            or int8_token_scales.shape != (x.shape[0],)
        ):
            raise ValueError("external INT8 activation audit requires codes and FP32 token scales")
        scales = int8_token_scales.detach().to(device).reshape(-1, 1)
        if not bool((torch.isfinite(scales) & (scales > 0)).all()) or bool((x == -128).any()):
            raise ValueError("external INT8 activation has invalid scales or reserved codes")
        x = x.float() * scales
    elif compute_mode == "wna8" and isinstance(weight, QuantizedMatrix):
        from cubic_serve.quantization.registry import get_extension

        reference = get_extension(weight.encoding.format).module("reference")
        x = reference.wna8_operand(x)
    elif fp8_token_scales is not None:
        # A TP row-parallel producer owns the complete-K scale domain. Its
        # supplied immutable scale must be witnessed independently by the caller;
        # recomputing a rank-local maximum would audit a different operator.
        if (
            not isinstance(weight, QuantizedMatrix)
            or weight.encoding.activation is not ActivationFormat.FP8_TOKEN
            or fp8_token_scales.dtype is not torch.float32
            or fp8_token_scales.shape != (x.shape[0],)
        ):
            raise ValueError("external scale audit requires FP8-token FP32 [M]")
        scales = fp8_token_scales.detach().to(device).reshape(-1, 1)
        if not bool((torch.isfinite(scales) & (scales > 0)).all()):
            raise ValueError("external FP8 token scales must be positive and finite")
        codes = (x.double() / scales.double()).float().clamp(-448.0, 448.0)
        x = codes.to(torch.float8_e4m3fn).float() * scales
    elif isinstance(weight, QuantizedMatrix):
        payload = {name: value.to(device) if value.numel() == 1 else value
                   for name, value in weight.payload.items()}
        x = activation_from_payload(x, payload, weight.encoding).dequantized
    x = x.double()
    absolute_x = x.abs()
    y = observed.detach().to(device).reshape(x.shape[0], -1).double()
    k, n = x.shape[1], y.shape[1]
    if not bool(x.isfinite().all() and y.isfinite().all()):
        raise AssertionError("native linear contains nonfinite operands or outputs")
    # Dot reduction plus format-scale multiplication/division boundaries. The
    # output term is the specified FP16/BF16 RNE error, including underflow.
    steps = k + 12
    gamma32 = steps * 2.0**-24 / (1.0 - steps * 2.0**-24)
    gamma64 = k * 2.0**-53 / (1.0 - k * 2.0**-53)
    uout = torch.finfo(observed.dtype).eps / 2.0
    underflow = torch.finfo(observed.dtype).tiny * uout
    maximum_error = maximum_bound = 0.0
    if output_rows is None:
        # Bound transient decode/FP64 work by geometry, not model or GPU. All
        # output columns and the full K reduction remain independently checked.
        # Account conservatively for packed-code expansion and FP64 temporaries.
        output_rows = max(1, min(n, (256 << 20) // (64 * k + 32 * x.shape[0])))
        if isinstance(weight, QuantizedMatrix):
            from cubic_serve.quantization.registry import get_extension
            alignment = math.lcm(
                weight.encoding.group_out,
                get_extension(weight.encoding.format).module("layout").ROW_ALIGNMENT,
            )
            output_rows = max(alignment, output_rows // alignment * alignment)
    for start in range(0, n, output_rows):
        end = min(start + output_rows, n)
        if isinstance(weight, Tensor):
            w = weight[start:end].detach().to(device).double()
        else:
            part = weight.rows(start, end)
            reference_weight = QuantizedMatrix(
                {name: value.detach().to(device) for name, value in part.payload.items()},
                part.encoding,
                part.shape,
                part.partitions,
            )
            if compute_mode == "wna8":
                from cubic_serve.quantization.registry import get_extension
                decoded = get_extension(weight.encoding.format).module("reference").wna8_weight(
                    reference_weight
                )
            else:
                decoded = decode_matrix(reference_weight)
            if (compute_mode != "wna8"
                    and weight.encoding.activation is ActivationFormat.IDENTITY):
                decoded = decoded.to(inputs.dtype)
            w = decoded.double()
        if not bool(w.isfinite().all()):
            raise AssertionError("reference decoding produced nonfinite weights")
        anchor = x @ w.t()
        absolute_sum = absolute_x @ w.abs().t()
        bound = absolute_sum * ((gamma32 + gamma64) / (1.0 - gamma64)) * (1.0 + uout)
        bound += anchor.abs() * uout + underflow
        error = (y[:, start:end] - anchor).abs()
        if not bool(error.isfinite().all()) or bool((error > bound).any()):
            excess = (error - bound).max().item()
            raise AssertionError(
                f"native linear exceeds FP32/format/RNE enclosure: excess={excess}"
            )
        maximum_error = max(maximum_error, error.max().item())
        maximum_bound = max(maximum_bound, bound.max().item())
    return dict(
        rows=x.shape[0],
        outputs=n,
        reduction_length=k,
        allowance="fp32-dot-and-format-scales-plus-output-rne",
        max_abs_error=maximum_error,
        max_bound=maximum_bound,
    )


class ReferenceMathMemo:
    """Memoize pure reference arithmetic only during startup audits.

    Keys bind versioned weights, precision and full typed operand bytes. A hash
    only indexes exact byte comparison. Tensor payload and entry counts are
    bounded separately; immutable host copies never retain candidate state or
    decoded weights. Producers, state, routes, collectives and joint-plan
    receipts are always witnessed by the caller. No serving path uses this.
    """

    def clear(self):
        self.entries.clear()
        self.bytes = 0
        self.integer_geometry.clear()

    def __init__(self, max_bytes=1 << 30, max_entries=131072):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("reference memo requires a nonnegative byte budget")
        if type(max_entries) is not int or max_entries < 0:
            raise ValueError("reference memo requires a nonnegative entry budget")
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.bytes = 0
        self.entries = OrderedDict()
        self.hits = self.misses = 0
        self._linear = QuantizedReferenceOperators.linear
        self._enclosure = check_linear_enclosure
        from cubic_serve.quantization.common.reference import IntegerGeometryMemo

        geometry_bytes = min(1 << 20, max_bytes // 16)
        geometry_entries = min(128, max_entries // 16)
        if not geometry_bytes or not geometry_entries:
            geometry_bytes = geometry_entries = 0
        self.integer_geometry = IntegerGeometryMemo(geometry_bytes, geometry_entries)
        self._math_byte_budget = max_bytes - geometry_bytes
        self._math_entry_budget = max_entries - geometry_entries

    @staticmethod
    def stamp(weight):
        if weight is None:
            return ("weightless-reference-arithmetic",)
        tensors = (
            (("weight", weight),)
            if isinstance(weight, torch.Tensor)
            else tuple(sorted(weight.payload.items()))
        )
        try:
            values = tuple(
                (
                    name,
                    id(t),
                    t._version,
                    t.data_ptr(),
                    str(t.device),
                    t.dtype,
                    tuple(t.shape),
                    tuple(t.stride()),
                )
                for name, t in tensors
            )
        except RuntimeError:
            return None  # No mutation counter: recompute the original oracle.
        metadata = (
            ()
            if isinstance(weight, torch.Tensor)
            else (weight.encoding, weight.shape, weight.partitions)
        )
        return id(weight), metadata, values

    @staticmethod
    def size(value):
        return value.numel() * value.element_size() if isinstance(value, torch.Tensor) else 0

    @staticmethod
    def copy(value, device="cpu"):
        return (
            value.detach().to(device=device, copy=True)
            if isinstance(value, torch.Tensor)
            else dict(value)
        )

    def _compute(self, function):
        from cubic_serve.quantization.common.reference import integer_geometry_scope

        with integer_geometry_scope(self.integer_geometry):
            return function()

    def evaluate(self, kind, options, operands, weight, function):
        stamp = self.stamp(weight)
        if stamp is None or not self.max_bytes or not self.max_entries or torch.is_grad_enabled():
            self.misses += 1
            return function()
        host = tuple(t.detach().to(device="cpu", copy=True) for t in operands)
        raw = tuple(t.contiguous().reshape(-1).view(torch.uint8) for t in host)
        layout = tuple(
            (t.dtype, tuple(t.shape), tuple(t.stride()), str(t.device)) for t in operands
        )
        digest = tuple(hashlib.blake2s(t.numpy().tobytes()).digest() for t in raw)
        key = (
            kind,
            stamp,
            options,
            layout,
            digest,
            torch.get_float32_matmul_precision(),
            torch.backends.cuda.matmul.allow_tf32,
        )
        prior = self.entries.get(key)
        if prior is not None:
            owner, previous, result = prior
            if owner is weight and all(
                torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b)
                for a, b in zip(previous, raw, strict=True)
            ):
                self.entries.move_to_end(key)
                self.hits += 1
                return self.copy(result, operands[0].device)
        self.misses += 1
        result = self._compute(function)
        size = sum(map(self.size, host)) + self.size(result)
        if size <= self._math_byte_budget:
            saved = self.copy(result)
            if prior is not None:
                self.bytes -= sum(map(self.size, prior[1])) + self.size(prior[2])
                del self.entries[key]
            while (self.bytes + size > self._math_byte_budget
                   or len(self.entries) >= self._math_entry_budget):
                _, evicted = self.entries.popitem(last=False)
                self.bytes -= sum(map(self.size, evicted[1])) + self.size(evicted[2])
            self.entries[key] = weight, host, saved
            self.bytes += size
        return result

    def linear(self, operator, inputs, weight):
        require_reference_execution()
        return self.evaluate(
            "linear",
            (operator.compute_mode, operator.output_rows),
            (inputs,),
            weight,
            lambda: self._linear(operator, inputs, weight),
        )

    def activation(self, function, inputs, *, storage_dtype, compute_dtype):
        # The function identity and resolved arithmetic type bind the pure
        # reference rule. Producers and every candidate boundary remain fresh.
        require_reference_execution()
        return self.evaluate(
            "activation", (function, storage_dtype, compute_dtype), (inputs,), None,
            lambda: function(inputs, compute_dtype),
        )

    def enclosure(self, inputs, weight, observed, **kwargs):
        # Pure local proof only. The caller still witnesses producer provenance,
        # schedules, state propagation, ownership and each actual selected plan.
        tensors = tuple((k, v) for k, v in sorted(kwargs.items()) if isinstance(v, torch.Tensor))
        options = tuple(
            (k, str(v)) for k, v in sorted(kwargs.items()) if not isinstance(v, torch.Tensor)
        )
        return self.evaluate(
            "enclosure",
            (_reference_compute_mode(), options, tuple(k for k, _ in tensors)),
            (inputs, observed, *(v for _, v in tensors)),
            weight,
            lambda: self._enclosure(inputs, weight, observed, **kwargs),
        )


def with_reference_integer_geometry(function):
    """Own constant-index metadata for one completed independent audit only."""
    @wraps(function)
    def audited(*args, **kwargs):
        from cubic_serve.quantization.common.reference import (
            IntegerGeometryMemo,
            integer_geometry_scope,
        )
        from cubic_serve.ops.eager.activation import reference_activation_memo

        memo = kwargs.get("reference_math_memo")
        geometry = getattr(memo, "integer_geometry", None)
        if geometry is None:
            geometry = IntegerGeometryMemo()
        try:
            with integer_geometry_scope(geometry), reference_activation_memo(memo):
                return function(*args, **kwargs)
        finally:
            # Release device indices before the caller can price its KV pool.
            # Pure reference dots/enclosures remain in their existing host memo.
            geometry.clear()

    return audited
