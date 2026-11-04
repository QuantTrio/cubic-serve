"""Fixed-address residual-stream math; projections and graph ownership are external.

Storage boundaries follow the independent eager hyperconnection definition.
All dimensions come from operands. No model, topology or device-name dispatch.
"""

from __future__ import annotations

from cubic_serve.ops.cuda.startup import weight_bound_calibration_key

import math

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from torch import Tensor
from triton.language.extra import libdevice  # type: ignore[import-untyped]

from cubic_serve.ops.cuda.activation import activation_type, sigmoid, silu
from cubic_serve.ops.cuda.normalization.rms_norm import _zero_centered_rms_norm
from cubic_serve.quantization.common.binding import PackedOperatorBinding, _PackedLaunch


@triton.jit  # type: ignore[untyped-decorator]
def _scaled_activation(  # type: ignore[no-untyped-def]
    X,
    Y,
    COUNT: tl.constexpr,
    DIVISOR: tl.constexpr,
    MULTIPLIER: tl.constexpr,
    SIGMOID: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < COUNT, 0).to(tl.float32)
    divided = tl.div_rn(x, DIVISOR).to(Y.dtype.element_ty).to(tl.float32)
    activated = sigmoid(divided, ACT) if SIGMOID else silu(divided, ACT)
    stored = activated.to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + i, stored * MULTIPLIER, i < COUNT)


@triton.jit  # type: ignore[untyped-decorator]
def _stream_mix(  # type: ignore[no-untyped-def]
    X, GATE, Y, WIDTH: tl.constexpr, STREAMS: tl.constexpr, ACT: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    total = tl.full((BLOCK,), 0, tl.float32)
    for stream in tl.static_range(STREAMS):
        offset = (row * STREAMS + stream) * WIDTH + column
        x = tl.load(X + offset, column < WIDTH, 0).to(tl.float32)
        g = tl.load(GATE + offset, column < WIDTH, 0).to(tl.float32)
        gate = sigmoid(g, ACT).to(Y.dtype.element_ty).to(tl.float32)
        weighted = (x * gate).to(Y.dtype.element_ty).to(tl.float32)
        total = total + weighted
    tl.store(Y + row * WIDTH + column, tl.div_rn(total, float(STREAMS)), column < WIDTH)


@triton.jit  # type: ignore[untyped-decorator]
def _stream_combine(  # type: ignore[no-untyped-def]
    RESIDUAL, UPDATE, GATES, Y, WIDTH: tl.constexpr, STREAMS: tl.constexpr, BLOCK: tl.constexpr
):
    row, stream = tl.program_id(0), tl.program_id(1)
    column = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    offset = (row * STREAMS + stream) * WIDTH + column
    residual = tl.load(RESIDUAL + offset, column < WIDTH, 0).to(tl.float32)
    update = tl.load(UPDATE + row * WIDTH + column, column < WIDTH, 0).to(tl.float32)
    gate = tl.load(GATES + row * STREAMS + stream).to(tl.float32)
    stored_update = (update * gate).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + offset, residual + stored_update, column < WIDTH)


def _validate(*operands: Tensor) -> None:
    x = operands[0]
    if x.device.type != "cuda" or x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("residual streams require CUDA FP16/BF16 operands")
    if any(
        t.device != x.device
        or t.dtype != x.dtype
        or not t.is_contiguous()
        or t.ndim != 2
        or min(t.shape) <= 0
        for t in operands
    ):
        raise ValueError("residual stream operands must be nonempty contiguous same-type matrices")


def bind_scaled_activation(
    x: Tensor,
    output: Tensor,
    *,
    divisor: float,
    multiplier: float = 1.0,
    activation: str = "silu",
) -> PackedOperatorBinding:
    """Divide -> stored activation -> multiply, keeping each rounding boundary."""
    _validate(x, output)
    if (
        x.shape != output.shape
        or activation not in ("silu", "sigmoid")
        or not math.isfinite(divisor)
        or divisor <= 0
        or not math.isfinite(multiplier)
    ):
        raise ValueError("scaled activation geometry or arithmetic parameters differ")
    call = _PackedLaunch.compile(
        _scaled_activation,
        (x, output),
        (
            x.numel(),
            float(divisor),
            float(multiplier),
            activation == "sigmoid",
            activation_type(x.dtype, x.device),
            256,
        ),
        (triton.cdiv(x.numel(), 256),),
        {"num_warps": 4, "enable_fp_fusion": False},
    )
    return PackedOperatorBinding(output, (call,))


def bind_stream_mix(
    normalized: Tensor,
    gate_logits: Tensor,
    output: Tensor,
) -> PackedOperatorBinding:
    """Stored sigmoid/product then ordered FP32 mean of the residual streams."""
    _validate(normalized, gate_logits, output)
    if (
        normalized.shape != gate_logits.shape
        or normalized.shape[0] != output.shape[0]
        or normalized.shape[1] % output.shape[1]
        or any(
            output.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
            for t in (normalized, gate_logits)
        )
    ):
        raise ValueError("stream mix requires complete groups and matching token rows")
    width, streams = output.shape[1], normalized.shape[1] // output.shape[1]
    call = _PackedLaunch.compile(
        _stream_mix,
        (normalized, gate_logits, output),
        (width, streams, activation_type(output.dtype, output.device), 256),
        (output.shape[0], triton.cdiv(width, 256)),
        {"num_warps": 4, "enable_fp_fusion": False},
    )
    return PackedOperatorBinding(output, (call,))


def bind_stream_combine(
    residual: Tensor,
    update: Tensor,
    gates: Tensor,
    output: Tensor,
) -> PackedOperatorBinding:
    """Broadcast a stored gated update to each stream, then add the residual."""
    _validate(residual, update, gates, output)
    rows, width = update.shape
    streams = gates.shape[1]
    if (
        gates.shape[0] != rows
        or residual.shape != (rows, streams * width)
        or output.shape != residual.shape
        or any(
            output.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
            for t in (update, gates)
        )
        or (
            output.untyped_storage().data_ptr() == residual.untyped_storage().data_ptr()
            and output.data_ptr() != residual.data_ptr()
        )
    ):
        raise ValueError("stream combine residual, update and gate geometry differs")
    call = _PackedLaunch.compile(
        _stream_combine,
        (residual, update, gates, output),
        (width, streams, 256),
        (rows, streams, triton.cdiv(width, 256)),
        {"num_warps": 4, "enable_fp_fusion": False},
    )
    return PackedOperatorBinding(output, (call,))


@triton.jit  # type: ignore[untyped-decorator]
def _stream_norm_only(  # type: ignore[no-untyped-def]
    RESIDUAL, WEIGHT, NORMALIZED,
    WIDTH: tl.constexpr, STREAMS: tl.constexpr, EPSILON: tl.constexpr, BLOCK: tl.constexpr,
):
    # The original primitive accepts runtime strides. The packed launch ABI
    # carries tensors and compile-time geometry, so bind contiguous strides here.
    _zero_centered_rms_norm(
        RESIDUAL, WEIGHT, NORMALIZED, WIDTH * STREAMS, WIDTH * STREAMS,
        WIDTH, EPSILON, BLOCK, STREAMS,
    )


@triton.jit  # type: ignore[untyped-decorator]
def _stream_combine_norm(  # type: ignore[no-untyped-def]
    RESIDUAL, UPDATE, LOGITS, WEIGHT, NORMALIZED,
    WIDTH: tl.constexpr, STREAMS: tl.constexpr, EPSILON: tl.constexpr,
    ACT: tl.constexpr, BLOCK: tl.constexpr,
):
    row, stream = tl.program_id(0) // STREAMS, tl.program_id(0) % STREAMS
    column = tl.arange(0, BLOCK)
    mask = column < WIDTH
    offset = (row * STREAMS + stream) * WIDTH + column
    residual = tl.load(RESIDUAL + offset, mask, 0).to(tl.float32)
    update = tl.load(UPDATE + row * WIDTH + column, mask, 0).to(tl.float32)
    logit = tl.load(LOGITS + row * STREAMS + stream).to(tl.float32)
    divided = tl.div_rn(logit, float(STREAMS)).to(RESIDUAL.dtype.element_ty).to(tl.float32)
    activated = sigmoid(divided, ACT).to(RESIDUAL.dtype.element_ty).to(tl.float32)
    gate = (activated * 2.0).to(RESIDUAL.dtype.element_ty).to(tl.float32)
    stored_update = (update * gate).to(RESIDUAL.dtype.element_ty).to(tl.float32)
    summed = (residual + stored_update).to(RESIDUAL.dtype.element_ty)
    tl.store(RESIDUAL + offset, summed, mask)
    # The following norm observes the same rounded residual as the caller.
    values = tl.where(mask, summed.to(tl.float32), 0.0)
    weight = tl.load(WEIGHT + stream * WIDTH + column, mask, 0).to(tl.float32)
    variance = libdevice.div_rn(tl.sum(values * values, axis=0), float(WIDTH))
    inverse = libdevice.rsqrt_rn(variance + EPSILON)
    tl.store(NORMALIZED + offset, values * inverse * (weight + 1.0), mask)


def bind_stream_combine_norm(
    residual: Tensor, update: Tensor, logits: Tensor, weight: Tensor,
    normalized: Tensor, gates: Tensor, *, epsilon: float,
    choices: dict[tuple, bool],
) -> PackedOperatorBinding:
    """Price complete adjacent gate/combine/norm work on private operands.

    Choose argmin(T_three_launches, T_fused) for this geometry/dtype/device.
    Both candidates retain the same fixed per-stream reduction and stores.
    Shared cold calibration verifies exact residual/output identity; its cache
    owns only the choice, never request buffers or a device/model patch table.
    """
    from cubic_serve.ops.cuda.launch_cost import select_launch
    from cubic_serve.ops.cuda.normalization.rms_norm import ZeroCenteredRMSNormPrimitive
    from cubic_serve.ops.strategies import launch_parameters

    _validate(residual, update, logits, normalized, gates)
    if gates.shape != logits.shape:
        raise ValueError("residual gate scratch differs from the logits")
    # Reuse the existing geometry and operand-ownership contracts.
    bind_stream_combine(residual, update, gates, residual)
    width, streams = update.shape[1], logits.shape[1]
    norm = ZeroCenteredRMSNormPrimitive(
        residual, weight, normalized, epsilon=epsilon, group_size=width
    )
    if any(normalized.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
           for value in (residual, update, logits, weight, gates)):
        raise ValueError("normalized output aliases a combine operand")
    if any(residual.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
           for value in (logits, weight)):
        raise ValueError("residual aliases a combine input")
    block = norm.meta.block
    warps = 8 if block >= 4096 else 4  # Same reduction tree as the standalone norm.
    act = activation_type(residual.dtype, residual.device)

    def bind(fused: bool, r: Tensor, u: Tensor, logit_input: Tensor, w: Tensor,
             y: Tensor, g: Tensor) -> PackedOperatorBinding:
        if fused:
            call = _PackedLaunch.compile(
                _stream_combine_norm, (r, u, logit_input, w, y),
                (width, streams, float(epsilon), act, block), (r.shape[0] * streams,),
                {"num_warps": warps, "enable_fp_fusion": False},
            )
            return PackedOperatorBinding(y, (call,))
        activation = bind_scaled_activation(
            logit_input, g, divisor=float(streams), multiplier=2.0, activation="sigmoid"
        )
        combine = bind_stream_combine(r, u, g, r)
        call = _PackedLaunch.compile(
            _stream_norm_only, (r, w, y),
            (width, streams, float(epsilon), block),
            (r.shape[0] * streams,), {"num_warps": warps},
        )
        return PackedOperatorBinding(y, (*activation._launches, *combine._launches, call))

    key = ("residual.combine_norm", residual.device, residual.dtype, residual.shape,
           width, streams, float(epsilon), act, launch_parameters().calibration_samples)
    if key not in choices:
        generator = torch.Generator(device=residual.device).manual_seed(193)

        def sample(value: Tensor) -> Tensor:
            return torch.randn(value.shape, dtype=value.dtype, device=value.device,
                               generator=generator).mul_(0.1)

        initial = sample(residual)
        r, u = torch.empty_like(initial), sample(update)
        sampled_logits, w = sample(logits), sample(weight)
        y, g = torch.empty_like(normalized), torch.empty_like(gates)
        eviction = torch.empty(
            torch.cuda.get_device_properties(residual.device).L2_cache_size * 2,
            device=residual.device, dtype=torch.uint8,
        )
        selected = None

        def select(fused: bool) -> None:
            nonlocal selected
            selected = bind(fused, r, u, sampled_logits, w, y, g)

        def launch() -> None:
            assert selected is not None
            selected.launch()

        def prepare() -> None:
            r.copy_(initial)
            eviction.zero_()

        choices[key] = select_launch(
            (False, True), device=residual.device, bind=select, launch=launch,
            prepare=prepare, snapshot=lambda: (r, y),
            cache_key=weight_bound_calibration_key(("residual-streams", key)),
        )
    return bind(choices[key], residual, update, logits, weight, normalized, gates)


@triton.jit  # type: ignore[untyped-decorator]
def _signed_root_gate(  # type: ignore[no-untyped-def]
    KEY,
    QUERY,
    VALUE,
    GATES,
    OUTPUT,
    WIDTH: tl.constexpr,
    STREAMS: tl.constexpr,
    DIVISOR: tl.constexpr,
    FLOOR: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, stream = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, BLOCK)
    offset = (row * STREAMS + stream) * WIDTH + d
    key = tl.load(KEY + offset, d < WIDTH, 0).to(tl.float32)
    query = tl.load(QUERY + offset, d < WIDTH, 0).to(tl.float32)
    stored_product = (key * query).to(OUTPUT.dtype.element_ty).to(tl.float32)
    stored_sum = tl.sum(stored_product, 0).to(OUTPUT.dtype.element_ty).to(tl.float32)
    score = tl.div_rn(stored_sum, DIVISOR).to(OUTPUT.dtype.element_ty).to(tl.float32)
    floor = tl.full((), FLOOR, tl.float32).to(OUTPUT.dtype.element_ty).to(tl.float32)
    magnitude = (
        libdevice.sqrt_rn(tl.maximum(tl.abs(score), floor))
        .to(OUTPUT.dtype.element_ty)
        .to(tl.float32)
    )
    signed = tl.where(score > 0, magnitude, tl.where(score < 0, -magnitude, 0.0))
    gate = sigmoid(signed, ACT).to(OUTPUT.dtype.element_ty).to(tl.float32)
    value = tl.load(VALUE + row * WIDTH + d, d < WIDTH, 0).to(tl.float32)
    tl.store(OUTPUT + offset, value * gate, d < WIDTH)
    tl.store(GATES + row * STREAMS + stream, gate)


def bind_signed_root_gate(
    key: Tensor,
    query: Tensor,
    value: Tensor,
    gates: Tensor,
    output: Tensor,
    *,
    magnitude_floor: float = 1e-6,
) -> PackedOperatorBinding:
    """Stored per-stream dot / sqrt(width), signed root, sigmoid and broadcast.

    Every low-precision product, sum, scaling, root and gate is an explicit
    storage boundary. Gates are exposed for causal auditing, and are exactly
    the values consumed by the broadcast multiply.
    """
    _validate(key, query, value, gates, output)
    rows, width = value.shape
    streams = gates.shape[1]
    if (
        key.shape != (rows, streams * width)
        or query.shape != key.shape
        or output.shape != key.shape
        or gates.shape[0] != rows
        or not math.isfinite(magnitude_floor)
        or magnitude_floor <= 0
    ):
        raise ValueError("signed-root gate requires complete residual streams and a positive floor")
    for writable in (gates, output):
        if any(
            t is not writable
            and t.untyped_storage().data_ptr() == writable.untyped_storage().data_ptr()
            for t in (key, query, value, gates, output)
        ):
            raise ValueError("signed-root outputs may not alias another operand")
    call = _PackedLaunch.compile(
        _signed_root_gate,
        (key, query, value, gates, output),
        (
            width,
            streams,
            math.sqrt(width),
            magnitude_floor,
            activation_type(output.dtype, output.device),
            triton.next_power_of_2(width),
        ),
        (rows, streams),
        {"num_warps": 4, "enable_fp_fusion": False},
    )
    return PackedOperatorBinding(output, (call,))
