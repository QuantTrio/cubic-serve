"""Eager hybrid-decoder operators used as correctness oracles.

These functions are checkpoint-independent, state-explicit mathematical building blocks.
They accept no model config, request, scheduler, cache owner, or checkpoint identity.
Normalization and convolution reduce in FP32; GDN has an explicit operand precision policy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor

from cubic_serve.executor.runner import require_reference_execution
from cubic_serve.ops.eager.activation import sigmoid as activation_sigmoid
from cubic_serve.ops.eager.activation import silu as activation_silu
from cubic_serve.ops.strategies import attention_precision

_ACCUMULATOR_DTYPE = torch.float32
_POSITION_DTYPE = torch.int64


def _require_tensor(name: str, tensor: Tensor) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


def _require_dtype(name: str, tensor: Tensor, dtype: torch.dtype) -> None:
    _require_tensor(name, tensor)
    if tensor.dtype is not dtype:
        raise TypeError(f"{name} must use {dtype}; found {tensor.dtype}")


def _require_same_device(reference_name: str, reference: Tensor, **tensors: Tensor) -> None:
    for name, tensor in tensors.items():
        if tensor.device != reference.device:
            raise ValueError(
                f"{name} must be on the same device as {reference_name}; "
                f"found {tensor.device} and {reference.device}"
            )


def _require_finite(name: str, tensor: Tensor) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must contain only finite values")


def _require_positive_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _require_activation_output(output_dtype: torch.dtype) -> None:
    if output_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("reference output_dtype must be torch.float16 or torch.bfloat16")


def _require_nonempty(name: str, tensor: Tensor) -> None:
    if tensor.numel() == 0 or any(dimension <= 0 for dimension in tensor.shape):
        raise ValueError(f"{name} must not have an empty dimension")


def _checked_result(name: str, tensor: Tensor) -> Tensor:
    _require_finite(name, tensor)
    return tensor


def zero_centered_rms_norm(
    inputs: Tensor,
    weight: Tensor,
    *,
    epsilon: float,
    output_dtype: torch.dtype,
) -> Tensor:
    """Apply Qwen3.5 RMSNorm: ``x / rms(x) * (1 + weight)``.

    Each row is reduced independently with FP32 variance and FP32 weight multiplication.  The
    explicit row loop keeps the reduction shape stable across isolated and packed execution.
    """

    require_reference_execution()
    _require_activation_output(output_dtype)
    _require_dtype("inputs", inputs, output_dtype)
    _require_dtype("weight", weight, output_dtype)
    _require_positive_finite("epsilon", epsilon)
    if inputs.ndim < 1:
        raise ValueError("inputs must have rank at least one")
    if weight.ndim != 1 or weight.shape[0] != inputs.shape[-1]:
        raise ValueError("weight must be rank one and match the input hidden width")
    _require_nonempty("inputs", inputs)
    _require_same_device("inputs", inputs, weight=weight)
    _require_finite("inputs", inputs)
    _require_finite("weight", weight)

    hidden_width = inputs.shape[-1]
    fp32_weight = weight.to(dtype=_ACCUMULATOR_DTYPE) + 1.0
    rows: list[Tensor] = []
    for row in inputs.reshape(-1, hidden_width).unbind(dim=0):
        fp32_row = row.to(dtype=_ACCUMULATOR_DTYPE)
        variance = (fp32_row * fp32_row).sum(dim=0) / hidden_width
        normalized = fp32_row * torch.rsqrt(variance + float(epsilon))
        rows.append(normalized * fp32_weight)
    result = torch.stack(rows, dim=0).reshape(inputs.shape).to(dtype=output_dtype)
    return _checked_result("zero-centered RMSNorm output", result)


def interleaved_mrope_cos_sin(
    position_ids: Tensor,
    *,
    rotary_dim: int,
    mrope_section: Sequence[int],
    rope_theta: float,
    output_dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Build partial, three-axis interleaved M-RoPE cosine and sine tensors.

    ``position_ids`` is exactly ``[3, batch, tokens]`` in temporal/height/width order.  Frequency
    slots follow the checkpoint's interleaving rule: temporal is the default axis, height replaces
    ``1::3``, and width replaces ``2::3`` for the counts declared by ``mrope_section``.
    """

    require_reference_execution()
    _require_activation_output(output_dtype)
    _require_dtype("position_ids", position_ids, _POSITION_DTYPE)
    _require_positive_finite("rope_theta", rope_theta)
    if position_ids.ndim != 3 or position_ids.shape[0] != 3:
        raise ValueError("position_ids must have shape [3, batch, tokens]")
    _require_nonempty("position_ids", position_ids)
    if bool((position_ids < 0).any().item()):
        raise ValueError("position_ids must be nonnegative")
    if isinstance(rotary_dim, bool) or not isinstance(rotary_dim, int):
        raise TypeError("rotary_dim must be an integer")
    if rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError("rotary_dim must be positive and even")
    if len(mrope_section) != 3 or any(
        isinstance(count, bool) or not isinstance(count, int) for count in mrope_section
    ):
        raise TypeError("mrope_section must contain exactly three integer counts")
    section = tuple(int(count) for count in mrope_section)
    if any(count <= 0 for count in section):
        raise ValueError("mrope_section counts must be positive")

    frequency_count = rotary_dim // 2
    if sum(section) != frequency_count:
        raise ValueError("mrope_section must sum to rotary_dim / 2")

    # Match qwen3_5's model-type ABI exactly rather than accepting an ambiguous generalized
    # three-axis permutation.  Reject sections for which the declared replacement slices would
    # not contain the promised number of frequency slots.
    axes = [0] * frequency_count
    height_indices = tuple(range(1, section[1] * 3, 3))
    width_indices = tuple(range(2, section[2] * 3, 3))
    if (
        len(height_indices) != section[1]
        or len(width_indices) != section[2]
        or any(index >= frequency_count for index in (*height_indices, *width_indices))
    ):
        raise ValueError("mrope_section cannot be represented by qwen3_5 interleaving")
    for index in height_indices:
        axes[index] = 1
    for index in width_indices:
        axes[index] = 2
    if axes.count(0) != section[0]:
        raise ValueError("mrope_section temporal count contradicts qwen3_5 interleaving")

    exponent = torch.arange(
        0,
        rotary_dim,
        2,
        dtype=_ACCUMULATOR_DTYPE,
        device=position_ids.device,
    ) / float(rotary_dim)
    theta = torch.tensor(float(rope_theta), dtype=_ACCUMULATOR_DTYPE, device=position_ids.device)
    inverse_frequency = torch.reciprocal(torch.pow(theta, exponent))
    axis_frequencies = position_ids.to(dtype=_ACCUMULATOR_DTYPE).unsqueeze(-1)
    axis_frequencies = axis_frequencies * inverse_frequency
    frequencies = torch.stack(
        tuple(axis_frequencies[axis, ..., index] for index, axis in enumerate(axes)),
        dim=-1,
    )
    doubled = torch.cat((frequencies, frequencies), dim=-1)
    cosine = doubled.cos().to(dtype=output_dtype)
    sine = doubled.sin().to(dtype=output_dtype)
    return (
        _checked_result("M-RoPE cosine", cosine),
        _checked_result("M-RoPE sine", sine),
    )


def apply_partial_rotary(
    query: Tensor,
    key: Tensor,
    cosine: Tensor,
    sine: Tensor,
    *,
    output_dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Rotate in FP32, then round once to the requested activation dtype.

    Stored trigonometric operands retain their supplied activation precision.
    The unrotated tail is copied without arithmetic.
    """

    require_reference_execution()
    _require_activation_output(output_dtype)
    for name, tensor in (
        ("query", query),
        ("key", key),
        ("cosine", cosine),
        ("sine", sine),
    ):
        _require_dtype(name, tensor, output_dtype)
        _require_finite(name, tensor)
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query and key must use BHSD layout")
    if cosine.ndim != 3 or sine.shape != cosine.shape:
        raise ValueError("cosine and sine must have matching [batch, tokens, rotary] shape")
    _require_nonempty("query", query)
    _require_nonempty("key", key)
    batch, _, tokens, head_dim = query.shape
    key_batch, _, key_tokens, key_head_dim = key.shape
    rotary_dim = cosine.shape[-1]
    if (key_batch, key_tokens, key_head_dim) != (batch, tokens, head_dim):
        raise ValueError("query and key batch/token/head dimensions must match")
    if cosine.shape[:2] != (batch, tokens):
        raise ValueError("cosine and sine batch/token dimensions must match query and key")
    if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > head_dim:
        raise ValueError("rotary width must be positive, even, and no larger than head_dim")
    _require_same_device("query", query, key=key, cosine=cosine, sine=sine)

    broadcast_cosine = cosine.float().unsqueeze(1)
    broadcast_sine = sine.float().unsqueeze(1)

    def rotate(tensor: Tensor) -> Tensor:
        rotary = tensor[..., :rotary_dim].float()
        half = rotary_dim // 2
        rotated_half = torch.cat((-rotary[..., half:], rotary[..., :half]), dim=-1)
        rotated = rotary * broadcast_cosine + rotated_half * broadcast_sine
        passthrough = tensor[..., rotary_dim:]
        return torch.cat((rotated, passthrough), dim=-1).to(dtype=output_dtype)

    rotated_query = rotate(query)
    rotated_key = rotate(key)
    return (
        _checked_result("rotated query", rotated_query),
        _checked_result("rotated key", rotated_key),
    )


def depthwise_causal_conv1d(
    inputs: Tensor,
    weight: Tensor,
    state: Tensor,
    bias: Tensor | None = None,
    *,
    activation: str,
    output_dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Run a BF16 depthwise causal convolution with explicit per-token state.

    Layouts are ``inputs=[batch, tokens, channels]``, ``weight=[channels, kernel]``, and
    ``state=[batch, channels, kernel - 1]``.  State is ordered oldest-to-newest and matches the
    serving-kernel ABI rather than the padded full-convolution workspace.  The caller's state is
    never modified; the returned state is a new BF16 tensor.
    """

    require_reference_execution()
    _require_activation_output(output_dtype)
    _require_dtype("inputs", inputs, output_dtype)
    _require_dtype("weight", weight, output_dtype)
    _require_dtype("state", state, output_dtype)
    if bias is not None:
        _require_dtype("bias", bias, output_dtype)
    if activation != "silu":
        raise ValueError("qwen3_5 depthwise convolution activation must be 'silu'")
    if inputs.ndim != 3:
        raise ValueError("inputs must have shape [batch, tokens, channels]")
    if weight.ndim != 2:
        raise ValueError("weight must have shape [channels, kernel]")
    batch, _, channels = inputs.shape
    weight_channels, kernel = weight.shape
    if weight_channels != channels:
        raise ValueError("weight channel count must match inputs")
    if kernel < 2:
        raise ValueError("qwen3_5 convolution kernel width must be at least two")
    if state.shape != (batch, channels, kernel - 1):
        raise ValueError("state must have shape [batch, channels, kernel - 1]")
    if bias is not None and bias.shape != (channels,):
        raise ValueError("bias must have shape [channels]")
    _require_nonempty("inputs", inputs)
    _require_nonempty("weight", weight)
    peers = {"weight": weight, "state": state}
    if bias is not None:
        peers["bias"] = bias
    _require_same_device("inputs", inputs, **peers)
    for name, tensor in (("inputs", inputs), ("weight", weight), ("state", state)):
        _require_finite(name, tensor)
    if bias is not None:
        _require_finite("bias", bias)

    work_state = state.clone()
    fp32_weight = weight.to(dtype=_ACCUMULATOR_DTYPE)
    fp32_bias = bias.to(dtype=_ACCUMULATOR_DTYPE) if bias is not None else None
    outputs: list[Tensor] = []
    for token in inputs.unbind(dim=1):
        window = torch.cat((work_state, token.unsqueeze(-1)), dim=-1)
        accumulated = (window.to(dtype=_ACCUMULATOR_DTYPE) * fp32_weight).sum(dim=-1)
        if fp32_bias is not None:
            accumulated = accumulated + fp32_bias
        outputs.append(
            activation_silu(accumulated, storage_dtype=output_dtype).to(dtype=output_dtype)
        )
        work_state = window[..., 1:]
    output = torch.stack(outputs, dim=1)
    return (
        _checked_result("depthwise causal convolution output", output),
        _checked_result("depthwise causal convolution state", work_state),
    )


def _l2_normalize_stored(
    values: Tensor, *, epsilon: float, output_dtype: torch.dtype
) -> Tensor:
    """Independent FP32 L2 reduction followed by the stored Q/K boundary.

    Keep this producer distinct from recurrence: a numerical witness must prove
    its input and stored output before that output becomes a recurrent operand.
    """
    require_reference_execution()
    fp32 = values.float()
    inverse = torch.rsqrt((fp32 * fp32).sum(dim=-1, keepdim=True) + float(epsilon))
    return (fp32 * inverse).to(dtype=output_dtype)


def gated_delta_net_recurrent(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    decay_input: Tensor,
    beta_input: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    state: Tensor,
    *,
    l2_epsilon: float,
    output_dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Evaluate the Qwen3.5 gated-delta recurrent rule one token at a time.

    ``query`` and ``key`` use ``[batch, tokens, key_heads, key_width]``.  ``value``,
    ``decay_input`` and ``beta_input`` use value heads; key heads are repeated only when value
    heads are a divisible multiple. The recurrent matrix uses the bound precision rule
    and is returned in its declared FP32 or BF16 storage dtype as
    ``[batch, value_heads, value_width, key_width]``.  All
    activations and ``dt_bias`` are BF16; ``a_log`` is BF16 in the dense checkpoint and FP32 in
    the MoE checkpoint.  Q/K L2 reductions run in FP32, then normalized Q/K are explicitly rounded
    to BF16 before the FP32 recurrence; prefill and decode must preserve this boundary.
    """

    require_reference_execution()
    _require_activation_output(output_dtype)
    bf16_tensors = (
        ("query", query),
        ("key", key),
        ("value", value),
        ("decay_input", decay_input),
        ("beta_input", beta_input),
        ("dt_bias", dt_bias),
    )
    for name, tensor in bf16_tensors:
        _require_dtype(name, tensor, output_dtype)
    _require_tensor("a_log", a_log)
    if a_log.dtype not in (output_dtype, _ACCUMULATOR_DTYPE):
        raise TypeError("a_log must use torch.bfloat16 or torch.float32")
    _require_tensor("state", state)
    if state.dtype not in (output_dtype, torch.bfloat16, _ACCUMULATOR_DTYPE):
        raise TypeError(
            f"state must use {output_dtype} or torch.float32 (BF16 storage also supported)"
        )
    _require_positive_finite("l2_epsilon", l2_epsilon)
    if query.ndim != 4 or key.shape != query.shape:
        raise ValueError("query and key must have matching [batch, tokens, key_heads, width]")
    if value.ndim != 4:
        raise ValueError("value must have shape [batch, tokens, value_heads, width]")
    _require_nonempty("query", query)
    _require_nonempty("value", value)
    batch, tokens, key_heads, key_width = query.shape
    value_batch, value_tokens, value_heads, value_width = value.shape
    if (value_batch, value_tokens) != (batch, tokens):
        raise ValueError("query/key and value batch/token dimensions must match")
    if value_heads % key_heads:
        raise ValueError("value head count must be divisible by key head count")
    head_shape = (batch, tokens, value_heads)
    if decay_input.shape != head_shape or beta_input.shape != head_shape:
        raise ValueError("decay_input and beta_input must have shape [batch, tokens, value_heads]")
    if a_log.shape != (value_heads,) or dt_bias.shape != (value_heads,):
        raise ValueError("a_log and dt_bias must have shape [value_heads]")
    if state.shape != (batch, value_heads, value_width, key_width):
        raise ValueError("state must have shape [batch, value_heads, value_width, key_width]")
    peers = {name: tensor for name, tensor in bf16_tensors if name != "query"}
    peers["a_log"] = a_log
    peers["state"] = state
    _require_same_device("query", query, **peers)
    for name, tensor in (*bf16_tensors, ("a_log", a_log), ("state", state)):
        _require_finite(name, tensor)

    state_storage_dtype = state.dtype
    work_state = state.to(dtype=_ACCUMULATOR_DTYPE)
    fp32_a_log = a_log.to(dtype=_ACCUMULATOR_DTYPE)
    fp32_dt_bias = dt_bias.to(dtype=_ACCUMULATOR_DTYPE)
    head_repeats = value_heads // key_heads
    query_scale = 1.0 / math.sqrt(key_width)
    outputs: list[Tensor] = []
    for token_index in range(tokens):
        fp32_query = query[:, token_index].to(dtype=_ACCUMULATOR_DTYPE)
        fp32_key = key[:, token_index].to(dtype=_ACCUMULATOR_DTYPE)
        # L2 normalization is reduced in FP32, but normalized q/k are BF16 graph
        # activations.  Make that boundary explicit before head expansion and the
        # recurrent FP32 arithmetic.  This is also the canonical serving ABI that
        # prefill and decode implementations must share.
        fp32_query = _l2_normalize_stored(
            fp32_query, epsilon=l2_epsilon, output_dtype=output_dtype
        ).to(dtype=_ACCUMULATOR_DTYPE)
        fp32_key = _l2_normalize_stored(
            fp32_key, epsilon=l2_epsilon, output_dtype=output_dtype
        ).to(dtype=_ACCUMULATOR_DTYPE)
        fp32_query = (
            (fp32_query.to(output_dtype) * torch.tensor(query_scale, dtype=output_dtype,
                                                       device=query.device)).float()
            if attention_precision() == "auto" else fp32_query * query_scale
        )
        if head_repeats > 1:
            fp32_query = fp32_query.repeat_interleave(head_repeats, dim=1)
            fp32_key = fp32_key.repeat_interleave(head_repeats, dim=1)

        fp32_value = value[:, token_index].to(dtype=_ACCUMULATOR_DTYPE)
        fp32_decay_input = decay_input[:, token_index].to(dtype=_ACCUMULATOR_DTYPE)
        fp32_beta_input = beta_input[:, token_index].to(dtype=_ACCUMULATOR_DTYPE)
        # sigmoid(b) is materialized in the BF16 activation dtype before recurrence.
        beta = (
            activation_sigmoid(fp32_beta_input, storage_dtype=output_dtype)
            .to(dtype=output_dtype)
            .to(dtype=_ACCUMULATOR_DTYPE)
        )
        decay_log = -torch.exp(fp32_a_log) * functional.softplus(fp32_decay_input + fp32_dt_bias)
        decay = torch.exp(decay_log).unsqueeze(-1).unsqueeze(-1)
        for intermediate_name, intermediate in (
            ("normalized query", fp32_query),
            ("normalized key", fp32_key),
            ("beta", beta),
            ("decay log", decay_log),
            ("decay", decay),
        ):
            _require_finite(intermediate_name, intermediate)

        if attention_precision() == "auto":
            # Independent staged rule: low pointwise boundaries and exact low
            # operand products, followed by FP32 reduction / fused-update RNE.
            decayed = (work_state.to(output_dtype) * decay.to(output_dtype))
            memory = (decayed.float() * fp32_key.unsqueeze(-2)).sum(dim=-1)
            residual = (fp32_value.to(output_dtype) - memory.to(output_dtype))
            delta = residual * beta.to(output_dtype).unsqueeze(-1)
            # FP64 evaluates the single low-product-plus-FP32-addend rounding;
            # it is an oracle only, never the candidate CUDA arithmetic.
            work_state = (decayed.double() + delta.double().unsqueeze(-1)
                          * fp32_key.double().unsqueeze(-2)).float()
        else:
            work_state = work_state * decay
            memory = (work_state * fp32_key.unsqueeze(-2)).sum(dim=-1)
            delta = (fp32_value - memory) * beta.unsqueeze(-1)
            work_state = work_state + delta.unsqueeze(-1) * fp32_key.unsqueeze(-2)
        if state_storage_dtype is not _ACCUMULATOR_DTYPE:
            work_state = work_state.to(dtype=state_storage_dtype).to(dtype=_ACCUMULATOR_DTYPE)
        output_state = (work_state.to(output_dtype).float()
                        if attention_precision() == "auto" else work_state)
        output = (output_state * fp32_query.unsqueeze(-2)).sum(dim=-1)
        _require_finite("gated-delta recurrent state", work_state)
        _require_finite("gated-delta output", output)
        outputs.append(output.to(dtype=output_dtype))

    stacked = torch.stack(outputs, dim=1)
    return (
        _checked_result("gated-delta output", stacked),
        _checked_result("gated-delta final state", work_state.to(dtype=state_storage_dtype)),
    )


def gated_rms_norm(
    inputs: Tensor,
    gate: Tensor,
    weight: Tensor,
    *,
    epsilon: float,
    output_dtype: torch.dtype,
    activation: str = "silu",
) -> Tensor:
    """Apply Qwen3.5's regular RMSNorm followed by its SiLU output gate.

    Normalized values use the activation storage dtype. Low-precision norm weights
    retain that weighted boundary; a declared FP32 norm weight keeps the weighted
    value in FP32. The SiLU gate follows the startup activation policy
    before final output conversion.
    """

    require_reference_execution()
    if activation not in ("silu", "sigmoid"):
        raise ValueError("gated norm activation must be silu or sigmoid")
    _require_activation_output(output_dtype)
    _require_dtype("inputs", inputs, output_dtype)
    _require_dtype("gate", gate, output_dtype)
    _require_tensor("weight", weight)
    if weight.dtype not in (output_dtype, _ACCUMULATOR_DTYPE):
        raise TypeError("weight must use torch.bfloat16 or torch.float32")
    _require_positive_finite("epsilon", epsilon)
    if inputs.ndim < 1 or gate.shape != inputs.shape:
        raise ValueError("inputs and gate must have the same non-scalar shape")
    if weight.ndim != 1 or weight.shape[0] != inputs.shape[-1]:
        raise ValueError("weight must be rank one and match the input hidden width")
    _require_nonempty("inputs", inputs)
    _require_same_device("inputs", inputs, gate=gate, weight=weight)
    for name, tensor in (("inputs", inputs), ("gate", gate), ("weight", weight)):
        _require_finite(name, tensor)

    hidden_width = inputs.shape[-1]
    input_rows = inputs.reshape(-1, hidden_width).unbind(dim=0)
    gate_rows = gate.reshape(-1, hidden_width).unbind(dim=0)
    rows: list[Tensor] = []
    for input_row, gate_row in zip(input_rows, gate_rows, strict=True):
        fp32_input = input_row.to(dtype=_ACCUMULATOR_DTYPE)
        variance = (fp32_input * fp32_input).sum(dim=0) / hidden_width
        normalized = fp32_input * torch.rsqrt(variance + float(epsilon))
        normalized_bf16 = normalized.to(dtype=output_dtype)
        weighted = normalized_bf16 * weight
        if weight.dtype is output_dtype:
            weighted = weighted.to(dtype=output_dtype)
        gate_function = activation_silu if activation == "silu" else activation_sigmoid
        activated_gate = gate_function(gate_row, storage_dtype=output_dtype)
        rows.append(weighted.to(dtype=_ACCUMULATOR_DTYPE) * activated_gate)
    result = torch.stack(rows, dim=0).reshape(inputs.shape).to(dtype=output_dtype)
    return _checked_result("gated RMSNorm output", result)


def sigmoid_output_gate(
    inputs: Tensor,
    gate: Tensor,
    *,
    output_dtype: torch.dtype,
) -> Tensor:
    """Apply the separate Qwen3.5 full-attention output gate in BF16."""

    require_reference_execution()
    _require_activation_output(output_dtype)
    _require_dtype("inputs", inputs, output_dtype)
    _require_dtype("gate", gate, output_dtype)
    if inputs.shape != gate.shape or inputs.ndim < 1:
        raise ValueError("inputs and gate must have the same non-scalar shape")
    _require_nonempty("inputs", inputs)
    _require_same_device("inputs", inputs, gate=gate)
    _require_finite("inputs", inputs)
    _require_finite("gate", gate)
    output = (inputs * activation_sigmoid(gate, storage_dtype=output_dtype).to(output_dtype)).to(
        dtype=output_dtype
    )
    return _checked_result("sigmoid output gate", output)


__all__ = (
    "apply_partial_rotary",
    "depthwise_causal_conv1d",
    "gated_delta_net_recurrent",
    "gated_rms_norm",
    "interleaved_mrope_cos_sin",
    "sigmoid_output_gate",
    "zero_centered_rms_norm",
)
