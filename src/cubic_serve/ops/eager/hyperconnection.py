"""Independent gated residual-stream mixing with explicit storage boundaries."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

from cubic_serve.executor.runner import require_reference_execution
from cubic_serve.ops.eager.activation import sigmoid_stored, silu_stored
from cubic_serve.ops.eager.linear import linear


def signed_root_gate(key: Tensor, query: Tensor, value: Tensor, *, magnitude_floor=1e-6) -> Tensor:
    """Stored dot, scaling and signed root, followed by the declared sigmoid."""
    require_reference_execution()
    if key.ndim < 2 or key.shape != query.shape or key.shape[:-2] != value.shape[:-1]:
        raise ValueError("signed-root gate row geometry differs")
    if (key.shape[-1] != value.shape[-1] or min(key.shape[-2:]) <= 0
            or not math.isfinite(magnitude_floor) or magnitude_floor <= 0):
        raise ValueError("signed-root gate width or magnitude floor differs")
    if any(t.dtype != key.dtype or t.device != key.device for t in (query, value)):
        raise ValueError("signed-root gate operands require the same dtype and device")
    score = (key * query).float().sum(-1).to(key.dtype) / math.sqrt(key.shape[-1])
    transformed = score.abs().clamp_min(magnitude_floor).sqrt() * score.sign()
    return (sigmoid_stored(transformed).unsqueeze(-1) * value.unsqueeze(-2)).flatten(-2)


def grouped_rms_norm(x: Tensor, weight: Tensor, *, group_size: int, eps: float) -> Tensor:
    require_reference_execution()
    if group_size <= 0 or x.shape[-1] % group_size or weight.shape != x.shape[-1:]:
        raise ValueError("grouped norm requires complete groups and a full affine vector")
    if eps <= 0:
        raise ValueError("norm epsilon must be positive")
    # CUDA's batched reduction launch can change its FP32 reduction tree when
    # the number of rows changes. The oracle uses one fixed-width group per
    # call, just like the existing canonical ordinary RMSNorm reference.
    groups = x.float().reshape(-1, group_size)
    normalized = torch.stack(
        [row * torch.rsqrt(row.square().sum() / group_size + eps) for row in groups.unbind(0)]
    ).reshape_as(x)
    return (normalized * (1 + weight.float())).to(x.dtype)


def mix(
    x: Tensor,
    norm: Tensor,
    down: Tensor,
    up: Tensor,
    *,
    streams: int,
    eps: float,
    project: Callable[[Tensor, Tensor], Tensor] = linear,
) -> tuple[Tensor, Tensor]:
    """Return one block input and the normalized residual used for injection."""
    require_reference_execution()
    if streams <= 0 or x.shape[-1] % streams:
        raise ValueError("residual width must divide into positive streams")
    width = x.shape[-1] // streams
    normalized = grouped_rms_norm(x, norm, group_size=width, eps=eps)
    low = silu_stored(project(normalized, down) / streams)
    gate = sigmoid_stored(project(low, up))
    gated = gate * normalized
    return gated.reshape(*gated.shape[:-1], streams, width).float().mean(-2).to(x.dtype), normalized


def injection(
    normalized: Tensor,
    weight: Tensor,
    *,
    streams: int,
    project: Callable[[Tensor, Tensor], Tensor] = linear,
) -> Tensor:
    require_reference_execution()
    if streams <= 0 or weight.shape[0] != streams:
        raise ValueError("injection requires one gate per residual stream")
    projected = project(normalized, weight) / streams
    return sigmoid_stored(projected) * 2


def combine(residual: Tensor, output: Tensor, gates: Tensor) -> Tensor:
    require_reference_execution()
    streams = gates.shape[-1]
    if residual.shape != (*output.shape[:-1], streams * output.shape[-1]):
        raise ValueError("block output and residual stream geometry differ")
    if gates.shape[:-1] != output.shape[:-1]:
        raise ValueError("injection gates and block rows differ")
    update = output.unsqueeze(-2) * gates.unsqueeze(-1)
    return residual + update.flatten(-2)
