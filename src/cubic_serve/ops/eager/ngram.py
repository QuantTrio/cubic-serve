"""State-explicit lexical hashing and dilated depthwise convolution oracles."""

from __future__ import annotations

import torch
from torch import Tensor

from cubic_serve.executor.runner import require_reference_execution
from cubic_serve.ops.eager.activation import silu_stored


def hash_indices(
    tokens: Tensor,
    history: Tensor,
    multipliers: Tensor,
    vocab_sizes: Tensor,
    offsets: Tensor,
    *,
    eos_token_id: int,
) -> tuple[Tensor, Tensor]:
    """Hash [batch, time] tokens, returning IDs and a new immutable history.

    Multiplication wraps in signed INT64 before XOR and positive remainder.
    An EOS resets earlier context for the following token, not for itself.
    Checkpoint metadata supplies all salts, moduli and offsets.
    """
    require_reference_execution()
    order = multipliers.numel()
    if tokens.ndim != 2 or order < 2 or tokens.shape[1] == 0:
        raise ValueError("ngram hashing requires nonempty sequences and order >= 2")
    if history.shape != (tokens.shape[0], order - 1):
        raise ValueError("ngram history must contain exactly order-1 tokens per sequence")
    if any(t.dtype != torch.int64 for t in (tokens, history, multipliers, vocab_sizes, offsets)):
        raise TypeError("ngram tokens and checkpoint metadata must be INT64")
    if multipliers.ndim != 1 or vocab_sizes.ndim != 1 or offsets.shape != vocab_sizes.shape:
        raise ValueError("ngram metadata must be vectors")
    if not vocab_sizes.numel() or vocab_sizes.numel() % (order - 1):
        raise ValueError("each ngram order needs the same positive number of heads")
    if bool((vocab_sizes <= 0).any()) or bool((offsets < 0).any()):
        raise ValueError("ngram moduli must be positive and offsets nonnegative")
    heads = vocab_sizes.numel() // (order - 1)
    context = history.clone()
    rows = []
    for current in tokens.unbind(1):
        window = torch.cat((context, current[:, None]), dim=1)
        shifted = [current]
        reset = torch.zeros_like(current, dtype=torch.bool)
        for lag in range(1, order):
            previous = window[:, -1 - lag]
            reset = reset | (previous == eos_token_id)
            shifted.append(torch.where(reset, eos_token_id, previous))
        hashed = shifted[0] * multipliers[0]
        ids = []
        for lag in range(1, order):
            hashed = torch.bitwise_xor(hashed, shifted[lag] * multipliers[lag])
            start = (lag - 1) * heads
            ids.append(
                hashed[:, None].remainder(vocab_sizes[start : start + heads])
                + offsets[start : start + heads]
            )
        rows.append(torch.cat(ids, dim=-1))
        context = window[:, 1:].clone()
    return torch.stack(rows, dim=1), context


def dilated_depthwise_conv(
    x: Tensor,
    history: Tensor,
    weight: Tensor,
    *,
    dilation: int,
) -> tuple[Tensor, Tensor]:
    """Causal [batch,time,channel] convolution followed by stored SiLU.

    History contains the last (kernel-1)*dilation *inputs*, including every
    intervening token. No stride or segment marker is interpreted as padding.
    """
    require_reference_execution()
    if x.ndim != 3 or weight.ndim != 3 or weight.shape[:2] != (x.shape[-1], 1):
        raise ValueError("depthwise weights must be [channels,1,kernel]")
    if dilation <= 0 or weight.shape[-1] == 0 or x.shape[1] == 0:
        raise ValueError("dilation, kernel and sequence length must be positive")
    extent = (weight.shape[-1] - 1) * dilation
    if history.shape != (x.shape[0], extent, x.shape[-1]) or history.dtype != x.dtype:
        raise ValueError("convolution history geometry or storage dtype differs")
    combined = torch.cat((history, x), dim=1)
    rows = []
    for t in range(x.shape[1]):
        accumulated = torch.zeros_like(x[:, t], dtype=torch.float32)
        for k in range(weight.shape[-1]):
            accumulated = (
                accumulated + combined[:, t + k * dilation].float() * weight[:, 0, k].float()
            )
        stored = accumulated.to(x.dtype)
        rows.append(silu_stored(stored))
    next_history = combined[:, -extent:].clone() if extent else combined[:, :0].clone()
    return torch.stack(rows, dim=1), next_history
