"""Explicit eager full-attention oracle."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from cubic_serve.executor.runner import require_reference_execution
from cubic_serve.ops.eager.activation import compute_dtype, exp_nonpositive, softmax
from cubic_serve.ops.strategies import attention_precision, softmax_precision


def qkv_attention_row(
    query: Tensor, key: Tensor, value: Tensor, *, scale: float, kv_splits: int = 32,
    output_dtype: torch.dtype | None = None,
) -> Tensor:
    """Independent staged oracle for low operands and FP32 dot accumulation.

    Wide scalar products model exact products of 16-bit operands, without calling
    the candidate MMA kernel. The unnormalized P operand is rounded before PV;
    online softmax normalization is independent of the dot operand policy.
    """
    require_reference_execution()
    if query.dtype not in (torch.float16, torch.bfloat16) or not (
        query.dtype == key.dtype == value.dtype
    ):
        raise ValueError("native attention requires matching FP16/BF16 QKV")
    if kv_splits not in (1, 32):
        raise ValueError("attention reference supports the declared 1/32 split plans")
    length = key.shape[0]
    span = (length + kv_splits - 1) // kv_splits
    starts = torch.arange(kv_splits, device=query.device) * span
    ends = torch.minimum(starts + span, torch.tensor(length, device=query.device))
    maximum = torch.full((kv_splits,), -torch.inf, device=query.device)
    denominator = torch.zeros_like(maximum)
    accumulator = torch.zeros((kv_splits, value.shape[-1]), device=query.device)
    dtype = (torch.float32 if softmax_precision() == "fp32"
             else compute_dtype(query.dtype, query.device))
    for offset in range(0, span, 16):
        indices = starts[:, None] + offset + torch.arange(16, device=query.device)[None, :]
        valid = indices < ends[:, None]
        selected = indices.clamp(max=length - 1)
        keys = key[selected].float()
        scores = (keys * query.float()).sum(-1) * scale
        scores = scores.masked_fill(~valid, -torch.inf)
        next_maximum = torch.maximum(maximum, scores.amax(-1))
        active = valid.any(-1)
        safe_maximum = torch.where(active, next_maximum, 0.0)
        # Exhausted splits retain their completed partial. Rescaling them by
        # exp(maximum) would change their weight in the final split merge.
        old_scale = torch.exp(torch.where(active, maximum - safe_maximum, 0.0))
        probability = exp_nonpositive(scores - safe_maximum[:, None], dtype).float()
        # This cast is a mathematical boundary, not an FP32 reduction allowance.
        operand = probability.to(value.dtype).float()
        values = value[selected].float().masked_fill(~valid[:, :, None], 0.0)
        weighted = (operand[:, :, None] * values).sum(1)
        accumulator = accumulator * old_scale[:, None] + weighted
        denominator = denominator * old_scale + probability.sum(-1)
        maximum = next_maximum
    merged = torch.zeros(value.shape[-1], device=query.device)
    total = torch.zeros((), device=query.device)
    peak = torch.full((), -torch.inf, device=query.device)
    for part in range(kv_splits):
        new_peak = torch.maximum(peak, maximum[part])
        old_scale = torch.exp(peak - new_peak)
        part_scale = torch.exp(maximum[part] - new_peak)
        merged = merged * old_scale + accumulator[part] * part_scale
        total = total * old_scale + denominator[part] * part_scale
        peak = new_peak
    return (merged / total).to(output_dtype or query.dtype)


def full_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    query_ordinals: Tensor,
    key_ordinals: Tensor,
    attention_mask: Tensor | None = None,
    causal: bool = True,
    scale: float | None = None,
    accumulator_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype | None = None,
) -> Tensor:
    """Reference GQA/MHA with FP32 reductions and the bound activation exponent rule.

    Canonical token ordinals define causality. RoPE/M-RoPE position axes are intentionally not
    accepted here: multimodal axes may repeat or be non-monotonic and are not sequence order.
    """

    require_reference_execution()
    if accumulator_dtype is not torch.float32:
        raise ValueError("reference attention accumulation is fixed to torch.float32")

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("attention expects query/key/value in BHSD layout")
    if query.device != key.device or query.device != value.device:
        raise ValueError("attention tensors must share a device")
    if not all(tensor.is_floating_point() for tensor in (query, key, value)):
        raise TypeError("attention tensors must be floating point")

    batch, query_heads, query_tokens, head_dim = query.shape
    key_batch, kv_heads, key_tokens, key_dim = key.shape
    value_batch, value_heads, value_tokens, value_dim = value.shape
    if (value_batch, value_heads, value_tokens) != (key_batch, kv_heads, key_tokens):
        raise ValueError("key and value batch/head/token dimensions must match")
    if batch != key_batch or head_dim != key_dim:
        raise ValueError("query and KV batch/head dimensions must match")
    if min(batch, query_heads, query_tokens, head_dim, kv_heads, key_tokens, value_dim) <= 0:
        raise ValueError("reference attention dimensions must be positive")
    if query_heads % kv_heads:
        raise ValueError("query head count must be divisible by KV head count")
    if query_ordinals.shape != (batch, query_tokens):
        raise ValueError("query_ordinals must have shape [batch, query_tokens]")
    if key_ordinals.shape != (batch, key_tokens):
        raise ValueError("key_ordinals must have shape [batch, key_tokens]")
    if query_ordinals.dtype is not torch.int64 or key_ordinals.dtype is not torch.int64:
        raise TypeError("attention ordinals must use int64")
    if query_ordinals.device != query.device or key_ordinals.device != query.device:
        raise ValueError("attention ordinals must be on the tensor device")
    if query_tokens > 1 and bool((query_ordinals[:, 1:] <= query_ordinals[:, :-1]).any().item()):
        raise ValueError("query ordinals must be strictly increasing within each sequence")
    if key_tokens > 1 and bool((key_ordinals[:, 1:] <= key_ordinals[:, :-1]).any().item()):
        raise ValueError("key ordinals must be strictly increasing within each sequence")

    allowed = torch.ones(
        (batch, query_tokens, key_tokens),
        dtype=torch.bool,
        device=query.device,
    )
    if causal:
        allowed &= key_ordinals[:, None, :] <= query_ordinals[:, :, None]
    if attention_mask is not None:
        if attention_mask.shape != allowed.shape or attention_mask.dtype is not torch.bool:
            raise ValueError("attention_mask must be bool [batch, query_tokens, key_tokens]")
        if attention_mask.device != query.device:
            raise ValueError("attention_mask must be on the query device")
        allowed &= attention_mask
    if not bool(torch.all(allowed.any(dim=-1)).item()):
        raise ValueError("every query row must attend to at least one key")

    repeats = query_heads // kv_heads
    score_scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    batch_outputs: list[Tensor] = []
    for batch_index in range(batch):
        head_outputs: list[Tensor] = []
        for query_head in range(query_heads):
            kv_head = query_head // repeats
            query_outputs: list[Tensor] = []
            for query_index in range(query_tokens):
                allowed_indices = torch.nonzero(
                    allowed[batch_index, query_index], as_tuple=False
                ).flatten()
                if attention_precision() == "auto":
                    query_outputs.append(qkv_attention_row(
                        query[batch_index, query_head, query_index],
                        key[batch_index, kv_head].index_select(0, allowed_indices),
                        value[batch_index, kv_head].index_select(0, allowed_indices),
                        scale=score_scale,
                        output_dtype=torch.float32 if output_dtype is torch.float32 else None,
                    ))
                    continue
                query_row = query[batch_index, query_head, query_index].to(accumulator_dtype)
                key_rows = (
                    key[batch_index, kv_head]
                    .index_select(0, allowed_indices)
                    .to(accumulator_dtype)
                )
                value_rows = (
                    value[batch_index, kv_head]
                    .index_select(0, allowed_indices)
                    .to(accumulator_dtype)
                )
                scores = (key_rows * query_row.unsqueeze(0)).sum(dim=-1) * score_scale
                probabilities = softmax(
                    scores, dim=0, storage_dtype=output_dtype or query.dtype
                ).to(accumulator_dtype)
                query_outputs.append((value_rows * probabilities.unsqueeze(-1)).sum(dim=0))
            head_outputs.append(torch.stack(query_outputs, dim=0))
        batch_outputs.append(torch.stack(head_outputs, dim=0))
    output = torch.stack(batch_outputs, dim=0)
    return output.to(dtype=output_dtype or query.dtype)
