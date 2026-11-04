"""Independent causal compressed-block selection, with no cache ownership."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from cubic_serve.executor.runner import require_reference_execution


def select_blocks(
    queries: Tensor,
    block_keys: Tensor,
    visible_lengths: Tensor,
    *,
    compression: int,
    token_budget: int,
) -> Tensor:
    """Return [batch,queries,budget+compression-1] token indices; -1 is absent.

    Keys are already mean-pooled, normalized and rotated at block-start
    positions. Scores sum rectified per-head dot products. Only complete
    visible blocks compete; every visible token in the incomplete tail stays.
    Equal scores use increasing original block index as the canonical tie rule.
    """
    require_reference_execution()
    if queries.ndim != 4 or block_keys.ndim != 3:
        raise ValueError("queries must be [B,T,H,D] and block keys [B,N,D]")
    if queries.shape[0] != block_keys.shape[0] or queries.shape[-1] != block_keys.shape[-1]:
        raise ValueError("indexer query and key geometry differ")
    if visible_lengths.shape != queries.shape[:2] or visible_lengths.dtype != torch.int64:
        raise ValueError("one INT64 visible length is required per query")
    if compression <= 0 or token_budget < compression or token_budget % compression:
        raise ValueError("token budget must contain a positive integer number of blocks")
    if bool((visible_lengths < 0).any()) or bool(
        (visible_lengths // compression > block_keys.shape[1]).any()
    ):
        raise ValueError("visible lengths exceed the supplied compressed cache")
    result = torch.full(
        (*queries.shape[:2], token_budget + compression - 1),
        -1,
        dtype=torch.int32,
        device=queries.device,
    )
    for batch in range(queries.shape[0]):
        for row in range(queries.shape[1]):
            visible = int(visible_lengths[batch, row])
            complete, tail = divmod(visible, compression)
            if complete:
                scores = queries[batch, row].float() @ block_keys[batch, :complete].float().T
                scores = scores.relu().sum(0) / math.sqrt(queries.shape[-1])
                selected = torch.argsort(scores, descending=True, stable=True)[
                    : token_budget // compression
                ]
                tokens = (
                    selected[:, None] * compression
                    + torch.arange(compression, device=queries.device)
                ).flatten()
            else:
                tokens = torch.empty(0, dtype=torch.int64, device=queries.device)
            tail_tokens = torch.arange(
                complete * compression, complete * compression + tail, device=queries.device
            )
            tokens = torch.cat((tokens, tail_tokens))
            result[batch, row, : tokens.numel()] = tokens.to(torch.int32)
    return result


def attend_selected(
    query: Tensor, key: Tensor, value: Tensor, indices: Tensor, *,
    output_dtype: torch.dtype | None = None,
) -> Tensor:
    """GQA over explicitly selected original token IDs, one query at a time."""
    require_reference_execution()
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("attention operands require [B,H,T,D] layouts")
    batch, heads, tokens, width = query.shape
    if key.shape[0] != batch or key.shape[-1] != width or heads % key.shape[1]:
        raise ValueError("query/KV grouped-head geometry differs")
    if indices.shape[:2] != (batch, tokens):
        raise ValueError("selected indices must cover every query")
    result = torch.empty_like(query, dtype=output_dtype or query.dtype)
    for b in range(batch):
        for t in range(tokens):
            selected = indices[b, t].long()
            selected = selected[selected >= 0].sort().values
            if not selected.numel() or bool((selected >= key.shape[2]).any()):
                raise ValueError("query must select valid KV entries")
            for h in range(heads):
                kv_head = h // (heads // key.shape[1])
                keys = key[b, kv_head, selected].float()
                values = value[b, kv_head, selected].float()
                scores = (keys * query[b, h, t].float()).sum(-1) / math.sqrt(width)
                probability = torch.softmax(scores, dim=0)
                result[b, h, t] = (probability[:, None] * values).sum(0).to(result.dtype)
    return result
