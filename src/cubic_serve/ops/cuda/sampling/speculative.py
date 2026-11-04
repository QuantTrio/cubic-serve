"""Fused K1 speculative verification over TP top-k candidates."""

from __future__ import annotations

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from torch import Tensor


@triton.jit  # type: ignore[untyped-decorator]
def _select_topk(scores, token_ids, real, lanes, K: tl.constexpr):  # type: ignore[no-untyped-def]
    selected_scores = tl.full(lanes.shape, -float("inf"), tl.float32)
    selected_ids = tl.full(lanes.shape, 0x7FFFFFFFFFFFFFFF, tl.int64)
    available = real
    for position in tl.static_range(0, K):
        best_score = tl.max(tl.where(available, scores, -float("inf")), axis=0)
        best_id = tl.min(
            tl.where(
                available & (scores == best_score),
                token_ids,
                0x7FFFFFFFFFFFFFFF,
            ),
            axis=0,
        )
        selected_scores = tl.where(lanes == position, best_score, selected_scores)
        selected_ids = tl.where(lanes == position, best_id, selected_ids)
        available &= ~((scores == best_score) & (token_ids == best_id))
    return selected_scores, selected_ids, tl.sum(real.to(tl.int32), axis=0)


@triton.jit  # type: ignore[untyped-decorator]
def _probabilities(scores, temperature, top_k, top_p, lanes):  # type: ignore[no-untyped-def]
    selected = lanes < top_k
    safe_temperature = tl.where(temperature == 0.0, 1.0, temperature)
    logits = tl.where(selected, scores / safe_temperature, -float("inf"))
    maximum = tl.max(logits, axis=0)
    weights = tl.where(selected, tl.exp(logits - maximum), 0.0)
    probabilities = weights / tl.sum(weights, axis=0)
    exclusive_cdf = tl.cumsum(probabilities, axis=0) - probabilities
    retained = tl.where(selected & (exclusive_cdf <= top_p), probabilities, 0.0)
    mass = tl.sum(retained, axis=0)
    return retained / mass, mass


@triton.jit(  # type: ignore[untyped-decorator]
    do_not_specialize=["active_rows", "Q_ROWS", "P_ROWS"],
    do_not_specialize_on_alignment=[
        "q_scores_ptr", "q_ids_ptr", "p_scores_ptr", "p_ids_ptr", "target_rows_ptr",
        "draft_tokens_ptr", "temperatures_ptr", "top_ps_ptr", "top_ks_ptr", "seeds_ptr",
        "ordinals_ptr", "uniforms_ptr", "decisions_ptr",
    ],
)
def _verify_k1_kernel(  # type: ignore[no-untyped-def]
    q_scores_ptr,
    q_ids_ptr,
    p_scores_ptr,
    p_ids_ptr,
    target_rows_ptr,
    draft_tokens_ptr,
    temperatures_ptr,
    top_ps_ptr,
    top_ks_ptr,
    seeds_ptr,
    ordinals_ptr,
    uniforms_ptr,
    decisions_ptr,
    constrained_rows_ptr,
    active_rows,
    Q_ROWS,
    P_ROWS,
    TP: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_UNIFORMS: tl.constexpr,
    ALLOW_UNDERFILLED: tl.constexpr,
    HAS_CONSTRAINT_ROWS: tl.constexpr,
):
    row = tl.program_id(0)
    active = row < active_rows
    lanes = tl.arange(0, BLOCK)
    candidate_mask = lanes < TP * K
    rank = lanes // K
    local = lanes % K
    target_row = tl.load(target_rows_ptr + row, mask=active, other=0)

    q_offsets = rank * Q_ROWS * K + row * K + local
    p0_offsets = rank * P_ROWS * K + target_row * K + local
    p1_offsets = p0_offsets + K
    q_scores = tl.load(
        q_scores_ptr + q_offsets, mask=active & candidate_mask, other=-float("inf")
    ).to(tl.float32)
    q_ids = tl.load(q_ids_ptr + q_offsets, mask=active & candidate_mask, other=-1)
    p0_scores = tl.load(
        p_scores_ptr + p0_offsets, mask=active & candidate_mask, other=-float("inf")
    ).to(tl.float32)
    p0_ids = tl.load(p_ids_ptr + p0_offsets, mask=active & candidate_mask, other=-1)
    p1_scores = tl.load(
        p_scores_ptr + p1_offsets, mask=active & candidate_mask, other=-float("inf")
    ).to(tl.float32)
    p1_ids = tl.load(p_ids_ptr + p1_offsets, mask=active & candidate_mask, other=-1)

    q_real = candidate_mask & (q_ids >= 0) & (tl.abs(q_scores) < float("inf"))
    p0_real = candidate_mask & (p0_ids >= 0) & (tl.abs(p0_scores) < float("inf"))
    p1_real = candidate_mask & (p1_ids >= 0) & (tl.abs(p1_scores) < float("inf"))
    q_padding = candidate_mask & (q_ids == -1) & (q_scores != q_scores)
    p0_padding = candidate_mask & (p0_ids == -1) & (p0_scores != p0_scores)
    p1_padding = candidate_mask & (p1_ids == -1) & (p1_scores != p1_scores)
    pairs_valid = (
        (tl.sum((~candidate_mask | q_real | q_padding).to(tl.int32), axis=0) == BLOCK)
        & (tl.sum((~candidate_mask | p0_real | p0_padding).to(tl.int32), axis=0) == BLOCK)
        & (tl.sum((~candidate_mask | p1_real | p1_padding).to(tl.int32), axis=0) == BLOCK)
    )
    q_scores, q_ids, q_count = _select_topk(q_scores, q_ids, q_real, lanes, K)
    p0_scores, p0_ids, p0_count = _select_topk(
        p0_scores, p0_ids, p0_real, lanes, K
    )
    p1_scores, p1_ids, p1_count = _select_topk(
        p1_scores, p1_ids, p1_real, lanes, K
    )

    temperature = tl.load(temperatures_ptr + row, mask=active, other=1.0).to(
        tl.float32
    )
    top_p = tl.load(top_ps_ptr + row, mask=active, other=1.0).to(tl.float32)
    top_k = tl.load(top_ks_ptr + row, mask=active, other=1)
    q_probabilities, q_mass = _probabilities(
        q_scores, temperature, top_k, top_p, lanes
    )
    p0_probabilities, p0_mass = _probabilities(
        p0_scores, temperature, top_k, top_p, lanes
    )
    p1_probabilities, p1_mass = _probabilities(
        p1_scores, temperature, top_k, top_p, lanes
    )

    draft_token = tl.load(draft_tokens_ptr + row, mask=active, other=-1)
    q_draft = tl.sum(tl.where(q_ids == draft_token, q_probabilities, 0.0), axis=0)
    p_draft = tl.sum(tl.where(p0_ids == draft_token, p0_probabilities, 0.0), axis=0)
    ratio = tl.minimum(p_draft / q_draft, 1.0)
    seed = tl.load(seeds_ptr + row, mask=active, other=0)
    ordinal = tl.load(ordinals_ptr + row, mask=active, other=0)
    random_seed = (seed ^ (seed >> 32) ^ 0x9E3779B9).to(tl.uint32)
    accept_uniform = (
        tl.load(uniforms_ptr + row * 2)
        if USE_UNIFORMS
        else tl.rand(random_seed, (ordinal * 2).to(tl.uint32))
    )
    random_accepted = accept_uniform < ratio
    greedy = temperature == 0.0
    p0_top = tl.min(
        tl.where(lanes == 0, p0_ids, 0x7FFFFFFFFFFFFFFF), axis=0
    )
    p1_top = tl.min(
        tl.where(lanes == 0, p1_ids, 0x7FFFFFFFFFFFFFFF), axis=0
    )
    greedy_accepted = draft_token == p0_top
    accepted = tl.where(greedy, greedy_accepted, random_accepted)

    q_at_p = tl.zeros(lanes.shape, tl.float32)
    for position in tl.static_range(0, K):
        q_id = tl.min(
            tl.where(lanes == position, q_ids, 0x7FFFFFFFFFFFFFFF), axis=0
        )
        q_probability = tl.sum(
            tl.where(lanes == position, q_probabilities, 0.0), axis=0
        )
        q_at_p = tl.where(p0_ids == q_id, q_probability, q_at_p)
    residual = tl.maximum(p0_probabilities - q_at_p, 0.0)
    residual_mass = tl.sum(residual, axis=0)
    sample_uniform = (
        tl.load(uniforms_ptr + row * 2 + 1)
        if USE_UNIFORMS
        else tl.rand(random_seed ^ 0x85EBCA6B, (ordinal * 2 + 1).to(tl.uint32))
    )
    p1_cdf = tl.cumsum(p1_probabilities, axis=0)
    residual_cdf = tl.cumsum(residual, axis=0)
    p1_slot = tl.min(
        tl.where(p1_cdf > sample_uniform, lanes, BLOCK).to(tl.int32), axis=0
    )
    residual_slot = tl.min(
        tl.where(
            residual_cdf > sample_uniform * residual_mass, lanes, BLOCK
        ).to(tl.int32),
        axis=0,
    )
    p1_token = tl.min(tl.where(lanes == p1_slot, p1_ids, 0x7FFFFFFFFFFFFFFF), axis=0)
    residual_token = tl.min(tl.where(lanes == residual_slot, p0_ids, 0x7FFFFFFFFFFFFFFF), axis=0)
    underfilled = ALLOW_UNDERFILLED
    if HAS_CONSTRAINT_ROWS:
        underfilled = underfilled & tl.load(constrained_rows_ptr + row, mask=active, other=False)
    valid = (
        active
        & pairs_valid
        & (target_row >= 0)
        & (target_row + 1 < P_ROWS)
        & (temperature >= 0.0)
        & (temperature < float("inf"))
        & (top_k >= 1)
        & (top_k <= K)
        & (top_p > 0.0)
        & (top_p <= 1.0)
        & (seed >= 0)
        & (ordinal >= 0)
        & (q_count >= tl.where(greedy | underfilled, 1, top_k))
        & (p0_count >= tl.where(greedy | underfilled, 1, top_k))
        & (p1_count >= tl.where(greedy | underfilled, 1, top_k))
        & (q_mass > 0.0)
        & (p0_mass > 0.0)
        & (p1_mass > 0.0)
        & (q_draft > 0.0)
        & (greedy | accepted | (residual_mass > 0.0))
    )
    greedy_next = tl.where(greedy_accepted, p1_top, p0_top)
    next_token = tl.where(
        greedy, greedy_next, tl.where(accepted, p1_token, residual_token)
    )
    tl.store(decisions_ptr + row * 2, tl.where(valid, accepted.to(tl.int64), -1))
    tl.store(decisions_ptr + row * 2 + 1, tl.where(valid, next_token, -1))


def verify_one_draft_candidates(
    *,
    q_scores: Tensor,
    q_token_ids: Tensor,
    p_scores: Tensor,
    p_token_ids: Tensor,
    target_rows: Tensor,
    draft_tokens: Tensor,
    temperatures: Tensor,
    top_ps: Tensor,
    top_ks: Tensor,
    seeds: Tensor,
    ordinals: Tensor,
    decisions: Tensor,
    active_rows: int,
    uniforms: Tensor | None = None,
    allow_underfilled: bool = False,
    constrained_rows: Tensor | None = None,
) -> None:
    """Launch one device-only K1 verification step for active proposal rows."""
    if type(allow_underfilled) is not bool:
        raise TypeError("speculative underfill policy must be boolean")
    if constrained_rows is not None and (
        not allow_underfilled
        or constrained_rows.dtype != torch.bool
        or constrained_rows.shape != temperatures.shape
        or constrained_rows.device != temperatures.device
        or not constrained_rows.is_contiguous()
    ):
        raise ValueError("invalid constrained-row ownership for speculative verification")

    operands = (
        q_scores,
        q_token_ids,
        p_scores,
        p_token_ids,
        target_rows,
        draft_tokens,
        temperatures,
        top_ps,
        top_ks,
        seeds,
        ordinals,
        decisions if uniforms is None else uniforms,
        decisions,
    )
    if any(tensor.device.type != "cuda" or not tensor.is_contiguous() for tensor in operands):
        raise ValueError("speculative verification requires contiguous CUDA buffers")
    if len({tensor.device for tensor in operands}) != 1:
        raise ValueError("speculative verification buffers must share one device")
    if (
        q_scores.ndim != 3
        or p_scores.ndim != 3
        or q_scores.dtype is not torch.bfloat16
        or p_scores.dtype is not torch.bfloat16
        or q_token_ids.shape != q_scores.shape
        or p_token_ids.shape != p_scores.shape
        or q_token_ids.dtype is not torch.int64
        or p_token_ids.dtype is not torch.int64
        or q_scores.shape[0] != p_scores.shape[0]
        or q_scores.shape[2] != p_scores.shape[2]
        or not 1 <= q_scores.shape[0] <= 8
        or not 1 <= q_scores.shape[2] <= 20
    ):
        raise ValueError("speculative candidates require matching BF16/int64 [TP,C,K]")
    q_rows = q_scores.shape[1]
    row_shape = (q_rows,)
    if (
        type(active_rows) is not int
        or not 1 <= active_rows <= q_rows
        or target_rows.shape != row_shape
        or target_rows.dtype is not torch.int32
        or draft_tokens.shape != row_shape
        or draft_tokens.dtype is not torch.int64
        or temperatures.shape != row_shape
        or temperatures.dtype is not torch.float32
        or top_ps.shape != row_shape
        or top_ps.dtype is not torch.float32
        or top_ks.shape != row_shape
        or top_ks.dtype is not torch.int64
        or seeds.shape != row_shape
        or seeds.dtype is not torch.int64
        or ordinals.shape != row_shape
        or ordinals.dtype is not torch.int64
        or decisions.shape != (q_rows, 2)
        or decisions.dtype is not torch.int64
        or (
            uniforms is not None
            and (uniforms.shape != (q_rows, 2) or uniforms.dtype is not torch.float32)
        )
    ):
        raise ValueError("speculative row buffers differ from proposal capacity")
    tp, _, candidates = q_scores.shape
    _verify_k1_kernel[(q_rows,)](
        *operands,
        constrained_rows,
        active_rows,
        q_rows,
        p_scores.shape[1],
        TP=tp,
        K=candidates,
        BLOCK=triton.next_power_of_2(tp * candidates),
        USE_UNIFORMS=uniforms is not None,
        ALLOW_UNDERFILLED=allow_underfilled,
        HAS_CONSTRAINT_ROWS=constrained_rows is not None,
        num_warps=4,
    )


__all__ = ["verify_one_draft_candidates"]
