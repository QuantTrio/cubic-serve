"""One-kernel root sampling from tensor-parallel top-k candidates.

The ordering follows llama.cpp ``common/sampling.cpp``, vLLM 0.27.1
``v1/sample/sampler.py``, and SGLang ``srt/layers/sampler.py``: global top-k,
then temperature, FP32 softmax, top-p, and categorical sampling. Instances
retain only immutable launch metadata; every tensor and RNG coordinate remains
caller-owned so the launch is safe to reuse from a fixed CUDA graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from torch import Tensor

_TensorABI = tuple[tuple[int, ...], tuple[int, ...], torch.dtype, torch.device]


@triton.jit  # type: ignore[untyped-decorator]
def _root_topk_topp_kernel(  # type: ignore[no-untyped-def]
    scores_ptr,
    ids_ptr,
    temperatures_ptr,
    top_ps_ptr,
    top_ks_ptr,
    seeds_ptr,
    ordinals_ptr,
    output_tokens_ptr,
    output_valid_ptr,
    constrained_rows_ptr,
    ROWS: tl.constexpr,
    TP: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    ALLOW_UNDERFILLED: tl.constexpr,
    HAS_CONSTRAINT_ROWS: tl.constexpr,
):
    row = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    candidate_count: tl.constexpr = TP * K
    candidate_mask = lanes < candidate_count
    rank = lanes // K
    local = lanes % K
    offsets = rank * ROWS * K + row * K + local

    scores = tl.load(scores_ptr + offsets, mask=candidate_mask, other=-float("inf")).to(
        tl.float32
    )
    token_ids = tl.load(ids_ptr + offsets, mask=candidate_mask, other=-1)
    finite = (scores == scores) & (scores > -float("inf")) & (scores < float("inf"))
    real = candidate_mask & finite & (token_ids >= 0)
    sentinel = candidate_mask & (scores != scores) & (token_ids == -1)
    pairs_valid = tl.sum((~candidate_mask | real | sentinel).to(tl.int32), axis=0) == BLOCK
    real_count = tl.sum(real.to(tl.int32), axis=0)

    selected_scores = tl.full((BLOCK,), -float("inf"), tl.float32)
    selected_ids = tl.full((BLOCK,), 0x7FFFFFFFFFFFFFFF, tl.int64)
    available = real
    for position in tl.static_range(0, K):
        best_score = tl.max(tl.where(available, scores, -float("inf")), axis=0)
        best_id = tl.min(
            tl.where(available & (scores == best_score), token_ids, 0x7FFFFFFFFFFFFFFF),
            axis=0,
        )
        selected_scores = tl.where(lanes == position, best_score, selected_scores)
        selected_ids = tl.where(lanes == position, best_id, selected_ids)
        available &= ~((scores == best_score) & (token_ids == best_id))

    temperature = tl.load(temperatures_ptr + row).to(tl.float32)
    top_p = tl.load(top_ps_ptr + row).to(tl.float32)
    top_k = tl.load(top_ks_ptr + row)
    seed = tl.load(seeds_ptr + row)
    ordinal = tl.load(ordinals_ptr + row)
    greedy = temperature == 0.0
    parameters_valid = (
        (temperature == temperature)
        & (temperature >= 0.0)
        & (temperature < float("inf"))
        & (top_p == top_p)
        & (top_p > 0.0)
        & (top_p <= 1.0)
        & (top_k >= 1)
        & (top_k <= K)
        & (seed >= 0)
        & (ordinal >= 0)
    )
    underfilled = ALLOW_UNDERFILLED
    if HAS_CONSTRAINT_ROWS:
        underfilled = underfilled & tl.load(constrained_rows_ptr + row)
    enough_candidates = tl.where(greedy | underfilled, real_count >= 1, real_count >= top_k)
    row_valid = pairs_valid & parameters_valid & enough_candidates

    safe_temperature = tl.where(greedy, 1.0, temperature)
    selected = lanes < top_k
    logits = tl.where(selected, selected_scores / safe_temperature, -float("inf"))
    max_logit = tl.max(logits, axis=0)
    weights = tl.where(selected, tl.exp(logits - max_logit), 0.0)
    probability = weights / tl.sum(weights, axis=0)
    exclusive_cdf = tl.cumsum(probability, axis=0) - probability
    retained = tl.where(selected & (exclusive_cdf <= top_p), probability, 0.0)
    retained_mass = tl.sum(retained, axis=0)

    # Philox uses all seed bits; xor the high ordinal bits into its key and use
    # the low bits as its counter so (seed, ordinal) names one request-local draw.
    random_seed = seed ^ (seed >> 32) ^ (ordinal >> 32)
    uniform = tl.rand(random_seed, ordinal.to(tl.uint32))
    target = uniform * retained_mass
    retained_cdf = tl.cumsum(retained, axis=0)
    sampled_lane = tl.min(
        tl.where(retained_cdf > target, lanes, BLOCK).to(tl.int32), axis=0
    )
    chosen_lane = tl.where(greedy, 0, sampled_lane)
    sampled_id = tl.min(
        tl.where(lanes == chosen_lane, selected_ids, 0x7FFFFFFFFFFFFFFF), axis=0
    )
    row_valid &= greedy | ((retained_mass > 0.0) & (sampled_lane < BLOCK))
    tl.store(output_tokens_ptr + row, tl.where(row_valid, sampled_id, -1))
    tl.store(output_valid_ptr + row, row_valid)


def _abi(tensor: Tensor) -> _TensorABI:
    if not isinstance(tensor, Tensor):
        raise TypeError("sampling operands must be tensors")
    return tuple(tensor.shape), tensor.stride(), tensor.dtype, tensor.device


@dataclass(frozen=True, slots=True)
class RootTopKTopPSamplingMeta:
    """Complete fixed launch identity for gathered ``[TP,C,K]`` candidates."""

    device: torch.device
    tp: int
    rows: int
    candidates_per_rank: int
    block: int


class RootTopKTopPSamplingPrimitive:
    """Sample root-owned TP candidates without allocations or host RNG state."""

    def __init__(
        self,
        scores: Tensor,
        token_ids: Tensor,
        temperatures: Tensor,
        top_ps: Tensor,
        top_ks: Tensor,
        seeds: Tensor,
        ordinals: Tensor,
        output_tokens: Tensor,
        output_valid: Tensor,
        *,
        allow_underfilled: bool = False,
        constrained_rows: Tensor | None = None,
    ) -> None:
        if type(allow_underfilled) is not bool:
            raise TypeError("candidate underfill policy must be boolean")
        self.allow_underfilled = allow_underfilled
        self._constraint_abi = None if constrained_rows is None else _abi(constrained_rows)
        operands = (
            scores,
            token_ids,
            temperatures,
            top_ps,
            top_ks,
            seeds,
            ordinals,
            output_tokens,
            output_valid,
        )
        self._validate(operands)
        if constrained_rows is not None and (
            not allow_underfilled
            or constrained_rows.dtype != torch.bool
            or constrained_rows.shape != temperatures.shape
            or constrained_rows.device != temperatures.device
            or not constrained_rows.is_contiguous()
        ):
            raise ValueError("invalid constrained-row ownership for root sampling")
        tp, rows, candidates_per_rank = scores.shape
        self._abis = tuple(_abi(tensor) for tensor in operands)
        self.meta = RootTopKTopPSamplingMeta(
            device=scores.device,
            tp=tp,
            rows=rows,
            candidates_per_rank=candidates_per_rank,
            block=triton.next_power_of_2(tp * candidates_per_rank),
        )

    @property
    def launch_signature(self) -> str:
        meta = self.meta
        return (
            f"tp={meta.tp},rows={meta.rows},k={meta.candidates_per_rank},"
            f"block={meta.block},scores=bf16,prob=fp32,rng=philox"
            f",allow_underfilled={int(self.allow_underfilled)}"
        )

    def launch(
        self,
        scores: Tensor,
        token_ids: Tensor,
        temperatures: Tensor,
        top_ps: Tensor,
        top_ks: Tensor,
        seeds: Tensor,
        ordinals: Tensor,
        output_tokens: Tensor,
        output_valid: Tensor,
        *,
        constrained_rows: Tensor | None = None,
    ) -> None:
        """Enqueue one fixed-shape kernel on the current CUDA stream."""

        operands = (
            scores,
            token_ids,
            temperatures,
            top_ps,
            top_ks,
            seeds,
            ordinals,
            output_tokens,
            output_valid,
        )
        if tuple(_abi(tensor) for tensor in operands) != self._abis:
            raise ValueError("sampling operands do not match the fixed launch ABI")
        if (None if constrained_rows is None else _abi(constrained_rows)) != self._constraint_abi:
            raise ValueError("constrained-row ownership differs from the fixed launch ABI")
        meta = self.meta
        _root_topk_topp_kernel[(meta.rows,)](
            *operands,
            constrained_rows,
            ROWS=meta.rows,
            TP=meta.tp,
            K=meta.candidates_per_rank,
            BLOCK=meta.block,
            ALLOW_UNDERFILLED=self.allow_underfilled,
            HAS_CONSTRAINT_ROWS=constrained_rows is not None,
            num_warps=4,
        )

    @staticmethod
    def _validate(operands: tuple[Tensor, ...]) -> None:
        if any(not isinstance(tensor, Tensor) for tensor in operands):
            raise TypeError("sampling operands must be tensors")
        scores, token_ids, temperatures, top_ps, top_ks, seeds, ordinals, tokens, valid = (
            operands
        )
        if any(tensor.device.type != "cuda" or not tensor.is_contiguous() for tensor in operands):
            raise ValueError("all sampling operands must be contiguous CUDA tensors")
        if len({tensor.device for tensor in operands}) != 1:
            raise ValueError("all sampling operands must share one CUDA device")
        if scores.ndim != 3 or not (1 <= scores.shape[0] <= 8) or not (1 <= scores.shape[2] <= 20):
            raise ValueError("scores must be FP16/BF16 [TP,C,K] with TP<=8 and K<=20")
        rows = scores.shape[1]
        row_shape = (rows,)
        if (
            rows <= 0
            or scores.dtype not in (torch.bfloat16, torch.float16)
            or token_ids.dtype is not torch.int64
            or token_ids.shape != scores.shape
            or temperatures.dtype is not torch.float32
            or top_ps.dtype is not torch.float32
            or top_ks.dtype is not torch.int64
            or seeds.dtype is not torch.int64
            or ordinals.dtype is not torch.int64
            or tokens.dtype is not torch.int64
            or valid.dtype is not torch.bool
            or any(tensor.shape != row_shape for tensor in operands[2:])
        ):
            raise ValueError(
                "expected FP16/BF16 scores/int64 ids [TP,C,K], FP32 temperature/top_p, "
                "int64 top_k/seed/ordinal/token, and bool valid [C]"
            )


__all__ = ["RootTopKTopPSamplingMeta", "RootTopKTopPSamplingPrimitive"]
