"""Sampling boundaries for tensor-parallel sharded vocabulary logits."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from cubic_serve.ops.cuda.collective import all_gather_single, broadcast
from cubic_serve.models.parallel_weights import AxisSpan

if TYPE_CHECKING:
    from cubic_serve.ops.cuda.sampling.greedy import CapturedGreedySampler

__all__ = [
    "FUSED_ROOT_SAMPLING_PROFILE_METRIC",
    "GreedySamplingProfiler",
    "TensorParallelSamplingError",
    "ShardedSamplingParameters",
    "ShardedSamplingWorkspace",
    "SparseSamplingDistribution",
    "SpeculativeSamplingDecision",
    "candidate_probabilities",
    "gather_greedy_broadcast_tp",
    "sample_sparse_distribution",
    "sample_sharded_broadcast_tp",
    "verify_speculative_chain",
]

FUSED_ROOT_SAMPLING_PROFILE_METRIC = (
    "guard_root_candidates_root_global_topk_"
    "temperature_softmax_top_p_sample_gpu_ms"
)


class TensorParallelSamplingError(ValueError):
    """A tensor-parallel sampling boundary was violated."""


@dataclass(frozen=True, slots=True)
class ShardedSamplingParameters:
    """Caller-owned per-row sampling parameters at captured-M capacity."""

    temperatures: Tensor
    top_ks: Tensor
    top_ps: Tensor
    max_top_k: int
    seeds: Tensor | None = None
    ordinals: Tensor | None = None


@dataclass(frozen=True, slots=True)
class ShardedSamplingWorkspace:
    """Caller-owned local and gathered candidate buffers."""

    local_scores: Tensor
    local_token_ids: Tensor
    gathered_scores: Tensor
    gathered_token_ids: Tensor


@dataclass(frozen=True, slots=True)
class SparseSamplingDistribution:
    """One normalized post-temperature/top-k/top-p token distribution."""

    token_ids: Tensor
    probabilities: Tensor


@dataclass(frozen=True, slots=True)
class SpeculativeSamplingDecision:
    """Lossless chain-verification result; the final token is the next root."""

    accepted_draft_tokens: int
    next_root_token_id: int


def candidate_probabilities(
    sorted_scores: Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> Tensor:
    """Apply the canonical sampler transform to descending candidate scores."""

    if (
        sorted_scores.ndim != 1
        or sorted_scores.numel() == 0
        or not sorted_scores.is_floating_point()
        or type(temperature) is not float
        or not 0.0 <= temperature <= 2.0
        or type(top_k) is not int
        or not 1 <= top_k <= sorted_scores.numel()
        or type(top_p) is not float
        or not 0.0 < top_p <= 1.0
    ):
        raise TensorParallelSamplingError("candidate distribution is outside the qualified domain")
    if not bool(torch.all(torch.isfinite(sorted_scores[:top_k])).item()):
        raise TensorParallelSamplingError("active candidate scores must be finite")
    probabilities = _candidate_probability_rows(
        sorted_scores.unsqueeze(0),
        temperatures=sorted_scores.new_tensor([temperature], dtype=torch.float32),
        top_ks=torch.tensor([top_k], dtype=torch.int64, device=sorted_scores.device),
        top_ps=sorted_scores.new_tensor([top_p], dtype=torch.float32),
    ).squeeze(0)
    if not bool(torch.all(torch.isfinite(probabilities)).item()):
        raise TensorParallelSamplingError("candidate distribution has no finite probability mass")
    return probabilities


def _candidate_probability_rows(
    sorted_scores: Tensor,
    *,
    temperatures: Tensor,
    top_ks: Tensor,
    top_ps: Tensor,
) -> Tensor:
    positions = torch.arange(sorted_scores.shape[1], device=sorted_scores.device)
    inside_top_k = positions.unsqueeze(0) < top_ks.unsqueeze(1)
    safe_temperatures = torch.where(temperatures == 0, 1, temperatures)
    scaled = sorted_scores.float() / safe_temperatures.unsqueeze(1)
    probabilities = torch.softmax(scaled.masked_fill(~inside_top_k, float("-inf")), dim=-1)
    inside_top_p = (
        torch.cumsum(probabilities, dim=-1) - probabilities
    ) <= top_ps.unsqueeze(1)
    probabilities.masked_fill_(~inside_top_p, 0)
    probabilities /= probabilities.sum(dim=-1, keepdim=True)
    greedy_rows = temperatures == 0
    greedy = torch.zeros_like(probabilities)
    greedy[:, 0] = 1
    return torch.where(greedy_rows.unsqueeze(1), greedy, probabilities)


def sample_sparse_distribution(
    distribution: SparseSamplingDistribution,
    *,
    generator: torch.Generator,
) -> int:
    """Sample one token with the same request-owned multinomial primitive."""

    _require_sparse_distribution(distribution)
    if not isinstance(generator, torch.Generator) or generator.device != distribution.probabilities.device:
        raise TensorParallelSamplingError("sampling requires a device-matched request generator")
    slot = torch.multinomial(distribution.probabilities, 1, generator=generator)
    return int(distribution.token_ids[slot].item())


def verify_speculative_chain(
    *,
    draft_token_ids: Tensor,
    draft_distributions: tuple[SparseSamplingDistribution, ...],
    target_distributions: tuple[SparseSamplingDistribution, ...],
    generator: torch.Generator,
    acceptance_uniforms: Tensor | None = None,
) -> SpeculativeSamplingDecision:
    """Verify a nonempty linear proposal chain with p/q rejection sampling.

    The provider and transaction planner own the admitted depth limit; the
    mathematical reference has no model-specific depth table.
    """

    if (
        draft_token_ids.ndim != 1
        or draft_token_ids.dtype is not torch.int64
        or not draft_token_ids.is_contiguous()
        or draft_token_ids.numel() == 0
    ):
        raise TensorParallelSamplingError("speculative verification requires nonempty int64 drafts")
    draft_count = draft_token_ids.numel()
    if len(draft_distributions) != draft_count or len(target_distributions) != draft_count + 1:
        raise TensorParallelSamplingError("speculative distributions do not match the draft chain")
    for distribution in (*draft_distributions, *target_distributions):
        _require_sparse_distribution(distribution)
        if distribution.probabilities.device != draft_token_ids.device:
            raise TensorParallelSamplingError("speculative distributions must share one device")
    if not isinstance(generator, torch.Generator) or generator.device != draft_token_ids.device:
        raise TensorParallelSamplingError("verification requires a device-matched request generator")
    if acceptance_uniforms is None:
        acceptance_uniforms = torch.rand(
            (draft_count,), dtype=torch.float32, device=draft_token_ids.device, generator=generator
        )
    elif (
        tuple(acceptance_uniforms.shape) != (draft_count,)
        or acceptance_uniforms.dtype is not torch.float32
        or acceptance_uniforms.device != draft_token_ids.device
        or not acceptance_uniforms.is_contiguous()
        or not bool(torch.all(torch.isfinite(acceptance_uniforms)).item())
        or not bool(torch.all((acceptance_uniforms >= 0) & (acceptance_uniforms < 1)).item())
    ):
        raise TensorParallelSamplingError("acceptance uniforms must be finite float32 values in [0, 1)")

    for depth in range(draft_count):
        token_id = int(draft_token_ids[depth].item())
        draft_probability = _token_probability(draft_distributions[depth], token_id)
        target_probability = _token_probability(target_distributions[depth], token_id)
        ratio = 0.0 if draft_probability == 0.0 else min(1.0, target_probability / draft_probability)
        if draft_probability > 0.0 and float(acceptance_uniforms[depth].item()) < ratio:
            continue
        residual = _residual_distribution(
            target_distributions[depth], draft_distributions[depth]
        )
        return SpeculativeSamplingDecision(
            accepted_draft_tokens=depth,
            next_root_token_id=sample_sparse_distribution(residual, generator=generator),
        )
    return SpeculativeSamplingDecision(
        accepted_draft_tokens=draft_count,
        next_root_token_id=sample_sparse_distribution(
            target_distributions[draft_count], generator=generator
        ),
    )


def _require_sparse_distribution(distribution: SparseSamplingDistribution) -> None:
    if not isinstance(distribution, SparseSamplingDistribution):
        raise TensorParallelSamplingError("a sparse sampling distribution is required")
    token_ids = distribution.token_ids
    probabilities = distribution.probabilities
    if (
        token_ids.ndim != 1
        or token_ids.dtype is not torch.int64
        or not token_ids.is_contiguous()
        or probabilities.ndim != 1
        or probabilities.dtype is not torch.float32
        or not probabilities.is_contiguous()
        or token_ids.shape != probabilities.shape
        or token_ids.numel() == 0
        or token_ids.device != probabilities.device
        or not bool(torch.all(token_ids >= 0).item())
        or not bool(torch.all(torch.isfinite(probabilities) & (probabilities >= 0)).item())
        or len(set(token_ids.tolist())) != token_ids.numel()
        or not torch.isclose(
            probabilities.sum(), probabilities.new_tensor(1.0), rtol=1e-6, atol=1e-6
        ).item()
    ):
        raise TensorParallelSamplingError("sparse distribution must be unique and normalized")


def _token_probability(distribution: SparseSamplingDistribution, token_id: int) -> float:
    matching = distribution.token_ids == token_id
    if not bool(torch.any(matching).item()):
        return 0.0
    return float(distribution.probabilities[matching].item())


def _residual_distribution(
    target: SparseSamplingDistribution,
    draft: SparseSamplingDistribution,
) -> SparseSamplingDistribution:
    token_ids = torch.unique(torch.cat((target.token_ids, draft.token_ids)), sorted=True)
    target_probabilities = torch.zeros(token_ids.shape, dtype=torch.float32, device=token_ids.device)
    draft_probabilities = torch.zeros_like(target_probabilities)
    target_slots = torch.searchsorted(token_ids, target.token_ids)
    draft_slots = torch.searchsorted(token_ids, draft.token_ids)
    target_probabilities[target_slots] = target.probabilities
    draft_probabilities[draft_slots] = draft.probabilities
    probabilities = torch.clamp(target_probabilities - draft_probabilities, min=0)
    mass = probabilities.sum()
    if not bool(torch.isfinite(mass).item()) or not bool((mass > 0).item()):
        raise TensorParallelSamplingError("a rejected proposal requires positive residual mass")
    return SparseSamplingDistribution(token_ids.contiguous(), (probabilities / mass).contiguous())


class GreedySamplingProfiler(Protocol):
    """Optional step-timing sink; production sampling never depends on it."""

    def begin_cuda(self, metric: str) -> None: ...

    def end_cuda(self, metric: str) -> None: ...

    def flush_cuda(self, metrics: tuple[str, ...]) -> None: ...

    def record_cpu(self, metric: str, milliseconds: float) -> None: ...


class RootSamplingPrimitive(Protocol):
    """Fixed-ABI device sampler used by the performance path."""

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
    ) -> None: ...


def _require_buffer(
    value: Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    label: str,
) -> None:
    if (
        tuple(value.shape) != shape
        or value.dtype is not dtype
        or value.device != device
        or value.layout is not torch.strided
        or not value.is_contiguous()
    ):
        raise TensorParallelSamplingError(
            f"{label} must be contiguous {dtype} {shape} on {device}"
        )


def _group_ranks(group: ProcessGroup, label: str) -> tuple[int, ...]:
    try:
        ranks = tuple(dist.get_process_group_ranks(group))
    except (RuntimeError, ValueError) as error:
        raise TensorParallelSamplingError(f"{label} group is not usable") from error
    return ranks


def _require_group(group: ProcessGroup, expected: tuple[int, ...], label: str) -> None:
    ranks = _group_ranks(group, label)
    if ranks != expected:
        raise TensorParallelSamplingError(f"{label} group ranks must be {expected}, got {ranks}")


def _padded_vocab_size(spans_by_rank: tuple[AxisSpan, ...]) -> int:
    if (
        type(spans_by_rank) is not tuple
        or not spans_by_rank
        or not all(isinstance(span, AxisSpan) for span in spans_by_rank)
    ):
        raise TensorParallelSamplingError("final vocab spans require AxisSpan values")
    ordered = sorted(spans_by_rank)
    cursor = 0
    for span in ordered:
        if span.start != cursor:
            raise TensorParallelSamplingError(
                "final vocab spans must cover the padded vocabulary once without gaps"
            )
        cursor = span.stop
    return cursor


def _require_sampling_parameter_buffers(
    parameters: ShardedSamplingParameters,
    *,
    capacity: int,
    device: torch.device,
) -> None:
    if not isinstance(parameters, ShardedSamplingParameters):
        raise TensorParallelSamplingError("sharded sampling parameters are required")
    _require_buffer(
        parameters.temperatures,
        shape=(capacity,),
        dtype=torch.float32,
        device=device,
        label="temperatures",
    )
    _require_buffer(
        parameters.top_ks,
        shape=(capacity,),
        dtype=torch.int64,
        device=device,
        label="top-k values",
    )
    _require_buffer(
        parameters.top_ps,
        shape=(capacity,),
        dtype=torch.float32,
        device=device,
        label="top-p values",
    )
    if (parameters.seeds is None) != (parameters.ordinals is None):
        raise TensorParallelSamplingError("seed and ordinal buffers must be paired")
    if parameters.seeds is not None and parameters.ordinals is not None:
        _require_buffer(
            parameters.seeds,
            shape=(capacity,),
            dtype=torch.int64,
            device=device,
            label="sampling seeds",
        )
        _require_buffer(
            parameters.ordinals,
            shape=(capacity,),
            dtype=torch.int64,
            device=device,
            label="sampling ordinals",
        )
    if type(parameters.max_top_k) is not int or not 1 <= parameters.max_top_k <= 20:
        raise TensorParallelSamplingError("maximum top-k must be an integer from 1 through 20")


def _require_sampling_parameters(
    parameters: ShardedSamplingParameters,
    *,
    capacity: int,
    active_rows: int,
    device: torch.device,
) -> Tensor:
    _require_sampling_parameter_buffers(parameters, capacity=capacity, device=device)
    temperatures = parameters.temperatures[:active_rows]
    top_ks = parameters.top_ks[:active_rows]
    top_ps = parameters.top_ps[:active_rows]
    return (
        torch.all(torch.isfinite(temperatures) & (temperatures >= 0))
        & torch.all((top_ks >= 1) & (top_ks <= parameters.max_top_k))
        & torch.all(torch.isfinite(top_ps) & (top_ps > 0) & (top_ps <= 1))
    )


def _require_sampling_workspace(
    workspace: ShardedSamplingWorkspace,
    *,
    tp_size: int,
    capacity: int,
    candidate_count: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    if not isinstance(workspace, ShardedSamplingWorkspace):
        raise TensorParallelSamplingError("final ranks require candidate workspaces")
    _require_buffer(
        workspace.local_scores,
        shape=(capacity, candidate_count),
        dtype=dtype,
        device=device,
        label="local candidate scores",
    )
    _require_buffer(
        workspace.local_token_ids,
        shape=(capacity, candidate_count),
        dtype=torch.int64,
        device=device,
        label="local candidate token ids",
    )
    _require_buffer(
        workspace.gathered_scores,
        shape=(tp_size, capacity, candidate_count),
        dtype=dtype,
        device=device,
        label="gathered candidate scores",
    )
    _require_buffer(
        workspace.gathered_token_ids,
        shape=(tp_size, capacity, candidate_count),
        dtype=torch.int64,
        device=device,
        label="gathered candidate token ids",
    )


def _sample_root_candidates(
    *,
    workspace: ShardedSamplingWorkspace,
    parameters: ShardedSamplingParameters,
    final_vocab_spans: tuple[AxisSpan, ...],
    true_vocab_size: int,
    active_rows: int,
    token_ids: Tensor,
    request_generators: tuple[torch.Generator | None, ...] | None,
    parameters_valid: Tensor,
    profiler: GreedySamplingProfiler | None,
) -> Tensor:
    """Sample active rows on the sole authority rank and return one validity scalar."""

    candidate_count = parameters.max_top_k
    gathered_scores = workspace.gathered_scores
    gathered_ids = workspace.gathered_token_ids
    candidates_valid = parameters_valid
    if profiler is not None:
        profiler.begin_cuda("guard_root_candidates_gpu_ms")
    for shard_index, span in enumerate(final_vocab_spans):
        visible_width = max(0, min(span.stop, true_vocab_size) - span.start)
        random_slots = min(candidate_count, visible_width)
        scores = gathered_scores[shard_index]
        ids = gathered_ids[shard_index]
        expected_counts = torch.full(
            (active_rows,), random_slots, dtype=torch.int64, device=scores.device
        )
        if visible_width:
            expected_counts.masked_fill_(parameters.temperatures[:active_rows] == 0, 1)
        expected = torch.arange(candidate_count, device=scores.device).unsqueeze(0)
        expected = expected < expected_counts.unsqueeze(1)
        active_scores = scores[:active_rows]
        active_ids = ids[:active_rows]
        real = (
            torch.isfinite(active_scores)
            & (active_ids >= span.start)
            & (active_ids < min(span.stop, true_vocab_size))
        )
        sentinel = torch.isnan(active_scores) & (active_ids == -1)
        candidates_valid &= torch.all(
            torch.where(expected, real, sentinel)
        )
        if random_slots > 1:
            random_rows = parameters.temperatures[:active_rows] > 0
            ordered_ids = torch.sort(active_ids[random_rows, :random_slots], dim=-1).values
            candidates_valid &= torch.all(
                ordered_ids[:, 1:] != ordered_ids[:, :-1]
            )
        if active_rows < scores.shape[0]:
            candidates_valid &= torch.all(
                torch.isnan(scores[active_rows:]) & (ids[active_rows:] == -1)
            )
    if profiler is not None:
        profiler.end_cuda("guard_root_candidates_gpu_ms")

    token_ids.fill_(-1)
    guard_started_ns = time.monotonic_ns() if profiler is not None else 0
    candidates_are_valid = bool(candidates_valid.item())
    if profiler is not None:
        profiler.record_cpu(
            "guard_root_candidates_host_wait_ms",
            (time.monotonic_ns() - guard_started_ns) / 1_000_000,
        )
    if not candidates_are_valid:
        return candidates_valid

    if profiler is not None:
        profiler.begin_cuda("root_global_topk_gpu_ms")
    scores = workspace.gathered_scores[:, :active_rows].permute(1, 0, 2).reshape(
        active_rows, -1
    )
    ids = workspace.gathered_token_ids[:, :active_rows].permute(1, 0, 2).reshape(
        active_rows, -1
    )
    real = (ids >= 0) & torch.isfinite(scores)
    sortable_ids = torch.where(real, ids, true_vocab_size)
    id_order = torch.argsort(sortable_ids, dim=-1, stable=True)
    sortable_ids = torch.gather(sortable_ids, 1, id_order)
    sortable_scores = torch.gather(
        torch.where(real, scores.float(), float("-inf")), 1, id_order
    )
    score_order = torch.argsort(sortable_scores, dim=-1, descending=True, stable=True)
    sorted_scores = torch.gather(sortable_scores, 1, score_order)
    sorted_ids = torch.gather(sortable_ids, 1, score_order)
    if profiler is not None:
        profiler.end_cuda("root_global_topk_gpu_ms")

    temperatures = parameters.temperatures[:active_rows]
    greedy_rows = temperatures == 0
    token_ids[:active_rows].copy_(sorted_ids[:, 0])
    random_rows = torch.nonzero(~greedy_rows, as_tuple=False).flatten()
    if random_rows.numel():
        if profiler is not None:
            profiler.begin_cuda("root_temperature_softmax_gpu_ms")
        random_scores = sorted_scores[random_rows]
        random_ids = sorted_ids[random_rows]
        random_top_ks = parameters.top_ks[random_rows]
        probabilities = _candidate_probability_rows(
            random_scores,
            temperatures=temperatures[random_rows],
            top_ks=random_top_ks,
            top_ps=parameters.top_ps[random_rows],
        )
        if profiler is not None:
            profiler.end_cuda("root_temperature_softmax_gpu_ms")
            profiler.begin_cuda("root_top_p_gpu_ms")
        if profiler is not None:
            profiler.end_cuda("root_top_p_gpu_ms")
            profiler.begin_cuda("root_sample_gpu_ms")

        if request_generators is None or all(generator is None for generator in request_generators):
            sampled_slots = torch.multinomial(probabilities, 1).squeeze(1)
        else:
            sampled_slots = torch.empty(
                random_rows.shape, dtype=torch.int64, device=random_rows.device
            )
            index_started_ns = time.monotonic_ns() if profiler is not None else 0
            random_row_indices = random_rows.tolist()
            if profiler is not None:
                profiler.record_cpu(
                    "root_sample_index_host_wait_ms",
                    (time.monotonic_ns() - index_started_ns) / 1_000_000,
                )
            for target, row in enumerate(random_row_indices):
                sampled_slots[target] = torch.multinomial(
                    probabilities[target], 1, generator=request_generators[row]
                )
        token_ids[random_rows] = torch.gather(
            random_ids, 1, sampled_slots.unsqueeze(1)
        ).squeeze(1)
        if profiler is not None:
            profiler.end_cuda("root_sample_gpu_ms")
    return candidates_valid


def _prepare_local_candidates(
    *,
    local_logits: Tensor,
    workspace: ShardedSamplingWorkspace,
    temperatures: Tensor,
    active_rows: int,
    visible_width: int,
    local_count: int,
    vocab_start: int,
    all_random_rows: bool = False,
    row_activity: Tensor | None = None,
    profiler: GreedySamplingProfiler | None = None,
    token_mask: Tensor | None = None,
) -> None:
    """Shared top-k/greedy math for ordinary and captured TP sampling.

    The caller clears candidate storage and owns its validated, fixed ABI.
    An ordinary execution slices its logical prefix. A captured composition
    supplies explicit device activity over the fixed physical row portfolio.
    This boundary changes no score, tie, probability, or RNG arithmetic.
    """
    visible_logits = local_logits[:active_rows, :visible_width]
    if profiler is not None:
        profiler.begin_cuda("guard_rank_logits_finite_gpu_ms")
    finite_rows = torch.all(torch.isfinite(visible_logits), dim=-1)
    if token_mask is not None:
        global_ids = torch.arange(
            vocab_start, vocab_start + visible_width, device=visible_logits.device
        )
        allowed = (
            (token_mask[:active_rows, global_ids // 32].to(torch.int64) >> (global_ids % 32)) & 1
        ).bool()
        visible_logits = visible_logits.masked_fill(~allowed, float("-inf"))
    if profiler is not None:
        profiler.end_cuda("guard_rank_logits_finite_gpu_ms")
    greedy_rows = temperatures[:active_rows] == 0
    owned_finite = finite_rows if row_activity is None else finite_rows & row_activity
    invalid_owned = ~finite_rows if row_activity is None else ~finite_rows & row_activity
    random_finite_rows = owned_finite & ~greedy_rows
    if profiler is not None:
        profiler.begin_cuda("rank_local_topk_gpu_ms")
    scores, indices = torch.topk(visible_logits, local_count, dim=-1)
    selected_rows = random_finite_rows.unsqueeze(1)
    if token_mask is not None:
        selected_rows = selected_rows & torch.isfinite(scores)
    workspace.local_scores[:active_rows, :local_count].copy_(
        torch.where(selected_rows, scores, float("nan"))
    )
    workspace.local_token_ids[:active_rows, :local_count].copy_(
        torch.where(
            selected_rows,
            indices + vocab_start,
            # -1 is ownerless candidate padding. A nonfinite owned
            # shard must survive gathering as an invalid pair, even
            # when other shards supply enough finite top-k candidates.
            torch.where(invalid_owned.unsqueeze(1), -2, -1),
        )
    )
    if profiler is not None:
        profiler.end_cuda("rank_local_topk_gpu_ms")
    if not all_random_rows:
        if profiler is not None:
            profiler.begin_cuda("rank_greedy_max_gpu_ms")
        exact_scores, exact_indices = torch.max(visible_logits, dim=-1)
        exact_rows = owned_finite & greedy_rows
        if token_mask is not None:
            exact_rows &= torch.isfinite(exact_scores)
        workspace.local_scores[:active_rows, 0].copy_(
            torch.where(
                exact_rows,
                exact_scores,
                workspace.local_scores[:active_rows, 0],
            )
        )
        workspace.local_token_ids[:active_rows, 0].copy_(
            torch.where(
                exact_rows,
                exact_indices + vocab_start,
                workspace.local_token_ids[:active_rows, 0],
            )
        )
        if profiler is not None:
            profiler.end_cuda("rank_greedy_max_gpu_ms")


def sample_sharded_broadcast_tp(
    *,
    local_logits: Tensor | None,
    local_vocab_span: AxisSpan | None,
    final_ranks: tuple[int, ...],
    final_vocab_spans: tuple[AxisSpan, ...],
    true_vocab_size: int,
    active_rows: int,
    token_ids: Tensor,
    parameters: ShardedSamplingParameters,
    workspace: ShardedSamplingWorkspace | None,
    final_tp_group: ProcessGroup | None,
    world_group: ProcessGroup,
    device: torch.device,
    request_generators: tuple[torch.Generator | None, ...] | None = None,
    root_sampler: RootSamplingPrimitive | None = None,
    root_valid: Tensor | None = None,
    all_random_rows: bool = False,
    cancel_requested: bool = False,
    profiler: GreedySamplingProfiler | None = None,
    candidates_pre_gathered: bool = False,
    candidates_only: bool = False,
    groups_prevalidated: bool = False,
    prevalidated_rank: int | None = None,
    greedy_plan: CapturedGreedySampler | None = None,
    all_greedy_rows: bool = False,
    validity_out: Tensor | None = None,
) -> Tensor:
    """Sample top-k/top-p from TP vocab shards without gathering full logits.

    Each final rank contributes at most ``max_top_k <= 20`` candidates per active
    row. The first final rank is the only stochastic authority and broadcasts its
    token ids to every pipeline rank. Temperature zero is exact greedy sampling
    and never advances request RNG state. Inactive captured-M rows are sentinels,
    never candidates.

    A speculative transaction may supply a private scalar ``validity_out``
    to retain the complete device guard without a per-proposal host barrier.
    The caller must reject any false guard before target review or publication,
    and use bounded safe token inputs for intervening private draft work.
    """

    boundary_started_ns = time.monotonic_ns() if profiler is not None else 0
    if not dist.is_initialized():
        raise TensorParallelSamplingError("distributed sampling requires initialization")
    if (
        type(final_ranks) is not tuple
        or not final_ranks
        or any(type(item) is not int or item < 0 for item in final_ranks)
        or tuple(sorted(set(final_ranks))) != final_ranks
        or len(final_vocab_spans) != len(final_ranks)
    ):
        raise TensorParallelSamplingError(
            "final ranks must be canonical and align one-to-one with vocab spans"
        )
    if world_group is None:
        raise TensorParallelSamplingError("an explicit world group is required")
    rank = prevalidated_rank if groups_prevalidated else dist.get_rank()
    if type(rank) is not int or rank < 0:
        raise TensorParallelSamplingError("prevalidated sampling rank is invalid")
    world_ranks = final_ranks if groups_prevalidated else _group_ranks(world_group, "world")
    if (
        not world_ranks
        or tuple(sorted(set(world_ranks))) != world_ranks
        or any(rank not in world_ranks for rank in final_ranks)
    ):
        raise TensorParallelSamplingError("final ranks must form a subset of the world")
    if rank not in world_ranks:
        raise TensorParallelSamplingError("sampling rank is outside the explicit world")

    if token_ids.ndim != 1 or token_ids.numel() == 0:
        raise TensorParallelSamplingError("token buffer must have non-empty captured-M shape")
    capacity = token_ids.shape[0]
    _require_buffer(
        token_ids,
        shape=(capacity,),
        dtype=torch.int64,
        device=device,
        label="token buffer",
    )
    if type(active_rows) is not int or not 0 < active_rows <= capacity:
        raise TensorParallelSamplingError("active rows must be within captured-M capacity")
    _require_sampling_parameter_buffers(parameters, capacity=capacity, device=device)
    if validity_out is not None:
        protected = [token_ids, parameters.temperatures, parameters.top_ks, parameters.top_ps]
        protected.extend(t for t in (local_logits, root_valid, parameters.seeds,
                                     parameters.ordinals) if t is not None)
        if workspace is not None:
            protected.extend((workspace.local_scores, workspace.local_token_ids,
                              workspace.gathered_scores, workspace.gathered_token_ids))
        _require_buffer(validity_out, shape=(), dtype=torch.bool, device=device,
                        label="deferred sampling validity")
        if any(validity_out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
               for t in protected):
            raise TensorParallelSamplingError("deferred sampling validity must be private")
        if candidates_only:
            raise TensorParallelSamplingError("candidate-only sampling has no token guard")
    if type(all_greedy_rows) is not bool:
        raise TensorParallelSamplingError("all-greedy row state must be boolean")
    if type(all_random_rows) is not bool or type(candidates_only) is not bool:
        raise TensorParallelSamplingError("all-random row state must be boolean")
    root_rank = final_ranks[0]
    if type(cancel_requested) is not bool or (cancel_requested and rank != root_rank):
        raise TensorParallelSamplingError("only the first final rank may request cancellation")
    if request_generators is not None and (
        rank != root_rank
        or type(request_generators) is not tuple
        or len(request_generators) != active_rows
        or any(
            generator is not None
            and (
                not isinstance(generator, torch.Generator)
                or generator.device != device
            )
            for generator in request_generators
        )
    ):
        raise TensorParallelSamplingError(
            "only the sampling root may own one device-matched generator per active row"
        )

    padded_vocab_size = _padded_vocab_size(final_vocab_spans)
    if (
        type(true_vocab_size) is not int
        or true_vocab_size <= 0
        or true_vocab_size > padded_vocab_size
    ):
        raise TensorParallelSamplingError("true vocabulary must fit the padded vocabulary")

    masked_plan = greedy_plan is not None and greedy_plan.token_mask is not None
    if ((all_greedy_rows and not candidates_only) or masked_plan) and not candidates_pre_gathered:
        if greedy_plan is None:
            raise TensorParallelSamplingError("greedy sampling requires a qualified startup graph")
        greedy_plan.validate_call(
            local_logits=local_logits,
            local_vocab_span=local_vocab_span,
            final_ranks=final_ranks,
            final_vocab_spans=final_vocab_spans,
            true_vocab_size=true_vocab_size,
            token_ids=token_ids,
            parameters=parameters,
            workspace=workspace,
            final_tp_group=final_tp_group,
            world_group=world_group,
            device=device,
            root_valid=root_valid,
        )
        if candidates_only:
            greedy_plan.replay_candidates(active_rows)
            return token_ids
        if masked_plan:
            result = greedy_plan.replay(
                active_rows,
                cancel_requested,
                profiler,
                validity_out=validity_out,
                stochastic=not all_greedy_rows,
            )
        else:
            result = greedy_plan.replay(
                active_rows, cancel_requested, profiler, validity_out=validity_out
            )
        if profiler is not None:
            profiler.record_cpu("sampling_boundary_wall_cpu_ms",
                                (time.monotonic_ns() - boundary_started_ns) / 1_000_000)
        return result
    parameters_valid = _require_sampling_parameters(
        parameters, capacity=capacity, active_rows=active_rows, device=device
    )
    is_final = rank in final_ranks
    candidates_valid: Tensor | None = None
    local_count = 0
    if not is_final:
        non_final_values = (local_logits, local_vocab_span, workspace, final_tp_group)
        if any(value is not None for value in non_final_values):
            raise TensorParallelSamplingError("non-final pipeline ranks cannot own logits")
    else:
        if final_tp_group is None:
            raise TensorParallelSamplingError("final ranks require their TP group")
        if not groups_prevalidated:
            _require_group(final_tp_group, final_ranks, "final TP")
        final_index = final_ranks.index(rank)
        expected_span = final_vocab_spans[final_index]
        if local_vocab_span != expected_span or local_logits is None:
            raise TensorParallelSamplingError("local logits differ from the rank vocab span")
        if local_logits.dtype not in (torch.bfloat16, torch.float16):
            raise TensorParallelSamplingError("local logits require FP16 or BF16")
        _require_buffer(
            local_logits,
            shape=(capacity, expected_span.size),
            dtype=local_logits.dtype,
            device=device,
            label="local logits",
        )
        if workspace is None:
            raise TensorParallelSamplingError("final ranks require candidate workspaces")
        _require_sampling_workspace(
            workspace,
            dtype=local_logits.dtype,
            tp_size=len(final_ranks),
            capacity=capacity,
            candidate_count=parameters.max_top_k,
            device=device,
        )
        buffers = (
            local_logits,
            token_ids,
            workspace.local_scores,
            workspace.local_token_ids,
            workspace.gathered_scores,
            workspace.gathered_token_ids,
        )
        if len({value.untyped_storage().data_ptr() for value in buffers}) != len(buffers):
            raise TensorParallelSamplingError("TP sampling buffers cannot alias")

        token_ids.fill_(-1)
        visible_width = max(0, min(expected_span.stop, true_vocab_size) - expected_span.start)
        local_count = min(parameters.max_top_k, visible_width)
        if not candidates_pre_gathered:
            workspace.local_scores.fill_(float("nan"))
            workspace.local_token_ids.fill_(-1)
        if local_count and not candidates_pre_gathered:
            _prepare_local_candidates(
                local_logits=local_logits, workspace=workspace,
                temperatures=parameters.temperatures, active_rows=active_rows,
                visible_width=visible_width, local_count=local_count,
                vocab_start=expected_span.start, all_random_rows=all_random_rows,
                profiler=profiler,
            )

        if profiler is not None:
            profiler.begin_cuda("tp_candidate_all_gather_gpu_ms")
        if not candidates_pre_gathered:
            all_gather_single(
                workspace.gathered_scores.view(-1),
                workspace.local_scores.view(-1),
                group=final_tp_group,
            )
            all_gather_single(
                workspace.gathered_token_ids.view(-1),
                workspace.local_token_ids.view(-1),
                group=final_tp_group,
            )
        if profiler is not None:
            profiler.end_cuda("tp_candidate_all_gather_gpu_ms")
        if not candidates_only and rank == root_rank:
            if root_sampler is None:
                if device.type == "cuda":
                    raise TensorParallelSamplingError(
                        "performance sampling requires the fused root sampler"
                    )
                candidates_valid = _sample_root_candidates(
                    workspace=workspace,
                    parameters=parameters,
                    final_vocab_spans=final_vocab_spans,
                    true_vocab_size=true_vocab_size,
                    active_rows=active_rows,
                    token_ids=token_ids,
                    request_generators=request_generators,
                    parameters_valid=parameters_valid,
                    profiler=profiler,
                )
            else:
                if (
                    parameters.seeds is None
                    or parameters.ordinals is None
                    or root_valid is None
                ):
                    raise TensorParallelSamplingError(
                        "fused root sampling requires seed, ordinal, and validity buffers"
                    )
                _require_buffer(
                    root_valid,
                    shape=(capacity,),
                    dtype=torch.bool,
                    device=device,
                    label="root sampling validity",
                )
                if profiler is not None:
                    profiler.begin_cuda(FUSED_ROOT_SAMPLING_PROFILE_METRIC)
                root_sampler.launch(
                    workspace.gathered_scores,
                    workspace.gathered_token_ids,
                    parameters.temperatures,
                    parameters.top_ps,
                    parameters.top_ks,
                    parameters.seeds,
                    parameters.ordinals,
                    token_ids,
                    root_valid,
                )
                if profiler is not None:
                    profiler.end_cuda(FUSED_ROOT_SAMPLING_PROFILE_METRIC)
                candidates_valid = parameters_valid & torch.all(
                    root_valid[:active_rows]
                )
            if active_rows < capacity and cancel_requested:
                token_ids[active_rows] = -2

    if candidates_only:
        if profiler is not None and is_final:
            metrics = ["tp_candidate_all_gather_gpu_ms"]
            if local_count:
                metrics.extend(
                    ("guard_rank_logits_finite_gpu_ms", "rank_local_topk_gpu_ms")
                )
                if not all_random_rows:
                    metrics.append("rank_greedy_max_gpu_ms")
            torch.cuda.current_stream(device).synchronize()
            profiler.flush_cuda(tuple(metrics))
            profiler.record_cpu(
                "sampling_boundary_wall_cpu_ms",
                (time.monotonic_ns() - boundary_started_ns) / 1_000_000,
            )
        return token_ids

    if profiler is not None:
        profiler.begin_cuda("tp_token_broadcast_gpu_ms")
    broadcast(token_ids, group=world_group, group_src=root_rank)
    if profiler is not None:
        profiler.end_cuda("tp_token_broadcast_gpu_ms")
        profiler.begin_cuda("guard_final_token_gpu_ms")
    active_token_ids = token_ids[:active_rows]
    valid = torch.all((active_token_ids >= 0) & (active_token_ids < true_vocab_size))
    if candidates_valid is not None:
        valid &= candidates_valid
    if active_rows < capacity:
        padding = token_ids[active_rows:]
        valid &= (padding[0] == -1) | (padding[0] == -2)
        if padding.numel() > 1:
            valid &= torch.all(padding[1:] == -1)
    if profiler is not None:
        profiler.end_cuda("guard_final_token_gpu_ms")
    validity_started_ns = time.monotonic_ns() if profiler is not None else 0
    if validity_out is None:
        is_valid = bool(valid.item())
    else:
        validity_out.copy_(valid)
        is_valid = True  # The leased transaction consumes this guard before review.
    if profiler is not None:
        final_wait_ms = (time.monotonic_ns() - validity_started_ns) / 1_000_000
        profiler.record_cpu("guard_final_token_host_wait_ms", final_wait_ms)
        profiler.record_cpu("validity_host_sync_cpu_ms", final_wait_ms)
        profiler.record_cpu(
            "sampling_boundary_wall_cpu_ms",
            (time.monotonic_ns() - boundary_started_ns) / 1_000_000,
        )
        metrics = ["tp_token_broadcast_gpu_ms", "guard_final_token_gpu_ms"]
        if is_final:
            if local_count:
                metrics.extend(("guard_rank_logits_finite_gpu_ms", "rank_local_topk_gpu_ms"))
                if not all_random_rows:
                    metrics.append("rank_greedy_max_gpu_ms")
            metrics.append("tp_candidate_all_gather_gpu_ms")
        if rank == root_rank and candidates_valid is not None:
            if root_sampler is None:
                metrics.extend(("guard_root_candidates_gpu_ms", "root_global_topk_gpu_ms"))
                if request_generators is None or any(
                    generator is not None for generator in request_generators
                ):
                    metrics.extend(
                        (
                            "root_temperature_softmax_gpu_ms",
                            "root_top_p_gpu_ms",
                            "root_sample_gpu_ms",
                        )
                    )
            else:
                metrics.append(FUSED_ROOT_SAMPLING_PROFILE_METRIC)
        profiler.flush_cuda(tuple(metrics))
    if not is_valid:
        raise TensorParallelSamplingError("TP sampling rejected invalid candidates or tokens")
    return token_ids


def gather_greedy_broadcast_tp(
    *,
    local_logits: Tensor | None,
    local_vocab_span: AxisSpan | None,
    final_ranks: tuple[int, ...],
    final_vocab_spans: tuple[AxisSpan, ...],
    true_vocab_size: int,
    active_rows: int,
    token_ids: Tensor,
    local_max_scores: Tensor | None,
    gathered_max_scores: Tensor | None,
    gathered_token_ids: Tensor | None,
    final_tp_group: ProcessGroup | None,
    world_group: ProcessGroup,
    device: torch.device,
    cancel_requested: bool = False,
    profiler: GreedySamplingProfiler | None = None,
) -> Tensor:
    """Select greedy ids from arbitrary TP vocab shards without gathering logits.

    Final ranks all-gather one BF16 score and one global int64 token id per row.
    Every final rank performs the same lexicographic reduction: greatest score,
    then smallest global token id.  The large TP workspaces are caller-owned and
    a world broadcast is needed only when ranks outside the final TP group need
    the ids.
    """

    if not dist.is_initialized():
        raise TensorParallelSamplingError("distributed sampling requires initialization")
    if (
        type(final_ranks) is not tuple
        or not final_ranks
        or any(type(item) is not int or item < 0 for item in final_ranks)
        or tuple(sorted(set(final_ranks))) != final_ranks
        or len(final_vocab_spans) != len(final_ranks)
    ):
        raise TensorParallelSamplingError(
            "final ranks must be canonical and align one-to-one with vocab spans"
        )
    if world_group is None:
        raise TensorParallelSamplingError("an explicit world group is required")
    world_ranks = _group_ranks(world_group, "world")
    if (
        not world_ranks
        or tuple(sorted(set(world_ranks))) != world_ranks
        or any(rank not in world_ranks for rank in final_ranks)
    ):
        raise TensorParallelSamplingError("final ranks must form a subset of the world")
    rank = dist.get_rank()
    if rank not in world_ranks:
        raise TensorParallelSamplingError("sampling rank is outside the explicit world")

    if token_ids.ndim != 1 or token_ids.numel() == 0:
        raise TensorParallelSamplingError("token buffer must have non-empty captured-M shape")
    capacity = token_ids.shape[0]
    _require_buffer(
        token_ids,
        shape=(capacity,),
        dtype=torch.int64,
        device=device,
        label="token buffer",
    )
    if type(active_rows) is not int or not 0 < active_rows <= capacity:
        raise TensorParallelSamplingError("active rows must be within captured-M capacity")
    root_rank = final_ranks[0]
    if type(cancel_requested) is not bool or (cancel_requested and rank != root_rank):
        raise TensorParallelSamplingError("only the first final rank may request cancellation")

    padded_vocab_size = _padded_vocab_size(final_vocab_spans)
    if (
        type(true_vocab_size) is not int
        or true_vocab_size <= 0
        or true_vocab_size > padded_vocab_size
    ):
        raise TensorParallelSamplingError("true vocabulary must fit the padded vocabulary")

    is_final = rank in final_ranks
    candidate_valid: Tensor | None = None
    visible_width = 0
    if not is_final:
        if any(
            value is not None
            for value in (
                local_logits,
                local_vocab_span,
                local_max_scores,
                gathered_max_scores,
                gathered_token_ids,
                final_tp_group,
            )
        ):
            raise TensorParallelSamplingError("non-final pipeline ranks cannot own logits")
    else:
        if final_tp_group is None:
            raise TensorParallelSamplingError("final ranks require their TP group")
        _require_group(final_tp_group, final_ranks, "final TP")
        final_index = final_ranks.index(rank)
        expected_span = final_vocab_spans[final_index]
        if local_vocab_span != expected_span or local_logits is None:
            raise TensorParallelSamplingError("local logits differ from the rank vocab span")
        if local_logits.dtype not in (torch.bfloat16, torch.float16):
            raise TensorParallelSamplingError("local logits require FP16 or BF16")
        _require_buffer(
            local_logits,
            shape=(capacity, expected_span.size),
            dtype=local_logits.dtype,
            device=device,
            label="local logits",
        )
        if local_max_scores is None or gathered_max_scores is None or gathered_token_ids is None:
            raise TensorParallelSamplingError("final ranks require TP candidate workspaces")
        _require_buffer(
            local_max_scores,
            shape=(capacity,),
            dtype=local_logits.dtype,
            device=device,
            label="local max scores",
        )
        _require_buffer(
            gathered_max_scores,
            shape=(len(final_ranks), capacity),
            dtype=local_logits.dtype,
            device=device,
            label="gathered max scores",
        )
        _require_buffer(
            gathered_token_ids,
            shape=(len(final_ranks), capacity),
            dtype=torch.int64,
            device=device,
            label="gathered token ids",
        )
        if (
            len(
                {
                    local_logits.untyped_storage().data_ptr(),
                    local_max_scores.untyped_storage().data_ptr(),
                    gathered_max_scores.untyped_storage().data_ptr(),
                    gathered_token_ids.untyped_storage().data_ptr(),
                    token_ids.untyped_storage().data_ptr(),
                }
            )
            != 5
        ):
            raise TensorParallelSamplingError("TP sampling buffers cannot alias")

        visible_stop = min(expected_span.stop, true_vocab_size)
        visible_width = max(0, visible_stop - expected_span.start)
        token_ids.fill_(-1)
        local_max_scores.fill_(float("nan"))
        if visible_width > 0:
            if profiler is not None:
                profiler.begin_cuda("local_greedy_max_gpu_ms")
            candidates = local_logits[:active_rows, :visible_width]
            scores, indices = torch.max(candidates, dim=-1)
            finite = torch.all(torch.isfinite(candidates), dim=-1)
            local_max_scores[:active_rows].copy_(
                torch.where(finite, scores, torch.full_like(scores, float("nan")))
            )
            token_ids[:active_rows].copy_(
                torch.where(finite, indices + expected_span.start, -1)
            )
            if profiler is not None:
                profiler.end_cuda("local_greedy_max_gpu_ms")
        if active_rows < capacity and cancel_requested:
            token_ids[active_rows].fill_(-2)

        if profiler is not None:
            profiler.begin_cuda("final_tp_all_gather_gpu_ms")
        all_gather_single(
            gathered_max_scores.view(-1), local_max_scores, group=final_tp_group
        )
        all_gather_single(
            gathered_token_ids.view(-1), token_ids, group=final_tp_group
        )
        if profiler is not None:
            profiler.end_cuda("final_tp_all_gather_gpu_ms")

        candidate_valid = torch.ones((), dtype=torch.bool, device=device)
        best_scores = local_max_scores[:active_rows]
        best_ids = token_ids[:active_rows]
        best_scores.fill_(float("-inf"))
        best_ids.fill_(true_vocab_size)
        for shard_index, span in enumerate(final_vocab_spans):
            shard_scores = gathered_max_scores[shard_index, :active_rows]
            shard_ids = gathered_token_ids[shard_index, :active_rows]
            shard_stop = min(span.stop, true_vocab_size)
            if shard_stop <= span.start:
                candidate_valid &= torch.all(torch.isnan(shard_scores) & (shard_ids == -1))
                continue
            shard_valid = (
                torch.isfinite(shard_scores)
                & (shard_ids >= span.start)
                & (shard_ids < shard_stop)
            )
            candidate_valid &= torch.all(shard_valid)
            choose = (shard_scores > best_scores) | (
                (shard_scores == best_scores) & (shard_ids < best_ids)
            )
            best_scores.copy_(torch.where(choose, shard_scores, best_scores))
            best_ids.copy_(torch.where(choose, shard_ids, best_ids))

        padding_valid = torch.ones((), dtype=torch.bool, device=device)
        cancel_seen = torch.zeros((), dtype=torch.bool, device=device)
        if active_rows < capacity:
            padding_valid &= torch.all(torch.isnan(gathered_max_scores[:, active_rows:]))
            padding_ids = gathered_token_ids[:, active_rows:]
            cancel_seen = padding_ids[0, 0] == -2
            padding_valid &= (padding_ids[0, 0] == -1) | cancel_seen
            if padding_ids.shape[1] > 1:
                padding_valid &= torch.all(padding_ids[0, 1:] == -1)
            if padding_ids.shape[0] > 1:
                padding_valid &= torch.all(padding_ids[1:] == -1)
        candidate_valid &= padding_valid
        best_ids.copy_(torch.where(candidate_valid, best_ids, -1))
        token_ids[active_rows:].fill_(-1)
        if active_rows < capacity:
            token_ids[active_rows].copy_(gathered_token_ids[0, active_rows])

    needs_broadcast = final_ranks != world_ranks
    if needs_broadcast:
        if profiler is not None:
            profiler.begin_cuda("world_broadcast_gpu_ms")
        broadcast(token_ids, group=world_group, group_src=root_rank)
        if profiler is not None:
            profiler.end_cuda("world_broadcast_gpu_ms")

    active_token_ids = token_ids[:active_rows]
    valid = torch.all((active_token_ids >= 0) & (active_token_ids < true_vocab_size))
    if candidate_valid is not None:
        valid &= candidate_valid
    if active_rows < capacity:
        padding = token_ids[active_rows:]
        valid &= (padding[0] == -1) | (padding[0] == -2)
        if padding.numel() > 1:
            valid &= torch.all(padding[1:] == -1)
    validity_started_ns = time.monotonic_ns() if profiler is not None else 0
    is_valid = bool(valid.item())
    if profiler is not None:
        profiler.record_cpu(
            "validity_host_sync_cpu_ms",
            (time.monotonic_ns() - validity_started_ns) / 1_000_000,
        )
        cuda_metrics: list[str] = []
        if is_final:
            if visible_width > 0:
                cuda_metrics.append("local_greedy_max_gpu_ms")
            cuda_metrics.append("final_tp_all_gather_gpu_ms")
        if needs_broadcast:
            cuda_metrics.append("world_broadcast_gpu_ms")
        profiler.flush_cuda(tuple(cuda_metrics))
    if not is_valid:
        raise TensorParallelSamplingError("TP sampling rejected invalid candidates or tokens")
    return token_ids
