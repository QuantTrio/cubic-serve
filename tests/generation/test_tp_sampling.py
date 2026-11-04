from __future__ import annotations

import unittest
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from unittest.mock import Mock, patch

import torch

from cubic_serve.generation.tp_sampling import (
    ShardedSamplingParameters,
    ShardedSamplingWorkspace,
    SparseSamplingDistribution,
    SpeculativeSamplingDecision,
    TensorParallelSamplingError,
    candidate_probabilities,
    gather_greedy_broadcast_tp,
    sample_sharded_broadcast_tp,
    sample_sparse_distribution,
    verify_speculative_chain,
)
from cubic_serve.models.parallel_weights import AxisSpan


@contextmanager
def _distributed_all_gather(
    rank: int,
    *,
    world_ranks: tuple[int, ...],
    final_ranks: tuple[int, ...],
    scores: torch.Tensor,
    token_ids: torch.Tensor,
) -> Iterator[tuple[object, object, Mock, Mock]]:
    world = object()
    final_tp = object()

    def group_ranks(group: object) -> list[int]:
        if group is world:
            return list(world_ranks)
        if group is final_tp:
            return list(final_ranks)
        return []

    def all_gather(output: torch.Tensor, value: torch.Tensor, **_: object) -> None:
        source = scores if value.dtype is torch.bfloat16 else token_ids
        output.copy_(source.reshape(-1))

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "cubic_serve.generation.tp_sampling.dist.is_initialized",
                return_value=True,
            )
        )
        stack.enter_context(
            patch("cubic_serve.generation.tp_sampling.dist.get_rank", return_value=rank)
        )
        stack.enter_context(
            patch("cubic_serve.generation.tp_sampling.dist.get_world_size",
                  side_effect=lambda group=None: len(group_ranks(group)))
        )
        stack.enter_context(
            patch(
                "cubic_serve.generation.tp_sampling.dist.get_process_group_ranks",
                side_effect=group_ranks,
            )
        )
        gather_mock = stack.enter_context(
            patch(
                "cubic_serve.generation.tp_sampling.dist.all_gather_single",
                side_effect=all_gather,
            )
        )
        broadcast_mock = stack.enter_context(
            patch("cubic_serve.generation.tp_sampling.dist.broadcast")
        )
        yield world, final_tp, gather_mock, broadcast_mock


def _sampling_workspace(capacity: int, top_k: int, tp: int = 4) -> ShardedSamplingWorkspace:
    return ShardedSamplingWorkspace(
        local_scores=torch.empty((capacity, top_k), dtype=torch.bfloat16),
        local_token_ids=torch.empty((capacity, top_k), dtype=torch.int64),
        gathered_scores=torch.empty((tp, capacity, top_k), dtype=torch.bfloat16),
        gathered_token_ids=torch.empty((tp, capacity, top_k), dtype=torch.int64),
    )


def _sampling_parameters(
    temperatures: list[float], top_ks: list[int], top_ps: list[float]
) -> ShardedSamplingParameters:
    return ShardedSamplingParameters(
        temperatures=torch.tensor(temperatures, dtype=torch.float32),
        top_ks=torch.tensor(top_ks, dtype=torch.int64),
        top_ps=torch.tensor(top_ps, dtype=torch.float32),
        max_top_k=max(top_ks),
    )


def _distribution(token_ids: list[int], probabilities: list[float]) -> SparseSamplingDistribution:
    return SparseSamplingDistribution(
        token_ids=torch.tensor(token_ids, dtype=torch.int64),
        probabilities=torch.tensor(probabilities, dtype=torch.float32),
    )


class TensorParallelSamplingTests(unittest.TestCase):
    def test_candidate_distribution_is_the_one_used_by_root_sampling(self) -> None:
        probabilities = candidate_probabilities(
            torch.tensor([8.0, 7.0, 6.0, 5.0]),
            temperature=0.7,
            top_k=3,
            top_p=0.8,
        )
        expected = torch.softmax(torch.tensor([8.0, 7.0, 6.0]) / 0.7, dim=0)
        expected[(torch.cumsum(expected, 0) - expected) > 0.8] = 0
        expected /= expected.sum()
        torch.testing.assert_close(probabilities, torch.cat((expected, torch.zeros(1))))

    def test_speculative_p_over_q_forces_each_chain_boundary(self) -> None:
        draft = _distribution([1, 2], [0.8, 0.2])
        target = _distribution([1, 2], [0.4, 0.6])
        bonus = _distribution([9], [1.0])
        generator = torch.Generator().manual_seed(191)

        reject_first = verify_speculative_chain(
            draft_token_ids=torch.tensor([1, 1, 1], dtype=torch.int64),
            draft_distributions=(draft, draft, draft),
            target_distributions=(target, target, target, bonus),
            generator=generator,
            acceptance_uniforms=torch.tensor([0.9, 0.1, 0.1]),
        )
        self.assertEqual(reject_first, SpeculativeSamplingDecision(0, 2))

        reject_second = verify_speculative_chain(
            draft_token_ids=torch.tensor([1, 1, 1], dtype=torch.int64),
            draft_distributions=(draft, draft, draft),
            target_distributions=(target, target, target, bonus),
            generator=generator,
            acceptance_uniforms=torch.tensor([0.1, 0.9, 0.1]),
        )
        self.assertEqual(reject_second, SpeculativeSamplingDecision(1, 2))

        reject_third = verify_speculative_chain(
            draft_token_ids=torch.tensor([1, 1, 1], dtype=torch.int64),
            draft_distributions=(draft, draft, draft),
            target_distributions=(target, target, target, bonus),
            generator=generator,
            acceptance_uniforms=torch.tensor([0.1, 0.1, 0.9]),
        )
        self.assertEqual(reject_third, SpeculativeSamplingDecision(2, 2))

        all_accepted = verify_speculative_chain(
            draft_token_ids=torch.tensor([1, 1, 1], dtype=torch.int64),
            draft_distributions=(draft, draft, draft),
            target_distributions=(target, target, target, bonus),
            generator=generator,
            acceptance_uniforms=torch.tensor([0.1, 0.1, 0.1]),
        )
        self.assertEqual(all_accepted, SpeculativeSamplingDecision(3, 9))

    def test_speculative_reference_depth_is_owned_by_the_provider(self) -> None:
        draft = _distribution([1, 2], [0.8, 0.2])
        target = _distribution([1, 2], [0.4, 0.6])
        bonus = _distribution([9], [1.0])
        for depth in (1, 3, 4, 7):
            for accepted in range(depth + 1):
                uniforms = torch.full((depth,), 0.1)
                if accepted < depth:
                    uniforms[accepted] = 0.9
                decision = verify_speculative_chain(
                    draft_token_ids=torch.ones(depth, dtype=torch.int64),
                    draft_distributions=(draft,) * depth,
                    target_distributions=(*(target,) * depth, bonus),
                    generator=torch.Generator().manual_seed(191), acceptance_uniforms=uniforms,
                )
                self.assertEqual(decision, SpeculativeSamplingDecision(
                    accepted, 9 if accepted == depth else 2))

    def test_speculative_sampling_fixed_request_rng_is_rank_consistent(self) -> None:
        draft = _distribution([4, 8, 12], [0.6, 0.3, 0.1])
        target = _distribution([4, 8, 12], [0.2, 0.5, 0.3])
        decisions = []
        states = []
        for _rank in range(4):
            generator = torch.Generator().manual_seed(9701)
            decisions.append(
                verify_speculative_chain(
                    draft_token_ids=torch.tensor([4, 8], dtype=torch.int64),
                    draft_distributions=(draft, draft),
                    target_distributions=(target, target, target),
                    generator=generator,
                )
            )
            states.append(generator.get_state())
        self.assertEqual(decisions, [decisions[0]] * 4)
        self.assertTrue(all(torch.equal(state, states[0]) for state in states))

    def test_speculative_output_distribution_matches_target_only(self) -> None:
        draft = _distribution([0, 1], [0.7, 0.3])
        target = _distribution([0, 1], [0.2, 0.8])
        generator = torch.Generator().manual_seed(4301)
        counts = torch.zeros(2, dtype=torch.int64)
        trials = 5000
        for _ in range(trials):
            proposal = sample_sparse_distribution(draft, generator=generator)
            decision = verify_speculative_chain(
                draft_token_ids=torch.tensor([proposal], dtype=torch.int64),
                draft_distributions=(draft,),
                target_distributions=(target, target),
                generator=generator,
            )
            emitted = proposal if decision.accepted_draft_tokens else decision.next_root_token_id
            counts[emitted] += 1
        observed = counts.float() / trials
        torch.testing.assert_close(observed, target.probabilities, rtol=0, atol=0.025)

    def test_tp4_seeded_top_k_top_p_matches_centralized_sampling(self) -> None:
        ranks = (0, 1, 2, 3)
        spans = tuple(AxisSpan(3 * rank, 3 * rank + 3) for rank in ranks)
        full_logits = torch.tensor(
            [[8.0, 7.0, 1.0, 6.0, 5.0, 0.0, 4.0, 3.0, -1.0, 2.0, -2.0, -3.0]],
            dtype=torch.bfloat16,
        )
        gathered_scores = torch.empty((4, 1, 3), dtype=torch.bfloat16)
        gathered_ids = torch.empty((4, 1, 3), dtype=torch.int64)
        for shard, span in enumerate(spans):
            scores, indices = torch.topk(full_logits[:, span.start : span.stop], 3, dim=-1)
            gathered_scores[shard] = scores
            gathered_ids[shard] = indices + span.start

        actual_generator = torch.Generator().manual_seed(451)
        expected_generator = torch.Generator().manual_seed(451)
        parameters = _sampling_parameters([0.7], [3], [0.8])
        tokens = torch.empty(1, dtype=torch.int64)
        profiler = Mock()
        with _distributed_all_gather(
            0,
            world_ranks=ranks,
            final_ranks=ranks,
            scores=gathered_scores,
            token_ids=gathered_ids,
        ) as (world, final_tp, gather, broadcast):
            result = sample_sharded_broadcast_tp(
                local_logits=full_logits[:, :3].contiguous(),
                local_vocab_span=spans[0],
                final_ranks=ranks,
                final_vocab_spans=spans,
                true_vocab_size=12,
                active_rows=1,
                token_ids=tokens,
                parameters=parameters,
                workspace=_sampling_workspace(1, 3),
                final_tp_group=final_tp,  # type: ignore[arg-type]
                world_group=world,  # type: ignore[arg-type]
                device=torch.device("cpu"),
                request_generators=(actual_generator,),
                profiler=profiler,
            )

        top_scores, top_ids = torch.topk(full_logits[0].float() / 0.7, 3)
        probabilities = torch.softmax(top_scores, dim=-1)
        probabilities[(torch.cumsum(probabilities, 0) - probabilities) > 0.8] = 0
        expected_slot = torch.multinomial(probabilities, 1, generator=expected_generator)
        self.assertIs(result, tokens)
        self.assertEqual(tokens.item(), top_ids[expected_slot].item())
        self.assertEqual(gather.call_count, 2)
        broadcast.assert_called_once()
        flushed = profiler.flush_cuda.call_args.args[0]
        self.assertEqual(
            set(flushed),
            {
                "guard_rank_logits_finite_gpu_ms",
                "rank_local_topk_gpu_ms",
                "rank_greedy_max_gpu_ms",
                "tp_candidate_all_gather_gpu_ms",
                "guard_root_candidates_gpu_ms",
                "root_global_topk_gpu_ms",
                "root_temperature_softmax_gpu_ms",
                "root_top_p_gpu_ms",
                "root_sample_gpu_ms",
                "tp_token_broadcast_gpu_ms",
                "guard_final_token_gpu_ms",
            },
        )

    def test_tp4_temperature_zero_is_exact_and_does_not_advance_rng(self) -> None:
        ranks = (0, 1, 2, 3)
        spans = tuple(AxisSpan(2 * rank, 2 * rank + 2) for rank in ranks)
        gathered_scores = torch.tensor(
            [
                [[9.0, float("nan")]],
                [[9.0, float("nan")]],
                [[8.0, float("nan")]],
                [[6.0, float("nan")]],
            ],
            dtype=torch.bfloat16,
        )
        gathered_ids = torch.tensor(
            [[[0, -1]], [[2, -1]], [[4, -1]], [[6, -1]]], dtype=torch.int64
        )
        generator = torch.Generator().manual_seed(72)
        initial_state = generator.get_state().clone()
        workspace = _sampling_workspace(1, 2)
        with _distributed_all_gather(
            0,
            world_ranks=ranks,
            final_ranks=ranks,
            scores=gathered_scores,
            token_ids=gathered_ids,
        ) as (world, final_tp, _gather, broadcast):
            tokens = sample_sharded_broadcast_tp(
                local_logits=torch.tensor([[9.0, 9.0]], dtype=torch.bfloat16),
                local_vocab_span=spans[0],
                final_ranks=ranks,
                final_vocab_spans=spans,
                true_vocab_size=8,
                active_rows=1,
                token_ids=torch.empty(1, dtype=torch.int64),
                parameters=_sampling_parameters([0.0], [2], [1.0]),
                workspace=workspace,
                final_tp_group=final_tp,  # type: ignore[arg-type]
                world_group=world,  # type: ignore[arg-type]
                device=torch.device("cpu"),
                request_generators=(generator,),
            )

        self.assertEqual(tokens.tolist(), [0])
        self.assertTrue(torch.equal(generator.get_state(), initial_state))
        self.assertEqual(workspace.local_token_ids.tolist(), [[0, -1]])
        broadcast.assert_called_once()

    def test_sharded_sampling_mixed_rows_excludes_captured_padding(self) -> None:
        ranks = (0, 1, 2, 3)
        spans = tuple(AxisSpan(2 * rank, 2 * rank + 2) for rank in ranks)
        scores = torch.tensor(
            [
                [[9.0, float("nan")], [4.0, 3.0], [float("nan"), float("nan")]],
                [[8.0, float("nan")], [7.0, 1.0], [float("nan"), float("nan")]],
                [[6.0, float("nan")], [6.0, 2.0], [float("nan"), float("nan")]],
                [[4.0, float("nan")], [5.0, 0.0], [float("nan"), float("nan")]],
            ],
            dtype=torch.bfloat16,
        )
        ids = torch.tensor(
            [
                [[0, -1], [0, 1], [-1, -1]],
                [[2, -1], [2, 3], [-1, -1]],
                [[4, -1], [4, 5], [-1, -1]],
                [[6, -1], [6, 7], [-1, -1]],
            ],
            dtype=torch.int64,
        )
        with _distributed_all_gather(
            0,
            world_ranks=ranks,
            final_ranks=ranks,
            scores=scores,
            token_ids=ids,
        ) as (world, final_tp, _gather, _broadcast):
            tokens = sample_sharded_broadcast_tp(
                local_logits=torch.tensor(
                    [[9.0, 1.0], [4.0, 3.0], [float("inf"), float("nan")]],
                    dtype=torch.bfloat16,
                ),
                local_vocab_span=spans[0],
                final_ranks=ranks,
                final_vocab_spans=spans,
                true_vocab_size=8,
                active_rows=2,
                token_ids=torch.empty(3, dtype=torch.int64),
                parameters=_sampling_parameters([0.0, 1.0, float("nan")], [2, 2, 2], [1, 1, 0]),
                workspace=_sampling_workspace(3, 2),
                final_tp_group=final_tp,  # type: ignore[arg-type]
                world_group=world,  # type: ignore[arg-type]
                device=torch.device("cpu"),
            )

        self.assertEqual(tokens[0].item(), 0)
        self.assertIn(tokens[1].item(), range(8))
        self.assertEqual(tokens[2].item(), -1)

    def test_sharded_sampling_non_final_rank_only_receives_broadcast(self) -> None:
        world_ranks = tuple(range(8))
        final_ranks = (1, 3, 5, 7)
        spans = tuple(AxisSpan(2 * rank, 2 * rank + 2) for rank in range(4))
        with _distributed_all_gather(
            0,
            world_ranks=world_ranks,
            final_ranks=final_ranks,
            scores=torch.empty((4, 1, 2), dtype=torch.bfloat16),
            token_ids=torch.empty((4, 1, 2), dtype=torch.int64),
        ) as (world, _final_tp, gather, broadcast):
            broadcast.side_effect = lambda target, **_: target.copy_(torch.tensor([6]))
            tokens = sample_sharded_broadcast_tp(
                local_logits=None,
                local_vocab_span=None,
                final_ranks=final_ranks,
                final_vocab_spans=spans,
                true_vocab_size=8,
                active_rows=1,
                token_ids=torch.empty(1, dtype=torch.int64),
                parameters=_sampling_parameters([0.9], [2], [0.95]),
                workspace=None,
                final_tp_group=None,
                world_group=world,  # type: ignore[arg-type]
                device=torch.device("cpu"),
            )

        self.assertEqual(tokens.tolist(), [6])
        gather.assert_not_called()
        broadcast.assert_called_once()

    def test_sharded_sampling_bad_workspace_fails_before_collectives(self) -> None:
        ranks = (0, 1, 2, 3)
        spans = tuple(AxisSpan(2 * rank, 2 * rank + 2) for rank in ranks)
        workspace = _sampling_workspace(1, 2)
        workspace = ShardedSamplingWorkspace(
            local_scores=torch.empty((1, 1), dtype=torch.bfloat16),
            local_token_ids=workspace.local_token_ids,
            gathered_scores=workspace.gathered_scores,
            gathered_token_ids=workspace.gathered_token_ids,
        )
        with (
            _distributed_all_gather(
                0,
                world_ranks=ranks,
                final_ranks=ranks,
                scores=torch.empty((4, 1, 2), dtype=torch.bfloat16),
                token_ids=torch.empty((4, 1, 2), dtype=torch.int64),
            ) as (world, final_tp, gather, broadcast),
            self.assertRaisesRegex(TensorParallelSamplingError, "local candidate scores"),
        ):
            sample_sharded_broadcast_tp(
                local_logits=torch.zeros((1, 2), dtype=torch.bfloat16),
                local_vocab_span=spans[0],
                final_ranks=ranks,
                final_vocab_spans=spans,
                true_vocab_size=8,
                active_rows=1,
                token_ids=torch.empty(1, dtype=torch.int64),
                parameters=_sampling_parameters([1.0], [2], [1.0]),
                workspace=workspace,
                final_tp_group=final_tp,  # type: ignore[arg-type]
                world_group=world,  # type: ignore[arg-type]
                device=torch.device("cpu"),
            )
        gather.assert_not_called()
        broadcast.assert_not_called()

    def test_generic_tp4_all_gathers_candidates_and_resolves_ties_by_token(self) -> None:
        ranks = (0, 1, 2, 3)
        spans = tuple(AxisSpan(2 * rank, 2 * rank + 2) for rank in ranks)
        scores = torch.tensor(
            [[5.0, 1.0, float("nan")], [5.0, 8.0, float("nan")],
             [5.0, 8.0, float("nan")], [4.0, 2.0, float("nan")]],
            dtype=torch.bfloat16,
        )
        candidates = torch.tensor(
            [[0, 0, -1], [2, 3, -1], [4, 4, -1], [6, 6, -1]],
            dtype=torch.int64,
        )
        tokens = torch.empty(3, dtype=torch.int64)
        with _distributed_all_gather(
            0,
            world_ranks=ranks,
            final_ranks=ranks,
            scores=scores,
            token_ids=candidates,
        ) as (world, final_tp, gather, broadcast):
            result = gather_greedy_broadcast_tp(
                local_logits=torch.tensor(
                    [[5.0, 0.0], [1.0, 0.0], [float("nan"), float("nan")]],
                    dtype=torch.bfloat16,
                ),
                local_vocab_span=spans[0],
                final_ranks=ranks,
                final_vocab_spans=spans,
                true_vocab_size=7,
                active_rows=2,
                token_ids=tokens,
                local_max_scores=torch.empty(3, dtype=torch.bfloat16),
                gathered_max_scores=torch.empty((4, 3), dtype=torch.bfloat16),
                gathered_token_ids=torch.empty((4, 3), dtype=torch.int64),
                final_tp_group=final_tp,  # type: ignore[arg-type]
                world_group=world,  # type: ignore[arg-type]
                device=torch.device("cpu"),
            )

        self.assertIs(result, tokens)
        self.assertEqual(tokens.tolist(), [0, 3, -1])
        self.assertEqual(gather.call_count, 2)
        broadcast.assert_not_called()

    def test_generic_tp4_ignores_padded_vocab_columns(self) -> None:
        ranks = (0, 1, 2, 3)
        spans = tuple(AxisSpan(2 * rank, 2 * rank + 2) for rank in ranks)
        scores = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.bfloat16)
        candidates = torch.tensor([[0], [2], [4], [6]], dtype=torch.int64)
        with _distributed_all_gather(
            3,
            world_ranks=ranks,
            final_ranks=ranks,
            scores=scores,
            token_ids=candidates,
        ) as (world, final_tp, _gather, _broadcast):
            tokens = gather_greedy_broadcast_tp(
                local_logits=torch.tensor([[4.0, float("inf")]], dtype=torch.bfloat16),
                local_vocab_span=spans[3],
                final_ranks=ranks,
                final_vocab_spans=spans,
                true_vocab_size=7,
                active_rows=1,
                token_ids=torch.empty(1, dtype=torch.int64),
                local_max_scores=torch.empty(1, dtype=torch.bfloat16),
                gathered_max_scores=torch.empty((4, 1), dtype=torch.bfloat16),
                gathered_token_ids=torch.empty((4, 1), dtype=torch.int64),
                final_tp_group=final_tp,  # type: ignore[arg-type]
                world_group=world,  # type: ignore[arg-type]
                device=torch.device("cpu"),
            )
        self.assertEqual(tokens.tolist(), [6])

    def test_generic_tp4_rejects_nonfinite_real_candidates_and_bad_tail(self) -> None:
        ranks = (0, 1, 2, 3)
        spans = tuple(AxisSpan(2 * rank, 2 * rank + 2) for rank in ranks)
        cases = (
            (
                torch.tensor(
                    [[float("nan")], [3.0], [5.0], [7.0]],
                    dtype=torch.bfloat16,
                ),
                torch.tensor([[0], [2], [4], [6]], dtype=torch.int64),
            ),
            (
                torch.tensor(
                    [[1.0, float("nan")], [2.0, float("nan")],
                     [3.0, float("nan")], [4.0, float("nan")]],
                    dtype=torch.bfloat16,
                ),
                torch.tensor([[0, -1], [2, 99], [4, -1], [6, -1]], dtype=torch.int64),
            ),
        )
        for scores, candidates in cases:
            with (
                self.subTest(candidates=candidates.tolist()),
                _distributed_all_gather(
                    0,
                    world_ranks=ranks,
                    final_ranks=ranks,
                    scores=scores,
                    token_ids=candidates,
                ) as (world, final_tp, _gather, broadcast),
                self.assertRaisesRegex(TensorParallelSamplingError, "invalid"),
            ):
                capacity = scores.shape[1]
                gather_greedy_broadcast_tp(
                    local_logits=torch.tensor(
                        [[float("inf"), 2.0]] * capacity, dtype=torch.bfloat16
                    ),
                    local_vocab_span=spans[0],
                    final_ranks=ranks,
                    final_vocab_spans=spans,
                    true_vocab_size=8,
                    active_rows=1,
                    token_ids=torch.empty(capacity, dtype=torch.int64),
                    local_max_scores=torch.empty(capacity, dtype=torch.bfloat16),
                    gathered_max_scores=torch.empty_like(scores),
                    gathered_token_ids=torch.empty_like(candidates),
                    final_tp_group=final_tp,  # type: ignore[arg-type]
                    world_group=world,  # type: ignore[arg-type]
                    device=torch.device("cpu"),
                )
            broadcast.assert_not_called()

    def test_generic_tp4_broadcasts_to_non_final_pipeline_ranks(self) -> None:
        world_ranks = tuple(range(8))
        final_ranks = (1, 3, 5, 7)
        spans = tuple(AxisSpan(2 * index, 2 * index + 2) for index in range(4))
        scores = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.bfloat16)
        candidates = torch.tensor([[0], [2], [4], [6]], dtype=torch.int64)
        with _distributed_all_gather(
            3,
            world_ranks=world_ranks,
            final_ranks=final_ranks,
            scores=scores,
            token_ids=candidates,
        ) as (world, final_tp, gather, broadcast):
            tokens = gather_greedy_broadcast_tp(
                local_logits=torch.tensor([[2.0, 1.0]], dtype=torch.bfloat16),
                local_vocab_span=spans[1],
                final_ranks=final_ranks,
                final_vocab_spans=spans,
                true_vocab_size=8,
                active_rows=1,
                token_ids=torch.empty(1, dtype=torch.int64),
                local_max_scores=torch.empty(1, dtype=torch.bfloat16),
                gathered_max_scores=torch.empty((4, 1), dtype=torch.bfloat16),
                gathered_token_ids=torch.empty((4, 1), dtype=torch.int64),
                final_tp_group=final_tp,  # type: ignore[arg-type]
                world_group=world,  # type: ignore[arg-type]
                device=torch.device("cpu"),
            )
        self.assertEqual(tokens.tolist(), [6])
        self.assertEqual(gather.call_count, 2)
        self.assertEqual(broadcast.call_args.kwargs["group_src"], 1)

    def test_optional_profiler_observes_sampling_stages_without_changing_tokens(self) -> None:
        class _Profiler:
            def __init__(self) -> None:
                self.events: list[tuple[str, str]] = []

            def begin_cuda(self, metric: str) -> None:
                self.events.append(("begin", metric))

            def end_cuda(self, metric: str) -> None:
                self.events.append(("end", metric))

            def flush_cuda(self, metrics: tuple[str, ...]) -> None:
                self.events.extend(("flush", metric) for metric in metrics)

            def record_cpu(self, metric: str, milliseconds: float) -> None:
                self.assert_nonnegative(milliseconds)
                self.events.append(("cpu", metric))

            @staticmethod
            def assert_nonnegative(value: float) -> None:
                if value < 0:
                    raise AssertionError("negative duration")

        world_ranks = (0, 1, 2, 3)
        final_ranks = (2, 3)
        spans = (AxisSpan(0, 2), AxisSpan(2, 4))
        profiler = _Profiler()
        tokens = torch.empty(1, dtype=torch.int64)
        scores = torch.tensor([[3.0], [2.0]], dtype=torch.bfloat16)
        candidates = torch.tensor([[0], [3]], dtype=torch.int64)
        with _distributed_all_gather(
            2,
            world_ranks=world_ranks,
            final_ranks=final_ranks,
            scores=scores,
            token_ids=candidates,
        ) as (world, final_tp, _gather, _broadcast):
            result = gather_greedy_broadcast_tp(
                local_logits=torch.tensor([[3.0, 1.0]], dtype=torch.bfloat16),
                local_vocab_span=spans[0],
                final_ranks=final_ranks,
                final_vocab_spans=spans,
                true_vocab_size=4,
                active_rows=1,
                token_ids=tokens,
                local_max_scores=torch.empty(1, dtype=torch.bfloat16),
                gathered_max_scores=torch.empty((2, 1), dtype=torch.bfloat16),
                gathered_token_ids=torch.empty((2, 1), dtype=torch.int64),
                final_tp_group=final_tp,  # type: ignore[arg-type]
                world_group=world,  # type: ignore[arg-type]
                device=torch.device("cpu"),
                profiler=profiler,
            )

        self.assertEqual(result.tolist(), [0])
        self.assertIn(("cpu", "validity_host_sync_cpu_ms"), profiler.events)
        self.assertIn(("flush", "local_greedy_max_gpu_ms"), profiler.events)
        self.assertIn(("flush", "final_tp_all_gather_gpu_ms"), profiler.events)
        self.assertIn(("flush", "world_broadcast_gpu_ms"), profiler.events)


if __name__ == "__main__":
    unittest.main()


class DeferredProposalGuardTests(unittest.TestCase):
    def test_deferred_guard_preserves_root_rejection_for_a_legal_token(self):
        class Root:
            valid = True

            def launch(self, scores, ids, temperatures, top_ps, top_ks, seeds, ordinals,
                       output_tokens, output_valid):
                output_tokens.fill_(1)
                output_valid.fill_(self.valid)

        root = Root()
        parameters = ShardedSamplingParameters(
            torch.ones(1), torch.full((1,), 2, dtype=torch.int64), torch.ones(1), 2,
            torch.ones(1, dtype=torch.int64), torch.zeros(1, dtype=torch.int64),
        )
        scores = torch.tensor([[[2., 1.]]], dtype=torch.bfloat16)
        ids = torch.tensor([[[0, 1]]], dtype=torch.int64)
        with _distributed_all_gather(0, world_ranks=(0,), final_ranks=(0,),
                                     scores=scores, token_ids=ids) as (world, tp, _, _):
            kwargs = dict(
                local_logits=scores[0].clone(), local_vocab_span=AxisSpan(0, 2),
                final_ranks=(0,), final_vocab_spans=(AxisSpan(0, 2),), true_vocab_size=2,
                active_rows=1, token_ids=torch.empty(1, dtype=torch.int64),
                parameters=parameters, workspace=_sampling_workspace(1, 2, tp=1),
                final_tp_group=tp, world_group=world, device=torch.device('cpu'),
                root_sampler=root, root_valid=torch.empty(1, dtype=torch.bool),
                all_random_rows=True,
            )
            witnesses = []
            for valid in (True, False, True):
                root.valid = valid
                witness = torch.empty((), dtype=torch.bool)
                output = sample_sharded_broadcast_tp(**kwargs, validity_out=witness)
                self.assertEqual(output.tolist(), [1])
                witnesses.append(witness)
            self.assertEqual([bool(value) for value in witnesses], [True, False, True])
            root.valid = False
            with self.assertRaisesRegex(TensorParallelSamplingError, 'invalid'):
                sample_sharded_broadcast_tp(**kwargs)
            with self.assertRaisesRegex(TensorParallelSamplingError, 'private'):
                sample_sharded_broadcast_tp(**kwargs, validity_out=kwargs['root_valid'][0])


def test_caller_graph_metadata_write_invalidates_cached_sampler_ingress():
    """A reduced proposal wave must not inherit a caller graph's zero count."""
    from types import SimpleNamespace
    from cubic_serve.ops.cuda.sampling.greedy import CapturedGreedySampler

    plan = CapturedGreedySampler.__new__(CapturedGreedySampler)
    plan._closed = False
    plan.capacity = 2
    plan.rank = plan.root_rank = 0
    plan.metadata = torch.tensor([0, 0], dtype=torch.int64)
    plan._metadata_hosts = torch.tensor([
        [[0, 0], [0, 1]], [[1, 0], [1, 1]], [[2, 0], [2, 1]]
    ], dtype=torch.int32)
    plan._metadata_key = (2, False)
    plan._call = {"token_ids": torch.full((2,), -1, dtype=torch.int64)}
    plan.final_valid = torch.tensor(True)

    def sample():
        count = int(plan.metadata[0])
        plan._call["token_ids"].fill_(-1)
        plan._call["token_ids"][:count] = torch.arange(count) + 11

    plan.graphs = (SimpleNamespace(replay=sample), SimpleNamespace(replay=lambda: None))
    plan.invalidate_metadata()
    assert plan._replay(2, False).tolist() == [11, 12]
    assert plan.metadata.tolist() == [2, 0]
    # Returning to the same count after a different caller-graph replay also
    # requires fresh ingress; this is independent of token or model geometry.
    plan.metadata[0] = 1
    plan.invalidate_metadata()
    assert plan._replay(2, False).tolist() == [11, 12]
