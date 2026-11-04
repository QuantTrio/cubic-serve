from __future__ import annotations

import math
import os
import random
import unittest

import torch

from cubic_serve.ops.cuda.sampling.speculative import verify_one_draft_candidates

RUN_REAL = os.environ.get("CUBIC_RUN_CUDA_SPECULATIVE_TESTS") == "1"


def _distribution(pairs: tuple[tuple[float, int], ...], temperature: float) -> dict[int, float]:
    ordered = sorted(pairs, key=lambda item: (-item[0], item[1]))
    if temperature == 0.0:
        return {ordered[0][1]: 1.0}
    maximum = ordered[0][0] / temperature
    weights = tuple(math.exp(score / temperature - maximum) for score, _ in ordered)
    total = sum(weights)
    return {
        token_id: weight / total for weight, (_, token_id) in zip(weights, ordered, strict=True)
    }


def _sample(probabilities: dict[int, float], uniform: float) -> int:
    cumulative = 0.0
    for token_id, probability in probabilities.items():
        cumulative += probability
        if cumulative > uniform:
            return token_id
    raise AssertionError("normalized distribution did not select a token")


def _oracle(
    q: tuple[tuple[float, int], ...],
    p0: tuple[tuple[float, int], ...],
    p1: tuple[tuple[float, int], ...],
    *,
    draft: int,
    temperature: float,
    accept_uniform: float,
    sample_uniform: float,
) -> tuple[int, int]:
    q_probability = _distribution(q, temperature)
    p0_probability = _distribution(p0, temperature)
    p1_probability = _distribution(p1, temperature)
    if temperature == 0.0:
        accepted = draft == next(iter(p0_probability))
    else:
        accepted = accept_uniform < min(1.0, p0_probability.get(draft, 0.0) / q_probability[draft])
    if accepted:
        return 1, _sample(p1_probability, sample_uniform)
    residual = {
        token_id: max(0.0, p0_probability.get(token_id, 0.0) - q_probability.get(token_id, 0.0))
        for token_id in p0_probability.keys() | q_probability.keys()
    }
    mass = sum(residual.values())
    normalized = {
        token_id: probability / mass
        for token_id, probability in sorted(residual.items())
        if probability > 0.0
    }
    return 0, _sample(normalized, sample_uniform)


@unittest.skipUnless(
    RUN_REAL,
    "set CUBIC_RUN_CUDA_SPECULATIVE_TESTS=1 for CUDA speculative verification",
)
class SpeculativeVerificationCudaTests(unittest.TestCase):
    def test_grammar_underfill_never_relaxes_unconstrained_request_guard(self) -> None:
        device = torch.device("cuda:0")
        scores = torch.full((1, 2, 20), math.nan, dtype=torch.bfloat16, device=device)
        ids = torch.full((1, 2, 20), -1, dtype=torch.int64, device=device)
        target = torch.full((1, 4, 20), math.nan, dtype=torch.bfloat16, device=device)
        target_ids = torch.full((1, 4, 20), -1, dtype=torch.int64, device=device)
        scores[:, :, 0] = target[:, :, 0] = 1.0
        ids[:, :, 0] = target_ids[:, :, 0] = 3
        decisions = torch.empty((2, 2), dtype=torch.int64, device=device)
        verify_one_draft_candidates(
            q_scores=scores,
            q_token_ids=ids,
            p_scores=target,
            p_token_ids=target_ids,
            target_rows=torch.tensor([0, 2], dtype=torch.int32, device=device),
            draft_tokens=torch.full((2,), 3, dtype=torch.int64, device=device),
            temperatures=torch.ones(2, device=device),
            top_ps=torch.ones(2, device=device),
            top_ks=torch.full((2,), 20, dtype=torch.int64, device=device),
            seeds=torch.ones(2, dtype=torch.int64, device=device),
            ordinals=torch.zeros(2, dtype=torch.int64, device=device),
            decisions=decisions,
            active_rows=2,
            allow_underfilled=True,
            constrained_rows=torch.tensor([True, False], device=device),
            uniforms=torch.full((2, 2), 0.5, device=device),
        )
        self.assertEqual(decisions.tolist(), [[1, 3], [-1, -1]])

    def test_owned_failure_is_not_padding_when_other_shards_supply_topk(self) -> None:
        device = torch.device("cuda:0")
        for source in ("q", "p0", "p1"):
            for score, token in ((math.nan, -2), (math.nan, 9),
                                 (math.inf, 9), (-math.inf, 9)):
                q = torch.full((4, 4, 20), math.nan, device=device, dtype=torch.bfloat16)
                qi = torch.full_like(q, -1, dtype=torch.int64)
                p = torch.full((4, 8, 20), math.nan, device=device, dtype=torch.bfloat16)
                pi = torch.full_like(p, -1, dtype=torch.int64)
                q[0, :3, 0], qi[0, :3, 0] = 2, 10
                p[0, :, 0] = 2
                pi[0, ::2, 0], pi[0, 1::2, 0] = 10, 11
                physical_rows = (4, 2, 0)
                scores, ids, row = ((q, qi, 1) if source == "q" else
                                   (p, pi, physical_rows[1] + (source == "p1")))
                scores[3, row, 0], ids[3, row, 0] = score, token
                decisions = torch.empty((4, 2), device=device, dtype=torch.int64)
                verify_one_draft_candidates(
                    q_scores=q, q_token_ids=qi, p_scores=p, p_token_ids=pi,
                    target_rows=torch.tensor((*physical_rows, 0), device=device, dtype=torch.int32),
                    draft_tokens=torch.full((4,), 10, device=device, dtype=torch.int64),
                    temperatures=torch.ones(4, device=device), top_ps=torch.ones(4, device=device),
                    top_ks=torch.ones(4, device=device, dtype=torch.int64),
                    seeds=torch.ones(4, device=device, dtype=torch.int64),
                    ordinals=torch.zeros(4, device=device, dtype=torch.int64),
                    decisions=decisions, active_rows=3,
                    uniforms=torch.full((4, 2), .5, device=device),
                )
                self.assertEqual(decisions.cpu().tolist(),
                                 [[1, 11], [-1, -1], [1, 11], [-1, -1]])

    def test_frozen_probabilistic_draws_across_tp_padding_and_row_maps(self) -> None:
        """Reference the same ordered CDF, with non-degenerate recovery mass.

        Operands are exact BF16 numbers and draws stay away from decision
        boundaries. This tests decisions, not a tolerance on model logits.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        device = torch.device("cuda:0")
        rng = random.Random(8149)
        for tp in (1, 2, 4):
            for capacity in (1, 4, 16):
                active = max(1, capacity - 1)
                for reverse in (False, True):
                    q = torch.full((tp, capacity, 20), math.nan, dtype=torch.bfloat16)
                    qi = torch.full_like(q, -1, dtype=torch.int64)
                    p = torch.full((tp, 2 * capacity, 20), math.nan, dtype=torch.bfloat16)
                    pi = torch.full_like(p, -1, dtype=torch.int64)
                    expected, drafts, draws, temperatures, top_ps = [], [], [], [], []
                    target_rows = list(range(0, 2 * active, 2))
                    if reverse:
                        target_rows.reverse()
                    for row, target_row in enumerate(target_rows):
                        ids = list(range(8))
                        rng.shuffle(ids)
                        distributions = []
                        temperature = (0.5, 1.0, 2.0)[row % 3]
                        top_p = (0.6, 0.9, 1.0)[row % 3]
                        for phase in range(3):
                            pairs = tuple((rng.randrange(-12, 13) / 4, token)
                                          for token in ids)
                            ordered = sorted(pairs, key=lambda item: (-item[0], item[1]))
                            probabilities = _distribution(tuple(ordered[:6]), temperature)
                            mass = 0.0
                            retained = {}
                            for token, probability in probabilities.items():
                                if mass <= top_p:
                                    retained[token] = probability
                                mass += probability
                            total = sum(retained.values())
                            distributions.append({token: value / total
                                                  for token, value in retained.items()})
                            scores, tokens = (q, qi) if phase == 0 else (p, pi)
                            physical = row if phase == 0 else target_row + phase - 1
                            for ordinal, (score, token) in enumerate(pairs):
                                rank, local = ordinal % tp, ordinal // tp
                                scores[rank, physical, local] = score
                                tokens[rank, physical, local] = token
                        q_prob, p0_prob, p1_prob = distributions
                        draft = next(iter(q_prob))
                        ratio = min(1.0, p0_prob.get(draft, 0.0) / q_prob[draft])
                        accept_draw = 0.125 if ratio > 0.25 else 0.875
                        accepted = accept_draw < ratio
                        distribution = p1_prob if accepted else {
                            token: max(0.0, probability - q_prob.get(token, 0.0))
                            for token, probability in p0_prob.items()
                        }
                        total = sum(distribution.values())
                        self.assertGreater(total, 0)
                        probabilities = {token: probability / total
                                         for token, probability in distribution.items()}
                        # Midpoint of a positive CDF interval avoids equality
                        # ambiguity while witnessing every residual ordering.
                        token, probability = max(probabilities.items(), key=lambda x: x[1])
                        cumulative = 0.0
                        for candidate, value in probabilities.items():
                            if candidate == token:
                                break
                            cumulative += value
                        sample_draw = cumulative + probability / 2
                        expected.append((int(accepted), _sample(probabilities, sample_draw)))
                        draws.append((accept_draw, sample_draw))
                        drafts.append(draft)
                        temperatures.append(temperature)
                        top_ps.append(top_p)
                    decisions = torch.full((capacity, 2), 713, device=device, dtype=torch.int64)
                    def cuda(values, dtype):
                        return torch.tensor(values, device=device, dtype=dtype)
                    verify_one_draft_candidates(
                        q_scores=q.to(device), q_token_ids=qi.to(device),
                        p_scores=p.to(device), p_token_ids=pi.to(device),
                        target_rows=cuda(target_rows + [0] * (capacity - active), torch.int32),
                        draft_tokens=cuda(drafts + [0] * (capacity - active), torch.int64),
                        temperatures=cuda(temperatures + [1.] * (capacity - active), torch.float32),
                        top_ps=cuda(top_ps + [1.] * (capacity - active), torch.float32),
                        top_ks=torch.full((capacity,), 6, device=device, dtype=torch.int64),
                        seeds=torch.zeros(capacity, device=device, dtype=torch.int64),
                        ordinals=torch.zeros(capacity, device=device, dtype=torch.int64),
                        decisions=decisions, active_rows=active,
                        uniforms=cuda(draws + [(0., 0.)] * (capacity - active), torch.float32),
                    )
                    self.assertEqual(tuple(map(tuple, decisions[:active].cpu().tolist())),
                                     tuple(expected))
                    # Inactive verification lanes return invalid decisions;
                    # poisoned candidate payload never becomes a live owner.
                    self.assertTrue(bool((decisions[active:] == -1).all()))

    def test_k1_decisions_match_independent_eager_oracle(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        device = torch.device("cuda:0")
        tp, rows, candidates, target_rows_count = 4, 4, 20, 8
        q_scores = torch.full((tp, rows, candidates), math.nan, device=device, dtype=torch.bfloat16)
        q_ids = torch.full_like(q_scores, -1, dtype=torch.int64)
        p_scores = torch.full(
            (tp, target_rows_count, candidates),
            math.nan,
            device=device,
            dtype=torch.bfloat16,
        )
        p_ids = torch.full_like(p_scores, -1, dtype=torch.int64)
        q_rows = (
            ((2.0, 10), (0.0, 11)),
            ((2.0, 10), (0.0, 11)),
            ((2.0, 10), (0.0, 11)),
            ((0.0, 10), (1.0, 11)),
        )
        p0_rows = (
            ((2.0, 10), (0.0, 11)),
            ((0.0, 10), (2.0, 11)),
            ((1.0, 10), (1.0, 11)),
            ((2.0, 10), (0.0, 11)),
        )
        p1_rows = tuple(((0.0, 90 + row * 2), (2.0, 91 + row * 2)) for row in range(rows))
        target_rows = torch.arange(0, target_rows_count, 2, device=device, dtype=torch.int32)
        for row in range(rows):
            for rank, pair in enumerate(q_rows[row]):
                q_scores[rank, row, 0] = pair[0]
                q_ids[rank, row, 0] = pair[1]
            for rank, pair in enumerate(p0_rows[row]):
                p_scores[rank, 2 * row, 0] = pair[0]
                p_ids[rank, 2 * row, 0] = pair[1]
            for rank, pair in enumerate(p1_rows[row]):
                p_scores[rank, 2 * row + 1, 0] = pair[0]
                p_ids[rank, 2 * row + 1, 0] = pair[1]

        temperatures = torch.tensor((0.0, 0.0, 1.0, 1.0), device=device)
        top_ps = torch.ones(rows, device=device)
        top_ks = torch.full((rows,), 2, device=device, dtype=torch.int64)
        seeds = torch.arange(rows, device=device, dtype=torch.int64)
        ordinals = torch.zeros(rows, device=device, dtype=torch.int64)
        uniforms = torch.tensor(
            ((0.5, 0.2), (0.5, 0.2), (0.9, 0.2), (0.2, 0.2)),
            device=device,
            dtype=torch.float32,
        )
        decisions = torch.empty((rows, 2), device=device, dtype=torch.int64)
        verify_one_draft_candidates(
            q_scores=q_scores,
            q_token_ids=q_ids,
            p_scores=p_scores,
            p_token_ids=p_ids,
            target_rows=target_rows,
            draft_tokens=torch.full((rows,), 10, device=device, dtype=torch.int64),
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            seeds=seeds,
            ordinals=ordinals,
            decisions=decisions,
            active_rows=rows,
            uniforms=uniforms,
        )
        expected = tuple(
            _oracle(
                q_rows[row],
                p0_rows[row],
                p1_rows[row],
                draft=10,
                temperature=float(temperatures[row].item()),
                accept_uniform=float(uniforms[row, 0].item()),
                sample_uniform=float(uniforms[row, 1].item()),
            )
            for row in range(rows)
        )
        torch.cuda.synchronize(device)
        self.assertEqual(tuple(map(tuple, decisions.cpu().tolist())), expected)


if __name__ == "__main__":
    unittest.main()
