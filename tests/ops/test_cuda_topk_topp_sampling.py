from __future__ import annotations

import math
import os
import unittest

import torch
import triton  # type: ignore[import-untyped]
from torch import Tensor

from cubic_serve.ops.cuda.sampling import RootTopKTopPSamplingPrimitive

RUN_REAL = os.environ.get("CUBIC_RUN_CUDA_TOPK_TOPP_TESTS") == "1"


def _oracle(
    scores: list[float],
    token_ids: list[int],
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    uniform: float,
) -> tuple[int, bool]:
    """Small independent oracle; ``uniform`` makes categorical choice explicit."""

    candidates: list[tuple[float, int]] = []
    for score, token_id in zip(scores, token_ids, strict=True):
        if math.isnan(score) and token_id == -1:
            continue
        if not math.isfinite(score) or token_id < 0:
            return -1, False
        candidates.append((score, token_id))
    if (
        not math.isfinite(temperature)
        or temperature < 0.0
        or not math.isfinite(top_p)
        or not 0.0 < top_p <= 1.0
        or top_k < 1
        or not candidates
        or (temperature > 0.0 and top_k > len(candidates))
        or not 0.0 <= uniform < 1.0
    ):
        return -1, False
    ordered = sorted(candidates, key=lambda item: (-item[0], item[1]))[:top_k]
    if temperature == 0.0:
        return ordered[0][1], True
    maximum = ordered[0][0] / temperature
    weights = [math.exp(score / temperature - maximum) for score, _ in ordered]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    kept: list[tuple[int, float]] = []
    exclusive = 0.0
    for probability, (_, token_id) in zip(probabilities, ordered, strict=True):
        if exclusive <= top_p:
            kept.append((token_id, probability))
        exclusive += probability
    target = uniform * sum(probability for _, probability in kept)
    cumulative = 0.0
    for token_id, probability in kept:
        cumulative += probability
        if cumulative > target:
            return token_id, True
    raise AssertionError("uniform must select a retained token")


class RootTopKTopPSamplingOracleTests(unittest.TestCase):
    def test_exact_greedy_tie_uses_smallest_token_id(self) -> None:
        self.assertEqual(
            _oracle([2.0, 4.0, 4.0], [7, 19, 3], temperature=0.0, top_p=0.2, top_k=2, uniform=0.9),
            (3, True),
        )
        self.assertEqual(
            _oracle([4.0, math.nan], [7, -1], temperature=0.0, top_p=1.0, top_k=20, uniform=0.0),
            (7, True),
        )

    def test_temperature_top_k_then_top_p(self) -> None:
        arguments = dict(
            scores=[3.0, 2.0, 1.0, math.nan],
            token_ids=[30, 20, 10, -1],
            temperature=1.0,
            top_k=3,
        )
        self.assertEqual(_oracle(**arguments, top_p=0.5, uniform=0.99), (30, True))
        self.assertEqual(_oracle(**arguments, top_p=0.7, uniform=0.8), (20, True))

    def test_invalid_and_empty_sentinels_fail_closed(self) -> None:
        parameters = dict(temperature=0.0, top_p=1.0, top_k=1, uniform=0.0)
        self.assertEqual(_oracle([math.nan], [-1], **parameters), (-1, False))
        self.assertEqual(_oracle([math.nan, 2.0], [4, 7], **parameters), (-1, False))

    def test_rejects_cpu_storage(self) -> None:
        scores = torch.empty((4, 1, 20), dtype=torch.bfloat16)

        def rows(dtype: torch.dtype) -> Tensor:
            return torch.empty((1,), dtype=dtype)

        with self.assertRaisesRegex(ValueError, "contiguous CUDA"):
            RootTopKTopPSamplingPrimitive(
                scores,
                torch.empty_like(scores, dtype=torch.int64),
                rows(torch.float32),
                rows(torch.float32),
                rows(torch.int64),
                rows(torch.int64),
                rows(torch.int64),
                rows(torch.int64),
                rows(torch.bool),
            )


@unittest.skipUnless(RUN_REAL, "set CUBIC_RUN_CUDA_TOPK_TOPP_TESTS=1 for CUDA tests")
class RootTopKTopPSamplingCudaTests(unittest.TestCase):
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    def test_grammar_underfill_is_owned_per_row_and_cannot_relax_neighbor_guards(self) -> None:
        operands = self._case(3)
        operands[0].fill_(float("nan"))
        operands[1].fill_(-1)
        operands[0][0, :, 0] = 2.0
        operands[1][0, :, 0] = torch.tensor([7, 8, 9], device=self.device)
        flags = torch.tensor([True, False, True], device=self.device)
        primitive = RootTopKTopPSamplingPrimitive(
            *operands, allow_underfilled=True, constrained_rows=flags
        )
        primitive.launch(*operands, constrained_rows=flags)
        self.assertEqual(operands[7].tolist(), [7, -1, 9])
        self.assertEqual(operands[8].tolist(), [True, False, True])
        flags.copy_(torch.tensor([False, True, True], device=self.device))
        operands[1][1, 2, 0] = 17  # A malformed NaN/id pair remains an error.
        primitive.launch(*operands, constrained_rows=flags)
        self.assertEqual(operands[7].tolist(), [-1, 8, -1])
        self.assertEqual(operands[8].tolist(), [False, True, False])

    def _case(self, rows: int) -> tuple[Tensor, ...]:
        tp, candidates_per_rank = 4, 20
        scores = torch.randn(
            (tp, rows, candidates_per_rank), device=self.device, dtype=self.dtype
        )
        token_ids = (
            torch.arange(tp * candidates_per_rank, device=self.device, dtype=torch.int64)
            .view(tp, 1, candidates_per_rank)
            .expand(tp, rows, candidates_per_rank)
            .contiguous()
        )
        return (
            scores,
            token_ids,
            torch.full((rows,), 0.8, device=self.device, dtype=torch.float32),
            torch.full((rows,), 0.9, device=self.device, dtype=torch.float32),
            torch.full((rows,), 20, device=self.device, dtype=torch.int64),
            torch.arange(rows, device=self.device, dtype=torch.int64) + 0x123456789,
            torch.zeros(rows, device=self.device, dtype=torch.int64),
            torch.empty(rows, device=self.device, dtype=torch.int64),
            torch.empty(rows, device=self.device, dtype=torch.bool),
        )

    def test_greedy_tie_and_invalid_sentinel(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        operands = list(self._case(3))
        scores, token_ids = operands[:2]
        scores.fill_(math.nan)
        token_ids.fill_(-1)
        scores[0, 0, :2] = torch.tensor([4.0, 4.0], device=self.device)
        token_ids[0, 0, :2] = torch.tensor([19, 3], device=self.device)
        scores[0, 1, 0] = 2.0
        token_ids[0, 1, 0] = 7
        token_ids[1, 1, 0] = 8  # Mismatched NaN/id sentinel invalidates row 1.
        operands[2].zero_()
        operands[4].fill_(20)
        primitive = RootTopKTopPSamplingPrimitive(*operands)
        primitive.launch(*operands)
        torch.cuda.synchronize(self.device)
        self.assertEqual(operands[7].cpu().tolist(), [3, -1, -1])
        self.assertEqual(operands[8].cpu().tolist(), [True, False, False])
        self.assertFalse(any(isinstance(value, Tensor) for value in vars(primitive).values()))

    def test_seed_and_ordinal_are_deterministic(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        operands = list(self._case(128))
        operands[0].zero_()
        operands[4].fill_(2)
        primitive = RootTopKTopPSamplingPrimitive(*operands)
        primitive.launch(*operands)
        first = operands[7].clone()
        primitive.launch(*operands)
        torch.cuda.synchronize(self.device)
        self.assertTrue(torch.equal(first, operands[7]))
        self.assertTrue(bool(operands[8].all()))
        operands[6].add_(1)
        primitive.launch(*operands)
        torch.cuda.synchronize(self.device)
        self.assertTrue(bool(torch.any(first != operands[7])))
        self.assertTrue(bool(torch.all((operands[7] == 0) | (operands[7] == 1))))

    def test_global_top_k_temperature_softmax_and_top_p_support(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        operands = list(self._case(4))
        scores, token_ids = operands[:2]
        scores.fill_(math.nan)
        token_ids.fill_(-1)
        scores[:, :, :2] = torch.tensor(
            [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], device=self.device
        ).view(4, 1, 2)
        token_ids[:, :, :2] = torch.arange(8, device=self.device).view(4, 1, 2)
        operands[2].fill_(1.0)
        operands[3].fill_(0.01)
        operands[4].fill_(4)
        primitive = RootTopKTopPSamplingPrimitive(*operands)
        primitive.launch(*operands)
        torch.cuda.synchronize(self.device)
        self.assertEqual(operands[7].cpu().tolist(), [0, 0, 0, 0])
        self.assertTrue(bool(operands[8].all()))

    def test_fixed_abi_and_benchmark_c1_c256(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        timings: dict[int, float] = {}
        for rows in (1, 256):
            operands = self._case(rows)
            primitive = RootTopKTopPSamplingPrimitive(*operands)
            primitive.launch(*operands)
            torch.cuda.synchronize(self.device)

            def launch(
                selected: RootTopKTopPSamplingPrimitive = primitive,
                buffers: tuple[Tensor, ...] = operands,
            ) -> None:
                selected.launch(*buffers)

            timings[rows] = 1000.0 * triton.testing.do_bench(
                launch, warmup=25, rep=100, return_mode="median"
            )
        print(f"root top-k/top-p kernel: C1={timings[1]:.3f} us C256={timings[256]:.3f} us")
        with self.assertRaisesRegex(ValueError, "fixed launch ABI"):
            primitive.launch(*(tensor[:128] if tensor.ndim > 0 else tensor for tensor in operands))


class FP16RootTopKTopPSamplingCudaTests(RootTopKTopPSamplingCudaTests):
    dtype = torch.float16


if __name__ == "__main__":
    unittest.main()
