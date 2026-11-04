"""Leaf audit callbacks preserve operands and defer verifier work."""

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "qwen3_5"))
from tp2_pp2_leaf_audit import _DeferredObserver


@dataclass(frozen=True)
class _Event:
    global_layer: int
    weight_name: str
    linear_input: torch.Tensor
    local_weight: torch.Tensor
    local_output: torch.Tensor
    tp_partial: torch.Tensor | None = None
    tp_reduced: torch.Tensor | None = None


class DeferredLeafObserverTests(unittest.TestCase):
    def test_reference_waits_for_finish_and_activations_are_immutable(self):
        consumed = []
        observer = _DeferredObserver(lambda event, ordinal: consumed.append((event, ordinal)))
        x = torch.tensor([[1.0, 2.0]])
        weight = torch.tensor([[3.0, 4.0]])
        output = torch.tensor([[11.0]])
        event = _Event(0, "projection", x, weight, output)
        observer(event)
        self.assertFalse(consumed)
        self.assertEqual(observer.schedule, ((0, "projection", False),))
        x.zero_()
        output.zero_()
        observer.finish(1)
        saved, ordinal = consumed[0]
        self.assertEqual(ordinal, 0)
        self.assertTrue(torch.equal(saved.linear_input, torch.tensor([[1.0, 2.0]])))
        self.assertTrue(torch.equal(saved.local_output, torch.tensor([[11.0]])))
        self.assertIs(saved.local_weight, weight)

    def test_invalid_event_count_does_not_run_reference(self):
        consumed = []
        observer = _DeferredObserver(lambda *args: consumed.append(args))
        with self.assertRaisesRegex(RuntimeError, "expected 1"):
            observer.finish(1)
        self.assertFalse(consumed)


if __name__ == "__main__":
    unittest.main()
