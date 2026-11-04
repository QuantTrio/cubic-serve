from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import torch

from cubic_serve.generation import GreedySampler
from cubic_serve.generation.constraints import (
    BackendTokenMask,
    ConstraintLifecycleError,
    ConstraintMatcher,
    ConstraintSpec,
    create_logit_constraint,
)
from cubic_serve.generation.sampler import load_eos_token_ids


class _Matcher:
    def __init__(
        self,
        digest: str,
        events: list[str],
        *,
        allowed: tuple[int, ...] = (1,),
        fail_apply: bool = False,
    ) -> None:
        self._digest = digest
        self._events = events
        self._allowed = allowed
        self._fail_apply = fail_apply

    @property
    def constraint_digest(self) -> str:
        return self._digest

    def fill_token_mask(self) -> BackendTokenMask:
        self._events.append("mask")
        return BackendTokenMask(self._allowed, True)

    def apply_token_mask(self, logits: torch.Tensor, payload: object) -> None:
        self._events.append("apply")
        if self._fail_apply:
            raise RuntimeError("backend apply failed")
        allowed = payload
        if type(allowed) is not tuple:
            raise TypeError("test mask must be a tuple")
        for token_id in range(logits.numel()):
            if token_id not in allowed:
                logits[token_id] = -torch.inf

    def accept_token(self, token_id: int) -> bool:
        self._events.append(f"accept:{token_id}")
        return token_id in self._allowed

    def is_terminated(self) -> bool:
        return False

    def fork(self) -> ConstraintMatcher:
        return _Matcher(self._digest, self._events, allowed=self._allowed)


@dataclass(slots=True)
class _Factory:
    events: list[str]
    allowed: tuple[int, ...] = (1,)
    fail_apply: bool = False
    backend_id: str = "test"

    def require_ready(self, spec: ConstraintSpec) -> None:
        return None

    def create_matcher(self, spec: ConstraintSpec) -> ConstraintMatcher:
        return _Matcher(
            spec.digest,
            self.events,
            allowed=self.allowed,
            fail_apply=self.fail_apply,
        )


class EosMetadataTests(unittest.TestCase):
    def test_collects_generation_and_nested_model_eos_without_losing_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(load_eos_token_ids(root, tokenizer_eos=9), (9,))
            (root / "config.json").write_text(json.dumps({"text_config": {"eos_token_id": 7}}))
            (root / "generation_config.json").write_text(json.dumps({"eos_token_id": [8, 9, 8]}))
            self.assertEqual(load_eos_token_ids(root, tokenizer_eos=9), (7, 8, 9))
            (root / "generation_config.json").write_text(json.dumps({"eos_token_id": 8}))
            self.assertEqual(load_eos_token_ids(root, tokenizer_eos=9), (7, 8, 9))

    def test_rejects_malformed_eos_instead_of_silently_dropping_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for value in (True, -1, [9, False], [9, -1], "9", [1.5]):
                with self.subTest(value=value):
                    (root / "generation_config.json").write_text(
                        json.dumps({"eos_token_id": value})
                    )
                    with self.assertRaisesRegex(ValueError, "eos_token_id"):
                        load_eos_token_ids(root, tokenizer_eos=9)


class GreedySamplerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sampler = GreedySampler()

    def test_unconstrained_selection_and_validation_are_unchanged(self) -> None:
        self.assertEqual(self.sampler.sample(torch.tensor([1.0, 3.0, 3.0])), 1)
        with self.assertRaisesRegex(ValueError, "rank-1"):
            self.sampler.sample(torch.zeros((1, 2)))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.sampler.sample(torch.tensor([0.0, torch.inf]))

    def test_identity_constraint_matches_plain_selection_exactly(self) -> None:
        logits = torch.tensor([1.0, 4.0, 4.0], dtype=torch.bfloat16)
        before = logits.clone()
        expected = self.sampler.sample(logits)
        constraint = create_logit_constraint(
            spec=ConstraintSpec.identity(),
            request_id="identity",
        )

        self.assertEqual(self.sampler.sample(logits, constraint), expected)
        self.assertTrue(torch.equal(logits, before))
        self.assertEqual(constraint.accepted_token_ids, (expected,))

    def test_constraint_masks_private_copy_then_accepts_selected_token(self) -> None:
        events: list[str] = []
        constraint = create_logit_constraint(
            spec=ConstraintSpec.json_schema(True),
            request_id="structured",
            factory=_Factory(events, allowed=(1, 2)),
        )
        logits = torch.tensor([100.0, 4.0, 8.0])
        before = logits.clone()

        token_id = self.sampler.sample(logits, constraint)

        self.assertEqual(token_id, 2)
        self.assertEqual(events, ["mask", "apply", "accept:2"])
        self.assertTrue(torch.equal(logits, before))
        self.assertEqual(constraint.accepted_token_ids, (2,))

    def test_backend_and_lifecycle_failures_propagate(self) -> None:
        backend = create_logit_constraint(
            spec=ConstraintSpec.json_schema(True),
            request_id="backend-failure",
            factory=_Factory([], fail_apply=True),
        )
        with self.assertRaisesRegex(ConstraintLifecycleError, "backend failed to apply mask"):
            self.sampler.sample(torch.zeros(3), backend)
        self.assertTrue(backend.failed)

        lifecycle = create_logit_constraint(
            spec=ConstraintSpec.identity(),
            request_id="lifecycle-failure",
        )
        lifecycle.mask()
        with self.assertRaisesRegex(ConstraintLifecycleError, "unconsumed token mask"):
            self.sampler.sample(torch.zeros(3), lifecycle)
        self.assertTrue(lifecycle.failed)

    def test_all_masked_candidates_fail_instead_of_selecting_token_zero(self) -> None:
        constraint = create_logit_constraint(
            spec=ConstraintSpec.json_schema(True),
            request_id="all-masked",
            factory=_Factory([], allowed=()),
        )
        with self.assertRaisesRegex(ValueError, "masked every candidate"):
            self.sampler.sample(torch.tensor([3.0, 2.0]), constraint)


if __name__ == "__main__":
    unittest.main()
