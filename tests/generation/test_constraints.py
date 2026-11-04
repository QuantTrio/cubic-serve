from __future__ import annotations

import unittest
from dataclasses import dataclass

import torch

from cubic_serve.generation.constraints import (
    BackendTokenMask,
    ConstraintKind,
    ConstraintLifecycleError,
    ConstraintMatcher,
    ConstraintReadinessError,
    ConstraintSpec,
    LogitConstraint,
    UnavailableConstraintFactory,
    create_logit_constraint,
    replay_logit_constraint,
)

_VALID_RUNS = ((1, 2, 9), (1, 3, 9))


class _FakeMatcher:
    def __init__(self, constraint_digest: str, prefix: tuple[int, ...] = ()) -> None:
        self._constraint_digest = constraint_digest
        self._prefix = prefix

    @property
    def constraint_digest(self) -> str:
        return self._constraint_digest

    def _allowed(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    run[len(self._prefix)]
                    for run in _VALID_RUNS
                    if len(run) > len(self._prefix) and run[: len(self._prefix)] == self._prefix
                }
            )
        )

    def fill_token_mask(self) -> BackendTokenMask:
        return BackendTokenMask(payload=self._allowed(), needs_apply=True)

    def apply_token_mask(self, logits: torch.Tensor, payload: object) -> None:
        if type(payload) is not tuple:
            raise TypeError("fake mask payload must be a tuple")
        allowed = payload
        for token_id in range(logits.numel()):
            if token_id not in allowed:
                logits[token_id] = -torch.inf

    def accept_token(self, token_id: int) -> bool:
        if token_id not in self._allowed():
            return False
        self._prefix += (token_id,)
        return True

    def is_terminated(self) -> bool:
        return self._prefix in _VALID_RUNS

    def fork(self) -> ConstraintMatcher:
        return _FakeMatcher(self._constraint_digest, self._prefix)


@dataclass(slots=True)
class _FakeFactory:
    backend_id: str = "fake-xgrammar"
    create_count: int = 0

    def require_ready(self, spec: ConstraintSpec) -> None:
        return None

    def create_matcher(self, spec: ConstraintSpec) -> ConstraintMatcher:
        self.create_count += 1
        return _FakeMatcher(spec.digest)


def _schema() -> ConstraintSpec:
    return ConstraintSpec.json_schema(
        {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
        }
    )


def _step(constraint: LogitConstraint, token_id: int) -> torch.Tensor:
    mask = constraint.mask()
    logits = torch.arange(10, dtype=torch.float32)
    constraint.apply(logits, mask)
    constraint.accept(token_id, mask)
    return logits


class ConstraintIdentityTests(unittest.TestCase):
    def test_json_schema_is_canonical_and_domain_separated(self) -> None:
        from_text = ConstraintSpec.json_schema(
            '{ "required": ["answer"], "properties": '
            '{"answer": {"type": "integer"}}, "type": "object" }'
        )
        from_mapping = _schema()

        self.assertEqual(from_text.canonical_payload, from_mapping.canonical_payload)
        self.assertEqual(from_text.digest, from_mapping.digest)
        self.assertTrue(from_text.digest.startswith("b2s256:"))
        tool = ConstraintSpec.tool_schema(
            {
                "type": "object",
                "properties": {"answer": {"type": "integer"}},
                "required": ["answer"],
            }
        )
        self.assertIn(from_text.canonical_payload, tool.canonical_payload)
        self.assertNotEqual(tool.digest, from_text.digest)

    def test_tool_array_order_is_semantic_but_object_order_is_not(self) -> None:
        first = ConstraintSpec.tool_schema(
            [
                {"type": "function", "function": {"name": "a", "parameters": {}}},
                {"type": "function", "function": {"name": "b", "parameters": {}}},
            ]
        )
        same = ConstraintSpec.tool_schema(
            [
                {"function": {"parameters": {}, "name": "a"}, "type": "function"},
                {"function": {"parameters": {}, "name": "b"}, "type": "function"},
            ]
        )
        reversed_tools = ConstraintSpec.tool_schema(
            [
                {"type": "function", "function": {"name": "b", "parameters": {}}},
                {"type": "function", "function": {"name": "a", "parameters": {}}},
            ]
        )

        self.assertEqual(first.digest, same.digest)
        self.assertNotEqual(first.digest, reversed_tools.digest)

    def test_malformed_or_noncanonical_schema_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            ConstraintSpec.json_schema('{"type":"object","type":"array"}')
        with self.assertRaisesRegex(ValueError, "non-finite"):
            ConstraintSpec.json_schema({"minimum": float("nan")})
        with self.assertRaisesRegex(ValueError, "object or boolean"):
            ConstraintSpec.json_schema([])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            ConstraintSpec.tool_schema([])

    def test_grammar_identity_preserves_source_and_dialect(self) -> None:
        source = 'root ::= "yes" | "no"\n'
        first = ConstraintSpec.grammar(source)
        same = ConstraintSpec.grammar(source)
        another_dialect = ConstraintSpec.grammar(source, dialect="gbnf")
        another_source = ConstraintSpec.grammar(source + "\n")

        self.assertEqual(first.digest, same.digest)
        self.assertNotEqual(first.digest, another_dialect.digest)
        self.assertNotEqual(first.digest, another_source.digest)
        with self.assertRaisesRegex(ValueError, "canonical LF"):
            ConstraintSpec.grammar('root ::= "yes"\r\n')


class ConstraintReadinessTests(unittest.TestCase):
    def test_no_constraint_uses_explicit_identity(self) -> None:
        constraint = create_logit_constraint(
            spec=ConstraintSpec.identity(),
            request_id="identity-request",
        )
        self.assertEqual(constraint.spec.kind, ConstraintKind.IDENTITY)
        mask = constraint.mask()
        self.assertFalse(mask.needs_apply)
        logits = torch.tensor([1.0, 2.0, 3.0])
        before = logits.clone()
        constraint.apply(logits, mask)
        constraint.accept(2, mask)
        snapshot = constraint.finish()

        self.assertTrue(torch.equal(logits, before))
        self.assertEqual(snapshot.accepted_token_ids, (2,))
        self.assertTrue(constraint.finished)

    def test_requested_constraint_without_backend_is_not_ignored(self) -> None:
        with self.assertRaisesRegex(ConstraintReadinessError, "not downgraded to identity"):
            create_logit_constraint(spec=_schema(), request_id="missing-backend")

    def test_unavailable_xgrammar_is_an_explicit_readiness_failure(self) -> None:
        unavailable = UnavailableConstraintFactory(
            backend_id="xgrammar",
            reason="python package xgrammar is not installed",
        )
        with self.assertRaisesRegex(
            ConstraintReadinessError,
            "xgrammar.*not ready.*not installed.*not downgraded",
        ):
            create_logit_constraint(
                spec=_schema(),
                request_id="xgrammar-missing",
                factory=unavailable,
            )


class ConstraintLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.factory = _FakeFactory()

    def create(self, request_id: str = "request-a") -> LogitConstraint:
        return create_logit_constraint(
            spec=_schema(),
            request_id=request_id,
            factory=self.factory,
        )

    def test_mask_apply_accept_finish_lifecycle(self) -> None:
        constraint = self.create()
        logits = _step(constraint, 1)
        self.assertTrue(torch.isfinite(logits[1]))
        self.assertEqual(
            torch.isneginf(logits).nonzero().flatten().tolist(), [0, 2, 3, 4, 5, 6, 7, 8, 9]
        )
        _step(constraint, 2)
        _step(constraint, 9)

        snapshot = constraint.finish()
        self.assertEqual(snapshot.accepted_token_ids, (1, 2, 9))
        self.assertTrue(snapshot.matcher_terminated)
        self.assertTrue(constraint.finished)
        with self.assertRaisesRegex(ConstraintLifecycleError, "finished"):
            constraint.mask()

    def test_accept_before_apply_fails_closed(self) -> None:
        constraint = self.create()
        mask = constraint.mask()
        with self.assertRaisesRegex(ConstraintLifecycleError, "before its mask is applied"):
            constraint.accept(1, mask)
        self.assertTrue(constraint.failed)
        with self.assertRaisesRegex(ConstraintLifecycleError, "failed"):
            constraint.mask()

    def test_stale_or_duplicate_mask_use_fails_closed(self) -> None:
        duplicate = self.create("duplicate")
        mask = duplicate.mask()
        logits = torch.zeros(10)
        duplicate.apply(logits, mask)
        with self.assertRaisesRegex(ConstraintLifecycleError, "only once"):
            duplicate.apply(logits, mask)

        stale = self.create("stale")
        old_mask = stale.mask()
        old_logits = torch.zeros(10)
        stale.apply(old_logits, old_mask)
        stale.accept(1, old_mask)
        new_mask = stale.mask()
        with self.assertRaisesRegex(ConstraintLifecycleError, "stale"):
            stale.apply(torch.zeros(10), old_mask)
        self.assertIsNot(old_mask, new_mask)

    def test_invalid_token_and_incomplete_finish_are_rejected(self) -> None:
        invalid = self.create("invalid-token")
        mask = invalid.mask()
        invalid.apply(torch.zeros(10), mask)
        with self.assertRaisesRegex(ConstraintLifecycleError, "rejected sampled token 4"):
            invalid.accept(4, mask)

        incomplete = self.create("incomplete")
        _step(incomplete, 1)
        with self.assertRaisesRegex(ConstraintLifecycleError, "incomplete"):
            incomplete.finish()
        self.assertTrue(incomplete.failed)

    def test_requests_are_isolated_and_forks_diverge(self) -> None:
        parent = self.create("parent")
        untouched = self.create("untouched")
        _step(parent, 1)

        untouched_mask = untouched.mask()
        untouched_logits = torch.zeros(10)
        untouched.apply(untouched_logits, untouched_mask)
        self.assertTrue(torch.isfinite(untouched_logits[1]))
        self.assertTrue(torch.isneginf(untouched_logits[2]))

        branch = parent.fork("branch")
        _step(parent, 2)
        _step(branch, 3)
        self.assertEqual(parent.accepted_token_ids, (1, 2))
        self.assertEqual(branch.accepted_token_ids, (1, 3))
        self.assertEqual(untouched.accepted_token_ids, ())

    def test_snapshot_replay_reconstructs_private_state(self) -> None:
        original = self.create("original")
        _step(original, 1)
        snapshot = original.snapshot()
        restored = replay_logit_constraint(
            snapshot=snapshot,
            request_id="restored",
            factory=self.factory,
        )

        self.assertEqual(restored.accepted_token_ids, (1,))
        self.assertEqual(restored.snapshot(), snapshot)
        _step(original, 2)
        _step(restored, 3)
        _step(original, 9)
        _step(restored, 9)
        self.assertTrue(original.finish().matcher_terminated)
        self.assertTrue(restored.finish().matcher_terminated)
        self.assertEqual(self.factory.create_count, 2)


if __name__ == "__main__":
    unittest.main()
