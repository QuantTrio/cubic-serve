from __future__ import annotations

import importlib.util
import unittest

import torch

from cubic_serve.contracts.request import ReasoningControl, StructuredOutputMeta
from cubic_serve.generation.constraints import (
    ConstraintCompilationError,
    ConstraintLifecycleError,
    ConstraintSpec,
    ForcedTokenMask,
    UnavailableConstraintFactory,
    create_logit_constraint,
    replay_logit_constraint,
)
from cubic_serve.generation.structured_sampling import (
    ConstraintBinding,
    GrammarPool,
    reasoning_control_from_prompt,
)
from cubic_serve.generation.xgrammar import XGrammarConstraintFactory


class ReasoningBudgetTests(unittest.TestCase):
    def test_private_rewind_restores_partial_budget_ends_and_reuses_owned_states(self):
        control = ReasoningControl((7,), (9, 10), 1, True)
        pool = GrammarPool(None, vocab_size=16, stop_token_ids=(1,))
        meta = StructuredOutputMeta.identity()
        step = pool.begin((ConstraintBinding('r', meta, 0, 16, reasoning_control=control),))
        original = pool.sessions['r']
        branch = step.branches['r']
        step.mask('r', 0)
        step.propose('r', 2, 0)
        step.mask('r', 1)
        step.propose('r', 9, 1)
        step.mask('r', 2)
        step.propose('r', 10, 2)
        # Only the first output committed; the speculative end was rejected.
        step.commit({'r': (2,)})
        self.assertIs(pool.sessions['r'], branch)
        self.assertEqual(original.revision, 0)
        following = pool.begin((ConstraintBinding('r', meta, 1, 15, reasoning_control=control),))
        self.assertIs(following.branches['r'], original)
        self.assertEqual(int(following.mask('r', 0)[0]), 1 << 9)
        following.propose('r', 9, 0)
        following.mask('r', 1)
        following.propose('r', 10, 1)
        following.commit({'r': (9,)})
        last = pool.begin((ConstraintBinding('r', meta, 2, 14, reasoning_control=control),))
        self.assertIs(last.branches['r'], branch)
        self.assertEqual(int(last.mask('r', 0)[0]), 1 << 10)
        last.commit({'r': (10,)})
        answer = pool.begin((ConstraintBinding('r', meta, 3, 13, reasoning_control=control),))
        self.assertIsNone(answer.mask('r', 0))
        pool.release(('r',))
        self.assertFalse(pool.sessions)
        self.assertFalse(pool._standby)

    def test_cpu_cost_witness_follows_actual_durable_owners_and_commit(self):
        from dataclasses import replace
        from math import nan

        pool = GrammarPool(None, vocab_size=8, stop_token_ids=(1,))
        binding = ConstraintBinding('owned', StructuredOutputMeta.identity(), 0, 8,
                                    reasoning_control=ReasoningControl((7,), (6,), 99, False))
        round_ = pool.begin((binding,))
        with self.assertRaises(ConstraintLifecycleError):
            round_.cost_witness()
        round_.mask('owned', 0)
        round_.mask('owned', 0)  # Cached masks do not invent another CPU call.
        round_.propose('owned', 2, 0)
        round_.mask('owned', 1)
        round_.commit({'owned': (2, 3)})
        cost = round_.cost_witness()
        self.assertTrue(cost.valid())
        self.assertEqual(cost.owners, (('owned', 0),))
        self.assertEqual((cost.mask_calls, cost.output_tokens), (2, 2))
        self.assertEqual(pool.sessions['owned'].revision, 2)
        for changed in ({'begin_ms': nan}, {'owners': (('owned', -1),)},
                        {'mask_calls': True}, {'output_tokens': 0}):
            self.assertFalse(replace(cost, **changed).valid())

    def test_control_mask_storage_does_not_follow_the_process_default_device(self):
        with torch.device("meta"):
            pool = GrammarPool(UnavailableConstraintFactory("absent", "not needed"),
                               vocab_size=8, stop_token_ids=(1,))
            control = ReasoningControl((0,), (7,), 0, True)
            binding = ConstraintBinding("r", StructuredOutputMeta.identity(), 0, 8,
                                        reasoning_control=control)
            mask = pool.begin((binding,)).mask("r", 0)
        self.assertEqual(mask.device.type, "cpu")

    def test_forced_mask_preserves_signed_words_and_vocabulary_boundaries(self):
        for token in (31, 32, 33):
            pool = GrammarPool(UnavailableConstraintFactory("absent", "not needed"),
                               vocab_size=34, stop_token_ids=(1,))
            control = ReasoningControl((2,), (token,), 0, True)
            binding = ConstraintBinding("r", StructuredOutputMeta.identity(), 0, 8,
                                        reasoning_control=control)
            mask = pool.begin((binding,)).mask("r", 0)
            actual = {i for i in range(34) if int(mask[i // 32]) & (1 << (i % 32))}
            self.assertEqual(actual, {token})

    def test_concurrent_budget_state_is_independent_of_physical_row_permutations(self):
        for rows in (1, 4, 16, 32):
            pool = GrammarPool(UnavailableConstraintFactory("absent", "not needed"),
                               vocab_size=8, stop_token_ids=(1,))
            controls = {str(i): ReasoningControl((0,), (7,), i % 3, True) for i in range(rows)}
            for revision in range(3):
                order = list(reversed(controls)) if revision % 2 else list(controls)
                bindings = tuple(
                    ConstraintBinding(key, StructuredOutputMeta.identity(), revision, 8-revision,
                                      reasoning_control=controls[key]) for key in order
                )
                round_ = pool.begin(bindings)
                outputs = {}
                for key in order:
                    budget = controls[key].token_budget
                    token = 7 if revision == budget else 2
                    mask = round_.mask(key, 0)
                    if revision == budget:
                        self.assertEqual(int(mask[0]), 1 << 7)
                    else:
                        self.assertIsNone(mask)
                    round_.propose(key, token, 0)
                    outputs[key] = (token,)
                round_.commit(outputs)
            self.assertEqual({state.revision for state in pool.sessions.values()}, {3})

    def constraint(self, budget=2, *, active=True, seed=0, tail=()):
        return create_logit_constraint(
            spec=ConstraintSpec.identity(), request_id="parent",
            reasoning_control=ReasoningControl((7, 8), (9, 10), budget, active, seed,
                                               initial_tail=tail),
        )

    def advance(self, constraint, token):
        mask = constraint.mask()
        logits = torch.zeros(16)
        constraint.apply(logits, mask)
        allowed = set(torch.isfinite(logits).nonzero().flatten().tolist())
        self.assertIn(token, allowed)
        constraint.accept(token, mask)
        return allowed

    def test_reference_budget_forces_the_complete_end_and_not_answer_tokens(self):
        for budget in (0, 1, 3):
            c = self.constraint(budget)
            for _ in range(budget):
                self.assertEqual(self.advance(c, 2), set(range(16)))
            self.assertEqual(self.advance(c, 9), {9})
            self.assertEqual(self.advance(c, 10), {10})
            self.assertEqual(self.advance(c, 3), set(range(16)))

    def test_budget_mask_is_compact_until_existing_sampler_consumes_it(self):
        c = self.constraint(0)
        mask = c.mask()
        self.assertIsInstance(mask._payload, ForcedTokenMask)
        self.assertEqual(mask._payload.token_id, 9)

    def test_start_control_tokens_do_not_consume_a_generated_thought_budget(self):
        c = self.constraint(1, active=False)
        for _ in range(2):
            for token in (7, 8, 2):
                self.advance(c, token)
            self.assertEqual(self.advance(c, 9), {9})
            self.assertEqual(self.advance(c, 10), {10})

    def test_partial_start_and_natural_end_are_not_restarted_at_the_budget_boundary(self):
        c = self.constraint(1)
        for token in (7, 8, 2):
            self.advance(c, token)
        self.assertEqual(self.advance(c, 9), {9})
        self.assertEqual(self.advance(c, 10), {10})
        c = self.constraint(3)
        for token in (2, 9):
            self.advance(c, token)
        self.advance(c, 10)
        self.assertEqual(self.advance(c, 4), set(range(16)))
        c = self.constraint(1, seed=1, tail=(9,))
        self.assertEqual(self.advance(c, 10), {10})

    def test_snapshot_and_forks_preserve_the_same_budget_without_consuming_parent(self):
        parent = self.constraint(2)
        self.advance(parent, 2)
        branch = parent.fork("draft")
        replay = replay_logit_constraint(snapshot=parent.snapshot(), request_id="replayed")
        self.advance(branch, 3)
        self.assertEqual(self.advance(branch, 9), {9})
        for current in (parent, replay):
            self.assertEqual(self.advance(current, 4), set(range(16)))
            self.assertEqual(self.advance(current, 9), {9})

    def test_prompt_seed_uses_the_last_open_thought_and_excludes_closed_history(self):
        # Construct the canonical request directly until wire budget admission is wired.
        from dataclasses import replace
        from types import SimpleNamespace

        from cubic_serve.inputs.normalize import normalize_openai_chat_completions

        request, _ = normalize_openai_chat_completions({
            "model": "m", "messages": [{"role": "user", "content": "question"}],
            "max_tokens": 32,
        }, request_id="budget")
        request = replace(request, thinking_token_budget=2)
        codec = SimpleNamespace(reasoning_start_token_ids=(7, 8), reasoning_end_token_ids=(9, 10))
        control = reasoning_control_from_prompt(request, codec, (7, 8, 2, 9, 10, 7, 8, 3, 9))
        self.assertTrue(control.initially_reasoning)
        self.assertEqual(control.initial_reasoning_tokens, 2)
        self.assertEqual(control.initial_tail, (9,))
        c = create_logit_constraint(spec=ConstraintSpec.identity(), request_id="seed",
                                    reasoning_control=control)
        self.assertEqual(self.advance(c, 10), {10})

    def test_budget_intersects_existing_grammar_and_never_overrides_it(self):
        pool = StructuredSamplingTests().pool()
        control = ReasoningControl((0,), (7,), 0, True)
        with self.assertRaisesRegex(ConstraintCompilationError, "conflicts"):
            pool.begin((ConstraintBinding("conflict", StructuredOutputMeta.regex("a"),
                                           0, 4, reasoning_control=control),))

    def test_every_speculative_acceptance_boundary_rebuilds_from_committed_tokens(self):
        # One complete speculative prefix crosses the forced end into the answer.
        proposed = (5, 6, 7, 2, 3)
        for accepted in range(len(proposed) + 1):
            pool = StructuredSamplingTests().pool()
            control = ReasoningControl((0,), (7,), 2, True)
            binding = ConstraintBinding("r", StructuredOutputMeta.identity(), 0, 20,
                                        reasoning_control=control)
            step = pool.begin((binding,))
            for depth, token in enumerate(proposed):
                step.mask("r", depth)
                step.propose("r", token, depth)
            # The repair is deliberately different when ordinary tokens are allowed.
            repair = 7 if accepted == 2 else 4
            committed = (*proposed[:accepted], repair)
            step.commit({"r": committed})
            next_binding = ConstraintBinding("r", binding.meta, len(committed), 20-len(committed),
                                             reasoning_control=control)
            candidate = pool.begin((next_binding,)).mask("r", 0)
            # Independent discrete rule: only committed ordinary tokens consume
            # the budget; token 7 ends it. Lookahead must not change this result.
            forced = 7 not in committed and len(committed) >= 2
            if forced:
                self.assertEqual(int(candidate[0]), 1 << 7)
            else:
                self.assertIsNone(candidate)

    def test_budget_only_ignore_eos_keeps_every_committed_token(self):
        pool = StructuredSamplingTests().pool()
        control = ReasoningControl((0,), (7,), 0, True)
        binding = ConstraintBinding("r", StructuredOutputMeta.identity(), 0, 20,
                                    reasoning_control=control, stop_token_ids=())
        first = pool.begin((binding,))
        first.commit({"r": (7, 1, 2, 3)})
        self.assertEqual(pool.sessions["r"].revision, 4)
        following = ConstraintBinding("r", binding.meta, 4, 16,
                                      reasoning_control=control, stop_token_ids=())
        self.assertIsNone(pool.begin((following,)).mask("r", 0))


class ImplicitReasoningTests(unittest.TestCase):
    def test_rejected_implicit_boundary_is_masked_until_explicit_reasoning_end(self):
        from cubic_serve.generation.structured_sampling import DelayedConstraintMatcher

        pool = StructuredSamplingTests().pool()
        base = pool.factory.create_matcher(ConstraintSpec.regex("a"))
        matcher = DelayedConstraintMatcher(base, (7,), (1,), ((3,),))
        mask = matcher.fill_token_mask()
        self.assertTrue(mask.needs_apply)
        self.assertFalse(int(mask.payload[0, 0]) & (1 << 3))
        self.assertTrue(int(mask.payload[0, 0]) & (1 << 7))
        branch = matcher.fork()
        self.assertTrue(branch.accept_token(7))
        self.assertTrue(branch.accept_token(2))
        self.assertTrue(branch.accept_token(1))
        self.assertTrue(branch.is_terminated())
        self.assertTrue(matcher.reasoning)

    def test_multitoken_implicit_boundary_only_masks_its_actual_completion(self):
        from cubic_serve.generation.structured_sampling import DelayedConstraintMatcher

        pool = StructuredSamplingTests().pool()
        base = pool.factory.create_matcher(ConstraintSpec.regex("a"))
        matcher = DelayedConstraintMatcher(base, (7,), (1,), ((3, 4),))
        self.assertFalse(matcher.fill_token_mask().needs_apply)
        self.assertTrue(matcher.accept_token(3))
        mask = matcher.fill_token_mask()
        self.assertFalse(int(mask.payload[0, 0]) & (1 << 4))
        self.assertTrue(matcher.accept_token(5))
        self.assertFalse(matcher.fill_token_mask().needs_apply)

    def test_implicit_native_boundary_feeds_grammar_prefix_and_forks_privately(self):
        from cubic_serve.generation.structured_sampling import DelayedConstraintMatcher

        class Base:
            constraint_digest = "test"
            def __init__(self): self.tokens = []
            def accept_token(self, token): self.tokens.append(token); return True
            def is_terminated(self): return False
            def fork(self):
                other = Base(); other.tokens = list(self.tokens); return other

        for end, implicit in (((9,), ((7, 8),)), ((8,), ((7, 8),))):
            base = Base()
            matcher = DelayedConstraintMatcher(base, end, (0,), implicit)
            self.assertTrue(matcher.accept_token(1))
            child = matcher.fork()
            self.assertTrue(child.accept_token(7))
            self.assertTrue(child.accept_token(8))
            self.assertFalse(child.reasoning)
            self.assertEqual(child.base.tokens, [7, 8])
            self.assertTrue(matcher.reasoning)
            self.assertEqual(base.tokens, [])


@unittest.skipUnless(importlib.util.find_spec("xgrammar"), "xgrammar extra required")
class StructuredSamplingTests(unittest.TestCase):
    def test_json_and_tags_rewind_matches_bytewise_independent_matchers(self):
        import json

        factory = XGrammarConstraintFactory(
            (b'<s>', b'</s>', *(bytes([i]) for i in range(256))),
            vocab_type='raw', vocab_size=258, stop_token_ids=(1,), max_threads=2,
        )
        schema = {
            'type': 'object', 'properties': {
                'items': {'type': 'array', 'minItems': 8, 'items': {
                    'type': 'object', 'properties': {
                        's': {'type': 'string'}, 'x': {'type': 'integer'}},
                    'required': ['s', 'x'], 'additionalProperties': False}}},
            'required': ['items'], 'additionalProperties': False,
        }
        text = json.dumps({'items': [dict(s='blue天空\\\"\n'*3, x=i)
                                     for i in range(8)]}, ensure_ascii=False)
        tagged = StructuredOutputMeta.structural_tag({
            'type': 'structural_tag', 'format': {
                'type': 'tag', 'begin': '<answer>',
                'content': {'type': 'const_string', 'value': text}, 'end': '</answer>'}})
        for meta, output in ((StructuredOutputMeta.json_schema(schema), text),
                             (tagged, '<answer>'+text+'</answer>')):
            pool = GrammarPool(factory, vocab_size=258, stop_token_ids=(1,))
            keys = tuple(str(i) for i in range(8))
            oracle = {key: factory.create_matcher(ConstraintSpec(meta)) for key in keys}
            gold = tuple(b+2 for b in output.encode()) + (1,)
            revisions = dict.fromkeys(keys, 0)
            ordinal = 0
            while pool.sessions or ordinal == 0:
                order = tuple(key for key in keys if revisions[key] < len(gold))
                if ordinal % 2:
                    order = order[::-1]
                step = pool.begin(tuple(ConstraintBinding(key, meta, revisions[key],
                                        len(gold)-revisions[key]) for key in order))
                updates = {}
                for index, key in enumerate(order):
                    start = revisions[key]
                    depth = min((ordinal+index) % 4, len(gold)-start-1)
                    proposals = gold[start:start+depth]
                    actual, valid = step.target_prefix_masks(key, proposals)
                    self.assertEqual(valid, depth)
                    probe = oracle[key].fork()
                    for d, mask in enumerate(actual):
                        expected = probe.fill_token_mask()
                        self.assertTrue(torch.equal(mask, expected.payload[0]))
                        if d < depth:
                            self.assertTrue(probe.accept_token(proposals[d]))
                    accepted = (ordinal+index) % (depth+1)
                    updates[key] = gold[start:start+accepted+1]
                    for token in updates[key]:
                        self.assertTrue(oracle[key].accept_token(token))
                step.commit(updates)
                for key in order:
                    revisions[key] += len(updates[key])
                    if revisions[key] == len(gold):
                        self.assertTrue(oracle[key].is_terminated())
                        self.assertNotIn(key, pool.sessions)
                        self.assertNotIn(key, pool._standby)
                ordinal += 1
            self.assertFalse(pool._standby)

    def test_private_reuse_matches_independent_committed_prefixes_across_row_counts(self):
        # The oracle never rewinds or swaps its durable matcher. It advances
        # only actual output, and forks solely to witness a proposal prefix.
        meta = StructuredOutputMeta.regex('(a[bc]|b[xy]){128}')
        for count in (1, 4, 8, 16, 32, 64, 128, 256):
            with self.subTest(count=count):
                pool = self.pool()
                keys = tuple(str(i) for i in range(count))
                oracle = {key: pool.factory.create_matcher(ConstraintSpec(meta)) for key in keys}
                prefix = (2, 4, 3, 6) * 48
                first = pool.begin(tuple(ConstraintBinding(key, meta, 0, 512) for key in keys))
                first.commit({key: prefix for key in keys})
                for matcher in oracle.values():
                    self.assertTrue(all(matcher.accept_token(t) for t in prefix))
                revisions = dict.fromkeys(keys, len(prefix))
                for ordinal in range(6):
                    order = keys[ordinal % count:] + keys[:ordinal % count]
                    if ordinal % 2:
                        order = order[::-1]
                    step = pool.begin(tuple(ConstraintBinding(key, meta, revisions[key],
                                            512-revisions[key]) for key in order))
                    outputs = {}
                    for index, key in enumerate(order):
                        probe = oracle[key].fork()
                        proposals, masks = [], []
                        depth = (ordinal + index) % 4
                        for _ in range(depth + 1):
                            mask = probe.fill_token_mask().payload[0].clone()
                            masks.append(mask)
                            if len(proposals) < depth:
                                token = max(self.allowed(mask))
                                proposals.append(token)
                                self.assertTrue(probe.accept_token(token))
                        observed, valid = step.target_prefix_masks(key, tuple(proposals))
                        self.assertEqual(valid, depth)
                        self.assertTrue(all(torch.equal(a, b)
                                            for a, b in zip(observed, masks, strict=True)))
                        accepted = (ordinal + index) % (depth + 1)
                        actual = oracle[key]
                        for token in proposals[:accepted]:
                            self.assertTrue(actual.accept_token(token))
                        repair = min(self.allowed(actual.fill_token_mask().payload[0]))
                        self.assertTrue(actual.accept_token(repair))
                        outputs[key] = (*proposals[:accepted], repair)
                    step.commit(outputs)
                    for key in order:
                        revisions[key] += len(outputs[key])
                        self.assertEqual(pool.sessions[key].revision, revisions[key])
                        self.assertTrue(torch.equal(
                            pool.sessions[key].matcher.fill_token_mask().payload,
                            oracle[key].fill_token_mask().payload))
                pool.release(keys)
                self.assertFalse(pool.sessions)
                self.assertFalse(pool._standby)

    def test_commits_reuse_only_two_private_states_across_rejection_and_row_permutations(self):
        pool = self.pool()
        meta = StructuredOutputMeta.regex('[ab]{64}')
        owned = {'left': set(), 'right': set()}
        for revision in range(12):
            order = ('left', 'right') if revision % 2 else ('right', 'left')
            step = pool.begin(tuple(ConstraintBinding(key, meta, revision, 64-revision)
                                    for key in order))
            parents = {key: pool.sessions[key] for key in order}
            for key in order:
                owned[key].update((id(parents[key]), id(step.branches[key])))
                step.mask(key, 0)
                step.propose(key, 2, 0)
                step.mask(key, 1)
            # Repairs disagree with the private proposal on every other round.
            output = 2 if revision % 2 else 3
            step.commit({key: (output,) for key in order})
            for key in order:
                self.assertEqual(parents[key].revision, revision)
                self.assertEqual(pool.sessions[key].revision, revision+1)
                self.assertIs(pool.sessions[key], step.branches[key])
        self.assertEqual({len(states) for states in owned.values()}, {2})
        self.assertTrue(owned['left'].isdisjoint(owned['right']))
        pool.release(('right', 'left'))
        self.assertFalse(pool._standby)

    def test_failed_commit_restores_cached_proposal_masks_before_retry(self):
        pool = self.pool()
        meta = StructuredOutputMeta.regex('[ab]{4}')
        step = pool.begin(tuple(ConstraintBinding(key, meta, 0, 4) for key in ('x', 'y')))
        snapshots = {}
        for key in ('x', 'y'):
            first = step.mask(key, 0).clone()
            step.propose(key, 2, 0)
            second = step.mask(key, 1).clone()
            snapshots[key] = first, second
        with self.assertRaises(ConstraintLifecycleError):
            step.commit({'x': (3,), 'y': (5,)})
        self.assertEqual({s.revision for s in pool.sessions.values()}, {0})
        self.assertFalse(pool._standby)
        with self.assertRaises(ConstraintLifecycleError):
            step.cost_witness()
        for key in ('x', 'y'):
            self.assertEqual(step.branches[key].revision, 1)
            self.assertTrue(torch.equal(step.mask(key, 0), snapshots[key][0]))
            self.assertTrue(torch.equal(step.mask(key, 1), snapshots[key][1]))
        step.commit({'x': (3,), 'y': (2,)})
        self.assertTrue(step.cost_witness().valid())
        durable = {key: state.revision for key, state in pool.sessions.items()}
        with self.assertRaises(ConstraintLifecycleError):
            step.commit({'x': (2,), 'y': (3,)})
        self.assertEqual({key: state.revision for key, state in pool.sessions.items()}, durable)

    def test_standby_state_cannot_be_reassigned_to_another_request(self):
        pool = self.pool()
        meta = StructuredOutputMeta.regex('[ab]{8}')
        step = pool.begin(tuple(ConstraintBinding(key, meta, 0, 8) for key in ('x', 'y')))
        step.commit({'x': (2,), 'y': (3,)})
        foreign, journal, _ = pool._standby['y']
        pool._standby['x'] = foreign, journal, pool.sessions['x']
        with self.assertRaises(ConstraintLifecycleError):
            pool.begin((ConstraintBinding('x', meta, 1, 7),))
        self.assertEqual(pool.sessions['x'].revision, 1)
        self.assertEqual(pool.sessions['y'].revision, 1)

    def pool(self):
        factory = XGrammarConstraintFactory(
            ("<s>", "</s>", "a", "b", "c", "x", "y", "</think>"),
            vocab_type="raw",
            vocab_size=8,
            stop_token_ids=(1,),
            max_threads=2,
        )
        return GrammarPool(factory, vocab_size=8, stop_token_ids=(1,))

    def allowed(self, mask):
        return {i for i in range(8) if mask is None or ((int(mask[i // 32]) >> (i % 32)) & 1)}

    def test_q_and_p_reuse_immutable_prefix_masks_and_partial_rejection_commits_only_output(self):
        pool = self.pool()
        meta = StructuredOutputMeta.regex("a(b|c)[xy]")
        step = pool.begin((ConstraintBinding("a", meta, 0, 8),))
        first = step.mask("a", 0)
        saved = first.clone()
        self.assertEqual(self.allowed(first), {2})
        step.propose("a", 2, 0)
        self.assertEqual(self.allowed(step.mask("a", 1)), {3, 4})
        step.propose("a", 3, 1)
        self.assertEqual(self.allowed(step.mask("a", 2)), {5, 6})
        self.assertIs(step.mask("a", 0), first)
        self.assertTrue(torch.equal(first, saved))
        self.assertEqual(pool.sessions["a"].revision, 0)
        # Target rejects the second proposal and repairs with c.
        step.commit({"a": (2, 4)})
        following = pool.begin((ConstraintBinding("a", meta, 2, 6),))
        self.assertEqual(self.allowed(following.mask("a", 0)), {5, 6})
        following.commit({"a": (6, 1)})
        self.assertNotIn("a", pool.sessions)

    def test_target_masks_stop_at_first_invalid_unconstrained_proposal(self):
        meta = StructuredOutputMeta.regex("a(b|c)[xy]")
        for proposals, prefix in (((6, 3, 5), 0), ((2, 6, 5), 1), ((2, 3, 5), 3)):
            pool = self.pool()
            step = pool.begin((ConstraintBinding("a", meta, 0, 8),))
            masks, valid = step.target_prefix_masks("a", proposals)
            self.assertEqual(valid, prefix)
            self.assertEqual(len(masks), len(proposals) + 1)
            self.assertEqual(self.allowed(masks[0]), {2})
            self.assertEqual(pool.sessions["a"].revision, 0)
            if prefix < len(proposals):
                self.assertNotIn(proposals[prefix], self.allowed(masks[prefix]))
                self.assertTrue(all(m is None for m in masks[prefix + 1:]))
            # Commit only valid target output, never the invalid private q.
            step.commit({"a": (2, 4)})
            following = pool.begin((ConstraintBinding("a", meta, 2, 6),))
            self.assertEqual(self.allowed(following.mask("a", 0)), {5, 6})

    def test_target_masks_absorb_eos_but_reject_later_nonterminal_draft(self):
        pool = self.pool()
        step = pool.begin((ConstraintBinding("a", StructuredOutputMeta.regex("a"), 0, 8),))
        masks, valid = step.target_prefix_masks("a", (2, 1, 3))
        self.assertEqual(valid, 2)
        self.assertEqual(self.allowed(masks[2]), {1})
        step.commit({"a": (2, 1)})
        self.assertFalse(pool.sessions)

    def test_delayed_reasoning_and_eos_lookahead_do_not_advance_durable_state(self):
        pool = self.pool()
        meta = StructuredOutputMeta.regex("a")
        binding = ConstraintBinding("a", meta, 0, 16, (7,))
        step = pool.begin((binding,))
        self.assertIsNone(step.mask("a", 0))
        step.propose("a", 5, 0)
        self.assertIsNone(step.mask("a", 1))
        step.propose("a", 7, 1)
        self.assertEqual(self.allowed(step.mask("a", 2)), {2})
        step.propose("a", 2, 2)
        self.assertEqual(self.allowed(step.mask("a", 3)), {1})
        step.propose("a", 1, 3)
        self.assertEqual(self.allowed(step.mask("a", 4)), {1})
        self.assertTrue(pool.sessions["a"].reasoning)
        step.commit({"a": (5, 7, 2, 1, 1)})
        self.assertNotIn("a", pool.sessions)

    def test_invalid_commit_is_atomic_and_request_state_and_row_maps_are_independent(self):
        pool = self.pool()
        meta = StructuredOutputMeta.regex("[ab][xy]")
        step = pool.begin(
            (ConstraintBinding("left", meta, 0, 4), ConstraintBinding("right", meta, 0, 4))
        )
        self.assertEqual(self.allowed(step.mask("right", 0)), {2, 3})
        self.assertEqual(self.allowed(step.mask("left", 0)), {2, 3})
        with self.assertRaises(ConstraintLifecycleError):
            step.commit({"left": (2,), "right": (4,)})
        self.assertEqual(pool.sessions["left"].revision, 0)
        self.assertEqual(pool.sessions["right"].revision, 0)
        step.commit({"left": (2,), "right": (3,)})
        with self.assertRaises(ConstraintLifecycleError):
            pool.begin((ConstraintBinding("left", meta, 0, 4),))
        next_ = pool.begin(
            (ConstraintBinding("right", meta, 1, 3), ConstraintBinding("left", meta, 1, 3))
        )
        next_.commit({"right": (5, 1), "left": (6, 1)})
        self.assertFalse(pool.sessions)

    def test_length_limit_releases_an_incomplete_language_without_claiming_eos(self):
        pool = self.pool()
        meta = StructuredOutputMeta.regex("ab")
        step = pool.begin((ConstraintBinding("a", meta, 0, 1),))
        step.mask("a", 0)
        step.commit({"a": (2,)})
        self.assertNotIn("a", pool.sessions)
        with self.assertRaises(ConstraintLifecycleError):
            step.mask("a", 0)


if __name__ == "__main__":
    unittest.main()
