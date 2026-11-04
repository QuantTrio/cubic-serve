from __future__ import annotations

import unittest
import pytest
from copy import deepcopy
from itertools import pairwise
from random import Random
from unittest.mock import patch

from cubic_serve.runtime.speculative import (
    DraftAction,
    DraftPolicyConfig,
    DynamicDraftPlanner,
    DynamicDraftPolicy,
    draft_decision_horizon,
)


@pytest.mark.parametrize("owners", [1, 2, 3, 4, 7, 8, 16, 31, 32])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_coherent_budget_requires_uniform_candidates_and_full_price_coverage(owners, depth):
    planner = DynamicDraftPlanner(tuple(1 << i for i in range(9)))
    policies = tuple(DynamicDraftPolicy(depth, 3) for _ in range(owners))
    planner.seed_compound_costs(tuple((owners * (k + 1), k + 1, 1.) for k in (1, 2, 3)))
    eligible = all((gap := planner.bucket(owners * (k + 1)) - owners * (k + 1)) == 0
                   or gap >= owners for k in range(3))
    assert planner.permits_coherent_budget(policies, row_limit=256) == eligible
    assert not planner.permits_coherent_budget(policies, row_limit=owners * 4 - 1)
    planner._seed_compound.clear()
    assert not planner.permits_coherent_budget(policies, row_limit=256)


def test_decision_budget_counts_emitted_bonus_and_rounds_separately():
    policy = DynamicDraftPolicy(3, 3, config=DraftPolicyConfig(decision_tokens=7, decision_min_rounds=3))
    assert policy.decision_budget == (7, 3)
    for accepted, expected in ((3, (3, 2)), (2, (0, 1)), (0, (0, 0))):
        policy.observe(accepted=accepted, proposed=3)
        assert policy.decision_budget == expected
    assert policy.ready


class CompoundRoundCostTests(unittest.TestCase):
    def test_complete_round_updates_target_correction_without_double_counting(self):
        for depth in (2, 3, 4):
            with self.subTest(depth=depth):
                planner = DynamicDraftPlanner((1, 2, 4, 8, 16))
                rows = 4 * depth
                capacity = planner.bucket(rows)
                planner.seed_costs(target_samples=((4, 6.), (capacity, 2.)), proposal_samples=(),
                                   target_depth_samples=((capacity, depth, 8.),))
                planner.observe_compound_cost(target_rows=rows, segment_depth=depth,
                    milliseconds=18., target_ms=12., completed_steps=7)
                self.assertEqual(planner._target_calibration.values, {capacity: 1.5})
                self.assertAlmostEqual(planner._zero_score(4), 4. / 9.)
                self.assertAlmostEqual(planner._work_score(5., rows, (4,) * (depth-1)), 5. / 18.)

    def test_complete_ingress_or_missing_matching_depth_cannot_calibrate_plain_target(self):
        planner = DynamicDraftPlanner((1, 2, 4, 8))
        planner.seed_costs(target_samples=((4, 2.),), proposal_samples=(),
                           target_depth_samples=((4, 3, 8.),))
        planner.observe_compound_cost(target_rows=4, segment_depth=3,
            milliseconds=18., target_ms=12., ingress=True)
        planner.observe_compound_cost(target_rows=4, segment_depth=2,
            milliseconds=18., target_ms=12.)
        self.assertEqual(planner._target_calibration.values, {})
        for invalid in (-1., float('nan'), float('inf'), 19., True):
            with self.assertRaises(ValueError):
                planner.observe_compound_cost(target_rows=4, segment_depth=2,
                    milliseconds=18., target_ms=invalid)

    def test_request_costs_do_not_turn_unknown_k0_work_into_zero_work(self):
        from cubic_serve.generation.structured_sampling import GrammarRoundCost

        planner = self.planner()
        planner.require_complete_rounds = True
        planner.seed_compound_costs(((4, 2, 10.), (6, 3, 16.)))
        cost = GrammarRoundCost((('a', 20), ('b', 21)), 2., 6., 3., 0., 1., 2., 1., 4, 4)
        planner.observe_request_round(target_rows=4, proposal_rows=(2,), target_ms=8.,
                                      worker_ms=18., round_ms=20., grammar_cost=cost)
        self.assertEqual(planner._zero_score(2), -1.)
        self.assertEqual(planner._work_score(3., 3, (1,)), -1.)
        # Native work + shared copies + output-dependent acceptance + handoff
        # + unknown remainder + only the exposed portion of mask preparation.
        self.assertAlmostEqual(planner._work_score(5., 6, (2, 2)), 5. / 27.5)
        planner.observe_compound_cost(target_rows=6, segment_depth=3, milliseconds=25.)
        self.assertEqual(planner._work_score(5., 6, (2, 2)), 5. / 25.)
        # An actual complete K0 round immediately supplies a valid stop price.
        planner.observe_request_round(target_rows=2, proposal_rows=(), target_ms=6.,
                                      worker_ms=10., round_ms=12., grammar_cost=cost)
        self.assertEqual(planner._zero_score(2), 2. / 12.)
        self.assertGreater(planner._work_score(3., 3, (1,)), 0.)
        self.assertEqual(planner.new_cost_domain()._request_work, {})

    def test_request_price_requires_matching_masked_occupancy_and_steady_path(self):
        from cubic_serve.generation.structured_sampling import GrammarRoundCost

        planner = self.planner()
        planner.require_complete_rounds = True
        planner.seed_compound_costs(((4, 2, 10.),))
        cost = GrammarRoundCost((('a', 20),), 2., 6., 3., 0., 1., 2., 1., 2, 2)
        for ingress in (False, True):
            planner.observe_request_round(target_rows=4, proposal_rows=(2,), target_ms=8.,
                                          worker_ms=18., round_ms=20., grammar_cost=cost,
                                          ingress=ingress)
        self.assertEqual(planner._request_work, {})
        self.assertEqual(planner._resident_zero_cost.values, {})

    def test_request_window_sums_are_normalized_once_not_per_receipt(self):
        from dataclasses import replace
        from cubic_serve.generation.structured_sampling import GrammarRoundCost

        cost = GrammarRoundCost((('a', 20), ('b', 21)), 2., 6., 3., 0., 1., 2., 1., 4, 4)
        sums = replace(cost, **{name: 3 * getattr(cost, name) for name in (
            'begin_ms', 'mask_ms', 'first_mask_ms', 'advance_ms', 'fork_ms', 'accept_ms',
            'publish_ms', 'mask_calls', 'output_tokens')})
        prices = []
        for steps, witness in ((1, cost), (3, sums)):
            planner = self.planner()
            planner.require_complete_rounds = True
            planner.seed_compound_costs(((4, 2, 10.),))
            planner.observe_request_round(target_rows=4, proposal_rows=(2,), target_ms=8.,
                                          worker_ms=18., round_ms=20., grammar_cost=witness,
                                          steps=steps)
            prices.append(planner._request_work)
        self.assertEqual(prices[0], prices[1])

    def test_qualified_masked_tail_completes_cold_zero_price_without_forcing_k(self):
        from cubic_serve.generation.structured_sampling import GrammarRoundCost

        planner = self.planner()
        planner.seed_costs(target_samples=((2, 6.),), proposal_samples=())
        planner.require_complete_rounds = True
        planner.seed_compound_costs(((4, 2, 10.),))
        planner.seed_constraint_sampling_costs(((2, 1.5),))
        self.assertEqual(planner._zero_score(2), -1.)  # No request CPU work yet.
        cost = GrammarRoundCost((('a', 20), ('b', 21)), 2., 6., 3., 0., 1., 2., 1., 4, 4)
        planner.observe_request_round(target_rows=4, proposal_rows=(2,), target_ms=8.,
                                      worker_ms=18., round_ms=20., grammar_cost=cost)
        # target6 + head1.5 + common4 + accept1 + handoff2 + unknown2.
        self.assertAlmostEqual(planner._zero_score(2), 2. / 16.5)
        planner.observe_resident_cost(target_rows=2, milliseconds=13.)
        self.assertEqual(planner._zero_score(2), 2. / 13.)
        self.assertEqual(planner.new_cost_domain()._seed_constraint_sampling, {})
        for invalid in (((3, 1.),), ((2, 1.), (2, 2.)), ((2, 0.),), ((2, float('nan')),)):
            with self.assertRaises(ValueError):
                planner.seed_constraint_sampling_costs(invalid)

    def test_target_calibration_uses_matching_segment_depth_not_plain_body(self):
        from cubic_serve.generation.structured_sampling import GrammarRoundCost

        planner = self.planner()
        planner.require_complete_rounds = True
        planner.seed_costs(target_samples=(), proposal_samples=(),
                           target_depth_samples=((4, 2, 4.),))
        planner.seed_compound_costs(((4, 2, 10.),))
        cost = GrammarRoundCost((('a', 20), ('b', 21)), 2., 6., 3., 0., 1., 2., 1., 4, 4)
        planner.observe_request_round(target_rows=4, proposal_rows=(2,), target_ms=8.,
                                      worker_ms=18., round_ms=20., grammar_cost=cost)
        self.assertEqual(planner._target_calibration.values, {4: 2.})

    def test_request_native_calibration_does_not_scale_fixed_sampler_work(self):
        planner = self.planner()
        planner.seed_costs(target_samples=(), proposal_samples=(),
                           target_depth_samples=((4, 2, 4.),))
        planner._target_calibration.observe(4, 2.)
        # 4ms target doubles; the remaining6ms retains its measured price.
        self.assertEqual(planner._request_native_price(10., 4, 2), 14.)

    def test_cold_complete_calibration_preserves_non_target_work_in_both_domains(self):
        for constrained in (False, True):
            with self.subTest(constrained=constrained):
                planner = self.planner()
                planner.require_complete_rounds = constrained
                planner.seed_costs(target_samples=(), proposal_samples=(),
                                   target_depth_samples=((4, 2, 4.),))
                planner.seed_compound_costs(((4, 2, 10.),))
                planner._target_calibration.observe(4, 2.)
                # Only the four target milliseconds double. Six milliseconds
                # of proposal, sampling and other work retain their own price.
                self.assertAlmostEqual(planner._work_score(4., 4, (2,)), 4. / 14.)

    def test_cold_complete_price_carries_observed_non_target_excess_once(self):
        planner = self.planner()
        planner.seed_costs(target_samples=(), proposal_samples=(),
                           target_depth_samples=((4, 2, 4.), (8, 2, 8.)))
        planner.seed_compound_costs(((4, 2, 10.), (8, 2, 18.)))
        planner._target_calibration.observe(4, 2.)
        planner.observe_compound_cost(target_rows=4, segment_depth=2, milliseconds=16.)
        # Observed16 = target4*2 + fixed6 + excess2. At eight rows, retain
        # the same measured excess: target8*2 + fixed10 + excess2 = 28.
        self.assertAlmostEqual(planner._work_score(8., 8, (4,)), 8. / 28.)
        planner.observe_compound_cost(target_rows=8, segment_depth=2, milliseconds=21.)
        # A matching inclusive measurement already accounts for every stage.
        self.assertAlmostEqual(planner._work_score(8., 8, (4,)), 8. / 21.)

    def test_staged_cold_price_includes_serial_masks_and_first_live_replaces_seed(self):
        planner = self.planner()
        planner.require_complete_rounds = True
        planner._request_work[2] = (4., .5, 1.5, 2., 2., 1.5)
        planner.seed_constraint_sampling_costs(((2, 1.5), (8, 4.)),
                                              greedy_samples=((2, .5), (8, 2.)))
        # target8; two proposals each native3 + serial mask3 + sampler delta1;
        # common4 + accept2.5 + handoff2 + remaining2 + target sampling4.
        self.assertEqual(planner._work_score(5., 6, (2, 2)), 5. / 36.5)
        planner.observe_cost(target_rows=6, target_ms=8., segment_depth=3,
                             proposal_samples=((2, 7.), (2, 7.)), overhead_ms=14.5)
        self.assertEqual(planner.proposal_cost.values[2], 7.)
        self.assertEqual(planner._request_proposal_observed, {2})
        self.assertEqual(planner._request_stage_overhead[3].values, {8: 14.5})
        # The measured inclusive stage is not charged for masks a second time.
        self.assertEqual(planner._work_score(5., 6, (2, 2)), 5. / 36.5)
        self.assertEqual(planner.new_cost_domain()._request_proposal_observed, set())

    def planner(self):
        planner = DynamicDraftPlanner((1, 2, 4, 8, 16))
        planner.seed_costs(target_samples=((1, 5.), (4, 6.), (8, 8.)),
                           proposal_samples=((1, 2.), (2, 3.)))
        return planner

    def test_complete_round_does_not_add_its_proposals_or_handoff_twice(self):
        planner = self.planner()
        planner.overhead_cost.observe(4, 3.)
        planner.overhead_cost.observe(8, 3.)
        planner.observe_compound_cost(target_rows=4, segment_depth=4, milliseconds=8.)
        self.assertEqual(planner._work_score(3., 4, (1, 1, 1)), 3. / 8.)
        # This retained price belongs to the executing mode, not its grammar
        # cost domain or a larger unobserved graph with a different workload.
        self.assertEqual(planner.new_cost_domain().compound_cost, {})
        self.assertEqual(planner._work_score(6., 8, (2, 2, 2)), 6. / 20.)

    def test_uniform_round_price_cannot_discount_ragged_or_target_only_owners(self):
        planner = self.planner()
        planner.observe_compound_cost(target_rows=4, segment_depth=3, milliseconds=1.)
        self.assertEqual(planner._work_score(3., 4, (2, 1)), 3. / 11.)
        planner.observe_compound_cost(target_rows=4, segment_depth=2, milliseconds=1.)
        self.assertEqual(planner._work_score(3., 4, (1,)), 3. / 8.)

    def test_one_time_proposal_ingress_cannot_poison_the_steady_round_price(self):
        planner = self.planner()
        planner.observe_compound_cost(target_rows=4, segment_depth=4,
                                      milliseconds=17.568738, ingress=True)
        self.assertEqual(planner.compound_cost, {})
        planner.observe_compound_cost(target_rows=4, segment_depth=4,
                                      milliseconds=12.35232)
        self.assertEqual(planner.compound_cost[4].values, {4: 12.35232})
        planner.observe_compound_cost(target_rows=4, segment_depth=4,
                                      milliseconds=90., ingress=True)
        self.assertEqual(planner.compound_cost[4]._pending, {})
        self.assertAlmostEqual(planner._work_score(3.2, 4, (1, 1, 1)), 3.2 / 12.35232)
        with self.assertRaises(ValueError):
            planner.observe_compound_cost(target_rows=4, segment_depth=4,
                                          milliseconds=12., ingress=1)

    def test_cold_complete_price_is_exact_and_live_price_supersedes_it(self):
        planner = self.planner()
        before = planner._work_score(3.2, 8, (2, 2, 2))
        planner.seed_compound_costs(((4, 4, 7.),))
        self.assertAlmostEqual(planner._work_score(3.2, 4, (1, 1, 1)), 3.2 / 7.)
        # No extrapolation to another bucket, depth, or ragged ownership.
        self.assertEqual(planner._work_score(3.2, 8, (2, 2, 2)), before)
        self.assertNotEqual(planner._work_score(3.2, 4, (1, 1)), 3.2 / 7.)
        self.assertNotEqual(planner._work_score(3.2, 4, (1,)), 3.2 / 7.)
        self.assertEqual(planner.new_cost_domain()._seed_compound, {})
        planner.observe_compound_cost(target_rows=4, segment_depth=4, milliseconds=9.)
        self.assertAlmostEqual(planner._work_score(3.2, 4, (1, 1, 1)), 3.2 / 9.)
        planner.reset_runtime_costs()
        self.assertAlmostEqual(planner._work_score(3.2, 4, (1, 1, 1)), 3.2 / 9.)

    def test_qualified_complete_price_prevents_false_terminal_stop(self):
        config = DraftPolicyConfig(decision_tokens=1, decision_min_rounds=1)
        planner = DynamicDraftPlanner((1, 2, 4, 8, 16), config=config)
        planner.seed_costs(
            target_samples=((1, 2.), (2, 3.), (4, 4.), (8, 20.), (16, 35.)),
            proposal_samples=((1, 2.), (2, 3.), (4, 5.)))
        policies = tuple(DynamicDraftPolicy(1, 3, config) for _ in range(4))
        for policy in policies:
            policy.observe(accepted=1, proposed=1)
        self.assertEqual(planner.plan(policies, row_limit=16), (0,) * 4)
        planner.seed_compound_costs(((16, 4, 8.),))
        targets = planner.plan(policies, row_limit=16)
        self.assertEqual(targets, (3,) * 4)
        # The prior changes the cost evidence, never the request transition rule.
        for policy, target in zip(policies, targets, strict=True):
            self.assertEqual(policy.move_toward(target).next_tokens, 2)
            self.assertFalse(policy.stopped)

    def test_new_cold_bucket_keeps_same_family_observed_slowdown(self):
        planner = self.planner()
        planner.seed_compound_costs(((4, 4, 7.), (8, 4, 10.), (16, 4, 20.)))
        planner._target_calibration.observe(4, 1.)
        planner._target_calibration.observe(8, 1.2)
        planner.observe_compound_cost(target_rows=4, segment_depth=4, milliseconds=9.)
        planner.observe_compound_cost(target_rows=8, segment_depth=4, milliseconds=13.)
        # Review scales by1.2; the outer full round has1ms remaining excess.
        # Applying its complete ratio would predict26ms; a pure offset23ms.
        self.assertAlmostEqual(planner._work_score(12., 16, (4, 4, 4)), 12. / 25.)
        planner.observe_compound_cost(target_rows=16, segment_depth=4, milliseconds=28.)
        self.assertAlmostEqual(planner._work_score(12., 16, (4, 4, 4)), 12. / 28.)

    def test_review_slowdown_reprices_cold_complete_before_a_full_live_round(self):
        planner = self.planner()
        planner.seed_compound_costs(((16, 4, 20.),))
        planner._target_calibration.observe(8, 1.5)
        self.assertAlmostEqual(planner._work_score(12., 16, (4, 4, 4)), 12. / 30.)

    def test_cold_logical_occupancy_interpolates_only_inside_its_physical_graph(self):
        planner = DynamicDraftPlanner((1, 2, 4, 8, 16, 32))
        planner.seed_costs(target_samples=((16, 9.), (32, 12.)),
                           proposal_samples=((1, 1.),))
        planner.seed_compound_costs(((20, 4, 20.), (32, 4, 26.)))
        planner._target_calibration.observe(32, 1.1)
        planner.observe_compound_cost(target_rows=20, segment_depth=4, milliseconds=24.)
        # Same graph's24-row cold price22ms, review factor1.1, remaining2ms.
        self.assertAlmostEqual(planner._work_score(12., 24, (6, 6, 6)), 12. / 26.2)
        self.assertIsNone(planner._compound_seed_estimate(16, 4))
        self.assertIsNone(planner._compound_seed_estimate(24, 2))

    def test_non_power_of_two_segment_depth_uses_its_actual_logical_prior(self):
        planner = self.planner()
        planner.seed_compound_costs(((3, 3, 9.),))
        self.assertAlmostEqual(planner._work_score(3., 3, (1, 1)), 3. / 9.)

    def test_partial_logical_round_cannot_replace_full_capacity_price(self):
        planner = self.planner()
        planner.seed_compound_costs(((16, 4, 20.),))
        self.assertNotEqual(planner._work_score(9., 12, (3, 3, 3)), 9. / 20.)
        planner.observe_compound_cost(target_rows=12, segment_depth=4, milliseconds=7.)
        self.assertEqual(planner.compound_cost[4].values, {12: 7.})
        self.assertAlmostEqual(planner._work_score(9., 12, (3, 3, 3)), 9. / 7.)
        self.assertAlmostEqual(planner._work_score(12., 16, (4, 4, 4)), 12. / 20.)

    def test_complete_residual_does_not_transfer_across_depth_or_sampling_domain(self):
        planner = self.planner()
        planner.seed_compound_costs(((4, 4, 7.), (8, 4, 10.), (8, 2, 12.)))
        planner.observe_compound_cost(target_rows=4, segment_depth=4, milliseconds=17.)
        self.assertAlmostEqual(planner._work_score(6., 8, (4,)), 6. / 12.)
        self.assertEqual(planner.new_cost_domain().compound_cost, {})

    def test_cold_complete_profile_rejects_invalid_or_duplicate_evidence(self):
        invalid = (((3, 2, 1.),), ((4, 5, 1.),), ((4, 1, 1.),),
                   ((4, 4, float("nan")),), ((4, 4, 0.),), ((4, 4, True),),
                   ((4, 4, 1.), (4, 4, 2.)))
        for samples in invalid:
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                self.planner().seed_compound_costs(samples)

    def test_idle_reset_discards_incomplete_samples_but_keeps_measured_price(self):
        planner = self.planner()
        planner.observe_compound_cost(target_rows=4, segment_depth=4, milliseconds=8.)
        planner.observe_compound_cost(target_rows=4, segment_depth=4, milliseconds=80.)
        planner.reset_runtime_costs()
        self.assertEqual(planner.compound_cost[4].values, {4: 8.})
        self.assertEqual(planner.compound_cost[4]._pending, {})


class DynamicDraftPolicyTests(unittest.TestCase):
    def policy(self, *, initial: int = 1, maximum: int = 7) -> DynamicDraftPolicy:
        return DynamicDraftPolicy(
            initial,
            maximum,
            DraftPolicyConfig(
                decision_tokens=8,
                decision_min_rounds=4,
            ),
        )

    def test_resident_budget_preserves_every_decision_boundary_and_observation(self):
        for initial in (1, 2, 3):
            with self.subTest(initial=initial):
                serial = self.policy(initial=initial, maximum=3)
                windowed = self.policy(initial=initial, maximum=3)
                rounds = 0
                decisions = 0
                while rounds < 128:
                    budget = windowed.rounds_until_decision
                    self.assertGreater(budget, 0)
                    for _ in range(budget):
                        self.assertFalse(serial.ready)
                        self.assertFalse(windowed.ready)
                        accepted = (rounds * 7 + rounds // 3) % (serial.current_tokens + 1)
                        serial.observe(accepted=accepted, proposed=serial.current_tokens)
                        windowed.observe(accepted=accepted, proposed=windowed.current_tokens)
                        rounds += 1
                    self.assertEqual(serial, windowed)
                    if serial.ready:
                        self.assertEqual(windowed.rounds_until_decision, 0)
                        target = 1 if decisions % 2 else 3
                        self.assertEqual(serial.move_toward(target), windowed.move_toward(target))
                        decisions += 1
                self.assertGreater(decisions, 4)

    def test_terminal_stop_margin_is_finite_nonnegative_configuration(self) -> None:
        self.assertEqual(DraftPolicyConfig().terminal_stop_margin, 0.03)
        self.assertEqual(
            DraftPolicyConfig.from_dict({"terminal_stop_margin": 0.5}).terminal_stop_margin, 0.5
        )
        for value in (-0.1, float("nan"), float("inf"), True, "0.2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DraftPolicyConfig.from_dict({"terminal_stop_margin": value})

    def test_default_observation_window_uses_committed_tokens_and_a_round_floor(self) -> None:
        config = DraftPolicyConfig()
        self.assertEqual(config.decision_tokens, 24)
        self.assertEqual(config.decision_min_rounds, 8)
        self.assertEqual(config.history_decay, 0.99)

        policy = DynamicDraftPolicy(1, 7, config)
        for _ in range(11):
            policy.observe(accepted=1, proposed=1)
        self.assertFalse(policy.ready)
        policy.observe(accepted=1, proposed=1)
        self.assertTrue(policy.ready)

    def test_window_counts_committed_output_instead_of_wall_time(self) -> None:
        policy = self.policy(initial=3)
        for _ in range(3):
            policy.observe(accepted=1, proposed=3)

        self.assertEqual(policy._rounds, 3)
        self.assertEqual(policy._window_tokens, 6)
        self.assertFalse(policy.ready)

        policy.observe(accepted=1, proposed=3)
        self.assertTrue(policy.ready)
        self.assertEqual(policy.move_toward(2).action, DraftAction.DEMOTE)

    def test_decision_resets_window_evidence_but_retains_history(self) -> None:
        policy = self.policy(initial=3)
        for _ in range(4):
            policy.observe(accepted=3, proposed=3)
        retained_trials = tuple(policy._trials)

        self.assertEqual(policy.move_toward(3).action, DraftAction.HOLD)
        self.assertEqual(policy._window_tokens, 0)
        self.assertEqual(tuple(policy._trials), retained_trials)

    def test_accepted_prefix_updates_only_reached_conditional_depths(self) -> None:
        policy = self.policy(initial=5)
        policy.observe(accepted=2, proposed=5)

        self.assertEqual(policy._trials, [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(policy._successes, [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        survival = policy.survival_probabilities()
        self.assertTrue(all(left >= right for left, right in pairwise(survival)))

    def test_request_moves_one_level_only_after_observation_interval(self) -> None:
        policy = self.policy()
        for _ in range(3):
            policy.observe(accepted=1, proposed=1)
        self.assertEqual(policy.move_toward(6).action, DraftAction.HOLD)
        policy.observe(accepted=1, proposed=1)
        decision = policy.move_toward(6)
        self.assertEqual((decision.action, decision.next_tokens), (DraftAction.PROMOTE, 2))

    def test_hold_consumes_the_completed_observation_interval(self) -> None:
        policy = self.policy(initial=3)
        for _ in range(4):
            policy.observe(accepted=3, proposed=3)

        self.assertTrue(policy.ready)
        self.assertEqual(policy.move_toward(3).action, DraftAction.HOLD)
        self.assertFalse(policy.ready)
        self.assertEqual(policy.move_toward(4).action, DraftAction.HOLD)

        for _ in range(4):
            policy.observe(accepted=3, proposed=3)
        self.assertEqual(policy.move_toward(4).action, DraftAction.PROMOTE)

    def test_k0_is_terminal(self) -> None:
        policy = self.policy()
        for _ in range(8):
            policy.observe(accepted=0, proposed=1)
        decision = policy.move_toward(0)
        self.assertEqual((decision.action, decision.next_tokens), (DraftAction.DEMOTE, 0))
        self.assertTrue(policy.stopped)
        with self.assertRaisesRegex(RuntimeError, "after K0"):
            policy.observe(accepted=0, proposed=1)

    def test_each_request_owns_independent_survival(self) -> None:
        strong, weak = self.policy(), self.policy()
        for _ in range(4):
            strong.observe(accepted=1, proposed=1)
            weak.observe(accepted=0, proposed=1)
        self.assertGreater(strong.survival_probabilities()[0], weak.survival_probabilities()[0])

    def test_observation_accepts_load_trim_below_request_k(self) -> None:
        policy = self.policy(initial=5)
        policy.observe(accepted=2, proposed=3)
        with self.assertRaisesRegex(ValueError, "active request depth"):
            policy.observe(accepted=1, proposed=6)

    def test_recent_acceptance_can_override_old_request_history(self) -> None:
        policy = DynamicDraftPolicy(
            1,
            7,
            DraftPolicyConfig(decision_tokens=8, decision_min_rounds=4, history_decay=0.8),
        )
        for _ in range(32):
            policy.observe(accepted=1, proposed=1)
        old_survival = policy.survival_probabilities()[0]
        for _ in range(16):
            policy.observe(accepted=0, proposed=1)

        self.assertLess(policy.survival_probabilities()[0], old_survival - 0.2)

    def test_unobserved_depth_inherits_last_reached_conditional(self) -> None:
        policy = self.policy(initial=3)
        for _ in range(8):
            policy.observe(accepted=1, proposed=3)

        survival = policy.survival_probabilities()
        conditional_at_two = survival[1] / survival[0]
        conditional_at_four = survival[3] / survival[2]
        self.assertAlmostEqual(conditional_at_four, conditional_at_two)


class DynamicDraftPlannerTests(unittest.TestCase):
    def test_steady_zero_residual_reprices_an_unobserved_bucket(self) -> None:
        planner = DynamicDraftPlanner((1, 2, 4, 8))
        planner.seed_costs(target_samples=((4, 5.), (8, 9.)), proposal_samples=((1, 1.),))
        planner.observe_cost(target_rows=4, target_ms=5., proposal_samples=(), overhead_ms=6.)
        self.assertAlmostEqual(planner._zero_score(8), 8. / 15.)
        planner.observe_resident_cost(target_rows=4, milliseconds=5.5, target_ms=5.)
        self.assertAlmostEqual(planner._zero_score(8), 8. / 9.5)
        self.assertAlmostEqual(planner._zero_score(4), 4. / 5.5)
        self.assertEqual(planner._zero_overhead.values, {4: 6.})
        self.assertEqual(planner.new_cost_domain()._resident_zero_overhead.values, {})

    def test_zero_residual_is_valid_and_partial_cost_cannot_alias_full_bucket(self) -> None:
        planner = DynamicDraftPlanner((1, 2, 4))
        planner.seed_costs(target_samples=((4, 5.),), proposal_samples=((1, 1.),))
        planner.observe_cost(target_rows=4, target_ms=5., proposal_samples=(), overhead_ms=6.)
        planner.observe_resident_cost(target_rows=3, milliseconds=4., target_ms=4.)
        self.assertEqual(planner._resident_zero_cost.values, {3: 4.})
        self.assertEqual(planner._resident_zero_overhead.values, {3: 0.})
        self.assertAlmostEqual(planner._zero_score(3), 3. / 4.)
        self.assertAlmostEqual(planner._zero_score(4), 4. / 5.)

    def test_resident_components_require_a_coherent_window(self) -> None:
        planner = DynamicDraftPlanner((1,))
        for target in (-1., 13., float("nan"), True):
            with self.subTest(target=target), self.assertRaises(ValueError):
                planner.observe_resident_cost(target_rows=1, milliseconds=12., target_ms=target)
        self.assertEqual(planner._resident_zero_cost.values, {})

    def test_complete_resident_prices_update_across_idle_boundaries(self) -> None:
        planner = DynamicDraftPlanner((1,))
        planner.seed_costs(target_samples=((1, 10.),), proposal_samples=((1, 1.),))
        planner.observe_resident_cost(target_rows=1, milliseconds=12.)
        planner.reset_runtime_costs()
        planner.observe_resident_cost(target_rows=1, milliseconds=20.)
        self.assertAlmostEqual(planner._resident_zero_cost.values[1], 13.6)
        self.assertAlmostEqual(planner._zero_score(1), 1. / 13.6)

    def test_cost_domain_starts_from_qualified_seeds_without_mutable_history(self) -> None:
        planner = DynamicDraftPlanner((1, 2, 4), config=DraftPolicyConfig(cost_sample_window=1))
        planner.seed_costs(target_samples=((1, 2.), (2, 3.), (4, 5.)),
            proposal_samples=((1, 1.),), target_depth_samples=((4, 2, 7.),))
        planner.observe_cost(target_rows=4, target_ms=50., proposal_samples=((1, 20.),), overhead_ms=40.)
        planner.observe_resident_cost(target_rows=4, milliseconds=12.)
        domain = planner.new_cost_domain()
        self.assertEqual(domain.target_cost.values[4], 5.)
        self.assertEqual(domain.target_depth_cost[2].values[4], 7.)
        self.assertEqual(domain.overhead_cost.values, {})
        self.assertEqual(domain._zero_overhead.values, {})
        self.assertEqual(domain._resident_zero_cost.values, {})
        self.assertEqual(domain._resident_zero_overhead.values, {})
        self.assertEqual(domain._target_calibration.values, {})
        before = dict(planner.target_cost.values)
        domain.observe_cost(target_rows=4, target_ms=100., proposal_samples=(), overhead_ms=80.)
        self.assertEqual(planner.target_cost.values, before)
        self.assertNotEqual(domain.target_cost.values, planner.target_cost.values)
        domain.observe_resident_cost(target_rows=4, milliseconds=20.)
        self.assertEqual(planner._resident_zero_cost.values, {4: 12.})

    def test_prefill_cost_does_not_reprice_independent_decode_rows(self) -> None:
        planner = DynamicDraftPlanner((1, 2, 4))
        planner.seed_costs(
            target_samples=((1, 2.0), (2, 3.0), (4, 5.0)), proposal_samples=((1, 1.0),)
        )
        for _ in range(9):
            planner.observe_cost(
                target_rows=4,
                target_ms=100.0,
                proposal_samples=(),
                overhead_ms=40.0,
                is_decode=False,
            )
        self.assertEqual(planner.target_cost.values[4], 5.0)
        self.assertEqual(planner.overhead_cost.values, {})
        self.assertEqual(planner._zero_overhead.values, {})

    def test_live_decode_reprices_only_unvisited_target_buckets(self) -> None:
        planner = DynamicDraftPlanner((1, 2, 4), config=DraftPolicyConfig(cost_sample_window=1))
        planner.seed_costs(
            target_samples=((1, 2.0), (2, 3.0), (4, 5.0)), proposal_samples=((1, 1.0),)
        )
        planner.observe_cost(
            target_rows=4, target_ms=10.0, proposal_samples=((1, 1.0),), overhead_ms=1.0
        )
        self.assertEqual(planner._target_estimate(1), 4.0)
        self.assertEqual(planner._target_estimate(4), planner.target_cost.values[4])
        planner.target_cost.values[1] = 1.8
        self.assertEqual(planner._target_estimate(1), 1.8)

    def test_cost_curve_changes_global_budget_without_changing_survival(self) -> None:
        policies = tuple(
            DynamicDraftPolicy(1, 3, DraftPolicyConfig(decision_tokens=1, decision_min_rounds=1))
            for _ in range(2)
        )
        for policy in policies:
            policy.observe(accepted=1, proposed=1)
        planner = DynamicDraftPlanner((1, 2, 3, 4, 5, 6, 8))
        planner.target_cost.values.update({2: 4.0, 3: 4.0, 4: 4.0, 5: 4.0, 6: 4.0})
        planner.proposal_cost.values.update({1: 0.01, 2: 0.01})
        cheap = planner.plan(policies, row_limit=8)

        planner.proposal_cost.values.update({1: 20.0, 2: 20.0})
        expensive = planner.plan(policies, row_limit=8)

        self.assertGreater(sum(cheap), sum(expensive))
        self.assertEqual(policies[0].survival_probabilities(), policies[1].survival_probabilities())

    def test_allocation_is_prefix_closed_and_transition_remains_one_level(self) -> None:
        config = DraftPolicyConfig(decision_tokens=1, decision_min_rounds=1)
        strong = DynamicDraftPolicy(1, 4, config)
        weak = DynamicDraftPolicy(1, 4, config)
        strong.observe(accepted=1, proposed=1)
        weak.observe(accepted=0, proposed=1)
        planner = DynamicDraftPlanner((1, 2, 3, 4, 5, 6, 8))
        planner.seed_costs(
            target_samples=((2, 4.0), (3, 4.0), (4, 4.0)),
            proposal_samples=((1, 0.1), (2, 0.1)),
        )
        allocation = planner.plan((strong, weak), row_limit=8)
        self.assertGreaterEqual(allocation[0], allocation[1])
        decisions = tuple(
            policy.move_toward(target)
            for policy, target in zip((strong, weak), allocation, strict=True)
        )
        self.assertTrue(all(0 <= decision.next_tokens <= 2 for decision in decisions))

    def test_full_depth_lookahead_crosses_one_step_graph_bucket_valley(self) -> None:
        policy = DynamicDraftPolicy(
            1,
            3,
            DraftPolicyConfig(
                decision_tokens=8,
                decision_min_rounds=4,
            ),
        )
        for _ in range(4):
            policy.observe(accepted=1, proposed=1)
        planner = DynamicDraftPlanner((1, 2, 4))
        planner.seed_costs(
            target_samples=((1, 4.0), (2, 4.0), (4, 7.0)),
            proposal_samples=((1, 0.1),),
        )

        target = planner.plan((policy,), row_limit=4)[0]

        self.assertEqual(target, 3)
        decision = policy.move_toward(target)
        self.assertEqual(
            (decision.action, decision.previous_tokens, decision.next_tokens),
            (DraftAction.PROMOTE, 1, 2),
        )

    def test_budget_zero_is_a_normal_cost_decision(self) -> None:
        policy = DynamicDraftPolicy(
            1, 3, DraftPolicyConfig(decision_tokens=1, decision_min_rounds=1)
        )
        policy.observe(accepted=0, proposed=1)
        planner = DynamicDraftPlanner((1, 2, 4, 8))
        planner.seed_costs(
            target_samples=((1, 4.0), (2, 4.0)),
            proposal_samples=((1, 20.0),),
        )

        self.assertEqual(planner.plan((policy,), row_limit=8), (0,))

    def test_demotion_looks_past_a_slower_intermediate_depth(self) -> None:
        policy = DynamicDraftPolicy(
            5, 5, DraftPolicyConfig(decision_tokens=1, decision_min_rounds=1)
        )
        policy.observe(accepted=3, proposed=5)
        planner = DynamicDraftPlanner((1, 2, 4, 8))
        planner.seed_costs(
            target_samples=((1, 8.0), (2, 9.0), (4, 12.0), (8, 20.0)),
            proposal_samples=((1, 1.0),),
        )

        target = planner.plan((policy,), row_limit=8)[0]

        self.assertEqual(target, 3)
        self.assertEqual(policy.move_toward(target).next_tokens, 4)
        policy.observe(accepted=3, proposed=4)
        self.assertEqual(policy.move_toward(planner.plan((policy,), row_limit=8)[0]).next_tokens, 3)

    def test_lookahead_does_not_spend_rows_before_demotion_releases_them(self) -> None:
        config = DraftPolicyConfig(decision_tokens=1, decision_min_rounds=1)
        weak, strong = DynamicDraftPolicy(5, 5, config), DynamicDraftPolicy(1, 5, config)
        for _ in range(64):
            weak.observe(accepted=0, proposed=5)
            strong.observe(accepted=1, proposed=1)
        planner = DynamicDraftPlanner((1, 2, 4, 8))
        planner.seed_costs(
            target_samples=((1, 10.0), (2, 10.0), (4, 10.0), (8, 10.0)),
            proposal_samples=((1, 0.01), (2, 0.01)),
        )
        targets = planner.plan((weak, strong), row_limit=7)
        next_k = tuple(
            policy.move_toward(target).next_tokens
            for policy, target in zip((weak, strong), targets, strict=True)
        )
        self.assertLessEqual(2 + sum(next_k), 7)

    def test_unseeded_planner_holds_instead_of_guessing(self) -> None:
        policy = DynamicDraftPolicy(
            3, 7, DraftPolicyConfig(decision_tokens=1, decision_min_rounds=1)
        )
        policy.observe(accepted=3, proposed=3)

        self.assertEqual(DynamicDraftPlanner((1, 2, 4, 8)).plan((policy,), row_limit=8), (3,))

    def test_nonready_request_is_priced_at_its_actual_k(self) -> None:
        fixed = DynamicDraftPolicy(
            3, 7, DraftPolicyConfig(decision_tokens=8, decision_min_rounds=8)
        )
        movable = DynamicDraftPolicy(
            1, 7, DraftPolicyConfig(decision_tokens=1, decision_min_rounds=1)
        )
        movable.observe(accepted=1, proposed=1)
        planner = DynamicDraftPlanner((1, 2, 4, 8, 16))
        planner.seed_costs(
            target_samples=((4, 4.0), (8, 20.0)),
            proposal_samples=((1, 0.1), (2, 0.1)),
        )

        allocation = planner.plan((fixed, movable), row_limit=16)

        self.assertEqual(allocation[0], 3)

    def test_aligned_depth_can_outperform_probability_sorted_ragged_work(self) -> None:
        capacities = (1, 2, 4, 8)
        planner = DynamicDraftPlanner(capacities)
        planner.seed_costs(
            target_samples=tuple((c, 4.0) for c in capacities),
            proposal_samples=tuple((c, 1.0) for c in capacities),
        )
        policies = tuple(DynamicDraftPolicy(3, 5) for _ in range(2))
        for policy, probability in zip(policies, (0.95, 0.7), strict=True):
            policy._rounds, policy._window_tokens = 8, 24
            policy._trials = [100.0] * 5
            policy._successes = [probability * 105 - 4] * 5
        # More accepted work from a deep/short split does not pay for its extra
        # proposal launches. Both candidates obey the same eight-row limit.
        self.assertEqual(planner.plan(policies, row_limit=8), (3, 3))

    def test_nonanchor_cohort_fills_padding_with_one_extra_wave(self) -> None:
        capacities = (1, 2, 4, 8, 16)
        planner = DynamicDraftPlanner(capacities)
        planner.seed_costs(
            target_samples=tuple((c, 10.0 if c <= 8 else 100.0) for c in capacities),
            proposal_samples=tuple((c, 0.2) for c in capacities),
        )
        policies = tuple(DynamicDraftPolicy(1, 3) for _ in range(3))
        for policy in policies:
            policy._rounds, policy._window_tokens = 8, 24
        # K1 alone leaves two rows unused; uniform K2 would require M16.
        # Two extra proposals fill the existing M8 bucket at lower total cost.
        self.assertEqual(planner.plan(policies, row_limit=8), (2, 2, 1))

    def test_extra_wave_stops_at_reachable_transition_capacity(self) -> None:
        capacities = (1, 2, 4, 8, 16)
        planner = DynamicDraftPlanner(capacities)
        planner.seed_costs(
            target_samples=tuple((c, 10.0) for c in capacities),
            proposal_samples=tuple((c, 0.2) for c in capacities),
        )
        policies = tuple(DynamicDraftPolicy(k, 3) for k in (3, 3, 2, 2))
        for policy, probability in zip(policies, (0.95, 0.95, 0.9, 0.85), strict=True):
            policy._trials = [100.0] * 3
            policy._successes = [probability * 105.0 - 4.0] * 3
        for policy in policies[2:]:
            policy._rounds, policy._window_tokens = 8, 24
        targets = planner.plan(policies, row_limit=16, fixed_decode_requests=3)
        # The two cooling-down requests retain K3. A fourth extra proposal
        # cannot spend their hypothetical future demotions; three still fit.
        self.assertEqual(targets, (3, 3, 2, 1))
        actual = tuple(p.move_toward(k).next_tokens for p, k in zip(policies, targets, strict=True))
        self.assertEqual(3 + len(policies) + sum(actual), 16)

    def test_zero_comparison_includes_decode_requests_waiting_for_rows(self) -> None:
        capacities = (1, 2, 4, 8)
        planner = DynamicDraftPlanner(capacities)
        planner.seed_costs(
            target_samples=tuple((c, 8.0 if c <= 4 else 9.0) for c in capacities),
            proposal_samples=tuple((c, 1.0) for c in capacities),
        )
        policies = tuple(DynamicDraftPolicy(1, 1) for _ in range(4))
        for policy in policies:
            policy._rounds, policy._window_tokens = 8, 24
        self.assertEqual(planner.plan(policies, row_limit=8), (1, 1, 1, 1))
        self.assertEqual(
            planner.plan(policies, row_limit=8, available_decode_requests=8), (0, 0, 0, 0)
        )

    def test_asynchronous_windows_can_approach_a_smaller_positive_cohort(self) -> None:
        capacities = (1, 2, 4, 8, 16)
        planner = DynamicDraftPlanner(capacities)
        planner.seed_costs(
            target_samples=((1, 7.0), (2, 7.0), (4, 7.0), (8, 8.0), (16, 25.0)),
            proposal_samples=tuple((c, 0.2) for c in capacities),
        )
        policies = tuple(DynamicDraftPolicy(3, 3) for _ in range(4))
        policies[0]._rounds, policies[0]._window_tokens = 8, 24
        targets = planner.plan(policies, row_limit=16)
        self.assertEqual(targets, (1, 3, 3, 3))
        self.assertEqual(
            tuple(p.move_toward(k).next_tokens for p, k in zip(policies, targets, strict=True)),
            (2, 3, 3, 3),
        )

    def test_unpriced_mixed_prefill_cannot_promote_draft_depth(self) -> None:
        capacities = (1, 2, 4, 8, 16, 32, 64, 128, 256)
        planner = DynamicDraftPlanner(capacities)
        planner.seed_costs(
            target_samples=tuple((c, 1.0) for c in capacities),
            proposal_samples=tuple((c, 0.01) for c in capacities),
        )
        policy = DynamicDraftPolicy(1, 7)
        policy._rounds, policy._window_tokens = 8, 24
        self.assertEqual(planner.plan((policy,), row_limit=128), (7,))
        self.assertEqual(planner.plan((policy,), row_limit=128, target_extra_rows=128), (1,))

    def test_mixed_prefill_cannot_stop_requests_to_fill_a_cheaper_bucket(self) -> None:
        for count in (2, 4, 8, 16):
            with self.subTest(requests=count):
                capacities = tuple(1 << n for n in range((4 * count).bit_length()))
                planner = DynamicDraftPlanner(capacities)
                planner.seed_costs(
                    target_samples=tuple(
                        (c, 1.0 if c <= 2 * count else 100.0) for c in capacities
                    ),
                    proposal_samples=tuple((c, 0.01) for c in capacities),
                )
                policies = tuple(DynamicDraftPolicy(1, 3) for _ in range(count))
                for policy in policies:
                    policy._rounds, policy._window_tokens = 8, 24
                # The small bucket has room for only one proposal alongside
                # prefill. Pricing that mixture as decode would permanently
                # stop the other requests to save one transient graph step.
                targets = planner.plan(
                    policies, row_limit=4 * count, target_extra_rows=count - 1
                )
                self.assertEqual(targets, (1,) * count)
                for policy, target in zip(policies, targets, strict=True):
                    policy.move_toward(target)
                    self.assertFalse(policy.stopped)

    def test_target_depth_prices_do_not_pollute_one_token_decode(self) -> None:
        capacities = (1, 2, 4, 8)
        planner = DynamicDraftPlanner(capacities)
        planner.seed_costs(
            target_samples=tuple((c, 14.0 + c * 0.5) for c in capacities),
            proposal_samples=tuple((c, 1.0 + c * 0.02) for c in capacities),
            target_depth_samples=tuple(
                (c, d, 12.0 + 2 * d if d <= 4 else 20.0 + 10 * (d - 4))
                for c in capacities
                for d in range(2, min(c, 8) + 1)
            ),
        )
        policy = DynamicDraftPolicy(3, 7)
        for _ in range(12):
            policy.observe(accepted=3, proposed=3)
        self.assertEqual(planner.plan((policy,), row_limit=8), (3,))
        single = planner.target_cost.values[8]
        depth4 = planner.target_depth_cost[4].values[8]
        planner.observe_cost(
            target_rows=8,
            target_ms=100.0,
            proposal_samples=((1, 1.0),),
            overhead_ms=4.0,
            segment_depth=5,
        )
        self.assertEqual(planner.target_cost.values[8], single)
        self.assertEqual(planner.target_depth_cost[4].values[8], depth4)
        planner.reset_runtime_costs()
        self.assertEqual(planner.target_depth_cost[5].values[8], 30.0)

    def test_low_capacity_cost_does_not_scale_below_observed_anchor(self) -> None:
        planner = DynamicDraftPlanner((1, 2, 4, 8))
        planner.target_cost.values[4] = 9.0

        self.assertEqual(planner.target_cost.estimate(1), 9.0)

    def test_observed_costs_are_keyed_by_graph_capacity(self) -> None:
        planner = DynamicDraftPlanner((1, 2, 4, 8))
        planner.observe_cost(
            target_rows=3,
            target_ms=9.0,
            proposal_samples=((1, 2.0), (3, 5.0)),
            overhead_ms=1.0,
        )
        self.assertEqual(planner.target_cost.values, {4: 9.0})
        self.assertEqual(planner.proposal_cost.values, {1: 2.0, 4: 5.0})
        self.assertEqual(planner.overhead_cost.values, {4: 1.0})

    def test_idle_request_window_preserves_executor_cost_history(self) -> None:
        planner = DynamicDraftPlanner(
            (1, 2, 4, 8), config=DraftPolicyConfig(cost_sample_window=1)
        )
        planner.seed_costs(
            target_samples=((1, 4.0), (2, 5.0)),
            proposal_samples=((1, 1.0), (2, 2.0)),
        )
        planner.observe_cost(
            target_rows=2,
            target_ms=20.0,
            proposal_samples=((1, 8.0),),
            overhead_ms=3.0,
        )
        expected_target = dict(planner.target_cost.values)
        expected_proposal = dict(planner.proposal_cost.values)
        expected_calibration = dict(planner._target_calibration.values)

        planner.reset_runtime_costs()

        self.assertEqual(planner.target_cost.values, expected_target)
        self.assertEqual(planner.proposal_cost.values, expected_proposal)
        self.assertEqual(planner.overhead_cost.values, {2: 3.0})
        self.assertEqual(planner._target_calibration.values, expected_calibration)

    def test_legacy_cost_update_is_parameterized_without_changing_survival_defaults(self) -> None:
        config = DraftPolicyConfig(cost_sample_window=1)
        planner = DynamicDraftPlanner((1, 2, 4, 8), config=config)
        planner.seed_costs(target_samples=((4, 20.0),), proposal_samples=((1, 2.0),))
        planner.observe_cost(
            target_rows=4, target_ms=15.0, proposal_samples=((1, 1.0),), overhead_ms=1.0
        )
        self.assertAlmostEqual(planner.target_cost.values[4], 19.0)
        self.assertAlmostEqual(planner.proposal_cost.values[1], 1.8)
        self.assertEqual(config.decision_tokens, DraftPolicyConfig().decision_tokens)
        planner.reset_runtime_costs()
        self.assertEqual(planner.target_cost.values[4], 19.0)

    def test_cold_counterfactual_uses_matching_shape_correction(self) -> None:
        planner = DynamicDraftPlanner((1, 2, 4, 8, 16))
        planner.seed_costs(
            target_samples=((2, 10.0), (4, 20.0), (8, 40.0), (16, 80.0)),
            proposal_samples=((1, 1.0),),
            target_depth_samples=((4, 2, 10.0), (8, 2, 30.0)),
        )
        planner.observe_cost(
            target_rows=4, target_ms=15.0, proposal_samples=((2, 1.0),),
            overhead_ms=2.0, segment_depth=2,
        )
        planner.observe_cost(
            target_rows=8, target_ms=33.0, proposal_samples=((4, 1.0),),
            overhead_ms=3.0, segment_depth=2,
        )
        # A lightly corrected larger review cannot erase the smaller graph's
        # measured slowdown when pricing an irreversible K0 decision.
        self.assertAlmostEqual(planner._target_estimate(4), 30.0)
        self.assertAlmostEqual(planner._target_estimate(8), 44.0)
        self.assertAlmostEqual(planner._target_estimate(2), 15.0)
        self.assertAlmostEqual(planner._target_estimate(16), 88.0)
        planner.reset_runtime_costs()
        self.assertAlmostEqual(planner._target_estimate(4), 30.0)

    def test_cold_zero_envelope_does_not_override_measured_cost(self) -> None:
        planner = DynamicDraftPlanner((1, 2, 4, 8))
        planner.seed_costs(target_samples=((2, 10.0), (4, 20.0), (8, 40.0)),
                           proposal_samples=((1, 1.0),))
        planner._target_calibration.values.update({2: 1.2, 4: 1.0, 8: 1.1})
        planner._zero_overhead.values[1] = 2.0
        self.assertAlmostEqual(planner._zero_score(4), 4.0 / 26.0)
        for _ in range(3):
            planner.observe_cost(target_rows=4, target_ms=16.0, proposal_samples=(),
                                 overhead_ms=2.0)
        self.assertAlmostEqual(planner._zero_score(4), 4.0 / (19.2 + 2.0))

    def test_measured_bucket_cost_removes_false_k3_barrier(self) -> None:
        # Frozen Dense public-trace survival: isolate cost estimation from text/RNG.
        survival = (0.8844, 0.8037, 0.6337, 0.4996, 0.3939, 0.3106, 0.2449)
        policy = DynamicDraftPolicy(
            3, 7, DraftPolicyConfig(decision_tokens=1, decision_min_rounds=1)
        )
        policy.observe(accepted=3, proposed=3)
        planner = DynamicDraftPlanner((1, 2, 4, 8))
        planner.seed_costs(
            target_samples=((1, 21.102), (2, 18.684), (4, 18.476), (8, 29.654)),
            proposal_samples=((1, 1.25),),
        )
        planner.overhead_cost.values.update({2: 1.045, 4: 1.358})
        with patch.object(DynamicDraftPolicy, "survival_probabilities", return_value=survival):
            self.assertEqual(planner.plan((policy,), row_limit=8), (3,))
            planner.target_cost.values[8] = 21.7
            target = planner.plan((policy,), row_limit=8)[0]
            self.assertGreater(target, 3)
            self.assertEqual(policy.move_toward(target).next_tokens, 4)

    def test_seeded_curve_uses_a_three_sample_median_before_adapting(self) -> None:
        planner = DynamicDraftPlanner((1, 2, 4))
        planner.seed_costs(
            target_samples=((2, 5.0),),
            proposal_samples=((1, 1.0),),
        )

        planner.observe_cost(
            target_rows=2,
            target_ms=50.0,
            proposal_samples=((1, 10.0),),
            overhead_ms=1.0,
        )
        self.assertEqual(planner.target_cost.values[2], 5.0)
        self.assertEqual(planner.proposal_cost.values[1], 1.0)
        planner.observe_cost(
            target_rows=2,
            target_ms=6.0,
            proposal_samples=((1, 1.2),),
            overhead_ms=1.0,
        )
        planner.observe_cost(
            target_rows=2,
            target_ms=5.5,
            proposal_samples=((1, 1.1),),
            overhead_ms=1.0,
        )

        self.assertAlmostEqual(planner.target_cost.values[2], 5.2)
        self.assertAlmostEqual(planner.proposal_cost.values[1], 1.04)

    def test_runtime_overhead_is_isolated_by_graph_load(self) -> None:
        planner = DynamicDraftPlanner((1, 2, 4, 8))
        planner.observe_cost(
            target_rows=1,
            target_ms=4.0,
            proposal_samples=((1, 1.0),),
            overhead_ms=0.5,
        )
        planner.observe_cost(
            target_rows=8,
            target_ms=20.0,
            proposal_samples=((8, 8.0),),
            overhead_ms=12.0,
        )

        self.assertEqual(planner.overhead_cost.values, {1: 0.5, 8: 12.0})

    def test_fixed_decode_requests_consume_the_shared_row_budget(self) -> None:
        policy = DynamicDraftPolicy(
            1, 3, DraftPolicyConfig(decision_tokens=1, decision_min_rounds=1)
        )
        policy.observe(accepted=0, proposed=1)
        planner = DynamicDraftPlanner((1, 2, 4, 8))
        planner.seed_costs(
            target_samples=((4, 4.0),),
            proposal_samples=((1, 20.0),),
        )

        self.assertEqual(
            planner.plan((policy,), row_limit=4, fixed_decode_requests=3),
            (0,),
        )

    def test_prefill_rows_price_the_actual_target_graph_bucket(self) -> None:
        policy = DynamicDraftPolicy(
            1, 3, DraftPolicyConfig(decision_tokens=1, decision_min_rounds=1)
        )
        policy.observe(accepted=1, proposed=1)
        planner = DynamicDraftPlanner((1, 2, 4, 8))
        planner.seed_costs(
            target_samples=((1, 4.0), (2, 4.0), (4, 4.0), (8, 100.0)),
            proposal_samples=((1, 0.01),),
        )

        self.assertEqual(planner.plan((policy,), row_limit=4), (3,))
        self.assertLess(
            planner.plan(
                (policy,),
                row_limit=4,
                target_extra_rows=2,
            )[0],
            3,
        )


if __name__ == "__main__":
    unittest.main()


class CoherentCompleteRoundTests(unittest.TestCase):
    def planner(self):
        config = DraftPolicyConfig(decision_tokens=1, decision_min_rounds=1)
        planner = DynamicDraftPlanner((1, 2, 4, 8, 16), config=config)
        planner.seed_costs(target_samples=((1, 2.), (2, 3.), (4, 4.), (8, 20.), (16, 35.)),
                           proposal_samples=((1, 2.), (2, 3.), (4, 5.)))
        planner.seed_compound_costs(((8, 4, 3.),))
        return planner, config

    def test_coherent_goal_does_not_consume_individual_observation_windows(self):
        planner, config = self.planner()
        policies = (DynamicDraftPolicy(1, 3, config), DynamicDraftPolicy(1, 3, config))
        policies[0].observe(accepted=1, proposed=1)
        snapshot = tuple((p.current_tokens, p._rounds, p._window_tokens,
                          p.survival_probabilities()) for p in policies)
        self.assertEqual(planner.plan(policies, row_limit=8), (3, 1))
        self.assertEqual(planner.plan(policies, row_limit=8, coherent_rounds=True), (3, 3))
        self.assertEqual(snapshot, tuple((p.current_tokens, p._rounds, p._window_tokens,
                                         p.survival_probabilities()) for p in policies))
        self.assertTrue(planner.has_compound_cost(2, 3))
        self.assertFalse(planner.has_compound_cost(3, 3))
        self.assertFalse(planner.has_compound_cost(2, 2))

    def test_complete_transition_is_positive_bounded_and_preserves_survival(self):
        _, config = self.planner()
        policy = DynamicDraftPolicy(1, 3, config)
        policy.observe(accepted=1, proposed=1)
        survival = policy.survival_probabilities()
        decision = policy.move_toward(3, complete_round=True)
        self.assertEqual((decision.action, decision.previous_tokens, decision.next_tokens),
                         (DraftAction.PROMOTE, 1, 3))
        self.assertEqual(policy.survival_probabilities(), survival)
        self.assertFalse(policy.ready or policy.stopped)
        policy.observe(accepted=2, proposed=3)
        self.assertEqual(policy.move_toward(1, complete_round=True).next_tokens, 1)
        policy.observe(accepted=1, proposed=1)
        before = (policy.current_tokens, policy._rounds, policy._window_tokens)
        for target in (0, 4, -1):
            with self.assertRaises(ValueError):
                policy.move_toward(target, complete_round=True)
            self.assertEqual(before, (policy.current_tokens, policy._rounds, policy._window_tokens))
        self.assertEqual(policy.move_toward(0).next_tokens, 0)
        self.assertTrue(policy.stopped)

    def test_coherent_goal_needs_a_complete_positive_pure_decode_geometry(self):
        planner, config = self.planner()
        policies = (DynamicDraftPolicy(1, 3, config), DynamicDraftPolicy(1, 3, config))
        policies[0].observe(accepted=1, proposed=1)
        for extras in ({'fixed_decode_requests': 1}, {'target_extra_rows': 1},
                       {'available_decode_requests': 3}):
            targets = planner.plan(policies, row_limit=16, coherent_rounds=True, **extras)
            self.assertEqual(targets[1], 1)
        planner._seed_compound.clear()
        targets = planner.plan(policies, row_limit=8, coherent_rounds=True)
        self.assertEqual(targets, (0, 0))
        self.assertFalse(planner.has_compound_cost(2, 3))
        for args in ((0, 1), (1, 0), (True, 1)):
            with self.assertRaises(ValueError):planner.has_compound_cost(*args)

    def test_terminal_exit_waits_for_all_owners_without_spending_observations(self):
        for count in (2, 4, 8):
            for ready_owner in (0, count - 1):
                with self.subTest(count=count, ready_owner=ready_owner):
                    config = DraftPolicyConfig(decision_tokens=1, decision_min_rounds=1)
                    planner = DynamicDraftPlanner((1, 2, 4, 8, 16, 32), config=config)
                    planner.seed_costs(target_samples=((n, 1.) for n in planner.graph_capacities),
                                       proposal_samples=((1, 10.), (2, 10.), (4, 10.), (8, 10.)))
                    policies = tuple(DynamicDraftPolicy(1, 3, config) for _ in range(count))
                    policies[ready_owner].observe(accepted=1, proposed=1)
                    before = tuple((p.current_tokens, p._rounds, p._window_tokens,
                                    p.survival_probabilities()) for p in policies)
                    self.assertEqual(planner.plan(policies, row_limit=count * 4,
                                                  coherent_rounds=True), (0,) * count)
                    self.assertEqual(before, tuple((p.current_tokens, p._rounds,
                                                    p._window_tokens, p.survival_probabilities())
                                                   for p in policies))
                    for index, policy in enumerate(policies):
                        if index != ready_owner:
                            policy.observe(accepted=1, proposed=1)
                    targets = planner.plan(policies, row_limit=count * 4,
                                           coherent_rounds=True)
                    self.assertEqual(targets, (0,) * count)
                    for policy, target in zip(policies, targets, strict=True):
                        self.assertEqual(policy.move_toward(target).next_tokens, 0)
                        self.assertTrue(policy.stopped)
                        with self.assertRaises(RuntimeError):
                            policy.move_toward(1)
                        with self.assertRaises(RuntimeError):
                            policy.observe(accepted=1, proposed=1)

    def test_terminal_wait_does_not_apply_to_partial_or_mixed_decode_cohorts(self):
        config = DraftPolicyConfig(decision_tokens=1, decision_min_rounds=1)
        planner = DynamicDraftPlanner((1, 2, 4, 8, 16), config=config)
        planner.seed_costs(target_samples=((n, 1.) for n in planner.graph_capacities),
                           proposal_samples=((1, 10.), (2, 10.), (4, 10.)))
        policies = (DynamicDraftPolicy(1, 3, config), DynamicDraftPolicy(1, 3, config))
        policies[0].observe(accepted=1, proposed=1)
        for extras, expected in (
            ({'coherent_rounds': False}, (0, 1)),
            ({'coherent_rounds': True, 'fixed_decode_requests': 1}, (0, 1)),
            ({'coherent_rounds': True, 'target_extra_rows': 1}, (1, 1)),
            ({'coherent_rounds': True, 'available_decode_requests': 3}, (0, 1)),
        ):
            with self.subTest(extras=extras):
                self.assertEqual(planner.plan(policies, row_limit=16, **extras), expected)

    def test_terminal_goal_keeps_adjacent_depth_transitions_coherent(self):
        for initial in (2, 3):
            with self.subTest(initial=initial):
                config = DraftPolicyConfig(decision_tokens=1, decision_min_rounds=1)
                planner = DynamicDraftPlanner((1, 2, 4, 8, 16), config=config)
                planner.seed_costs(target_samples=((n, 1.) for n in planner.graph_capacities),
                                   proposal_samples=((1, 10.), (2, 10.), (4, 10.)))
                policies = tuple(DynamicDraftPolicy(initial, 3, config) for _ in range(2))
                for depth in range(initial, 0, -1):
                    policies[0].observe(accepted=depth, proposed=depth)
                    self.assertEqual(planner.plan(policies, row_limit=16, coherent_rounds=True),
                                     (0, 0))
                    policies[1].observe(accepted=0, proposed=depth)
                    targets = planner.plan(policies, row_limit=16, coherent_rounds=True)
                    self.assertEqual(targets, (0, 0))
                    for policy, target in zip(policies, targets, strict=True):
                        self.assertEqual(policy.move_toward(target).next_tokens, depth - 1)
                        self.assertEqual(policy.stopped, depth == 1)

    def test_mixed_current_depths_keep_individual_terminal_decision_timing(self):
        for depths in ((1, 2), (3, 1), (2, 3)):
            for ready_owner in (0, 1):
                with self.subTest(depths=depths, ready_owner=ready_owner):
                    config = DraftPolicyConfig(decision_tokens=1, decision_min_rounds=1)
                    planner = DynamicDraftPlanner((1, 2, 4, 8, 16), config=config)
                    planner.seed_costs(target_samples=((n, 1.) for n in planner.graph_capacities),
                                       proposal_samples=((1, 10.), (2, 10.), (4, 10.)))
                    policies = tuple(DynamicDraftPolicy(depth, 3, config) for depth in depths)
                    policies[ready_owner].observe(accepted=1, proposed=depths[ready_owner])
                    expected = tuple(0 if i == ready_owner else depth
                                     for i, depth in enumerate(depths))
                    self.assertEqual(planner.plan(policies, row_limit=16, coherent_rounds=True),
                                     expected)


class DraftDecisionHorizonTests(unittest.TestCase):
    def test_bounds_match_independent_acceptance_simulation_and_row_permutations(self):
        rng = Random(20261024)
        for count in (1, 2, 4, 8, 17, 64, 128, 256):
            for depth in (1, 2, 3):
                for repetition in range(4):
                    with self.subTest(count=count, depth=depth, repetition=repetition):
                        policies = tuple(DynamicDraftPolicy(depth, 3) for _ in range(count))
                        for policy in policies:
                            for _ in range(rng.randrange(25)):
                                policy.observe(accepted=rng.randrange(depth + 1), proposed=depth)
                        initial = deepcopy(policies)
                        horizon = draft_decision_horizon(policies, coherent_rounds=True)
                        self.assertEqual(policies, initial)
                        for coherent in (False, True):
                            reference = deepcopy(initial)
                            ready = all if coherent else any
                            rounds = 0
                            while not ready(policy.ready for policy in reference):
                                for policy in reference:
                                    policy.observe(accepted=depth, proposed=depth)
                                rounds += 1
                            self.assertEqual(draft_decision_horizon(
                                initial, coherent_rounds=coherent), rounds)
                        # Actual acceptance can delay the decision, but cannot
                        # make the complete cohort ready before this bound.
                        for _ in range(horizon):
                            self.assertFalse(all(policy.ready for policy in policies))
                            for policy in policies:
                                policy.observe(accepted=rng.randrange(depth + 1), proposed=depth)
                        self.assertEqual(draft_decision_horizon(
                            tuple(reversed(initial)), coherent_rounds=True), horizon)
        self.assertEqual(draft_decision_horizon(()), 0)
        self.assertEqual(draft_decision_horizon((), coherent_rounds=True), 0)


class MaskOverlapBudgetTests(unittest.TestCase):
    def test_service_waits_count_when_active_cpu_time_underprices_mask_latency(self):
        from types import SimpleNamespace as S
        from cubic_serve.runtime import performance

        for prefix in ((), (2,), (2, 3, 4)):
            for limit, expected in ((1, [1, 1, 1]), (2, [1, 2, 2]),
                                    (3, [1, 2, 3]), (4, [1, 2, 4])):
                with self.subTest(prefix=prefix, limit=limit):
                    scale = len(prefix) + 1
                    clock = S(wall=0, cpu=0)
                    clock.monotonic_ns = lambda: clock.wall
                    clock.process_time_ns = lambda: clock.cpu
                    selected, witness = [], object()
                    proposals = {str(owner): prefix for owner in range(8)}

                    def prepare(actual, *, workers):
                        self.assertIs(actual, proposals)
                        selected.append(workers)
                        # Service includes serial/copy/wait time: active CPU
                        # alone would keep choosing one despite a missed window.
                        clock.wall += {1: 20, 2: 12, 3: 9, 4: 8}[workers] * scale * 1_000_000
                        clock.cpu += {1: 5, 2: 8, 3: 9, 4: 10}[workers] * scale * 1_000_000
                        return witness

                    grammar = S(pool=S(batch_mask_workers=limit),
                                target_prefix_masks_many=prepare)
                    state = S(_constraint_forward_ms=10. * scale)
                    with patch.object(performance, 'time', clock):
                        for _ in range(3):
                            state._constraint_forward_queued_ns = clock.wall
                            self.assertIs(performance._constraint_prefix_masks_with_overlap(
                                state, grammar, proposals), witness)
                    self.assertEqual(selected, expected)

    def test_absent_mask_work_uses_no_clock_or_pool(self):
        from types import SimpleNamespace as S
        from cubic_serve.runtime import performance

        def forbidden():
            self.fail('absent grammar work touched a resource clock')

        with patch.object(performance, 'time', S(monotonic_ns=forbidden,
                                               process_time_ns=forbidden)):
            self.assertEqual(performance._constraint_prefix_masks_with_overlap(
                object(), object(), {}), {})


class CompleteRoundTemporalWeightTests(unittest.TestCase):
    def test_window_handoffs_do_not_falsely_terminate_faster_drafting(self):
        from cubic_serve.runtime.speculative import _EmaCurve
        old, complete = _EmaCurve(), _EmaCurve()
        old.values[8] = complete.values[8] = 54.
        for _ in range(90):
            for price, steps in ((54., 8), (110., 1)):
                old.observe(8, price)
                complete.observe_complete(8, price, completed_steps=steps)
        cfg = DraftPolicyConfig(decision_tokens=1, decision_min_rounds=1)
        def decision(curve):
            planner = DynamicDraftPlanner((1, 2, 4, 8), config=cfg)
            planner.seed_costs(target_samples=((4, 10.), (8, 20.)),
                               proposal_samples=((4, 4.),))
            planner.compound_cost[2] = curve
            planner.observe_resident_cost(target_rows=4, milliseconds=41.)
            owners = tuple(DynamicDraftPolicy(1, 1, config=cfg) for _ in range(4))
            for owner in owners:
                owner.observe(accepted=1, proposed=1)
            return planner.plan(owners, row_limit=8, coherent_rounds=True)
        self.assertEqual(decision(old), (0,) * 4)
        self.assertEqual(decision(complete), (1,) * 4)

    def test_complete_cost_is_weighted_by_executed_work(self):
        from cubic_serve.runtime.speculative import _EmaCurve
        for steps in (1, 2, 3, 8, 16, 31, 1024):
            curve = _EmaCurve(alpha=1., values={8: 60.})
            for price, count in ((54., steps), (110., 1), (100., 1)):
                curve.observe_complete(8, price, completed_steps=count)
            self.assertAlmostEqual(curve.values[8], (54. * steps + 210.) / (steps + 2))
            self.assertFalse(curve._pending[8])

    def test_runtime_reset_clears_the_existing_complete_sample_buffer(self):
        planner = DynamicDraftPlanner((1, 2, 4, 8))
        planner.seed_costs(target_samples=((4, 10.), (8, 20.)),
                           proposal_samples=((4, 4.),))
        planner.observe_compound_cost(target_rows=8, segment_depth=2,
                                      milliseconds=54., completed_steps=8)
        planner.observe_compound_cost(target_rows=8, segment_depth=2,
                                      milliseconds=110., completed_steps=1)
        self.assertTrue(planner.compound_cost[2]._pending)
        planner.reset_runtime_costs()
        self.assertFalse(planner.compound_cost[2]._pending)

    def test_complete_timing_rejects_invalid_step_metadata(self):
        from cubic_serve.runtime.speculative import _EmaCurve
        for steps in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                _EmaCurve().observe_complete(8, 1., completed_steps=steps)


@pytest.mark.parametrize('depth', [1, 2, 3, 7])
def test_reservation_bound_covers_rejected_proposals_without_moving_execution_boundary(depth):
    import copy
    from cubic_serve.runtime.speculative import DynamicDraftPolicy

    policy = DynamicDraftPolicy(depth, depth)
    for accepted in (0, depth, 0):
        policy.observe(accepted=accepted, proposed=depth)
    earliest = policy.rounds_until_decision
    latest = policy.maximum_rounds_until_decision
    for accepted in (0, depth):
        witness = copy.deepcopy(policy)
        rounds = 0
        while not witness.ready:
            witness.observe(accepted=accepted, proposed=depth)
            rounds += 1
        assert earliest <= rounds <= latest
        assert witness.rounds_until_decision == witness.maximum_rounds_until_decision == 0
    assert policy.rounds_until_decision == earliest


def test_capacity_stop_is_terminal_without_fabricating_decision_readiness():
    policy=DynamicDraftPolicy(3,3)
    policy.observe(accepted=1,proposed=3)
    before=(policy._rounds,policy._window_tokens,tuple(policy._successes),tuple(policy._trials))
    assert not policy.ready
    decision=policy.stop_for_capacity()
    assert decision.next_tokens==0 and policy.stopped and policy.current_tokens==0
    assert (policy._rounds,policy._window_tokens,tuple(policy._successes),tuple(policy._trials))==before
    with pytest.raises(RuntimeError,match='cannot resume'):policy.observe(accepted=0,proposed=1)
