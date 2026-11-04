"""Provider-independent, load-aware speculative draft planning."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, fields
from cubic_serve._compat import StrEnum
from math import isfinite


class DraftAction(StrEnum):
    PROMOTE = "promote"
    HOLD = "hold"
    DEMOTE = "demote"


@dataclass(frozen=True, slots=True)
class DraftPolicyConfig:
    decision_tokens: int = 24
    decision_min_rounds: int = 8
    prior_success: float = 4.0
    prior_failure: float = 1.0
    history_decay: float = 0.99
    cost_ema_alpha: float = 0.2
    cost_sample_window: int = 3
    terminal_stop_margin: float = 0.03

    @classmethod
    def from_dict(cls, value: object) -> DraftPolicyConfig:
        if not isinstance(value, dict) or set(value) - {f.name for f in fields(cls)}:
            raise ValueError("unknown dynamic draft policy fields")
        return cls(**value)

    def __post_init__(self) -> None:
        for value in (
            self.prior_success,
            self.prior_failure,
            self.history_decay,
            self.cost_ema_alpha,
            self.terminal_stop_margin,
        ):
            if type(value) not in (int, float) or not isfinite(value):
                raise ValueError("draft policy parameters must be finite numbers")
        if type(self.cost_sample_window) is not int or self.cost_sample_window not in (1, 3, 5, 7):
            raise ValueError("draft cost sample window must be 1, 3, 5, or 7")
        if self.terminal_stop_margin < 0.0:
            raise ValueError("terminal stop margin must be non-negative")
        if not 0.0 < self.cost_ema_alpha <= 1.0:
            raise ValueError("draft cost EMA alpha must lie in (0, 1]")
        if type(self.decision_tokens) is not int or self.decision_tokens < 1:
            raise ValueError("draft decision token interval must be positive")
        if type(self.decision_min_rounds) is not int or self.decision_min_rounds < 1:
            raise ValueError("draft minimum decision rounds must be positive")
        if self.prior_success <= 0.0 or self.prior_failure <= 0.0:
            raise ValueError("draft survival prior must be positive")
        if not 0.0 < self.history_decay < 1.0:
            raise ValueError("draft survival history decay must lie between zero and one")


@dataclass(frozen=True, slots=True)
class DraftDecision:
    action: DraftAction
    previous_tokens: int
    next_tokens: int


@dataclass(slots=True)
class DynamicDraftPolicy:
    """One request's conditional draft-survival observations and K state."""

    initial_tokens: int
    max_tokens: int
    config: DraftPolicyConfig = field(default_factory=DraftPolicyConfig)
    current_tokens: int = field(init=False)
    _successes: list[float] = field(init=False)
    _trials: list[float] = field(init=False)
    _window_tokens: int = field(init=False, default=0)
    _rounds: int = field(init=False, default=0)
    _stopped: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if (
            type(self.initial_tokens) is not int
            or type(self.max_tokens) is not int
            or not 1 <= self.initial_tokens <= self.max_tokens
        ):
            raise ValueError("draft bounds must satisfy 1 <= initial <= maximum")
        self.current_tokens = self.initial_tokens
        self._successes = [0.0] * self.max_tokens
        self._trials = [0.0] * self.max_tokens

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def ready(self) -> bool:
        return (
            self._rounds >= self.config.decision_min_rounds
            and self._window_tokens >= self.config.decision_tokens
        )

    @property
    def rounds_until_decision(self) -> int:
        """Earliest individual boundary if every remaining proposal accepts.

        Sealed transactions combine these lower bounds with
        :func:`draft_decision_horizon` for their eligible decision boundary.
        """
        if self.stopped or self.ready:
            return 0
        tokens = max(0, self.config.decision_tokens - self._window_tokens)
        stride = self.current_tokens + 1
        return max(
            self.config.decision_min_rounds - self._rounds,
            (tokens + stride - 1) // stride,
        )

    @property
    def decision_budget(self) -> tuple[int, int]:
        """Unconsumed emitted tokens and rounds at the current observation window."""
        return (max(0, self.config.decision_tokens - self._window_tokens),
                max(0, self.config.decision_min_rounds - self._rounds))

    @property
    def maximum_rounds_until_decision(self) -> int:
        """Latest boundary: every completed round emits at least one target token.

        Page admission uses this upper bound. Uniform, qualified cohorts may
        also grant it to workers that stop at the actual all-owner boundary;
        other execution retains the earliest conservative checkpoint.
        """
        if self.stopped or self.ready:
            return 0
        return max(
            0, self.config.decision_min_rounds - self._rounds,
            self.config.decision_tokens - self._window_tokens,
        )

    def observe(self, *, accepted: int, proposed: int) -> None:
        if self._stopped:
            raise RuntimeError("draft observations cannot resume after K0")
        if not 1 <= proposed <= self.current_tokens or not 0 <= accepted <= proposed:
            raise ValueError("draft observation differs from the active request depth")
        decay = self.config.history_decay
        for depth in range(self.max_tokens):
            self._trials[depth] *= decay
            self._successes[depth] *= decay
        for depth in range(1, proposed + 1):
            if accepted < depth - 1:
                break
            self._trials[depth - 1] += 1.0
            self._successes[depth - 1] += float(accepted >= depth)
        self._rounds += 1
        self._window_tokens += accepted + 1

    def survival_probabilities(self) -> tuple[float, ...]:
        """Return monotone P(accepted_length >= depth) through max K."""

        survival = 1.0
        result: list[float] = []
        conditional = self.config.prior_success / (
            self.config.prior_success + self.config.prior_failure
        )
        for successes, trials in zip(self._successes, self._trials, strict=True):
            if trials:
                conditional = (self.config.prior_success + successes) / (
                    self.config.prior_success + self.config.prior_failure + trials
                )
            survival *= conditional
            result.append(survival)
        return tuple(result)

    def stop_for_capacity(self) -> DraftDecision:
        """End optional drafting at a committed boundary without inventing observations.

        K0 is terminal in the carried-provider contract. Resource admission can
        require it before a cost-observation window is ready; it must never
        silently resume a provider whose state no longer advances with target.
        """
        previous=self.current_tokens
        self.current_tokens=0
        self._stopped=True
        return DraftDecision(DraftAction.HOLD if previous==0 else DraftAction.DEMOTE,previous,0)

    def move_toward(self, target_tokens: int, *, complete_round: bool = False) -> DraftDecision:
        """Use an adjacent transition, or an engine-qualified complete round."""

        if self._stopped:
            raise RuntimeError("K0 requests no longer participate in draft decisions")
        if type(target_tokens) is not int or not 0 <= target_tokens <= self.max_tokens:
            raise ValueError("draft target lies outside request bounds")
        if type(complete_round) is not bool or complete_round and target_tokens == 0:
            raise ValueError("complete draft transitions require a positive qualified depth")
        previous = self.current_tokens
        if not self.ready:
            return DraftDecision(DraftAction.HOLD, previous, previous)
        self._rounds = 0
        self._window_tokens = 0
        if target_tokens == previous:
            return DraftDecision(DraftAction.HOLD, previous, previous)
        next_tokens = (target_tokens if complete_round else
                       previous + (1 if target_tokens > previous else -1))
        self.current_tokens = next_tokens
        if next_tokens == 0:
            self._stopped = True
        action = DraftAction.PROMOTE if next_tokens > previous else DraftAction.DEMOTE
        return DraftDecision(action, previous, next_tokens)


@dataclass(slots=True)
class _EmaCurve:
    alpha: float = 0.2
    sample_window: int = 3
    values: dict[int, float] = field(default_factory=dict)
    _pending: dict[int, list[tuple[float, int]]] = field(default_factory=dict)
    allow_zero: bool = False

    def observe(self, capacity: int, milliseconds: float) -> None:
        if capacity <= 0 or milliseconds < 0.0 or (milliseconds == 0.0 and not self.allow_zero):
            return
        previous = self.values.get(capacity)
        if previous is None:
            self.values[capacity] = milliseconds
            return
        samples = self._pending.setdefault(capacity, [])
        samples.append((milliseconds, 1))
        if len(samples) < self.sample_window:
            return
        median = sorted(samples)[len(samples) // 2][0]
        self.values[capacity] = previous + self.alpha * (median - previous)
        samples.clear()

    def observe_complete(
        self, capacity: int, milliseconds: float, *, completed_steps: int = 1,
    ) -> None:
        """Average inclusive time by executed rounds, then smooth complete groups.

        One multi-round window and one host-dispatched round are observations
        of different amounts of work. Their per-round prices cannot receive
        equal weight. Operator-local single-call curves retain ``observe``.
        """
        if (type(completed_steps) is not int or completed_steps < 1
                or type(capacity) is not int or capacity < 1
                or type(milliseconds) not in (int, float) or not isfinite(milliseconds)
                or milliseconds < 0.0 or milliseconds == 0.0 and not self.allow_zero):
            raise ValueError("complete timing requires finite cost and executed rounds")
        previous = self.values.get(capacity)
        if previous is None:
            self.values[capacity] = float(milliseconds)
            return
        samples = self._pending.setdefault(capacity, [])
        samples.append((milliseconds, completed_steps))
        if len(samples) < self.sample_window:
            return
        mean = sum(price * steps for price, steps in samples) / sum(steps for _, steps in samples)
        self.values[capacity] = previous + self.alpha * (mean - previous)
        samples.clear()

    def estimate(self, capacity: int) -> float | None:
        if not self.values:
            return None
        if capacity in self.values:
            return self.values[capacity]
        points = sorted(self.values.items())
        lower = max((point for point in points if point[0] < capacity), default=None)
        upper = min((point for point in points if point[0] > capacity), default=None)
        if lower and upper:
            fraction = (capacity - lower[0]) / (upper[0] - lower[0])
            return lower[1] + fraction * (upper[1] - lower[1])
        if upper is not None:
            # Low-row decode is commonly bandwidth-bound: scaling the first
            # observed graph cost down by M invents savings that do not exist.
            return upper[1]
        assert lower is not None
        return lower[1] * capacity / lower[0]


@dataclass(slots=True)
class DynamicDraftPlanner:
    """dSpARk-style expected-output/step-cost allocator."""

    graph_capacities: tuple[int, ...]
    config: DraftPolicyConfig = field(default_factory=DraftPolicyConfig)
    target_cost: _EmaCurve = field(default_factory=_EmaCurve)
    target_depth_cost: dict[int, _EmaCurve] = field(default_factory=dict)
    proposal_cost: _EmaCurve = field(default_factory=_EmaCurve)
    overhead_cost: _EmaCurve = field(default_factory=_EmaCurve)
    compound_cost: dict[int, _EmaCurve] = field(default_factory=dict)
    _target_calibration: _EmaCurve = field(default_factory=_EmaCurve)
    _zero_overhead: _EmaCurve = field(default_factory=_EmaCurve)
    _resident_zero_cost: _EmaCurve = field(default_factory=_EmaCurve)
    _resident_zero_overhead: _EmaCurve = field(
        default_factory=lambda: _EmaCurve(allow_zero=True),
    )
    _seed_target: dict[int, float] = field(init=False, default_factory=dict)
    _seed_target_depth: dict[int, dict[int, float]] = field(init=False, default_factory=dict)
    _seed_proposal: dict[int, float] = field(init=False, default_factory=dict)
    _seed_compound: dict[tuple[int, int], float] = field(init=False, default_factory=dict)
    require_complete_rounds: bool = False
    # CPU request work is indexed by actual masked occupancy, not padded M.
    # Values: state copies/publication, accept per output, mask per reachable
    # prefix, controller handoff, unattributed native/worker remainder, first
    # mask per owner (K0 has no draft-prefix advancement).
    _request_work: dict[int, tuple[float, ...]] = field(default_factory=dict)
    _seed_constraint_sampling: dict[int, float] = field(default_factory=dict)
    _seed_constraint_greedy: dict[int, float] = field(default_factory=dict)
    _request_proposal_observed: set[int] = field(default_factory=set)
    _request_stage_overhead: dict[int, _EmaCurve] = field(default_factory=dict)

    def seed_constraint_sampling_costs(
        self, samples: Iterable[tuple[int, float]], *,
        greedy_samples: Iterable[tuple[int, float]] = (),
    ) -> None:
        def read(samples):
            values = {}
            for rows, milliseconds in samples:
                if (type(rows) is not int or rows not in self.graph_capacities or rows in values
                        or type(milliseconds) not in (int, float)
                        or not isfinite(milliseconds) or milliseconds <= 0):
                    raise ValueError("invalid qualified masked sampling profile")
                values[rows] = float(milliseconds)
            return values
        values, greedy_values = read(samples), read(greedy_samples)
        self._seed_constraint_sampling = values
        self._seed_constraint_greedy = greedy_values

    def _request_native_price(self, seed: float, rows: int, depth: int) -> float:
        capacity = self.bucket(rows)
        scale = self._target_scale(capacity)
        body = self._seed_target_depth.get(depth, {}).get(capacity)
        # A target-body calibration does not calibrate sampler/transport or
        # fixed host work. Keep unknown cold fixtures on the existing prior.
        return (seed * scale if body is None else
                seed + min(seed, body) * (scale - 1.0))

    def observe_request_round(
        self, *, target_rows: int, proposal_rows: tuple[int, ...],
        target_ms: float, worker_ms: float, round_ms: float, grammar_cost,
        ingress: bool = False, steps: int = 1,
    ) -> None:
        """Reconcile one immutable worker witness with its controller receipt.

        Complete prices remain inclusive. Only missing cold positive-depth
        prices use this decomposition. K0 needs both a qualified masked sampler
        prior and live request work; a target-body seed alone is incomplete.
        """
        if not self.require_complete_rounds or ingress or grammar_cost is None:
            return
        if (type(steps) is not int or steps < 1 or not grammar_cost.valid()
                or any(type(value) not in (int, float) or not isfinite(value) or value < 0
                       for value in (target_ms, worker_ms, round_ms))
                or round_ms <= 0 or worker_ms > round_ms):
            raise ValueError("request cost requires one coherent completed window")
        owners = target_rows - sum(proposal_rows)
        if owners != len(grammar_cost.owners) or owners < 1:
            # A partially masked cohort cannot train an all-masked cohort.
            return
        self.bucket(target_rows)
        if not proposal_rows:
            self.observe_resident_cost(target_rows=owners, milliseconds=round_ms,
                                       target_ms=min(target_ms, round_ms))
            return
        depth = len(proposal_rows) + 1
        if (len(set(proposal_rows)) != 1 or proposal_rows[0] != owners
                or not grammar_cost.mask_calls or not grammar_cost.output_tokens):
            return
        seed = self._compound_seed_estimate(target_rows, depth)
        if seed is None:
            return
        capacity = self.bucket(target_rows)
        target_seed = self._seed_target_depth.get(depth, {}).get(capacity)
        if target_seed and target_ms > 0:
            self._target_calibration.observe(capacity, target_ms / target_seed)
        shared = (grammar_cost.begin_ms + grammar_cost.fork_ms + grammar_cost.publish_ms) / steps
        mask = (grammar_cost.mask_ms + grammar_cost.advance_ms) / steps
        publication = max(0.0, round_ms - worker_ms)
        native = self._request_native_price(seed, target_rows, depth)
        # GPU event intervals may include stream waits. The mask/target pair
        # overlaps; adding both would count that critical path twice.
        exposed_mask = max(0.0, mask - target_ms)
        remainder = max(0.0, round_ms - native - shared - grammar_cost.accept_ms / steps
                        - exposed_mask - publication)
        sample = (shared, grammar_cost.accept_ms / grammar_cost.output_tokens,
                  (grammar_cost.mask_ms + grammar_cost.advance_ms) / grammar_cost.mask_calls,
                  publication, remainder, grammar_cost.first_mask_ms / (steps * owners))
        previous = self._request_work.get(owners)
        alpha = self.config.cost_ema_alpha
        self._request_work[owners] = (sample if previous is None else tuple(
            (1.0 - alpha) * old + alpha * new for old, new in zip(previous, sample, strict=True)
        ))

    def __post_init__(self) -> None:
        if (
            not self.graph_capacities
            or tuple(sorted(set(self.graph_capacities))) != self.graph_capacities
        ):
            raise ValueError("planner graph capacities must be sorted and unique")
        for curve in (
            self.target_cost,
            self.proposal_cost,
            self.overhead_cost,
            self._target_calibration,
            self._zero_overhead,
            self._resident_zero_cost,
            self._resident_zero_overhead,
        ):
            curve.alpha = self.config.cost_ema_alpha
            curve.sample_window = self.config.cost_sample_window
        # Each resident observation already averages one complete steady
        # execution window, rather than one potentially jittery receipt.
        self._resident_zero_cost.sample_window = 1
        self._resident_zero_overhead.sample_window = 1

    def bucket(self, logical_rows: int) -> int:
        for capacity in self.graph_capacities:
            if logical_rows <= capacity:
                return capacity
        raise ValueError("dynamic draft plan exceeds graph capacity")

    def observe_cost(
        self,
        *,
        target_rows: int,
        target_ms: float,
        proposal_samples: Iterable[tuple[int, float]],
        overhead_ms: float,
        is_decode: bool = True,
        segment_depth: int = 1,
    ) -> None:
        # A multi-token recurrent prefill is a different workload from
        # independent decode/review rows, even at the same graph capacity.
        if not is_decode:
            return
        proposal_samples = tuple(proposal_samples)
        capacity = self.bucket(target_rows)
        if type(segment_depth) is not int or not 1 <= segment_depth <= target_rows:
            raise ValueError("target segment depth must fit the logical rows")
        if segment_depth == 1:
            self.target_cost.observe(capacity, target_ms)
        else:
            curve = self.target_depth_cost.setdefault(
                segment_depth,
                _EmaCurve(
                    alpha=self.config.cost_ema_alpha, sample_window=self.config.cost_sample_window
                ),
            )
            curve.observe(capacity, target_ms)
        for rows, milliseconds in proposal_samples:
            proposal_bucket = self.bucket(rows)
            if (self.require_complete_rounds
                    and proposal_bucket not in self._request_proposal_observed):
                # The cold prior has neither CPU matcher work nor stochastic
                # sampling. The first actual stage supersedes that prior.
                self.proposal_cost.values.pop(proposal_bucket, None)
                self.proposal_cost._pending.pop(proposal_bucket, None)
                self._request_proposal_observed.add(proposal_bucket)
            self.proposal_cost.observe(proposal_bucket, milliseconds)
        self.overhead_cost.observe(capacity, overhead_ms)
        if self.require_complete_rounds:
            curve = self._request_stage_overhead.setdefault(segment_depth, _EmaCurve(
                alpha=self.config.cost_ema_alpha, sample_window=self.config.cost_sample_window))
            curve.observe(capacity, overhead_ms)
        seed = self._seed_target_depth.get(segment_depth, self._seed_target).get(capacity)
        if is_decode and seed and target_ms > 0.0:
            self._target_calibration.observe(capacity, target_ms / seed)
        if is_decode and not proposal_samples:
            self._zero_overhead.observe(capacity, overhead_ms)

    def observe_resident_cost(
        self, *, target_rows: int, milliseconds: float, target_ms: float | None = None,
    ) -> None:
        """Complete target-only step price from one consumed steady window."""
        if (type(target_rows) is not int or target_rows <= 0
                or type(milliseconds) not in (int, float)
                or not isfinite(milliseconds) or milliseconds <= 0
                or (target_ms is not None and (
                    type(target_ms) not in (int, float) or not isfinite(target_ms)
                    or not 0 <= target_ms <= milliseconds))):
            raise ValueError("resident cost requires a coherent positive window price")
        self.bucket(target_rows)
        self._resident_zero_cost.observe(target_rows, milliseconds)
        if target_ms is not None:
            # A completed steady window supersedes ordinary transition costs
            # when pricing another resident K0 bucket. This includes all work
            # outside the measured review, not an attribution as pure host time.
            self._resident_zero_overhead.observe(target_rows, milliseconds - target_ms)

    def observe_compound_cost(
        self, *, target_rows: int, segment_depth: int, milliseconds: float,
        ingress: bool = False, completed_steps: int = 1,
        target_ms: float | None = None,
    ) -> None:
        """Complete carried-round price, distinct from its individual stages."""
        if (type(target_rows) is not int or type(segment_depth) is not int
                or not 2 <= segment_depth <= target_rows):
            raise ValueError("compound segment depth must fit the logical rows")
        if type(ingress) is not bool:
            raise ValueError("compound ingress must be an explicit boolean")
        if target_ms is not None and (
            type(target_ms) not in (int, float) or not isfinite(target_ms)
            or not 0 <= target_ms <= milliseconds
        ):
            raise ValueError("compound target time must belong to its complete round")
        if ingress:
            return
        curve = self.compound_cost.setdefault(segment_depth, _EmaCurve(
            alpha=self.config.cost_ema_alpha, sample_window=self.config.cost_sample_window,
        ))
        self.bucket(target_rows)  # Validate admission without aliasing padded workloads.
        curve.observe_complete(target_rows, milliseconds, completed_steps=completed_steps)
        if target_ms:
            capacity = self.bucket(target_rows)
            seed = self._seed_target_depth.get(segment_depth, {}).get(capacity)
            if seed:
                # Carried rounds bypass observe_cost, but their live target
                # still witnesses the existing cold-to-live correction. Without
                # it a learned full-round price competes with an untouched K0
                # prior. Never add this nested interval to the inclusive price.
                self._target_calibration.observe_complete(
                    capacity, target_ms / seed, completed_steps=completed_steps)

    def seed_compound_costs(self, samples: Iterable[tuple[int, int, float]]) -> None:
        """Cold worker-round priors from already-qualified captured transactions.

        These omit controller feeding, and are not serving measurements. The
        first actual complete-round observation supersedes its prior. An
        intermediate staged K cannot assign its host overhead to a different,
        already-captured complete transaction.
        """
        values = {}
        for rows, depth, milliseconds in samples:
            if (type(rows) is not int or not 0 < rows <= self.graph_capacities[-1]
                    or type(depth) is not int or not 2 <= depth <= rows or rows % depth
                    or type(milliseconds) not in (int, float)
                    or not isfinite(milliseconds) or milliseconds <= 0
                    or (rows, depth) in values):
                raise ValueError("invalid captured draft round cold profile")
            self.bucket(rows)
            values[rows, depth] = float(milliseconds)
        self._seed_compound = values

    def seed_costs(
        self,
        *,
        target_samples: Iterable[tuple[int, float]],
        proposal_samples: Iterable[tuple[int, float]],
        target_depth_samples: Iterable[tuple[int, int, float]] = (),
        compound_samples: Iterable[tuple[int, int, float]] = (),
    ) -> None:
        """Install costs measured from the already-qualified resident graphs."""

        for capacity, milliseconds in target_samples:
            self.target_cost.observe(capacity, milliseconds)
        for capacity, milliseconds in proposal_samples:
            self.proposal_cost.observe(capacity, milliseconds)
        self._seed_target = dict(self.target_cost.values)
        self._seed_proposal = dict(self.proposal_cost.values)
        for capacity, depth, milliseconds in target_depth_samples:
            if not 2 <= depth <= capacity:
                raise ValueError("target depth sample exceeds graph capacity")
            curve = self.target_depth_cost.setdefault(
                depth,
                _EmaCurve(
                    alpha=self.config.cost_ema_alpha, sample_window=self.config.cost_sample_window
                ),
            )
            curve.observe(capacity, milliseconds)
        self._seed_target_depth = {
            depth: dict(curve.values) for depth, curve in self.target_depth_cost.items()
        }

        self.seed_compound_costs(compound_samples)

    def new_cost_domain(self) -> DynamicDraftPlanner:
        """Share qualified startup prices, never another execution mode's history."""
        if not self._seed_target or not self._seed_proposal:
            raise RuntimeError("dynamic draft startup cost profile is incomplete")
        other = DynamicDraftPlanner(self.graph_capacities, config=self.config)
        other.require_complete_rounds = self.require_complete_rounds
        other.seed_costs(
            target_samples=self._seed_target.items(),
            proposal_samples=self._seed_proposal.items(),
            target_depth_samples=(
                (capacity, depth, milliseconds)
                for depth, values in self._seed_target_depth.items()
                for capacity, milliseconds in values.items()
            ),
        )
        return other

    def reset_runtime_costs(self) -> None:
        """Discard incomplete timing windows, retaining this executor's learned costs."""

        if not self._seed_target or not self._seed_proposal:
            raise RuntimeError("dynamic draft startup cost profile is incomplete")
        # Acceptance history is request-owned. Execution prices belong to the
        # still-loaded graphs; an idle interval does not make them cold again.
        for curve in (
            self.target_cost, self.proposal_cost, self.overhead_cost,
            self._target_calibration, self._zero_overhead,
            self._resident_zero_cost,
            self._resident_zero_overhead,
            *self.target_depth_cost.values(),
            *self.compound_cost.values(),
            *self._request_stage_overhead.values(),
        ):
            curve._pending.clear()

    def _target_estimate(self, capacity: int, segment_depth: int = 1) -> float | None:
        curve = self.target_depth_cost.get(segment_depth, self.target_cost)
        estimate = curve.estimate(capacity)
        seed = self._seed_target_depth.get(segment_depth, self._seed_target).get(capacity)
        if estimate is None or seed is None or estimate != seed:
            return estimate
        # Reprice cold buckets with observed decode slowdown. In particular,
        # never compare live review cost with an unrealistically cheap K0 seed.
        return max(estimate, seed * self._target_scale(capacity))

    def _target_scale(self, capacity: int) -> float:
        points = self._target_calibration.values
        # This dimensionless correction is shape-dependent. Clamp outside its
        # measured domain instead of extrapolating a latency slope into a ratio.
        calibrated_capacity = min(max(capacity, min(points)), max(points)) if points else capacity
        return max(1.0, self._target_calibration.estimate(calibrated_capacity) or 1.0)

    def _compound_seed_estimate(self, rows: int, depth: int) -> float | None:
        capacity = self.bucket(rows)
        points = {logical: price for (logical, segment), price in self._seed_compound.items()
                  if segment == depth and self.bucket(logical) == capacity}
        # Interpolate occupancy inside one actual sealed graph only. A price
        # from another physical bucket cannot supply an unmeasured workload.
        if not points or not min(points) <= rows <= max(points):
            return None
        return _EmaCurve(values=points).estimate(rows)

    def _work_score(
        self,
        expected: float,
        target_rows: int,
        proposal_rows: tuple[int, ...],
    ) -> float:
        """One denominator for ragged and aligned candidate shapes."""

        capacity = self.bucket(target_rows)
        owners = target_rows - sum(proposal_rows)
        if (self.require_complete_rounds and proposal_rows
                and proposal_rows[0] < owners and self._zero_score(owners) < 0):
            # Partial K0 waves terminate owners too. They require the same
            # complete-price witness as the explicit all-zero candidate.
            return -1.0
        # A carried round covers a uniform cohort only. Never apply its price
        # to ragged waves, target-only owners or a mixed prefill. Use actual
        # observed graph buckets; a small-C fused price cannot predict a new
        # routed workload at a neighboring, still-unobserved bucket.
        if (proposal_rows and len(set(proposal_rows)) == 1
                and target_rows == proposal_rows[0] * (len(proposal_rows) + 1)):
            curve = self.compound_cost.get(len(proposal_rows) + 1)
            complete = None if curve is None else curve.values.get(target_rows)
            if complete is None:
                complete = self._compound_seed_estimate(target_rows, len(proposal_rows) + 1)
                if complete is not None:
                    work = self._request_work.get(owners) if self.require_complete_rounds else None
                    if work is not None:
                        shared, accept, mask, publication, remainder, _first_mask = work
                        target = self._target_estimate(capacity, len(proposal_rows) + 1)
                        complete = (self._request_native_price(
                                        complete, target_rows, len(proposal_rows) + 1)
                                    + shared + accept * expected + publication + remainder
                                    + max(0.0, mask * target_rows - (target or 0.0)))
                        return expected / max(1e-6, complete)
                    # Review slowdown and complete-round excess are different
                    # predictors. Scaling the whole excess with a larger GPU
                    # graph can prematurely stop drafting. The remaining work
                    # is not attributed as pure host time. Only matched logical
                    # occupancy witnesses calibrate the remaining complete work.
                    excess = {} if curve is None else {
                        rows: max(0.0, milliseconds
                                  - self._request_native_price(
                                      self._seed_compound[key], rows, len(proposal_rows) + 1))
                        for rows, milliseconds in curve.values.items()
                        if (key := (rows, len(proposal_rows) + 1)) in self._seed_compound
                    }
                    complete = self._request_native_price(
                        complete, target_rows, len(proposal_rows) + 1)
                    if excess:
                        anchor = min(max(target_rows, min(excess)), max(excess))
                        complete += _EmaCurve(values=excess).estimate(anchor) or 0.0
            if complete is not None:
                return expected / max(1e-6, complete)
        target = self._target_estimate(capacity, len(proposal_rows) + 1)
        if target is None:
            return -1.0
        if not self.require_complete_rounds:
            proposal = sum(
                self.proposal_cost.estimate(self.bucket(rows)) or 0.0 for rows in proposal_rows)
            overhead = self.overhead_cost.estimate(capacity) or 0.0
            return expected / max(1e-6, target + proposal + overhead)
        work = self._request_work.get(owners) if self.require_complete_rounds else None
        proposal = 0.0
        for rows in proposal_rows:
            bucket = self.bucket(rows)
            price = self.proposal_cost.estimate(bucket) or 0.0
            if work is not None and bucket not in self._request_proposal_observed:
                # Staged q prepares its mask before the provider can launch.
                # Its event interval includes that exposed CPU wait; captured
                # carried rounds instead overlap p masks with the target.
                price += work[5] * rows
                stochastic = _EmaCurve(values=self._seed_constraint_sampling).estimate(bucket)
                greedy = _EmaCurve(values=self._seed_constraint_greedy).estimate(bucket)
                if stochastic is not None and greedy is not None:
                    price += max(0.0, stochastic - greedy)
            proposal += price
        overhead = self.overhead_cost.estimate(capacity) or 0.0
        if work is not None:
            depth = len(proposal_rows) + 1
            curve = self._request_stage_overhead.get(depth)
            actual = None if curve is None else curve.values.get(capacity)
            if actual is not None:
                overhead = actual
            else:
                shared, accept, mask, handoff, remainder, _first = work
                tail = _EmaCurve(values=self._seed_constraint_sampling).estimate(capacity)
                if tail is not None:
                    overhead = (shared + accept * expected + handoff + remainder + tail
                                + max(0.0, mask * owners - target))
        return expected / max(1e-6, target + proposal + overhead)

    def _zero_score(self, requests: int) -> float:
        capacity = self.bucket(requests)
        resident = self._resident_zero_cost.values.get(requests)
        if resident is not None:
            # This already includes target, sampling, communication and host
            # publication. Recalibration or adding overhead would double count.
            return requests / max(1e-6, resident)
        if self.require_complete_rounds:
            sampling = self._seed_constraint_sampling.get(requests)
            work = self._request_work.get(requests)
            target = self._target_estimate(capacity)
            if (sampling is None or work is None or target is None
                    or requests not in self._seed_target):
                # Missing request work or masked sampling is unknown, not free.
                return -1.0
            shared, accept, _mask, publication, remainder, first_mask = work
            # The sampling prior includes its own CPU staging, mask upload,
            # GPU sampling/communication and durable D2H. Do not scale that
            # inclusive wall span by the target model's GPU calibration.
            price = (target + sampling + shared + accept * requests
                     + publication + remainder + max(0.0, first_mask * requests - target))
            return requests / max(1e-6, price)
        target = self._target_estimate(capacity)
        if target is None:
            return -1.0
        seed = self._seed_target.get(capacity)
        calibration = self._target_calibration.values
        if seed and self.target_cost.estimate(capacity) == seed and calibration:
            # K0 cannot recover from an optimistic counterfactual. Cover the
            # observed local variation of live/cold corrections until this
            # target-only bucket has its own runtime measurement.
            anchors = sorted(calibration)
            nearest = min(range(len(anchors)), key=lambda i: abs(anchors[i] - capacity))
            neighbors = anchors[max(0, nearest - 1):nearest + 2]
            target = max(target, seed * max(calibration[n] for n in neighbors))
        steady = self._resident_zero_overhead.values
        points = steady or self._zero_overhead.values
        if points:
            rows = requests if steady else capacity
            overhead = points[min(points, key=lambda n: abs(n - rows))]
        else:
            overhead = min(self.overhead_cost.values.values(), default=0.0)
        return requests / max(1e-6, target + overhead)

    def has_compound_cost(self, requests: int, draft_tokens: int) -> bool:
        """Only an actual qualified geometry can authorize a complete transition."""
        if (type(requests) is not int or requests < 1
                or type(draft_tokens) is not int or draft_tokens < 1):
            raise ValueError("compound draft geometry must be positive integers")
        rows, depth = requests * (draft_tokens + 1), draft_tokens + 1
        if rows > self.graph_capacities[-1]:
            return False
        curve = self.compound_cost.get(depth)
        return (curve is not None and rows in curve.values
                or self._compound_seed_estimate(rows, depth) is not None)

    def permits_coherent_budget(self, policies, *, row_limit: int) -> bool:
        """Prove every candidate allocation is uniform before deferring a decision.

        A partially filled optional wave can change an individually ready owner.
        Such cohorts must retain the conservative coordinator checkpoints.
        """
        if (not policies or len({p.max_tokens for p in policies}) != 1
                or len({p.current_tokens for p in policies}) != 1
                or any(p.stopped for p in policies)):
            return False
        n, maximum = len(policies), policies[0].max_tokens
        if n * (maximum + 1) > row_limit:
            return False
        return (all((gap := self.bucket(n * (k + 1)) - n * (k + 1)) == 0 or gap >= n
                    for k in range(maximum))
                and all(self.has_compound_cost(n, k) for k in range(1, maximum + 1)))

    def plan(
        self,
        policies: tuple[DynamicDraftPolicy, ...],
        *,
        row_limit: int,
        fixed_decode_requests: int = 0,
        target_extra_rows: int = 0,
        available_decode_requests: int | None = None,
        coherent_rounds: bool = False,
    ) -> tuple[int, ...]:
        if type(coherent_rounds) is not bool:
            raise ValueError("coherent draft planning must be an explicit boolean")
        if not policies:
            return ()
        if fixed_decode_requests < 0 or target_extra_rows < 0:
            raise ValueError("dynamic draft fixed load cannot be negative")
        request_count = len(policies)
        base_decode_rows = request_count + fixed_decode_requests
        if not self.target_cost.values or not self.proposal_cost.values:
            # Never terminate a request from an arbitrary cold-start prior.
            return tuple(policy.current_tokens for policy in policies)
        if not any(policy.ready for policy in policies):
            return tuple(policy.current_tokens for policy in policies)
        transition_floor = tuple(
            max(0, policy.current_tokens - 1) if policy.ready else policy.current_tokens
            for policy in policies
        )
        # Price the future cohort across graph boundaries independently of
        # request cooldowns. Only the next one-level, ready transition consumes
        # this dispatch's rows; unready requests hold their actual K.
        # Mixed prefill recurrence is deliberately excluded from these decode
        # prices. Do not promote from apparently free rows in its large graph;
        # retain resource-safe demotion and resume exploration in pure decode.
        upper = tuple(
            policy.current_tokens if target_extra_rows else policy.max_tokens for policy in policies
        )
        max_budget = min(row_limit - base_decode_rows, sum(upper))
        if max_budget < sum(transition_floor):
            # The caller must choose a smaller request set; inventing a plan
            # that cannot be reached by one stable transition misprices load.
            return tuple(policy.current_tokens for policy in policies)
        survivals = tuple(policy.survival_probabilities() for policy in policies)
        best_score = -1.0
        best = tuple(policy.current_tokens for policy in policies)
        zero_score = self._zero_score(base_decode_rows) if target_extra_rows == 0 else -1.0
        # A physical bucket/depth price cannot identify arbitrary mixtures of
        # many draft depths. Use a common cohort depth and at most one extra
        # proposal wave. The extra wave fills non-anchor graph occupancy while
        # keeping the measured workload comparable to the candidate workload.
        # A depth-zero cohort with a partial extra wave also stops requests.
        # Excluding only the all-zero score is insufficient during unpriced
        # mixed prefill: filling a cheaper bucket must not permanently disable
        # drafting. Resource pressure may reduce positive K or the request set.
        for depth in range(1 if target_extra_rows else 0, max(upper) + 1):
            base = tuple(min(depth, bound) for bound in upper)
            base_budget = sum(base)
            if base_budget > max_budget:
                break
            optional = sorted(
                (request for request, bound in enumerate(upper) if bound > depth),
                key=lambda request: (-survivals[request][depth], request),
            )
            target_rows = base_decode_rows + base_budget + target_extra_rows
            extra_limit = min(
                len(optional),
                max_budget - base_budget,
                self.bucket(target_rows) - target_rows,
            )
            next_rows = base_decode_rows + sum(
                policy.current_tokens
                + (int(target > policy.current_tokens) - int(target < policy.current_tokens))
                * int(policy.ready)
                for policy, target in zip(policies, base, strict=True)
            )
            if next_rows > row_limit:
                continue
            for count, request in enumerate(optional[:extra_limit]):
                policy = policies[request]
                next_rows += int(
                    policy.ready
                    and base[request]
                    in (
                        policy.current_tokens - 1,
                        policy.current_tokens,
                    )
                )
                if next_rows > row_limit:
                    extra_limit = count
                    break
            # Compare the common cohort with its fullest reachable occupancy in
            # the same target bucket. Partly filled intermediate waves alias
            # different routed workloads onto one bucket/depth price and can
            # repeatedly reverse the cohort's K decision.
            extra_counts = {0, extra_limit}
            for extra in sorted(extra_counts):
                allocation = list(base)
                for request in optional[:extra]:
                    allocation[request] += 1
                if not any(allocation):
                    continue
                next_rows = base_decode_rows + sum(
                    policy.current_tokens
                    + (int(target > policy.current_tokens) - int(target < policy.current_tokens))
                    * int(policy.ready)
                    for policy, target in zip(policies, allocation, strict=True)
                )
                if next_rows > row_limit:
                    continue
                expected = base_decode_rows + sum(
                    sum(survival[:k]) for survival, k in zip(survivals, allocation, strict=True)
                )
                score = self._work_score(
                    expected,
                    base_decode_rows + base_budget + extra + target_extra_rows,
                    tuple(sum(k >= d for k in allocation) for d in range(1, max(allocation) + 1)),
                )
                if score > best_score:
                    best_score, best = score, tuple(allocation)
        if target_extra_rows == 0 and available_decode_requests is not None:
            if available_decode_requests < base_decode_rows:
                raise ValueError("available decode load is smaller than selected requests")
            zero_requests = min(row_limit, available_decode_requests)
            zero_score = self._zero_score(zero_requests)
        if zero_score >= 0.0 and zero_score > best_score * (1.0 + self.config.terminal_stop_margin):
            best = (0,) * request_count
        if (coherent_rounds and fixed_decode_requests == 0 and target_extra_rows == 0
                and (available_decode_requests is None
                     or available_decode_requests == request_count)
                and not any(best) and len({p.current_tokens for p in policies}) == 1):
            # Return the complete terminal goal, as with positive coherent
            # goals. The owner preserves every observation window until all
            # requests can consume their transition together.
            return best
        if (coherent_rounds and fixed_decode_requests == 0 and target_extra_rows == 0
                and (available_decode_requests is None
                     or available_decode_requests == request_count)
                and len(set(best)) == 1 and best[0] > 0
                and self.has_compound_cost(request_count, best[0])):
            # The caller waits without consuming individual observation windows,
            # then restores checkpoints and selects this complete depth together.
            # Otherwise an individually ready prefix can fragment a cohort and
            # price a captured graph that its actual mixed depths cannot use.
            return best
        return tuple(
            target if policy.ready else policy.current_tokens
            for policy, target in zip(policies, best, strict=True)
        )


def draft_decision_horizon(
    policies: tuple[DynamicDraftPolicy, ...], *, coherent_rounds: bool = False,
) -> int:
    """Lower bound on rounds before the next eligible observation boundary.

    An individual boundary uses the earliest request; a complete cohort waits
    for every owner, so its earliest possible boundary is the latest bound.
    Callers still prove complete ownership and matching captured depth before
    using the coherent bound. Acceptance can postpone, never hasten, this bound.
    """
    bounds = tuple(policy.rounds_until_decision for policy in policies)
    return (max if coherent_rounds else min)(bounds, default=0)


__all__ = [
    "draft_decision_horizon",
    "DraftAction",
    "DraftDecision",
    "DraftPolicyConfig",
    "DynamicDraftPlanner",
    "DynamicDraftPolicy",
]
