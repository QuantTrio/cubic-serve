"""Qualification for CUDA Graph replay and padding isolation.

The oracle executes only the logical rows.  Every adversarial performance replay
owns a fresh :class:`~cubic_serve.contracts.execution.LaunchPlan` and keeps its
topology-wide lease until durable output/state observation and fixture cleanup.
Replay adapters expose canonical logical outputs through the same production
state/transaction path used after admission; padding-only scratch is deliberately
excluded from numeric comparison, while its schema and every surrounding guard
are checked.

This module is independent of CUDA so the same qualification protocol can drive
an in-process fake, an eager kernel adapter, or a real captured-graph runner.
"""

from __future__ import annotations

from typing_extensions import TypeAliasType

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import combinations, pairwise
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from cubic_serve.contracts.execution import (
    ActiveLaunchRow,
    GraphBucket,
    LaunchPlan,
    LaunchRow,
    PaddingLaunchRow,
)
from cubic_serve.guardrail.compare import (
    ComparisonContext,
    ComparisonPolicy,
    ComparisonResult,
    DifferenceClass,
    VerifiedWitness,
    compare_tensors,
    comparison_failure,
    stable_digest,
    tensor_digest,
)
from cubic_serve.guardrail.oracle import OracleCallable, clone_tensor_map, run_oracle


@dataclass(frozen=True, slots=True)
class GraphPaddingPoison:
    """One adversarial initialization of all non-logical graph resources."""

    poison_id: str
    padding_inputs: Mapping[str, Tensor]
    scratch_state: Mapping[str, Tensor]
    guards: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        if not self.poison_id:
            raise ValueError("padding poison identity is required")
        clone_tensor_map(self.padding_inputs, "padding poison input")
        clone_tensor_map(self.scratch_state, "padding poison scratch")
        clone_tensor_map(self.guards, "padding poison guard")


@dataclass(slots=True)
class GraphPaddingBuffers:
    """Private mutable buffers owned by one topology-wide replay lease."""

    poison_id: str
    padding_inputs: dict[str, Tensor]
    scratch_state: dict[str, Tensor]
    guards: dict[str, Tensor]

    @classmethod
    def from_poison(cls, poison: GraphPaddingPoison) -> GraphPaddingBuffers:
        return cls(
            poison_id=poison.poison_id,
            padding_inputs=clone_tensor_map(poison.padding_inputs, "replay padding input"),
            scratch_state=clone_tensor_map(poison.scratch_state, "replay padding scratch"),
            guards=clone_tensor_map(poison.guards, "replay padding guard"),
        )


@runtime_checkable
class GraphQualificationRun(Protocol):
    """One fresh qualification fixture using the production state path."""

    @property
    def production_state(self) -> object: ...

    def committed_snapshot(self) -> Mapping[str, Tensor]: ...

    def close(self) -> None: ...

    def abort(self) -> None: ...


@runtime_checkable
class GraphQualificationStateFactory(Protocol):
    """Create an isolated production-state run from one canonical tensor seed."""

    @property
    def state_path_id(self) -> str: ...

    @property
    def state_path_digest(self) -> str: ...

    def create(self, canonical_seed: Mapping[str, Tensor]) -> GraphQualificationRun: ...


@runtime_checkable
class GraphQualificationAttempt(Protocol):
    """One fresh lease-bound candidate replay, closed only after observation."""

    @property
    def launch_plan(self) -> LaunchPlan: ...

    @property
    def state_factory(self) -> GraphQualificationStateFactory: ...

    def replay(
        self,
        inputs: Mapping[str, Tensor],
        state: object,
        padding: GraphPaddingBuffers,
    ) -> Mapping[str, Tensor]: ...

    def close(self) -> None: ...


GraphQualificationAttemptFactory = TypeAliasType("GraphQualificationAttemptFactory", Callable[[], GraphQualificationAttempt])


@dataclass(frozen=True, slots=True)
class GraphPaddingCase:
    """Logical inputs/state plus the poison corpus for one captured bucket."""

    case_id: str
    logical_inputs: Mapping[str, Tensor]
    initial_committed_state: Mapping[str, Tensor]
    poisons: tuple[GraphPaddingPoison, ...]

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("graph padding case identity is required")
        if type(self.poisons) is not tuple or any(
            not isinstance(poison, GraphPaddingPoison) for poison in self.poisons
        ):
            raise TypeError("graph padding poisons must be an immutable typed tuple")
        clone_tensor_map(self.logical_inputs, "graph padding logical input")
        clone_tensor_map(self.initial_committed_state, "graph padding committed state")


@dataclass(frozen=True, slots=True)
class GraphPaddingCheck:
    path: str
    result: ComparisonResult

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("graph padding check path is required")


@dataclass(frozen=True, slots=True)
class CrossMReplaySnapshot:
    """Physical replay tensors plus the row identity needed to compare two M values."""

    snapshot_id: str
    corpus_id: str
    reduction_plan_id: str
    rows: tuple[LaunchRow, ...]
    tensors: Mapping[str, Tensor]
    row_axes: Mapping[str, int]

    def __post_init__(self) -> None:
        if not all((self.snapshot_id, self.corpus_id, self.reduction_plan_id)):
            raise ValueError("cross-M snapshot identities are required")
        if type(self.rows) is not tuple or not self.rows or any(
            not isinstance(row, (ActiveLaunchRow, PaddingLaunchRow)) for row in self.rows
        ):
            raise TypeError("cross-M snapshot requires immutable typed physical rows")
        active = tuple(
            row.logical_index for row in self.rows if isinstance(row, ActiveLaunchRow)
        )
        if tuple(sorted(active)) != tuple(range(len(active))):
            raise ValueError("cross-M logical rows must form one complete identity domain")
        if not self.tensors or set(self.tensors) != set(self.row_axes):
            raise ValueError("cross-M tensors require one explicit physical row axis each")
        for name, value in self.tensors.items():
            axis = self.row_axes[name]
            if not isinstance(value, Tensor) or type(axis) is not int:
                raise TypeError("cross-M observations require tensors and exact integer axes")
            if value.ndim == 0 or not -value.ndim <= axis < value.ndim:
                raise ValueError("cross-M tensor row axis is outside its rank")
            if value.shape[axis % value.ndim] != len(self.rows):
                raise ValueError("cross-M tensor physical extent differs from its row map")

    @property
    def logical_m(self) -> int:
        return sum(isinstance(row, ActiveLaunchRow) for row in self.rows)


@dataclass(frozen=True, slots=True)
class CrossMArithmeticWitness:
    """An oracle-issued allowance for one canonical common-row comparison.

    The witness binds the actual tensors and arithmetic, not a caller-selected
    tolerance. State tensors require evidence for their producing computation;
    a local linear witness cannot stand in for propagated state evidence.
    """

    policy: ComparisonPolicy
    context: ComparisonContext
    witness: VerifiedWitness


@dataclass(frozen=True, slots=True)
class CrossMReplayResult:
    checks: tuple[GraphPaddingCheck, ...]
    reduction_plans: tuple[tuple[str, str], ...] = ()

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.result.passed for check in self.checks)


def compare_cross_m_replays(
    snapshots: tuple[CrossMReplaySnapshot, ...],
    *,
    arithmetic: Mapping[tuple[str, str, str], CrossMArithmeticWitness] | None = None,
) -> CrossMReplayResult:
    """Audit common rows, admitting only independently witnessed floating deltas.

    Arithmetic keys are (left snapshot ID, right snapshot ID, tensor path), in
    snapshot order. Witnesses are issued for canonical common rows, after row-map
    normalization. Changed reduction plans are recorded, not rejected outright.
    """

    checks: list[GraphPaddingCheck] = []
    allowances = dict(arithmetic or {})
    used: set[tuple[str, str, str]] = set()
    if type(snapshots) is not tuple or any(
        not isinstance(item, CrossMReplaySnapshot) for item in snapshots
    ):
        raise TypeError("cross-M qualification requires an immutable snapshot tuple")
    if len(snapshots) < 2 or len({item.logical_m for item in snapshots}) < 2:
        return CrossMReplayResult(
            (_failure("cross-m/contract", "at least two distinct logical M values are required"),)
        )
    if len({item.snapshot_id for item in snapshots}) != len(snapshots):
        checks.append(_failure("cross-m/contract", "snapshot identities must be unique"))

    for left, right in combinations(snapshots, 2):
        pair = f"{left.snapshot_id}~{right.snapshot_id}"
        if left.corpus_id != right.corpus_id:
            checks.append(_failure(f"cross-m:{pair}/corpus", "logical corpus identity changed"))
            continue
        names = sorted(set(left.tensors) | set(right.tensors))
        for name in names:
            if name not in left.tensors or name not in right.tensors:
                checks.append(
                    _failure(f"cross-m:{pair}:{name}", "observed tensor path changed across M")
                )
                continue
            common_m = min(left.logical_m, right.logical_m)
            expected = _canonical_logical_rows(left, name)[:common_m]
            actual = _canonical_logical_rows(right, name)[:common_m]
            key = (left.snapshot_id, right.snapshot_id, name)
            allowance = allowances.get(key)
            if allowance is not None:
                used.add(key)
                if not expected.is_floating_point() or not actual.is_floating_point():
                    checks.append(_failure(
                        f"cross-m:{pair}:{name}",
                        "discrete semantic tensors cannot use floating arithmetic allowances",
                    ))
                    continue
                result = compare_tensors(
                    expected, actual, allowance.policy,
                    context=allowance.context, witness=allowance.witness,
                )
            else:
                result = compare_tensors(
                    expected, actual, ComparisonPolicy(DifferenceClass.EXACT)
                )
            checks.append(
                GraphPaddingCheck(
                    path=f"cross-m:{pair}:{name}",
                    result=result,
                )
            )
    for key in sorted(allowances.keys() - used):
        checks.append(_failure(f"cross-m:unused:{key}", "arithmetic witness has no compared field"))
    return CrossMReplayResult(
        tuple(checks), tuple((s.snapshot_id, s.reduction_plan_id) for s in snapshots)
    )


def _canonical_logical_rows(snapshot: CrossMReplaySnapshot, name: str) -> Tensor:
    physical_by_logical = [0] * snapshot.logical_m
    for physical, row in enumerate(snapshot.rows):
        if isinstance(row, ActiveLaunchRow):
            physical_by_logical[row.logical_index] = physical
    value = snapshot.tensors[name]
    axis = snapshot.row_axes[name] % value.ndim
    indices = torch.tensor(physical_by_logical, dtype=torch.int64, device=value.device)
    return value.index_select(axis, indices).movedim(axis, 0)


@dataclass(frozen=True, slots=True)
class GraphReplayLaunchIdentity:
    """Ordered proof that one poison used one distinct topology-wide lease."""

    poison_id: str
    launch_plan_id: str
    lease_id: str
    lease_epoch: int
    execution_plan_id: str
    source_plan_id: str
    bucket_id: str
    row_map_digest: str

    @classmethod
    def from_plan(cls, poison_id: str, launch_plan: LaunchPlan) -> GraphReplayLaunchIdentity:
        lease = launch_plan.replay_lease
        return cls(
            poison_id,
            launch_plan.launch_plan_id,
            lease.lease_id,
            lease.epoch,
            launch_plan.execution_plan_id,
            launch_plan.source_plan_id,
            launch_plan.bucket.bucket_id,
            row_map_digest(launch_plan.rows),
        )


@dataclass(frozen=True, slots=True)
class GraphPaddingResult:
    """Fail-closed qualification result; it carries no execution fallback."""

    case_id: str
    bucket_id: str
    logical_m: int
    row_map_digest: str
    poison_ids: tuple[str, ...]
    replay_launches: tuple[GraphReplayLaunchIdentity, ...]
    state_path_id: str | None
    state_path_digest: str | None
    checks: tuple[GraphPaddingCheck, ...]

    def __post_init__(self) -> None:
        if not all((self.case_id, self.bucket_id)) or self.logical_m <= 0:
            raise ValueError("graph padding result identities are required")
        _require_digest(self.row_map_digest, "row map")
        if not self.checks:
            raise ValueError("graph padding result requires at least one check")
        paths = tuple(check.path for check in self.checks)
        if len(set(paths)) != len(paths):
            raise ValueError("graph padding result check paths must be unique")
        launches = self.replay_launches
        if any(
            launch.bucket_id != self.bucket_id or launch.row_map_digest != self.row_map_digest
            for launch in launches
        ):
            raise ValueError("qualification replays must share one bucket and row map")
        launch_poison_ids = tuple(launch.poison_id for launch in launches)
        positions = tuple(
            self.poison_ids.index(poison_id)
            for poison_id in launch_poison_ids
            if poison_id in self.poison_ids
        )
        if len(positions) != len(launches) or positions != tuple(sorted(set(positions))):
            raise ValueError("qualification replay identities must follow poison order")
        if len({launch.launch_plan_id for launch in launches}) != len(launches) or len(
            {launch.lease_id for launch in launches}
        ) != len(launches):
            raise ValueError("each qualification replay requires a distinct launch and lease")
        epochs = tuple(launch.lease_epoch for launch in launches)
        if any(right <= left for left, right in pairwise(epochs)):
            raise ValueError("qualification replay lease epochs must increase")
        if self.passed and (
            len(launches) != len(self.poison_ids)
            or not self.state_path_id
            or not self.state_path_digest
        ):
            raise ValueError("passing qualification requires complete replay provenance")

    @property
    def passed(self) -> bool:
        return all(check.result.passed for check in self.checks)


@dataclass(frozen=True, slots=True)
class _ReplaySnapshot:
    poison_id: str
    logical_outputs: dict[str, Tensor]
    committed_state: dict[str, Tensor]


class GraphPaddingHarness:
    """Prove that padding cannot affect logical results or committed state."""

    def __init__(self, exact_m_oracle: OracleCallable) -> None:
        self._exact_m_oracle = exact_m_oracle

    def qualify(
        self,
        *,
        case: GraphPaddingCase,
        bucket: GraphBucket,
        rows: tuple[LaunchRow, ...],
        attempt_factory: GraphQualificationAttemptFactory,
    ) -> GraphPaddingResult:
        checks: list[GraphPaddingCheck] = []
        logical_m = sum(isinstance(row, ActiveLaunchRow) for row in rows)
        target_digest = row_map_digest(rows)
        try:
            config_error = _configuration_error(case, bucket, rows)
        except Exception as error:
            config_error = (
                f"padding qualification contract inspection failed: {type(error).__name__}: {error}"
            )
        if config_error is not None:
            checks.append(_failure("contract", config_error))
            return _result(case, bucket, logical_m, target_digest, (), None, checks)

        try:
            oracle = run_oracle(
                self._exact_m_oracle,
                case.logical_inputs,
                case.initial_committed_state,
            )
        except Exception as error:
            checks.append(
                _failure(
                    "oracle/runtime",
                    f"exact-M oracle execution failed: {type(error).__name__}: {error}",
                )
            )
            return _result(case, bucket, logical_m, target_digest, (), None, checks)

        snapshots: list[_ReplaySnapshot] = []
        launches: list[GraphReplayLaunchIdentity] = []
        state_path: tuple[str, str] | None = None
        for poison in case.poisons:
            # Admission failures retain their typed overload/poison semantics;
            # no candidate replay exists for the harness to compare in that case.
            attempt = attempt_factory()
            state_run: GraphQualificationRun | None = None
            snapshot: _ReplaySnapshot | None = None
            completed = False
            try:
                buffers = GraphPaddingBuffers.from_poison(poison)
                expected_guards = clone_tensor_map(buffers.guards, "expected replay guard")
                scratch_schema = _tensor_map_schema(buffers.scratch_state)
                inputs = clone_tensor_map(case.logical_inputs, "performance logical input")
                if not isinstance(attempt, GraphQualificationAttempt):
                    raise TypeError("qualification attempt factory returned an invalid attempt")
                launch_plan = attempt.launch_plan
                if (
                    launch_plan.bucket != bucket
                    or launch_plan.rows != rows
                    or launch_plan.logical_m != logical_m
                ):
                    raise ValueError("qualification attempt changed its bucket or row map")
                identity = GraphReplayLaunchIdentity.from_plan(poison.poison_id, launch_plan)
                if launches and identity.lease_epoch <= launches[-1].lease_epoch:
                    raise ValueError("qualification attempt reused a stale lease epoch")
                if any(
                    identity.launch_plan_id == prior.launch_plan_id
                    or identity.lease_id == prior.lease_id
                    for prior in launches
                ):
                    raise ValueError("qualification attempt reused a launch or lease identity")
                launches.append(identity)
                factory = attempt.state_factory
                if not isinstance(factory, GraphQualificationStateFactory):
                    raise TypeError("qualification attempt requires a typed state factory")
                observed_path = (factory.state_path_id, factory.state_path_digest)
                _require_digest(observed_path[1], "state path")
                if state_path is not None and state_path != observed_path:
                    raise ValueError("qualification attempts changed the production state path")
                state_path = observed_path
                state_run = factory.create(
                    clone_tensor_map(
                        case.initial_committed_state, "qualification canonical state seed"
                    )
                )
                if not isinstance(state_run, GraphQualificationRun):
                    raise TypeError("qualification state factory returned an invalid run")
                outputs = attempt.replay(inputs, state_run.production_state, buffers)
                snapshot = _ReplaySnapshot(
                    poison_id=poison.poison_id,
                    logical_outputs=clone_tensor_map(outputs, "performance logical output"),
                    committed_state=clone_tensor_map(
                        state_run.committed_snapshot(), "performance committed state"
                    ),
                )
                observed_guards = clone_tensor_map(buffers.guards, "observed replay guard")
                observed_scratch_schema = _tensor_map_schema(buffers.scratch_state)
                completed = True
            except Exception as error:
                snapshot = None
                checks.append(
                    _failure(
                        f"poison:{poison.poison_id}/runtime",
                        f"performance graph replay failed: {type(error).__name__}: {error}",
                    )
                )
            finally:
                if state_run is not None:
                    try:
                        if completed:
                            state_run.close()
                        else:
                            state_run.abort()
                    except Exception as error:
                        snapshot = None
                        checks.append(
                            _failure(
                                f"poison:{poison.poison_id}/state-cleanup",
                                "qualification state cleanup failed: "
                                f"{type(error).__name__}: {error}",
                            )
                        )
                if isinstance(attempt, GraphQualificationAttempt):
                    try:
                        attempt.close()
                    except Exception as error:
                        snapshot = None
                        checks.append(
                            _failure(
                                f"poison:{poison.poison_id}/lease-cleanup",
                                "qualification replay lease cleanup failed: "
                                f"{type(error).__name__}: {error}",
                            )
                        )

            if snapshot is None:
                continue
            snapshots.append(snapshot)
            _compare_maps(
                checks,
                prefix=f"poison:{poison.poison_id}/oracle/output",
                expected=oracle.outputs,
                actual=snapshot.logical_outputs,
                require_non_empty=True,
            )
            _compare_maps(
                checks,
                prefix=f"poison:{poison.poison_id}/oracle/committed",
                expected=oracle.mutated_state,
                actual=snapshot.committed_state,
                require_non_empty=True,
            )
            _compare_maps(
                checks,
                prefix=f"poison:{poison.poison_id}/guard",
                expected=expected_guards,
                actual=observed_guards,
                require_non_empty=True,
            )
            checks.append(
                GraphPaddingCheck(
                    path=f"poison:{poison.poison_id}/scratch-schema",
                    result=_schema_result(scratch_schema, observed_scratch_schema),
                )
            )

        if len(snapshots) < 2:
            checks.append(
                _failure(
                    "poison-invariance/runtime",
                    "fewer than two adversarial graph replays produced observable snapshots",
                )
            )
        else:
            baseline = snapshots[0]
            for candidate in snapshots[1:]:
                pair = f"{baseline.poison_id}~{candidate.poison_id}"
                _compare_maps(
                    checks,
                    prefix=f"poison-invariance:{pair}/output",
                    expected=baseline.logical_outputs,
                    actual=candidate.logical_outputs,
                    require_non_empty=True,
                )
                _compare_maps(
                    checks,
                    prefix=f"poison-invariance:{pair}/committed",
                    expected=baseline.committed_state,
                    actual=candidate.committed_state,
                    require_non_empty=True,
                )

        return _result(
            case,
            bucket,
            logical_m,
            target_digest,
            tuple(launches),
            state_path,
            checks,
        )


def _configuration_error(
    case: GraphPaddingCase,
    bucket: GraphBucket,
    rows: tuple[LaunchRow, ...],
) -> str | None:
    if not isinstance(bucket, GraphBucket) or type(rows) is not tuple:
        return "qualification target requires a typed graph bucket and immutable rows"
    if len(rows) != bucket.captured_m or any(
        not isinstance(row, (ActiveLaunchRow, PaddingLaunchRow)) for row in rows
    ):
        return "qualification rows must exactly fill captured M"
    active = tuple(row.logical_index for row in rows if isinstance(row, ActiveLaunchRow))
    logical_m = len(active)
    scratch = tuple(row.scratch_index for row in rows if isinstance(row, PaddingLaunchRow))
    if (
        logical_m not in bucket.logical_m_values
        or tuple(sorted(active)) != tuple(range(logical_m))
        or tuple(sorted(scratch)) != tuple(range(bucket.captured_m - logical_m))
    ):
        return "qualification rows do not form one valid logical-to-physical map"
    if len(case.poisons) < 2:
        return "performance graph qualification requires at least two padding poisons"

    poison_ids = tuple(poison.poison_id for poison in case.poisons)
    if len(set(poison_ids)) != len(poison_ids):
        return "padding poison identities must be unique"
    padding_rows = bucket.captured_m - logical_m
    if any(not poison.guards for poison in case.poisons):
        return "each graph replay poison must cover guards"
    if padding_rows and any(
        not poison.padding_inputs or not poison.scratch_state for poison in case.poisons
    ):
        return "each padded replay poison must cover padding inputs and scratch state"
    if not padding_rows and any(
        poison.padding_inputs or poison.scratch_state for poison in case.poisons
    ):
        return "a full-bucket replay cannot carry nonexistent padding resources"

    first = case.poisons[0]
    expected_schemas = (
        _tensor_map_schema(first.padding_inputs),
        _tensor_map_schema(first.scratch_state),
        _tensor_map_schema(first.guards),
    )
    if any(
        (
            _tensor_map_schema(poison.padding_inputs),
            _tensor_map_schema(poison.scratch_state),
            _tensor_map_schema(poison.guards),
        )
        != expected_schemas
        for poison in case.poisons[1:]
    ):
        return "all padding poisons must cover the same tensor schemas"

    if padding_rows:
        try:
            padding_digests = {_tensor_map_digest(poison.padding_inputs) for poison in case.poisons}
            scratch_digests = {_tensor_map_digest(poison.scratch_state) for poison in case.poisons}
        except Exception as error:
            return f"padding poison cannot be digested: {type(error).__name__}: {error}"
        if len(padding_digests) < 2 or len(scratch_digests) < 2:
            return "padding and scratch values must both vary across adversarial poisons"
    return None


def _compare_maps(
    checks: list[GraphPaddingCheck],
    *,
    prefix: str,
    expected: Mapping[str, Tensor],
    actual: Mapping[str, Tensor],
    require_non_empty: bool,
) -> None:
    names = sorted(set(expected) | set(actual))
    if require_non_empty and not names:
        checks.append(_failure(prefix, "required tensor map is empty"))
        return
    policy = ComparisonPolicy(DifferenceClass.EXACT)
    for name in names:
        expected_tensor = expected.get(name)
        actual_tensor = actual.get(name)
        if expected_tensor is None or actual_tensor is None:
            present = actual_tensor if expected_tensor is None else expected_tensor
            assert present is not None
            side = "actual" if expected_tensor is None else "expected"
            result = comparison_failure(
                DifferenceClass.EXACT,
                present.numel(),
                f"tensor exists only in {side} map",
            )
        else:
            result = compare_tensors(expected_tensor, actual_tensor, policy)
        checks.append(GraphPaddingCheck(path=f"{prefix}:{name}", result=result))


def _tensor_map_schema(values: Mapping[str, Tensor]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            name,
            tuple(tensor.shape),
            tuple(tensor.stride()),
            str(tensor.dtype),
            str(tensor.layout),
            tensor.device.type,
            tensor.device.index,
        )
        for name, tensor in sorted(values.items())
    )


def _tensor_map_digest(values: Mapping[str, Tensor]) -> str:
    return stable_digest({name: tensor_digest(tensor) for name, tensor in sorted(values.items())})


def _schema_result(
    expected: tuple[tuple[object, ...], ...],
    actual: tuple[tuple[object, ...], ...],
) -> ComparisonResult:
    if expected != actual:
        return comparison_failure(
            DifferenceClass.EXACT,
            0,
            "padding scratch tensor schema changed during replay",
        )
    return ComparisonResult(
        passed=True,
        difference_class=DifferenceClass.EXACT,
        element_count=0,
        first_mismatch_index=None,
        max_absolute_error=0.0,
        max_ulp_error=0,
        max_allowed_error=0.0,
        reason="padding scratch schema preserved; scratch values are intentionally unconstrained",
    )


def _failure(path: str, reason: str) -> GraphPaddingCheck:
    return GraphPaddingCheck(
        path=path,
        result=comparison_failure(DifferenceClass.EXACT, 0, reason),
    )


def _result(
    case: GraphPaddingCase,
    bucket: GraphBucket,
    logical_m: int,
    target_digest: str,
    launches: tuple[GraphReplayLaunchIdentity, ...],
    state_path: tuple[str, str] | None,
    checks: list[GraphPaddingCheck],
) -> GraphPaddingResult:
    return GraphPaddingResult(
        case_id=case.case_id,
        bucket_id=bucket.bucket_id,
        logical_m=logical_m,
        row_map_digest=target_digest,
        poison_ids=tuple(poison.poison_id for poison in case.poisons),
        replay_launches=launches,
        state_path_id=None if state_path is None else state_path[0],
        state_path_digest=None if state_path is None else state_path[1],
        checks=tuple(checks),
    )


def row_map_digest(rows: tuple[LaunchRow, ...]) -> str:
    return stable_digest(
        [
            ("active", row.logical_index)
            if isinstance(row, ActiveLaunchRow)
            else ("padding", row.scratch_index)
            for row in rows
        ]
    )


def _require_digest(value: str, label: str) -> None:
    algorithm, separator, digest = value.partition(":")
    if algorithm != "b2s256" or separator != ":" or len(digest) != 64:
        raise ValueError(f"qualification {label} fingerprint must be b2s256:<64 hex chars>")
    try:
        int(digest, 16)
    except ValueError as error:
        raise ValueError(
            f"qualification {label} fingerprint contains non-hex characters"
        ) from error
