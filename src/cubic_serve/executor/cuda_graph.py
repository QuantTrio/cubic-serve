"""Fail-closed coordination for qualified multi-M CUDA Graph programs.

The coordinator deliberately knows nothing about a model's tensors, state
schema, kernels, or communication schedule. A :class:`GraphProgram` owns that
ABI and the concrete captured resources behind each leased slot.
"""
# Sources: llama.cpp c841aeeb, vLLM 73fac1e1, SGLang ba433bb4.

from __future__ import annotations

from typing_extensions import TypeAliasType

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from threading import Condition
from time import monotonic
from typing import Protocol, runtime_checkable
from uuid import uuid4

from torch import Tensor

from cubic_serve.contracts import execution as ec
from cubic_serve.executor.replay_lease import ReplayLeasePool, ReplayLeaseUnavailable
from cubic_serve.guardrail import graph_padding as gp
from cubic_serve.guardrail.compare import stable_digest, tensor_digest
from cubic_serve.guardrail.oracle import OracleCallable


class CudaGraphRejected(RuntimeError):
    """The request does not close over one qualified performance graph."""


class CudaGraphOverload(RuntimeError):
    """A complete topology-wide lease was unavailable before its deadline."""


class CudaGraphPoisoned(RuntimeError):
    """The captured domain is permanently unusable after a runtime failure."""


TensorMap = TypeAliasType("TensorMap", Mapping[str, Tensor])
Rows = TypeAliasType("Rows", tuple[ec.LaunchRow, ...])
ResourceSets = TypeAliasType("ResourceSets", tuple[tuple[ec.RankReplayResources, ...], ...])
_WITNESS_ISSUER = object()


@dataclass(frozen=True, slots=True)
class GraphRuntimeBinding:
    """Close one bucket over immutable execution, topology, and partition plans."""

    execution_plan: ec.ExecutionPlan
    topology: ec.ParallelTopology
    partition: ec.PartitionPlan
    bucket: ec.GraphBucket
    participating_ranks: tuple[int, ...] = field(init=False)
    domain_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if self.execution_plan.mode is not ec.ExecutionMode.PERFORMANCE:
            raise CudaGraphRejected("CUDA Graph runtime requires a performance plan")
        if self.bucket not in self.execution_plan.graph_buckets:
            raise CudaGraphRejected("graph bucket is not owned by the execution plan")
        if self.topology.topology_id != self.bucket.topology_id:
            raise CudaGraphRejected("execution topology does not match the graph bucket")
        if self.partition.topology_id != self.topology.topology_id:
            raise CudaGraphRejected("partition does not belong to the execution topology")
        if self.partition.partition_id != self.bucket.partition_id:
            raise CudaGraphRejected("partition does not match the graph bucket")

        ranks = tuple(range(self.topology.world_size))
        _validate_partition_closure(self.topology, self.partition, ranks)
        object.__setattr__(self, "participating_ranks", ranks)
        object.__setattr__(
            self,
            "domain_fingerprint",
            stable_digest(
                {
                    "execution": repr(self.execution_plan),
                    "topology": repr(self.topology),
                    "partition": repr(self.partition),
                    "bucket": repr(self.bucket),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class AcquireDeadline:
    """One finite monotonic deadline shared by complete lease acquisition."""

    expires_at: float

    def __post_init__(self) -> None:
        if not isfinite(self.expires_at):
            raise CudaGraphRejected("replay acquire deadline must be finite")

    @classmethod
    def after(cls, timeout_s: float) -> AcquireDeadline:
        if not isfinite(timeout_s) or timeout_s < 0:
            raise CudaGraphRejected("replay acquire timeout must be finite and non-negative")
        expires_at = monotonic() + timeout_s
        if not isfinite(expires_at):
            raise CudaGraphRejected("replay acquire deadline must be finite")
        return cls(expires_at)

    def remaining(self) -> float:
        return max(0.0, self.expires_at - monotonic())


@dataclass(frozen=True, slots=True)
class ReplayIntent:
    """Logical work accepted by performance replay; launch identity is derived."""

    batch_plan: ec.BatchPlan
    rows: Rows
    deadline: AcquireDeadline

    def __post_init__(self) -> None:
        if not isinstance(self.batch_plan, ec.BatchPlan):
            raise TypeError("replay intent requires a typed BatchPlan")
        if type(self.rows) is not tuple:
            raise TypeError("replay rows must be an immutable tuple")
        if any(not isinstance(row, (ec.ActiveLaunchRow, ec.PaddingLaunchRow)) for row in self.rows):
            raise CudaGraphRejected("replay rows must contain typed launch rows")
        if not isinstance(self.deadline, AcquireDeadline):
            raise TypeError("replay intent requires a typed finite deadline")


@runtime_checkable
class GraphProgramSlot(Protocol):
    """One program-owned, captured all-rank static resource set."""

    @property
    def resources(self) -> tuple[ec.RankReplayResources, ...]: ...


@runtime_checkable
class GraphProgramSession(Protocol):
    """One lease-bound replay transaction.

    ``consume`` is a graph-completion barrier and returns durable logical
    outputs. ``commit`` publishes durable state and releases transaction
    ownership. ``abort`` is the completion barrier for every unsuccessful path
    and must discard pending state before it returns.
    """

    def replay(self) -> None: ...

    def consume(self) -> TensorMap: ...

    def commit(self) -> None: ...

    def abort(self) -> None: ...


@runtime_checkable
class GraphProgram(Protocol):
    """The sole model/kernel-specific boundary of graph execution.

    The program validates its complete ABI and supported topology, captures all
    concrete tensors/graphs/streams/workspaces for a slot, and binds request
    state transactionally during ``fill``. A failing ``fill`` must release any
    partial transaction ownership before raising.
    """

    @property
    def program_id(self) -> str: ...

    @property
    def replay_abi_id(self) -> str: ...

    @property
    def state_path_id(self) -> str: ...

    def validate_binding(self, binding: GraphRuntimeBinding) -> None: ...

    def capture(
        self,
        binding: GraphRuntimeBinding,
        resources: tuple[ec.RankReplayResources, ...],
    ) -> GraphProgramSlot: ...

    def qualification_state_factory(
        self,
        slot: GraphProgramSlot,
        plan: ec.LaunchPlan,
        intent: ReplayIntent,
    ) -> gp.GraphQualificationStateFactory: ...

    def fill(
        self,
        slot: GraphProgramSlot,
        plan: ec.LaunchPlan,
        intent: ReplayIntent,
        logical_inputs: TensorMap,
        state: object,
        padding: gp.GraphPaddingBuffers | None,
    ) -> GraphProgramSession: ...


@dataclass(frozen=True, slots=True, init=False)
class QualificationWitness:
    """Runtime-issued proof for one logical M of one concrete capture."""

    domain_fingerprint: str
    capture_fingerprint: str
    capture_epoch: int
    bucket: ec.GraphBucket
    logical_m: int
    row_map_digest: str
    case_digest: str
    state_path_id: str
    state_path_digest: str
    poison_ids: tuple[str, ...]
    result: gp.GraphPaddingResult
    _runtime_token: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        domain_fingerprint: str,
        capture_fingerprint: str,
        capture_epoch: int,
        bucket: ec.GraphBucket,
        logical_m: int,
        row_map_digest: str,
        case_digest: str,
        state_path_id: str,
        state_path_digest: str,
        poison_ids: tuple[str, ...],
        result: gp.GraphPaddingResult,
        _runtime_token: object,
        _issuer: object | None = None,
    ) -> None:
        if _issuer is not _WITNESS_ISSUER:
            raise TypeError("qualification witnesses may only be issued by the graph runtime")
        if capture_epoch <= 0 or logical_m not in bucket.logical_m_values:
            raise ValueError("qualification witness has an invalid capture epoch or logical M")
        if len(poison_ids) < 2 or len(set(poison_ids)) != len(poison_ids):
            raise ValueError("qualification witness requires unique replay poison identities")
        if type(state_path_id) is not str or not state_path_id:
            raise ValueError("qualification witness requires a state path identity")
        if not result.passed or result.bucket_id != bucket.bucket_id:
            raise ValueError("qualification witness requires passing exact bucket evidence")
        for label, fingerprint in (
            ("domain", domain_fingerprint),
            ("capture", capture_fingerprint),
            ("row map", row_map_digest),
            ("case", case_digest),
            ("state path", state_path_digest),
        ):
            _require_digest(fingerprint, label)
        values = {
            "domain_fingerprint": domain_fingerprint,
            "capture_fingerprint": capture_fingerprint,
            "capture_epoch": capture_epoch,
            "bucket": bucket,
            "logical_m": logical_m,
            "row_map_digest": row_map_digest,
            "case_digest": case_digest,
            "state_path_id": state_path_id,
            "state_path_digest": state_path_digest,
            "poison_ids": poison_ids,
            "result": result,
            "_runtime_token": _runtime_token,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def passed(self) -> bool:
        return self.result.passed


@dataclass(slots=True)
class _ReplaySlot:
    resources: tuple[ec.RankReplayResources, ...]
    program_slot: GraphProgramSlot
    owner_lease_id: str | None = None
    owner_epoch: int = 0

    def bind(self, lease: ec.ReplayLease) -> None:
        if self.resources != lease.rank_resources or self.owner_lease_id is not None:
            raise RuntimeError("replay slot ownership does not match the acquired lease")
        self.owner_lease_id, self.owner_epoch = lease.lease_id, lease.epoch

    def unbind(self, lease: ec.ReplayLease) -> None:
        if (self.owner_lease_id, self.owner_epoch) != (lease.lease_id, lease.epoch):
            raise RuntimeError("replay slot cannot release a foreign or stale lease")
        self.owner_lease_id, self.owner_epoch = None, 0


@dataclass(slots=True)
class _QualificationAttempt:
    runtime: CudaGraphRuntime
    lease: ec.ReplayLease
    plan: ec.LaunchPlan
    intent: ReplayIntent
    factory: gp.GraphQualificationStateFactory
    replayed: bool = False
    closed: bool = False

    @property
    def launch_plan(self) -> ec.LaunchPlan:
        return self.plan

    @property
    def state_factory(self) -> gp.GraphQualificationStateFactory:
        return self.factory

    def replay(
        self,
        inputs: TensorMap,
        state: object,
        padding: gp.GraphPaddingBuffers,
    ) -> TensorMap:
        if self.replayed or self.closed:
            raise RuntimeError("qualification attempt permits exactly one replay")
        self.replayed = True
        return self.runtime._run(self.plan, self.intent, inputs, state, padding)

    def close(self) -> None:
        if self.closed:
            raise RuntimeError("qualification attempt is already closed")
        try:
            self.runtime._finish(self.lease)
        finally:
            self.closed = True


class CudaGraphRuntime:
    """Coordinate one program's captured, qualified multi-M graph bucket."""

    def __init__(
        self,
        *,
        binding: GraphRuntimeBinding,
        program: GraphProgram,
        resource_sets: ResourceSets,
    ) -> None:
        if not isinstance(program, GraphProgram):
            raise CudaGraphRejected("CUDA Graph runtime requires a typed GraphProgram")
        if not program.program_id or program.replay_abi_id != binding.bucket.replay_abi_id:
            raise CudaGraphRejected("graph program identity or replay ABI does not match bucket")
        if type(program.state_path_id) is not str or not program.state_path_id:
            raise CudaGraphRejected("graph program requires a stable state path identity")
        if not resource_sets:
            raise CudaGraphRejected("CUDA Graph runtime requires a complete replay slot")
        expected = binding.participating_ranks
        if any(tuple(resource.rank for resource in slot) != expected for slot in resource_sets):
            raise CudaGraphRejected("every replay slot must exactly cover every topology rank")
        try:
            program.validate_binding(binding)
            pool = ReplayLeasePool(resource_sets)
        except CudaGraphRejected:
            raise
        except (TypeError, ValueError) as error:
            raise CudaGraphRejected(f"invalid graph program binding: {error}") from error

        self.binding, self.bucket, self.program = binding, binding.bucket, program
        self._state_path_id = program.state_path_id
        self._resource_sets, self._pool = resource_sets, pool
        self._slots: dict[tuple[ec.RankReplayResources, ...], _ReplaySlot] = {}
        self._condition = Condition()
        self._capture_started = self._captured = self._sealed = False
        self._capture_epoch = 0
        self._capture_fingerprint: str | None = None
        self._sealed_row_maps: dict[int, str] = {}
        self._capture_nonce, self._witness_token = uuid4().hex, object()
        self._poison_reason: str | None = None
        self._launch_serial = self._qualifications_in_flight = 0

    @property
    def in_flight_leases(self) -> int:
        return self._pool.in_flight

    def capture(self) -> None:
        with self._condition:
            if self._capture_started:
                raise CudaGraphRejected("CUDA Graph capture is single-use")
            self._capture_started = True
        try:
            slots = tuple(
                _ReplaySlot(resources, self.program.capture(self.binding, resources))
                for resources in self._resource_sets
            )
            if any(
                not isinstance(slot.program_slot, GraphProgramSlot)
                or slot.program_slot.resources != slot.resources
                for slot in slots
            ):
                raise RuntimeError("graph program captured a foreign resource slot")
        except Exception as error:
            self._poison("capture", error)
            raise CudaGraphPoisoned(self._poison_reason) from error
        with self._condition:
            self._slots = {slot.resources: slot for slot in slots}
            self._capture_epoch += 1
            self._capture_fingerprint = stable_digest(
                {
                    "domain": self.binding.domain_fingerprint,
                    "program": self.program.program_id,
                    "state_path": self._state_path_id,
                    "epoch": self._capture_epoch,
                    "nonce": self._capture_nonce,
                    "resources": repr(self._resource_sets),
                }
            )
            self._captured = True

    def qualify(
        self,
        *,
        case: gp.GraphPaddingCase,
        intent: ReplayIntent,
        exact_m_oracle: OracleCallable,
    ) -> QualificationWitness:
        self._admit_qualification()
        try:
            self._validate_intent(intent)
            result = gp.GraphPaddingHarness(exact_m_oracle).qualify(
                case=case,
                bucket=self.bucket,
                rows=intent.rows,
                attempt_factory=lambda: self._qualification_attempt(intent),
            )
            return self._issue_witness(result, intent, case)
        finally:
            self._finish_qualification()

    def seal(self, witnesses: tuple[QualificationWitness, ...]) -> None:
        self._require_ready(production=False)
        if type(witnesses) is not tuple or any(
            not isinstance(witness, QualificationWitness) for witness in witnesses
        ):
            raise CudaGraphRejected("seal requires immutable runtime-issued witnesses")
        logical_ms = tuple(sorted(witness.logical_m for witness in witnesses))
        if logical_ms != self.bucket.logical_m_values:
            raise CudaGraphRejected(
                "qualification witnesses do not cover the exact logical M domain"
            )
        capture = self._capture_fingerprint
        assert capture is not None
        if self.program.state_path_id != self._state_path_id:
            raise CudaGraphRejected("graph program state path identity changed")
        state_path_digests: set[str] = set()
        for witness in witnesses:
            if (
                witness._runtime_token is not self._witness_token
                or witness.capture_fingerprint != capture
                or witness.capture_epoch != self._capture_epoch
            ):
                raise CudaGraphRejected("qualification witness belongs to a different capture")
            if (
                witness.domain_fingerprint != self.binding.domain_fingerprint
                or witness.bucket != self.bucket
            ):
                raise CudaGraphRejected("qualification witness belongs to a different graph domain")
            if not witness.passed:
                raise CudaGraphRejected("qualification witness did not pass exact comparison")
            if witness.state_path_id != self._state_path_id:
                raise CudaGraphRejected("qualification witness used a different state path")
            state_path_digests.add(witness.state_path_digest)
        if len(state_path_digests) != 1:
            raise CudaGraphRejected("qualification witnesses used different state factories")
        with self._condition:
            if self._sealed:
                raise CudaGraphRejected("CUDA Graph bucket is already sealed")
            if self._qualifications_in_flight:
                raise CudaGraphRejected("cannot seal while qualification replay is in flight")
            self._sealed_row_maps = {
                witness.logical_m: witness.row_map_digest for witness in witnesses
            }
            self._sealed = True

    def replay(
        self,
        *,
        intent: ReplayIntent,
        logical_inputs: TensorMap,
        state: object,
    ) -> TensorMap:
        self._require_ready(production=True)
        lease, plan = self._begin(intent, production=True)
        try:
            return self._run(plan, intent, logical_inputs, state, None)
        finally:
            self._finish(lease)

    def _begin(
        self, intent: ReplayIntent, production: bool
    ) -> tuple[ec.ReplayLease, ec.LaunchPlan]:
        self._require_ready(production=production)
        self._validate_intent(intent)
        with self._condition:
            self._launch_serial += 1
            launch_id = f"{self.bucket.bucket_id}:launch:{self._launch_serial}"
        lease = self._acquire(launch_id, intent.deadline)
        try:
            plan = ec.LaunchPlan(
                launch_plan_id=launch_id,
                execution_plan_id=self.binding.execution_plan.execution_plan_id,
                source_plan_id=intent.batch_plan.plan_id,
                logical_m=intent.batch_plan.shape.logical_m,
                bucket=self.bucket,
                rows=intent.rows,
                replay_lease=lease,
            )
            self.binding.execution_plan.validate_launch(intent.batch_plan, plan)
        except (TypeError, ValueError) as error:
            self._release(lease)
            raise CudaGraphRejected(f"invalid graph launch plan: {error}") from error
        return lease, plan

    def _validate_intent(self, intent: ReplayIntent) -> None:
        if not isinstance(intent, ReplayIntent):
            raise CudaGraphRejected("performance replay requires a typed ReplayIntent")
        batch = intent.batch_plan
        try:
            selected = self.binding.execution_plan.bucket_for(
                scope_id=self.bucket.scope_id,
                replay_abi_id=self.bucket.replay_abi_id,
                phase=batch.phase,
                logical_m=batch.shape.logical_m,
                capacity_items=batch.shape.capacity_items,
                topology_id=batch.topology_id,
                partition_id=batch.partition_id,
            )
        except ValueError as error:
            raise CudaGraphRejected(f"batch has no sealed performance graph: {error}") from error
        if selected != self.bucket:
            raise CudaGraphRejected("batch resolves to a different captured graph runtime")
        with self._condition:
            qualified_rows = self._sealed_row_maps.get(batch.shape.logical_m)
        if qualified_rows is not None and _row_map_digest(intent.rows) != qualified_rows:
            raise CudaGraphRejected("replay row map differs from its sealed qualification")

    def _run(
        self,
        plan: ec.LaunchPlan,
        intent: ReplayIntent,
        inputs: TensorMap,
        state: object,
        padding: gp.GraphPaddingBuffers | None,
    ) -> TensorMap:
        self._require_ready(production=False)
        slot = self._owned_slot(plan.replay_lease)
        session: GraphProgramSession | None = None
        try:
            session = self.program.fill(slot.program_slot, plan, intent, inputs, state, padding)
            if not isinstance(session, GraphProgramSession):
                raise RuntimeError("graph program returned an invalid replay session")
            session.replay()
            outputs = session.consume()
            if not isinstance(outputs, Mapping) or any(
                type(name) is not str or not isinstance(tensor, Tensor)
                for name, tensor in outputs.items()
            ):
                raise RuntimeError("graph program returned an invalid logical output map")
            with self._condition:
                if self._poison_reason is not None:
                    raise CudaGraphPoisoned(self._poison_reason)
                session.commit()
            return outputs
        except (CudaGraphRejected, CudaGraphOverload):
            self._abort(session)
            raise
        except CudaGraphPoisoned:
            self._abort(session)
            raise
        except Exception as error:
            self._poison("replay", error)
            self._abort(session)
            raise CudaGraphPoisoned(self._poison_reason) from error

    def _abort(self, session: GraphProgramSession | None) -> None:
        if session is None:
            return
        try:
            session.abort()
        except Exception as error:
            self._poison("replay abort", error)
            raise CudaGraphPoisoned(self._poison_reason) from error

    def _issue_witness(
        self,
        result: gp.GraphPaddingResult,
        intent: ReplayIntent,
        case: gp.GraphPaddingCase,
    ) -> QualificationWitness:
        self._require_ready(production=False)
        self._validate_intent(intent)
        poison_ids = tuple(poison.poison_id for poison in case.poisons)
        if (
            not result.passed
            or result.case_id != case.case_id
            or result.bucket_id != self.bucket.bucket_id
            or result.logical_m != intent.batch_plan.shape.logical_m
            or result.row_map_digest != _row_map_digest(intent.rows)
            or result.poison_ids != poison_ids
            or tuple(launch.poison_id for launch in result.replay_launches) != poison_ids
            or any(
                launch.execution_plan_id != self.binding.execution_plan.execution_plan_id
                or launch.source_plan_id != intent.batch_plan.plan_id
                for launch in result.replay_launches
            )
        ):
            raise CudaGraphRejected("qualification result is not bound to its exact replay case")
        capture = self._capture_fingerprint
        assert capture is not None
        if result.state_path_id != self._state_path_id or result.state_path_digest is None:
            raise CudaGraphRejected("qualification result used a different state path")
        try:
            _require_digest(result.state_path_digest, "state path")
        except ValueError as error:
            raise CudaGraphRejected(f"invalid qualification state path: {error}") from error
        try:
            return QualificationWitness(
                domain_fingerprint=self.binding.domain_fingerprint,
                capture_fingerprint=capture,
                capture_epoch=self._capture_epoch,
                bucket=self.bucket,
                logical_m=result.logical_m,
                row_map_digest=result.row_map_digest,
                case_digest=_case_digest(case),
                state_path_id=result.state_path_id,
                state_path_digest=result.state_path_digest,
                poison_ids=poison_ids,
                result=result,
                _runtime_token=self._witness_token,
                _issuer=_WITNESS_ISSUER,
            )
        except (TypeError, ValueError) as error:
            raise CudaGraphRejected(f"invalid qualification witness: {error}") from error

    def _qualification_attempt(self, intent: ReplayIntent) -> gp.GraphQualificationAttempt:
        lease, plan = self._begin(intent, production=False)
        try:
            factory = self._qualification_state_factory(plan, intent)
            return _QualificationAttempt(self, lease, plan, intent, factory)
        except Exception:
            self._finish(lease)
            raise

    def _qualification_state_factory(
        self, plan: ec.LaunchPlan, intent: ReplayIntent
    ) -> gp.GraphQualificationStateFactory:
        if self.program.state_path_id != self._state_path_id:
            raise CudaGraphRejected("graph program state path identity changed")
        slot = self._owned_slot(plan.replay_lease)
        return self._validate_state_factory(
            self.program.qualification_state_factory(slot.program_slot, plan, intent)
        )

    def _validate_state_factory(
        self, factory: gp.GraphQualificationStateFactory
    ) -> gp.GraphQualificationStateFactory:
        if (
            not isinstance(factory, gp.GraphQualificationStateFactory)
            or factory.state_path_id != self._state_path_id
        ):
            raise CudaGraphRejected("graph program returned a foreign state factory")
        try:
            _require_digest(factory.state_path_digest, "state path")
        except ValueError as error:
            raise CudaGraphRejected(f"invalid qualification state factory: {error}") from error
        return factory

    def _acquire(self, launch_id: str, deadline: AcquireDeadline) -> ec.ReplayLease:
        with self._condition:
            while True:
                if self._poison_reason is not None:
                    raise CudaGraphPoisoned(self._poison_reason)
                try:
                    lease = self._pool.acquire(
                        launch_plan_id=launch_id,
                        execution_plan_id=self.binding.execution_plan.execution_plan_id,
                        bucket_id=self.bucket.bucket_id,
                        template_id=self.bucket.template_id,
                        topology_id=self.bucket.topology_id,
                        partition_id=self.bucket.partition_id,
                    )
                except ReplayLeaseUnavailable:
                    remaining = deadline.remaining()
                    if remaining <= 0:
                        raise CudaGraphOverload(
                            "topology-wide replay lease unavailable before deadline"
                        ) from None
                    self._condition.wait(remaining)
                    continue
                try:
                    self._slot(lease).bind(lease)
                except Exception as error:
                    self._poison_locked("lease bind", error)
                    try:
                        self._pool.release(lease)
                    except Exception as release_error:
                        self._poison_locked("lease bind cleanup", release_error)
                    raise CudaGraphPoisoned(self._poison_reason) from error
                return lease

    def _slot(self, lease: ec.ReplayLease) -> _ReplaySlot:
        slot = self._slots.get(lease.rank_resources)
        if slot is None:
            raise RuntimeError("replay lease does not identify a captured program slot")
        return slot

    def _owned_slot(self, lease: ec.ReplayLease) -> _ReplaySlot:
        slot = self._slot(lease)
        if (slot.owner_lease_id, slot.owner_epoch) != (lease.lease_id, lease.epoch):
            raise RuntimeError("program slot is not owned by the active topology-wide lease")
        return slot

    def _finish(self, lease: ec.ReplayLease) -> None:
        self._release(lease)

    def _release(self, lease: ec.ReplayLease) -> None:
        with self._condition:
            errors: list[Exception] = []
            try:
                self._slot(lease).unbind(lease)
            except Exception as error:
                errors.append(error)
                self._poison_locked("lease unbind", error)
            try:
                self._pool.release(lease)
            except Exception as error:
                errors.append(error)
                self._poison_locked("lease pool release", error)
            self._condition.notify_all()
            if errors:
                raise CudaGraphPoisoned(self._poison_reason) from errors[0]

    def _require_ready(self, *, production: bool) -> None:
        with self._condition:
            if self._poison_reason is not None:
                raise CudaGraphPoisoned(self._poison_reason)
            if not self._captured:
                raise CudaGraphRejected("CUDA Graph bucket has not been captured")
            if production and not self._sealed:
                raise CudaGraphRejected("CUDA Graph bucket has no complete qualification evidence")

    def _admit_qualification(self) -> None:
        with self._condition:
            if self._poison_reason is not None:
                raise CudaGraphPoisoned(self._poison_reason)
            if not self._captured:
                raise CudaGraphRejected("CUDA Graph bucket has not been captured")
            if self._sealed:
                raise CudaGraphRejected("a sealed CUDA Graph bucket cannot be requalified")
            self._qualifications_in_flight += 1

    def _finish_qualification(self) -> None:
        with self._condition:
            if self._qualifications_in_flight <= 0:
                raise RuntimeError("qualification admission counter underflow")
            self._qualifications_in_flight -= 1
            self._condition.notify_all()

    def _poison(self, stage: str, error: Exception) -> None:
        with self._condition:
            self._poison_locked(stage, error)

    def _poison_locked(self, stage: str, error: Exception) -> None:
        if self._poison_reason is None:
            self._poison_reason = f"{stage} graph poison: {type(error).__name__}: {error}"
        self._condition.notify_all()


class CudaGraphDecodeDispatcher:
    """Route one immutable decode batch to its exact registered graph runtime."""

    def __init__(self, runtimes: tuple[CudaGraphRuntime, ...]) -> None:
        if type(runtimes) is not tuple or not runtimes:
            raise CudaGraphRejected("decode dispatcher requires registered graph runtimes")
        if any(type(runtime) is not CudaGraphRuntime for runtime in runtimes):
            raise CudaGraphRejected("decode dispatcher requires typed CUDA Graph runtimes")

        execution_plan = runtimes[0].binding.execution_plan
        first = runtimes[0].bucket
        domain = (
            first.scope_id,
            first.replay_abi_id,
            first.topology_id,
            first.partition_id,
        )
        routes: dict[tuple[ec.CapacityItems, int], CudaGraphRuntime] = {}
        for runtime in runtimes:
            bucket = runtime.bucket
            if runtime.binding.execution_plan is not execution_plan:
                raise CudaGraphRejected(
                    "decode dispatcher runtimes must share one immutable execution plan"
                )
            if bucket.phase is not ec.BatchPhase.DECODE:
                raise CudaGraphRejected("decode dispatcher cannot register a non-decode graph")
            if (
                bucket.scope_id,
                bucket.replay_abi_id,
                bucket.topology_id,
                bucket.partition_id,
            ) != domain:
                raise CudaGraphRejected(
                    "decode dispatcher runtimes must share one replay ABI and graph domain"
                )
            for logical_m in bucket.logical_m_values:
                key = (bucket.capacity_items, logical_m)
                if key in routes:
                    raise CudaGraphRejected(
                        "decode dispatcher graph domains overlap for one logical batch"
                    )
                routes[key] = runtime

        self._execution_plan = execution_plan
        self._scope_id, self._replay_abi_id, self._topology_id, self._partition_id = domain
        self._runtimes = runtimes
        self._routes = routes

    @property
    def bucket_ids(self) -> tuple[str, ...]:
        return tuple(runtime.bucket.bucket_id for runtime in self._runtimes)

    def replay(
        self,
        *,
        batch_plan: ec.BatchPlan,
        rows: Rows,
        deadline: AcquireDeadline,
        logical_inputs: TensorMap,
        state: object,
    ) -> TensorMap:
        """Replay the unique exact bucket; missing capacity is a typed rejection."""

        if not isinstance(batch_plan, ec.BatchPlan):
            raise CudaGraphRejected("graph dispatch requires a typed immutable BatchPlan")
        if batch_plan.phase is not ec.BatchPhase.DECODE:
            raise CudaGraphRejected("decode graph dispatcher accepts decode-only batches")
        if not isinstance(deadline, AcquireDeadline):
            raise CudaGraphRejected("graph dispatch requires a typed finite acquire deadline")
        if (
            batch_plan.topology_id,
            batch_plan.partition_id,
        ) != (self._topology_id, self._partition_id):
            raise CudaGraphRejected("decode batch belongs to a different graph domain")

        runtime = self._routes.get(
            (batch_plan.shape.capacity_items, batch_plan.shape.logical_m)
        )
        if runtime is None:
            raise CudaGraphRejected(
                "logical decode batch has no registered sealed graph bucket; "
                "eager/exact-M fallback is forbidden"
            )
        try:
            selected = self._execution_plan.bucket_for(
                scope_id=self._scope_id,
                replay_abi_id=self._replay_abi_id,
                phase=batch_plan.phase,
                logical_m=batch_plan.shape.logical_m,
                capacity_items=batch_plan.shape.capacity_items,
                topology_id=batch_plan.topology_id,
                partition_id=batch_plan.partition_id,
            )
        except ValueError as error:
            raise CudaGraphRejected(f"decode graph dispatch rejected the batch: {error}") from error
        if selected != runtime.bucket:
            raise CudaGraphRejected("registered decode runtime differs from the execution plan")
        return runtime.replay(
            intent=ReplayIntent(batch_plan, rows, deadline),
            logical_inputs=logical_inputs,
            state=state,
        )


def _validate_partition_closure(
    topology: ec.ParallelTopology,
    partition: ec.PartitionPlan,
    ranks: tuple[int, ...],
) -> None:
    valid_ranks = set(ranks)
    placements = {placement.rank: placement for placement in topology.placements}
    owners = {
        assignment.node_id: set(assignment.owner_ranks) for assignment in partition.assignments
    }
    if any(
        not set(assignment.owner_ranks) <= valid_ranks
        or any(
            placements[rank].pipeline_stage != assignment.pipeline_stage
            for rank in assignment.owner_ranks
        )
        for assignment in partition.assignments
    ):
        raise CudaGraphRejected("partition assignment has an unknown rank or pipeline stage")
    if set().union(*owners.values()) != valid_ranks:
        raise CudaGraphRejected("partition assignments must cover every topology rank")
    group_ids = {group.group_id for group in topology.groups}
    if any(step.group_id not in group_ids for step in partition.collectives):
        raise CudaGraphRejected("partition collective references an unknown topology group")
    if any(
        assignment.node_id not in owners or assignment.owner_rank not in owners[assignment.node_id]
        for assignment in partition.expert_assignments
    ):
        raise CudaGraphRejected("partition expert owner does not own its graph node")
    if any(
        edge.source_node_id not in owners
        or edge.target_node_id not in owners
        or edge.source_rank not in owners[edge.source_node_id]
        or edge.target_rank not in owners[edge.target_node_id]
        for edge in partition.pipeline_edges
    ):
        raise CudaGraphRejected("partition pipeline edge rank does not own its endpoint")
    if 1 < topology.ep_size < topology.tp_size or any(
        assignment.expert_tensor_ranks for assignment in partition.expert_assignments
    ):
        from cubic_serve.executor.parallel.topology import seal_partition

        try:
            seal_partition(topology, partition)
        except ValueError as error:
            raise CudaGraphRejected(
                f"partition expert tensor ownership is open: {error}"
            ) from error


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


def _row_map_digest(rows: Rows) -> str:
    return gp.row_map_digest(rows)


def _case_digest(case: gp.GraphPaddingCase) -> str:
    return stable_digest(
        {
            "case_id": case.case_id,
            "inputs": {
                name: tensor_digest(value) for name, value in sorted(case.logical_inputs.items())
            },
            "state": {
                name: tensor_digest(value)
                for name, value in sorted(case.initial_committed_state.items())
            },
            "poisons": [
                {
                    "id": poison.poison_id,
                    "inputs": {
                        name: tensor_digest(value)
                        for name, value in sorted(poison.padding_inputs.items())
                    },
                    "scratch": {
                        name: tensor_digest(value)
                        for name, value in sorted(poison.scratch_state.items())
                    },
                    "guards": {
                        name: tensor_digest(value) for name, value in sorted(poison.guards.items())
                    },
                }
                for poison in case.poisons
            ],
        }
    )
