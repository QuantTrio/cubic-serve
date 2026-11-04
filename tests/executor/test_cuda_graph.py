from __future__ import annotations

import inspect
import os
import unittest
from collections.abc import Mapping, MutableMapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from threading import TIMEOUT_MAX, Event, Lock
from unittest.mock import patch

import torch
from torch import Tensor

from cubic_serve.contracts.execution import (
    ActiveLaunchRow,
    BatchPhase,
    BatchPlan,
    BatchShape,
    CandidateBackend,
    CandidateBinding,
    CollectiveStep,
    ExecutionPlan,
    ExpertAssignment,
    GraphAssignment,
    GraphBucket,
    GraphSafetyContract,
    LaunchPlan,
    LaunchRow,
    OpBinding,
    OpSignature,
    PaddingIsolation,
    PaddingLaunchRow,
    ParallelTopology,
    PartitionPlan,
    PlanSegment,
    RankReplayResources,
    ReplayLease,
    StateAccess,
    StateTxn,
    TensorSpec,
)
from cubic_serve.contracts.model import ScalarDType
from cubic_serve.contracts.request import TokenPosition
from cubic_serve.executor.cuda_graph import (
    AcquireDeadline,
    CudaGraphDecodeDispatcher,
    CudaGraphOverload,
    CudaGraphPoisoned,
    CudaGraphRejected,
    CudaGraphRuntime,
    GraphProgramSession,
    GraphProgramSlot,
    GraphRuntimeBinding,
    QualificationWitness,
    ReplayIntent,
)
from cubic_serve.guardrail.compare import (
    ComparisonPolicy,
    ComparisonResult,
    DifferenceClass,
    compare_tensors,
    stable_digest,
)
from cubic_serve.guardrail.graph_padding import (
    GraphPaddingBuffers,
    GraphPaddingCase,
    GraphPaddingCheck,
    GraphPaddingPoison,
    GraphPaddingResult,
    GraphQualificationRun,
    GraphQualificationStateFactory,
    GraphReplayLaunchIdentity,
    row_map_digest,
)
from cubic_serve.guardrail.oracle import clone_tensor_map
from cubic_serve.runtime.cache import (
    CacheArena,
    CacheArenaBinding,
    CacheArenaSpec,
    CacheStateStore,
    KVLayout,
    RecurrentStateProvision,
    ScratchArenaLeaseError,
    StateSlotSpec,
    StateTransaction,
)
from tests.guardrail.operator_evidence_fixture import sealed_test_performance_plan

RUN_REAL = os.environ.get("CUBIC_RUN_CUDA_GRAPH_TESTS") == "1"
REPLAY_ABI = "cubic.replay.toy-state-row.v1"
PRODUCTION_CANARY = 0xC0B1C
TOY_STATE_PATH_ID = "test.toy-committed-state-txn.v1"
TOY_STATE_PATH_DIGEST = stable_digest(
    {"state_path": TOY_STATE_PATH_ID, "factory": "ToyQualificationStateFactory.v1"}
)
PROBE_STATE_PATH_ID = "test.no-state.v1"
PROBE_STATE_PATH_DIGEST = stable_digest(
    {"state_path": PROBE_STATE_PATH_ID, "factory": "ProbeQualificationStateFactory.v1"}
)


class ExpertTensorGraphClosureTests(unittest.TestCase):
    def test_graph_binding_requires_complete_expert_tensor_ownership(self):
        from cubic_serve.executor.cuda_graph import _validate_partition_closure

        topology = ParallelTopology.single_node(
            topology_id="tp4.ep2",
            device_indices=(0, 1, 2, 3),
            tp_size=4,
            pp_size=1,
            ep_size=2,
        )
        partition = PartitionPlan(
            "experts",
            topology.topology_id,
            (GraphAssignment("moe", 0, (0, 1, 2, 3)),),
            expert_assignments=(ExpertAssignment("moe", 0, 0, (0, 1)),),
        )
        _validate_partition_closure(topology, partition, (0, 1, 2, 3))
        for experts in (
            (),
            (ExpertAssignment("moe", 0, 0),),
            (ExpertAssignment("moe", 0, 0, (0, 2)),),
        ):
            with self.assertRaisesRegex(CudaGraphRejected, "expert tensor ownership"):
                _validate_partition_closure(
                    topology,
                    replace(partition, expert_assignments=experts),
                    (0, 1, 2, 3),
                )


STATE_DTYPES = {
    "kv": torch.bfloat16,
    "recurrent": torch.float32,
    "route": torch.int64,
    "sampler": torch.int64,
    "txn": torch.int64,
}


def _bucket() -> GraphBucket:
    return GraphBucket(
        bucket_id="decode:row4",
        template_id="toy:state-row:4",
        scope_id="model.forward",
        replay_abi_id=REPLAY_ABI,
        topology_id="tp2:pp1",
        partition_id="toy:tp2",
        phase=BatchPhase.DECODE,
        capacity_items=(("cache_pages", 4), ("request_slots", 4)),
        logical_m_values=(2, 3, 4),
        captured_m=4,
        row_map_schema_id="cubic.rows.v1",
        scratch_schema_id="cubic.scratch.toy-state-row.v1",
        graph_safety_contract_ids=("graph-safe:toy-state-row:v1",),
        evidence_id="evidence:cuda-graph:toy-state-row:exact",
    )


def _topology() -> ParallelTopology:
    return ParallelTopology.single_node(
        topology_id="tp2:pp1", device_indices=(0, 1), tp_size=2, pp_size=1
    )


def _partition() -> PartitionPlan:
    return PartitionPlan(
        partition_id="toy:tp2",
        topology_id="tp2:pp1",
        assignments=(GraphAssignment("toy.state_row", 0, (0, 1)),),
    )


def _execution(bucket: GraphBucket | None = None) -> ExecutionPlan:
    selected = bucket or _bucket()
    safety = GraphSafetyContract(
        contract_id=selected.graph_safety_contract_ids[0],
        row_map_schema_id=selected.row_map_schema_id,
        isolation=PaddingIsolation.PRIVATE_SCRATCH,
        evidence_id="evidence:toy:scratch",
    )
    signature = OpSignature(
        semantic_op="toy.state_row",
        phase=BatchPhase.DECODE,
        inputs=(TensorSpec("x", ScalarDType.BF16, ("m",)),),
        outputs=(TensorSpec("logits", ScalarDType.BF16, ("m", 2)),),
        accumulator_dtype=ScalarDType.FP32,
        layout="row",
        shape_items=(),
        cache_form="bf16",
        state_form="toy",
        topology_id=selected.topology_id,
    )
    candidate = CandidateBinding(
        candidate_id="toy:cuda",
        implementation_id="b2s256:toy",
        backend=CandidateBackend.CUDA,
        launch_signature="row4",
        workspace_bytes=0,
        evidence_id="evidence:toy:exact",
        graph_safety=safety,
    )
    return sealed_test_performance_plan(
        "performance:toy",
        (OpBinding(signature=signature, candidate=candidate),),
        (selected,),
    )


def _binding(
    *,
    bucket: GraphBucket | None = None,
    topology: ParallelTopology | None = None,
    partition: PartitionPlan | None = None,
    execution: ExecutionPlan | None = None,
) -> GraphRuntimeBinding:
    selected = bucket or _bucket()
    plan = execution or _execution(selected)
    if execution is None:
        selected = plan.graph_buckets[0]
    return GraphRuntimeBinding(
        plan,
        topology or _topology(),
        partition or _partition(),
        selected,
    )


def _resources(slots: int = 2) -> tuple[tuple[RankReplayResources, ...], ...]:
    return tuple(
        tuple(
            RankReplayResources(
                rank=rank,
                graph_instance_id=f"graph:{slot}:rank{rank}",
                stream_lane_id=f"stream:{slot}:rank{rank}",
                buffer_set_id=f"buffers:{slot}:rank{rank}",
                workspace_arena_id=f"workspace:{slot}:rank{rank}",
                scratch_arena_id=f"scratch:{slot}:rank{rank}",
            )
            for rank in (0, 1)
        )
        for slot in range(slots)
    )


def _rows(logical_m: int) -> tuple[LaunchRow, ...]:
    if logical_m == 2:
        return (
            PaddingLaunchRow(1),
            ActiveLaunchRow(1),
            PaddingLaunchRow(0),
            ActiveLaunchRow(0),
        )
    if logical_m == 3:
        return (
            ActiveLaunchRow(2),
            PaddingLaunchRow(0),
            ActiveLaunchRow(0),
            ActiveLaunchRow(1),
        )
    return tuple(ActiveLaunchRow(index) for index in range(4))


def _batch(
    logical_m: int,
    *,
    topology_id: str = "tp2:pp1",
    capacity_items: tuple[tuple[str, int], ...] = (
        ("cache_pages", 4),
        ("request_slots", 4),
    ),
) -> BatchPlan:
    request_id = f"request:m{logical_m}"
    return BatchPlan(
        plan_id=f"batch:m{logical_m}",
        phase=BatchPhase.DECODE,
        input_plan_ids=(request_id,),
        segments=(
            PlanSegment(
                request_id=request_id,
                input_plan_digest=f"b2s256:{'a' * 64}",
                execution_input_digest=f"b2s256:{'b' * 64}",
                phase=BatchPhase.DECODE,
                produces_output=True,
                token_start=0,
                token_count=logical_m,
                token_ids=tuple(range(logical_m)),
                positions=tuple(TokenPosition((index,)) for index in range(logical_m)),
                cache_slots=tuple(range(logical_m)),
                cache_pages=(0,),
            ),
        ),
        shape=BatchShape(logical_m, capacity_items),
        topology_id=topology_id,
        partition_id="toy:tp2",
        state_transactions=(
            StateTxn(
                transaction_id=f"txn:m{logical_m}",
                request_id=request_id,
                base_revision=0,
                target_revision=1,
                reads=tuple(
                    StateAccess(
                        kind,
                        rank,
                        slots=tuple(range(logical_m)),
                        pages=(0,) if kind == "kv" else (),
                    )
                    for rank in (0, 1)
                    for kind in ("kv", "recurrent", "txn")
                ),
                writes=tuple(
                    StateAccess(
                        kind,
                        rank,
                        slots=tuple(range(logical_m)),
                        pages=(0,) if kind == "kv" else (),
                    )
                    for rank in (0, 1)
                    for kind in STATE_DTYPES
                ),
                expected_owner_ranks=(0, 1),
            ),
        ),
    )


def _intent(logical_m: int, *, timeout_s: float = 1.0) -> ReplayIntent:
    return ReplayIntent(_batch(logical_m), _rows(logical_m), AcquireDeadline.after(timeout_s))


def _state(logical_m: int) -> dict[str, Tensor]:
    values: dict[str, Tensor] = {}
    for rank in (0, 1):
        device = torch.device(f"cuda:{rank}")
        for ordinal, (kind, dtype) in enumerate(STATE_DTYPES.items()):
            base = torch.arange(logical_m, dtype=torch.float32, device=device) + rank * 13 + ordinal
            values[f"{kind}:rank{rank}"] = base.to(dtype=dtype)
    return values


def _scratch(padding_m: int, salt: int) -> dict[str, Tensor]:
    values: dict[str, Tensor] = {}
    for rank in (0, 1):
        device = torch.device(f"cuda:{rank}")
        for ordinal, (kind, dtype) in enumerate(STATE_DTYPES.items()):
            base = torch.arange(padding_m, dtype=torch.float32, device=device)
            values[f"{kind}:rank{rank}"] = (base + salt + rank * 17 + ordinal).to(dtype)
    return values


def _case(logical_m: int) -> GraphPaddingCase:
    x = torch.tensor((1.5, -2.25, 3.0, -4.5)[:logical_m], dtype=torch.bfloat16)
    padding_m = 4 - logical_m
    poisons = tuple(
        GraphPaddingPoison(
            poison_id=name,
            padding_inputs=(
                {"x": torch.full((padding_m,), value, dtype=torch.bfloat16)} if padding_m else {}
            ),
            scratch_state=_scratch(padding_m, salt) if padding_m else {},
            guards={"sentinel": torch.tensor([salt], dtype=torch.int64)},
        )
        for name, value, salt in (
            ("positive", 37.0, 101),
            ("negative", -73.0, 211),
        )
    )
    return GraphPaddingCase(
        case_id=f"cuda-graph/m{logical_m}/bucket4",
        logical_inputs={"x": x},
        initial_committed_state=_state(logical_m),
        poisons=poisons,
    )


def _oracle(
    inputs: Mapping[str, Tensor], state: MutableMapping[str, Tensor]
) -> Mapping[str, Tensor]:
    outputs: dict[str, Tensor] = {}
    for rank in (0, 1):
        x = inputs["x"].to(device=f"cuda:{rank}")
        logits = torch.stack((x, -x), dim=1)
        state[f"kv:rank{rank}"].add_(x)
        state[f"recurrent:rank{rank}"].add_(x)
        state[f"route:rank{rank}"].copy_(x > 0)
        state[f"sampler:rank{rank}"].copy_(torch.argmax(logits, dim=1))
        state[f"txn:rank{rank}"].add_(1)
        outputs[f"logits:rank{rank}"] = logits
    return outputs


class ToyCommittedState:
    def __init__(self, *, request_id: str, revision: int, tensors: Mapping[str, Tensor]) -> None:
        self.request_id = request_id
        self._revision = revision
        self._tensors = clone_tensor_map(tensors, "toy committed state")
        self._lock = Lock()
        self._active: str | None = None
        self._used: set[str] = set()

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def snapshot(self) -> dict[str, Tensor]:
        with self._lock:
            return clone_tensor_map(self._tensors, "toy committed snapshot")

    def begin(self, transaction: StateTxn, deadline: AcquireDeadline) -> dict[str, Tensor]:
        if not self._lock.acquire(timeout=min(deadline.remaining(), TIMEOUT_MAX)):
            raise CudaGraphOverload("toy state transaction unavailable before deadline")
        try:
            if (
                transaction.request_id != self.request_id
                or transaction.transaction_id in self._used
                or transaction.base_revision != self._revision
            ):
                raise CudaGraphRejected("toy state transaction or revision is stale")
            self._active = transaction.transaction_id
            self._used.add(transaction.transaction_id)
            return self._tensors
        except Exception:
            self._lock.release()
            raise

    def commit(self, transaction: StateTxn, state: Mapping[str, Tensor]) -> None:
        if (
            self._active != transaction.transaction_id
            or transaction.base_revision != self._revision
            or set(state) != set(self._tensors)
        ):
            raise RuntimeError("toy state commit lost its transaction or schema")
        self._tensors = dict(state)
        self._revision = transaction.target_revision

    def finish(self, transaction: StateTxn) -> None:
        if self._active != transaction.transaction_id:
            raise RuntimeError("toy state cannot finish a foreign transaction")
        self._active = None
        self._lock.release()


@dataclass(slots=True)
class ToyQualificationRun:
    state: ToyCommittedState
    target_revision: int
    finished: str | None = None

    @property
    def production_state(self) -> object:
        return self.state

    def committed_snapshot(self) -> Mapping[str, Tensor]:
        if self.state.revision != self.target_revision:
            raise RuntimeError("toy qualification state was not committed")
        return self.state.snapshot()

    def close(self) -> None:
        self.finished = "closed"

    def abort(self) -> None:
        self.finished = "aborted"


@dataclass(slots=True)
class ToyQualificationStateFactory:
    request_id: str
    base_revision: int
    target_revision: int
    runs: list[ToyQualificationRun] = field(default_factory=list)
    state_path_id: str = field(init=False, default=TOY_STATE_PATH_ID)
    state_path_digest: str = field(init=False, default=TOY_STATE_PATH_DIGEST)

    def create(self, canonical_seed: Mapping[str, Tensor]) -> GraphQualificationRun:
        run = ToyQualificationRun(
            ToyCommittedState(
                request_id=self.request_id,
                revision=self.base_revision,
                tensors=canonical_seed,
            ),
            self.target_revision,
        )
        self.runs.append(run)
        return run


@dataclass(slots=True)
class _ToyLane:
    resource: RankReplayResources
    device: torch.device
    stream: torch.cuda.Stream
    graph: object
    inputs: Tensor
    logits: Tensor
    states: dict[str, Tensor]
    guard: Tensor

    @classmethod
    def capture(
        cls, resource: RankReplayResources, device: torch.device, captured_m: int
    ) -> _ToyLane:
        with torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
            inputs = torch.zeros(captured_m, dtype=torch.bfloat16, device=device)
            logits = torch.empty((captured_m, 2), dtype=torch.bfloat16, device=device)
            states = {
                kind: torch.zeros(captured_m, dtype=dtype, device=device)
                for kind, dtype in STATE_DTYPES.items()
            }
            guard = torch.zeros(2, dtype=torch.int64, device=device)

            def row_program() -> None:
                logits[:, 0].copy_(inputs)
                logits[:, 1].copy_(inputs)
                logits[:, 1].neg_()
                states["kv"].add_(inputs)
                states["recurrent"].add_(inputs)
                states["route"].copy_(inputs > 0)
                torch.argmax(logits, dim=1, out=states["sampler"])
                states["txn"].add_(1)

            with torch.cuda.stream(stream):
                row_program()
                row_program()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                row_program()
            stream.synchronize()
        return cls(resource, device, stream, graph, inputs, logits, states, guard)


@dataclass(slots=True)
class ToyGraphSlot:
    resources: tuple[RankReplayResources, ...]
    lanes: tuple[_ToyLane, ...]


@dataclass(slots=True)
class ToyGraphSession:
    slot: ToyGraphSlot
    plan: LaunchPlan
    state: MutableMapping[str, Tensor]
    sentinel: int
    transaction: tuple[ToyCommittedState, StateTxn]
    outputs: dict[str, Tensor] = field(default_factory=dict)
    durable_state: dict[str, Tensor] = field(default_factory=dict)
    finished: bool = False

    def replay(self) -> None:
        for lane in self.slot.lanes:
            with torch.cuda.device(lane.device), torch.cuda.stream(lane.stream):
                lane.guard[0].fill_(self.plan.replay_lease.epoch)
                lane.guard[1].fill_(self.sentinel)
                lane.graph.replay()  # type: ignore[attr-defined]

    def consume(self) -> Mapping[str, Tensor]:
        indices = _physical_indices(self.plan.rows)
        guards: list[Tensor] = []
        for lane in self.slot.lanes:
            lane.stream.synchronize()
            index = torch.tensor(indices, dtype=torch.int64, device=lane.device)
            self.outputs[f"logits:rank{lane.resource.rank}"] = lane.logits.index_select(
                0, index
            ).clone()
            for kind in STATE_DTYPES:
                key = f"{kind}:rank{lane.resource.rank}"
                self.durable_state[key] = lane.states[kind].index_select(0, index).clone()
            guards.append(lane.guard.clone())
        expected = (self.plan.replay_lease.epoch, self.sentinel)
        if any(tuple(int(value) for value in guard.tolist()) != expected for guard in guards):
            raise RuntimeError("toy lease guard was corrupted")
        return self.outputs

    def commit(self) -> None:
        if self.finished:
            raise RuntimeError("toy replay session is already finished")
        owner, transaction = self.transaction
        owner.commit(transaction, self.durable_state)
        owner.finish(transaction)
        self.finished = True

    def abort(self) -> None:
        if self.finished:
            return
        for lane in self.slot.lanes:
            lane.stream.synchronize()
        owner, transaction = self.transaction
        owner.finish(transaction)
        self.finished = True


class ToyGraphProgram:
    program_id = "test.toy-state-row.cuda.v1"
    replay_abi_id = REPLAY_ABI
    state_path_id = TOY_STATE_PATH_ID

    def __init__(self) -> None:
        self.binding: GraphRuntimeBinding | None = None
        self.captured_slots: list[ToyGraphSlot] = []
        self.qualification_factories: list[ToyQualificationStateFactory] = []

    def validate_binding(self, binding: GraphRuntimeBinding) -> None:
        if binding.topology.pp_size != 1 or binding.topology.ep_size != 1:
            raise CudaGraphRejected("toy program does not support PP or EP")
        if (
            binding.partition.collectives
            or binding.partition.expert_assignments
            or binding.partition.pipeline_edges
        ):
            raise CudaGraphRejected("toy program does not support communication schedules")
        if len(binding.execution_plan.bindings) != 1:
            raise CudaGraphRejected("toy program requires one candidate")
        candidate = binding.execution_plan.bindings[0]
        if (
            candidate.signature.semantic_op != "toy.state_row"
            or candidate.candidate is None
            or candidate.candidate.backend is not CandidateBackend.CUDA
            or candidate.candidate.workspace_bytes != 0
            or {item.node_id for item in binding.partition.assignments} != {"toy.state_row"}
        ):
            raise CudaGraphRejected("binding does not close the toy replay ABI")
        self.binding = binding

    def capture(
        self,
        binding: GraphRuntimeBinding,
        resources: tuple[RankReplayResources, ...],
    ) -> GraphProgramSlot:
        if binding is not self.binding:
            raise RuntimeError("toy capture received a foreign binding")
        devices = {
            placement.rank: placement.device_index for placement in binding.topology.placements
        }
        slot = ToyGraphSlot(
            resources,
            tuple(
                _ToyLane.capture(
                    resource,
                    torch.device(f"cuda:{devices[resource.rank]}"),
                    binding.bucket.captured_m,
                )
                for resource in resources
            ),
        )
        self.captured_slots.append(slot)
        return slot

    def qualification_state_factory(
        self,
        slot: GraphProgramSlot,
        plan: LaunchPlan,
        intent: ReplayIntent,
    ) -> GraphQualificationStateFactory:
        if not isinstance(slot, ToyGraphSlot) or plan.rows != intent.rows:
            raise CudaGraphRejected("toy qualification factory requires its replay slot")
        transaction = intent.batch_plan.state_transactions[0]
        factory = ToyQualificationStateFactory(
            transaction.request_id,
            transaction.base_revision,
            transaction.target_revision,
        )
        self.qualification_factories.append(factory)
        return factory

    def fill(
        self,
        slot: GraphProgramSlot,
        plan: LaunchPlan,
        intent: ReplayIntent,
        logical_inputs: Mapping[str, Tensor],
        state: object,
        padding: GraphPaddingBuffers | None,
    ) -> GraphProgramSession:
        if not isinstance(slot, ToyGraphSlot) or self.binding is None:
            raise CudaGraphRejected("toy fill requires its captured slot")
        self._validate_batch(intent.batch_plan)
        if not isinstance(state, ToyCommittedState):
            raise CudaGraphRejected("toy replay requires committed transactional state")
        txn = intent.batch_plan.state_transactions[0]
        mutable = state.begin(txn, intent.deadline)
        transaction = (state, txn)
        try:
            self._validate_payload(slot, plan, logical_inputs, mutable, padding)
            sentinel = (
                int(padding.guards["sentinel"][0].item())
                if padding is not None
                else PRODUCTION_CANARY
            )
            for lane in slot.lanes:
                lane.stream.wait_stream(torch.cuda.current_stream(lane.device))
                with torch.cuda.device(lane.device), torch.cuda.stream(lane.stream):
                    for physical, row in enumerate(plan.rows):
                        if isinstance(row, ActiveLaunchRow):
                            lane.inputs[physical].copy_(logical_inputs["x"][row.logical_index])
                            for kind in STATE_DTYPES:
                                key = f"{kind}:rank{lane.resource.rank}"
                                lane.states[kind][physical].copy_(mutable[key][row.logical_index])
                        elif padding is None:
                            lane.inputs[physical].zero_()
                            for kind in STATE_DTYPES:
                                lane.states[kind][physical].zero_()
                        else:
                            lane.inputs[physical].copy_(
                                padding.padding_inputs["x"][row.scratch_index]
                            )
                            for kind in STATE_DTYPES:
                                key = f"{kind}:rank{lane.resource.rank}"
                                lane.states[kind][physical].copy_(
                                    padding.scratch_state[key][row.scratch_index]
                                )
            return ToyGraphSession(slot, plan, mutable, sentinel, transaction)
        except Exception:
            if transaction is not None:
                transaction[0].finish(transaction[1])
            raise

    def _validate_batch(self, batch: BatchPlan) -> None:
        assert self.binding is not None
        if len(batch.segments) != 1 or len(batch.state_transactions) != 1:
            raise CudaGraphRejected("toy ABI requires one request transaction")
        expected_resources = {
            (kind, rank) for rank in self.binding.participating_ranks for kind in STATE_DTYPES
        }
        expected_reads = {
            (kind, rank)
            for rank in self.binding.participating_ranks
            for kind in ("kv", "recurrent", "txn")
        }
        segment = batch.segments[0]
        transaction = batch.state_transactions[0]
        if {
            (item.resource_id, item.owner_rank) for item in transaction.writes
        } != expected_resources or {
            (item.resource_id, item.owner_rank) for item in transaction.reads
        } != expected_reads:
            raise CudaGraphRejected("toy transaction does not cover its complete state ABI")
        slots, pages = (
            tuple(sorted(set(segment.cache_slots))),
            tuple(sorted(set(segment.cache_pages))),
        )
        for access in (*transaction.reads, *transaction.writes):
            expected_pages = pages if access.resource_id == "kv" else ()
            if access.slots != slots or access.pages != expected_pages:
                raise CudaGraphRejected("toy transaction regions differ from the batch plan")

    def _validate_payload(
        self,
        slot: ToyGraphSlot,
        plan: LaunchPlan,
        inputs: Mapping[str, Tensor],
        state: Mapping[str, Tensor],
        padding: GraphPaddingBuffers | None,
    ) -> None:
        if set(inputs) != {"x"}:
            raise CudaGraphRejected("toy input ABI requires x")
        _check_tensor("x", inputs["x"], torch.bfloat16, plan.logical_m)
        expected = {
            f"{kind}:rank{lane.resource.rank}": (dtype, lane.device)
            for lane in slot.lanes
            for kind, dtype in STATE_DTYPES.items()
        }
        if set(state) != set(expected):
            raise CudaGraphRejected("toy state map differs from its complete ABI")
        for name, (dtype, device) in expected.items():
            _check_tensor(name, state[name], dtype, plan.logical_m, device)
        if padding is None:
            return
        padding_m = plan.captured_m - plan.logical_m
        if set(padding.padding_inputs) != ({"x"} if padding_m else set()):
            raise CudaGraphRejected("toy padding input schema differs")
        if set(padding.scratch_state) != (set(expected) if padding_m else set()):
            raise CudaGraphRejected("toy padding scratch schema differs")
        if set(padding.guards) != {"sentinel"}:
            raise CudaGraphRejected("toy qualification guard is missing")
        _check_tensor("sentinel", padding.guards["sentinel"], torch.int64, 1)
        if padding_m:
            _check_tensor("padding:x", padding.padding_inputs["x"], torch.bfloat16, padding_m)
            for name, (dtype, device) in expected.items():
                _check_tensor(
                    f"scratch:{name}", padding.scratch_state[name], dtype, padding_m, device
                )


@dataclass(slots=True)
class ProbeSlot:
    resources: tuple[RankReplayResources, ...]


@dataclass(slots=True)
class ProbeSession:
    entered: Event | None = None
    release: Event | None = None
    replay_error: Exception | None = None
    abort_error: Exception | None = None
    committed: bool = False
    aborted: bool = False

    def replay(self) -> None:
        if self.replay_error is not None:
            raise self.replay_error

    def consume(self) -> Mapping[str, Tensor]:
        if self.entered is not None:
            self.entered.set()
        if self.release is not None and not self.release.wait(1.0):
            raise RuntimeError("probe consume release timed out")
        return {"out": torch.tensor([1])}

    def commit(self) -> None:
        self.committed = True

    def abort(self) -> None:
        self.aborted = True
        if self.abort_error is not None:
            raise self.abort_error


@dataclass(slots=True)
class ProbeQualificationRun:
    state: dict[str, Tensor]

    @property
    def production_state(self) -> object:
        return self.state

    def committed_snapshot(self) -> Mapping[str, Tensor]:
        return clone_tensor_map(self.state, "probe qualification snapshot")

    def close(self) -> None:
        return None

    def abort(self) -> None:
        return None


class ProbeQualificationStateFactory:
    state_path_id = PROBE_STATE_PATH_ID
    state_path_digest = PROBE_STATE_PATH_DIGEST

    def create(self, canonical_seed: Mapping[str, Tensor]) -> GraphQualificationRun:
        return ProbeQualificationRun(clone_tensor_map(canonical_seed, "probe qualification state"))


class ProbeProgram:
    program_id = "test.probe.v1"
    replay_abi_id = REPLAY_ABI
    state_path_id = PROBE_STATE_PATH_ID

    def __init__(self, session: ProbeSession | None = None) -> None:
        self.session = session or ProbeSession()
        self.validated: GraphRuntimeBinding | None = None

    def validate_binding(self, binding: GraphRuntimeBinding) -> None:
        self.validated = binding

    def capture(
        self,
        binding: GraphRuntimeBinding,
        resources: tuple[RankReplayResources, ...],
    ) -> GraphProgramSlot:
        if binding is not self.validated:
            raise RuntimeError("probe received a foreign binding")
        return ProbeSlot(resources)

    def qualification_state_factory(
        self,
        slot: GraphProgramSlot,
        plan: LaunchPlan,
        intent: ReplayIntent,
    ) -> GraphQualificationStateFactory:
        del slot, plan, intent
        return ProbeQualificationStateFactory()

    def fill(
        self,
        slot: GraphProgramSlot,
        plan: LaunchPlan,
        intent: ReplayIntent,
        logical_inputs: Mapping[str, Tensor],
        state: object,
        padding: GraphPaddingBuffers | None,
    ) -> GraphProgramSession:
        del slot, plan, intent, logical_inputs, state, padding
        return self.session


ARENA_STATE_PATH_ID = "test.shared-arena-qualification.v1"
ARENA_STATE_PATH_DIGEST = stable_digest(ARENA_STATE_PATH_ID)


@dataclass(slots=True)
class ArenaQualificationRun:
    stores: tuple[CacheStateStore, ...]
    owners: tuple[str, ...]
    finished: str | None = None

    @property
    def production_state(self) -> object:
        return self

    def begin(self, launch: LaunchPlan) -> tuple[StateTransaction, ...]:
        return tuple(
            store.begin(
                store.plan(
                    f"txn:{launch.launch_plan_id}:{owner}:rank{rank}",
                    owner,
                    writes=(store.recurrent_region(owner, "state"),),
                    target_position=1,
                )
            )
            for rank, (store, owner) in enumerate(zip(self.stores, self.owners, strict=True))
        )

    def committed_snapshot(self) -> Mapping[str, Tensor]:
        return {
            f"revision:rank{rank}": torch.tensor([store.current_revision(owner)])
            for rank, (store, owner) in enumerate(zip(self.stores, self.owners, strict=True))
        }

    def close(self) -> None:
        self._retire("closed")

    def abort(self) -> None:
        self._retire("aborted")

    def _retire(self, status: str) -> None:
        for store, owner in zip(self.stores, self.owners, strict=True):
            store.retire_owner(
                owner,
                expected_revision=store.current_revision(owner),
                expected_page_ids=(),
            )
        self.finished = status


class ArenaQualificationFactory:
    state_path_id = ARENA_STATE_PATH_ID
    state_path_digest = ARENA_STATE_PATH_DIGEST

    def __init__(self, stores: tuple[CacheStateStore, ...]) -> None:
        self.stores = stores
        self.runs: list[ArenaQualificationRun] = []
        self.serial = 0

    def create(self, canonical_seed: Mapping[str, Tensor]) -> ArenaQualificationRun:
        expected = {f"revision:rank{rank}" for rank in range(len(self.stores))}
        if set(canonical_seed) != expected:
            raise ValueError("arena qualification seed differs from its state ABI")
        self.serial += 1
        owners = tuple(
            f"qualification:{self.serial}:rank{rank}" for rank in range(len(self.stores))
        )
        for store, owner in zip(self.stores, owners, strict=True):
            store.provision_owner(
                owner,
                (RecurrentStateProvision("state", torch.zeros(1, dtype=torch.bfloat16), 0),),
            )
        run = ArenaQualificationRun(self.stores, owners)
        self.runs.append(run)
        return run


@dataclass(slots=True)
class ArenaSession:
    transactions: tuple[StateTransaction, ...]
    bindings: tuple[CacheArenaBinding, ...]
    output: Tensor
    poison_id: str
    fail_poison: str | None
    finished: bool = False

    def replay(self) -> None:
        if self.poison_id == self.fail_poison:
            raise RuntimeError("injected arena qualification replay failure")
        for binding in self.bindings:
            state = dict(binding.state_tensors)["state"]
            slots = binding.state_slots[0]
            state[slots.destination_slot].copy_(state[slots.source_slot] + 1)

    def consume(self) -> Mapping[str, Tensor]:
        return {"out": self.output.clone()}

    def commit(self) -> None:
        for transaction in self.transactions:
            transaction.accept_graph_writes()
            transaction.verify()
        for transaction in self.transactions:
            transaction.commit()
        self.finished = True

    def abort(self) -> None:
        if self.finished:
            return
        for transaction in self.transactions:
            if transaction.status not in ("committed", "aborted"):
                transaction.abort()
        self.finished = True


class ArenaProgram:
    program_id = "test.shared-arena-program.v1"
    replay_abi_id = REPLAY_ABI
    state_path_id = ARENA_STATE_PATH_ID

    def __init__(self, resources: tuple[RankReplayResources, ...], fail_poison: str | None = None):
        layout = KVLayout((0,), 1, 1, 1)
        self.arenas = tuple(
            CacheArena(
                CacheArenaSpec(
                    f"arena:rank{resource.rank}",
                    1,
                    2,
                    (StateSlotSpec("state", torch.bfloat16, (1,)),),
                    2,
                    (resource.scratch_arena_id,),
                ),
                layout,
                "cpu",
                rank_id=resource.rank,
            )
            for resource in resources
        )
        stores = tuple(
            CacheStateStore(layout, rank_id=rank, arena=arena)
            for rank, arena in enumerate(self.arenas)
        )
        self.factory = ArenaQualificationFactory(stores)
        self.fail_poison = fail_poison
        self.plans: list[LaunchPlan] = []
        self.binding: GraphRuntimeBinding | None = None

    def validate_binding(self, binding: GraphRuntimeBinding) -> None:
        self.binding = binding

    def capture(
        self,
        binding: GraphRuntimeBinding,
        resources: tuple[RankReplayResources, ...],
    ) -> GraphProgramSlot:
        if binding is not self.binding:
            raise RuntimeError("arena program received a foreign binding")
        return ProbeSlot(resources)

    def qualification_state_factory(
        self,
        slot: GraphProgramSlot,
        plan: LaunchPlan,
        intent: ReplayIntent,
    ) -> GraphQualificationStateFactory:
        del slot, plan, intent
        return self.factory

    def fill(
        self,
        slot: GraphProgramSlot,
        plan: LaunchPlan,
        intent: ReplayIntent,
        logical_inputs: Mapping[str, Tensor],
        state: object,
        padding: GraphPaddingBuffers | None,
    ) -> GraphProgramSession:
        del slot, intent
        if not isinstance(state, ArenaQualificationRun) or padding is None:
            raise CudaGraphRejected("arena program requires qualification state and poison")
        transactions = state.begin(plan)
        try:
            bindings = tuple(transaction.graph_binding(plan) for transaction in transactions)
        except Exception:
            for transaction in transactions:
                if transaction.status not in ("committed", "aborted"):
                    transaction.abort()
            raise
        self.plans.append(plan)
        return ArenaSession(
            transactions,
            bindings,
            logical_inputs["x"],
            padding.poison_id,
            self.fail_poison,
        )


def _check_tensor(
    name: str,
    tensor: Tensor,
    dtype: torch.dtype,
    length: int,
    device: torch.device | None = None,
) -> None:
    if (
        not isinstance(tensor, Tensor)
        or tensor.shape != (length,)
        or tensor.dtype != dtype
        or tensor.layout is not torch.strided
        or (device is not None and tensor.device != device)
    ):
        raise CudaGraphRejected(f"{name} has an incompatible tensor schema")


def _physical_indices(rows: tuple[LaunchRow, ...]) -> tuple[int, ...]:
    return tuple(
        physical
        for _logical, physical in sorted(
            (row.logical_index, physical)
            for physical, row in enumerate(rows)
            if isinstance(row, ActiveLaunchRow)
        )
    )


def _assert_exact(
    test: unittest.TestCase,
    expected: Mapping[str, Tensor],
    actual: Mapping[str, Tensor],
) -> None:
    test.assertEqual(set(expected), set(actual))
    policy = ComparisonPolicy(DifferenceClass.EXACT)
    for name in expected:
        result = compare_tensors(expected[name], actual[name], policy)
        test.assertTrue(result.passed, f"{name}: {result.reason}")


def _passing_result(
    logical_m: int,
    launch_plan_id: str | None = None,
    *,
    state_path_digest: str = PROBE_STATE_PATH_DIGEST,
) -> GraphPaddingResult:
    passed = ComparisonResult(
        passed=True,
        difference_class=DifferenceClass.EXACT,
        element_count=1,
        first_mismatch_index=None,
        max_absolute_error=0.0,
        max_ulp_error=0,
        max_allowed_error=0.0,
        reason="test witness",
    )
    launch_base = launch_plan_id or f"launch:m{logical_m}"
    rows = _rows(logical_m)
    launches = tuple(
        GraphReplayLaunchIdentity(
            poison_id=poison_id,
            launch_plan_id=f"{launch_base}:{poison_id}",
            lease_id=f"lease:{launch_base}:{poison_id}",
            lease_epoch=epoch,
            execution_plan_id="performance:toy",
            source_plan_id=f"batch:m{logical_m}",
            bucket_id=_bucket().bucket_id,
            row_map_digest=row_map_digest(rows),
        )
        for epoch, poison_id in enumerate(("positive", "negative"), start=1)
    )
    return GraphPaddingResult(
        case_id=f"cuda-graph/m{logical_m}/bucket4",
        bucket_id=_bucket().bucket_id,
        logical_m=logical_m,
        row_map_digest=row_map_digest(rows),
        poison_ids=("positive", "negative"),
        replay_launches=launches,
        state_path_id=PROBE_STATE_PATH_ID,
        state_path_digest=state_path_digest,
        checks=(GraphPaddingCheck("test", passed),),
    )


def _cpu_case(logical_m: int) -> GraphPaddingCase:
    padding_m = 4 - logical_m
    state = {
        f"{kind}:rank{rank}": torch.zeros(logical_m, dtype=dtype)
        for rank in (0, 1)
        for kind, dtype in STATE_DTYPES.items()
    }
    return GraphPaddingCase(
        case_id=f"cuda-graph/m{logical_m}/bucket4",
        logical_inputs={"x": torch.zeros(logical_m, dtype=torch.bfloat16)},
        initial_committed_state=state,
        poisons=tuple(
            GraphPaddingPoison(
                poison_id=poison_id,
                padding_inputs=(
                    {"x": torch.full((padding_m,), value, dtype=torch.bfloat16)}
                    if padding_m
                    else {}
                ),
                scratch_state=(
                    {
                        name: torch.full((padding_m,), value, dtype=tensor.dtype)
                        for name, tensor in state.items()
                    }
                    if padding_m
                    else {}
                ),
                guards={"sentinel": torch.tensor([sentinel], dtype=torch.int64)},
            )
            for poison_id, value, sentinel in (
                ("positive", 7.0, 11),
                ("negative", -9.0, 13),
            )
        ),
    )


def _arena_case() -> GraphPaddingCase:
    return GraphPaddingCase(
        case_id="arena-qualification/m2/bucket4",
        logical_inputs={"x": torch.tensor([2.0, -3.0])},
        initial_committed_state={
            "revision:rank0": torch.tensor([0]),
            "revision:rank1": torch.tensor([0]),
        },
        poisons=tuple(
            GraphPaddingPoison(
                poison_id=poison_id,
                padding_inputs={"x": torch.full((2,), value)},
                scratch_state={"state": torch.full((2,), value)},
                guards={"sentinel": torch.tensor([sentinel])},
            )
            for poison_id, value, sentinel in (
                ("positive", 17.0, 101),
                ("negative", -23.0, 211),
            )
        ),
    )


def _real_runtime() -> tuple[CudaGraphRuntime, ToyGraphProgram]:
    program = ToyGraphProgram()
    runtime = CudaGraphRuntime(binding=_binding(), program=program, resource_sets=_resources())
    runtime.capture()
    return runtime, program


class _BrokenGraph:
    def replay(self) -> None:
        raise RuntimeError("injected CUDA graph launch failure")


@unittest.skipUnless(RUN_REAL, "set CUBIC_RUN_CUDA_GRAPH_TESTS=1 for real CUDA Graph tests")
class RealCudaGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("real topology-wide CUDA Graph test requires two GPUs")

    def test_multi_m_exact_lease_backpressure_full_replay_and_poison(self) -> None:
        runtime, program = _real_runtime()
        case2, case3 = _case(2), _case(3)
        witnesses = {
            logical_m: runtime.qualify(
                case=case,
                intent=_intent(logical_m),
                exact_m_oracle=_oracle,
            )
            for logical_m, case in ((2, case2), (3, case3))
        }
        self.assertEqual(runtime.in_flight_leases, 0)

        with self.assertRaises(CudaGraphRejected):
            runtime.seal(tuple(witnesses.values()))
        witnesses[4] = runtime.qualify(case=_case(4), intent=_intent(4), exact_m_oracle=_oracle)
        self.assertEqual(len(program.qualification_factories), 6)
        for factory in program.qualification_factories:
            self.assertEqual(len(factory.runs), 1)
            self.assertEqual([run.finished for run in factory.runs], ["closed"])
            self.assertTrue(
                all(run.state.revision == factory.target_revision for run in factory.runs)
            )
        epochs = tuple(
            launch.lease_epoch
            for logical_m in runtime.bucket.logical_m_values
            for launch in witnesses[logical_m].result.replay_launches
        )
        self.assertEqual(epochs, tuple(sorted(set(epochs))))
        runtime.seal(tuple(witnesses[value] for value in runtime.bucket.logical_m_values))

        with self.assertRaisesRegex(CudaGraphRejected, "transactional state"):
            runtime.replay(
                intent=_intent(2),
                logical_inputs=case2.logical_inputs,
                state=clone_tensor_map(case2.initial_committed_state, "bare state"),
            )

        durable_outputs: list[Mapping[str, Tensor]] = []
        final_state: ToyCommittedState | None = None
        final_case: GraphPaddingCase | None = None
        for logical_m in runtime.bucket.logical_m_values:
            case = _case(logical_m)
            intent = _intent(logical_m)
            state = ToyCommittedState(
                request_id=intent.batch_plan.segments[0].request_id,
                revision=0,
                tensors=case.initial_committed_state,
            )
            expected_state = clone_tensor_map(case.initial_committed_state, "expected state")
            expected_output = _oracle(case.logical_inputs, expected_state)
            output = runtime.replay(intent=intent, logical_inputs=case.logical_inputs, state=state)
            _assert_exact(self, expected_output, output)
            _assert_exact(self, expected_state, state.snapshot())
            self.assertEqual(state.revision, 1)
            with self.assertRaisesRegex(CudaGraphRejected, "transaction|revision"):
                runtime.replay(intent=intent, logical_inputs=case.logical_inputs, state=state)
            durable_outputs.append(output)
            final_state, final_case = state, case

        self.assertEqual(runtime.in_flight_leases, 0)
        assert final_state is not None and final_case is not None
        lanes = tuple(lane for slot in program.captured_slots for lane in slot.lanes)
        for lane in lanes:
            self.assertTrue(
                all(
                    value.data_ptr() != lane.logits.data_ptr()
                    for output in durable_outputs
                    for value in output.values()
                )
            )
            static_state_ptrs = {item.data_ptr() for item in lane.states.values()}
            self.assertTrue(
                all(
                    value.data_ptr() not in static_state_ptrs
                    for value in final_state.snapshot().values()
                )
            )
            self.assertGreater(int(lane.guard[0].item()), 0)
        self.assertEqual(len({lane.guard.data_ptr() for lane in lanes}), len(lanes))

        for lane in lanes:
            lane.graph = _BrokenGraph()
        poison_intent = _intent(4)
        with self.assertRaises(CudaGraphPoisoned):
            runtime.replay(
                intent=poison_intent,
                logical_inputs=final_case.logical_inputs,
                state=ToyCommittedState(
                    request_id=poison_intent.batch_plan.segments[0].request_id,
                    revision=0,
                    tensors=final_case.initial_committed_state,
                ),
            )
        with self.assertRaises(CudaGraphPoisoned):
            runtime.seal(tuple(witnesses.values()))


class CudaGraphCoordinatorTests(unittest.TestCase):
    def test_decode_dispatch_selects_exact_capacity_and_rejects_missing_bucket(self) -> None:
        bucket = _bucket()
        alternate = replace(
            bucket,
            bucket_id="decode:row4:larger-cache",
            template_id="toy:state-row:4:larger-cache",
            capacity_items=(("cache_pages", 8), ("request_slots", 4)),
        )
        execution = sealed_test_performance_plan(
            "performance:toy", _execution(bucket).bindings, (bucket, alternate)
        )
        bucket, alternate = execution.graph_buckets
        sessions = (ProbeSession(), ProbeSession())
        runtimes = tuple(
            CudaGraphRuntime(
                binding=_binding(bucket=item, execution=execution),
                program=ProbeProgram(session),
                resource_sets=_resources(slots=1),
            )
            for item, session in zip((bucket, alternate), sessions, strict=True)
        )
        for runtime in runtimes:
            runtime.capture()
            runtime._sealed = True
        dispatcher = CudaGraphDecodeDispatcher(runtimes)

        output = dispatcher.replay(
            batch_plan=_batch(2, capacity_items=alternate.capacity_items),
            rows=_rows(2),
            deadline=AcquireDeadline.after(0.0),
            logical_inputs={},
            state=object(),
        )
        self.assertEqual(tuple(output), ("out",))
        self.assertFalse(sessions[0].committed)
        self.assertTrue(sessions[1].committed)
        self.assertEqual(dispatcher.bucket_ids, (bucket.bucket_id, alternate.bucket_id))

        with self.assertRaisesRegex(CudaGraphRejected, "fallback is forbidden"):
            dispatcher.replay(
                batch_plan=_batch(
                    2,
                    capacity_items=(("cache_pages", 16), ("request_slots", 4)),
                ),
                rows=_rows(2),
                deadline=AcquireDeadline.after(0.0),
                logical_inputs={},
                state=object(),
            )

    def test_decode_dispatch_preserves_lease_backpressure(self) -> None:
        runtime = CudaGraphRuntime(
            binding=_binding(),
            program=ProbeProgram(),
            resource_sets=_resources(slots=1),
        )
        runtime.capture()
        runtime._sealed = True
        dispatcher = CudaGraphDecodeDispatcher((runtime,))
        held, _plan = runtime._begin(_intent(2), production=True)
        try:
            with self.assertRaises(CudaGraphOverload):
                dispatcher.replay(
                    batch_plan=_batch(2),
                    rows=_rows(2),
                    deadline=AcquireDeadline.after(0.0),
                    logical_inputs={},
                    state=object(),
                )
        finally:
            runtime._finish(held)

    def test_each_poison_owns_a_fresh_arena_lease_and_stale_launch_fails(self) -> None:
        resources = _resources(slots=1)[0]
        program = ArenaProgram(resources)
        runtime = CudaGraphRuntime(binding=_binding(), program=program, resource_sets=(resources,))
        runtime.capture()
        case = _arena_case()

        def oracle(
            inputs: Mapping[str, Tensor], state: MutableMapping[str, Tensor]
        ) -> Mapping[str, Tensor]:
            for value in state.values():
                value.add_(1)
            return {"out": inputs["x"].clone()}

        held, _plan = runtime._begin(_intent(2), production=False)
        try:
            with self.assertRaises(CudaGraphOverload):
                runtime.qualify(
                    case=case,
                    intent=_intent(2, timeout_s=0.0),
                    exact_m_oracle=oracle,
                )
        finally:
            runtime._finish(held)

        witness = runtime.qualify(
            case=case,
            intent=_intent(2),
            exact_m_oracle=oracle,
        )
        launches = witness.result.replay_launches
        self.assertEqual(tuple(launch.poison_id for launch in launches), ("positive", "negative"))
        self.assertLess(launches[0].lease_epoch, launches[1].lease_epoch)
        self.assertEqual(len({launch.lease_id for launch in launches}), 2)
        self.assertEqual(runtime.in_flight_leases, 0)
        self.assertEqual([run.finished for run in program.factory.runs], ["closed", "closed"])

        stale_run = program.factory.create(case.initial_committed_state)
        transactions = stale_run.begin(program.plans[0])
        try:
            with self.assertRaisesRegex(ScratchArenaLeaseError, "stale"):
                transactions[0].graph_binding(program.plans[0])
        finally:
            for transaction in transactions:
                if transaction.status not in ("committed", "aborted"):
                    transaction.abort()
            stale_run.abort()

        failed_program = ArenaProgram(resources, fail_poison="negative")
        failed = CudaGraphRuntime(
            binding=_binding(), program=failed_program, resource_sets=(resources,)
        )
        failed.capture()
        with self.assertRaises(CudaGraphPoisoned):
            failed.qualify(case=case, intent=_intent(2), exact_m_oracle=oracle)
        self.assertEqual(failed.in_flight_leases, 0)
        self.assertEqual(
            [run.finished for run in failed_program.factory.runs],
            ["closed", "aborted"],
        )

    def test_capacity_profile_is_exact_and_never_falls_back(self) -> None:
        bucket = _bucket()
        alternate = replace(
            bucket,
            bucket_id="decode:row4:larger-cache",
            template_id="toy:state-row:4:larger-cache",
            capacity_items=(("cache_pages", 8), ("request_slots", 4)),
        )
        execution = sealed_test_performance_plan(
            "performance:toy", _execution(bucket).bindings, (bucket, alternate)
        )
        bucket, alternate = execution.graph_buckets
        self.assertIs(
            execution.bucket_for(
                scope_id=bucket.scope_id,
                replay_abi_id=bucket.replay_abi_id,
                phase=bucket.phase,
                logical_m=2,
                capacity_items=alternate.capacity_items,
                topology_id=bucket.topology_id,
                partition_id=bucket.partition_id,
            ),
            alternate,
        )
        with self.assertRaisesRegex(ValueError, "fallback is forbidden"):
            execution.bucket_for(
                scope_id=bucket.scope_id,
                replay_abi_id=bucket.replay_abi_id,
                phase=bucket.phase,
                logical_m=2,
                capacity_items=(("cache_pages", 16), ("request_slots", 4)),
                topology_id=bucket.topology_id,
                partition_id=bucket.partition_id,
            )

        runtime = CudaGraphRuntime(
            binding=_binding(bucket=bucket, execution=execution),
            program=ProbeProgram(),
            resource_sets=_resources(slots=1),
        )
        runtime.capture()
        runtime._sealed = True
        wrong = ReplayIntent(
            _batch(2, capacity_items=alternate.capacity_items),
            _rows(2),
            AcquireDeadline.after(0.0),
        )
        with self.assertRaisesRegex(CudaGraphRejected, "different captured graph runtime"):
            runtime.replay(intent=wrong, logical_inputs={}, state=object())
        self.assertEqual(runtime.in_flight_leases, 0)

    def test_core_closes_structure_and_program_owns_supported_topology(self) -> None:
        unsupported = replace(_bucket(), replay_abi_id="cubic.replay.unknown.v1")
        binding = _binding(bucket=unsupported)
        with self.assertRaisesRegex(CudaGraphRejected, "replay ABI"):
            CudaGraphRuntime(
                binding=binding,
                program=ToyGraphProgram(),
                resource_sets=_resources(slots=1),
            )

        pp = ParallelTopology.single_node(
            topology_id="toy:pp2", device_indices=(0, 1), tp_size=1, pp_size=2
        )
        pp_bucket = replace(_bucket(), topology_id=pp.topology_id)
        pp_partition = replace(
            _partition(),
            topology_id=pp.topology_id,
            assignments=(
                GraphAssignment("stage0", 0, (0,)),
                GraphAssignment("stage1", 1, (1,)),
            ),
        )
        pp_binding = _binding(bucket=pp_bucket, topology=pp, partition=pp_partition)
        with self.assertRaisesRegex(CudaGraphRejected, "PP or EP"):
            CudaGraphRuntime(
                binding=pp_binding,
                program=ToyGraphProgram(),
                resource_sets=_resources(slots=1),
            )

        ep = ParallelTopology.single_node(
            topology_id="toy:ep2",
            device_indices=(0, 1),
            tp_size=2,
            pp_size=1,
            ep_size=2,
        )
        ep_bucket = replace(_bucket(), topology_id=ep.topology_id)
        ep_partition = replace(_partition(), topology_id=ep.topology_id)
        ep_binding = _binding(bucket=ep_bucket, topology=ep, partition=ep_partition)
        with self.assertRaisesRegex(CudaGraphRejected, "PP or EP"):
            CudaGraphRuntime(
                binding=ep_binding,
                program=ToyGraphProgram(),
                resource_sets=_resources(slots=1),
            )

        scheduled = replace(
            _partition(),
            collectives=(CollectiveStep("all-reduce", 0, "tp:0", "all_reduce"),),
        )
        scheduled_binding = _binding(partition=scheduled)
        with self.assertRaisesRegex(CudaGraphRejected, "communication"):
            CudaGraphRuntime(
                binding=scheduled_binding,
                program=ToyGraphProgram(),
                resource_sets=_resources(slots=1),
            )

        forged = replace(_partition(), assignments=(GraphAssignment("toy.state_row", 0, (0, 2)),))
        with self.assertRaisesRegex(CudaGraphRejected, "unknown rank"):
            _binding(partition=forged)

    def test_complete_rank_lease_deadline_and_no_caller_padding(self) -> None:
        for timeout in (float("nan"), float("inf"), -1.0):
            with self.subTest(timeout=timeout), self.assertRaises(CudaGraphRejected):
                AcquireDeadline.after(timeout)
        self.assertNotIn("padding", inspect.signature(CudaGraphRuntime.replay).parameters)
        one_rank = tuple((slot[0],) for slot in _resources(slots=1))
        with self.assertRaisesRegex(CudaGraphRejected, "every topology rank"):
            CudaGraphRuntime(binding=_binding(), program=ProbeProgram(), resource_sets=one_rank)

    def test_typed_rows_and_program_owned_state_abi_fail_closed(self) -> None:
        with self.assertRaises(CudaGraphRejected):
            ReplayIntent(
                _batch(2),
                (object(), object(), object(), object()),  # type: ignore[arg-type]
                AcquireDeadline.after(0.0),
            )
        program = ToyGraphProgram()
        program.validate_binding(_binding())
        batch = _batch(2)
        incomplete = replace(
            batch.state_transactions[0],
            writes=tuple(StateAccess("kv", rank, slots=(0, 1), pages=(0,)) for rank in (0, 1)),
        )
        with self.assertRaisesRegex(CudaGraphRejected, "complete state ABI"):
            program._validate_batch(replace(batch, state_transactions=(incomplete,)))

    def test_witness_provenance_domain_and_seal_race_are_closed(self) -> None:
        with self.assertRaises(TypeError):
            QualificationWitness()  # type: ignore[call-arg]
        result = _passing_result(2)
        with self.assertRaisesRegex(TypeError, "only be issued"):
            QualificationWitness(
                domain_fingerprint="b2s256:" + "1" * 64,
                capture_fingerprint="b2s256:" + "2" * 64,
                capture_epoch=1,
                bucket=_bucket(),
                logical_m=2,
                row_map_digest="b2s256:" + "3" * 64,
                case_digest="b2s256:" + "4" * 64,
                state_path_id=PROBE_STATE_PATH_ID,
                state_path_digest=PROBE_STATE_PATH_DIGEST,
                poison_ids=result.poison_ids,
                result=result,
                _runtime_token=object(),
            )

        runtime = CudaGraphRuntime(
            binding=_binding(),
            program=ProbeProgram(),
            resource_sets=_resources(slots=1),
        )
        runtime.capture()
        intents = {value: _intent(value) for value in runtime.bucket.logical_m_values}
        with self.assertRaises(CudaGraphRejected):
            runtime._issue_witness(
                _passing_result(2),
                intents[3],
                _cpu_case(2),
            )
        witnesses = tuple(
            runtime._issue_witness(
                _passing_result(value),
                intents[value],
                _cpu_case(value),
            )
            for value in runtime.bucket.logical_m_values
        )
        alternate_digest = stable_digest("alternate qualification factory")
        mixed_factory = list(witnesses)
        mixed_factory[1] = runtime._issue_witness(
            _passing_result(3, state_path_digest=alternate_digest),
            intents[3],
            _cpu_case(3),
        )
        with self.assertRaisesRegex(CudaGraphRejected, "different state factories"):
            runtime.seal(tuple(mixed_factory))
        entered, release = Event(), Event()

        def blocked_begin(intent: ReplayIntent, production: bool) -> tuple[ReplayLease, LaunchPlan]:
            del intent, production
            entered.set()
            self.assertTrue(release.wait(1.0))
            raise CudaGraphRejected("stop after qualification admission")

        with (
            patch.object(runtime, "_begin", side_effect=blocked_begin),
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            future = executor.submit(
                runtime.qualify,
                case=_cpu_case(2),
                intent=intents[2],
                exact_m_oracle=lambda _inputs, _state: {"out": torch.tensor([1])},
            )
            self.assertTrue(entered.wait(1.0))
            with self.assertRaisesRegex(CudaGraphRejected, "qualification.*in flight"):
                runtime.seal(witnesses)
            release.set()
            with self.assertRaises(CudaGraphRejected):
                future.result(timeout=1.0)
        with self.assertRaisesRegex(CudaGraphRejected, "logical M domain"):
            runtime.seal((witnesses[0], witnesses[0], witnesses[0]))
        runtime.seal(witnesses)
        rows = (
            ActiveLaunchRow(0),
            PaddingLaunchRow(0),
            ActiveLaunchRow(1),
            PaddingLaunchRow(1),
        )
        with self.assertRaisesRegex(CudaGraphRejected, "row map"):
            runtime.replay(
                intent=ReplayIntent(_batch(2), rows, AcquireDeadline.after(0.0)),
                logical_inputs={},
                state=object(),
            )

        foreign = CudaGraphRuntime(
            binding=_binding(),
            program=ProbeProgram(),
            resource_sets=_resources(slots=1),
        )
        foreign.capture()
        with self.assertRaisesRegex(CudaGraphRejected, "capture"):
            foreign.seal(witnesses)

    def test_concurrent_poison_prevents_commit_and_aborts_session(self) -> None:
        entered, release = Event(), Event()
        session = ProbeSession(entered=entered, release=release)
        runtime = CudaGraphRuntime(
            binding=_binding(),
            program=ProbeProgram(session),
            resource_sets=_resources(slots=1),
        )
        runtime.capture()
        runtime._sealed = True
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                runtime.replay,
                intent=_intent(2),
                logical_inputs={},
                state=object(),
            )
            self.assertTrue(entered.wait(1.0))
            runtime._poison("peer", RuntimeError("injected peer failure"))
            release.set()
            with self.assertRaises(CudaGraphPoisoned):
                future.result(timeout=1.0)
        self.assertFalse(session.committed)
        self.assertTrue(session.aborted)
        self.assertEqual(runtime.in_flight_leases, 0)

    def test_bind_capture_replay_and_abort_failures_poison_without_leaks(self) -> None:
        bind_failure = CudaGraphRuntime(
            binding=_binding(),
            program=ProbeProgram(),
            resource_sets=_resources(slots=1),
        )
        with self.assertRaises(CudaGraphPoisoned):
            bind_failure._acquire("launch:bind-failure", AcquireDeadline.after(0.0))
        self.assertEqual(bind_failure.in_flight_leases, 0)

        capture_failure = CudaGraphRuntime(
            binding=_binding(),
            program=ProbeProgram(),
            resource_sets=_resources(slots=1),
        )
        with (
            patch.object(
                capture_failure.program,
                "capture",
                side_effect=RuntimeError("injected capture failure"),
            ),
            self.assertRaises(CudaGraphPoisoned),
        ):
            capture_failure.capture()
        with self.assertRaises(CudaGraphPoisoned):
            capture_failure.qualify(case=_cpu_case(2), intent=_intent(2), exact_m_oracle=_oracle)

        session = ProbeSession(
            replay_error=RuntimeError("injected replay failure"),
            abort_error=RuntimeError("injected abort failure"),
        )
        replay_failure = CudaGraphRuntime(
            binding=_binding(),
            program=ProbeProgram(session),
            resource_sets=_resources(slots=1),
        )
        replay_failure.capture()
        replay_failure._sealed = True
        with self.assertRaises(CudaGraphPoisoned):
            replay_failure.replay(intent=_intent(2), logical_inputs={}, state=object())
        self.assertTrue(session.aborted)
        self.assertEqual(replay_failure.in_flight_leases, 0)


if __name__ == "__main__":
    unittest.main()
