from __future__ import annotations

import unittest
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import replace

import torch
from torch import Tensor

from cubic_serve.contracts.execution import (
    ActiveLaunchRow,
    BatchPhase,
    GraphBucket,
    LaunchPlan,
    PaddingLaunchRow,
    RankReplayResources,
    ReplayLease,
)
from cubic_serve.contracts.model import ScalarDType
from cubic_serve.guardrail.compare import (
    ComparisonContext,
    ComparisonPolicy,
    DifferenceClass,
    ReductionOrderProof,
    ScaleCastABI,
    ScaleDTypeProof,
    scale_abi_digest,
    stable_digest,
    tensor_digest,
)
from cubic_serve.guardrail.graph_padding import (
    CrossMArithmeticWitness,
    CrossMReplaySnapshot,
    GraphPaddingBuffers,
    GraphPaddingCase,
    GraphPaddingHarness,
    GraphPaddingPoison,
    GraphPaddingResult,
    GraphQualificationStateFactory,
    compare_cross_m_replays,
)
from cubic_serve.guardrail.oracle import (
    ReductionWitnessMaterial,
    ScaleWitnessMaterial,
    clone_tensor_map,
    reduction_material_identity,
    verify_reduction_witness,
    verify_scale_witness,
)

TEST_STATE_PATH_ID = "test.qualification.tensor-map.v1"
TEST_STATE_PATH_DIGEST = "b2s256:" + "5" * 64


class _MappingQualificationRun:
    def __init__(self, seed: Mapping[str, Tensor]) -> None:
        self.state = clone_tensor_map(seed, "test qualification state")
        self.finished: str | None = None

    @property
    def production_state(self) -> object:
        return self.state

    def committed_snapshot(self) -> Mapping[str, Tensor]:
        return clone_tensor_map(self.state, "test qualification snapshot")

    def close(self) -> None:
        self.finished = "closed"

    def abort(self) -> None:
        self.finished = "aborted"


class _MappingQualificationFactory:
    state_path_id = TEST_STATE_PATH_ID
    state_path_digest = TEST_STATE_PATH_DIGEST

    def __init__(self) -> None:
        self.runs: list[_MappingQualificationRun] = []

    def create(self, canonical_seed: Mapping[str, Tensor]) -> _MappingQualificationRun:
        run = _MappingQualificationRun(canonical_seed)
        self.runs.append(run)
        return run


def _mutable_state(state: object) -> MutableMapping[str, Tensor]:
    if not isinstance(state, MutableMapping):
        raise TypeError("test replay requires its explicit mapping state fixture")
    return state


def _launch_plan() -> LaunchPlan:
    bucket = GraphBucket(
        bucket_id="decode:4",
        template_id="graph:decode:4",
        scope_id="model.forward",
        replay_abi_id="cubic.replay.decode.v1",
        topology_id="single",
        partition_id="toy",
        phase=BatchPhase.DECODE,
        capacity_items=(("request_slots", 4),),
        logical_m_values=(1, 2, 3, 4),
        captured_m=4,
        row_map_schema_id="cubic.rows.v1",
        scratch_schema_id="cubic.scratch.v1",
        graph_safety_contract_ids=("graph-safe:model-forward:v1",),
        evidence_id="evidence:decode:4",
    )
    return LaunchPlan(
        launch_plan_id="launch:decode:m2-to-4",
        execution_plan_id="performance:decode",
        source_plan_id="batch:decode:m2",
        logical_m=2,
        bucket=bucket,
        rows=(
            PaddingLaunchRow(0),
            ActiveLaunchRow(1),
            PaddingLaunchRow(1),
            ActiveLaunchRow(0),
        ),
        replay_lease=ReplayLease(
            lease_id="lease:decode:m2-to-4",
            epoch=7,
            launch_plan_id="launch:decode:m2-to-4",
            execution_plan_id="performance:decode",
            bucket_id=bucket.bucket_id,
            template_id=bucket.template_id,
            topology_id=bucket.topology_id,
            partition_id=bucket.partition_id,
            rank_resources=(
                RankReplayResources(
                    rank=0,
                    graph_instance_id="graph-instance:decode:4:rank0",
                    stream_lane_id="decode:0",
                    buffer_set_id="buffers:decode:0",
                    workspace_arena_id="workspace:decode:0",
                    scratch_arena_id="scratch:decode:0",
                ),
            ),
        ),
    )


def _poison(
    poison_id: str,
    *,
    padding: tuple[float, float],
    scratch: tuple[float, float],
    guard: int,
) -> GraphPaddingPoison:
    return GraphPaddingPoison(
        poison_id=poison_id,
        padding_inputs={"hidden": torch.tensor(padding, dtype=torch.float32)},
        scratch_state={"padding-kv": torch.tensor(scratch, dtype=torch.float32)},
        guards={
            "logical-guard": torch.tensor([guard, guard + 1], dtype=torch.int64),
            "state-sentinel": torch.tensor([guard + 2], dtype=torch.int64),
        },
    )


def _case() -> GraphPaddingCase:
    return GraphPaddingCase(
        case_id="decode/m2/bucket4",
        logical_inputs={"x": torch.tensor([2.0, -3.0], dtype=torch.float32)},
        initial_committed_state={
            "kv:layer0": torch.tensor([10.0, 20.0], dtype=torch.float32),
            "recurrent:layer0": torch.tensor([7.0], dtype=torch.float32),
            "route:layer0": torch.tensor([0, 0], dtype=torch.int64),
            "transaction:revision": torch.tensor([4], dtype=torch.int64),
        },
        poisons=(
            _poison("positive", padding=(37.0, -41.0), scratch=(101.0, 103.0), guard=211),
            _poison("negative", padding=(-73.0, 79.0), scratch=(-107.0, -109.0), guard=223),
        ),
    )


def _oracle(
    inputs: Mapping[str, Tensor], state: MutableMapping[str, Tensor]
) -> Mapping[str, Tensor]:
    x = inputs["x"]
    state["kv:layer0"].add_(x)
    state["recurrent:layer0"].add_(x.sum())
    state["route:layer0"].copy_(torch.tensor([1, 0], dtype=torch.int64))
    state["transaction:revision"].add_(1)
    return {"logits": x * 2.0}


def _good_replay(
    launch_plan: LaunchPlan,
    inputs: Mapping[str, Tensor],
    state: object,
    buffers: GraphPaddingBuffers,
) -> Mapping[str, Tensor]:
    assert launch_plan.active_physical_indices == (1, 3)
    outputs = _oracle(inputs, _mutable_state(state))
    buffers.scratch_state["padding-kv"].add_(buffers.padding_inputs["hidden"])
    return outputs


type _Replay = Callable[
    [LaunchPlan, Mapping[str, Tensor], object, GraphPaddingBuffers],
    Mapping[str, Tensor],
]


class _Attempt:
    def __init__(
        self,
        plan: LaunchPlan,
        factory: GraphQualificationStateFactory,
        replay: _Replay,
        closed: list[LaunchPlan],
    ) -> None:
        self._plan, self._factory, self._replay, self._closed = (
            plan,
            factory,
            replay,
            closed,
        )

    @property
    def launch_plan(self) -> LaunchPlan:
        return self._plan

    @property
    def state_factory(self) -> GraphQualificationStateFactory:
        return self._factory

    def replay(
        self,
        inputs: Mapping[str, Tensor],
        state: object,
        padding: GraphPaddingBuffers,
    ) -> Mapping[str, Tensor]:
        return self._replay(self._plan, inputs, state, padding)

    def close(self) -> None:
        self._closed.append(self._plan)


class _AttemptFactory:
    def __init__(
        self,
        plan: LaunchPlan,
        factory: GraphQualificationStateFactory,
        replay: _Replay,
    ) -> None:
        self.plan, self.factory, self.replay = plan, factory, replay
        self.closed: list[LaunchPlan] = []
        self._epoch = plan.replay_lease.epoch

    def __call__(self) -> _Attempt:
        self._epoch += 1
        launch_id = f"{self.plan.launch_plan_id}:poison:{self._epoch}"
        lease = replace(
            self.plan.replay_lease,
            lease_id=f"{self.plan.replay_lease.lease_id}:{self._epoch}",
            epoch=self._epoch,
            launch_plan_id=launch_id,
        )
        plan = replace(self.plan, launch_plan_id=launch_id, replay_lease=lease)
        return _Attempt(plan, self.factory, self.replay, self.closed)


def _qualify(
    oracle: Callable[[Mapping[str, Tensor], MutableMapping[str, Tensor]], Mapping[str, Tensor]],
    replay: _Replay,
    *,
    case: GraphPaddingCase | None = None,
    plan: LaunchPlan | None = None,
    factory: _MappingQualificationFactory | None = None,
) -> tuple[GraphPaddingResult, _MappingQualificationFactory, _AttemptFactory]:
    selected_case, selected_plan = case or _case(), plan or _launch_plan()
    selected_factory = factory or _MappingQualificationFactory()
    attempts = _AttemptFactory(selected_plan, selected_factory, replay)
    result = GraphPaddingHarness(oracle).qualify(
        case=selected_case,
        bucket=selected_plan.bucket,
        rows=selected_plan.rows,
        attempt_factory=attempts,
    )
    return result, selected_factory, attempts


class GraphPaddingHarnessTests(unittest.TestCase):
    def test_cross_m_common_rows_are_exact_after_physical_remap(self) -> None:
        small = CrossMReplaySnapshot(
            "m2",
            "corpus:prefix",
            "reduce:fixed",
            (PaddingLaunchRow(0), ActiveLaunchRow(1), ActiveLaunchRow(0)),
            {
                "hidden": torch.tensor([[91.0], [20.0], [10.0]], dtype=torch.bfloat16),
                "kv": torch.tensor([[[91.0], [20.0], [10.0]]], dtype=torch.bfloat16),
            },
            {"hidden": 0, "kv": 1},
        )
        large = CrossMReplaySnapshot(
            "m3",
            "corpus:prefix",
            "reduce:fixed",
            (
                ActiveLaunchRow(2),
                ActiveLaunchRow(0),
                ActiveLaunchRow(1),
                PaddingLaunchRow(0),
            ),
            {
                "hidden": torch.tensor(
                    [[30.0], [10.0], [20.0], [-77.0]], dtype=torch.bfloat16
                ),
                "kv": torch.tensor(
                    [[[30.0], [10.0], [20.0], [-77.0]]], dtype=torch.bfloat16
                ),
            },
            {"hidden": 0, "kv": 1},
        )

        result = compare_cross_m_replays((small, large))

        self.assertTrue(result.passed, result)

    def test_cross_m_rejects_unwitnessed_bf16_row_drift(self) -> None:
        base = CrossMReplaySnapshot(
            "m1",
            "corpus:same",
            "reduce:same",
            (ActiveLaunchRow(0), PaddingLaunchRow(0)),
            {"hidden": torch.tensor([[1.0], [99.0]], dtype=torch.bfloat16)},
            {"hidden": 0},
        )
        drifted = CrossMReplaySnapshot(
            "m2",
            "corpus:same",
            "reduce:same",
            (ActiveLaunchRow(1), ActiveLaunchRow(0)),
            {"hidden": torch.tensor([[2.0], [1.0078125]], dtype=torch.bfloat16)},
            {"hidden": 0},
        )

        result = compare_cross_m_replays((base, drifted))

        self.assertFalse(result.passed)
        self.assertTrue(any(not check.result.passed for check in result.checks))

    def test_cross_m_rejects_changed_corpus_but_records_reduction_plan(self) -> None:
        rows = (ActiveLaunchRow(0), PaddingLaunchRow(0))
        base = CrossMReplaySnapshot(
            "m1",
            "corpus:a",
            "reduce:a",
            rows,
            {"hidden": torch.tensor([[1.0], [0.0]])},
            {"hidden": 0},
        )
        for field, candidate in (
            (
                "corpus",
                CrossMReplaySnapshot(
                    "m2",
                    "corpus:b",
                    "reduce:a",
                    (ActiveLaunchRow(0), ActiveLaunchRow(1)),
                    {"hidden": torch.tensor([[1.0], [2.0]])},
                    {"hidden": 0},
                ),
            ),
            (
                "reduction",
                CrossMReplaySnapshot(
                    "m2",
                    "corpus:a",
                    "reduce:b",
                    (ActiveLaunchRow(0), ActiveLaunchRow(1)),
                    {"hidden": torch.tensor([[1.0], [2.0]])},
                    {"hidden": 0},
                ),
            ),
        ):
            with self.subTest(field=field):
                result = compare_cross_m_replays((base, candidate))
                if field == "corpus":
                    self.assertFalse(result.passed)
                    self.assertTrue(any(field in check.path for check in result.checks))
                else:
                    self.assertTrue(result.passed)
                    self.assertEqual(
                        result.reduction_plans, (("m1", "reduce:a"), ("m2", "reduce:b"))
                    )

    def test_cross_m_admits_verified_reduction_and_rejects_reused_evidence(self) -> None:
        expected, actual = torch.tensor([0.0]), torch.tensor([1.0])
        operands = torch.tensor([[1e8, 1.0, -1e8]])
        indices = torch.tensor([[0, 1, 2]])
        material = ReductionWitnessMaterial(
            reference_operands=operands, reference_indices=indices,
            candidate_operands=operands[:, [0, 2, 1]],
            candidate_indices=indices[:, [0, 2, 1]],
        )
        identity = reduction_material_identity(material)
        policy = ComparisonPolicy(DifferenceClass.REDUCTION_ORDER, reduction=ReductionOrderProof(
            operands_digest=identity.operands_digest, indices_digest=identity.indices_digest,
            reduction_length=identity.reduction_length, accumulator_dtype=ScalarDType.FP32,
        ))
        context = ComparisonContext(stable_digest("sum"), stable_digest("M2/reordered"))
        evidence = CrossMArithmeticWitness(policy, context, verify_reduction_witness(
            expected, actual, policy, context, material,
        ))
        base = CrossMReplaySnapshot(
            "m1", "same", "sequential", (ActiveLaunchRow(0),),
            {"hidden": expected, "owner": torch.tensor([0])}, {"hidden": 0, "owner": 0},
        )
        other = CrossMReplaySnapshot(
            "m2", "same", "reordered", (ActiveLaunchRow(1), ActiveLaunchRow(0)),
            {"hidden": torch.tensor([2.0, 1.0]), "owner": torch.tensor([1, 0])},
            {"hidden": 0, "owner": 0},
        )
        allowances = {("m1", "m2", "hidden"): evidence}
        result = compare_cross_m_replays((base, other), arithmetic=allowances)
        self.assertTrue(result.passed, result)
        self.assertTrue(any(check.result.witness_digest for check in result.checks))
        corrupted = replace(other, tensors={**other.tensors, "hidden": torch.tensor([2.0, 2.0])})
        self.assertFalse(compare_cross_m_replays((base, corrupted), arithmetic=allowances).passed)
        wrong_owner = replace(other, tensors={**other.tensors, "owner": torch.tensor([1, 9])})
        self.assertFalse(compare_cross_m_replays((base, wrong_owner), arithmetic=allowances).passed)
        self.assertFalse(compare_cross_m_replays(
            (base, other), arithmetic={**allowances, ("m1", "m2", "owner"): evidence},
        ).passed)
        self.assertFalse(compare_cross_m_replays(
            (base, other), arithmetic={**allowances, ("m1", "m2", "absent"): evidence},
        ).passed)

    def test_cross_m_admits_original_scale_cast_witness(self) -> None:
        operands, scale = torch.tensor([2.0]), torch.tensor([0.3333])
        abi = ScaleCastABI("scale:v1", "projection.scale", ScalarDType.FP32, ScalarDType.FP32)
        policy = ComparisonPolicy(DifferenceClass.SCALE_DTYPE, scale=ScaleDTypeProof(
            tensor_digest(scale), scale_abi_digest(abi), ScalarDType.FP32,
            ScalarDType.FP16, "projection.scale",
        ))
        context = ComparisonContext(stable_digest("scale"), stable_digest("M2/fp16-scale"))
        expected = operands * scale
        actual = operands * scale.half().float()
        self.assertFalse(torch.equal(expected, actual))
        witness = verify_scale_witness(
            expected, actual, policy, context, ScaleWitnessMaterial(operands, scale, abi)
        )
        base = CrossMReplaySnapshot(
            "m1", "same", "fp32-scale", (ActiveLaunchRow(0),),
            {"hidden": expected}, {"hidden": 0},
        )
        other = CrossMReplaySnapshot(
            "m2", "same", "fp16-scale", (ActiveLaunchRow(1), ActiveLaunchRow(0)),
            {"hidden": torch.cat((torch.tensor([7.0]), actual))}, {"hidden": 0},
        )
        result = compare_cross_m_replays((base, other), arithmetic={
            ("m1", "m2", "hidden"): CrossMArithmeticWitness(policy, context, witness),
        })
        self.assertTrue(result.passed, result)

    def test_full_bucket_replay_uses_the_same_exact_guardrail(self) -> None:
        padded = _launch_plan()
        full = LaunchPlan(
            launch_plan_id=padded.launch_plan_id,
            execution_plan_id=padded.execution_plan_id,
            source_plan_id="batch:decode:m4",
            logical_m=4,
            bucket=padded.bucket,
            rows=tuple(ActiveLaunchRow(index) for index in range(4)),
            replay_lease=padded.replay_lease,
        )
        case = GraphPaddingCase(
            case_id="decode/m4/bucket4",
            logical_inputs={"x": torch.tensor([1.0, -2.0, 3.0, -4.0])},
            initial_committed_state={"kv": torch.tensor([4.0, 5.0, 6.0, 7.0])},
            poisons=tuple(
                GraphPaddingPoison(
                    poison_id=poison_id,
                    padding_inputs={},
                    scratch_state={},
                    guards={"sentinel": torch.tensor([sentinel])},
                )
                for poison_id, sentinel in (("first", 17), ("second", 29))
            ),
        )

        def exact(
            inputs: Mapping[str, Tensor], state: MutableMapping[str, Tensor]
        ) -> Mapping[str, Tensor]:
            state["kv"].add_(inputs["x"])
            return {"logits": inputs["x"] * 2}

        result, _factory, attempts = _qualify(
            exact,
            lambda _plan, inputs, state, _buffers: exact(inputs, _mutable_state(state)),
            case=case,
            plan=full,
        )

        self.assertTrue(result.passed)
        self.assertEqual(len(attempts.closed), 2)
        self.assertLess(
            result.replay_launches[0].lease_epoch,
            result.replay_launches[1].lease_epoch,
        )

    def test_only_padding_scratch_values_may_differ(self) -> None:
        observed_scratch: list[Tensor] = []

        def replay(
            launch_plan: LaunchPlan,
            inputs: Mapping[str, Tensor],
            state: object,
            buffers: GraphPaddingBuffers,
        ) -> Mapping[str, Tensor]:
            outputs = _good_replay(launch_plan, inputs, state, buffers)
            observed_scratch.append(buffers.scratch_state["padding-kv"].clone())
            return outputs

        result, factory, attempts = _qualify(_oracle, replay)

        self.assertTrue(result.passed)
        self.assertEqual(len(observed_scratch), 2)
        self.assertFalse(torch.equal(observed_scratch[0], observed_scratch[1]))
        self.assertTrue(any(check.path.startswith("poison-invariance:") for check in result.checks))
        self.assertEqual([run.finished for run in factory.runs], ["closed", "closed"])
        self.assertIsNot(factory.runs[0].state, factory.runs[1].state)
        self.assertEqual(len(attempts.closed), 2)

    def test_padding_contamination_of_logical_output_is_rejected(self) -> None:
        def replay(
            launch_plan: LaunchPlan,
            inputs: Mapping[str, Tensor],
            state: object,
            buffers: GraphPaddingBuffers,
        ) -> Mapping[str, Tensor]:
            outputs = dict(_good_replay(launch_plan, inputs, state, buffers))
            outputs["logits"].add_(buffers.padding_inputs["hidden"].sum())
            return outputs

        result, _factory, _attempts = _qualify(_oracle, replay)

        self.assertFalse(result.passed)
        self.assertTrue(
            any(
                not check.result.passed and "/oracle/output:logits" in check.path
                for check in result.checks
            )
        )
        self.assertTrue(
            any(
                not check.result.passed and check.path.startswith("poison-invariance:")
                for check in result.checks
            )
        )

    def test_padding_contamination_of_each_committed_state_class_is_rejected(self) -> None:
        for state_name in (
            "kv:layer0",
            "recurrent:layer0",
            "route:layer0",
            "transaction:revision",
        ):
            with self.subTest(state_name=state_name):

                def replay(
                    launch_plan: LaunchPlan,
                    inputs: Mapping[str, Tensor],
                    state: object,
                    buffers: GraphPaddingBuffers,
                    _state_name: str = state_name,
                ) -> Mapping[str, Tensor]:
                    outputs = _good_replay(launch_plan, inputs, state, buffers)
                    delta = int(buffers.padding_inputs["hidden"][0].item())
                    _mutable_state(state)[_state_name].add_(delta)
                    return outputs

                result, _factory, _attempts = _qualify(_oracle, replay)

                self.assertFalse(result.passed)
                self.assertTrue(
                    any(
                        not check.result.passed
                        and check.path.endswith(f"/oracle/committed:{state_name}")
                        for check in result.checks
                    )
                )

    def test_guard_or_sentinel_write_is_rejected(self) -> None:
        def replay(
            launch_plan: LaunchPlan,
            inputs: Mapping[str, Tensor],
            state: object,
            buffers: GraphPaddingBuffers,
        ) -> Mapping[str, Tensor]:
            outputs = _good_replay(launch_plan, inputs, state, buffers)
            buffers.guards["state-sentinel"].add_(1)
            return outputs

        result, _factory, _attempts = _qualify(_oracle, replay)

        self.assertFalse(result.passed)
        self.assertTrue(
            any(
                not check.result.passed and check.path.endswith("/guard:state-sentinel")
                for check in result.checks
            )
        )

    def test_replay_failure_is_a_rejection_result_without_fallback(self) -> None:
        def replay(
            launch_plan: LaunchPlan,
            inputs: Mapping[str, Tensor],
            state: object,
            buffers: GraphPaddingBuffers,
        ) -> Mapping[str, Tensor]:
            raise RuntimeError("captured graph launch failed")

        result, factory, attempts = _qualify(_oracle, replay)

        self.assertFalse(result.passed)
        reasons = tuple(check.result.reason.lower() for check in result.checks)
        self.assertTrue(any("graph replay failed" in reason for reason in reasons))
        self.assertFalse(any("fallback" in reason for reason in reasons))
        self.assertEqual([run.finished for run in factory.runs], ["aborted", "aborted"])
        self.assertEqual(len(attempts.closed), 2)


if __name__ == "__main__":
    unittest.main()


def test_text_audit_retires_cyclic_temporary_owners_before_next_bucket():
    import gc
    import weakref
    from types import SimpleNamespace
    from unittest.mock import patch

    from cubic_serve.guardrail.quantized_graph import text_cross_m as audit
    from cubic_serve.runtime.cache.quantization import KVCacheStorageSpec

    owners = []

    class TemporaryStage:
        def __init__(self, *args, captured_m, **kwargs):
            # A live predecessor would retain its CUDA operator workspaces.
            assert all(owner() is None for owner in owners)
            owners.append(weakref.ref(self))
            self.captured_m = captured_m
            self.helper = SimpleNamespace(_buffers={})

        def launch(self):
            pass

        def capture(self):
            self.launch()

        def close(self):
            pass  # Graph reset alone intentionally does not break the cycle.

    arena = SimpleNamespace(
        device=torch.device("cpu"), layout=SimpleNamespace(
            block_size_tokens=4, storage=KVCacheStorageSpec().resolve(torch.bfloat16)),
        spec=SimpleNamespace(state_slot_capacity=4, page_capacity=4),
        reset_state_rows=lambda rows: None, reset_page_rows=lambda rows: None,
    )
    base = dict(
        config=SimpleNamespace(text=SimpleNamespace(
            qsa=SimpleNamespace(compression_ratio=4), vocabulary_size=100,
            hyperconnection=SimpleNamespace(stream_count=1),
        )),
        weights=None, topology=SimpleNamespace(pp_size=1, ep_size=1),
        buffers=SimpleNamespace(page_table=torch.empty(1, 1)), arena=arena,
        operators=SimpleNamespace(config=None), recurrent_state_dtype=torch.bfloat16,
        mrope_tables=None, rank=0,
    )
    executors = {
        m: SimpleNamespace(body=SimpleNamespace(**base, captured_m=m)) for m in (1, 2, 4)
    }
    report = {"steps": [{"boundary_digests": {}} for _ in range(4)]}
    enabled = gc.isenabled()
    gc.disable()  # Collection must be explicit at the audit ownership boundary.
    try:
        with (
            patch.object(audit, "FlashNextDecodeStage", TemporaryStage),
            patch.object(audit, "audit_original", return_value=report),
            patch.object(audit, "_run", lambda *args, **kwargs: [[({}, ())] for _ in range(4)]),
            patch.object(audit, "_run_segments", lambda *args, **kwargs: None),
            patch.object(audit, "_row_failure", return_value=None),
            patch.object(audit, "_coordinate"),
            patch.object(audit.dist, "get_world_size", return_value=1),
            patch.object(audit.dist, "all_gather_object",
                         side_effect=lambda outputs, value, **kwargs: outputs.__setitem__(0, value)),
            patch.object(torch.cuda, "empty_cache"),
            patch.object(torch.cuda, "memory_allocated", return_value=0),
            # This CPU fixture exercises retirement without retaining captures.
            # Device headroom belongs to the real CUDA resource checks.
            patch("cubic_serve.runtime.quantized_qualification._draft_reference_headroom",
                  return_value=0),
        ):
            result = audit.audit_cross_m(
                executors, SimpleNamespace(max_sequence_tokens=4), {}, report)
        assert len(result["temporary_retirements"]) == 4
        assert all(owner() is None for owner in owners)
    finally:
        if enabled:
            gc.enable()
        gc.collect()
