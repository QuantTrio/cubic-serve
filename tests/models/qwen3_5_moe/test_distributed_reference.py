from __future__ import annotations

import faulthandler
import json
import os
import pickle
import tempfile
import time
import traceback
import unittest
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
import torch.nn.functional as functional
from safetensors.torch import save_file
from torch import Tensor

from cubic_serve.contracts.execution import (
    CollectiveStep,
    ExpertAssignment,
    GraphAssignment,
    ParallelTopology,
    PartitionPlan,
    RankReplayResources,
)
from cubic_serve.contracts.model import ScalarDType
from cubic_serve.executor.distributed_reference import Qwen3_5DistributedCache
from cubic_serve.executor.model_runner import (
    GDN_CONV_STATE_ID,
    GDN_RECURRENT_STATE_ID,
    Qwen3_5TP2PP2DecodeBody,
    Qwen3_5TP2PP2GraphSlot,
)
from cubic_serve.executor.mtp_runner import Qwen3_5TPMTPBody, Qwen3_5TPMTPGraphSlot
from cubic_serve.executor.parallel import initialize_eager_nccl, seal_partition
from cubic_serve.executor.runner import ReferenceExecutionContext, configure_reference_math
from cubic_serve.generation.tp_sampling import (
    ShardedSamplingParameters,
    ShardedSamplingWorkspace,
    sample_sharded_broadcast_tp,
)
from cubic_serve.guardrail.compare import (
    ComparisonContext,
    ComparisonPolicy,
    DifferenceClass,
    PartitionedReductionProof,
    ReductionOrderProof,
    stable_digest,
    tensor_digest,
)
from cubic_serve.guardrail.oracle import (
    PartitionedReductionWitnessMaterial,
    ReductionWitnessMaterial,
    WitnessVerificationError,
    partitioned_reduction_material_identity,
    reduction_material_identity,
    verify_observed_bf16_linear_sample,
    verify_partitioned_reduction_witness,
    verify_reduction_witness,
)
from cubic_serve.guardrail.quantized_graph.bounds import R, check, product
from cubic_serve.guardrail.quantized_graph.state_bounds import sigmoid_range
from cubic_serve.models.checkpoint import inspect_checkpoint
from cubic_serve.models.qwen3_5.model import _DIRECT_REFERENCE_OPERATORS, AttentionKind
from cubic_serve.models.qwen3_5_moe import (
    parse_qwen3_5_moe_config,
    parse_qwen3_5_moe_config_json,
)
from cubic_serve.models.qwen3_5_moe.distributed import (
    Qwen3_5MoeDistributedMTPReferenceModel,
    Qwen3_5MoeDistributedReferenceError,
    Qwen3_5MoeDistributedReferenceModel,
    _combine_step,
    _deduplicate_kv_replicas,
    _dispatch_step,
    _DistributedMoe,
    _row_step,
    seal_qwen3_5_moe_distributed_reference,
    seal_qwen3_5_moe_mtp_distributed_reference,
)
from cubic_serve.models.qwen3_5_moe.model import route_qwen3_5_moe
from cubic_serve.models.qwen3_5_moe.parallel import (
    MoeWeightLayout,
    load_qwen3_5_moe_rank_weights,
    plan_qwen3_5_moe_tp4_pp2_ep4,
    plan_qwen3_5_moe_weights,
)
from cubic_serve.models.qwen3_5_moe.weights import (
    _expected_headers,
    plan_qwen3_5_moe_checkpoint_weights,
    plan_qwen3_5_moe_weight_headers,
)
from cubic_serve.models.weight_schema import TensorHeader
from cubic_serve.ops.cuda.sampling import RootTopKTopPSamplingPrimitive
from cubic_serve.ops.eager.activation import sigmoid as reference_sigmoid
from cubic_serve.ops.eager.hybrid import (
    apply_partial_rotary,
    depthwise_causal_conv1d,
    gated_delta_net_recurrent,
    gated_rms_norm,
    interleaved_mrope_cos_sin,
    zero_centered_rms_norm,
)
from cubic_serve.ops.eager.linear import linear as reference_linear
from cubic_serve.ops.strategies import attention_precision
from cubic_serve.runtime.performance import (
    GREEDY_SAMPLING,
    _BatchRow,
    _CapturedGraphRank,
    _direct_bind_production_audit_rows,
)
from tests.models.qwen3_5_moe.test_config import _fixture
from tests.models.qwen3_5_moe.test_parallel import _plan, _topology


class Qwen3_5MoeDistributedContractTests(unittest.TestCase):
    def test_tp4_ep2_reference_closes_local_partials_with_tp_sum(self) -> None:
        config = parse_qwen3_5_moe_config(_fixture())
        headers = tuple(
            TensorHeader(name, item.dtype, item.shape)
            for name, item in _expected_headers(config).items()
        )
        weights = plan_qwen3_5_moe_weight_headers(config, headers)
        topology = ParallelTopology.single_node(
            topology_id="moe:tp4:ep2", device_indices=(0, 1, 2, 3),
            tp_size=4, pp_size=1, ep_size=2,
        )
        plan = plan_qwen3_5_moe_weights(config, weights, topology, include_mtp=True)
        for seal in (seal_qwen3_5_moe_distributed_reference,
                     seal_qwen3_5_moe_mtp_distributed_reference):
            steps = seal(config, plan).partition.collectives
            self.assertFalse(any(step.operation == "all_to_all" for step in steps))
            self.assertTrue(any(step.operation == "all_reduce" for step in steps))

    def test_mtp_recipe_seals_tp_ep_and_row_operations_in_eager_order(self) -> None:
        config = parse_qwen3_5_moe_config(_fixture())
        headers = tuple(
            TensorHeader(name, item.dtype, item.shape)
            for name, item in _expected_headers(config).items()
        )
        weights = plan_qwen3_5_moe_weight_headers(config, headers)
        topology = ParallelTopology.single_node(
            topology_id="qwen3.5-122b:gpu0-3:tp4:pp1:ep4:mtp-reference",
            device_indices=tuple(range(4)),
            tp_size=4,
            pp_size=1,
            ep_size=4,
        )
        plan = plan_qwen3_5_moe_weights(config, weights, topology, include_mtp=True)
        sealed = seal_qwen3_5_moe_mtp_distributed_reference(config, plan)
        tp = tuple(item for item in sealed.partition.collectives if item.group_id == "tp:0")
        ep = tuple(item for item in sealed.partition.collectives if item.group_id == "ep:0")
        self.assertEqual(len(tp), 13)
        self.assertEqual(tuple(item.sequence_no for item in tp), tuple(range(13)))
        self.assertEqual(tuple(item.operation for item in ep), ("all_to_all", "all_to_all"))
        self.assertEqual(tuple(item.sequence_no for item in ep), (0, 1))

    def test_tp4_ep1_recipe_has_no_expert_dispatch(self) -> None:
        config = parse_qwen3_5_moe_config(_fixture())
        headers = tuple(TensorHeader(name, item.dtype, item.shape)
                        for name, item in _expected_headers(config).items())
        weights = plan_qwen3_5_moe_weight_headers(config, headers)
        topology = ParallelTopology.single_node(
            topology_id="moe:tp4:ep1", device_indices=(0, 1, 2, 3),
            tp_size=4, pp_size=1, ep_size=1,
        )
        plan = plan_qwen3_5_moe_weights(config, weights, topology)
        sealed = seal_qwen3_5_moe_distributed_reference(config, plan)
        self.assertFalse(any(step.operation == "all_to_all"
                             for step in sealed.partition.collectives))
        self.assertEqual(sum(step.operation == "all_reduce"
                             for step in sealed.partition.collectives),
                         config.graph.text.layer_count)
        self.assertTrue(all(a.expert_tensor_ranks == (0, 1, 2, 3)
                            for a in plan.partition.expert_assignments))

    def test_tp4_ep1_mtp_recipe_preserves_tp_sum_without_ep_dispatch(self) -> None:
        config = parse_qwen3_5_moe_config(_fixture())
        headers = tuple(TensorHeader(name, item.dtype, item.shape)
                        for name, item in _expected_headers(config).items())
        weights = plan_qwen3_5_moe_weight_headers(config, headers)
        topology = ParallelTopology.single_node(
            topology_id="mtp:tp4:ep1", device_indices=(0, 1, 2, 3),
            tp_size=4, pp_size=1, ep_size=1,
        )
        plan = plan_qwen3_5_moe_weights(config, weights, topology, include_mtp=True)
        sealed = seal_qwen3_5_moe_mtp_distributed_reference(config, plan)
        steps = sealed.partition.collectives
        self.assertEqual(len(steps), 13)
        self.assertTrue(all(step.group_id == "tp:0" for step in steps))
        self.assertEqual(sum(step.operation == "all_reduce" for step in steps), 1)
        self.assertEqual(tuple(step.sequence_no for step in steps), tuple(range(13)))

    def test_issued_inventory_and_plans_survive_the_spawn_pickle_boundary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cubic-pickle-inventory-") as directory:
            root = Path(directory)
            root.joinpath("config.json").write_text('{"model_type":"fixture"}', encoding="utf-8")
            save_file(
                {"weight": torch.ones((2, 2), dtype=torch.bfloat16)}, root / "model.safetensors"
            )
            inventory = inspect_checkpoint(root)
            restored_inventory = pickle.loads(pickle.dumps(inventory))
        config = parse_qwen3_5_moe_config_json(
            json.dumps(_fixture(), separators=(",", ":")).encode("utf-8")
        )
        headers = tuple(
            TensorHeader(name, item.dtype, item.shape)
            for name, item in _expected_headers(config).items()
        )
        weight_plan = plan_qwen3_5_moe_weight_headers(config, headers)
        parallel_plan = plan_qwen3_5_moe_tp4_pp2_ep4(config, weight_plan, _topology())
        restored_weight, restored_parallel = pickle.loads(
            pickle.dumps((weight_plan, parallel_plan))
        )
        self.assertEqual(restored_inventory.inventory_digest, inventory.inventory_digest)
        self.assertEqual(restored_weight.semantic_plan_digest, weight_plan.semantic_plan_digest)
        self.assertEqual(restored_parallel, parallel_plan)

    def test_recipe_closes_nonexpert_gathers_true_ep_and_row_reconstruction(self) -> None:
        plan = _plan()
        sealed = seal_qwen3_5_moe_distributed_reference(parse_qwen3_5_moe_config(_fixture()), plan)
        for stage in range(2):
            tp = tuple(
                item for item in sealed.partition.collectives if item.group_id == f"tp:{stage}"
            )
            ep = tuple(
                item for item in sealed.partition.collectives if item.group_id == f"ep:{stage}"
            )
            gathers = tuple(item for item in tp if item.operation == "all_gather")
            rows = tuple(item for item in tp if item.operation == "all_reduce")
            self.assertEqual((len(gathers), len(ep), len(rows)), (289, 48, 24))
            self.assertEqual(tuple(item.sequence_no for item in gathers), tuple(range(289)))
            self.assertEqual(tuple(item.sequence_no for item in ep), tuple(range(48)))
            self.assertEqual(tuple(item.sequence_no for item in rows), tuple(range(24)))
            self.assertEqual(
                tuple(item.operation for item in ep[0:2]), ("all_to_all", "all_to_all")
            )
        gathered_names = tuple(
            item.step_id for item in sealed.partition.collectives if item.operation == "all_gather"
        )
        self.assertFalse(any("mlp.experts." in name for name in gathered_names))
        self.assertEqual(
            tuple(edge.edge_id for edge in sealed.partition.pipeline_edges),
            tuple(f"qwen3_5_moe.pp:0->1:lane:{lane}" for lane in range(4)),
        )
        self.assertIn("eager-reference-ep", sealed.partition.partition_id)
        self.assertEqual(
            Qwen3_5MoeDistributedReferenceModel._cache_collective_step(0, "convolution"),
            "qwen3_5_moe.reference.state:0:convolution",
        )

    def test_kv_replica_reconstruction_validates_pairs_before_deduplication(self) -> None:
        source = torch.arange(12, dtype=torch.bfloat16).reshape(2, 2, 3)
        gathered = torch.cat((source[:1], source[:1], source[1:], source[1:]), dim=0)
        self.assertTrue(
            torch.equal(
                _deduplicate_kv_replicas(gathered, axis=0, local_size=1, global_size=2), source
            )
        )
        bad = gathered.clone()
        bad[1, 0, 0] += 1
        with self.assertRaisesRegex(Qwen3_5MoeDistributedReferenceError, "KV replicas differ"):
            _deduplicate_kv_replicas(bad, axis=0, local_size=1, global_size=2)


def _toy_partition():
    topology = ParallelTopology.single_node(
        topology_id="qwen3_5_moe:toy:tp4-pp2-ep4",
        device_indices=tuple(range(8)),
        tp_size=4,
        pp_size=2,
        ep_size=4,
    )
    assignments = (
        GraphAssignment("layer:0", 0, (0, 1, 2, 3)),
        GraphAssignment("layer:24", 1, (4, 5, 6, 7)),
    )
    collectives: list[CollectiveStep] = []
    experts: list[ExpertAssignment] = []
    for stage, layer in ((0, 0), (1, 24)):
        collectives.extend(
            (
                CollectiveStep(_dispatch_step(layer), 0, f"ep:{stage}", "all_to_all", "toy-ep"),
                CollectiveStep(_combine_step(layer), 1, f"ep:{stage}", "all_to_all", "toy-ep"),
                CollectiveStep(_row_step(layer), 0, f"tp:{stage}", "all_reduce", "toy-row"),
            )
        )
        experts.extend(
            ExpertAssignment(f"layer:{layer}", expert, stage * 4 + expert // 2)
            for expert in range(8)
        )
    return seal_partition(
        topology,
        PartitionPlan(
            "qwen3_5_moe:toy:true-ep",
            topology.topology_id,
            assignments,
            tuple(collectives),
            tuple(experts),
        ),
    )


def _weight(shape: tuple[int, ...], offset: int) -> Tensor:
    values = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32)
    return (((values + offset).remainder(29) - 14) / 32).reshape(shape).to(torch.bfloat16)


def _toy_payload(
    stage: int, lane: int, device: torch.device
) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
    layer = stage * 24
    prefix = f"model.language_model.layers.{layer}.mlp"
    router = torch.eye(8, dtype=torch.bfloat16)
    hidden = torch.tensor(
        (
            (4, 3, -1, -1, -1, -1, -1, -1),
            (-1, -1, 4, 3, -1, -1, -1, -1),
            (-1, -1, -1, -1, 4, -1, -1, 3),
        ),
        dtype=torch.bfloat16,
    )
    gate_up = _weight((8, 8, 8), 3 + stage)
    down = _weight((8, 8, 4), 11 + stage)
    full = {
        f"{prefix}.gate.weight": router,
        f"{prefix}.experts.gate_up_proj": gate_up,
        f"{prefix}.experts.down_proj": down,
        f"{prefix}.shared_expert.gate_proj.weight": _weight((4, 8), 7 + stage),
        f"{prefix}.shared_expert.up_proj.weight": _weight((4, 8), 13 + stage),
        f"{prefix}.shared_expert.down_proj.weight": _weight((8, 4), 17 + stage),
        f"{prefix}.shared_expert_gate.weight": _weight((1, 8), 19 + stage),
    }
    local = dict(full)
    local[f"{prefix}.experts.gate_up_proj"] = gate_up[lane * 2 : (lane + 1) * 2]
    local[f"{prefix}.experts.down_proj"] = down[lane * 2 : (lane + 1) * 2]
    return (
        hidden.to(device),
        {name: value.to(device) for name, value in local.items()},
        {name: value.to(device) for name, value in full.items()},
    )


def _oracle_linear(inputs: Tensor, weight: Tensor) -> Tensor:
    rows = inputs.reshape(-1, inputs.shape[-1]).float()
    outputs = tuple(
        torch.matmul(row.unsqueeze(0), weight.float().t()).squeeze(0) for row in rows.unbind()
    )
    return torch.stack(outputs).reshape(*inputs.shape[:-1], weight.shape[0]).to(torch.bfloat16)


def _independent_oracle(
    hidden: Tensor, weights: dict[str, Tensor], prefix: str
) -> tuple[Tensor, Tensor, Tensor]:
    logits = _oracle_linear(hidden, weights[f"{prefix}.gate.weight"])
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
    route_ids = torch.argsort(probabilities, dim=-1, descending=True, stable=True)[:, :2]
    route_weights = probabilities.gather(1, route_ids)
    route_weights = (route_weights / route_weights.sum(dim=-1, keepdim=True)).to(torch.bfloat16)
    route_ids, order = torch.sort(route_ids, dim=-1)
    route_weights = route_weights.gather(1, order)
    routed = torch.zeros_like(hidden)
    for row in range(hidden.shape[0]):
        for slot in range(2):
            expert = int(route_ids[row, slot].item())
            gate, up = _oracle_linear(
                hidden[row : row + 1], weights[f"{prefix}.experts.gate_up_proj"][expert]
            ).chunk(2, -1)
            value = _oracle_linear(
                functional.silu(gate) * up, weights[f"{prefix}.experts.down_proj"][expert]
            )
            routed[row] = (
                routed[row] + (value[0] * route_weights[row, slot]).to(torch.bfloat16)
            ).to(torch.bfloat16)
    gate = functional.silu(
        _oracle_linear(hidden, weights[f"{prefix}.shared_expert.gate_proj.weight"])
    )
    up = _oracle_linear(hidden, weights[f"{prefix}.shared_expert.up_proj.weight"])
    shared = _oracle_linear(gate * up, weights[f"{prefix}.shared_expert.down_proj.weight"])
    shared_gate = torch.sigmoid(
        _oracle_linear(hidden, weights[f"{prefix}.shared_expert_gate.weight"])
    )
    return (
        (routed + (shared * shared_gate).to(torch.bfloat16)).to(torch.bfloat16),
        route_ids,
        route_weights,
    )


class Qwen3_5MoeRoutingTests(unittest.TestCase):
    def test_topk_boundary_tie_prefers_lower_expert_id_then_accumulates_by_id(self) -> None:
        hidden = torch.ones((2, 4), dtype=torch.bfloat16)
        gate = torch.zeros((8, 4), dtype=torch.bfloat16)
        with ReferenceExecutionContext():
            actual = route_qwen3_5_moe(hidden, gate, top_k=3)
        self.assertTrue(torch.equal(actual.expert_ids, torch.tensor(((0, 1, 2), (0, 1, 2)))))
        self.assertTrue(
            torch.equal(
                actual.weights,
                torch.full((2, 3), 1 / 3, dtype=torch.bfloat16),
            )
        )


def _toy_worker(rank: int, init_method: str) -> None:
    configure_reference_math()
    sealed = _toy_partition()
    stage, lane = divmod(rank, 4)
    with initialize_eager_nccl(sealed, rank=rank, init_method=init_method) as runtime:
        hidden, local, full = _toy_payload(stage, lane, runtime.device)
        layer = stage * 24
        prefix = f"model.language_model.layers.{layer}.mlp"
        config = SimpleNamespace(
            graph=SimpleNamespace(
                text=SimpleNamespace(
                    moe=SimpleNamespace(
                        expert_count=8,
                        experts_per_token=2,
                        normalize_topk_probability=True,
                    )
                )
            )
        )
        plan = SimpleNamespace(topology=sealed.topology)
        with ReferenceExecutionContext(), runtime.collective_epoch(f"toy-stage-{stage}"):
            expected, expected_ids, expected_weights = _independent_oracle(hidden, full, prefix)
            routes = route_qwen3_5_moe(hidden, local[f"{prefix}.gate.weight"], top_k=2)
            actual = _DistributedMoe(cast(Any, config), cast(Any, local), cast(Any, plan), runtime)(
                hidden,
                prefix=prefix,
                operators=SimpleNamespace(linear=_oracle_linear),
            )
        if not torch.equal(routes.expert_ids, expected_ids) or not torch.equal(
            routes.weights, expected_weights
        ):
            raise AssertionError(
                "router indices or BF16 weights differ from the independent oracle"
            )
        if not torch.equal(actual, expected):
            raise AssertionError(
                "true-EP dispatch/combine differs from the independent dense oracle"
            )


@unittest.skipUnless(
    os.environ.get("CUBIC_RUN_QWEN_MOE_EP_FIXTURE") == "1",
    "set CUBIC_RUN_QWEN_MOE_EP_FIXTURE=1 to run the eight-GPU true-EP fixture",
)
class Qwen3_5MoeDistributedFixtureTests(unittest.TestCase):
    def test_tp4_pp2_ep4_true_ownership_matches_independent_exact_oracle(self) -> None:
        if not torch.distributed.is_nccl_available() or torch.cuda.device_count() < 8:
            self.skipTest("eight NCCL CUDA devices are required")
        with tempfile.TemporaryDirectory(prefix="cubic-qwen-moe-ep-") as directory:
            init_method = (Path(directory) / "store").resolve().as_uri()
            torch.multiprocessing.spawn(_toy_worker, args=(init_method,), nprocs=8, join=True)


_REAL_CHECKPOINT = Path("/pubdata/llm_models/Qwen/Qwen3.5-122B-A10B")
_RUN_REAL_FORWARD = os.environ.get("CUBIC_RUN_REAL_QWEN_MOE_FORWARD") == "1"
_RUN_REAL_PERFORMANCE_AUDIT = os.environ.get("CUBIC_RUN_REAL_QWEN_MOE_PERFORMANCE_AUDIT") == "1"
_RUN_REAL_PERFORMANCE_EAGER_ONLY = os.environ.get("CUBIC_QWEN_MOE_AUDIT_EAGER_ONLY") == "1"
_REAL_PERFORMANCE_EP_SIZE = int(os.environ.get("CUBIC_QWEN_MOE_AUDIT_EP_SIZE", "4"))
_RUN_REAL_MTP_AUDIT = os.environ.get("CUBIC_RUN_REAL_QWEN_MOE_MTP_AUDIT") == "1"
_RUN_REAL_MTP_EAGER_ONLY = os.environ.get("CUBIC_QWEN_MOE_MTP_AUDIT_EAGER_ONLY") == "1"
_REAL_PERFORMANCE_DIAGNOSTIC_ONLY = os.environ.get("CUBIC_QWEN_MOE_AUDIT_DIAGNOSTIC_ONLY") == "1"
_REAL_TOKENS = tuple(
    int(token)
    for token in os.environ.get("CUBIC_QWEN_MOE_REFERENCE_TOKENS", "814,20139").split(",")
)
_REAL_PERFORMANCE_STATE_DTYPE = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}[os.environ.get("CUBIC_QWEN_MOE_AUDIT_STATE_DTYPE", "float32")]


class _AuditHarnessError(RuntimeError):
    """The audit could not establish evidence; this is not a numerical verdict."""


def _capture_linear_event(event: Any) -> SimpleNamespace:
    """Own stream-ordered evidence without running an oracle or a collective."""

    return SimpleNamespace(
        global_layer=event.global_layer,
        weight_name=event.weight_name,
        weight_shape=event.weight_shape,
        artifact_digest=event.artifact_digest,
        launch_specialization_digest=event.launch_specialization_digest,
        linear_input=event.linear_input.detach().clone(),
        local_weight=event.local_weight,
        local_output=event.local_output.detach().clone(),
        tp_reduced=(
            None if event.tp_reduced is None else event.tp_reduced.detach().clone()
        ),
    )


def _require_identical_event_schedules(schedules: tuple[object, ...]) -> None:
    if not schedules or any(peer != schedules[0] for peer in schedules[1:]):
        raise _AuditHarnessError(
            "rank linear event schedules differ; diagnostic collectives were not run"
        )


def _audit_failure_kind(error: BaseException) -> str:
    if isinstance(error, _AuditHarnessError):
        return "harness"
    if isinstance(error, AssertionError | WitnessVerificationError):
        return "numerical"
    return "execution"


class AuditHarnessSelfTests(unittest.TestCase):
    def test_capture_owns_mutable_operands_without_verifying_them(self) -> None:
        linear_input = torch.tensor([[1.0]], dtype=torch.bfloat16)
        local_output = torch.tensor([[2.0]], dtype=torch.bfloat16)
        reduced = torch.tensor([[3.0]], dtype=torch.bfloat16)
        event = SimpleNamespace(
            global_layer=0,
            weight_name="linear.weight",
            weight_shape=(1, 1),
            artifact_digest="artifact",
            launch_specialization_digest="launch",
            linear_input=linear_input,
            local_weight=torch.tensor([[2.0]], dtype=torch.bfloat16),
            local_output=local_output,
            tp_reduced=reduced,
        )

        snapshot = _capture_linear_event(event)
        linear_input.zero_()
        local_output.zero_()
        reduced.zero_()

        self.assertEqual(snapshot.linear_input.item(), 1.0)
        self.assertEqual(snapshot.local_output.item(), 2.0)
        self.assertEqual(snapshot.tp_reduced.item(), 3.0)

    def test_rank_schedule_mismatch_is_a_harness_failure(self) -> None:
        with self.assertRaises(_AuditHarnessError):
            _require_identical_event_schedules(((0, "a", True), (0, "b", True)))
        self.assertEqual(_audit_failure_kind(_AuditHarnessError("schedule")), "harness")
        self.assertEqual(_audit_failure_kind(AssertionError("value")), "numerical")
        self.assertEqual(_audit_failure_kind(RuntimeError("cuda")), "execution")


def _complete_reduction_witness(
    expected: Tensor,
    actual: Tensor,
    operands: Tensor,
    *,
    boundary: str,
) -> str:
    indices = torch.arange(operands.shape[-1], dtype=torch.int64, device=operands.device).expand(
        operands.shape
    )
    material = ReductionWitnessMaterial(operands, indices, operands, indices)
    identity = reduction_material_identity(material)
    policy = ComparisonPolicy(
        DifferenceClass.REDUCTION_ORDER,
        reduction=ReductionOrderProof(
            operands_digest=identity.operands_digest,
            indices_digest=identity.indices_digest,
            reduction_length=identity.reduction_length,
            accumulator_dtype=ScalarDType.FP32,
            candidate_accumulator_dtype=ScalarDType.FP32,
        ),
    )
    try:
        return verify_reduction_witness(
            expected,
            actual,
            policy,
            ComparisonContext(
                stable_digest({"kind": "qwen3_5_moe_eager_boundary", "boundary": boundary}),
                stable_digest("triton/fp32-reassociation"),
            ),
            material,
        ).witness_digest
    except ValueError as error:
        delta = (expected.float() - actual.float()).abs().reshape(-1)
        worst = int(delta.argmax().item())
        raise AssertionError(
            f"{boundary} reduction witness failed: max_delta={float(delta[worst].item())} "
            f"expected={float(expected.reshape(-1)[worst].item())} "
            f"actual={float(actual.reshape(-1)[worst].item())} "
            f"operands={operands.reshape(-1, operands.shape[-1])[worst].tolist()}"
        ) from error


def _spawn_with_timeout(
    worker: Any,
    args: tuple[Any, ...],
    *,
    nprocs: int,
    timeout: float,
    report_directory: str | None = None,
    report_pattern: str | None = None,
) -> None:
    context = cast(Any, torch.multiprocessing.spawn)(worker, args=args, nprocs=nprocs, join=False)
    deadline = time.monotonic() + timeout
    try:
        while True:
            if (
                report_directory is not None
                and report_pattern is not None
                and len(tuple(Path(report_directory).glob(report_pattern))) == nprocs
            ):
                return
            if context.join(timeout=1):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"distributed eager audit exceeded {timeout:.0f}s")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()


def _cache_digest(cache: Qwen3_5DistributedCache) -> str:
    return stable_digest(
        {
            "identity": (
                cache.sequence_length,
                cache.batch_size,
                cache.rank,
                cache.layer_start,
                cache.layer_end,
            ),
            "layers": [
                {
                    name: tensor_digest(tensor)
                    for name in ("convolution", "recurrent", "key", "value")
                    if (tensor := getattr(layer, name)) is not None
                }
                for layer in cache.layers
            ],
        }
    )


def _cache_exact_error(cold: Qwen3_5DistributedCache, split: Qwen3_5DistributedCache) -> str | None:
    cold_identity = (
        cold.sequence_length,
        cold.batch_size,
        cold.rank,
        cold.layer_start,
        cold.layer_end,
        len(cold.layers),
    )
    split_identity = (
        split.sequence_length,
        split.batch_size,
        split.rank,
        split.layer_start,
        split.layer_end,
        len(split.layers),
    )
    if cold_identity != split_identity:
        return f"cache identity differs: cold={cold_identity}, split={split_identity}"
    dtypes = {
        "convolution": torch.bfloat16,
        "recurrent": torch.float32,
        "key": torch.bfloat16,
        "value": torch.bfloat16,
    }
    for layer_index, (cold_layer, split_layer) in enumerate(
        zip(cold.layers, split.layers, strict=True), start=cold.layer_start
    ):
        for name, dtype in dtypes.items():
            left, right = getattr(cold_layer, name), getattr(split_layer, name)
            if (left is None) != (right is None):
                return f"rank {cold.rank} layer {layer_index} {name} presence differs"
            if left is None or right is None:
                continue
            if left.dtype is not dtype or right.dtype is not dtype:
                return f"rank {cold.rank} layer {layer_index} {name} dtype is not {dtype}"
            if not torch.equal(left, right):
                return f"rank {cold.rank} layer {layer_index} {name} differs elementwise"
    return None


def _real_forward_worker(
    rank: int,
    init_method: str,
    report_directory: str,
    config: Any,
    inventory: Any,
    weight_plan: Any,
    plan: Any,
) -> None:
    report_path = Path(report_directory, f"forward-rank-{rank}.json")
    started = time.monotonic()
    try:
        configure_reference_math()
        sealed = seal_qwen3_5_moe_distributed_reference(config, plan)
        with initialize_eager_nccl(
            sealed, rank=rank, init_method=init_method, timeout_seconds=600
        ) as runtime:
            torch.cuda.reset_peak_memory_stats(runtime.device)
            load_started = time.monotonic()
            weights = load_qwen3_5_moe_rank_weights(
                inventory=inventory,
                weight_plan=weight_plan,
                parallel_plan=plan,
                rank=rank,
                device=runtime.device,
            )
            load_seconds = time.monotonic() - load_started
            model = Qwen3_5MoeDistributedReferenceModel(
                config=config,
                weights=weights,
                parallel_plan=plan,
                runtime=runtime,
            )
            tokens = torch.tensor([_REAL_TOKENS], dtype=torch.int64, device=runtime.device)
            with ReferenceExecutionContext():
                cold_started = time.monotonic()
                cold = model.forward(tokens, epoch_id="real-moe:cold-2")
                cold_seconds = time.monotonic() - cold_started
                split_started = time.monotonic()
                first = model.forward(tokens[:, :1], epoch_id="real-moe:split-1")
                second = model.forward(
                    tokens[:, 1:], cache=first.cache, epoch_id="real-moe:split-2"
                )
                split_seconds = time.monotonic() - split_started
            torch.cuda.synchronize(runtime.device)

            exact_errors = [
                error
                for error in (_cache_exact_error(cold.cache, second.cache),)
                if error is not None
            ]
            cache_exact = not exact_errors
            final_stage = model.pipeline_stage == 1
            logits_exact = True
            greedy_token: int | None = None
            logits_digest: str | None = None
            if final_stage:
                if cold.logits is None or second.logits is None:
                    raise AssertionError("last PP stage did not produce logits")
                cold_last, split_last = cold.logits[:, -1], second.logits[:, -1]
                logits_exact = (
                    cold_last.dtype is torch.bfloat16
                    and split_last.dtype is torch.bfloat16
                    and torch.equal(cold_last, split_last)
                )
                if not logits_exact:
                    exact_errors.append(f"rank {rank} final logits differ elementwise")
                greedy_token = int(torch.argmax(split_last, dim=-1).item())
                logits_digest = tensor_digest(split_last)
            elif cold.logits is not None or second.logits is not None:
                raise AssertionError("non-final PP stage produced logits")

            report: dict[str, object] = {
                "rank": rank,
                "pipeline_stage": model.pipeline_stage,
                "tensor_rank": model.tensor_rank,
                "resident_bytes": weights.resident_bytes,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(runtime.device),
                "max_reconstructed_operand_bytes": max(
                    cold.max_reconstructed_operand_bytes,
                    first.max_reconstructed_operand_bytes,
                    second.max_reconstructed_operand_bytes,
                ),
                "cache_exact": cache_exact,
                "cache_digest": _cache_digest(second.cache),
                "logits_exact": logits_exact,
                "logits_digest": logits_digest,
                "greedy_token": greedy_token,
                "exact_error": "; ".join(exact_errors) or None,
                "timing_seconds": {
                    "load": load_seconds,
                    "cold_2": cold_seconds,
                    "split_1_plus_1": split_seconds,
                    "total": time.monotonic() - started,
                },
            }
            report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
            torch.distributed.barrier()
    except BaseException:
        report_path.write_text(
            json.dumps({"rank": rank, "error": traceback.format_exc()}, sort_keys=True),
            encoding="utf-8",
        )
        raise


@unittest.skipUnless(
    _RUN_REAL_FORWARD,
    "set CUBIC_RUN_REAL_QWEN_MOE_FORWARD=1 to run the real TP4 x PP2 x EP4 forward",
)
class Qwen3_5MoeRealDistributedReferenceTests(unittest.TestCase):
    def test_cold_two_matches_prefill_decode_logits_and_rank_local_state_exactly(self) -> None:
        if not (_REAL_CHECKPOINT / "model.safetensors.index.json").is_file():
            self.skipTest("the local Qwen3.5-122B-A10B checkpoint is unavailable")
        if not torch.distributed.is_nccl_available() or torch.cuda.device_count() < 8:
            self.skipTest("eight CUDA devices with NCCL are required")
        with tempfile.TemporaryDirectory(prefix="cubic-qwen-moe-forward-") as directory:
            config = parse_qwen3_5_moe_config_json((_REAL_CHECKPOINT / "config.json").read_bytes())
            inventory = inspect_checkpoint(_REAL_CHECKPOINT, verify_payloads=False)
            weight_plan = plan_qwen3_5_moe_checkpoint_weights(config, inventory)
            parallel_plan = plan_qwen3_5_moe_tp4_pp2_ep4(config, weight_plan, _topology())
            issued = pickle.loads(pickle.dumps((config, inventory, weight_plan, parallel_plan)))
            init_method = (Path(directory) / "store").resolve().as_uri()
            try:
                cast(Any, torch.multiprocessing.spawn)(
                    _real_forward_worker,
                    args=(init_method, directory, *issued),
                    nprocs=8,
                    join=True,
                )
            except Exception as error:
                diagnostics = "\n".join(
                    path.read_text(encoding="utf-8")
                    for path in sorted(Path(directory).glob("forward-rank-*.json"))
                )
                self.fail(f"real MoE worker failed: {error}\n{diagnostics}")
            reports = tuple(
                json.loads(Path(directory, f"forward-rank-{rank}.json").read_text("utf-8"))
                for rank in range(8)
            )

        failures = tuple(report for report in reports if report.get("exact_error"))
        self.assertEqual(failures, (), json.dumps(failures, sort_keys=True))
        self.assertEqual(tuple(report["rank"] for report in reports), tuple(range(8)))
        self.assertEqual(
            tuple((report["pipeline_stage"], report["tensor_rank"]) for report in reports),
            ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)),
        )
        self.assertEqual(len({report["logits_digest"] for report in reports[4:]}), 1)
        expected_token = os.environ.get("CUBIC_QWEN_MOE_REFERENCE_GREEDY_TOKEN")
        if expected_token is not None:
            self.assertEqual(
                {report["greedy_token"] for report in reports[4:]},
                {int(expected_token)},
            )


def _real_performance_audit_worker(
    rank: int,
    init_method: str,
    report_directory: str,
    config: Any,
    inventory: Any,
    weight_plan: Any,
    plan: Any,
) -> None:
    report_path = Path(report_directory, f"performance-audit-rank-{rank}.json")
    done_path = Path(report_directory, f"performance-audit-rank-{rank}.done")
    try:

        def mark(phase: str) -> None:
            report_path.write_text(
                json.dumps({"rank": rank, "phase": phase}, sort_keys=True),
                encoding="utf-8",
            )

        mark("runtime")
        torch.set_num_threads(1)
        configure_reference_math()
        sealed = seal_qwen3_5_moe_distributed_reference(config, plan)
        with initialize_eager_nccl(
            sealed, rank=rank, init_method=init_method, timeout_seconds=600
        ) as runtime:
            mark("weights")
            weights = load_qwen3_5_moe_rank_weights(
                inventory=inventory,
                weight_plan=weight_plan,
                parallel_plan=plan,
                rank=rank,
                device=runtime.device,
            )
            mark("body")
            resource = RankReplayResources(
                rank,
                f"audit-rank{rank}-graph",
                f"audit-rank{rank}-stream",
                f"audit-rank{rank}-buffers",
                f"audit-rank{rank}-workspace",
                f"audit-rank{rank}-scratch",
            )
            page_count = (len(_REAL_TOKENS) + 15) // 16
            body = Qwen3_5TP2PP2DecodeBody.build(
                config=config,
                parallel_plan=plan,
                weights=weights,
                tp_group=runtime._groups["tp:0"],
                pp_group=None,
                captured_m=1,
                table_pages=page_count,
                page_capacity=2 * page_count + 2,
                state_capacity=2,
                scratch_arena_id=resource.scratch_arena_id,
                scratch_capacity=16,
                recurrent_state_dtype=_REAL_PERFORMANCE_STATE_DTYPE,
            )
            segment_body = body.build_sibling(captured_m=16)
            body.prepare_scratch(resource)
            mark("body_ready")
            body.set_active_rows(1)
            body.arena.key.zero_()
            body.arena.value.zero_()
            body.arena.written.zero_()
            for state in body.arena.states.values():
                state.zero_()
            buffers = body.buffers
            layer_slots = torch.arange(
                buffers.conv_slots.shape[1], dtype=torch.int32, device=runtime.device
            )

            def reset_body() -> tuple[list[int], int]:
                body.arena.key.zero_()
                body.arena.value.zero_()
                body.arena.written.zero_()
                for state in body.arena.states.values():
                    state.zero_()
                body.prepare_scratch(resource)
                return [2 * block for block in range(page_count)], 0

            def bind_token(
                position: int,
                token: int,
                current_pages: list[int],
                current_state: int,
            ) -> int:
                block = position // 16
                source_page = current_pages[block]
                destination_page = 2 * block + int(source_page == 2 * block)
                current_pages[block] = destination_page
                buffers.page_table[0].copy_(
                    torch.tensor(current_pages, dtype=torch.int32, device=runtime.device)
                )
                buffers.position_ids[:, 0].fill_(position)
                buffers.page_source[0].fill_(source_page)
                buffers.page_destination[0].fill_(destination_page)
                buffers.token_offsets[0].fill_(position % 16)
                buffers.sequence_lengths[0].fill_(position + 1)
                buffers.segment_lengths[0].fill_(1)
                destination_state = 1 if position == 32 else current_state
                source_slots = current_state * layer_slots.numel() + layer_slots
                destination_slots = destination_state * layer_slots.numel() + layer_slots
                buffers.conv_slots[0, :, 0].copy_(source_slots)
                buffers.conv_slots[1, :, 0].copy_(destination_slots)
                buffers.recurrent_slots[0, :, 0].copy_(source_slots)
                buffers.recurrent_slots[1, :, 0].copy_(destination_slots)
                body.fill_token_ids(torch.tensor([token], dtype=torch.int64, device=runtime.device))
                return destination_state

            if _RUN_REAL_PERFORMANCE_EAGER_ONLY:
                tokens = torch.tensor([_REAL_TOKENS], dtype=torch.int64, device=runtime.device)
                reference = Qwen3_5MoeDistributedReferenceModel(
                    config=config,
                    weights=weights,
                    parallel_plan=plan,
                    runtime=runtime,
                    recurrent_state_dtype=_REAL_PERFORMANCE_STATE_DTYPE,
                )
                reference_layer_snapshots: dict[str, Tensor] = {}

                def observe_reference(layer: int, phase: str, value: Tensor) -> None:
                    reference_layer_snapshots[f"{layer}:{phase}"] = value.detach().clone()

                with ReferenceExecutionContext():
                    if tokens.shape[1] == 1:
                        reference_step = reference.forward(
                            tokens,
                            epoch_id="audit:eager-only:reference-final",
                            layer_observer=observe_reference,
                        )
                    else:
                        prefix = reference.forward(
                            tokens[:, :-1], epoch_id="audit:eager-only:reference-prefix"
                        )
                        reference_step = reference.forward(
                            tokens[:, -1:],
                            cache=prefix.cache,
                            epoch_id="audit:eager-only:reference-final",
                            layer_observer=observe_reference,
                        )
                torch.cuda.synchronize(runtime.device)
                torch.distributed.barrier()
                reference_layers = {
                    name: value.to("cpu", copy=True)
                    for name, value in reference_layer_snapshots.items()
                }
                mark("reference_ready")
                performance_layer_snapshots: dict[str, Tensor] = {}
                linear_witnesses: dict[str, str] = {}
                boundary_linear_witnesses: dict[str, str] = {}
                linear_outputs: dict[str, Tensor] = {}
                linear_event_snapshots: list[SimpleNamespace] = []

                def wait_for_boundary_peers(weight_name: str, phase: str) -> None:
                    event_id = stable_digest(weight_name)[7:23]
                    ready = (
                        report_path.parent
                        / f"boundary-{event_id}-{phase}-rank-{rank}.ready"
                    )
                    ready.touch()
                    deadline = time.monotonic() + 120.0
                    while len(
                        tuple(
                            report_path.parent.glob(
                                f"boundary-{event_id}-{phase}-rank-*.ready"
                            )
                        )
                    ) != 4:
                        if time.monotonic() >= deadline:
                            raise _AuditHarnessError(
                                f"ranks did not finish boundary witness for {weight_name}"
                            )
                        time.sleep(0.01)

                def validate_linear_event_schedule() -> None:
                    schedule = [
                        [event.global_layer, event.weight_name, event.tp_reduced is not None]
                        for event in linear_event_snapshots
                    ]
                    schedule_path = report_path.parent / f"linear-schedule-rank-{rank}.json"
                    pending_path = schedule_path.with_suffix(".pending")
                    pending_path.write_text(json.dumps(schedule), encoding="utf-8")
                    pending_path.replace(schedule_path)
                    deadline = time.monotonic() + 120.0
                    paths = tuple(
                        report_path.parent / f"linear-schedule-rank-{peer}.json"
                        for peer in range(4)
                    )
                    while not all(path.exists() for path in paths):
                        if time.monotonic() >= deadline:
                            raise _AuditHarnessError(
                                "not every rank published its linear event schedule"
                            )
                        time.sleep(0.01)
                    schedules = tuple(json.loads(path.read_text()) for path in paths)
                    _require_identical_event_schedules(schedules)

                def observe_linear(event: Any) -> None:
                    snapshot = _capture_linear_event(event)
                    linear_outputs[event.weight_name] = snapshot.local_output
                    linear_event_snapshots.append(snapshot)

                def verify_linear_event(event: Any) -> None:
                    if _REAL_PERFORMANCE_DIAGNOSTIC_ONLY:
                        return
                    report_path.write_text(
                        json.dumps(
                            {
                                "phase": "boundary_local_oracle",
                                "rank": rank,
                                "weight": event.weight_name,
                            }
                        )
                    )
                    with ReferenceExecutionContext():
                        local_oracle = reference_linear(event.linear_input, event.local_weight)
                    report_path.write_text(
                        json.dumps(
                            {
                                "phase": "boundary_local_compare",
                                "rank": rank,
                                "weight": event.weight_name,
                            }
                        )
                    )
                    unequal_columns = tuple(
                        int(index)
                        for index in torch.nonzero(
                            (event.local_output != local_oracle).any(dim=0),
                            as_tuple=False,
                        )
                        .flatten()
                        .tolist()
                    )
                    boundary_key = f"{event.global_layer}:{event.weight_name}:local"
                    if not unequal_columns:
                        boundary_linear_witnesses[f"{boundary_key}:exact"] = stable_digest(
                            {
                                "input": tensor_digest(event.linear_input),
                                "output": tensor_digest(event.local_output),
                                "artifact": event.artifact_digest,
                            }
                        )
                    for column in unequal_columns:
                        report_path.write_text(
                            json.dumps(
                                {
                                    "phase": "boundary_local_witness",
                                    "rank": rank,
                                    "weight": event.weight_name,
                                    "column": column,
                                }
                            )
                        )
                        witness = verify_observed_bf16_linear_sample(
                            x=event.linear_input,
                            weight=event.local_weight,
                            output=event.local_output,
                            input_rows=(0,),
                            output_columns=(column,),
                            operand_binding_digest=stable_digest(
                                {
                                    "kind": "qwen3_5_moe_boundary_local_linear",
                                    "rank": rank,
                                    "weight": event.weight_name,
                                    "input": tensor_digest(event.linear_input),
                                    "column": column,
                                }
                            ),
                            artifact_digest=event.artifact_digest,
                            launch_specialization_digest=(
                                event.launch_specialization_digest
                            ),
                            progress=lambda phase,
                            weight_name=event.weight_name,
                            column=column: report_path.write_text(
                                json.dumps(
                                    {
                                        "phase": f"boundary_local_{phase}",
                                        "rank": rank,
                                        "weight": weight_name,
                                        "column": column,
                                    }
                                )
                            ),
                        )
                        boundary_linear_witnesses[
                            f"{boundary_key}:column:{column}"
                        ] = witness.witness_digest
                    output_width = event.local_output.shape[1]
                    columns = tuple(
                        sorted({0, output_width // 3, 2 * output_width // 3, output_width - 1})
                    )
                    if event.tp_reduced is None:
                        witness = verify_observed_bf16_linear_sample(
                            x=event.linear_input,
                            weight=event.local_weight,
                            output=event.local_output,
                            input_rows=(0,),
                            output_columns=columns,
                            operand_binding_digest=stable_digest(
                                {
                                    "kind": "qwen3_5_moe_local_linear",
                                    "weight": event.weight_name,
                                    "input": tensor_digest(event.linear_input),
                                    "checkpoint": inventory.inventory_digest,
                                    "weight_shape": event.weight_shape,
                                }
                            ),
                            artifact_digest=event.artifact_digest,
                            launch_specialization_digest=event.launch_specialization_digest,
                        )
                        linear_witnesses[event.weight_name] = witness.witness_digest
                        wait_for_boundary_peers(event.weight_name, "done")
                        return
                    wait_for_boundary_peers(event.weight_name, "local")
                    local_x = event.linear_input.detach().clone()
                    local_weight = event.local_weight.index_select(
                        0,
                        torch.tensor(columns, dtype=torch.int64, device=runtime.device),
                    )
                    gathered_x = [torch.empty_like(local_x) for _ in range(4)]
                    gathered_weight = [torch.empty_like(local_weight) for _ in range(4)]
                    torch.distributed.all_gather(gathered_x, local_x, group=runtime._groups["tp:0"])
                    torch.distributed.all_gather(
                        gathered_weight, local_weight, group=runtime._groups["tp:0"]
                    )
                    logical: list[Tensor] = []
                    partitions: list[Tensor] = []
                    for peer_rank in range(4):
                        binding = next(
                            item
                            for item in plan.bindings_for_rank(peer_rank)
                            if item.tensor_name == event.weight_name
                        )
                        peer_indices = torch.cat(
                            tuple(
                                torch.arange(
                                    span.start,
                                    span.stop,
                                    dtype=torch.int64,
                                    device=runtime.device,
                                )
                                for span in binding.spans
                            )
                        )
                        logical.append(peer_indices.expand(len(columns), -1))
                        partitions.append(
                            torch.full(
                                (len(columns), peer_indices.numel()),
                                peer_rank,
                                dtype=torch.int64,
                                device=runtime.device,
                            )
                        )
                    material = PartitionedReductionWitnessMaterial(
                        left_operands=torch.cat(
                            tuple(x.expand(len(columns), -1) for x in gathered_x), dim=1
                        ).unsqueeze(0),
                        right_operands=torch.cat(tuple(gathered_weight), dim=1).unsqueeze(0),
                        logical_indices=torch.cat(tuple(logical), dim=1).unsqueeze(0),
                        partition_ids=torch.cat(tuple(partitions), dim=1).unsqueeze(0),
                    )
                    identity = partitioned_reduction_material_identity(material)
                    expected = torch.mm(
                        torch.cat(tuple(gathered_x), dim=1).float(),
                        torch.cat(tuple(gathered_weight), dim=1).float().t(),
                    ).to(torch.bfloat16)
                    actual = event.tp_reduced[:, columns]
                    context = ComparisonContext(
                        stable_digest(
                            {
                                "kind": "qwen3_5_moe_attention_row",
                                "weight": event.weight_name,
                                "columns": columns,
                                "binding": tensor_digest(local_x),
                            }
                        ),
                        stable_digest(
                            {
                                "artifact": event.artifact_digest,
                                "launch": event.launch_specialization_digest,
                            }
                        ),
                    )
                    witness = verify_partitioned_reduction_witness(
                        expected,
                        actual,
                        ComparisonPolicy(
                            DifferenceClass.PARTITIONED_REDUCTION,
                            partitioned_reduction=PartitionedReductionProof(
                                operands_digest=identity.operands_digest,
                                indices_digest=identity.indices_digest,
                                partitions_digest=identity.partitions_digest,
                                reduction_length=identity.reduction_length,
                                partition_count=4,
                            ),
                        ),
                        context,
                        material,
                    )
                    linear_witnesses[event.weight_name] = witness.witness_digest
                    wait_for_boundary_peers(event.weight_name, "done")

                current_pages, current_state = reset_body()
                for position, token in enumerate(_REAL_TOKENS):
                    current_state = bind_token(position, token, current_pages, current_state)
                    body.launch_qualification(
                        observe_linear if position + 1 == len(_REAL_TOKENS) else None,
                        layer_observer=(
                            lambda layer, phase, value: performance_layer_snapshots.__setitem__(
                                f"{layer}:{phase}", value.detach().clone()
                            )
                        )
                        if position + 1 == len(_REAL_TOKENS)
                        else None,
                    )
                    torch.cuda.synchronize(runtime.device)
                    torch.distributed.barrier()
                try:
                    validate_linear_event_schedule()
                    for event in linear_event_snapshots:
                        verify_linear_event(event)
                except (AssertionError, WitnessVerificationError):
                    raise
                except BaseException as error:
                    raise _AuditHarnessError(
                        "deferred linear evidence verification failed"
                    ) from error
                torch.cuda.synchronize(runtime.device)
                torch.distributed.barrier()
                performance_layers = {
                    name: value.to("cpu", copy=True)
                    for name, value in performance_layer_snapshots.items()
                }
                mark("candidate_ready")
                shadow_layers: dict[str, Tensor] = {}
                shadow_cache: dict[int, Any] = {}
                direct_state: dict[str, Tensor] = {}
                direct_state_witnesses: dict[str, Any] = {}
                diagnostic_failures: list[str] = []
                direct_boundary_snapshots: list[SimpleNamespace] = []
                if len(_REAL_TOKENS) == 1:
                    text = config.graph.text
                    rotary_dim = int(
                        text.full_attention.head_dim * text.mrope.partial_rotary_factor
                    )
                    position_ids = torch.zeros((3, 1, 1), dtype=torch.int64, device=runtime.device)
                    with ReferenceExecutionContext():
                        cosine, sine = interleaved_mrope_cos_sin(
                            position_ids,
                            rotary_dim=rotary_dim,
                            mrope_section=text.mrope.section,
                            rope_theta=text.mrope.rope_theta,
                            output_dtype=torch.bfloat16,
                        )
                    empty = reference.empty_cache(1)
                    reference._weights.open()
                    reference._moe.observer = lambda layer, phase, value: shadow_layers.__setitem__(
                        f"{layer}:{phase}", value.detach().clone()
                    )
                    try:
                        shadow_epoch = (
                            nullcontext()
                            if _REAL_PERFORMANCE_DIAGNOSTIC_ONLY
                            else runtime.collective_epoch("audit:eager-only:shadow-layers")
                        )
                        with ReferenceExecutionContext(), shadow_epoch:
                            reference._canonical.embed_input_ids(tokens)
                            for layer in range(text.layer_count):
                                prefix = f"model.language_model.layers.{layer}"
                                candidate_input = (
                                    performance_layers[f"{layer}:layer_input"]
                                    .to(runtime.device)
                                    .reshape(1, 1, text.hidden_size)
                                )
                                candidate_norm = (
                                    performance_layers[f"{layer}:input_norm"]
                                    .to(runtime.device)
                                    .reshape_as(candidate_input)
                                )
                                candidate_attention = (
                                    performance_layers[f"{layer}:attention"]
                                    .to(runtime.device)
                                    .reshape_as(candidate_input)
                                )
                                candidate_mlp_input = (
                                    performance_layers[f"{layer}:mlp_input"]
                                    .to(runtime.device)
                                    .reshape_as(candidate_input)
                                )
                                candidate_attention_mixed = (
                                    performance_layers[f"{layer}:attention_mixed"]
                                    .to(runtime.device)
                                    .reshape_as(candidate_input)
                                )
                                candidate_mlp_output = (
                                    performance_layers[f"{layer}:mlp_output"]
                                    .to(runtime.device)
                                    .reshape_as(candidate_input)
                                )
                                kind = text.layer_schedule[layer]
                                if (
                                    _REAL_PERFORMANCE_DIAGNOSTIC_ONLY
                                    and kind is not AttentionKind.GATED_DELTA_NET
                                ):
                                    continue
                                full_previous = (
                                    None
                                    if _REAL_PERFORMANCE_DIAGNOSTIC_ONLY
                                    else reference._restore_layer_cache(layer, empty.layers[layer])
                                )
                                shadow_layers[f"{layer}:layer_input"] = candidate_input.clone()
                                shadow_layers[f"{layer}:input_norm"] = zero_centered_rms_norm(
                                    candidate_input,
                                    reference._weights[f"{prefix}.input_layernorm.weight"],
                                    epsilon=text.rms_norm_epsilon,
                                    output_dtype=torch.bfloat16,
                                ).clone()
                                if kind is AttentionKind.GATED_DELTA_NET:
                                    if not _REAL_PERFORMANCE_DIAGNOSTIC_ONLY:
                                        assert full_previous is not None
                                        mixed, next_cache = reference._canonical._gated_delta(
                                            candidate_norm,
                                            prefix=prefix,
                                            previous=full_previous,
                                            operators=_DIRECT_REFERENCE_OPERATORS,
                                        )
                                    local_gdn = body._body.model.config.graph.text.gated_delta_net
                                    local_weights = body._body.model.weights
                                    local_op = f"{prefix}.linear_attn"
                                    local_state_layer = body._body._gdn_local[layer]
                                    projected = linear_outputs[
                                        f"{local_op}.in_proj_qkv.weight"
                                    ].unsqueeze(1)
                                    zero_conv = torch.zeros_like(
                                        body.arena.states[GDN_CONV_STATE_ID][0, local_state_layer]
                                    ).unsqueeze(0)
                                    convolved, next_conv = depthwise_causal_conv1d(
                                        projected,
                                        local_weights[f"{local_op}.conv1d.weight"].squeeze(1),
                                        zero_conv,
                                        activation="silu",
                                        output_dtype=torch.bfloat16,
                                    )
                                    query_width = local_gdn.key_heads * local_gdn.key_head_dim
                                    value_width = local_gdn.value_heads * local_gdn.value_head_dim
                                    query, key, value = torch.split(
                                        convolved,
                                        (query_width, query_width, value_width),
                                        dim=-1,
                                    )
                                    normalized_projected = (
                                        performance_layers[f"{layer}:gdn_normalized_qk"]
                                        .to(runtime.device)
                                        .reshape_as(convolved)
                                    )
                                    candidate_convolved = (
                                        performance_layers[f"{layer}:gdn_convolved"]
                                        .to(runtime.device)
                                        .reshape_as(convolved)
                                    )
                                    candidate_query, candidate_key, _ = torch.split(
                                        normalized_projected,
                                        (query_width, query_width, value_width),
                                        dim=-1,
                                    )
                                    _, _, candidate_value = torch.split(
                                        candidate_convolved,
                                        (query_width, query_width, value_width),
                                        dim=-1,
                                    )
                                    query = query.view(
                                        1,
                                        1,
                                        local_gdn.key_heads,
                                        local_gdn.key_head_dim,
                                    )
                                    key = key.view_as(query)
                                    value = value.view(
                                        1,
                                        1,
                                        local_gdn.value_heads,
                                        local_gdn.value_head_dim,
                                    )
                                    candidate_value = candidate_value.view_as(value)
                                    zero_recurrent = torch.zeros_like(
                                        body.arena.states[GDN_RECURRENT_STATE_ID][
                                            0, local_state_layer
                                        ]
                                    ).unsqueeze(0)
                                    core, next_recurrent = gated_delta_net_recurrent(
                                        query,
                                        key,
                                        value,
                                        linear_outputs[f"{local_op}.in_proj_a.weight"].unsqueeze(1),
                                        linear_outputs[f"{local_op}.in_proj_b.weight"].unsqueeze(1),
                                        local_weights[f"{local_op}.A_log"],
                                        local_weights[f"{local_op}.dt_bias"],
                                        zero_recurrent,
                                        l2_epsilon=1e-6,
                                        output_dtype=torch.bfloat16,
                                    )
                                    repeats = local_gdn.value_heads // local_gdn.key_heads
                                    candidate_query_f32 = (
                                        candidate_query[:, 0]
                                        .reshape(
                                            1,
                                            local_gdn.key_heads,
                                            local_gdn.key_head_dim,
                                        )
                                        .float()
                                        .repeat_interleave(repeats, dim=1)
                                    )
                                    candidate_key_f32 = (
                                        candidate_key[:, 0]
                                        .reshape_as(
                                            candidate_query[:, 0].reshape(
                                                1,
                                                local_gdn.key_heads,
                                                local_gdn.key_head_dim,
                                            )
                                        )
                                        .float()
                                        .repeat_interleave(repeats, dim=1)
                                    )
                                    reference_beta_input = linear_outputs[
                                        f"{local_op}.in_proj_b.weight"
                                    ]
                                    reference_beta = (
                                        reference_sigmoid(reference_beta_input,
                                                          storage_dtype=torch.bfloat16)
                                        .to(torch.bfloat16)
                                        .float()
                                    )
                                    candidate_beta_input = (
                                        performance_layers[f"{layer}:gdn_beta_input"]
                                        .to(runtime.device)
                                        .reshape_as(reference_beta_input)
                                    )
                                    candidate_beta = (
                                        reference_sigmoid(candidate_beta_input,
                                                          storage_dtype=torch.bfloat16)
                                        .to(torch.bfloat16)
                                        .float()
                                    )
                                    actual_recurrent = body.arena.states[GDN_RECURRENT_STATE_ID][
                                        current_state, local_state_layer
                                    ].unsqueeze(0)
                                    native = attention_precision() == "auto"
                                    # Witness the declared staged rule on the actual
                                    # candidate operands. Native recurrence rounds
                                    # the residual/beta product before its exact
                                    # low-operand outer product; wide recurrence does
                                    # not have that low intermediate boundary.
                                    delta = (
                                        candidate_value[:, 0].to(torch.bfloat16)
                                        * candidate_beta[:, :, None].to(torch.bfloat16)
                                        if native else candidate_value[:, 0].float()
                                        * candidate_beta[:, :, None]
                                    )
                                    boundary_recurrent = (
                                        delta.float().unsqueeze(-1)
                                        * candidate_key_f32[:, :, None, :]
                                    ).to(_REAL_PERFORMANCE_STATE_DTYPE)
                                    state_interval = None
                                    if native:
                                        beta_source = candidate_beta_input.detach().cpu().double()
                                        beta_lo, beta_hi = sigmoid_range(
                                            beta_source, beta_source, dtype=torch.bfloat16
                                        )
                                        stored_value = candidate_value[:, 0].detach().cpu().double()
                                        stored_key = candidate_key_f32.detach().cpu().double()
                                        lower, upper = product(
                                            stored_value, stored_value,
                                            beta_lo[:, :, None], beta_hi[:, :, None],
                                        )
                                        lower, upper = product(
                                            R(lower, torch.bfloat16).unsqueeze(-1),
                                            R(upper, torch.bfloat16).unsqueeze(-1),
                                            stored_key[:, :, None, :], stored_key[:, :, None, :],
                                        )
                                        state_interval = (
                                            R(lower, _REAL_PERFORMANCE_STATE_DTYPE),
                                            R(upper, _REAL_PERFORMANCE_STATE_DTYPE),
                                        )
                                    output_state = (
                                        actual_recurrent.to(torch.bfloat16).float()
                                        if native else actual_recurrent.float()
                                    )
                                    scaled_query = (
                                        candidate_query_f32.to(torch.bfloat16)
                                        * torch.tensor(local_gdn.key_head_dim**-0.5,
                                                       dtype=torch.bfloat16,
                                                       device=runtime.device)
                                        if native else candidate_query_f32
                                        * local_gdn.key_head_dim**-0.5
                                    ).float()
                                    boundary_core_terms = (
                                        output_state * scaled_query[:, :, None, :]
                                    )
                                    boundary_core = boundary_core_terms.sum(-1).to(torch.bfloat16)
                                    actual_core = (
                                        performance_layers[f"{layer}:gdn_core"]
                                        .to(runtime.device)
                                        .reshape_as(core[:, -1])
                                    )
                                    direct_boundary_snapshots.append(
                                        SimpleNamespace(
                                            layer=layer,
                                            candidate_beta=candidate_beta.detach().clone(),
                                            reference_beta=reference_beta.detach().clone(),
                                            candidate_value=candidate_value.detach().clone(),
                                            candidate_key=candidate_key_f32.detach().clone(),
                                            boundary_recurrent=boundary_recurrent.detach().clone(),
                                            state_interval=state_interval,
                                            actual_recurrent=actual_recurrent.detach().clone(),
                                            boundary_core=boundary_core.detach().clone(),
                                            actual_core=actual_core.detach().clone(),
                                            boundary_core_terms=boundary_core_terms.detach().clone(),
                                        )
                                    )
                                    normalized = gated_rms_norm(
                                        core,
                                        linear_outputs[f"{local_op}.in_proj_z.weight"].view_as(
                                            core
                                        ),
                                        local_weights[f"{local_op}.norm.weight"],
                                        epsilon=text.rms_norm_epsilon,
                                        output_dtype=torch.bfloat16,
                                    )
                                    direct_state[f"{layer}:convolved"] = convolved.clone()
                                    direct_state[f"{layer}:core"] = core[:, -1].clone()
                                    direct_state[f"{layer}:normalized"] = normalized[:, -1].clone()
                                    direct_state[f"{layer}:conv"] = next_conv.clone()
                                    direct_state[f"{layer}:recurrent"] = next_recurrent.clone()
                                else:
                                    assert full_previous is not None
                                    mixed, next_cache = reference._canonical._full_attention(
                                        candidate_norm,
                                        prefix=prefix,
                                        previous=full_previous,
                                        cosine=cosine,
                                        sine=sine,
                                        previous_length=0,
                                        operators=_DIRECT_REFERENCE_OPERATORS,
                                    )
                                if _REAL_PERFORMANCE_DIAGNOSTIC_ONLY:
                                    continue
                                shadow_layers[f"{layer}:attention_mixed"] = mixed.clone()
                                shadow_layers[f"{layer}:attention"] = (
                                    candidate_input + candidate_attention_mixed
                                ).clone()
                                if not _REAL_PERFORMANCE_DIAGNOSTIC_ONLY:
                                    shadow_layers[f"{layer}:mlp_input"] = zero_centered_rms_norm(
                                        candidate_attention,
                                        reference._weights[
                                            f"{prefix}.post_attention_layernorm.weight"
                                        ],
                                        epsilon=text.rms_norm_epsilon,
                                        output_dtype=torch.bfloat16,
                                    ).clone()
                                    mlp_output = reference._canonical._mlp(
                                        candidate_mlp_input,
                                        prefix=f"{prefix}.mlp",
                                        operators=_DIRECT_REFERENCE_OPERATORS,
                                    )
                                    shadow_layers[f"{layer}:mlp_output"] = mlp_output.clone()
                                    shadow_layers[f"{layer}:mlp"] = (
                                        candidate_attention + candidate_mlp_output
                                    ).clone()
                                shadow_cache[layer] = reference._slice_layer_cache(
                                    layer, next_cache
                                )
                            if not _REAL_PERFORMANCE_DIAGNOSTIC_ONLY:
                                final_hidden = (
                                    performance_layers[f"{text.layer_count - 1}:mlp"]
                                    .to(runtime.device)
                                    .reshape(1, 1, text.hidden_size)
                                )
                                reference._canonical.project_normalized_logits(
                                    reference._canonical.normalize_hidden(final_hidden)
                                )
                    finally:
                        reference._moe.observer = None
                        reference._weights.close()
                    torch.cuda.synchronize(runtime.device)
                    torch.distributed.barrier()
                    for snapshot in direct_boundary_snapshots:
                        layer = snapshot.layer
                        if _REAL_PERFORMANCE_DIAGNOSTIC_ONLY and not torch.equal(
                            snapshot.candidate_beta, snapshot.reference_beta
                        ):
                            diagnostic_failures.append(
                                f"layer:{layer}:observed beta input differs from "
                                "the packed-linear boundary"
                            )
                        if not torch.equal(
                            snapshot.boundary_recurrent, snapshot.actual_recurrent
                        ):
                            if snapshot.state_interval is not None:
                                proof = check(snapshot.actual_recurrent.detach().cpu(),
                                              *snapshot.state_interval)
                                if proof["outside_enclosure"] == 0:
                                    corrupted = snapshot.actual_recurrent.detach().cpu().clone()
                                    upper = snapshot.state_interval[1].flatten()[0]
                                    corrupted.flatten()[0] = upper.abs() * 2 + 1
                                    bad = check(corrupted, *snapshot.state_interval)
                                    if bad["outside_enclosure"] == 0:
                                        raise AssertionError(
                                            "state beta interval accepts corruption"
                                        )
                                    direct_state_witnesses[f"{layer}:state_beta_interval"] = proof
                                    # The output witness below consumes this witnessed
                                    # actual rounded state, not a substituted oracle state.
                                else:
                                    raise AssertionError(
                                        f"layer:{layer}:state beta interval {proof}"
                                    )
                            delta = (
                                snapshot.boundary_recurrent.float()
                                - snapshot.actual_recurrent.float()
                            ).abs()
                            coordinates = torch.nonzero(delta)
                            heads = torch.unique(coordinates[:, 1]).tolist()
                            values = coordinates[:, 2]
                            worst = torch.nonzero(delta == delta.max(), as_tuple=False)[0].tolist()
                            _, head, value_index, key_index = worst
                            expected_value = float(
                                snapshot.boundary_recurrent[
                                    0, head, value_index, key_index
                                ]
                            )
                            actual_value = float(
                                snapshot.actual_recurrent[0, head, value_index, key_index]
                            )
                            message = (
                                f"layer:{layer}:recurrent differs after actual candidate "
                                f"Q/K/V/beta boundaries: count="
                                f"{int(torch.count_nonzero(delta).item())} "
                                f"max_delta={float(delta.max().item())} heads={heads} "
                                f"value_range=({int(values.min().item())},"
                                f"{int(values.max().item())}) worst={worst} "
                                f"v={float(snapshot.candidate_value[0, 0, head, value_index])} "
                                f"beta={float(snapshot.candidate_beta[0, head])} "
                                f"k={float(snapshot.candidate_key[0, head, key_index])} "
                                f"expected={expected_value} actual={actual_value}"
                            )
                            if snapshot.state_interval is not None:
                                pass
                            elif _REAL_PERFORMANCE_DIAGNOSTIC_ONLY:
                                diagnostic_failures.append(message)
                            else:
                                raise AssertionError(message)
                        if not torch.equal(snapshot.boundary_core, snapshot.actual_core):
                            direct_state_witnesses[f"{layer}:core"] = _complete_reduction_witness(
                                snapshot.boundary_core,
                                snapshot.actual_core,
                                snapshot.boundary_core_terms,
                                boundary=f"layer:{layer}:core",
                            )
                    shadow_layers = {
                        name: value.to("cpu", copy=True) for name, value in shadow_layers.items()
                    }
                    direct_state = {
                        name: value.to("cpu", copy=True) for name, value in direct_state.items()
                    }
                if reference_step.logits is None or body.logits is None:
                    raise AssertionError("eager-only audit produced no final logits")
                gathered = [torch.empty_like(body.logits) for _ in range(4)]
                torch.distributed.all_gather(gathered, body.logits, group=runtime._groups["tp:0"])
                performance_logits = torch.cat(gathered, dim=-1)[
                    ..., : config.graph.text.vocabulary_size
                ]
                cache_comparisons: dict[str, dict[str, float | int | bool]] = {}
                shadow_cache_comparisons: dict[str, dict[str, float | int | bool]] = {}
                gdn_local = 0
                full_local = 0
                for layer, expected_cache in enumerate(reference_step.cache.layers):
                    expected_shadow_cache = shadow_cache.get(layer)
                    if expected_cache.recurrent is not None:
                        expected_states = (
                            ("conv", expected_cache.convolution),
                            ("recurrent", expected_cache.recurrent),
                        )
                        actual_states = (
                            (
                                "conv",
                                body.arena.states[GDN_CONV_STATE_ID][
                                    current_state, gdn_local
                                ].unsqueeze(0),
                            ),
                            (
                                "recurrent",
                                body.arena.states[GDN_RECURRENT_STATE_ID][
                                    current_state, gdn_local
                                ].unsqueeze(0),
                            ),
                        )
                        for (name, expected_state), (_, actual_state) in zip(
                            expected_states, actual_states, strict=True
                        ):
                            assert expected_state is not None
                            actual_state = actual_state.reshape_as(expected_state)
                            delta = (actual_state.float() - expected_state.float()).abs()
                            cache_comparisons[f"{layer}:{name}"] = {
                                "exact": torch.equal(actual_state, expected_state),
                                "unequal": int(
                                    torch.count_nonzero(actual_state != expected_state).item()
                                ),
                                "max_abs": float(delta.max().item()),
                                "mean_abs": float(delta.mean().item()),
                            }
                            if expected_shadow_cache is not None:
                                shadow_state = (
                                    expected_shadow_cache.convolution
                                    if name == "conv"
                                    else expected_shadow_cache.recurrent
                                )
                                assert shadow_state is not None
                                shadow_state = shadow_state.reshape_as(actual_state)
                                shadow_delta = (actual_state.float() - shadow_state.float()).abs()
                                shadow_cache_comparisons[f"{layer}:{name}"] = {
                                    "exact": torch.equal(actual_state, shadow_state),
                                    "unequal": int(
                                        torch.count_nonzero(actual_state != shadow_state).item()
                                    ),
                                    "max_abs": float(shadow_delta.max().item()),
                                    "mean_abs": float(shadow_delta.mean().item()),
                                }
                        gdn_local += 1
                        continue
                    assert expected_cache.key is not None and expected_cache.value is not None
                    for name, arena_tensor, expected_state in (
                        ("key", body.arena.key, expected_cache.key),
                        ("value", body.arena.value, expected_cache.value),
                    ):
                        actual_state = (
                            torch.cat(
                                tuple(arena_tensor[page, full_local] for page in current_pages),
                                dim=0,
                            )[: len(_REAL_TOKENS)]
                            .permute(1, 0, 2)
                            .unsqueeze(0)
                        )
                        actual_state = actual_state.reshape_as(expected_state)
                        delta = (actual_state.float() - expected_state.float()).abs()
                        cache_comparisons[f"{layer}:{name}"] = {
                            "exact": torch.equal(actual_state, expected_state),
                            "unequal": int(
                                torch.count_nonzero(actual_state != expected_state).item()
                            ),
                            "max_abs": float(delta.max().item()),
                            "mean_abs": float(delta.mean().item()),
                        }
                        if expected_shadow_cache is not None:
                            shadow_state = getattr(expected_shadow_cache, name)
                            assert shadow_state is not None
                            shadow_state = shadow_state.reshape_as(actual_state)
                            shadow_delta = (actual_state.float() - shadow_state.float()).abs()
                            shadow_cache_comparisons[f"{layer}:{name}"] = {
                                "exact": torch.equal(actual_state, shadow_state),
                                "unequal": int(
                                    torch.count_nonzero(actual_state != shadow_state).item()
                                ),
                                "max_abs": float(shadow_delta.max().item()),
                                "mean_abs": float(shadow_delta.mean().item()),
                            }
                    full_local += 1
                comparisons = {}
                for name, expected in reference_layers.items():
                    actual = performance_layers[name].reshape_as(expected)
                    difference = (actual.float() - expected.float()).abs()
                    comparisons[name] = {
                        "exact": torch.equal(actual, expected),
                        "unequal": int(torch.count_nonzero(actual != expected).item()),
                        "max_abs": float(difference.max().item()),
                        "mean_abs": float(difference.mean().item()),
                    }
                    if name.endswith(":route") and not torch.equal(actual, expected):
                        layer = name.split(":", 1)[0]
                        reference_router = reference_layers[f"{layer}:router_logits"]
                        performance_router = performance_layers[
                            f"{layer}:router_logits"
                        ].reshape_as(reference_router)
                        expert_ids = tuple(
                            sorted(
                                set(expected.flatten().tolist()) | set(actual.flatten().tolist())
                            )
                        )
                        comparisons[name]["reference_ids"] = expected.flatten().tolist()
                        comparisons[name]["performance_ids"] = actual.flatten().tolist()
                        comparisons[name]["performance_top_ids"] = (
                            performance_layers[f"{layer}:router_top_ids"].flatten().tolist()
                        )
                        canonical_performance = torch.argsort(
                            performance_router.float(),
                            dim=-1,
                            descending=True,
                            stable=True,
                        )[:, : expected.shape[-1]]
                        comparisons[name]["performance_canonical_ids"] = (
                            torch.sort(canonical_performance, dim=-1).values.flatten().tolist()
                        )
                        comparisons[name]["boundary"] = {
                            str(expert): {
                                "reference": float(reference_router[0, expert].item()),
                                "performance": float(performance_router[0, expert].item()),
                            }
                            for expert in expert_ids
                        }
                shadow_comparisons: dict[str, dict[str, float | int | bool]] = {}
                for name, expected in shadow_layers.items():
                    actual = performance_layers[name].reshape_as(expected)
                    difference = (actual.float() - expected.float()).abs()
                    shadow_comparisons[name] = {
                        "exact": torch.equal(actual, expected),
                        "unequal": int(torch.count_nonzero(actual != expected).item()),
                        "max_abs": float(difference.max().item()),
                        "mean_abs": float(difference.mean().item()),
                    }
                direct_state_comparisons: dict[str, dict[str, float | int | bool]] = {}
                if direct_state:
                    gdn_local = 0
                    for layer, kind in enumerate(config.graph.text.layer_schedule):
                        if kind is not AttentionKind.GATED_DELTA_NET:
                            continue
                        actual_values = {
                            "convolved": performance_layers[f"{layer}:gdn_convolved"],
                            "core": performance_layers[f"{layer}:gdn_core"],
                            "normalized": performance_layers[f"{layer}:gdn_normalized"],
                            "conv": body.arena.states[GDN_CONV_STATE_ID][
                                current_state, gdn_local
                            ].unsqueeze(0),
                            "recurrent": body.arena.states[GDN_RECURRENT_STATE_ID][
                                current_state, gdn_local
                            ].unsqueeze(0),
                        }
                        for phase, actual in actual_values.items():
                            expected = direct_state[f"{layer}:{phase}"]
                            actual = actual.cpu().reshape_as(expected)
                            difference = (actual.float() - expected.float()).abs()
                            direct_state_comparisons[f"{layer}:{phase}"] = {
                                "exact": torch.equal(actual, expected),
                                "unequal": int(torch.count_nonzero(actual != expected).item()),
                                "max_abs": float(difference.max().item()),
                                "mean_abs": float(difference.mean().item()),
                            }
                        gdn_local += 1
                shadow_routes: dict[str, dict[str, object]] = {}
                router_artifact = body._linear_moe_router.project_artifact()
                with ReferenceExecutionContext():
                    for layer in range(config.graph.text.layer_count):
                        layer_key = str(layer)
                        router_input = performance_layers[f"{layer_key}:mlp_input"].to(
                            runtime.device
                        )
                        router_weight = weights[
                            f"model.language_model.layers.{layer}.mlp.gate.weight"
                        ]
                        router_output = performance_layers[f"{layer_key}:router_logits"].to(
                            runtime.device
                        )
                        shadow_logits = reference_linear(
                            router_input,
                            router_weight,
                        ).cpu()
                        top_ids = torch.argsort(
                            shadow_logits.float(),
                            dim=-1,
                            descending=True,
                            stable=True,
                        )[:, : config.graph.text.moe.experts_per_token]
                        probabilities = torch.softmax(shadow_logits, dim=-1, dtype=torch.float32)
                        top_weights = probabilities.gather(1, top_ids)
                        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
                        top_weights = top_weights.to(torch.bfloat16)
                        shadow_ids, order = torch.sort(top_ids, dim=-1)
                        shadow_weights = top_weights.gather(1, order).cpu()
                        actual_ids = performance_layers[f"{layer_key}:route"].reshape_as(shadow_ids)
                        actual_weights = performance_layers[
                            f"{layer_key}:route_weights"
                        ].reshape_as(shadow_weights)
                        witness = verify_observed_bf16_linear_sample(
                            x=router_input,
                            weight=router_weight,
                            output=router_output,
                            input_rows=(0,),
                            output_columns=tuple(range(router_weight.shape[0])),
                            operand_binding_digest=stable_digest(
                                {
                                    "kind": "qwen3_5_moe_router",
                                    "layer": layer,
                                    "input": tensor_digest(router_input),
                                    "weight": tensor_digest(router_weight),
                                }
                            ),
                            artifact_digest=router_artifact.artifact_digest,
                            launch_specialization_digest=(
                                router_artifact.launch_specialization_digest
                            ),
                        )
                        shadow_routes[layer_key] = {
                            "exact": torch.equal(actual_ids, shadow_ids)
                            and torch.equal(actual_weights, shadow_weights),
                            "ids_exact": torch.equal(actual_ids, shadow_ids),
                            "weights_exact": torch.equal(actual_weights, shadow_weights),
                            "weights_max_abs": float(
                                (actual_weights.float() - shadow_weights.float()).abs().max().item()
                            ),
                            "shadow_ids": shadow_ids.flatten().tolist(),
                            "performance_ids": actual_ids.flatten().tolist(),
                            "complete_k_witness": witness.witness_digest,
                        }
                divergent_routes = tuple(
                    int(name.split(":", 1)[0])
                    for name, result in comparisons.items()
                    if name.endswith(":route") and not result["exact"]
                )
                # A first divergent router localizes a cause; it does not qualify
                # projections and TP reductions in the remaining layers.
                causal_layer_end = config.graph.text.layer_count - 1
                causal_linear_witnesses = {
                    key: witness
                    for key, witness in boundary_linear_witnesses.items()
                    if key.split(":", 1)[0].isdigit()
                    and int(key.split(":", 1)[0]) <= causal_layer_end
                }
                torch.set_num_threads(1)
                faulthandler.dump_traceback_later(30.0, repeat=True)
                try:
                    causal_events = tuple(
                        event
                        for event in linear_event_snapshots
                        if event.global_layer is not None
                        and event.global_layer <= causal_layer_end
                    )
                    for event_index, event in enumerate(causal_events):
                        report_path.write_text(
                            json.dumps(
                                {
                                    "phase": "causal_linear",
                                    "rank": rank,
                                    "event": event_index,
                                    "events": len(causal_events),
                                    "weight": event.weight_name,
                                }
                            )
                        )
                        if event.tp_reduced is None:
                            continue
                        gathered_x = [torch.empty_like(event.linear_input) for _ in range(4)]
                        gathered_weight = [torch.empty_like(event.local_weight) for _ in range(4)]
                        torch.distributed.all_gather(
                            gathered_x,
                            event.linear_input,
                            group=runtime._groups["tp:0"],
                        )
                        torch.distributed.all_gather(
                            gathered_weight,
                            event.local_weight,
                            group=runtime._groups["tp:0"],
                        )
                        report_path.write_text(
                            json.dumps(
                                {
                                    "phase": "causal_tp_collective",
                                    "rank": rank,
                                    "event": event_index,
                                    "events": len(causal_events),
                                    "weight": event.weight_name,
                                }
                            )
                        )
                        gathered_tp_output = [
                            torch.empty_like(event.tp_reduced) for _ in range(4)
                        ]
                        torch.distributed.all_gather(
                            gathered_tp_output,
                            event.tp_reduced,
                            group=runtime._groups["tp:0"],
                        )
                        torch.cuda.synchronize(runtime.device)
                        if any(
                            not torch.equal(gathered_tp_output[0], peer_output)
                            for peer_output in gathered_tp_output[1:]
                        ):
                            raise AssertionError(
                                f"TP reduction differs across ranks for {event.weight_name}"
                            )
                        shared_tp_witnesses: list[dict[str, str]] = []
                        shared_path = report_path.parent / f"causal-{event_index}-tp.json"
                        if rank != 0:
                            deadline = time.monotonic() + 120.0
                            while not shared_path.exists():
                                if time.monotonic() >= deadline:
                                    raise TimeoutError(
                                        f"rank 0 did not publish TP witnesses for "
                                        f"{event.weight_name}"
                                    )
                                time.sleep(0.01)
                            received = json.loads(shared_path.read_text())
                            if not isinstance(received, list):
                                raise AssertionError("rank 0 did not publish TP witnesses")
                            for item in received:
                                if not isinstance(item, dict):
                                    raise AssertionError("invalid shared TP witness payload")
                                causal_linear_witnesses.update(item)
                            continue
                        full_x = torch.cat(tuple(gathered_x), dim=1)
                        full_weight = torch.cat(tuple(gathered_weight), dim=1)
                        with ReferenceExecutionContext():
                            tp_oracle = reference_linear(full_x, full_weight)
                        tp_unequal_columns = tuple(
                            int(index)
                            for index in torch.nonzero(
                                (event.tp_reduced != tp_oracle).any(dim=0),
                                as_tuple=False,
                            )
                            .flatten()
                            .tolist()
                        )
                        if not tp_unequal_columns:
                            shared_tp_witnesses.append(
                                {
                                    f"{event.weight_name}:tp:exact": stable_digest(
                                        {
                                            "input": tensor_digest(full_x),
                                            "weight": tensor_digest(full_weight),
                                            "output": tensor_digest(event.tp_reduced),
                                            "artifact": event.artifact_digest,
                                        }
                                    )
                                }
                            )
                        peer_indices: list[Tensor] = []
                        for peer_rank in range(4):
                            binding = next(
                                item
                                for item in plan.bindings_for_rank(peer_rank)
                                if item.tensor_name == event.weight_name
                            )
                            peer_indices.append(
                                torch.cat(
                                    tuple(
                                        torch.arange(
                                            span.start,
                                            span.stop,
                                            dtype=torch.int64,
                                            device=runtime.device,
                                        )
                                        for span in binding.spans
                                    )
                                )
                            )
                        for offset in range(0, len(tp_unequal_columns), 2048):
                            columns = tp_unequal_columns[offset : offset + 2048]
                            report_path.write_text(
                                json.dumps(
                                    {
                                        "phase": "causal_tp_witness",
                                        "rank": rank,
                                        "event": event_index,
                                        "events": len(causal_events),
                                        "weight": event.weight_name,
                                        "columns": [offset, offset + len(columns)],
                                        "unequal_columns": len(tp_unequal_columns),
                                    }
                                )
                            )
                            column_index = torch.tensor(
                                columns, dtype=torch.int64, device=runtime.device
                            )
                            logical = tuple(
                                indices.expand(len(columns), -1) for indices in peer_indices
                            )
                            partitions = tuple(
                                torch.full(
                                    (len(columns), indices.numel()),
                                    peer_rank,
                                    dtype=torch.int64,
                                    device=runtime.device,
                                )
                                for peer_rank, indices in enumerate(peer_indices)
                            )
                            material = PartitionedReductionWitnessMaterial(
                                left_operands=torch.cat(
                                    tuple(x.expand(len(columns), -1) for x in gathered_x),
                                    dim=1,
                                ).unsqueeze(0),
                                right_operands=torch.cat(
                                    tuple(
                                        weight.index_select(0, column_index)
                                        for weight in gathered_weight
                                    ),
                                    dim=1,
                                ).unsqueeze(0),
                                logical_indices=torch.cat(logical, dim=1).unsqueeze(0),
                                partition_ids=torch.cat(partitions, dim=1).unsqueeze(0),
                            )
                            identity = partitioned_reduction_material_identity(material)
                            witness = verify_partitioned_reduction_witness(
                                tp_oracle[:, columns],
                                event.tp_reduced[:, columns],
                                ComparisonPolicy(
                                    DifferenceClass.PARTITIONED_REDUCTION,
                                    partitioned_reduction=PartitionedReductionProof(
                                        operands_digest=identity.operands_digest,
                                        indices_digest=identity.indices_digest,
                                        partitions_digest=identity.partitions_digest,
                                        reduction_length=identity.reduction_length,
                                        partition_count=4,
                                    ),
                                ),
                                ComparisonContext(
                                    stable_digest(
                                        {
                                            "kind": "qwen3_5_moe_causal_tp_linear",
                                            "weight": event.weight_name,
                                            "columns": columns,
                                        }
                                    ),
                                    stable_digest(
                                        {
                                            "artifact": event.artifact_digest,
                                            "launch": event.launch_specialization_digest,
                                        }
                                    ),
                                ),
                                material,
                            )
                            shared_tp_witnesses.append(
                                {
                                    f"{event.weight_name}:tp:{offset}": witness.witness_digest
                                }
                            )
                        for item in shared_tp_witnesses:
                            causal_linear_witnesses.update(item)
                        pending_path = shared_path.with_suffix(".pending")
                        pending_path.write_text(json.dumps(shared_tp_witnesses))
                        pending_path.replace(shared_path)
                    missing_causal_receipts: list[str] = []
                    for event in causal_events:
                        local_prefix = (
                            f"{event.global_layer}:{event.weight_name}:local:"
                        )
                        if not any(
                            key.startswith(local_prefix)
                            for key in causal_linear_witnesses
                        ):
                            missing_causal_receipts.append(f"local:{event.weight_name}")
                        if event.tp_reduced is not None and not any(
                            key.startswith(f"{event.weight_name}:tp:")
                            for key in causal_linear_witnesses
                        ):
                            missing_causal_receipts.append(f"tp:{event.weight_name}")
                    if missing_causal_receipts:
                        raise AssertionError(
                            "causal linear closure is incomplete: "
                            + ", ".join(missing_causal_receipts)
                        )
                finally:
                    faulthandler.cancel_dump_traceback_later()
                    torch.set_num_threads(1)
                report_path.write_text(
                    json.dumps(
                        {
                            "rank": rank,
                            "mode": "independent-eager",
                            "recurrent_state_dtype": str(_REAL_PERFORMANCE_STATE_DTYPE),
                            "layers": comparisons,
                            "cache": cache_comparisons,
                            "shadow_cache": shadow_cache_comparisons,
                            "direct_state": direct_state_comparisons,
                            "direct_state_witnesses": direct_state_witnesses,
                            **(
                                {"error": "; ".join(diagnostic_failures)}
                                if diagnostic_failures
                                else {}
                            ),
                            "shadow_routes": shadow_routes,
                            "shadow_layers": shadow_comparisons,
                            "linear_witnesses": linear_witnesses,
                            "causal_layer_end": causal_layer_end,
                            "first_divergent_route_layer": (
                                min(divergent_routes) if divergent_routes else None
                            ),
                            "causal_event_count": len(causal_events),
                            "causal_complete": True,
                            "causal_linear_witnesses": causal_linear_witnesses,
                            "reference_token": int(
                                torch.argmax(reference_step.logits, dim=-1).item()
                            ),
                            "performance_token": int(
                                torch.argmax(performance_logits, dim=-1).item()
                            ),
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                done_path.touch()
                torch.distributed.barrier()
                return

            current_pages, current_state = reset_body()
            current_state = bind_token(0, _REAL_TOKENS[0], current_pages, current_state)
            mark("warm")
            torch.distributed.barrier()
            body.launch()
            torch.cuda.synchronize(runtime.device)
            torch.distributed.barrier()
            mark("capture")
            torch.cuda.current_stream(runtime.device).synchronize()
            graph = Qwen3_5TP2PP2GraphSlot.capture(body, resource)
            mark("graph-replay")
            current_pages, current_state = reset_body()
            for position, token in enumerate(_REAL_TOKENS):
                current_state = bind_token(position, token, current_pages, current_state)
                torch.cuda.current_stream(runtime.device).synchronize()
                graph.replay()
                graph.synchronize()
                torch.distributed.barrier()
                mark(f"graph-replay:{position + 1}")
            assert body.logits is not None
            graph_local_logits = body.logits.detach().to("cpu", copy=True)
            graph_logits = [torch.empty_like(body.logits) for _ in range(4)]
            torch.distributed.all_gather(graph_logits, body.logits, group=runtime._groups["tp:0"])
            graph_full_logits = torch.cat(graph_logits, dim=-1)[
                ..., : config.graph.text.vocabulary_size
            ]
            graph_token = int(torch.argmax(graph_full_logits, dim=-1).item())

            current_pages, current_state = reset_body()
            production = _CapturedGraphRank(
                body,
                resource,
                graph,
                runtime,
                SimpleNamespace(
                    config=config,
                    max_sequence_tokens=len(_REAL_TOKENS),
                    parallel_plan=plan,
                ),
            )
            production_token = -1
            pages: list[int] = []
            for position, token in enumerate(_REAL_TOKENS):
                page_index = position // 16
                new_pages: tuple[int, ...] = ()
                if page_index == len(pages):
                    pages.append(2 * page_index)
                    new_pages = (pages[-1],)
                source_page = pages[page_index]
                destination_page = 2 * page_index + int(source_page == 2 * page_index)
                replay_pages = list(pages)
                replay_pages[page_index] = destination_page
                row = _BatchRow(
                    request_id="audit:production-binder",
                    token_id=token,
                    position=position,
                    state_source_slot=0,
                    state_destination_slot=0,
                    page_source=source_page,
                    page_destination=destination_page,
                    page_table=tuple(replay_pages),
                    reset_state=position == 0,
                    new_pages=new_pages,
                    segment_length=1,
                    sampling=GREEDY_SAMPLING,
                )
                production.replay_batch((row,))
                sampled = production.sample_tokens((row,))
                if sampled:
                    production_token = sampled[0]
                pages[page_index] = destination_page
                graph.synchronize()
                torch.distributed.barrier()

            current_pages, current_state = reset_body()

            tokens = torch.tensor([_REAL_TOKENS], dtype=torch.int64, device=runtime.device)
            reference = Qwen3_5MoeDistributedReferenceModel(
                config=config,
                weights=weights,
                parallel_plan=plan,
                runtime=runtime,
                recurrent_state_dtype=_REAL_PERFORMANCE_STATE_DTYPE,
            )
            reference_layers: dict[str, Tensor] = {}
            reference_segment_layers: dict[str, Tensor] = {}
            with ReferenceExecutionContext():
                prefix = reference.forward(tokens[:, :-1], epoch_id="audit:reference-prefix")
                reference_step = reference.forward(
                    tokens[:, -1:],
                    cache=prefix.cache,
                    epoch_id="audit:reference-final",
                    layer_observer=lambda layer, phase, value: reference_layers.__setitem__(
                        f"{layer}:{phase}", value.detach().to("cpu", copy=True)
                    ),
                )
                segment_first = reference.forward(
                    tokens[:, :16], epoch_id="audit:reference-segment-prefix-0"
                )
                segment_prefix = reference.forward(
                    tokens[:, 16:32],
                    cache=segment_first.cache,
                    epoch_id="audit:reference-segment-prefix-1",
                )
                reference_segment = reference.forward(
                    tokens[:, 32:],
                    cache=segment_prefix.cache,
                    epoch_id="audit:reference-segment-final",
                    layer_observer=lambda layer, phase, value: reference_segment_layers.__setitem__(
                        f"{layer}:{phase}", value.detach().to("cpu", copy=True)
                    ),
                )
            torch.distributed.barrier()

            performance_layers: dict[str, Tensor] = {}
            for position, token in enumerate(_REAL_TOKENS):
                current_state = bind_token(position, token, current_pages, current_state)
                body.launch_qualification(
                    layer_observer=(
                        (
                            lambda layer, phase, value: performance_layers.__setitem__(
                                f"{layer}:{phase}", value.detach().to("cpu", copy=True)
                            )
                        )
                        if position + 1 == len(_REAL_TOKENS)
                        else None
                    )
                )
                torch.cuda.synchronize(runtime.device)
                torch.distributed.barrier()

            segment_body.arena.key.zero_()
            segment_body.arena.value.zero_()
            segment_body.arena.written.zero_()
            for state in segment_body.arena.states.values():
                state.zero_()
            segment_body.prepare_scratch(resource)
            segment_performance_layers: dict[str, Tensor] = {}
            pages = [0, 2, 4]
            for start, stop in ((0, 16), (16, 32), (32, len(_REAL_TOKENS))):
                page_index = start // 16
                destination = pages[page_index] + 1
                replay_pages = list(pages)
                replay_pages[page_index] = destination
                rows = tuple(
                    _BatchRow(
                        request_id="audit:segment-prefill",
                        token_id=_REAL_TOKENS[position],
                        position=position,
                        state_source_slot=0,
                        state_destination_slot=0,
                        page_source=pages[page_index] if position == start else destination,
                        page_destination=destination,
                        page_table=tuple(replay_pages),
                        reset_state=start == 0 and position == start,
                        new_pages=(destination,) if position == start else (),
                        segment_length=stop - start if position == start else 0,
                        sampling=GREEDY_SAMPLING,
                    )
                    for position in range(start, stop)
                )
                _direct_bind_production_audit_rows(segment_body, rows)
                segment_body.set_active_rows(len(rows))
                active_rows = stop - start
                segment_body.launch_qualification(
                    layer_observer=(
                        lambda layer, phase, value, active_rows=active_rows: (
                            segment_performance_layers.__setitem__(
                                f"{layer}:{phase}",
                                value[:active_rows].detach().to("cpu", copy=True),
                            )
                        )
                    )
                    if stop == len(_REAL_TOKENS)
                    else None
                )
                torch.cuda.synchronize(runtime.device)
                torch.distributed.barrier()
                pages[page_index] = destination

            comparisons: dict[str, dict[str, float | int | bool]] = {}
            for name, expected in reference_layers.items():
                actual = performance_layers[name].reshape_as(expected)
                difference = (actual.float() - expected.float()).abs()
                comparisons[name] = {
                    "exact": torch.equal(actual, expected),
                    "unequal": int(torch.count_nonzero(actual != expected).item()),
                    "max_abs": float(difference.max().item()),
                    "mean_abs": float(difference.mean().item()),
                }
            segment_comparisons: dict[str, dict[str, float | int | bool]] = {}
            for name, expected in reference_segment_layers.items():
                actual = segment_performance_layers[name].reshape_as(expected)
                difference = (actual.float() - expected.float()).abs()
                segment_comparisons[name] = {
                    "exact": torch.equal(actual, expected),
                    "unequal": int(torch.count_nonzero(actual != expected).item()),
                    "max_abs": float(difference.max().item()),
                    "mean_abs": float(difference.mean().item()),
                }
            logits_comparison: dict[str, float | int | bool] | None = None
            token_comparison: dict[str, object] | None = None
            if reference_step.logits is not None and body.logits is not None:
                span = body.vocab_span
                assert span is not None
                expected = reference_step.logits[..., span.start : span.stop].cpu()
                actual = body.logits.detach().cpu().reshape_as(expected)
                difference = (actual.float() - expected.float()).abs()
                logits_comparison = {
                    "exact": torch.equal(actual, expected),
                    "unequal": int(torch.count_nonzero(actual != expected).item()),
                    "max_abs": float(difference.max().item()),
                    "mean_abs": float(difference.mean().item()),
                }
                gathered = [torch.empty_like(body.logits) for _ in range(4)]
                torch.distributed.all_gather(gathered, body.logits, group=runtime._groups["tp:0"])
                performance_logits = torch.cat(gathered, dim=-1)[
                    ..., : config.graph.text.vocabulary_size
                ]
                reference_logits = reference_step.logits
                token_comparison = {
                    "reference": int(torch.argmax(reference_logits, dim=-1).item()),
                    "performance": int(torch.argmax(performance_logits, dim=-1).item()),
                    "graph": graph_token,
                    "production_binder": production_token,
                    "reference_top5": (
                        torch.topk(reference_logits.flatten(), 5).indices.cpu().tolist()
                    ),
                    "performance_top5": (
                        torch.topk(performance_logits.flatten(), 5).indices.cpu().tolist()
                    ),
                    "graph_top5": torch.topk(graph_full_logits.flatten(), 5).indices.cpu().tolist(),
                }
                graph_difference = (graph_local_logits.float() - expected.float()).abs()
                logits_comparison.update(
                    {
                        "graph_exact": torch.equal(graph_local_logits, expected),
                        "graph_unequal": int(
                            torch.count_nonzero(graph_local_logits != expected).item()
                        ),
                        "graph_max_abs": float(graph_difference.max().item()),
                        "graph_mean_abs": float(graph_difference.mean().item()),
                    }
                )
                sample_tokens = torch.empty(1, dtype=torch.int64, device=runtime.device)
                temperatures = torch.zeros(1, dtype=torch.float32, device=runtime.device)
                top_ks = torch.ones(1, dtype=torch.int64, device=runtime.device)
                top_ps = torch.ones(1, dtype=torch.float32, device=runtime.device)
                seeds = torch.ones(1, dtype=torch.int64, device=runtime.device)
                ordinals = torch.ones(1, dtype=torch.int64, device=runtime.device)
                workspace = ShardedSamplingWorkspace(
                    torch.empty((1, 20), dtype=torch.bfloat16, device=runtime.device),
                    torch.empty((1, 20), dtype=torch.int64, device=runtime.device),
                    torch.empty((4, 1, 20), dtype=torch.bfloat16, device=runtime.device),
                    torch.empty((4, 1, 20), dtype=torch.int64, device=runtime.device),
                )
                valid = torch.empty(1, dtype=torch.bool, device=runtime.device)
                root_sampler = (
                    RootTopKTopPSamplingPrimitive(
                        workspace.gathered_scores,
                        workspace.gathered_token_ids,
                        temperatures,
                        top_ps,
                        top_ks,
                        seeds,
                        ordinals,
                        sample_tokens,
                        valid,
                    )
                    if rank == 0
                    else None
                )
                sampled = sample_sharded_broadcast_tp(
                    local_logits=body.logits,
                    local_vocab_span=body.vocab_span,
                    final_ranks=(0, 1, 2, 3),
                    final_vocab_spans=tuple(
                        next(
                            item.spans[0]
                            for item in plan.bindings_for_rank(final_rank)
                            if item.tensor_name == "lm_head.weight"
                        )
                        for final_rank in range(4)
                    ),
                    true_vocab_size=config.graph.text.vocabulary_size,
                    active_rows=1,
                    token_ids=sample_tokens,
                    parameters=ShardedSamplingParameters(
                        temperatures, top_ks, top_ps, 20, seeds, ordinals
                    ),
                    workspace=workspace,
                    final_tp_group=runtime._groups["tp:0"],
                    world_group=torch.distributed.group.WORLD,
                    device=runtime.device,
                    root_sampler=root_sampler,
                    root_valid=valid if rank == 0 else None,
                )
                token_comparison["public_sampler"] = int(sampled[0].item())

            segment_token_comparison: dict[str, object] | None = None
            if reference_segment.logits is not None and segment_body.logits is not None:
                local_segment_logits = segment_body.logits[2:3]
                gathered_segment = [torch.empty_like(local_segment_logits) for _ in range(4)]
                torch.distributed.all_gather(
                    gathered_segment, local_segment_logits, group=runtime._groups["tp:0"]
                )
                full_segment_logits = torch.cat(gathered_segment, dim=-1)[
                    ..., : config.graph.text.vocabulary_size
                ]
                reference_segment_logits = reference_segment.logits[:, -1:]
                segment_token_comparison = {
                    "reference": int(torch.argmax(reference_segment_logits, dim=-1).item()),
                    "performance_segment": int(torch.argmax(full_segment_logits, dim=-1).item()),
                    "reference_top5": torch.topk(reference_segment_logits.flatten(), 5)
                    .indices.cpu()
                    .tolist(),
                    "performance_top5": torch.topk(full_segment_logits.flatten(), 5)
                    .indices.cpu()
                    .tolist(),
                }

            report_path.write_text(
                json.dumps(
                    {
                        "rank": rank,
                        "recurrent_state_dtype": str(_REAL_PERFORMANCE_STATE_DTYPE),
                        "layers": comparisons,
                        "segment_layers": segment_comparisons,
                        "logits": logits_comparison,
                        "tokens": token_comparison,
                    "segment_tokens": segment_token_comparison,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            done_path.touch()
            torch.distributed.barrier()
    except BaseException as error:
        report_path.write_text(
            json.dumps(
                {
                    "rank": rank,
                    "failure_kind": _audit_failure_kind(error),
                    "error": traceback.format_exc(),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        done_path.touch()
        raise


@unittest.skipUnless(
    _RUN_REAL_PERFORMANCE_AUDIT,
    "set CUBIC_RUN_REAL_QWEN_MOE_PERFORMANCE_AUDIT=1 for eager/performance layer audit",
)
class Qwen3_5MoeRealPerformanceAuditTests(unittest.TestCase):
    def test_performance_body_against_independent_eager_layers(self) -> None:
        if not (_REAL_CHECKPOINT / "model.safetensors.index.json").is_file():
            self.skipTest("the local Qwen3.5-122B-A10B checkpoint is unavailable")
        devices = tuple(
            int(item)
            for item in os.environ.get("CUBIC_QWEN_MOE_AUDIT_DEVICES", "0,1,2,3").split(",")
        )
        if len(devices) != 4 or torch.cuda.device_count() <= max(devices):
            self.skipTest("four selected CUDA devices are required")
        with tempfile.TemporaryDirectory(prefix="cubic-qwen-moe-performance-audit-") as directory:
            config = parse_qwen3_5_moe_config_json((_REAL_CHECKPOINT / "config.json").read_bytes())
            inventory = inspect_checkpoint(_REAL_CHECKPOINT, verify_payloads=False)
            weight_plan = plan_qwen3_5_moe_checkpoint_weights(config, inventory)
            topology = ParallelTopology.single_node(
                topology_id=f"qwen3_5_moe:real:tp4:pp1:ep{_REAL_PERFORMANCE_EP_SIZE}:audit",
                device_indices=devices,
                tp_size=4,
                pp_size=1,
                ep_size=_REAL_PERFORMANCE_EP_SIZE,
            )
            plan = plan_qwen3_5_moe_weights(config, weight_plan, topology)
            issued = pickle.loads(pickle.dumps((config, inventory, weight_plan, plan)))
            init_method = (Path(directory) / "store").resolve().as_uri()
            try:
                _spawn_with_timeout(
                    _real_performance_audit_worker,
                    (init_method, directory, *issued),
                    nprocs=4,
                    timeout=600,
                    report_directory=directory,
                    report_pattern="performance-audit-rank-*.done",
                )
            except Exception as error:
                diagnostics = "\n".join(
                    path.read_text(encoding="utf-8")
                    for path in sorted(Path(directory).glob("performance-audit-rank-*.json"))
                )
                self.fail(f"real performance audit worker failed: {error}\n{diagnostics}")
            reports = tuple(
                json.loads(Path(directory, f"performance-audit-rank-{rank}.json").read_text())
                for rank in range(4)
            )
            if _REAL_PERFORMANCE_DIAGNOSTIC_ONLY:
                errors = tuple(
                    f"rank{report['rank']}: {report['error']}"
                    for report in reports
                    if "error" in report
                )
                if errors:
                    self.fail("\n".join(errors))
                self.assertTrue(
                    all(
                        route["exact"]
                        for report in reports
                        for route in report["shadow_routes"].values()
                    ),
                    "same-input eager router and candidate top-k differ",
                )
                print(
                    json.dumps(
                        {
                            "mode": "independent-eager-boundary-diagnostic",
                            "gdn_core_reduction_witnesses": [
                                len(report["direct_state_witnesses"]) for report in reports
                            ],
                            "same_input_router_exact": [
                                sum(route["exact"] for route in report["shadow_routes"].values())
                                for report in reports
                            ],
                            "end_to_end_route_exact": [
                                sum(
                                    value["exact"]
                                    for name, value in report["layers"].items()
                                    if name.endswith(":route")
                                )
                                for report in reports
                            ],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                return
            else:
                self.assertTrue(all("error" not in report for report in reports), reports)
            if _RUN_REAL_PERFORMANCE_EAGER_ONLY:
                self.assertEqual({report["mode"] for report in reports}, {"independent-eager"})
                self.assertTrue(
                    all(
                        report["causal_complete"]
                        and report["causal_layer_end"] == config.graph.text.layer_count - 1
                        for report in reports
                    ),
                    "causal qualification must cover every model layer",
                )
                self.assertEqual(
                    {report["reference_token"] for report in reports}
                    | {report["performance_token"] for report in reports},
                    {reports[0]["reference_token"]},
                )
                shadow_phase_summary = (
                    {
                        phase: {
                            "exact_layers": sum(
                                reports[0]["shadow_layers"][f"{layer}:{phase}"]["exact"]
                                for layer in range(48)
                            ),
                            "worst": max(
                                (
                                    reports[0]["shadow_layers"][f"{layer}:{phase}"]["max_abs"],
                                    layer,
                                    reports[0]["shadow_layers"][f"{layer}:{phase}"]["unequal"],
                                )
                                for layer in range(48)
                            ),
                        }
                        for phase in (
                            "layer_input",
                            "input_norm",
                            "attention_mixed",
                            "attention",
                            "mlp_input",
                            "mlp_output",
                            "mlp",
                        )
                    }
                    if reports[0]["shadow_layers"] and not _REAL_PERFORMANCE_DIAGNOSTIC_ONLY
                    else {}
                )
                print(
                    json.dumps(
                        {
                            "mode": "independent-eager",
                            "state": reports[0]["recurrent_state_dtype"],
                            "token": reports[0]["reference_token"],
                            "route_exact": [
                                sum(
                                    value["exact"]
                                    for name, value in report["layers"].items()
                                    if name.endswith(":route")
                                )
                                for report in reports
                            ],
                            "route_differences": {
                                name: value
                                for name, value in reports[0]["layers"].items()
                                if name.endswith(":route") and not value["exact"]
                            },
                            "shadow_route_differences": {
                                layer: value
                                for layer, value in reports[0]["shadow_routes"].items()
                                if not value["exact"]
                            },
                            "linear_witness_count": len(reports[0]["linear_witnesses"]),
                            "causal_layer_end": reports[0]["causal_layer_end"],
                            "causal_event_count": reports[0]["causal_event_count"],
                            "causal_witness_count": len(reports[0]["causal_linear_witnesses"]),
                            "shadow_phase_summary": shadow_phase_summary,
                            "phase_trace": {
                                layer: {
                                    phase: reports[0]["layers"][f"{layer}:{phase}"]
                                    for phase in (
                                        "input_norm",
                                        "attention_mixed",
                                        "attention",
                                        "mlp_input",
                                        "mlp_output",
                                        "mlp",
                                    )
                                }
                                for layer in ("0", "17", "18", "31", "32", "41", "42")
                            },
                            "cache_summary": {
                                kind: {
                                    "exact": sum(
                                        value["exact"]
                                        for name, value in reports[0]["cache"].items()
                                        if name.endswith(f":{kind}")
                                    ),
                                    "count": sum(
                                        name.endswith(f":{kind}") for name in reports[0]["cache"]
                                    ),
                                    "max_abs": max(
                                        value["max_abs"]
                                        for name, value in reports[0]["cache"].items()
                                        if name.endswith(f":{kind}")
                                    ),
                                }
                                for kind in ("conv", "recurrent", "key", "value")
                            },
                            "shadow_cache_summary": (
                                {
                                    kind: {
                                        "exact": sum(
                                            value["exact"]
                                            for name, value in reports[0]["shadow_cache"].items()
                                            if name.endswith(f":{kind}")
                                        ),
                                        "count": sum(
                                            name.endswith(f":{kind}")
                                            for name in reports[0]["shadow_cache"]
                                        ),
                                        "max_abs": max(
                                            value["max_abs"]
                                            for name, value in reports[0]["shadow_cache"].items()
                                            if name.endswith(f":{kind}")
                                        ),
                                    }
                                    for kind in ("conv", "recurrent", "key", "value")
                                }
                                if reports[0]["shadow_cache"]
                                and not _REAL_PERFORMANCE_DIAGNOSTIC_ONLY
                                else {}
                            ),
                            "direct_state_summary": (
                                {
                                    phase: {
                                        "exact": sum(
                                            value["exact"]
                                            for name, value in reports[0]["direct_state"].items()
                                            if name.endswith(f":{phase}")
                                        ),
                                        "count": sum(
                                            name.endswith(f":{phase}")
                                            for name in reports[0]["direct_state"]
                                        ),
                                        "max_abs": max(
                                            value["max_abs"]
                                            for name, value in reports[0]["direct_state"].items()
                                            if name.endswith(f":{phase}")
                                        ),
                                    }
                                    for phase in (
                                        "convolved",
                                        "core",
                                        "normalized",
                                        "conv",
                                        "recurrent",
                                    )
                                }
                                if reports[0]["direct_state"]
                                else {}
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                route_differences = {
                    name: value
                    for name, value in reports[0]["layers"].items()
                    if name.endswith(":route") and not value["exact"]
                }
                if route_differences:
                    self.assertTrue(
                        all(report["causal_complete"] for report in reports),
                        "independent eager route changes require complete causal receipts",
                    )
            else:
                print(json.dumps(reports, sort_keys=True), flush=True)


def _real_mtp_audit_worker(
    rank: int,
    init_method: str,
    report_directory: str,
    config: Any,
    inventory: Any,
    weight_plan: Any,
    plan: Any,
) -> None:
    report_path = Path(report_directory, f"mtp-audit-rank-{rank}.json")
    try:
        configure_reference_math()
        sealed = seal_qwen3_5_moe_mtp_distributed_reference(config, plan)
        with initialize_eager_nccl(
            sealed, rank=rank, init_method=init_method, timeout_seconds=600
        ) as runtime:
            weights = load_qwen3_5_moe_rank_weights(
                inventory=inventory,
                weight_plan=weight_plan,
                parallel_plan=plan,
                rank=rank,
                device=runtime.device,
            )
            body = Qwen3_5TPMTPBody(
                config=config,
                parallel_plan=plan,
                weights=weights,
                tp_group=runtime._groups["tp:0"],
                captured_m=1,
                table_pages=1,
                page_capacity=2,
                max_proposal_tokens=7,
            )
            reference = Qwen3_5MoeDistributedMTPReferenceModel(
                config=config,
                weights=weights,
                parallel_plan=plan,
                runtime=runtime,
                packed_experts=(body._expert_gate_up, body._expert_down),
            )
            hidden_size = config.graph.text.hidden_size
            target_hidden = (
                (torch.arange(hidden_size, device=runtime.device).remainder(257) - 128)
                .div(128)
                .to(torch.bfloat16)
                .reshape(1, 1, hidden_size)
            )
            token_ids = torch.tensor([[814]], dtype=torch.int64, device=runtime.device)
            position_ids = torch.zeros((3, 1, 1), dtype=torch.int64, device=runtime.device)
            reference_observations: dict[str, Tensor] = {}
            reference_boundaries: dict[str, Tensor] = {}
            reference._canonical._moe.observer = (  # type: ignore[attr-defined]
                lambda _layer, phase, value: reference_observations.__setitem__(
                    phase, value.detach().clone()
                )
            )
            with ReferenceExecutionContext():
                expected = reference.forward(
                    token_ids,
                    target_hidden,
                    position_ids=position_ids,
                    epoch_id="real-moe:mtp-eager-oracle",
                    observer=lambda phase, value: reference_boundaries.__setitem__(
                        phase, value.detach().clone()
                    ),
                )
            expected_route = reference_observations["route"].clone()
            reference._canonical._moe.observer = None  # type: ignore[attr-defined]

            body.arena.key.zero_()
            body.arena.value.zero_()
            body.arena.written.zero_()
            body.bind_token_ids(token_ids.flatten())
            body.buffers.position_ids.zero_()
            body.buffers.page_destination.zero_()
            body.buffers.token_offsets.zero_()
            body.buffers.page_table.zero_()
            body.buffers.sequence_lengths.fill_(1)
            body.buffers.target_hidden.copy_(target_hidden.reshape(1, hidden_size))
            body._active_rows.fill_(1)
            performance_observations: dict[str, Tensor] = {}
            body.launch(
                lambda phase, value: performance_observations.__setitem__(
                    phase, value.detach().clone()
                )
            )
            torch.cuda.synchronize(runtime.device)
            initial_route_exact = torch.equal(performance_observations["route"], expected_route)

            eager_hidden = body.buffers.norm.clone()
            eager_logits = body.buffers.logits.clone()
            eager_key = body.arena.key[0, 0, 0].clone()
            eager_value = body.arena.value[0, 0, 0].clone()
            if not _RUN_REAL_MTP_EAGER_ONLY:
                body.arena.key.zero_()
                body.arena.value.zero_()
                body.arena.written.zero_()
                graph = Qwen3_5TPMTPGraphSlot.capture(body)
                body.arena.key.zero_()
                body.arena.value.zero_()
                body.arena.written.zero_()
                body.bind_token_ids(token_ids.flatten())
                body.buffers.position_ids.zero_()
                body.buffers.page_destination.zero_()
                body.buffers.token_offsets.zero_()
                body.buffers.page_table.zero_()
                body.buffers.sequence_lengths.fill_(1)
                body.buffers.target_hidden.copy_(target_hidden.reshape(1, hidden_size))
                body._active_rows.fill_(1)
                graph.replay()
                torch.cuda.synchronize(runtime.device)

            span = body.vocab_span
            expected_logits = expected.logits[..., span.start : span.stop].reshape_as(
                body.buffers.logits
            )
            expected_hidden = expected.hidden_states.reshape_as(body.buffers.norm)
            logits_delta = (body.buffers.logits.float() - expected_logits.float()).abs()
            hidden_delta = (body.buffers.norm.float() - expected_hidden.float()).abs()
            initial_hidden_exact = torch.equal(body.buffers.norm, expected_hidden)
            initial_logits_exact = torch.equal(body.buffers.logits, expected_logits)
            expected_key = expected.cache.layer.key
            expected_value = expected.cache.layer.value
            if expected_key is None or expected_value is None:
                raise AssertionError("eager MTP oracle returned no KV state")
            key = body.arena.key[0, 0, 0].reshape_as(expected_key)
            value = body.arena.value[0, 0, 0].reshape_as(expected_value)
            gathered_logits = [torch.empty_like(body.buffers.logits) for _ in range(4)]
            torch.distributed.all_gather(
                gathered_logits, body.buffers.logits, group=runtime._groups["tp:0"]
            )
            full_logits = torch.cat(gathered_logits, dim=-1)[
                ..., : config.graph.text.vocabulary_size
            ]
            graph_checks = (
                {"skipped_eager_only": True}
                if _RUN_REAL_MTP_EAGER_ONLY
                else {
                    "hidden": torch.equal(body.buffers.norm, eager_hidden),
                    "logits": torch.equal(body.buffers.logits, eager_logits),
                    "key": torch.equal(key, eager_key.reshape_as(key)),
                    "value": torch.equal(value, eager_value.reshape_as(value)),
                }
            )
            graph_hidden_max_abs = float(
                (body.buffers.norm.float() - eager_hidden.float()).abs().max().item()
            )
            graph_logits_max_abs = float(
                (body.buffers.logits.float() - eager_logits.float()).abs().max().item()
            )
            chain: list[dict[str, object]] = []
            reference_cache = expected.cache
            next_token = int(torch.argmax(expected.logits, dim=-1).item())
            previous_body_key = body.arena.key[0, 0, :1].clone()
            previous_body_value = body.arena.value[0, 0, :1].clone()
            witnessed_value_prefix = True

            def linear_witnesses(
                *,
                primitive: Any,
                inputs: Tensor,
                oracle_inputs: Tensor,
                weight: Tensor,
                candidate: Tensor,
                oracle: Tensor,
            ) -> tuple[str, str] | None:
                columns = tuple(
                    int(index)
                    for index in torch.nonzero((candidate != oracle).any(dim=0), as_tuple=False)
                    .flatten()
                    .tolist()
                )
                if not columns:
                    return None
                artifact = primitive.project_artifact()
                candidate_binding = stable_digest(
                    {
                        "inputs": tensor_digest(inputs),
                        "weight": tensor_digest(weight),
                        "shape": tuple(candidate.shape),
                    }
                )
                oracle_binding = stable_digest(
                    {
                        "inputs": tensor_digest(oracle_inputs),
                        "weight": tensor_digest(weight),
                        "shape": tuple(oracle.shape),
                    }
                )
                candidate_witness = verify_observed_bf16_linear_sample(
                    x=inputs,
                    weight=weight,
                    output=candidate,
                    input_rows=(0,),
                    output_columns=columns,
                    operand_binding_digest=candidate_binding,
                    artifact_digest=artifact.artifact_digest,
                    launch_specialization_digest=artifact.launch_specialization_digest,
                )
                oracle_witness = verify_observed_bf16_linear_sample(
                    x=oracle_inputs,
                    weight=weight,
                    output=oracle,
                    input_rows=(0,),
                    output_columns=columns,
                    operand_binding_digest=oracle_binding,
                    artifact_digest=stable_digest({"implementation": "independent-eager-mm"}),
                    launch_specialization_digest=stable_digest(
                        {"shape": tuple(oracle.shape), "reduction": "complete-k"}
                    ),
                )
                return candidate_witness.witness_digest, oracle_witness.witness_digest

            def witness_cache_boundary(position: int) -> dict[str, object]:
                """Prove cache writes from this forward's actual producing operands."""

                observed = performance_observations
                prefix = "mtp.layers.0.self_attn"
                witnesses: dict[str, object] = {}
                stages = (
                    ("fc", body._fc, observed["fc_input"], body._weights["mtp.fc.weight"],
                     body.buffers.fc_local),
                    ("q", body._q, observed["attention_input"],
                     body._weights[f"{prefix}.q_proj.weight"],
                     observed["q_projection"].reshape(1, -1)),
                    ("k", body._kv, observed["attention_input"],
                     body._weights[f"{prefix}.k_proj.weight"],
                     observed["k_projection"].reshape(1, -1)),
                    ("v", body._kv, observed["attention_input"],
                     body._weights[f"{prefix}.v_proj.weight"],
                     observed["v_projection"].reshape(1, -1)),
                )
                for name, primitive, inputs, weight, candidate in stages:
                    with ReferenceExecutionContext():
                        oracle = reference_linear(inputs, weight)
                    witnesses[name] = linear_witnesses(
                        primitive=primitive, inputs=inputs, oracle_inputs=inputs,
                        weight=weight, candidate=candidate, oracle=oracle,
                    ) or "exact"
                fc = observed["fc"][:, body._fc_span.start : body._fc_span.stop]
                if not torch.equal(fc, body.buffers.fc_local):
                    raise AssertionError("MTP FC shard assembly changed its local producer")
                with ReferenceExecutionContext():
                    norm = zero_centered_rms_norm(
                        observed["fc"], body._weights["mtp.layers.0.input_layernorm.weight"],
                        epsilon=config.graph.text.rms_norm_epsilon,
                        output_dtype=torch.bfloat16,
                    )
                    query = observed["q_projection"][:, :, 0, :]
                    key = observed["k_projection"]
                    normalized_query = zero_centered_rms_norm(
                        query, body._weights[f"{prefix}.q_norm.weight"],
                        epsilon=config.graph.text.rms_norm_epsilon,
                        output_dtype=torch.bfloat16,
                    )
                    normalized_key = zero_centered_rms_norm(
                        key, body._weights[f"{prefix}.k_norm.weight"],
                        epsilon=config.graph.text.rms_norm_epsilon,
                        output_dtype=torch.bfloat16,
                    )
                    rope = config.graph.text.mrope
                    cosine, sine = interleaved_mrope_cos_sin(
                        torch.full((3, 1, 1), position, dtype=torch.int64,
                                   device=runtime.device),
                        rotary_dim=int(config.graph.text.full_attention.head_dim
                                       * rope.partial_rotary_factor),
                        mrope_section=rope.section,
                        rope_theta=rope.rope_theta, output_dtype=body.buffers.cosine.dtype,
                    )
                    _, rotated_key = apply_partial_rotary(
                        normalized_query.unsqueeze(0).transpose(1, 2),
                        normalized_key.unsqueeze(0).transpose(1, 2),
                        cosine, sine, output_dtype=torch.bfloat16,
                    )
                if not torch.equal(norm, observed["attention_input"]):
                    raise AssertionError("MTP attention norm differs on its actual FC input")
                if not torch.equal(normalized_key, body.buffers.normalized_key):
                    raise AssertionError("MTP key norm differs on its actual K projection")
                if not torch.equal(normalized_query, body.buffers.normalized_query):
                    raise AssertionError("MTP query norm differs on its actual Q projection")
                if not torch.equal(cosine.reshape_as(body.buffers.cosine), body.buffers.cosine):
                    raise AssertionError("MTP cosine table differs from independent M-RoPE")
                if not torch.equal(sine.reshape_as(body.buffers.sine), body.buffers.sine):
                    raise AssertionError("MTP sine table differs from independent M-RoPE")
                stored_key = body.arena.key[0, 0, position].reshape_as(rotated_key)
                stored_value = body.arena.value[0, 0, position].reshape_as(
                    observed["v_projection"]
                )
                if not torch.equal(stored_key, rotated_key):
                    raise AssertionError(
                        "MTP cache key differs from its independently rotated producer"
                    )
                if not torch.equal(stored_value, observed["v_projection"]):
                    raise AssertionError("MTP cache value differs from its audited V producer")
                witnesses["state_write"] = stable_digest({
                    "position": position, "key": tensor_digest(stored_key),
                    "value": tensor_digest(stored_value),
                })
                return witnesses

            if not torch.equal(
                performance_observations["fc_input"],
                reference_boundaries["fc_input"].reshape_as(performance_observations["fc_input"]),
            ):
                raise AssertionError(
                    "initial MTP FC operands differ from independent input producers"
                )
            initial_cache_witness = witness_cache_boundary(0)
            for depth in range(1, 7):
                step_hidden = (
                    (torch.arange(hidden_size, device=runtime.device) + 17 * depth)
                    .remainder(257)
                    .sub(128)
                    .div(128)
                    .to(torch.bfloat16)
                    .reshape(1, 1, hidden_size)
                )
                step_token = torch.tensor([[next_token]], dtype=torch.int64, device=runtime.device)
                step_position = torch.full(
                    (3, 1, 1), depth, dtype=torch.int64, device=runtime.device
                )
                reference_observations.clear()
                reference_boundaries.clear()
                reference._canonical._moe.observer = (  # type: ignore[attr-defined]
                    lambda _layer, phase, value: reference_observations.__setitem__(
                        phase, value.detach().clone()
                    )
                )
                with ReferenceExecutionContext():
                    reference_step = reference.forward(
                        step_token,
                        step_hidden,
                        position_ids=step_position,
                        cache=reference_cache,
                        epoch_id=f"real-moe:mtp-eager-oracle:{depth}",
                        observer=lambda phase, value: reference_boundaries.__setitem__(
                            phase, value.detach().clone()
                        ),
                    )
                reference._canonical._moe.observer = None  # type: ignore[attr-defined]
                body.bind_token_ids(step_token.flatten())
                body.buffers.position_ids.fill_(depth)
                body.buffers.page_destination.zero_()
                body.buffers.token_offsets.fill_(depth)
                body.buffers.page_table.zero_()
                body.buffers.sequence_lengths.fill_(depth + 1)
                body.buffers.target_hidden.copy_(step_hidden.reshape(1, hidden_size))
                performance_observations.clear()
                body.launch(
                    lambda phase, value: performance_observations.__setitem__(
                        phase, value.detach().clone()
                    )
                )
                torch.cuda.synchronize(runtime.device)
                gathered_step_logits = [torch.empty_like(body.buffers.logits) for _ in range(4)]
                torch.distributed.all_gather(
                    gathered_step_logits,
                    body.buffers.logits,
                    group=runtime._groups["tp:0"],
                )
                step_full_logits = torch.cat(gathered_step_logits, dim=-1)[
                    ..., : config.graph.text.vocabulary_size
                ]
                next_token = int(torch.argmax(reference_step.logits, dim=-1).item())
                reference_key = reference_step.cache.layer.key
                reference_value = reference_step.cache.layer.value
                if reference_key is None or reference_value is None:
                    raise AssertionError("MTP chain oracle returned no KV state")
                body_key = body.arena.key[0, 0, : depth + 1].permute(1, 0, 2).unsqueeze(0)
                body_value = body.arena.value[0, 0, : depth + 1].permute(1, 0, 2).unsqueeze(0)
                key_delta = (body_key.float() - reference_key.float()).abs()
                value_delta = (body_value.float() - reference_value.float()).abs()
                reference_attention_input = reference_boundaries["attention_input"].reshape_as(
                    performance_observations["attention_input"]
                )
                attention_input_delta = (
                    performance_observations["attention_input"].float()
                    - reference_attention_input.float()
                ).abs()
                reference_fc = reference_boundaries["fc"].reshape_as(performance_observations["fc"])
                fc_delta = (performance_observations["fc"].float() - reference_fc.float()).abs()
                reference_fc_input = reference_boundaries["fc_input"].reshape_as(
                    performance_observations["fc_input"]
                )
                with ReferenceExecutionContext():
                    performance_norm_oracle = zero_centered_rms_norm(
                        performance_observations["fc"].reshape(1, 1, hidden_size),
                        body._weights["mtp.layers.0.input_layernorm.weight"],
                        epsilon=config.graph.text.rms_norm_epsilon,
                        output_dtype=torch.bfloat16,
                    ).reshape_as(performance_observations["attention_input"])
                local_kv_width = body.buffers.value.shape[1] * body.buffers.value.shape[2]
                local_kv_start = (rank // 2) * local_kv_width
                reference_v = (
                    reference_boundaries["v_projection"]
                    .reshape(1, -1)
                    .narrow(1, local_kv_start, local_kv_width)
                )
                performance_v = performance_observations["v_projection"].reshape(1, -1)
                v_projection_delta = (performance_v.float() - reference_v.float()).abs()
                fc_span = body._fc_span
                reference_local_fc = reference_fc[:, fc_span.start : fc_span.stop]
                fc_witnesses = linear_witnesses(
                    primitive=body._fc,
                    inputs=performance_observations["fc_input"],
                    oracle_inputs=reference_fc_input,
                    weight=body._weights["mtp.fc.weight"],
                    candidate=body.buffers.fc_local,
                    oracle=reference_local_fc,
                )
                v_witnesses = linear_witnesses(
                    primitive=body._kv,
                    inputs=performance_observations["attention_input"],
                    oracle_inputs=reference_attention_input,
                    weight=body._weights["mtp.layers.0.self_attn.v_proj.weight"],
                    candidate=performance_v,
                    oracle=reference_v,
                )
                current_key_exact = torch.equal(body_key[:, :, -1:], reference_key[:, :, -1:])
                current_value_exact = torch.equal(body_value[:, :, -1:], reference_value[:, :, -1:])
                cache_witness = witness_cache_boundary(depth)
                current_value_witnessed = current_value_exact or v_witnesses is not None
                witnessed_value_prefix = witnessed_value_prefix and current_value_witnessed
                kv_reduction_witnessed = (
                    current_key_exact
                    and witnessed_value_prefix
                    and (fc_delta.count_nonzero().item() == 0 or fc_witnesses is not None)
                    and torch.equal(
                        performance_observations["attention_input"],
                        performance_norm_oracle,
                    )
                )
                chain.append(
                    {
                        "depth": depth + 1,
                        "reference_token": next_token,
                        "performance_token": int(torch.argmax(step_full_logits, dim=-1).item()),
                        "route_exact": torch.equal(
                            body._routed_experts.route_ids,
                            reference_observations["route"],
                        ),
                        "kv_exact": torch.equal(body_key, reference_key)
                        and torch.equal(body_value, reference_value),
                        "key_max_abs": float(key_delta.max().item()),
                        "key_unequal": int(torch.count_nonzero(body_key != reference_key).item()),
                        "value_max_abs": float(value_delta.max().item()),
                        "value_unequal": int(
                            torch.count_nonzero(body_value != reference_value).item()
                        ),
                        "attention_input_max_abs": float(attention_input_delta.max().item()),
                        "attention_input_unequal": int(
                            torch.count_nonzero(
                                performance_observations["attention_input"]
                                != reference_attention_input
                            ).item()
                        ),
                        "fc_max_abs": float(fc_delta.max().item()),
                        "fc_unequal": int(
                            torch.count_nonzero(
                                performance_observations["fc"] != reference_fc
                            ).item()
                        ),
                        "fc_input_exact": torch.equal(
                            performance_observations["fc_input"], reference_fc_input
                        ),
                        "attention_norm_self_exact": torch.equal(
                            performance_observations["attention_input"],
                            performance_norm_oracle,
                        ),
                        "v_projection_max_abs": float(v_projection_delta.max().item()),
                        "v_projection_unequal": int(
                            torch.count_nonzero(performance_v != reference_v).item()
                        ),
                        "fc_witnesses": fc_witnesses,
                        "v_witnesses": v_witnesses,
                        "kv_reduction_witnessed": kv_reduction_witnessed,
                        "cache_producer_witness": cache_witness,
                        "prefix_immutable": torch.equal(
                            body.arena.key[0, 0, :depth], previous_body_key
                        )
                        and torch.equal(body.arena.value[0, 0, :depth], previous_body_value),
                    }
                )
                previous_body_key = body.arena.key[0, 0, : depth + 1].clone()
                previous_body_value = body.arena.value[0, 0, : depth + 1].clone()
                reference_cache = reference_step.cache
            report_path.write_text(
                json.dumps(
                    {
                        "rank": rank,
                        "reference_token": int(torch.argmax(expected.logits, dim=-1).item()),
                        "performance_token": int(torch.argmax(full_logits, dim=-1).item()),
                        "hidden": {
                            "exact": initial_hidden_exact,
                            "max_abs": float(hidden_delta.max().item()),
                            "mean_abs": float(hidden_delta.mean().item()),
                        },
                        "logits": {
                            "exact": initial_logits_exact,
                            "max_abs": float(logits_delta.max().item()),
                            "mean_abs": float(logits_delta.mean().item()),
                        },
                        "kv_exact": torch.equal(key, expected_key)
                        and torch.equal(value, expected_value),
                        "cache_producer_witness": initial_cache_witness,
                        "route_exact": initial_route_exact,
                        "mode": (
                            "independent-eager"
                            if _RUN_REAL_MTP_EAGER_ONLY
                            else "independent-eager-plus-capture"
                        ),
                        "graph_exact": (
                            None if _RUN_REAL_MTP_EAGER_ONLY else all(graph_checks.values())
                        ),
                        "graph_checks": graph_checks,
                        "graph_hidden_max_abs": graph_hidden_max_abs,
                        "graph_logits_max_abs": graph_logits_max_abs,
                        "chain": chain,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            torch.distributed.barrier()
    except BaseException:
        report_path.write_text(
            json.dumps({"rank": rank, "error": traceback.format_exc()}, sort_keys=True),
            encoding="utf-8",
        )
        raise


@unittest.skipUnless(
    _RUN_REAL_MTP_AUDIT,
    "set CUBIC_RUN_REAL_QWEN_MOE_MTP_AUDIT=1 for the eager MTP oracle audit",
)
class Qwen3_5MoeRealMTPAuditTests(unittest.TestCase):
    def test_performance_mtp_body_against_independent_eager_oracle(self) -> None:
        if not (_REAL_CHECKPOINT / "model.safetensors.index.json").is_file():
            self.skipTest("the local Qwen3.5-122B-A10B checkpoint is unavailable")
        devices = tuple(
            int(item)
            for item in os.environ.get("CUBIC_QWEN_MOE_AUDIT_DEVICES", "0,1,2,3").split(",")
        )
        if len(devices) != 4 or torch.cuda.device_count() <= max(devices):
            self.skipTest("four selected CUDA devices are required")
        with tempfile.TemporaryDirectory(prefix="cubic-qwen-moe-mtp-audit-") as directory:
            config = parse_qwen3_5_moe_config_json((_REAL_CHECKPOINT / "config.json").read_bytes())
            inventory = inspect_checkpoint(_REAL_CHECKPOINT, verify_payloads=False)
            weight_plan = plan_qwen3_5_moe_checkpoint_weights(config, inventory)
            topology = ParallelTopology.single_node(
                topology_id=f"qwen3_5_moe:real:tp4:pp1:ep{_REAL_PERFORMANCE_EP_SIZE}:mtp-audit",
                device_indices=devices,
                tp_size=4,
                pp_size=1,
                ep_size=_REAL_PERFORMANCE_EP_SIZE,
            )
            plan = plan_qwen3_5_moe_weights(config, weight_plan, topology, include_mtp=True)
            issued = pickle.loads(pickle.dumps((config, inventory, weight_plan, plan)))
            init_method = (Path(directory) / "store").resolve().as_uri()
            try:
                _spawn_with_timeout(
                    _real_mtp_audit_worker,
                    (init_method, directory, *issued),
                    nprocs=4,
                    timeout=600,
                    report_directory=directory,
                    report_pattern="mtp-audit-rank-*.json",
                )
            except Exception as error:
                diagnostics = "\n".join(
                    path.read_text(encoding="utf-8")
                    for path in sorted(Path(directory).glob("mtp-audit-rank-*.json"))
                )
                self.fail(f"real MoE MTP audit failed: {error}\n{diagnostics}")
            reports = tuple(
                json.loads(Path(directory, f"mtp-audit-rank-{rank}.json").read_text())
                for rank in range(4)
            )
            self.assertTrue(all("error" not in report for report in reports), reports)
            self.assertTrue(all(
                report["kv_exact"] or report["cache_producer_witness"] for report in reports
            ), reports)
            if _RUN_REAL_MTP_EAGER_ONLY:
                self.assertEqual({report["mode"] for report in reports}, {"independent-eager"})
                self.assertTrue(all(report["graph_exact"] is None for report in reports), reports)
            else:
                self.assertTrue(all(report["graph_exact"] for report in reports), reports)
            self.assertTrue(all(report["route_exact"] for report in reports), reports)
            self.assertEqual(
                {report["reference_token"] for report in reports}
                | {report["performance_token"] for report in reports},
                {reports[0]["reference_token"]},
            )
            for chain_index in range(len(reports[0]["chain"])):
                steps = tuple(report["chain"][chain_index] for report in reports)
                with self.subTest(depth=steps[0]["depth"]):
                    self.assertTrue(
                        all(
                            step["route_exact"]
                            and step["reference_token"] == step["performance_token"]
                            and step["fc_input_exact"]
                            and step["attention_norm_self_exact"]
                            and step["prefix_immutable"]
                            and (step["key_unequal"] == 0 or step["cache_producer_witness"])
                            and (
                                step["v_projection_unequal"] == 0 or step["v_witnesses"] is not None
                            )
                            for step in steps
                        ),
                        steps,
                    )
                    self.assertEqual(
                        any(step["fc_unequal"] for step in steps),
                        any(step["fc_witnesses"] is not None for step in steps),
                        steps,
                    )
            print(json.dumps(reports, sort_keys=True), flush=True)


def _real_load_worker(
    rank: int,
    init_method: str,
    report_directory: str,
    inventory: Any,
    weight_plan: Any,
    plan: Any,
) -> None:
    started = time.monotonic()
    configure_reference_math()
    torch.cuda.set_device(rank)
    torch.cuda.reset_peak_memory_stats(rank)
    print(f"MOE_REAL_LOAD rank={rank} start", flush=True)
    torch.distributed.init_process_group(
        "gloo",
        init_method=init_method,
        world_size=8,
        rank=rank,
        timeout=timedelta(minutes=15),
    )
    try:
        weights = load_qwen3_5_moe_rank_weights(
            inventory=inventory,
            weight_plan=weight_plan,
            parallel_plan=plan,
            rank=rank,
            device=f"cuda:{rank}",
        )
        expected = 30_537_663_168 if rank < 4 else 30_537_669_312
        experts = tuple(item for item in weights.bindings if item.layout is MoeWeightLayout.EXPERT)
        if weights.resident_bytes != expected or len(experts) != 48:
            raise AssertionError("real rank-local resident bytes or expert inventory differ")
        if any(item.local_shape[0] != 64 for item in experts):
            raise AssertionError("a real rank owns a non-local expert extent")
        torch.cuda.synchronize(rank)
        print(f"MOE_REAL_LOAD rank={rank} ready bytes={weights.resident_bytes}", flush=True)
        report_path = Path(report_directory, f"rank-{rank}.json")
        report: dict[str, object] = {
            "rank": rank,
            "resident_bytes": weights.resident_bytes,
            "tensor_count": weights.tensor_count,
            "expert_tensor_count": len(experts),
            "inventory_digest": weights.inventory_digest,
            "semantic_plan_digest": plan.semantic_plan_digest,
            "allocated_bytes": torch.cuda.memory_allocated(rank),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(rank),
            "load_wall_seconds": time.monotonic() - started,
            "barrier_complete": False,
        }
        report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
        # Both barriers execute while every process retains its full rank-local object.
        torch.distributed.barrier()
        print(f"MOE_REAL_LOAD rank={rank} world8-resident", flush=True)
        if len(tuple(Path(report_directory).glob("rank-*.json"))) != 8:
            raise AssertionError("world8 did not reach simultaneous rank-local residency")
        report["barrier_complete"] = True
        report["world8_wall_seconds"] = time.monotonic() - started
        report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


@unittest.skipUnless(
    os.environ.get("CUBIC_RUN_REAL_QWEN_MOE_LOAD") == "1",
    "set CUBIC_RUN_REAL_QWEN_MOE_LOAD=1 to load the real 122B rank shards",
)
class Qwen3_5MoeRealRankLoadTests(unittest.TestCase):
    def test_real_checkpoint_is_simultaneously_rank_local_on_world8(self) -> None:
        if not (_REAL_CHECKPOINT / "model.safetensors.index.json").is_file():
            self.skipTest("the local Qwen3.5-122B-A10B checkpoint is unavailable")
        if torch.cuda.device_count() < 8:
            self.skipTest("eight CUDA devices are required")
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="cubic-qwen-moe-load-") as directory:
            config = parse_qwen3_5_moe_config_json((_REAL_CHECKPOINT / "config.json").read_bytes())
            inventory = inspect_checkpoint(_REAL_CHECKPOINT)
            weight_plan = plan_qwen3_5_moe_checkpoint_weights(config, inventory)
            parallel_plan = plan_qwen3_5_moe_tp4_pp2_ep4(config, weight_plan, _topology())
            restored = pickle.loads(pickle.dumps((inventory, weight_plan, parallel_plan)))
            if (
                restored[0].inventory_digest != inventory.inventory_digest
                or restored[1].semantic_plan_digest != weight_plan.semantic_plan_digest
                or restored[2] != parallel_plan
            ):
                raise AssertionError("issued inventory or plan changed at the spawn boundary")
            init_method = (Path(directory) / "store").resolve().as_uri()
            torch.multiprocessing.spawn(
                _real_load_worker,
                args=(init_method, directory, *restored),
                nprocs=8,
                join=True,
            )
            reports = tuple(
                json.loads(Path(directory, f"rank-{rank}.json").read_text(encoding="utf-8"))
                for rank in range(8)
            )
            self.assertEqual(
                tuple(item["resident_bytes"] for item in reports),
                (30_537_663_168,) * 4 + (30_537_669_312,) * 4,
            )
            self.assertEqual(len({item["inventory_digest"] for item in reports}), 1)
            self.assertEqual(len({item["semantic_plan_digest"] for item in reports}), 1)
            self.assertTrue(all(item["barrier_complete"] for item in reports))
            print(
                "MOE_REAL_LOAD_SUMMARY "
                + json.dumps(
                    {"wall_seconds": time.monotonic() - started, "ranks": reports},
                    sort_keys=True,
                ),
                flush=True,
            )


if __name__ == "__main__":
    unittest.main()
