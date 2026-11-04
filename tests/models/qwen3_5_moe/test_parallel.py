from __future__ import annotations

import math
import unittest

import torch

from cubic_serve.contracts.execution import ParallelTopology
from cubic_serve.models.parallel_weights import AxisSpan, materialize_axis_spans
from cubic_serve.models.qwen3_5_moe import parse_qwen3_5_moe_config
from cubic_serve.models.qwen3_5_moe.parallel import (
    MoeWeightLayout,
    Qwen3_5MoeParallelError,
    plan_qwen3_5_moe_tp4_pp2_ep4,
    plan_qwen3_5_moe_weights,
)
from cubic_serve.models.qwen3_5_moe.weights import (
    _expected_headers,
    plan_qwen3_5_moe_weight_headers,
)
from cubic_serve.models.weight_schema import TensorHeader
from tests.models.qwen3_5_moe.test_config import _fixture


def _topology() -> ParallelTopology:
    return ParallelTopology.single_node(
        topology_id="qwen3.5-122b:gpu0-7:tp4:pp2:ep4",
        device_indices=tuple(range(8)),
        tp_size=4,
        pp_size=2,
        ep_size=4,
    )


def _plan():
    config = parse_qwen3_5_moe_config(_fixture())
    headers = tuple(
        TensorHeader(name, item.dtype, item.shape)
        for name, item in _expected_headers(config).items()
    )
    weights = plan_qwen3_5_moe_weight_headers(config, headers)
    return plan_qwen3_5_moe_tp4_pp2_ep4(config, weights, _topology())


class Qwen3_5MoeParallelPlanTests(unittest.TestCase):
    def test_tp4_ep2_preserves_expert_owner_and_pairs_etp_slices(self) -> None:
        from cubic_serve.executor.parallel import seal_partition

        config = parse_qwen3_5_moe_config(_fixture())
        headers = tuple(
            TensorHeader(name, item.dtype, item.shape)
            for name, item in _expected_headers(config).items()
        )
        weights = plan_qwen3_5_moe_weight_headers(config, headers)
        topology = ParallelTopology.single_node(
            topology_id="qwen3.5:tp4:ep2",
            device_indices=(0, 1, 2, 3),
            tp_size=4,
            pp_size=1,
            ep_size=2,
        )
        plan = plan_qwen3_5_moe_weights(config, weights, topology, include_mtp=True)
        seal_partition(topology, plan.partition)
        for assignment in plan.partition.expert_assignments:
            owner = (assignment.expert_id // 128) * 2
            self.assertEqual(assignment.owner_rank, owner)
            self.assertEqual(assignment.expert_tensor_ranks, (owner, owner + 1))
        for rank in range(4):
            gate, down = (
                next(
                    binding for binding in plan.bindings_for_rank(rank)
                    if binding.tensor_name == f"model.language_model.layers.0.mlp.experts.{role}"
                )
                for role in ("gate_up_proj", "down_proj")
            )
            self.assertEqual(gate.spans, down.spans)
            self.assertEqual(gate.secondary_axis, 1)
            self.assertEqual(down.secondary_axis, 2)
            self.assertEqual(gate.local_shape[1], 2 * down.local_shape[2])
            mtp = [
                item for item in plan.bindings_for_rank(rank)
                if item.tensor_name.startswith("mtp.layers.0.mlp.experts.")
            ]
            self.assertEqual(len(mtp), 128 * 3)
            self.assertEqual(math.prod(mtp[0].local_shape) * 2,
                             math.prod(mtp[0].global_shape))

    def test_two_axis_checkpoint_slice_reads_only_owned_rectangle(self) -> None:
        from cubic_serve.models.parallel_weights import LogicalWeightSlice

        class Reader:
            def __init__(self, tensor):
                self.tensor = tensor
                self.indices = []

            def get_slice(self, name):
                self.name = name
                return self

            def get_shape(self):
                return list(self.tensor.shape)

            def __getitem__(self, index):
                self.indices.append(index)
                return self.tensor[index]

        source = torch.arange(4 * 8 * 2, dtype=torch.bfloat16).reshape(4, 8, 2)
        reader = Reader(source)
        binding = LogicalWeightSlice("expert", 0, (AxisSpan(2, 4),), tuple(source.shape))
        class Rectangular:
            tensor_name = binding.tensor_name
            shard_axis = binding.shard_axis
            spans = binding.spans
            global_shape = binding.global_shape
            local_shape = (2, 4, 2)

        loaded = materialize_axis_spans(
            reader, Rectangular(), dtype=torch.bfloat16, device=torch.device("cpu"),
            secondary_axis=1, secondary_spans=(AxisSpan(0, 2), AxisSpan(4, 6)),
        )
        self.assertTrue(torch.equal(loaded, torch.cat((source[2:4, 0:2], source[2:4, 4:6]), dim=1)))
        self.assertEqual(reader.indices, [(slice(2, 4), slice(0, 2), slice(None)),
                                          (slice(2, 4), slice(4, 6), slice(None))])

    def test_tp4_ep4_mtp_owns_one_expert_slice_and_replicated_kv_pairs(self) -> None:
        config = parse_qwen3_5_moe_config(_fixture())
        headers = tuple(
            TensorHeader(name, item.dtype, item.shape)
            for name, item in _expected_headers(config).items()
        )
        weights = plan_qwen3_5_moe_weight_headers(config, headers)
        topology = ParallelTopology.single_node(
            topology_id="qwen3.5-122b:gpu0-3:tp4:pp1:ep4:mtp",
            device_indices=tuple(range(4)),
            tp_size=4,
            pp_size=1,
            ep_size=4,
        )

        plan = plan_qwen3_5_moe_weights(config, weights, topology, include_mtp=True)

        self.assertTrue(plan.mtp_enabled)
        self.assertEqual(len(plan.global_names), 1616)
        self.assertEqual(len(plan.partition.expert_assignments), 49 * 256)
        self.assertEqual(
            tuple(item.head_span for item in plan.mtp_kv_heads),
            (AxisSpan(0, 1), AxisSpan(0, 1), AxisSpan(1, 2), AxisSpan(1, 2)),
        )
        for rank in range(4):
            local = tuple(
                item.tensor_name
                for item in plan.bindings_for_rank(rank)
                if item.tensor_name.startswith("mtp.layers.0.mlp.experts.")
            )
            self.assertEqual(len(local), 64 * 3)
            ids = {int(name.split(".experts.")[1].split(".")[0]) for name in local}
            self.assertEqual(ids, set(range(rank * 64, (rank + 1) * 64)))

    def test_mtp_tp_shards_pair_unpacked_gate_up_and_down(self) -> None:
        from cubic_serve.executor.parallel import seal_partition

        config = parse_qwen3_5_moe_config(_fixture())
        headers = tuple(
            TensorHeader(name, item.dtype, item.shape)
            for name, item in _expected_headers(config).items()
        )
        weights = plan_qwen3_5_moe_weight_headers(config, headers)
        topology = ParallelTopology.single_node(
            topology_id="mtp:tp4:ep1",
            device_indices=(0, 1, 2, 3),
            tp_size=4,
            pp_size=1,
            ep_size=1,
        )
        plan = plan_qwen3_5_moe_weights(config, weights, topology, include_mtp=True)
        seal_partition(topology, plan.partition)
        for rank in range(4):
            local = {
                b.tensor_name: b
                for b in plan.bindings_for_rank(rank)
                if b.tensor_name.startswith("mtp.layers.0.mlp.experts.")
            }
            self.assertEqual(len(local), 256 * 3)
            for expert in range(256):
                prefix = f"mtp.layers.0.mlp.experts.{expert}"
                gate, up, down = (
                    local[f"{prefix}.{op}_proj.weight"] for op in ("gate", "up", "down")
                )
                self.assertEqual((gate.shard_axis, up.shard_axis, down.shard_axis), (0, 0, 1))
                self.assertEqual(gate.spans, up.spans)
                self.assertEqual(gate.spans, down.spans)
                for binding in (gate, up, down):
                    self.assertEqual(
                        math.prod(binding.local_shape) * 4, math.prod(binding.global_shape)
                    )
        for assignment in plan.partition.expert_assignments:
            self.assertEqual(assignment.owner_rank, 0)
            self.assertEqual(assignment.expert_tensor_ranks, (0, 1, 2, 3))

    def test_tp4_ep1_pairs_intermediate_shards_without_expert_replication(self) -> None:
        from cubic_serve.executor.parallel import seal_partition

        config = parse_qwen3_5_moe_config(_fixture())
        headers = tuple(
            TensorHeader(name, item.dtype, item.shape)
            for name, item in _expected_headers(config).items()
        )
        weights = plan_qwen3_5_moe_weight_headers(config, headers)
        for pp in (1, 2):
            topology = ParallelTopology.single_node(
                topology_id=f"tp4-pp{pp}-ep1",
                device_indices=tuple(range(4 * pp)),
                tp_size=4,
                pp_size=pp,
                ep_size=1,
            )
            plan = plan_qwen3_5_moe_weights(config, weights, topology)
            seal_partition(topology, plan.partition)
            for assignment in plan.partition.expert_assignments:
                self.assertEqual(
                    assignment.expert_tensor_ranks,
                    tuple(range(assignment.owner_rank, assignment.owner_rank + 4)),
                )
            up = [
                b
                for b in plan.bindings
                if b.tensor_name == "model.language_model.layers.0.mlp.experts.gate_up_proj"
            ]
            down = [
                b
                for b in plan.bindings
                if b.tensor_name == "model.language_model.layers.0.mlp.experts.down_proj"
            ]
            for u, d in zip(up, down, strict=True):
                self.assertEqual(u.local_shape[0], config.graph.text.moe.expert_count)
                self.assertEqual(u.shard_axis, 1)
                self.assertEqual(d.shard_axis, 2)
                self.assertEqual(u.local_shape[1], 2 * d.local_shape[2])
                self.assertEqual(u.spans[0], d.spans[0])
                self.assertEqual(u.spans[1].start, u.global_shape[1] // 2 + d.spans[0].start)
                self.assertEqual(math.prod(u.local_shape) * 4, math.prod(u.global_shape))
                self.assertEqual(math.prod(d.local_shape) * 4, math.prod(d.global_shape))

    def test_true_ep_ownership_and_rank_memory_are_closed(self) -> None:
        plan = _plan()
        self.assertEqual(len(plan.global_names), 831)
        self.assertEqual(len(plan.partition.expert_assignments), 48 * 256)
        self.assertEqual(len(plan.kv_heads), 12 * 4)
        self.assertEqual(len(plan.gdn_states), 36 * 4)

        expected_bytes = (30_565_974_720,) * 4 + (30_565_980_864,) * 4
        actual_bytes: list[int] = []
        for rank in range(8):
            bindings = plan.bindings_for_rank(rank)
            actual_bytes.append(
                sum(
                    math.prod(item.local_shape) * (4 if item.dtype is torch.float32 else 2)
                    for item in bindings
                )
            )
            expert_bindings = tuple(
                item for item in bindings if item.layout is MoeWeightLayout.EXPERT
            )
            self.assertEqual(len(expert_bindings), 48)
            self.assertTrue(all(item.local_shape[0] == 64 for item in expert_bindings))
            self.assertEqual(
                sum(math.prod(item.local_shape) * 2 for item in expert_bindings),
                27 * 2**30,
            )
        self.assertEqual(tuple(actual_bytes), expected_bytes)

        layer_9 = tuple(
            item
            for item in plan.partition.expert_assignments
            if item.node_id == "text.decoder.layer.0009"
        )
        self.assertEqual(
            tuple((item.expert_id, item.owner_rank) for item in layer_9[::64]),
            ((0, 0), (64, 1), (128, 2), (192, 3)),
        )
        layer_33 = tuple(
            item
            for item in plan.partition.expert_assignments
            if item.node_id == "text.decoder.layer.0033"
        )
        self.assertEqual(
            tuple((item.expert_id, item.owner_rank) for item in layer_33[::64]),
            ((0, 4), (64, 5), (128, 6), (192, 7)),
        )

    def test_two_kv_heads_form_explicit_replica_factor_two(self) -> None:
        plan = _plan()
        owners = tuple(item for item in plan.kv_heads if item.layer_index == 3)
        self.assertEqual(
            tuple((item.head_span.start, item.head_span.stop) for item in owners),
            ((0, 1), (0, 1), (1, 2), (1, 2)),
        )
        self.assertEqual(
            tuple(item.replica_ranks for item in owners),
            ((0, 1), (0, 1), (2, 3), (2, 3)),
        )
        k_name = "model.language_model.layers.3.self_attn.k_proj.weight"
        bindings = tuple(item for item in plan.bindings if item.tensor_name == k_name)
        self.assertEqual(
            tuple((item.spans[0].start, item.spans[0].stop) for item in bindings),
            ((0, 256), (0, 256), (256, 512), (256, 512)),
        )

    def test_nearby_topologies_fail_instead_of_silently_changing_parallelism(self) -> None:
        config = parse_qwen3_5_moe_config(_fixture())
        headers = tuple(
            TensorHeader(name, item.dtype, item.shape)
            for name, item in _expected_headers(config).items()
        )
        weights = plan_qwen3_5_moe_weight_headers(config, headers)
        for tp, pp, ep, devices in (
            (4, 2, 1, tuple(range(8))),
            (2, 4, 2, tuple(range(8))),
        ):
            topology = ParallelTopology.single_node(
                topology_id=f"bad:{tp}:{pp}:{ep}",
                device_indices=devices,
                tp_size=tp,
                pp_size=pp,
                ep_size=ep,
            )
            with (
                self.subTest(topology=topology.topology_id),
                self.assertRaises(Qwen3_5MoeParallelError),
            ):
                plan_qwen3_5_moe_tp4_pp2_ep4(config, weights, topology)


if __name__ == "__main__":
    unittest.main()


def test_configured_layers_and_heads_cover_multiple_tensor_and_pipeline_sizes():
    raw = _fixture()
    text = raw["text_config"]
    text["num_hidden_layers"] = 12
    text["linear_num_key_heads"] = 8
    text["linear_num_value_heads"] = 16
    text["linear_key_head_dim"] = 64
    text["linear_value_head_dim"] = 32
    text["linear_conv_kernel_dim"] = 3
    text["head_dim"] = 128
    text["rope_parameters"]["partial_rotary_factor"] = 0.5
    if "partial_rotary_factor" in text:
        text["partial_rotary_factor"] = 0.5
    text["hidden_size"] = 1536
    text["mtp_num_hidden_layers"] = 0
    text["layer_types"] = [
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(12)
    ]
    raw["vision_config"]["out_hidden_size"] = 1536
    raw["vision_config"]["depth"] = 2
    text["num_experts"] = 16
    config = parse_qwen3_5_moe_config(raw)
    headers = [
        TensorHeader(name, item.dtype, item.shape)
        for name, item in _expected_headers(config).items()
    ]
    weights = plan_qwen3_5_moe_weight_headers(config, headers)
    for tp in (1, 2, 4, 8):
        for pp in (1, 2, 3):
            for ep in (size for size in range(1, tp + 1) if tp % size == 0):
                from cubic_serve.executor.parallel import seal_partition

                topology = ParallelTopology.single_node(
                    topology_id=f"generic:tp{tp}:pp{pp}:ep{ep}",
                    device_indices=tuple(range(tp * pp)),
                    tp_size=tp,
                    pp_size=pp,
                    ep_size=ep,
                )
                plan = plan_qwen3_5_moe_weights(config, weights, topology)
                seal_partition(topology, plan.partition)
                assert len(plan.partition.pipeline_edges) == tp * (pp - 1)
                assert {owner.layer_index for owner in plan.kv_heads} == {3, 7, 11}
                assert {owner.layer_index for owner in plan.gdn_states} == (
                    set(range(12)) - {3, 7, 11}
                )
                assert {
                    head
                    for owner in plan.kv_heads
                    for head in range(owner.head_span.start, owner.head_span.stop)
                } == set(range(text["num_key_value_heads"]))
                for binding in plan.bindings:
                    if binding.layout is MoeWeightLayout.EXPERT:
                        assert math.prod(binding.local_shape) * tp == math.prod(
                            binding.global_shape
                        )
