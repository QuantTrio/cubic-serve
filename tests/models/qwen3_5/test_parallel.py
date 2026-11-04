from __future__ import annotations

import gc
import json
import multiprocessing
import os
import queue
import time
import traceback
import unittest
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import cast

import torch

from cubic_serve.contracts.execution import ParallelTopology
from cubic_serve.models.checkpoint import CheckpointInventory, inspect_checkpoint
from cubic_serve.models.qwen3_5 import (
    AxisSpan,
    Qwen3_5Config,
    Qwen3_5ParallelWeightError,
    Qwen3_5ParallelWeightPlan,
    RankWeightBinding,
    WeightShardLayout,
    load_qwen3_5_rank_weights,
    parse_qwen3_5_config,
    parse_qwen3_5_config_json,
    plan_qwen3_5_checkpoint_weights,
    plan_qwen3_5_tp2_pp2_weights,
    plan_qwen3_5_weight_headers,
)
from cubic_serve.models.qwen3_5.weights import (
    Qwen3_5WeightPlan,
    _copy_binding,
    _expected_headers,
    plan_qwen3_5_dense_weights,
)
from cubic_serve.models.weight_schema import TensorHeader
from tests.models.qwen3_5.test_config import _fixture

_CHECKPOINT = Path("/pubdata/llm_models/Qwen/Qwen3.8-27B")
_RUN_REAL = os.environ.get("CUBIC_RUN_REAL_QWEN_SHARDS") == "1"


def _topology() -> ParallelTopology:
    return ParallelTopology.single_node(
        topology_id="qwen3.8-27b:gpu0-3:tp2:pp2",
        device_indices=(0, 1, 2, 3),
        tp_size=2,
        pp_size=2,
    )


def _tp4_pp1_topology() -> ParallelTopology:
    return ParallelTopology.single_node(
        topology_id="qwen3.8-27b:gpu0-3:tp4:pp1",
        device_indices=(0, 1, 2, 3),
        tp_size=4,
        pp_size=1,
    )


def _header_plan() -> tuple[Qwen3_5Config, Qwen3_5WeightPlan, Qwen3_5ParallelWeightPlan]:
    config = parse_qwen3_5_config(_fixture())
    headers = tuple(
        TensorHeader(name, contract.dtype, contract.shape)
        for name, contract in _expected_headers(config).items()
    )
    weight_plan = plan_qwen3_5_weight_headers(config, headers)
    return config, weight_plan, plan_qwen3_5_tp2_pp2_weights(config, weight_plan, _topology())


class _TrackingSlice:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor
        self.indexes: list[tuple[slice, ...]] = []

    def get_shape(self) -> list[int]:
        return list(self.tensor.shape)

    def __getitem__(self, index: tuple[slice, ...]) -> torch.Tensor:
        self.indexes.append(index)
        return self.tensor[index]


class _SliceOnlyReader:
    def __init__(self, name: str, tensor: torch.Tensor) -> None:
        self.name = name
        self.view = _TrackingSlice(tensor)

    def keys(self) -> list[str]:
        return [self.name]

    def get_slice(self, name: str) -> _TrackingSlice:
        if name != self.name:
            raise KeyError(name)
        return self.view


def _real_load_worker(
    rank: int,
    inventory: CheckpointInventory,
    weight_plan: Qwen3_5WeightPlan,
    parallel_plan: Qwen3_5ParallelWeightPlan,
    reports: multiprocessing.Queue[dict[str, object]],
) -> None:
    try:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
        free_before, total_memory = torch.cuda.mem_get_info(device)
        torch.cuda.reset_peak_memory_stats(device)

        weights = load_qwen3_5_rank_weights(
            inventory=inventory,
            weight_plan=weight_plan,
            parallel_plan=parallel_plan,
            rank=rank,
            device=device,
        )
        torch.cuda.synchronize(device)
        resident_allocated = torch.cuda.memory_allocated(device) - baseline_allocated
        resident_reserved = torch.cuda.memory_reserved(device) - baseline_reserved
        peak_allocated = torch.cuda.max_memory_allocated(device) - baseline_allocated
        free_loaded, _ = torch.cuda.mem_get_info(device)
        expected_bytes = sum(
            2 * torch.Size(binding.local_shape).numel()
            for binding in parallel_plan.bindings_for_rank(rank)
        )
        if weights.resident_bytes != expected_bytes or resident_allocated < expected_bytes:
            raise AssertionError(
                "rank logical/allocator bytes differ from its local contract: "
                f"logical={weights.resident_bytes}, expected={expected_bytes}, "
                f"allocated={resident_allocated}"
            )
        report: dict[str, object] = {
            "rank": rank,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "tensor_count": weights.tensor_count,
            "resident_bytes": weights.resident_bytes,
            "resident_allocated_bytes": resident_allocated,
            "allocator_overhead_bytes": resident_allocated - expected_bytes,
            "resident_reserved_bytes": resident_reserved,
            "peak_allocated_bytes": peak_allocated,
            "device_used_delta_bytes": free_before - free_loaded,
            "device_total_bytes": total_memory,
        }
        del weights
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        report["released_allocated_bytes"] = (
            torch.cuda.memory_allocated(device) - baseline_allocated
        )
        report["released_reserved_bytes"] = torch.cuda.memory_reserved(device) - baseline_reserved
        reports.put(report)
    except BaseException:
        reports.put({"rank": rank, "error": traceback.format_exc()})
        raise


class Qwen3_5ParallelWeightTests(unittest.TestCase):
    def test_dense_rank_bindings_pass_shared_executor_dtype_validation(self) -> None:
        from cubic_serve.executor.model_runner import (
            Qwen3_5ParallelPerformanceError,
            Qwen3_5TP2PP2DecodeBody,
        )
        from cubic_serve.models.qwen3_5.weights import Qwen3_5RankWeights

        config, weight_plan, _ = _header_plan()
        device = torch.device("meta")
        for topology in (_topology(), _tp4_pp1_topology()):
            plan = plan_qwen3_5_dense_weights(config, weight_plan, topology)
            for rank in range(4):
                with self.subTest(tp=topology.tp_size, rank=rank):
                    bindings = plan.bindings_for_rank(rank)
                    tensors = {
                        b.tensor_name: torch.empty(
                            b.local_shape, dtype=torch.bfloat16, device=device
                        )
                        for b in bindings
                    }
                    weights = Qwen3_5RankWeights(
                        rank=rank,
                        device=device,
                        inventory_digest="fixture",
                        bindings=bindings,
                        tensors=tensors,
                    )
                    stage = rank // topology.tp_size
                    Qwen3_5TP2PP2DecodeBody._validate_weights(config, plan, weights, stage)
                    first = bindings[0]
                    weights._tensors[first.tensor_name] = torch.empty(
                        first.local_shape, dtype=torch.float32, device=device
                    )
                    with self.assertRaisesRegex(Qwen3_5ParallelPerformanceError, "shard ABI"):
                        Qwen3_5TP2PP2DecodeBody._validate_weights(config, plan, weights, stage)

    def test_dense_tp4_pp1_contract_owns_all_layers_without_pipeline_edges(self) -> None:
        config, weight_plan, _ = _header_plan()
        topology = _tp4_pp1_topology()
        plan = plan_qwen3_5_dense_weights(config, weight_plan, topology)

        self.assertEqual([len(plan.bindings_for_rank(rank)) for rank in range(4)], [851] * 4)
        self.assertEqual(plan.partition.pipeline_edges, ())
        self.assertTrue(
            all(
                assignment.pipeline_stage == 0 and assignment.owner_ranks == (0, 1, 2, 3)
                for assignment in plan.partition.assignments
            )
        )

        embed = tuple(
            binding
            for binding in plan.bindings
            if binding.tensor_name == "model.language_model.embed_tokens.weight"
        )
        head = tuple(
            binding for binding in plan.bindings if binding.tensor_name == "lm_head.weight"
        )
        expected_vocab = tuple(
            (rank, (AxisSpan(rank * 62080, (rank + 1) * 62080),)) for rank in range(4)
        )
        self.assertEqual(tuple((item.rank, item.spans) for item in embed), expected_vocab)
        self.assertEqual(tuple((item.rank, item.spans) for item in head), expected_vocab)

        qkv_name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
        qkv = tuple(binding for binding in plan.bindings if binding.tensor_name == qkv_name)
        self.assertEqual(
            tuple(item.spans for item in qkv),
            (
                (AxisSpan(0, 512), AxisSpan(2048, 2560), AxisSpan(4096, 5632)),
                (AxisSpan(512, 1024), AxisSpan(2560, 3072), AxisSpan(5632, 7168)),
                (AxisSpan(1024, 1536), AxisSpan(3072, 3584), AxisSpan(7168, 8704)),
                (AxisSpan(1536, 2048), AxisSpan(3584, 4096), AxisSpan(8704, 10240)),
            ),
        )
        self.assertEqual([item.local_shape for item in qkv], [(2560, 5120)] * 4)
        self.assertEqual(len(plan.kv_heads), 16 * 4)
        self.assertEqual(len(plan.gdn_states), 48 * 4)
        state0 = next(
            owner for owner in plan.gdn_states if owner.layer_index == 0 and owner.rank == 0
        )
        self.assertEqual(state0.convolution_shape, (2560, 3))
        self.assertEqual(state0.recurrent_shape, (12, 128, 128))

        with self.assertRaisesRegex(Qwen3_5ParallelWeightError, "TP2 x PP2 planner"):
            plan_qwen3_5_tp2_pp2_weights(config, weight_plan, topology)

    def test_mtp_tp4_contract_is_explicit_and_does_not_enable_pp(self) -> None:
        config, weight_plan, _ = _header_plan()
        plan = plan_qwen3_5_dense_weights(
            config,
            weight_plan,
            _tp4_pp1_topology(),
            include_mtp=True,
        )

        self.assertTrue(plan.mtp_enabled)
        self.assertEqual(len(plan.global_names), 866)
        self.assertEqual([len(plan.bindings_for_rank(rank)) for rank in range(4)], [866] * 4)
        self.assertEqual(len(plan.mtp_kv_heads), 4)
        self.assertEqual(
            tuple(owner.head_span for owner in plan.mtp_kv_heads),
            tuple(AxisSpan(rank, rank + 1) for rank in range(4)),
        )
        fc = tuple(item for item in plan.bindings if item.tensor_name == "mtp.fc.weight")
        self.assertEqual([item.layout for item in fc], [WeightShardLayout.COLUMN] * 4)
        self.assertEqual([item.local_shape for item in fc], [(1280, 10240)] * 4)
        self.assertEqual(
            {
                item.layout
                for item in plan.bindings
                if item.tensor_name
                in {
                    "mtp.norm.weight",
                    "mtp.pre_fc_norm_embedding.weight",
                    "mtp.pre_fc_norm_hidden.weight",
                }
            },
            {WeightShardLayout.REPLICATED},
        )

        with self.assertRaisesRegex(Qwen3_5ParallelWeightError, "TP-only"):
            plan_qwen3_5_dense_weights(config, weight_plan, _topology(), include_mtp=True)

    def test_851_tensor_tp2_pp2_contract_is_locally_and_globally_closed(self) -> None:
        config, _, plan = _header_plan()
        self.assertEqual(len(plan.global_names), 851)
        self.assertEqual(
            [len(plan.bindings_for_rank(rank)) for rank in range(4)],
            [425, 425, 426, 426],
        )

        assignment = {item.node_id: item for item in plan.partition.assignments}
        self.assertEqual(assignment["text.token_embedding"].owner_ranks, (0, 1))
        self.assertEqual(assignment["text.final_norm"].owner_ranks, (2, 3))
        self.assertEqual(assignment["text.lm_head"].owner_ranks, (2, 3))
        self.assertEqual(
            {
                item.pipeline_stage
                for item in plan.partition.assignments
                if item.node_id.startswith("text.decoder.layer.")
                and int(item.node_id.rsplit(".", 1)[1]) < 32
            },
            {0},
        )
        self.assertEqual(
            {
                item.pipeline_stage
                for item in plan.partition.assignments
                if item.node_id.startswith("text.decoder.layer.")
                and int(item.node_id.rsplit(".", 1)[1]) >= 32
            },
            {1},
        )

        by_name: dict[str, list[RankWeightBinding]] = {}
        for binding in plan.bindings:
            by_name.setdefault(binding.tensor_name, []).append(binding)
        self.assertEqual(tuple(sorted(by_name)), plan.global_names)
        self.assertTrue(all(len(items) == 2 for items in by_name.values()))
        for name, items in by_name.items():
            if items[0].layout is WeightShardLayout.REPLICATED:
                self.assertTrue(all(item.local_shape == item.global_shape for item in items))
                continue
            spans = sorted(span for item in items for span in item.spans)
            self.assertEqual(spans[0].start, 0, name)
            self.assertEqual(spans[-1].stop, items[0].global_shape[items[0].shard_axis], name)
            self.assertTrue(
                all(left.stop == right.start for left, right in pairwise(spans)),
                name,
            )

        embed = by_name["model.language_model.embed_tokens.weight"]
        head = by_name["lm_head.weight"]
        self.assertEqual(
            [(item.rank, item.spans) for item in embed],
            [(0, (AxisSpan(0, 124160),)), (1, (AxisSpan(124160, 248320),))],
        )
        self.assertEqual([item.rank for item in head], [2, 3])

        layer = 0
        prefix = f"model.language_model.layers.{layer}.linear_attn"
        qkv = by_name[f"{prefix}.in_proj_qkv.weight"]
        self.assertEqual(
            [item.spans for item in qkv],
            [
                (AxisSpan(0, 1024), AxisSpan(2048, 3072), AxisSpan(4096, 7168)),
                (AxisSpan(1024, 2048), AxisSpan(3072, 4096), AxisSpan(7168, 10240)),
            ],
        )
        self.assertEqual([item.local_shape for item in qkv], [(5120, 5120)] * 2)
        self.assertEqual(
            {item.layout for item in by_name[f"{prefix}.out_proj.weight"]},
            {WeightShardLayout.ROW},
        )
        mlp = "model.language_model.layers.0.mlp"
        self.assertEqual(
            (
                by_name[f"{mlp}.gate_proj.weight"][0].shard_axis,
                by_name[f"{mlp}.down_proj.weight"][0].shard_axis,
            ),
            (0, 1),
        )

        self.assertEqual(len(plan.kv_heads), 16 * 2)
        self.assertEqual({owner.replica_ranks for owner in plan.kv_heads}, {(0,), (1,), (2,), (3,)})
        self.assertEqual(len(plan.gdn_states), 48 * 2)
        state0 = next(
            owner for owner in plan.gdn_states if owner.layer_index == 0 and owner.rank == 0
        )
        self.assertEqual(state0.convolution_shape, (5120, 3))
        self.assertEqual(state0.recurrent_shape, (24, 128, 128))
        self.assertIs(state0.convolution_dtype, torch.bfloat16)
        self.assertIs(state0.recurrent_dtype, torch.float32)
        self.assertEqual(config.graph.text.layer_count, 64)

    def test_tampered_weight_or_kv_coverage_fails_closed(self) -> None:
        _, _, plan = _header_plan()
        with self.assertRaisesRegex(Qwen3_5ParallelWeightError, "open|close"):
            replace(plan, bindings=plan.bindings[:-1])
        with self.assertRaisesRegex(Qwen3_5ParallelWeightError, "schedule"):
            replace(plan, kv_heads=())
        first = plan.kv_heads[0]
        with self.assertRaisesRegex(Qwen3_5ParallelWeightError, "replica"):
            replace(
                plan,
                kv_heads=(
                    replace(first, replica_ranks=(first.rank, first.rank + 1)),
                    *plan.kv_heads[1:],
                ),
            )
        k_name = "model.language_model.layers.3.self_attn.k_proj.weight"
        k_indexes = [
            index for index, binding in enumerate(plan.bindings) if binding.tensor_name == k_name
        ]
        swapped = list(plan.bindings)
        left, right = (plan.bindings[index] for index in k_indexes)
        swapped[k_indexes[0]] = replace(left, spans=right.spans)
        swapped[k_indexes[1]] = replace(right, spans=left.spans)
        with self.assertRaisesRegex(Qwen3_5ParallelWeightError, "weight and state"):
            replace(plan, bindings=tuple(swapped))

    def test_segmented_loader_uses_only_source_slices(self) -> None:
        source = torch.arange(24, dtype=torch.bfloat16).reshape(12, 2)
        reader = _SliceOnlyReader("gdn", source)
        binding = RankWeightBinding(
            tensor_name="gdn",
            node_id="text.decoder.layer.0000",
            rank=0,
            pipeline_stage=0,
            tensor_rank=0,
            layout=WeightShardLayout.GDN_QKV,
            shard_axis=0,
            spans=(AxisSpan(0, 2), AxisSpan(4, 6), AxisSpan(8, 10)),
            global_shape=(12, 2),
            local_shape=(6, 2),
        )
        loaded = _copy_binding(reader, binding, torch.device("cpu"))
        self.assertTrue(torch.equal(loaded, torch.cat((source[0:2], source[4:6], source[8:10]))))
        self.assertEqual(
            reader.view.indexes,
            [(slice(0, 2), slice(None)), (slice(4, 6), slice(None)), (slice(8, 10), slice(None))],
        )

    @unittest.skipUnless(
        _RUN_REAL and (_CHECKPOINT / "model.safetensors.index.json").is_file(),
        "real four-GPU Qwen shard load is opt-in",
    )
    def test_real_checkpoint_loads_rank_local_slices_on_four_gpus(self) -> None:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
            self.skipTest("four visible CUDA devices are required")
        config = parse_qwen3_5_config_json((_CHECKPOINT / "config.json").read_bytes())
        inventory = inspect_checkpoint(_CHECKPOINT)
        weight_plan = plan_qwen3_5_checkpoint_weights(config, inventory)
        parallel_plan = plan_qwen3_5_tp2_pp2_weights(config, weight_plan, _topology())

        context = multiprocessing.get_context("spawn")
        reports: multiprocessing.Queue[dict[str, object]] = context.Queue()
        workers = [
            context.Process(
                target=_real_load_worker,
                args=(rank, inventory, weight_plan, parallel_plan, reports),
            )
            for rank in range(4)
        ]
        deadline = time.monotonic() + 240
        try:
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(max(0.0, deadline - time.monotonic()))
            if any(worker.is_alive() for worker in workers):
                self.fail("four-rank Qwen weight load exceeded 240 seconds")
            self.assertEqual([worker.exitcode for worker in workers], [0, 0, 0, 0])
            received: list[dict[str, object]] = []
            for _ in workers:
                try:
                    received.append(reports.get(timeout=5))
                except queue.Empty as error:
                    raise AssertionError("a Qwen rank exited without a load report") from error
            received.sort(key=lambda item: cast(int, item["rank"]))
            errors = [item for item in received if "error" in item]
            self.assertEqual(errors, [])
            self.assertEqual([item["tensor_count"] for item in received], [425, 425, 426, 426])
            self.assertTrue(all(item["released_allocated_bytes"] == 0 for item in received))
            self.assertTrue(all(item["released_reserved_bytes"] == 0 for item in received))
            print("QWEN_TP_PP_LOAD_REPORT=" + json.dumps(received, sort_keys=True))
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                worker.join(timeout=5)
            reports.close()


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
    text["num_key_value_heads"] = 8
    config = parse_qwen3_5_config(raw)
    headers = [
        TensorHeader(name, item.dtype, item.shape)
        for name, item in _expected_headers(config).items()
    ]
    weights = plan_qwen3_5_weight_headers(config, headers)
    for tp in (1, 2, 4, 8):
        for pp in (1, 2, 3):
            topology = ParallelTopology.single_node(
                topology_id=f"generic:tp{tp}:pp{pp}",
                device_indices=tuple(range(tp * pp)),
                tp_size=tp,
                pp_size=pp,
                ep_size=1,
            )
            plan = plan_qwen3_5_dense_weights(config, weights, topology)
            assert len(plan.partition.pipeline_edges) == tp * (pp - 1)
            assert {owner.layer_index for owner in plan.kv_heads} == {3, 7, 11}
            assert {owner.layer_index for owner in plan.gdn_states} == set(range(12)) - {3, 7, 11}
            assert {
                head
                for owner in plan.kv_heads
                for head in range(owner.head_span.start, owner.head_span.stop)
            } == set(range(text["num_key_value_heads"]))
