from __future__ import annotations

import gc
import json
import os
import tempfile
import time
import traceback
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import torch
from torch.multiprocessing.spawn import spawn

from cubic_serve.contracts.execution import ParallelTopology
from cubic_serve.executor.distributed_reference import (
    Qwen3_5DistributedCache,
    Qwen3_5DistributedReferenceModel,
    _reconstruct_equal_shards,
    seal_qwen3_5_distributed_reference,
    seal_qwen3_5_target_mtp_distributed_reference,
    shard_qwen3_5_reference_cache,
)
from cubic_serve.executor.parallel import initialize_eager_nccl
from cubic_serve.executor.runner import ReferenceExecutionContext, configure_reference_math
from cubic_serve.guardrail.compare import (
    ComparisonPolicy,
    DifferenceClass,
    compare_tensors,
    stable_digest,
    tensor_digest,
)
from cubic_serve.models.checkpoint import inspect_checkpoint
from cubic_serve.models.qwen3_5 import (
    AxisSpan,
    Qwen3_5Config,
    Qwen3_5ParallelWeightPlan,
    load_qwen3_5_rank_weights,
    parse_qwen3_5_config,
    parse_qwen3_5_config_json,
    plan_qwen3_5_checkpoint_weights,
    plan_qwen3_5_tp2_pp2_weights,
    plan_qwen3_5_weight_headers,
)
from cubic_serve.models.qwen3_5.model import Qwen3_5LayerCache, materialize_qwen3_5_reference
from cubic_serve.models.qwen3_5.weights import _expected_headers, plan_qwen3_5_dense_weights
from cubic_serve.models.weight_schema import TensorHeader
from tests.models.qwen3_5.test_config import _fixture

_CHECKPOINT = Path("/pubdata/llm_models/Qwen/Qwen3.8-27B")
_RUN_REAL = os.environ.get("CUBIC_RUN_REAL_QWEN_TP_PP") == "1"
_PROMPT = (814, 20139)


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


def _header_plan() -> tuple[Qwen3_5Config, Qwen3_5ParallelWeightPlan]:
    config = parse_qwen3_5_config(_fixture())
    headers = tuple(
        TensorHeader(name, contract.dtype, contract.shape)
        for name, contract in _expected_headers(config).items()
    )
    weights = plan_qwen3_5_weight_headers(config, headers)
    return config, plan_qwen3_5_tp2_pp2_weights(config, weights, _topology())


def _cache_digest(cache: Qwen3_5DistributedCache) -> str:
    layers: list[dict[str, str | int]] = []
    for layer_index, layer in enumerate(cache.layers, start=cache.layer_start):
        payload: dict[str, str | int] = {"layer": layer_index}
        for name in ("convolution", "recurrent", "key", "value"):
            tensor = getattr(layer, name)
            if tensor is not None:
                payload[name] = tensor_digest(tensor)
        layers.append(payload)
    return stable_digest(
        {
            "sequence_length": cache.sequence_length,
            "batch_size": cache.batch_size,
            "rank": cache.rank,
            "layer_start": cache.layer_start,
            "layer_end": cache.layer_end,
            "layers": layers,
        }
    )


class Qwen3_5DistributedReferenceContractTests(unittest.TestCase):
    def test_tp4_pp1_recipe_has_one_ordered_tp_lane_and_no_pipeline(self) -> None:
        config, _ = _header_plan()
        headers = tuple(
            TensorHeader(name, contract.dtype, contract.shape)
            for name, contract in _expected_headers(config).items()
        )
        weights = plan_qwen3_5_weight_headers(config, headers)
        plan = plan_qwen3_5_dense_weights(config, weights, _tp4_pp1_topology())
        sealed = seal_qwen3_5_distributed_reference(config, plan)

        steps = sealed.partition.collectives
        self.assertEqual(len(steps), 770)
        self.assertEqual({step.group_id for step in steps}, {"tp:0"})
        self.assertEqual(tuple(step.sequence_no for step in steps), tuple(range(770)))
        self.assertEqual(
            steps[0].step_id,
            "qwen3_5.reference.weight:model.language_model.embed_tokens.weight",
        )
        self.assertEqual(steps[-1].step_id, "qwen3_5.reference.weight:lm_head.weight")
        self.assertEqual(sealed.partition.pipeline_edges, ())

    def test_mtp_tp4_recipe_is_a_distinct_complete_epoch(self) -> None:
        config, _ = _header_plan()
        headers = tuple(
            TensorHeader(name, contract.dtype, contract.shape)
            for name, contract in _expected_headers(config).items()
        )
        weights = plan_qwen3_5_weight_headers(config, headers)
        plan = plan_qwen3_5_dense_weights(
            config,
            weights,
            _tp4_pp1_topology(),
            include_mtp=True,
        )
        target = seal_qwen3_5_distributed_reference(config, plan)
        combined = seal_qwen3_5_target_mtp_distributed_reference(config, plan)

        self.assertNotEqual(target.partition_digest, combined.partition_digest)
        self.assertEqual(len(combined.partition.collectives), 782)
        self.assertEqual(
            {step.lane_id for step in combined.partition.collectives},
            {"qwen3_5-canonical-reconstruct-v1", "qwen3_5-mtp-canonical-reconstruct-v1"},
        )
        mtp_steps = tuple(
            step
            for step in combined.partition.collectives
            if step.lane_id == "qwen3_5-mtp-canonical-reconstruct-v1"
        )
        self.assertEqual(len(mtp_steps), 12)
        self.assertEqual(
            tuple(step.sequence_no for step in mtp_steps), tuple(range(12))
        )
        self.assertEqual(
            tuple(step.step_id for step in mtp_steps[:3]),
            (
                "qwen3_5.mtp.reference.state:0:key",
                "qwen3_5.mtp.reference.state:0:value",
                "qwen3_5.mtp.reference.weight:model.language_model.embed_tokens.weight",
            ),
        )
        self.assertEqual(
            {step.lane_id for step in mtp_steps},
            {"qwen3_5-mtp-canonical-reconstruct-v1"},
        )

    def test_pp1_forward_runs_first_and_last_endpoints_without_pipeline(self) -> None:
        config, _ = _header_plan()

        class _Weights:
            peak_operand_bytes = 0
            active = False

            def open(self) -> None:
                self.active = True

            def close(self) -> None:
                self.active = False

        class _Runtime:
            device = torch.device("cpu")
            pipeline_calls: list[str] = []

            @contextmanager
            def collective_epoch(self, _epoch_id: str):  # type: ignore[no-untyped-def]
                yield

            def pipeline_send(self, *_args: object) -> None:
                self.pipeline_calls.append("send")

            def pipeline_recv(self, *_args: object) -> torch.Tensor:
                self.pipeline_calls.append("recv")
                raise AssertionError("PP1 cannot receive a pipeline tensor")

        class _Canonical:
            calls: list[str] = []

            def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
                self.calls.append("embed")
                return torch.zeros(
                    (*input_ids.shape, config.graph.text.hidden_size),
                    dtype=torch.bfloat16,
                )

            def forward_layer(  # type: ignore[no-untyped-def]
                self, hidden: torch.Tensor, **_kwargs: object
            ):
                self.calls.append("layer")
                return hidden, Qwen3_5LayerCache()

            def normalize_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
                self.calls.append("normalize")
                return hidden

            def project_normalized_logits(self, hidden: torch.Tensor) -> torch.Tensor:
                self.calls.append("logits")
                return torch.zeros((*hidden.shape[:2], 4), dtype=torch.bfloat16)

        model = Qwen3_5DistributedReferenceModel.__new__(Qwen3_5DistributedReferenceModel)
        runtime = _Runtime()
        weights = _Weights()
        canonical = _Canonical()
        model.config = config
        model.parallel_plan = cast(Any, type("_Plan", (), {"mtp_enabled": False})())
        model.runtime = cast(Any, runtime)
        model.rank = 0
        model.pipeline_stage = 0
        model.tensor_rank = 0
        model.is_first_stage = True
        model.is_last_stage = True
        model.layer_start = 0
        model.layer_end = 1
        model._weights = cast(Any, weights)
        model._canonical = cast(Any, canonical)
        model._validate_cache = lambda _cache, _batch_size: None
        model._restore_layer_cache = lambda _layer, previous: previous
        model._slice_layer_cache = lambda _layer, state: state
        cache = Qwen3_5DistributedCache(
            sequence_length=0,
            batch_size=1,
            rank=0,
            layer_start=0,
            layer_end=1,
            layers=(Qwen3_5LayerCache(),),
        )

        with (
            patch("cubic_serve.executor.distributed_reference.require_reference_execution"),
            patch(
                "cubic_serve.executor.distributed_reference.interleaved_mrope_cos_sin",
                return_value=(torch.empty(0), torch.empty(0)),
            ),
        ):
            step = model.forward(
                torch.tensor([[1]], dtype=torch.int64),
                cache=cache,
                epoch_id="tp4-pp1",
            )

        self.assertEqual(canonical.calls, ["embed", "layer", "normalize", "logits"])
        self.assertEqual(runtime.pipeline_calls, [])
        self.assertIsNotNone(step.logits)
        self.assertFalse(weights.active)

    def test_recipe_is_complete_ordered_and_separate_from_performance_tp(self) -> None:
        config, plan = _header_plan()
        sealed = seal_qwen3_5_distributed_reference(config, plan)
        by_group = {
            group: tuple(item for item in sealed.partition.collectives if item.group_id == group)
            for group in ("tp:0", "tp:1")
        }
        self.assertEqual([len(by_group[group]) for group in ("tp:0", "tp:1")], [385, 385])
        for group, steps in by_group.items():
            self.assertEqual(tuple(step.sequence_no for step in steps), tuple(range(385)))
            self.assertEqual({step.operation for step in steps}, {"all_gather"})
            self.assertEqual({step.lane_id for step in steps}, {"qwen3_5-canonical-reconstruct-v1"})
            self.assertTrue(all(step.group_id == group for step in steps))
        self.assertEqual(
            by_group["tp:0"][0].step_id,
            "qwen3_5.reference.weight:model.language_model.embed_tokens.weight",
        )
        self.assertEqual(by_group["tp:1"][-1].step_id, "qwen3_5.reference.weight:lm_head.weight")
        self.assertEqual(
            tuple(edge.edge_id for edge in sealed.partition.pipeline_edges),
            ("qwen3_5.pp:lane:0", "qwen3_5.pp:lane:1"),
        )
        self.assertIn("eager-reference-reconstruct", sealed.partition.partition_id)

    def test_segmented_gdn_gather_restores_global_semantic_order(self) -> None:
        source = torch.arange(24, dtype=torch.bfloat16).reshape(12, 2)
        rank_spans = (
            (AxisSpan(0, 2), AxisSpan(4, 6), AxisSpan(8, 10)),
            (AxisSpan(2, 4), AxisSpan(6, 8), AxisSpan(10, 12)),
        )
        rank_chunks = tuple(
            torch.cat(tuple(source[span.start : span.stop] for span in spans), dim=0)
            for spans in rank_spans
        )
        gathered = torch.cat(rank_chunks, dim=0)
        actual = _reconstruct_equal_shards(
            gathered,
            axis=0,
            local_size=6,
            global_size=12,
            spans_by_rank=rank_spans,
        )
        self.assertTrue(torch.equal(actual, source))


def _oracle_payload(path: str) -> None:
    config = parse_qwen3_5_config_json((_CHECKPOINT / "config.json").read_bytes())
    inventory = inspect_checkpoint(_CHECKPOINT)
    weight_plan = plan_qwen3_5_checkpoint_weights(config, inventory)
    parallel_plan = plan_qwen3_5_tp2_pp2_weights(config, weight_plan, _topology())
    with ReferenceExecutionContext():
        model = materialize_qwen3_5_reference(
            config=config,
            inventory=inventory,
            plan=weight_plan,
            device="cuda:0",
        )
        prompt = torch.tensor([_PROMPT], dtype=torch.int64, device="cuda:0")
        prefill = model.forward(prompt)
        decode_token = int(torch.argmax(prefill.logits[0, -1]).item())
        decode = model.forward(
            torch.tensor([[decode_token]], dtype=torch.int64, device="cuda:0"),
            cache=prefill.cache,
        )
        cache_digests = tuple(
            _cache_digest(shard_qwen3_5_reference_cache(decode.cache, parallel_plan, rank=rank))
            for rank in range(4)
        )
        payload = {
            "prefill_logits": prefill.logits.detach().cpu(),
            "decode_logits": decode.logits.detach().cpu(),
            "decode_token": decode_token,
            "cache_digests": cache_digests,
        }
    del model, prefill, decode, prompt
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)
    torch.save(payload, path)


def _distributed_worker(
    rank: int,
    init_method: str,
    oracle_path: str,
    report_directory: str,
) -> None:
    report_path = Path(report_directory) / f"rank-{rank}.json"
    try:
        configure_reference_math()
        oracle = torch.load(oracle_path, map_location="cpu", weights_only=True)
        config = parse_qwen3_5_config_json((_CHECKPOINT / "config.json").read_bytes())
        inventory = inspect_checkpoint(_CHECKPOINT)
        weight_plan = plan_qwen3_5_checkpoint_weights(config, inventory)
        parallel_plan = plan_qwen3_5_tp2_pp2_weights(config, weight_plan, _topology())
        sealed = seal_qwen3_5_distributed_reference(config, parallel_plan)

        with initialize_eager_nccl(
            sealed,
            rank=rank,
            init_method=init_method,
            timeout_seconds=180,
        ) as runtime:
            rank_weights = load_qwen3_5_rank_weights(
                inventory=inventory,
                weight_plan=weight_plan,
                parallel_plan=parallel_plan,
                rank=rank,
                device=runtime.device,
            )
            resident_bytes = rank_weights.resident_bytes
            torch.cuda.reset_peak_memory_stats(runtime.device)
            model = Qwen3_5DistributedReferenceModel(
                config=config,
                weights=rank_weights,
                parallel_plan=parallel_plan,
                runtime=runtime,
            )
            with ReferenceExecutionContext():
                prompt = torch.tensor([_PROMPT], dtype=torch.int64, device=runtime.device)
                prefill = model.forward(prompt, epoch_id="prefill")
                decode = model.forward(
                    torch.tensor(
                        [[cast(int, oracle["decode_token"])]],
                        dtype=torch.int64,
                        device=runtime.device,
                    ),
                    cache=prefill.cache,
                    epoch_id="decode-1",
                )

            comparisons: dict[str, bool] = {}
            if runtime.sealed.topology.placements[rank].pipeline_stage == 1:
                assert prefill.logits is not None and decode.logits is not None
                for name, expected, actual in (
                    ("prefill", oracle["prefill_logits"], prefill.logits),
                    ("decode", oracle["decode_logits"], decode.logits),
                ):
                    result = compare_tensors(
                        cast(torch.Tensor, expected).to(runtime.device),
                        actual,
                        ComparisonPolicy(DifferenceClass.EXACT),
                    )
                    if not result.passed:
                        raise AssertionError(f"{name} logits: {result.reason}")
                    comparisons[name] = True
                comparisons["token"] = int(torch.argmax(prefill.logits[0, -1]).item()) == cast(
                    int, oracle["decode_token"]
                )
                if not comparisons["token"]:
                    raise AssertionError("distributed prefill token differs from oracle")
            else:
                if prefill.logits is not None or decode.logits is not None:
                    raise AssertionError("non-final PP stage produced logits")

            actual_cache_digest = _cache_digest(decode.cache)
            expected_cache_digest = cast(tuple[str, ...], oracle["cache_digests"])[rank]
            if actual_cache_digest != expected_cache_digest:
                raise AssertionError("rank-local hybrid cache differs from canonical oracle")
            report = {
                "rank": rank,
                "pipeline_stage": model.pipeline_stage,
                "tensor_rank": model.tensor_rank,
                "resident_bytes": resident_bytes,
                "max_reconstructed_operand_bytes": max(
                    prefill.max_reconstructed_operand_bytes,
                    decode.max_reconstructed_operand_bytes,
                ),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(runtime.device),
                "cache_digest": actual_cache_digest,
                "logit_comparisons": comparisons,
            }
        report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    except BaseException:
        report_path.write_text(
            json.dumps({"rank": rank, "error": traceback.format_exc()}, sort_keys=True),
            encoding="utf-8",
        )
        raise


def _spawn_with_timeout(
    init_method: str,
    oracle_path: str,
    report_directory: str,
    *,
    timeout: float,
) -> None:
    spawn_processes = cast(Any, spawn)
    context = spawn_processes(
        _distributed_worker,
        args=(init_method, oracle_path, report_directory),
        nprocs=4,
        join=False,
    )
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"real Qwen TP2 x PP2 reference exceeded {timeout:.0f}s")
            if context.join(timeout=remaining, grace_period=10):
                return
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()


@unittest.skipUnless(
    _RUN_REAL and (_CHECKPOINT / "model.safetensors.index.json").is_file(),
    "real Qwen TP2 x PP2 reference test is opt-in",
)
class Qwen3_5RealDistributedReferenceTests(unittest.TestCase):
    def test_short_prefill_decode_matches_single_rank_logits_token_and_state_exactly(self) -> None:
        if not torch.distributed.is_nccl_available() or torch.cuda.device_count() < 4:
            self.skipTest("four CUDA devices with NCCL are required")
        with tempfile.TemporaryDirectory(prefix="cubic-qwen-tp2-pp2-") as directory:
            oracle_path = str(Path(directory) / "oracle.pt")
            _oracle_payload(oracle_path)
            init_method = (Path(directory) / "store").resolve().as_uri()
            _spawn_with_timeout(
                init_method,
                oracle_path,
                directory,
                timeout=900,
            )
            reports = tuple(
                json.loads((Path(directory) / f"rank-{rank}.json").read_text("utf-8"))
                for rank in range(4)
            )

        errors = tuple(report for report in reports if "error" in report)
        self.assertEqual(errors, ())
        self.assertEqual(tuple(report["rank"] for report in reports), (0, 1, 2, 3))
        self.assertEqual(
            tuple((report["pipeline_stage"], report["tensor_rank"]) for report in reports),
            ((0, 0), (0, 1), (1, 0), (1, 1)),
        )
        self.assertTrue(
            all(
                report["max_reconstructed_operand_bytes"] < report["resident_bytes"]
                for report in reports
            )
        )
        for rank in (2, 3):
            self.assertEqual(
                reports[rank]["logit_comparisons"],
                {"decode": True, "prefill": True, "token": True},
            )
        print("QWEN_TP2_PP2_REFERENCE_REPORT=" + json.dumps(reports, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
