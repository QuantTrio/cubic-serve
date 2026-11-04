"""Qualify every native TP2 x PP2 Qwen3.8 BF16 linear occurrence at M1.

The observer exposes real eager operands.  Each rank verifies deterministic
output-column samples over complete K; row-parallel pairs additionally receive
one verifier-issued witness for the fixed local-GEMM -> TP2-sum hierarchy.
Missing occurrences, ABI bindings, samples, or witnesses fail closed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
from linear_witness import _indices
from tp2_pp2_audit import (
    _Admission,
    _admit,
    _build_body,
    _csv_ints,
    _emit,
    _load_performance_api,
    _prepare,
    _read_reports,
    _spawn_with_timeout,
)
from torch import Tensor

from cubic_serve.executor.parallel import initialize_eager_nccl
from cubic_serve.executor.runner import configure_reference_math
from cubic_serve.guardrail.compare import (
    ComparisonContext,
    ComparisonPolicy,
    DifferenceClass,
    PartitionedReductionProof,
    compare_tensors,
    stable_digest,
    tensor_digest,
)
from cubic_serve.guardrail.oracle import (
    PartitionedReductionWitnessMaterial,
    partitioned_reduction_material_identity,
    verify_observed_bf16_linear_sample,
    verify_partitioned_reduction_witness,
)
from cubic_serve.models.qwen3_5 import load_qwen3_5_rank_weights
from cubic_serve.models.qwen3_5.model import AttentionKind, Qwen3_5Config
from cubic_serve.models.qwen3_5.weights import RankWeightBinding, WeightShardLayout
from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

_CHECKPOINT = Path("/pubdata/llm_models/Qwen/Qwen3.8-27B")
_GIB = 1 << 30


class _DeferredObserver:
    """Retain immutable operands; evaluate reference math after the full forward."""

    def __init__(self, consume: Callable[[Any, int], None]) -> None:
        self._consume = consume
        self._count = 0
        self._first_error: tuple[int, Exception] | None = None
        self._events: list[Any] = []

    @property
    def count(self) -> int:
        return self._count

    def __call__(self, event: Any) -> None:
        ordinal = self._count
        self._count += 1
        if self._first_error is not None:
            return
        try:
            # Checkpoint weights are sealed read-only. Activations and partials
            # are reused during the forward and need stream-ordered snapshots.
            self._events.append(dataclasses.replace(
                event,
                linear_input=event.linear_input.clone(),
                local_output=event.local_output.clone(),
                tp_partial=None if event.tp_partial is None else event.tp_partial.clone(),
                tp_reduced=None if event.tp_reduced is None else event.tp_reduced.clone(),
            ))
        except Exception as error:
            self._first_error = (ordinal, error)

    def finish(self, expected_count: int) -> None:
        if self._first_error is not None:
            ordinal, error = self._first_error
            raise RuntimeError(
                "linear observer failed at occurrence "
                f"{ordinal}; failure deferred through {self._count} callbacks"
            ) from error
        if self._count != expected_count:
            raise RuntimeError(
                f"observer emitted {self._count} occurrences; expected {expected_count}"
            )
        for ordinal, event in enumerate(self._events):
            self._consume(event, ordinal)
        self._events.clear()

    @property
    def schedule(self) -> tuple[tuple[int | None, str, bool], ...]:
        return tuple(
            (event.global_layer, event.weight_name, event.tp_reduced is not None)
            for event in self._events
        )


def _expected_weights(
    config: Qwen3_5Config, stage: int, pp_size: int = 2
) -> tuple[str, ...]:
    layers_per_stage, remainder = divmod(config.graph.text.layer_count, pp_size)
    if remainder:
        raise RuntimeError("leaf audit requires an even PP layer partition")
    result: list[str] = []
    for layer in range(stage * layers_per_stage, (stage + 1) * layers_per_stage):
        prefix = f"model.language_model.layers.{layer}"
        if config.graph.text.layer_schedule[layer] is AttentionKind.GATED_DELTA_NET:
            op = f"{prefix}.linear_attn"
            result.extend(
                f"{op}.{name}.weight"
                for name in ("in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj")
            )
        else:
            op = f"{prefix}.self_attn"
            result.extend(
                f"{op}.{name}.weight"
                for name in ("q_proj", "k_proj", "v_proj", "o_proj")
            )
        result.extend(
            f"{prefix}.mlp.{name}.weight" for name in ("gate_proj", "up_proj", "down_proj")
        )
    if stage == pp_size - 1:
        result.append("lm_head.weight")
    return tuple(result)


def _global_columns(
    spans: tuple[tuple[int, int], ...], local_columns: tuple[int, ...]
) -> tuple[int, ...]:
    mapped: list[int] = []
    for column in local_columns:
        offset = column
        for start, stop in spans:
            size = stop - start
            if offset < size:
                mapped.append(start + offset)
                break
            offset -= size
        else:
            raise ValueError("sampled local column lies outside its TP spans")
    return tuple(mapped)


def _binding_payload(binding: RankWeightBinding) -> dict[str, object]:
    return {
        "weight_name": binding.tensor_name,
        "global_shape": binding.global_shape,
        "local_shape": binding.local_shape,
        "dtype": "bfloat16",
        "layout": binding.layout.value,
        "shard_axis": binding.shard_axis,
        "spans": tuple((span.start, span.stop) for span in binding.spans),
        "rank": binding.rank,
        "stage": binding.pipeline_stage,
        "tensor_rank": binding.tensor_rank,
    }


def _witness_payload(witness: Any) -> dict[str, object]:
    result = {
        field.name: getattr(witness, field.name)
        for field in dataclasses.fields(witness)
        if not field.name.startswith("_")
    }
    result["witness_digest"] = cast(Any, witness).witness_digest
    return result


def _record_event(
    event: Any,
    *,
    ordinal: int,
    expected_name: str,
    binding: RankWeightBinding,
    inventory_digest: str,
    semantic_plan_digest: str,
    sample_columns: int,
) -> dict[str, object]:
    payload = _binding_payload(binding)
    row = binding.layout is WeightShardLayout.ROW
    expected_layer = (
        None
        if expected_name == "lm_head.weight"
        else int(expected_name.split(".layers.", 1)[1].split(".", 1)[0])
    )
    tp_present = event.tp_partial is not None and event.tp_reduced is not None
    if (
        event.weight_name != expected_name
        or event.weight_name != binding.tensor_name
        or event.global_layer != expected_layer
        or tuple(event.weight_shape) != binding.global_shape
        or tuple(event.local_weight.shape) != binding.local_shape
        or event.local_weight.dtype is not torch.bfloat16
        or event.linear_input.dtype is not torch.bfloat16
        or event.local_output.dtype is not torch.bfloat16
        or event.linear_input.ndim != 2
        or event.local_output.ndim != 2
        or event.linear_input.shape[1] != binding.local_shape[1]
        or event.local_output.shape[1] != binding.local_shape[0]
        or binding.layout is WeightShardLayout.REPLICATED
        or (event.tp_partial is None) != (event.tp_reduced is None)
        or tp_present != row
    ):
        raise RuntimeError(f"linear occurrence {ordinal} differs from rank weight ABI")
    if row and (
        event.tp_partial.shape != event.local_output.shape
        or event.tp_reduced.shape != event.local_output.shape
        or event.tp_partial.dtype is not torch.bfloat16
        or event.tp_reduced.dtype is not torch.bfloat16
        or event.tp_reduced.data_ptr() == event.local_output.data_ptr()
    ):
        raise RuntimeError(f"row-linear occurrence {ordinal} aliases or changes its TP ABI")
    columns = _indices(event.local_output.shape[1], sample_columns)
    spans = cast(tuple[tuple[int, int], ...], payload["spans"])
    global_columns = (
        columns if row else _global_columns(spans, columns)
    )
    operand_binding = stable_digest(
        {
            "schema": "cubic.qwen3_5.tp2pp2.linear-leaf.v1",
            "inventory_digest": inventory_digest,
            "semantic_plan_digest": semantic_plan_digest,
            "ordinal": ordinal,
            **payload,
        }
    )
    local_witness = verify_observed_bf16_linear_sample(
        x=event.linear_input,
        weight=event.local_weight,
        output=event.local_output,
        input_rows=(0,),
        output_columns=columns,
        operand_binding_digest=operand_binding,
        artifact_digest=event.artifact_digest,
        launch_specialization_digest=event.launch_specialization_digest,
    )
    record: dict[str, object] = {
        "ordinal": ordinal,
        "global_layer": event.global_layer,
        **payload,
        "artifact_digest": event.artifact_digest,
        "launch_specialization_digest": event.launch_specialization_digest,
        "operand_binding_digest": operand_binding,
        "input_digest": tensor_digest(event.linear_input[0]),
        "sample_local_columns": columns,
        "sample_global_columns": global_columns,
        "local_k_witness": _witness_payload(local_witness),
        "row_parallel": row,
    }
    if row:
        record.update(
            {
                "row_x": event.linear_input[0].detach().to(device="cpu", copy=True),
                "row_weight": event.local_weight[list(columns)].detach().to(
                    device="cpu", copy=True
                ),
                "tp_partial": event.tp_partial[0, list(columns)].detach().to(
                    device="cpu", copy=True
                ),
                "tp_reduced": event.tp_reduced[0, list(columns)].detach().to(
                    device="cpu", copy=True
                ),
            }
        )
    return record


def _exact_bf16_dot(left: Tensor, right: Tensor) -> Tensor:
    products = left.to(torch.float64) * right.to(torch.float64)
    return torch.tensor(
        [math.fsum(row) for row in products.tolist()], dtype=torch.float64
    ).to(torch.bfloat16)


def _verify_tp4_row_reduction(
    expected: Tensor,
    actual: Tensor,
    material: PartitionedReductionWitnessMaterial,
    context: ComparisonContext,
) -> str:
    """Bind four complete K partitions and conservatively enclose their BF16 sum."""

    left = material.left_operands.to(torch.float64)
    right = material.right_operands.to(torch.float64)
    products = left * right
    partitions = material.partition_ids.to(torch.int64)
    if set(torch.unique(partitions).tolist()) != {0, 1, 2, 3}:
        raise RuntimeError("TP4 row witness does not contain four K partitions")
    exact = products.sum(dim=-1)
    error = torch.zeros_like(exact)
    partial_abs = torch.zeros_like(exact)
    fp32_u = 2.0**-24
    bf16_u = 2.0**-8
    for partition in range(4):
        selected = torch.where(partitions == partition, products, 0.0)
        count = (partitions == partition).sum(dim=-1)
        if bool((count == 0).any().item()):
            raise RuntimeError("TP4 row witness has an empty K partition")
        shard_exact = selected.sum(dim=-1)
        shard_abs = selected.abs().sum(dim=-1)
        steps = count.to(torch.float64) - 1.0
        local = shard_abs * (steps * fp32_u / (1.0 - steps * fp32_u))
        cast_bound = (shard_exact.abs() + local) * bf16_u / (1.0 - bf16_u)
        error += local + cast_bound
        partial_abs += shard_exact.abs() + local + cast_bound
    collective = partial_abs * (3.0 * fp32_u / (1.0 - 3.0 * fp32_u))
    output_cast = (partial_abs + collective) * bf16_u / (1.0 - bf16_u)
    bound = error + collective + output_cast
    if bool(((actual.to(torch.float64) - exact).abs() > bound).any().item()):
        raise RuntimeError("TP4 row output exceeds the partitioned reduction envelope")
    if not torch.equal(expected.view(torch.uint8), exact.to(torch.bfloat16).view(torch.uint8)):
        raise RuntimeError("TP4 exact oracle differs from complete K operands")
    digest: str = stable_digest(
        {
            "schema": "cubic.qwen3_5.tp4.row-reduction-witness.v1",
            "context_digest": context.digest,
            "operands_digest": stable_digest(
                {"left": tensor_digest(material.left_operands),
                 "right": tensor_digest(material.right_operands)}
            ),
            "indices_digest": tensor_digest(material.logical_indices),
            "partitions_digest": tensor_digest(material.partition_ids),
            "expected_digest": tensor_digest(expected),
            "actual_digest": tensor_digest(actual),
            "bound_digest": tensor_digest(bound),
        }
    )
    return digest


def _row_witness(*rank_records: dict[str, object]) -> dict[str, object]:
    if len(rank_records) not in (2, 4):
        raise RuntimeError("row witness requires a complete TP2 or TP4 partition set")
    left = rank_records[0]
    keys = (
        "ordinal", "global_layer", "weight_name", "global_shape", "layout",
        "sample_global_columns",
    )
    if any(record[key] != left[key] for record in rank_records[1:] for key in keys):
        raise RuntimeError("TP peers disagree on row-linear occurrence identity")
    if any(
        record["sample_local_columns"] != left["sample_local_columns"]
        for record in rank_records[1:]
    ):
        raise RuntimeError("TP peers sampled different row-linear output columns")
    reduced = cast(Tensor, left["tp_reduced"])
    if any(
        not torch.equal(
            reduced.view(torch.uint8),
            cast(Tensor, record["tp_reduced"]).view(torch.uint8),
        )
        for record in rank_records[1:]
    ):
        raise RuntimeError("TP peers received different row-linear reduced outputs")
    factors_left: list[Tensor] = []
    factors_right: list[Tensor] = []
    indices: list[Tensor] = []
    partitions: list[Tensor] = []
    for partition, record in enumerate(rank_records):
        x = cast(Tensor, record["row_x"])
        weight = cast(Tensor, record["row_weight"])
        factors_left.append(x.expand(weight.shape[0], -1))
        factors_right.append(weight)
        ((start, stop),) = cast(tuple[tuple[int, int], ...], record["spans"])
        indices.append(torch.arange(start, stop, dtype=torch.int64).expand(weight.shape[0], -1))
        partitions.append(torch.full_like(indices[-1], partition))
    material = PartitionedReductionWitnessMaterial(
        torch.cat(factors_left, dim=1).contiguous(),
        torch.cat(factors_right, dim=1).contiguous(),
        torch.cat(indices, dim=1).contiguous(),
        torch.cat(partitions, dim=1).contiguous(),
    )
    reduction_length = material.left_operands.shape[-1]
    context = ComparisonContext(
        stable_digest(
            {
                "schema": "cubic.qwen3_5.tp2pp2.row-linear.v1",
                "weight_name": left["weight_name"],
                "global_shape": left["global_shape"],
                "sample_columns": left["sample_global_columns"],
                "operand_bindings": tuple(
                    record["operand_binding_digest"] for record in rank_records
                ),
                "local_k_witnesses": tuple(
                    cast(dict[str, object], record["local_k_witness"])["witness_digest"]
                    for record in rank_records
                ),
                "artifacts": tuple(record["artifact_digest"] for record in rank_records),
            }
        ),
        stable_digest(
            {
                "rank_launches": tuple(
                    record["launch_specialization_digest"] for record in rank_records
                ),
                "collective": f"nccl-all-reduce-sum-bf16-tp{len(rank_records)}",
            }
        ),
    )
    expected = _exact_bf16_dot(material.left_operands, material.right_operands)
    if len(rank_records) == 2:
        identity = partitioned_reduction_material_identity(material)
        policy = ComparisonPolicy(
            DifferenceClass.PARTITIONED_REDUCTION,
            partitioned_reduction=PartitionedReductionProof(
                identity.operands_digest,
                identity.indices_digest,
                identity.partitions_digest,
                identity.reduction_length,
            ),
        )
        witness = verify_partitioned_reduction_witness(
            expected, reduced, policy, context, material
        )
        comparison = compare_tensors(
            expected, reduced, policy, context=context, witness=witness
        )
        if not comparison.passed:
            raise RuntimeError(f"partitioned row witness rejected: {comparison.reason}")
        witness_digest = witness.witness_digest
    else:
        witness_digest = _verify_tp4_row_reduction(
            expected, reduced, material, context
        )
    return {
        "weight_name": left["weight_name"],
        "global_layer": left["global_layer"],
        "sample_global_columns": left["sample_global_columns"],
        "partition_count": len(rank_records),
        "reduction_length": reduction_length,
        "witness_digest": witness_digest,
        "context_digest": context.digest,
    }


def _merge(
    records: list[list[dict[str, object]]],
    expected_counts: tuple[int, int, int, int],
    *,
    tp_size: int = 2,
    pp_size: int = 2,
    layer_count: int = 64,
) -> dict[str, object]:
    if len(records) != 4:
        raise RuntimeError("leaf audit requires exactly four rank reports")
    if (tp_size, pp_size) not in ((2, 2), (4, 1)):
        raise RuntimeError("leaf audit supports only TP2 x PP2 or TP4 x PP1")
    observed_counts = tuple(len(items) for items in records)
    if observed_counts != expected_counts:
        raise RuntimeError("rank-local observer coverage differs from its exact program")
    row_witnesses: list[dict[str, object]] = []
    paired_occurrences = 0
    for stage in range(pp_size):
        peers = records[stage * tp_size : (stage + 1) * tp_size]
        if any(len(peer) != len(peers[0]) for peer in peers[1:]):
            raise RuntimeError("TP peers observed different linear occurrence counts")
        for occurrence in zip(*peers, strict=True):
            first = occurrence[0]
            paired_occurrences += 1
            if any(item["weight_name"] != first["weight_name"] for item in occurrence[1:]):
                raise RuntimeError("TP peers observed a different linear order")
            if any(
                bool(item["row_parallel"]) != bool(first["row_parallel"])
                for item in occurrence[1:]
            ):
                raise RuntimeError("TP peers disagree on linear layout")
            if first["row_parallel"]:
                row_witnesses.append(_row_witness(*occurrence))
            elif any(item["input_digest"] != first["input_digest"] for item in occurrence[1:]):
                raise RuntimeError("column/local linear input is not TP-replicated exact")
    local_count = sum(len(items) for items in records)
    local_witness_count = sum(
        1 for items in records for item in items if item.get("local_k_witness")
    )
    local_witness_digests = [
        cast(dict[str, object], item["local_k_witness"])["witness_digest"]
        for items in records
        for item in items
    ]
    sampled_elements = sum(
        cast(int, cast(dict[str, object], item["local_k_witness"])["sampled_element_count"])
        for items in records
        for item in items
    )
    row_counts = tuple(
        sum(bool(item["row_parallel"]) for item in items) for items in records
    )
    gaps = []
    if local_count != local_witness_count:
        gaps.append("rank-local-k-witness")
    rows_per_rank = 2 * (layer_count // pp_size)
    expected_row_counts: tuple[int, int, int, int] = (rows_per_rank,) * 4
    if layer_count % pp_size or row_counts != expected_row_counts:
        gaps.append("row-occurrence-count")
    if paired_occurrences == 0:
        gaps.append("tp-pairing")
    passed = (
        local_count == local_witness_count
        and paired_occurrences > 0
        and row_counts == expected_row_counts
    )
    return {
        "status": "pass" if passed else "fail",
        "expected_rank_occurrences": expected_counts,
        "observed_rank_occurrences": observed_counts,
        "expected_row_occurrences_per_rank": expected_row_counts,
        "observed_row_occurrences_per_rank": row_counts,
        "rank_local_occurrences": local_count,
        "rank_local_k_witnesses": local_witness_count,
        "tp_paired_occurrences": paired_occurrences,
        "partitioned_row_witnesses": len(row_witnesses),
        "sampled_output_elements": sampled_elements,
        "local_witness_set_digest": stable_digest(local_witness_digests),
        "row_witness_set_digest": stable_digest(
            [item["witness_digest"] for item in row_witnesses]
        ),
        "coverage_gaps": gaps,
    }


def _worker(
    rank: int,
    init_method: str,
    report_directory: str,
    admission: _Admission,
    sample_columns: int,
    operator_config: OperatorConfig,
    operator_model_id: str,
    recurrent_state_dtype: str,
    captured_m: int,
) -> None:
    report_path = Path(report_directory) / f"rank-{rank}.json"
    tensor_path = Path(report_directory) / f"rank-{rank}.pt"
    try:
        api = _load_performance_api()
        # Hundreds of tiny sampled [1,K] x [8,K] CPU oracles are faster and
        # substantially less intrusive with one verifier thread per rank.
        torch.set_num_threads(1)
        configure_reference_math()
        with initialize_eager_nccl(
            admission.sealed, rank=rank, init_method=init_method, timeout_seconds=1800
        ) as runtime, operator_strategy_scope(
            operator_config, device=torch.cuda.get_device_name(runtime.device),
            model=operator_model_id,
        ):
            weights = load_qwen3_5_rank_weights(
                inventory=admission.inventory,
                weight_plan=admission.weight_plan,
                parallel_plan=admission.parallel_plan,
                rank=rank,
                device=runtime.device,
            )
            body, resource = _build_body(
                api, admission, weights, runtime, captured_m, 1, "leaf",
                recurrent_state_dtype=getattr(torch, recurrent_state_dtype),
            )
            active, _ = _prepare(
                body, resource, logical_m=1, context_length=16,
                poison=None if captured_m == 1 else 0,
                empty_context=True,
                active_layout="prefix",
            )
            if active != (0,):
                raise RuntimeError("leaf audit requires physical active row zero")
            expected = _expected_weights(
                admission.config,
                body.stage,
                admission.parallel_plan.topology.pp_size,
            )
            bindings = {item.tensor_name: item for item in weights.bindings}
            records: list[dict[str, object]] = []

            def consume(event: Any, ordinal: int) -> None:
                if ordinal >= len(expected) or event.weight_name not in bindings:
                    raise RuntimeError("observer emitted an unexpected linear occurrence")
                records.append(
                    _record_event(
                        event,
                        ordinal=ordinal,
                        expected_name=expected[ordinal],
                        binding=bindings[event.weight_name],
                        inventory_digest=weights.inventory_digest,
                        semantic_plan_digest=admission.parallel_plan.semantic_plan_digest,
                        sample_columns=sample_columns,
                    )
                )

            observer = _DeferredObserver(consume)
            torch.distributed.barrier()
            body.launch_qualification(observer)
            torch.cuda.synchronize(runtime.device)
            torch.distributed.barrier()
            # Validate every rank's recorded program before any reference math
            # or diagnostic reduction. PP stages have declared distinct names.
            schedules: list[Any] = [None] * admission.parallel_plan.topology.world_size
            torch.distributed.all_gather_object(schedules, observer.schedule)
            topology = admission.parallel_plan.topology
            for peer, schedule in enumerate(schedules):
                stage = topology.placements[peer].pipeline_stage
                names = _expected_weights(admission.config, stage, topology.pp_size)
                if tuple(item[1] for item in schedule) != names:
                    raise RuntimeError(f"rank {peer} recorded an invalid linear event schedule")
            for stage in range(topology.pp_size):
                peers = [
                    schedules[p.rank] for p in topology.placements if p.pipeline_stage == stage
                ]
                if any(schedule != peers[0] for schedule in peers[1:]):
                    raise RuntimeError("TP ranks recorded different ordered event schedules")
            observer.finish(len(expected))
            if tuple(item["weight_name"] for item in records) != expected:
                raise RuntimeError("observer coverage differs from the expected stage program")
            torch.save(records, tensor_path)
            report = {
                "rank": rank,
                "stage": body.stage,
                "tensor_rank": body.tensor_rank,
                "expected_occurrences": len(expected),
                "observed_occurrences": len(records),
                "row_occurrences": sum(bool(item["row_parallel"]) for item in records),
                "resident_weight_gib": round(weights.resident_bytes / _GIB, 3),
                "fallback": False,
                "operator_config": operator_config.digest,
                "operator_model_id": operator_model_id,
                "recurrent_state_dtype": recurrent_state_dtype,
            }
        report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    except BaseException:
        report_path.write_text(
            json.dumps({"rank": rank, "fallback": False, "error": traceback.format_exc()}),
            encoding="utf-8",
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=_CHECKPOINT)
    parser.add_argument(
        "--devices", type=lambda value: _csv_ints(value, allowed=tuple(range(64))),
        default=(0, 1, 2, 3),
    )
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--pp-size", type=int, default=2)
    parser.add_argument("--sample-output-columns", type=int, default=8)
    parser.add_argument("--operator-config", type=Path)
    parser.add_argument("--operator-model-id", default="")
    parser.add_argument(
        "--recurrent-state-dtype", choices=("float32", "bfloat16"), default="float32"
    )
    parser.add_argument(
        "--captured-m", type=int, choices=(1, 2, 4, 8, 16, 32, 64, 128, 256), default=2
    )
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    operator_config = (
        OperatorConfig.from_file(args.operator_config) if args.operator_config else OperatorConfig()
    )
    if any(p.model is not None for p in operator_config.profiles) and not args.operator_model_id:
        parser.error("model-scoped operator profiles require --operator-model-id")
    if (args.tp_size, args.pp_size) not in ((2, 2), (4, 1)):
        parser.error("four-rank leaf audit supports only TP2 x PP2 or TP4 x PP1")
    if len(args.devices) != 4 or len(set(args.devices)) != 4:
        parser.error("four-rank leaf audit requires four unique CUDA devices")
    if args.sample_output_columns <= 0 or args.timeout <= 0:
        parser.error("sample-output-columns and timeout must be positive")
    if not (args.checkpoint / "model.safetensors.index.json").is_file():
        _emit({"verdict": "error", "fallback": False, "error": "checkpoint unavailable"})
        return 1
    started = time.monotonic()
    try:
        admission = _admit(
            args.checkpoint, args.devices, args.tp_size, args.pp_size
        )
        with tempfile.TemporaryDirectory(
            prefix=f"cubic-qwen-tp{args.tp_size}-pp{args.pp_size}-leaf-"
        ) as raw:
            directory = Path(raw)
            try:
                _spawn_with_timeout(
                    _worker,
                    (
                        directory.joinpath("store").resolve().as_uri(), str(directory),
                        admission, args.sample_output_columns,
                        operator_config, args.operator_model_id,
                        args.recurrent_state_dtype,
                        args.captured_m,
                    ),
                    timeout=args.timeout,
                )
            except BaseException as error:
                _emit(
                    {
                        "verdict": "error", "fallback": False,
                        "error": f"{type(error).__name__}: {error}",
                        "ranks": _read_reports(directory),
                    }
                )
                return 1
            rank_reports = _read_reports(directory)
            if any("error" in report for report in rank_reports):
                _emit({"verdict": "error", "fallback": False, "ranks": rank_reports})
                return 1
            records = [
                torch.load(directory / f"rank-{rank}.pt", map_location="cpu", weights_only=True)
                for rank in range(4)
            ]
            expected_counts = cast(
                tuple[int, int, int, int],
                tuple(
                    cast(int, report["expected_occurrences"])
                    for report in rank_reports
                ),
            )
            planned_counts = cast(
                tuple[int, int, int, int],
                tuple(
                    len(
                        _expected_weights(
                            admission.config,
                            cast(int, report["stage"]),
                            args.pp_size,
                        )
                    )
                    for report in rank_reports
                ),
            )
            if expected_counts != planned_counts:
                raise RuntimeError("Qwen3.8 linear program count differs from topology plan")
            coverage = _merge(
                records,
                expected_counts,
                tp_size=args.tp_size,
                pp_size=args.pp_size,
                layer_count=admission.config.graph.text.layer_count,
            )
    except BaseException as error:
        _emit({"verdict": "error", "fallback": False, "error": f"{type(error).__name__}: {error}"})
        return 1
    passed = coverage["status"] == "pass"
    _emit(
        {
            "verdict": "pass" if passed else "fail",
            "fallback": False,
            "checkpoint": str(args.checkpoint.resolve()),
            "topology": {
                "tp": args.tp_size, "pp": args.pp_size, "ep": 1,
                "devices": args.devices,
            },
            "logical_m": 1,
            "captured_m": args.captured_m,
            "recurrent_state_dtype": args.recurrent_state_dtype,
            "operator_config": operator_config.digest,
            "operator_model_id": args.operator_model_id,
            "scope": "all_linear_occurrences_sampled_columns_full_k",
            "claim": "linear_leaf_sample_qualification_only",
            "sample_output_columns": args.sample_output_columns,
            "complete_k_per_sample": True,
            "coverage": coverage,
            "ranks": rank_reports,
            "seconds": round(time.monotonic() - started, 3),
        }
    )
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
