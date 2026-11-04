from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import torch

from cubic_serve.contracts.execution import (
    BatchPhase,
    CollectiveStep,
    ExpertAssignment,
    GraphAssignment,
    ParallelTopology,
    PartitionPlan,
    PipelineEdge,
)
from cubic_serve.contracts.model import ScalarDType
from cubic_serve.contracts.request import TokenPosition
from cubic_serve.executor.parallel import initialize_eager_nccl, seal_partition
from cubic_serve.guardrail.compare import ComparisonPolicy, DifferenceClass, compare_tensors
from cubic_serve.models import ModelSegment

_HIDDEN = 4
_TOKENS = (1, 2, 3, 4)
_W0 = (
    (1, 0, 1, 0),
    (0, 1, 0, 1),
    (1, 1, 0, 0),
    (0, 0, 1, 1),
)
_W1 = (
    (1, 0, 0, 1),
    (0, 1, 1, 0),
    (1, 1, 1, 0),
    (0, 1, 1, 1),
)


def _segment(token_start: int, token_ids: tuple[int, ...]) -> ModelSegment:
    return ModelSegment(
        phase=BatchPhase.PREFILL,
        token_start=token_start,
        token_ids=token_ids,
        positions=tuple(TokenPosition((token_start + index,)) for index in range(len(token_ids))),
    )


def _toy_partition():
    topology = ParallelTopology.single_node(
        topology_id="toy:gpu0-3:tp2:pp2:ep2",
        device_indices=(0, 1, 2, 3),
        tp_size=2,
        pp_size=2,
        ep_size=2,
    )
    collectives: list[CollectiveStep] = []
    for run_index, run_id in enumerate(("cold", "hot")):
        for ordinal in range(len(_TOKENS)):
            request_step = run_index * len(_TOKENS) + ordinal
            collectives.extend(
                (
                    CollectiveStep(
                        f"{run_id}.stage0.linear:{ordinal}",
                        request_step,
                        "tp:0",
                        "all_reduce",
                    ),
                    CollectiveStep(
                        f"{run_id}.ep.dispatch:{ordinal}",
                        request_step * 2,
                        "ep:1",
                        "all_to_all",
                    ),
                    CollectiveStep(
                        f"{run_id}.ep.combine:{ordinal}",
                        request_step * 2 + 1,
                        "ep:1",
                        "all_to_all",
                    ),
                    CollectiveStep(
                        f"{run_id}.stage1.rows:{ordinal}",
                        request_step * 2,
                        "tp:1",
                        "all_reduce",
                    ),
                    CollectiveStep(
                        f"{run_id}.stage1.linear:{ordinal}",
                        request_step * 2 + 1,
                        "tp:1",
                        "all_reduce",
                    ),
                )
            )
    partition = PartitionPlan(
        partition_id="distributed-toy-request",
        topology_id=topology.topology_id,
        assignments=(
            GraphAssignment("toy.stage0", 0, (0, 1)),
            GraphAssignment("toy.stage1", 1, (2, 3)),
        ),
        collectives=tuple(collectives),
        expert_assignments=(
            ExpertAssignment("toy.stage1", 0, 2),
            ExpertAssignment("toy.stage1", 1, 3),
        ),
        pipeline_edges=(
            PipelineEdge(
                "toy.pp:lane0",
                "toy.stage0",
                "toy.stage1",
                0,
                2,
                ScalarDType.BF16,
                "token_hidden",
            ),
            PipelineEdge(
                "toy.pp:lane1",
                "toy.stage0",
                "toy.stage1",
                1,
                3,
                ScalarDType.BF16,
                "token_hidden",
            ),
        ),
    )
    return seal_partition(topology, partition)


def _embedding(token_id: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        (token_id, token_id + 1, token_id * 2, token_id * 2 + 1),
        dtype=torch.bfloat16,
        device=device,
    )


def _weight(rows: tuple[tuple[int, ...], ...], device: torch.device) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.bfloat16, device=device)


def _weight_shard(
    rows: tuple[tuple[int, ...], ...],
    tensor_rank: int,
    device: torch.device,
) -> torch.Tensor:
    start = tensor_rank * (_HIDDEN // 2)
    return torch.tensor(
        tuple(row[start : start + _HIDDEN // 2] for row in rows),
        dtype=torch.bfloat16,
        device=device,
    )


def _single_rank_segment(
    segment: ModelSegment,
    state: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight0 = _weight(_W0, device)
    weight1 = _weight(_W1, device)
    outputs: list[torch.Tensor] = []
    running = state.detach().clone()
    for offset, token_id in enumerate(segment.token_ids):
        ordinal = segment.token_start + offset
        row = _embedding(token_id, device)
        running = running + row.to(torch.float32)
        stateful = (row.to(torch.float32) + running).to(torch.bfloat16).unsqueeze(0)
        hidden = torch.matmul(stateful, weight0.t()).to(torch.bfloat16)
        expert_rank = 2 + ((token_id + ordinal) % 2)
        hidden = (hidden.to(torch.float32) + float(expert_rank - 1)).to(torch.bfloat16)
        outputs.append(torch.matmul(hidden, weight1.t()).to(torch.bfloat16)[0])
    return torch.stack(outputs), running


def _distributed_segment(
    runtime,
    segment: ModelSegment,
    *,
    run_id: str,
    state_shard: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    placement = runtime.sealed.topology.placements[runtime.rank]
    stage = placement.pipeline_stage
    tensor_rank = placement.tensor_rank
    shard_start = tensor_rank * (_HIDDEN // 2)
    weight = _weight_shard(_W0 if stage == 0 else _W1, tensor_rank, runtime.device)
    outputs: list[torch.Tensor] = []

    if stage == 0 and state_shard is None:
        raise RuntimeError("stage 0 requires its TP-owned recurrent-state shard")
    if stage == 1 and state_shard is not None:
        raise RuntimeError("stage 1 cannot own stage 0 recurrent state")

    for offset, token_id in enumerate(segment.token_ids):
        ordinal = segment.token_start + offset
        edge_id = f"toy.pp:lane{tensor_rank}"
        if stage == 0:
            assert state_shard is not None
            input_shard = _embedding(token_id, runtime.device)[
                shard_start : shard_start + _HIDDEN // 2
            ]
            state_shard = state_shard + input_shard.to(torch.float32)
            stateful = (input_shard.to(torch.float32) + state_shard).to(torch.bfloat16)
            hidden = torch.matmul(stateful.unsqueeze(0), weight.t()).to(torch.bfloat16)
            runtime.tp_all_reduce(f"{run_id}.stage0.linear:{ordinal}", hidden)
            runtime.pipeline_send(edge_id, hidden)
            continue

        activation = runtime.pipeline_recv(edge_id, (1, _HIDDEN))
        source_tensor_rank = ordinal % 2
        owns_row = tensor_rank == source_tensor_rank
        local_rows = activation if owns_row else activation.new_empty((0, _HIDDEN))
        expert_rank = 2 + ((token_id + ordinal) % 2)
        destinations = torch.full(
            (local_rows.shape[0],),
            expert_rank,
            dtype=torch.int64,
            device=runtime.device,
        )
        routed = runtime.ep_dispatch(f"{run_id}.ep.dispatch:{ordinal}", local_rows, destinations)
        expert_rows = (routed.values.to(torch.float32) + float(runtime.rank - 1)).to(
            torch.bfloat16
        )
        combined = runtime.ep_combine(
            f"{run_id}.ep.combine:{ordinal}",
            routed,
            expert_rows,
        )
        replicated = activation.new_zeros((1, _HIDDEN))
        if owns_row:
            replicated.copy_(combined)
        runtime.tp_all_reduce(f"{run_id}.stage1.rows:{ordinal}", replicated)
        input_shard = replicated[:, shard_start : shard_start + _HIDDEN // 2]
        output = torch.matmul(input_shard, weight.t()).to(torch.bfloat16)
        runtime.tp_all_reduce(f"{run_id}.stage1.linear:{ordinal}", output)
        outputs.append(output[0].detach().clone())

    return (torch.stack(outputs) if outputs else None), state_shard


def _tensor_payload(value: torch.Tensor) -> dict[str, object]:
    cpu = value.detach().contiguous().cpu()
    return {
        "dtype": str(cpu.dtype).removeprefix("torch."),
        "shape": list(cpu.shape),
        "values": cpu.to(torch.float32).reshape(-1).tolist(),
    }


def _payload_tensor(payload: dict[str, object]) -> torch.Tensor:
    dtype_name = payload["dtype"]
    if dtype_name == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_name == "float32":
        dtype = torch.float32
    else:
        raise ValueError(f"unsupported test payload dtype {dtype_name!r}")
    return torch.tensor(payload["values"], dtype=dtype).reshape(payload["shape"])


def _distributed_toy_worker(rank: int, init_method: str, report_directory: str) -> None:
    sealed = _toy_partition()
    placement = sealed.topology.placements[rank]
    full = _segment(0, _TOKENS)
    prefix = _segment(0, _TOKENS[:2])
    suffix = _segment(2, _TOKENS[2:])

    with initialize_eager_nccl(
        sealed,
        rank=rank,
        init_method=init_method,
        timeout_seconds=45,
    ) as runtime:
        state = (
            torch.zeros((_HIDDEN // 2,), dtype=torch.float32, device=runtime.device)
            if placement.pipeline_stage == 0
            else None
        )
        cold_output, cold_state = _distributed_segment(
            runtime,
            full,
            run_id="cold",
            state_shard=None if state is None else state.detach().clone(),
        )
        hot_prefix, prefix_state = _distributed_segment(
            runtime,
            prefix,
            run_id="hot",
            state_shard=None if state is None else state.detach().clone(),
        )
        snapshot = None if prefix_state is None else prefix_state.detach().clone()
        hot_suffix, hot_state = _distributed_segment(
            runtime,
            suffix,
            run_id="hot",
            state_shard=snapshot,
        )

        if placement.pipeline_stage == 0:
            assert cold_output is None and hot_prefix is None and hot_suffix is None
            assert cold_state is not None and prefix_state is not None and hot_state is not None
            assert torch.equal(cold_state, hot_state)
            output_payload = None
        else:
            assert cold_state is None and prefix_state is None and hot_state is None
            assert cold_output is not None and hot_prefix is not None and hot_suffix is not None
            hot_output = torch.cat((hot_prefix, hot_suffix), dim=0)
            assert torch.equal(cold_output, hot_output)
            output_payload = _tensor_payload(cold_output)

        report: dict[str, object] = {
            "rank": rank,
            "device_index": placement.device_index,
            "pipeline_stage": placement.pipeline_stage,
            "tensor_rank": placement.tensor_rank,
            "weight_shard_shape": [_HIDDEN, _HIDDEN // 2],
            "cold_output": output_payload,
            "cold_state": None if cold_state is None else _tensor_payload(cold_state),
            "prefix_state": None if prefix_state is None else _tensor_payload(prefix_state),
            "hot_state": None if hot_state is None else _tensor_payload(hot_state),
        }
        if rank == 0:
            zero = torch.zeros((_HIDDEN,), dtype=torch.float32, device=runtime.device)
            oracle_cold, oracle_state = _single_rank_segment(full, zero, runtime.device)
            oracle_prefix, oracle_prefix_state = _single_rank_segment(
                prefix,
                zero,
                runtime.device,
            )
            oracle_suffix, oracle_hot_state = _single_rank_segment(
                suffix,
                oracle_prefix_state,
                runtime.device,
            )
            oracle_hot = torch.cat((oracle_prefix, oracle_suffix), dim=0)
            assert torch.equal(oracle_cold, oracle_hot)
            assert torch.equal(oracle_state, oracle_hot_state)
            report["oracle"] = {
                "output": _tensor_payload(oracle_cold),
                "prefix_state_shards": (
                    _tensor_payload(oracle_prefix_state[:2]),
                    _tensor_payload(oracle_prefix_state[2:]),
                ),
                "state_shards": (
                    _tensor_payload(oracle_state[:2]),
                    _tensor_payload(oracle_state[2:]),
                ),
            }

    target = Path(report_directory) / f"rank-{rank}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    temporary.replace(target)


def _spawn_with_timeout(init_method: str, report_directory: str, timeout: float) -> None:
    context = torch.multiprocessing.spawn(
        _distributed_toy_worker,
        args=(init_method, report_directory),
        nprocs=4,
        join=False,
    )
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"distributed toy exceeded {timeout:.0f}s")
            if context.join(timeout=remaining, grace_period=5):
                return
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()


@unittest.skipUnless(
    os.environ.get("CUBIC_RUN_NCCL_TESTS") == "1",
    "set CUBIC_RUN_NCCL_TESTS=1 to run the four-GPU distributed toy",
)
class DistributedToyModelTests(unittest.TestCase):
    def assert_payload_exact(
        self,
        expected_payload: dict[str, object],
        actual_payload: dict[str, object],
    ) -> None:
        result = compare_tensors(
            _payload_tensor(expected_payload),
            _payload_tensor(actual_payload),
            ComparisonPolicy(DifferenceClass.EXACT),
        )
        self.assertTrue(result.passed, result.reason)

    def test_request_matches_single_rank_oracle_and_prefix_resume(self) -> None:
        if not torch.distributed.is_nccl_available():
            self.skipTest("torch NCCL support is unavailable")
        if torch.cuda.device_count() < 4:
            self.skipTest("four visible CUDA devices are required")
        with tempfile.TemporaryDirectory(prefix="cubic-distributed-toy-") as directory:
            init_method = (Path(directory) / "store").resolve().as_uri()
            _spawn_with_timeout(init_method, directory, timeout=90)
            reports = tuple(
                json.loads((Path(directory) / f"rank-{rank}.json").read_text("utf-8"))
                for rank in range(4)
            )

        oracle = reports[0]["oracle"]
        self.assertEqual(tuple(report["device_index"] for report in reports), (0, 1, 2, 3))
        self.assertTrue(all(report["weight_shard_shape"] == [4, 2] for report in reports))
        for rank in (2, 3):
            self.assertIsNone(reports[rank]["cold_state"])
            self.assert_payload_exact(oracle["output"], reports[rank]["cold_output"])
        for rank in (0, 1):
            self.assertIsNone(reports[rank]["cold_output"])
            self.assert_payload_exact(
                oracle["prefix_state_shards"][rank],
                reports[rank]["prefix_state"],
            )
            self.assert_payload_exact(
                oracle["state_shards"][rank],
                reports[rank]["cold_state"],
            )
            self.assert_payload_exact(reports[rank]["cold_state"], reports[rank]["hot_state"])


if __name__ == "__main__":
    unittest.main()
