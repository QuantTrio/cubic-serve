from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import torch

from cubic_serve.contracts.execution import (
    CollectiveStep,
    ExpertAssignment,
    GraphAssignment,
    GroupKind,
    ParallelTopology,
    PartitionPlan,
    PipelineEdge,
    RankGroup,
)
from cubic_serve.contracts.model import ScalarDType
from cubic_serve.executor.parallel import (
    RankTensor,
    RoutedRows,
    all_gather,
    all_reduce_sum,
    all_to_all_combine,
    all_to_all_dispatch,
    initialize_eager_nccl,
    seal_partition,
    send_forward,
)
from cubic_serve.executor.runner import ReferenceExecutionContext


def _distributed_partition():
    topology = ParallelTopology.single_node(
        topology_id="gpu0-3:tp2:pp2:ep2",
        device_indices=(0, 1, 2, 3),
        tp_size=2,
        pp_size=2,
        ep_size=2,
    )
    partition = PartitionPlan(
        partition_id="eager-nccl-slice",
        topology_id=topology.topology_id,
        assignments=(
            GraphAssignment("stage:0", 0, (0, 1)),
            GraphAssignment("stage:1", 1, (2, 3)),
        ),
        collectives=(
            CollectiveStep("tp.reduce:0", 0, "tp:0", "all_reduce"),
            CollectiveStep("tp.reduce:1", 0, "tp:1", "all_reduce"),
            CollectiveStep("ep.dispatch:0", 0, "ep:0", "all_to_all"),
            CollectiveStep("ep.combine:0", 1, "ep:0", "all_to_all"),
            CollectiveStep("ep.dispatch:1", 0, "ep:1", "all_to_all"),
            CollectiveStep("ep.combine:1", 1, "ep:1", "all_to_all"),
        ),
        expert_assignments=(
            ExpertAssignment("stage:0", 0, 0),
            ExpertAssignment("stage:0", 1, 1),
            ExpertAssignment("stage:1", 0, 2),
            ExpertAssignment("stage:1", 1, 3),
        ),
        pipeline_edges=(
            PipelineEdge(
                "pp:lane0",
                "stage:0",
                "stage:1",
                0,
                2,
                ScalarDType.BF16,
                "token_hidden",
            ),
            PipelineEdge(
                "pp:lane1",
                "stage:0",
                "stage:1",
                1,
                3,
                ScalarDType.BF16,
                "token_hidden",
            ),
        ),
    )
    return seal_partition(topology, partition)


def _eager_nccl_worker(rank: int, init_method: str) -> None:
    sealed = _distributed_partition()
    placement = sealed.topology.placements[rank]
    stage = placement.pipeline_stage
    tensor_rank = placement.tensor_rank
    with initialize_eager_nccl(sealed, rank=rank, init_method=init_method) as runtime:
        assert runtime.creation_order == (
            "ep:0",
            "ep:1",
            "pp:0",
            "pp:1",
            "tp:0",
            "tp:1",
        )

        reduced = torch.full(
            (2, 2),
            float(tensor_rank + 1 + stage * 2),
            dtype=torch.bfloat16,
            device=runtime.device,
        )
        runtime.tp_all_reduce(f"tp.reduce:{stage}", reduced)
        expected_sum = 3.0 if stage == 0 else 7.0
        assert torch.equal(reduced, torch.full_like(reduced, expected_sum))

        edge_id = f"pp:lane{tensor_rank}"
        if stage == 0:
            sent = torch.tensor(
                [[10.0 + tensor_rank, 20.0 + tensor_rank]],
                dtype=torch.bfloat16,
                device=runtime.device,
            )
            runtime.pipeline_send(edge_id, sent)
        else:
            received = runtime.pipeline_recv(edge_id, (1, 2))
            expected = torch.tensor(
                [[10.0 + tensor_rank, 20.0 + tensor_rank]],
                dtype=torch.bfloat16,
                device=runtime.device,
            )
            assert torch.equal(received, expected)

        row_count = 3 if tensor_rank == 0 else 1
        values = (
            torch.arange(row_count * 2, dtype=torch.float32, device=runtime.device)
            .reshape(row_count, 2)
            .add_(float(stage * 32 + tensor_rank * 8))
            .to(torch.bfloat16)
        )
        peer_rank = stage * 2 + 1
        owner_rank = stage * 2
        destinations = (
            torch.tensor(
                [owner_rank, peer_rank, peer_rank],
                dtype=torch.int64,
                device=runtime.device,
            )
            if tensor_rank == 0
            else torch.tensor([owner_rank], dtype=torch.int64, device=runtime.device)
        )
        routed = runtime.ep_dispatch(f"ep.dispatch:{stage}", values, destinations)
        expert_values = routed.values + float(rank + 1)
        combined = runtime.ep_combine(f"ep.combine:{stage}", routed, expert_values)
        expected = values + (destinations + 1).to(torch.bfloat16).unsqueeze(1)
        assert torch.equal(combined, expected)


class ReferenceCollectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tp = RankGroup("tp:0", GroupKind.TENSOR, (0, 1))
        self.ep = RankGroup("ep:0", GroupKind.EXPERT, (0, 1, 2))

    def test_all_reduce_and_gather_follow_canonical_rank_order(self) -> None:
        shards = (
            RankTensor(1, torch.tensor([[3.0]], dtype=torch.bfloat16)),
            RankTensor(0, torch.tensor([[1.0]], dtype=torch.bfloat16)),
        )
        with ReferenceExecutionContext():
            reduced = all_reduce_sum(self.tp, shards)
            gathered = all_gather(self.tp, shards, dim=1)

        self.assertEqual(tuple(item.rank for item in reduced), (0, 1))
        self.assertTrue(torch.equal(reduced[0].value, torch.tensor([[4.0]], dtype=torch.bfloat16)))
        self.assertTrue(
            torch.equal(
                gathered[0].value,
                torch.tensor([[1.0, 3.0]], dtype=torch.bfloat16),
            )
        )

    def test_ep_dispatch_and_combine_preserve_every_source_row(self) -> None:
        shards = (
            RankTensor(0, torch.tensor([[10.0], [11.0], [12.0]])),
            RankTensor(1, torch.tensor([[20.0]])),
            RankTensor(2, torch.empty((0, 1))),
        )
        destinations = (
            RankTensor(0, torch.tensor([0, 1, 1], dtype=torch.int64)),
            RankTensor(1, torch.tensor([0], dtype=torch.int64)),
            RankTensor(2, torch.empty((0,), dtype=torch.int64)),
        )
        with ReferenceExecutionContext():
            routed = all_to_all_dispatch(self.ep, shards, destinations)
            # An expert is represented by a rank-specific, visible transformation.
            expert_outputs = tuple(
                RoutedRows(
                    item.destination_rank,
                    item.values + float(item.destination_rank + 1),
                    item.source_ranks,
                    item.source_rows,
                )
                for item in routed
            )
            combined = all_to_all_combine(
                self.ep,
                expert_outputs,
                source_row_counts=((0, 3), (1, 1), (2, 0)),
            )

        self.assertTrue(torch.equal(combined[0].value, torch.tensor([[11.0], [13.0], [14.0]])))
        self.assertTrue(torch.equal(combined[1].value, torch.tensor([[21.0]])))
        self.assertEqual(combined[2].value.shape, (0, 1))

    def test_combine_rejects_missing_route(self) -> None:
        with (
            ReferenceExecutionContext(),
            self.assertRaisesRegex(ValueError, "restore every source row"),
        ):
            all_to_all_combine(
                self.tp,
                (
                    RoutedRows(
                        0,
                        torch.tensor([[1.0]]),
                        torch.tensor([0], dtype=torch.int64),
                        torch.tensor([0], dtype=torch.int64),
                    ),
                    RoutedRows(
                        1,
                        torch.empty((0, 1)),
                        torch.empty((0,), dtype=torch.int64),
                        torch.empty((0,), dtype=torch.int64),
                    ),
                ),
                source_row_counts=((0, 2), (1, 0)),
            )

    def test_pipeline_send_is_a_detached_copy(self) -> None:
        source = torch.tensor([[1.0]], dtype=torch.bfloat16)
        with ReferenceExecutionContext():
            received = send_forward(source, source_rank=0, target_rank=2)
        source.zero_()
        self.assertEqual(received.rank, 2)
        self.assertEqual(float(received.value.item()), 1.0)


@unittest.skipUnless(
    os.environ.get("CUBIC_RUN_NCCL_TESTS") == "1",
    "set CUBIC_RUN_NCCL_TESTS=1 to run the four-GPU NCCL slice",
)
class EagerNcclCollectiveTests(unittest.TestCase):
    def test_tp2_pp2_ep2_matches_exact_bf16_reference(self) -> None:
        if not torch.distributed.is_nccl_available():
            self.skipTest("torch NCCL support is unavailable")
        if torch.cuda.device_count() < 4:
            self.skipTest("four visible CUDA devices are required")
        with tempfile.TemporaryDirectory(prefix="cubic-nccl-") as directory:
            init_method = (Path(directory) / "store").resolve().as_uri()
            torch.multiprocessing.spawn(
                _eager_nccl_worker,
                args=(init_method,),
                nprocs=4,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
