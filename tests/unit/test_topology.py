from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import asdict, replace

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
from cubic_serve.executor.parallel import seal_partition


class PartitionSealingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.topology = ParallelTopology.single_node(
            topology_id="tp2:pp2:ep2",
            device_indices=(3, 4, 5, 6),
            tp_size=2,
            pp_size=2,
            ep_size=2,
        )

    def _partition(self) -> PartitionPlan:
        return PartitionPlan(
            partition_id="toy",
            topology_id=self.topology.topology_id,
            assignments=(
                GraphAssignment("embed", 0, (0, 1)),
                GraphAssignment("moe", 0, (0, 1)),
                GraphAssignment("head", 1, (2, 3)),
            ),
            collectives=(
                CollectiveStep("moe.dispatch", 0, "ep:0", "all_to_all"),
                CollectiveStep("moe.combine", 1, "ep:0", "all_to_all"),
            ),
            expert_assignments=tuple(
                ExpertAssignment("moe", expert_id, expert_id % 2) for expert_id in range(4)
            ),
            pipeline_edges=(
                PipelineEdge(
                    "stage:0:1:lane0",
                    "moe",
                    "head",
                    0,
                    2,
                    ScalarDType.BF16,
                    "token_hidden",
                ),
                PipelineEdge(
                    "stage:0:1:lane1",
                    "moe",
                    "head",
                    1,
                    3,
                    ScalarDType.BF16,
                    "token_hidden",
                ),
            ),
        )

    def test_tp_pp_ep_partition_seals_with_stable_digests(self) -> None:
        first = seal_partition(self.topology, self._partition())
        second = seal_partition(self.topology, self._partition())

        self.assertEqual(first.topology_digest, second.topology_digest)
        self.assertEqual(first.partition_digest, second.partition_digest)

    def test_pipeline_edge_cannot_cross_tensor_lanes(self) -> None:
        partition = self._partition()
        invalid = PartitionPlan(
            partition_id=partition.partition_id,
            topology_id=partition.topology_id,
            assignments=partition.assignments,
            collectives=partition.collectives,
            expert_assignments=partition.expert_assignments,
            pipeline_edges=(
                PipelineEdge(
                    "crossed",
                    "moe",
                    "head",
                    0,
                    3,
                    ScalarDType.BF16,
                    "token_hidden",
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "tensor lane"):
            seal_partition(self.topology, invalid)

    def test_expert_ownership_is_not_just_a_group_name(self) -> None:
        partition = self._partition()
        missing_expert = PartitionPlan(
            partition_id=partition.partition_id,
            topology_id=partition.topology_id,
            assignments=partition.assignments,
            collectives=partition.collectives,
            expert_assignments=partition.expert_assignments[1:],
            pipeline_edges=partition.pipeline_edges,
        )

        with self.assertRaisesRegex(ValueError, "contiguous"):
            seal_partition(self.topology, missing_expert)


class ExpertTensorTopologyTests(unittest.TestCase):
    def test_explicit_expert_shards_cover_one_owner_group_across_topologies(self):
        for tp, pp, ep in ((1, 1, 1), (4, 1, 1), (4, 1, 2), (4, 2, 2), (4, 1, 4)):
            topology = ParallelTopology.single_node(
                topology_id=f"tp{tp}.pp{pp}.ep{ep}",
                device_indices=tuple(range(tp * pp)),
                tp_size=tp,
                pp_size=pp,
                ep_size=ep,
            )
            assignments = []
            experts = []
            for stage in range(pp):
                node = f"stage{stage}.moe"
                ranks = tuple(range(stage * tp, (stage + 1) * tp))
                assignments.append(GraphAssignment(node, stage, ranks))
                for expert in range(8):
                    owner = expert // (8 // ep)
                    shards = tuple(r for r in ranks if topology.expert_coordinates(r)[0] == owner)
                    experts.append(ExpertAssignment(node, expert, shards[0], shards))
            partition = PartitionPlan(
                "owned", topology.topology_id, tuple(assignments), expert_assignments=tuple(experts)
            )
            sealed = seal_partition(topology, partition)
            self.assertEqual(sealed.partition.expert_assignments, tuple(experts))

    def test_partial_ep_rejects_missing_mixed_or_cross_stage_expert_shards(self):
        topology = ParallelTopology.single_node(
            topology_id="tp4.pp2.ep2",
            device_indices=tuple(range(8)),
            tp_size=4,
            pp_size=2,
            ep_size=2,
        )
        partition = PartitionPlan(
            "owned",
            topology.topology_id,
            (GraphAssignment("moe", 0, (0, 1, 2, 3)),),
            expert_assignments=(ExpertAssignment("moe", 0, 0, (0, 1)),),
        )
        for shards in ((), (0,), (0, 2), (0, 4), (0, 1, 2, 3)):
            invalid = replace(
                partition, expert_assignments=(ExpertAssignment("moe", 0, 0, shards),)
            )
            with self.assertRaises(ValueError):
                seal_partition(topology, invalid)
        incomplete_node = replace(partition, assignments=(GraphAssignment("moe", 0, (0,)),))
        with self.assertRaisesRegex(ValueError, "graph node owner"):
            seal_partition(topology, incomplete_node)
        for shards in ((1, 0), (0, 0), (1, 2)):
            with self.assertRaises(ValueError):
                ExpertAssignment("moe", 0, 0, shards)

    def test_legacy_scalar_partition_digest_does_not_change(self):
        topology = ParallelTopology.single_node(
            topology_id="legacy",
            device_indices=(0, 1),
            tp_size=2,
            pp_size=1,
            ep_size=2,
        )
        partition = PartitionPlan(
            "legacy",
            topology.topology_id,
            (GraphAssignment("moe", 0, (0, 1)),),
            expert_assignments=(ExpertAssignment("moe", 0, 0),),
        )
        historical = asdict(partition)
        for expert in historical["expert_assignments"]:
            del expert["expert_tensor_ranks"]
        expected = (
            "b2s256:"
            + hashlib.blake2s(
                json.dumps(
                    historical,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
        )
        self.assertEqual(seal_partition(topology, partition).partition_digest, expected)
        explicit = replace(partition, expert_assignments=(ExpertAssignment("moe", 0, 0, (0,)),))
        self.assertNotEqual(seal_partition(topology, explicit).partition_digest, expected)

    def test_partial_ep_has_orthogonal_nonreplicated_expert_tensor_groups(self):
        t = ParallelTopology.single_node(
            topology_id="tp4.ep2.etp2",
            device_indices=(0, 1, 2, 3),
            tp_size=4,
            pp_size=1,
            ep_size=2,
        )
        self.assertEqual(t.expert_tensor_size, 2)
        self.assertEqual(
            [t.expert_coordinates(r) for r in range(4)], [(0, 0), (0, 1), (1, 0), (1, 1)]
        )
        self.assertEqual(
            [g.ranks for g in t.groups if g.kind is GroupKind.EXPERT], [(0, 2), (1, 3)]
        )
        self.assertEqual(
            [g.ranks for g in t.groups if g.kind is GroupKind.EXPERT_TENSOR], [(0, 1), (2, 3)]
        )
        for groups in (
            tuple(g for g in t.groups if g.kind is not GroupKind.EXPERT_TENSOR),
            (
                *(g for g in t.groups if g.kind is not GroupKind.EXPERT),
                RankGroup("wrong0", GroupKind.EXPERT, (0, 1)),
                RankGroup("wrong1", GroupKind.EXPERT, (2, 3)),
            ),
        ):
            with self.assertRaises(ValueError):
                replace(t, groups=groups)
        with self.assertRaisesRegex(ValueError, "shard ownership evidence"):
            seal_partition(
                t,
                PartitionPlan(
                    partition_id="unqualified",
                    topology_id=t.topology_id,
                    assignments=(GraphAssignment("moe", 0, (0, 1, 2, 3)),),
                    collectives=(),
                    expert_assignments=(),
                    pipeline_edges=(),
                ),
            )

    def test_existing_topologies_keep_their_serialized_fields_and_group_ids(self):
        for tp, pp, ep in ((4, 1, 1), (4, 1, 4), (2, 2, 2)):
            t = ParallelTopology.single_node(
                topology_id="compat",
                device_indices=tuple(range(tp * pp)),
                tp_size=tp,
                pp_size=pp,
                ep_size=ep,
            )
            self.assertEqual(
                set(asdict(t)),
                {"topology_id", "tp_size", "pp_size", "ep_size", "placements", "groups"},
            )
            self.assertFalse(any(g.kind is GroupKind.EXPERT_TENSOR for g in t.groups))
            self.assertEqual(
                [g.group_id for g in t.groups if g.kind is GroupKind.EXPERT],
                [f"ep:{stage}" for stage in range(pp)] if ep > 1 else [],
            )


if __name__ == "__main__":
    unittest.main()
