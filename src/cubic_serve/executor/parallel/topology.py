"""Cross-contract sealing for a concrete single-node partition."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from cubic_serve.contracts.execution import GroupKind, ParallelTopology, PartitionPlan


@dataclass(frozen=True, slots=True)
class SealedPartition:
    topology: ParallelTopology
    partition: PartitionPlan
    topology_digest: str
    partition_digest: str


def _digest(value: ParallelTopology | PartitionPlan) -> str:
    document = asdict(value)
    if isinstance(value, PartitionPlan):
        # Existing scalar partitions retain their published identity. Explicit
        # expert tensor ownership is new evidence and participates in the hash.
        for expert in document["expert_assignments"]:
            if not expert["expert_tensor_ranks"]:
                del expert["expert_tensor_ranks"]
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "b2s256:" + hashlib.blake2s(payload).hexdigest()


def seal_partition(topology: ParallelTopology, partition: PartitionPlan) -> SealedPartition:
    """Validate relationships that frozen leaf contracts cannot check in isolation."""

    if 1 < topology.ep_size < topology.tp_size and (
        not partition.expert_assignments
        or any(not expert.expert_tensor_ranks for expert in partition.expert_assignments)
    ):
        raise ValueError("legacy partition seals lack expert-tensor shard ownership evidence")
    if partition.topology_id != topology.topology_id:
        raise ValueError("partition references a different topology")

    placement_by_rank = {placement.rank: placement for placement in topology.placements}
    assignment_by_node = {assignment.node_id: assignment for assignment in partition.assignments}
    for assignment in partition.assignments:
        if assignment.pipeline_stage >= topology.pp_size:
            raise ValueError(f"graph node {assignment.node_id!r} references an unknown stage")
        if tuple(sorted(assignment.owner_ranks)) != assignment.owner_ranks:
            raise ValueError("graph assignment owner ranks must be sorted")
        for rank in assignment.owner_ranks:
            placement = placement_by_rank.get(rank)
            if placement is None:
                raise ValueError(f"graph node {assignment.node_id!r} references an unknown rank")
            if placement.pipeline_stage != assignment.pipeline_stage:
                raise ValueError(f"graph node {assignment.node_id!r} crosses pipeline stages")

    group_by_id = {group.group_id: group for group in topology.groups}
    for step in partition.collectives:
        if step.group_id not in group_by_id:
            raise ValueError(f"collective {step.step_id!r} references an unknown group")

    if topology.ep_size == 1 and any(
        not expert.expert_tensor_ranks for expert in partition.expert_assignments
    ):
        raise ValueError("expert ownership requires an enabled EP group")
    expert_ids_by_node: dict[str, list[int]] = {}
    for expert in partition.expert_assignments:
        graph_assignment = assignment_by_node.get(expert.node_id)
        if graph_assignment is None:
            raise ValueError(f"expert references unknown graph node {expert.node_id!r}")
        if expert.owner_rank not in graph_assignment.owner_ranks:
            raise ValueError("expert owner must be one of its graph node owners")
        owner_stage = placement_by_rank[expert.owner_rank].pipeline_stage
        if expert.expert_tensor_ranks:
            expert_coordinate, _ = topology.expert_coordinates(expert.owner_rank)
            expected = tuple(
                placement.rank
                for placement in topology.placements
                if placement.pipeline_stage == owner_stage
                and topology.expert_coordinates(placement.rank)[0] == expert_coordinate
            )
            if expert.expert_tensor_ranks != tuple(sorted(expected)):
                raise ValueError(
                    "expert tensor owners must close exactly one stage-local ETP group"
                )
            if not set(expert.expert_tensor_ranks).issubset(graph_assignment.owner_ranks):
                raise ValueError("every expert tensor shard must be a graph node owner")
            expert_ids_by_node.setdefault(expert.node_id, []).append(expert.expert_id)
            continue
        ep_group = next(
            (
                group
                for group in topology.groups
                if group.kind is GroupKind.EXPERT
                and expert.owner_rank in group.ranks
                and placement_by_rank[group.ranks[0]].pipeline_stage == owner_stage
            ),
            None,
        )
        if ep_group is None:
            raise ValueError("expert owner is not covered by a stage-local EP group")
        expert_ids_by_node.setdefault(expert.node_id, []).append(expert.expert_id)
    for node_id, expert_ids in expert_ids_by_node.items():
        if sorted(expert_ids) != list(range(len(expert_ids))):
            raise ValueError(f"expert ids for {node_id!r} must be contiguous from zero")

    for edge in partition.pipeline_edges:
        source = assignment_by_node.get(edge.source_node_id)
        target = assignment_by_node.get(edge.target_node_id)
        if source is None or target is None:
            raise ValueError(f"pipeline edge {edge.edge_id!r} references an unknown graph node")
        if edge.source_rank not in source.owner_ranks or edge.target_rank not in target.owner_ranks:
            raise ValueError(f"pipeline edge {edge.edge_id!r} rank is not a node owner")
        source_placement = placement_by_rank[edge.source_rank]
        target_placement = placement_by_rank[edge.target_rank]
        if target_placement.pipeline_stage != source_placement.pipeline_stage + 1:
            raise ValueError("pipeline edges must connect adjacent forward stages")
        if target_placement.tensor_rank != source_placement.tensor_rank:
            raise ValueError("pipeline edges must preserve the declared tensor lane")

    return SealedPartition(
        topology=topology,
        partition=partition,
        topology_digest=_digest(topology),
        partition_digest=_digest(partition),
    )
