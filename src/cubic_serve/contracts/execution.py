"""Execution, topology, plan, and state-transaction contracts."""

from __future__ import annotations

from typing_extensions import TypeAliasType

import hashlib
import json
from dataclasses import dataclass
from cubic_serve._compat import StrEnum

from cubic_serve.contracts.model import ScalarDType
from cubic_serve.contracts.request import TokenPosition


class ExecutionMode(StrEnum):
    REFERENCE = "reference"
    PERFORMANCE = "performance"


class BatchPhase(StrEnum):
    PREFILL = "prefill"
    DECODE = "decode"
    MIXED = "mixed"


class GroupKind(StrEnum):
    TENSOR = "tensor"
    PIPELINE = "pipeline"
    EXPERT = "expert"
    EXPERT_TENSOR = "expert_tensor"
    ENCODER = "encoder"


class DecodeProvider(StrEnum):
    NONE = "none"
    MTP = "mtp"
    DRAFT = "draft"
    DSPARK = "dspark"


class CandidateBackend(StrEnum):
    CUDA = "cuda"
    TRITON = "triton"


class PaddingIsolation(StrEnum):
    """How non-logical graph rows are prevented from touching request state."""

    PRIVATE_SCRATCH = "private_scratch"


CapacityItems = TypeAliasType("CapacityItems", tuple[tuple[str, int], ...])


def _validate_capacity_items(items: CapacityItems, *, required: bool, owner: str) -> None:
    if type(items) is not tuple:
        raise TypeError(f"{owner} capacity items must be an immutable tuple")
    if required and not items:
        raise ValueError(f"{owner} requires a non-empty capacity profile")
    if any(type(item) is not tuple or len(item) != 2 for item in items):
        raise TypeError(f"{owner} capacity items must contain immutable key/value pairs")
    if any(
        type(key) is not str or not key.isidentifier() or type(value) is not int or value <= 0
        for key, value in items
    ):
        raise ValueError(f"{owner} capacity items require canonical keys and positive integers")
    keys = tuple(key for key, _value in items)
    if tuple(sorted(keys)) != keys or len(set(keys)) != len(keys):
        raise ValueError(f"{owner} capacity items must have unique, sorted keys")


@dataclass(frozen=True, slots=True)
class RankReplayResources:
    """Static CUDA Graph resources exclusively owned on one rank."""

    rank: int
    graph_instance_id: str
    stream_lane_id: str
    buffer_set_id: str
    workspace_arena_id: str
    scratch_arena_id: str

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError("replay resource rank must be non-negative")
        if not all(
            (
                self.graph_instance_id,
                self.stream_lane_id,
                self.buffer_set_id,
                self.workspace_arena_id,
                self.scratch_arena_id,
            )
        ):
            raise ValueError("rank replay resource identities are required")


@dataclass(frozen=True, slots=True)
class ReplayLease:
    """Atomic, topology-wide ownership for one performance graph replay.

    The runtime owns acquisition and release.  The immutable contract binds an
    epoch to every rank-local graph instance, stream lane, static buffer set,
    workspace arena, and padding arena so a stale or partial lease cannot be reused.
    """

    lease_id: str
    epoch: int
    launch_plan_id: str
    execution_plan_id: str
    bucket_id: str
    template_id: str
    topology_id: str
    partition_id: str
    rank_resources: tuple[RankReplayResources, ...]

    def __post_init__(self) -> None:
        if not all(
            (
                self.lease_id,
                self.launch_plan_id,
                self.execution_plan_id,
                self.bucket_id,
                self.template_id,
                self.topology_id,
                self.partition_id,
            )
        ):
            raise ValueError("replay lease identities are required")
        if self.epoch <= 0:
            raise ValueError("replay lease epoch must be positive")
        if not self.rank_resources:
            raise ValueError("replay lease must atomically cover at least one rank")
        ranks = tuple(resource.rank for resource in self.rank_resources)
        if tuple(sorted(set(ranks))) != ranks:
            raise ValueError("replay lease ranks must be non-empty, unique, and sorted")


@dataclass(frozen=True, slots=True)
class RankPlacement:
    rank: int
    device_index: int
    pipeline_stage: int
    tensor_rank: int

    def __post_init__(self) -> None:
        if min(self.rank, self.device_index, self.pipeline_stage, self.tensor_rank) < 0:
            raise ValueError("rank placement values must be non-negative")


@dataclass(frozen=True, slots=True)
class RankGroup:
    group_id: str
    kind: GroupKind
    ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("group_id is required")
        if not self.ranks or tuple(sorted(set(self.ranks))) != self.ranks:
            raise ValueError("group ranks must be non-empty, unique, and sorted")


@dataclass(frozen=True, slots=True)
class ParallelTopology:
    """A single-node PP x TP rank grid with optional EP groups over each stage."""

    topology_id: str
    tp_size: int
    pp_size: int
    ep_size: int
    placements: tuple[RankPlacement, ...]
    groups: tuple[RankGroup, ...]

    def __post_init__(self) -> None:
        if not self.topology_id:
            raise ValueError("topology_id is required")
        if min(self.tp_size, self.pp_size, self.ep_size) <= 0:
            raise ValueError("parallel sizes must be positive")
        if self.tp_size % self.ep_size:
            raise ValueError("without a replica axis, EP must divide the stage TP width")

        expected_world_size = self.tp_size * self.pp_size
        if len(self.placements) != expected_world_size:
            raise ValueError("single-node topology requires world_size == tp_size * pp_size")
        if tuple(sorted(p.rank for p in self.placements)) != tuple(range(expected_world_size)):
            raise ValueError("placements must contain each world rank exactly once")
        if len({p.device_index for p in self.placements}) != expected_world_size:
            raise ValueError("each single-node rank must own a distinct device")

        for stage in range(self.pp_size):
            members = tuple(p for p in self.placements if p.pipeline_stage == stage)
            if len(members) != self.tp_size:
                raise ValueError(f"pipeline stage {stage} must contain tp_size ranks")
            if {p.tensor_rank for p in members} != set(range(self.tp_size)):
                raise ValueError(f"pipeline stage {stage} has invalid tensor ranks")

        group_ids = tuple(group.group_id for group in self.groups)
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("parallel group ids must be unique")
        valid_ranks = set(range(expected_world_size))
        if any(not set(group.ranks) <= valid_ranks for group in self.groups):
            raise ValueError("parallel group references an unknown rank")

        self._validate_partitioning_groups(GroupKind.TENSOR, self.tp_size)
        self._validate_partitioning_groups(GroupKind.PIPELINE, self.pp_size)
        self._validate_partitioning_groups(GroupKind.EXPERT, self.ep_size)
        if 1 < self.ep_size < self.tp_size or any(
            g.kind is GroupKind.EXPERT_TENSOR for g in self.groups
        ):
            self._validate_partitioning_groups(GroupKind.EXPERT_TENSOR, self.expert_tensor_size)

        placement_by_rank = {p.rank: p for p in self.placements}
        for group in self.groups:
            if group.kind in (GroupKind.TENSOR, GroupKind.EXPERT, GroupKind.EXPERT_TENSOR):
                stages = {placement_by_rank[rank].pipeline_stage for rank in group.ranks}
                if len(stages) != 1:
                    raise ValueError(f"{group.kind} group {group.group_id} crosses PP stages")
            if group.kind is GroupKind.PIPELINE:
                tensor_ranks = {placement_by_rank[rank].tensor_rank for rank in group.ranks}
                if len(tensor_ranks) != 1:
                    raise ValueError(f"pipeline group {group.group_id} crosses tensor lanes")

            if group.kind is GroupKind.EXPERT:
                lanes = {
                    placement_by_rank[r].tensor_rank % self.expert_tensor_size for r in group.ranks
                }
                if len(lanes) != 1:
                    raise ValueError("EP group must preserve one expert-tensor lane")
            if group.kind is GroupKind.EXPERT_TENSOR:
                owners = {
                    placement_by_rank[r].tensor_rank // self.expert_tensor_size for r in group.ranks
                }
                if len(owners) != 1:
                    raise ValueError("expert-tensor group must preserve one expert owner")

    @property
    def expert_tensor_size(self) -> int:
        """Expert-internal TP width; EP x ETP covers a stage without replicas."""
        return self.tp_size // self.ep_size

    def expert_coordinates(self, rank: int) -> tuple[int, int]:
        """Return immutable (expert owner, expert-tensor lane) coordinates."""
        placement = next((p for p in self.placements if p.rank == rank), None)
        if placement is None:
            raise ValueError("expert coordinates require a rank in this topology")
        return divmod(placement.tensor_rank, self.expert_tensor_size)

    @property
    def world_size(self) -> int:
        return len(self.placements)

    def _validate_partitioning_groups(self, kind: GroupKind, expected_size: int) -> None:
        groups = tuple(group for group in self.groups if group.kind is kind)
        if expected_size == 1 and not groups:
            return
        flattened = tuple(rank for group in groups for rank in group.ranks)
        if any(len(group.ranks) != expected_size for group in groups):
            raise ValueError(f"all {kind} groups must have size {expected_size}")
        if tuple(sorted(flattened)) != tuple(range(self.world_size)):
            raise ValueError(f"{kind} groups must partition all world ranks exactly once")

    @classmethod
    def single_node(
        cls,
        *,
        topology_id: str,
        device_indices: tuple[int, ...],
        tp_size: int,
        pp_size: int,
        ep_size: int = 1,
    ) -> ParallelTopology:
        if min(tp_size, pp_size, ep_size) <= 0 or tp_size % ep_size:
            raise ValueError("without a replica axis, positive EP must divide the stage TP width")
        if len(device_indices) != tp_size * pp_size:
            raise ValueError("device count must equal tp_size * pp_size")

        placements = tuple(
            RankPlacement(
                rank=rank,
                device_index=device,
                pipeline_stage=rank // tp_size,
                tensor_rank=rank % tp_size,
            )
            for rank, device in enumerate(device_indices)
        )

        groups: list[RankGroup] = []
        if tp_size > 1:
            for stage in range(pp_size):
                start = stage * tp_size
                groups.append(
                    RankGroup(
                        group_id=f"tp:{stage}",
                        kind=GroupKind.TENSOR,
                        ranks=tuple(range(start, start + tp_size)),
                    )
                )
        if pp_size > 1:
            for lane in range(tp_size):
                groups.append(
                    RankGroup(
                        group_id=f"pp:{lane}",
                        kind=GroupKind.PIPELINE,
                        ranks=tuple(stage * tp_size + lane for stage in range(pp_size)),
                    )
                )
        expert_tp = tp_size // ep_size
        if ep_size > 1:
            for stage in range(pp_size):
                stage_start = stage * tp_size
                for lane in range(expert_tp):
                    groups.append(
                        RankGroup(
                            group_id=f"ep:{stage}" if expert_tp == 1 else f"ep:{stage}:{lane}",
                            kind=GroupKind.EXPERT,
                            ranks=tuple(
                                stage_start + owner * expert_tp + lane for owner in range(ep_size)
                            ),
                        )
                    )
                if expert_tp > 1:
                    for owner in range(ep_size):
                        start = stage_start + owner * expert_tp
                        groups.append(
                            RankGroup(
                                group_id=f"etp:{stage}:{owner}",
                                kind=GroupKind.EXPERT_TENSOR,
                                ranks=tuple(range(start, start + expert_tp)),
                            )
                        )

        return cls(
            topology_id=topology_id,
            tp_size=tp_size,
            pp_size=pp_size,
            ep_size=ep_size,
            placements=placements,
            groups=tuple(groups),
        )


@dataclass(frozen=True, slots=True)
class GraphAssignment:
    node_id: str
    pipeline_stage: int
    owner_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.node_id or self.pipeline_stage < 0:
            raise ValueError("graph assignment requires a node and non-negative stage")
        if not self.owner_ranks or len(set(self.owner_ranks)) != len(self.owner_ranks):
            raise ValueError("graph assignment owner ranks must be non-empty and unique")


@dataclass(frozen=True, slots=True)
class CollectiveStep:
    step_id: str
    sequence_no: int
    group_id: str
    operation: str
    lane_id: str = "default"

    def __post_init__(self) -> None:
        if not self.step_id or not self.group_id or not self.operation or not self.lane_id:
            raise ValueError("collective step fields are required")
        if self.sequence_no < 0:
            raise ValueError("collective sequence number must be non-negative")


@dataclass(frozen=True, slots=True)
class ExpertAssignment:
    node_id: str
    expert_id: int
    owner_rank: int
    # Explicit tensor shards of one checkpoint-owned expert. Empty retains the
    # legacy single-rank representation; this never denotes expert replicas.
    expert_tensor_ranks: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.node_id or min(self.expert_id, self.owner_rank) < 0:
            raise ValueError("expert assignment requires a node and non-negative ids")
        if self.expert_tensor_ranks and (
            any(type(rank) is not int or rank < 0 for rank in self.expert_tensor_ranks)
            or tuple(sorted(set(self.expert_tensor_ranks))) != self.expert_tensor_ranks
            or self.expert_tensor_ranks[0] != self.owner_rank
        ):
            raise ValueError("expert tensor owners must be sorted, unique, and led by owner_rank")


@dataclass(frozen=True, slots=True)
class PipelineEdge:
    edge_id: str
    source_node_id: str
    target_node_id: str
    source_rank: int
    target_rank: int
    tensor_dtype: ScalarDType
    tensor_layout: str

    def __post_init__(self) -> None:
        if not all((self.edge_id, self.source_node_id, self.target_node_id, self.tensor_layout)):
            raise ValueError("pipeline edge identity and tensor layout are required")
        if min(self.source_rank, self.target_rank) < 0:
            raise ValueError("pipeline edge ranks must be non-negative")
        if self.source_rank == self.target_rank:
            raise ValueError("pipeline edge must cross ranks")


@dataclass(frozen=True, slots=True)
class PartitionPlan:
    partition_id: str
    topology_id: str
    assignments: tuple[GraphAssignment, ...]
    collectives: tuple[CollectiveStep, ...] = ()
    expert_assignments: tuple[ExpertAssignment, ...] = ()
    pipeline_edges: tuple[PipelineEdge, ...] = ()

    def __post_init__(self) -> None:
        if not self.partition_id or not self.topology_id:
            raise ValueError("partition and topology identity are required")
        node_ids = tuple(assignment.node_id for assignment in self.assignments)
        if not node_ids or len(set(node_ids)) != len(node_ids):
            raise ValueError("each graph node must have exactly one assignment")
        step_ids = tuple(step.step_id for step in self.collectives)
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("collective step ids must be unique")
        lanes = {(step.group_id, step.lane_id) for step in self.collectives}
        for group_id, lane_id in lanes:
            sequence_numbers = tuple(
                step.sequence_no
                for step in self.collectives
                if (step.group_id, step.lane_id) == (group_id, lane_id)
            )
            if tuple(sorted(sequence_numbers)) != tuple(range(len(sequence_numbers))):
                raise ValueError("collective sequence numbers must be contiguous per group lane")
        expert_keys = tuple(
            (assignment.node_id, assignment.expert_id) for assignment in self.expert_assignments
        )
        if len(set(expert_keys)) != len(expert_keys):
            raise ValueError("each graph node expert must have exactly one owner")
        edge_ids = tuple(edge.edge_id for edge in self.pipeline_edges)
        if len(set(edge_ids)) != len(edge_ids):
            raise ValueError("pipeline edge ids must be unique")


@dataclass(frozen=True, slots=True)
class BatchShape:
    logical_m: int
    capacity_items: CapacityItems = ()

    def __post_init__(self) -> None:
        if self.logical_m <= 0:
            raise ValueError("logical_m must be positive")
        _validate_capacity_items(self.capacity_items, required=False, owner="batch")


@dataclass(frozen=True, slots=True)
class PlanSegment:
    request_id: str
    input_plan_digest: str
    execution_input_digest: str
    phase: BatchPhase
    produces_output: bool
    token_start: int
    token_count: int
    token_ids: tuple[int, ...]
    positions: tuple[TokenPosition, ...]
    cache_slots: tuple[int, ...]
    cache_pages: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.request_id or self.token_start < 0 or self.token_count <= 0:
            raise ValueError("segment request, token start, and token count are invalid")
        if type(self.produces_output) is not bool:
            raise TypeError("segment output boundary must be an explicit boolean")
        for field_name, value in (
            ("input plan", self.input_plan_digest),
            ("execution input", self.execution_input_digest),
        ):
            algorithm, separator, digest = value.partition(":")
            if algorithm != "b2s256" or separator != ":" or len(digest) != 64:
                raise ValueError(f"segment {field_name} digest must use b2s256:<64 hex chars>")
            try:
                int(digest, 16)
            except ValueError as error:
                raise ValueError(
                    f"segment {field_name} digest contains non-hex characters"
                ) from error
        if len(self.cache_slots) != self.token_count:
            raise ValueError("every physical token must have one cache slot")
        if len(self.token_ids) != self.token_count or len(self.positions) != self.token_count:
            raise ValueError("every planned token must carry its exact id and position")
        if any(token_id < 0 for token_id in self.token_ids):
            raise ValueError("planned token ids must be non-negative")
        if any(value < 0 for value in (*self.cache_slots, *self.cache_pages)):
            raise ValueError("cache slots and pages must be non-negative")


@dataclass(frozen=True, slots=True)
class StateAccess:
    """One immutable resource region in a transaction plan."""

    resource_id: str
    owner_rank: int
    slots: tuple[int, ...] = ()
    pages: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.resource_id or self.owner_rank < 0:
            raise ValueError("state access requires an identity and non-negative owner rank")
        if tuple(sorted(set(self.slots))) != self.slots:
            raise ValueError("state access slots must be unique and sorted")
        if tuple(sorted(set(self.pages))) != self.pages:
            raise ValueError("state access pages must be unique and sorted")
        if any(value < 0 for value in (*self.slots, *self.pages)):
            raise ValueError("state access slots/pages must be non-negative")


@dataclass(frozen=True, slots=True)
class PrefixPublishPlan:
    """Relationship between a physical KV prefix and its arithmetic state boundary."""

    input_plan_id: str
    prefix_token_count: int
    state_commit_position: int
    commit_alignment_tokens: int

    def __post_init__(self) -> None:
        if not self.input_plan_id:
            raise ValueError("prefix publication requires an input plan identity")
        if self.prefix_token_count <= 0 or self.commit_alignment_tokens <= 0:
            raise ValueError("prefix length and state commit alignment must be positive")
        if not 0 <= self.state_commit_position <= self.prefix_token_count:
            raise ValueError("state commit position must fall within the prefix")
        if self.state_commit_position % self.commit_alignment_tokens:
            raise ValueError("state commit position must satisfy its arithmetic alignment")

    @property
    def replay_token_start(self) -> int:
        return self.state_commit_position

    @property
    def replay_token_count(self) -> int:
        return self.prefix_token_count - self.state_commit_position


@dataclass(frozen=True, slots=True)
class StateTxn:
    """A frozen transaction plan; runtime exclusively owns its mutable status machine."""

    transaction_id: str
    request_id: str
    base_revision: int
    target_revision: int
    reads: tuple[StateAccess, ...]
    writes: tuple[StateAccess, ...]
    expected_owner_ranks: tuple[int, ...]
    prefix_publish: PrefixPublishPlan | None = None

    def __post_init__(self) -> None:
        if not self.transaction_id or not self.request_id or self.base_revision < 0:
            raise ValueError(
                "transaction/request identity and non-negative base revision are required"
            )
        if self.target_revision != self.base_revision + 1:
            raise ValueError("transaction target revision must immediately follow its base")
        if not self.writes:
            raise ValueError("a state transaction must declare at least one private write")
        if tuple(sorted(set(self.expected_owner_ranks))) != self.expected_owner_ranks:
            raise ValueError("expected owner ranks must be non-empty, unique, and sorted")
        if not self.expected_owner_ranks:
            raise ValueError("a state transaction requires at least one expected owner")

        def region_key(access: StateAccess) -> tuple[str, int]:
            # The reference engine indexes accesses by this identity.  Allowing
            # two disjoint regions under one identity would silently collapse
            # one of them at that boundary.
            return (access.resource_id, access.owner_rank)

        if len({region_key(access) for access in self.reads}) != len(self.reads):
            raise ValueError("transaction permits one read region per resource/owner identity")
        if len({region_key(access) for access in self.writes}) != len(self.writes):
            raise ValueError("transaction permits one write region per resource/owner identity")
        known_owners = set(self.expected_owner_ranks)
        if any(access.owner_rank not in known_owners for access in (*self.reads, *self.writes)):
            raise ValueError("state access references an undeclared transaction owner")
        if {access.owner_rank for access in self.writes} != known_owners:
            raise ValueError("every expected owner must contribute a private write before commit")


@dataclass(frozen=True, slots=True)
class DecodeExtensionMeta:
    provider: DecodeProvider = DecodeProvider.NONE
    provider_version: str | None = None
    max_proposal_tokens: int = 0
    state_schema_id: str | None = None

    def __post_init__(self) -> None:
        if self.provider is DecodeProvider.NONE:
            if (
                any(value is not None for value in (self.provider_version, self.state_schema_id))
                or self.max_proposal_tokens
            ):
                raise ValueError("NONE decode provider cannot carry speculative configuration")
        elif not self.provider_version or not self.state_schema_id or self.max_proposal_tokens <= 0:
            raise ValueError("speculative providers require version, span, and state schema")


@dataclass(frozen=True, slots=True)
class SpeculativeIteration:
    """One speculative decision at an exact target-computed state boundary.

    ``target_state_token_count`` names ``S_t``: target state contains exactly the
    first ``t`` tokens.  ``root_token_id`` is the previously sampled token
    ``x_{t+1}``; it may already have been emitted, but target has not processed it.
    Verification consumes ``[root, *drafts]``.  The replacement (or all-accepted
    bonus) token is emitted by this iteration and becomes the next root, so it is
    deliberately absent from the committed target state and any published prefix.
    """

    target_state_token_count: int
    root_token_id: int
    draft_token_ids: tuple[int, ...]
    accepted_draft_count: int
    replacement_token_id: int

    def __post_init__(self) -> None:
        if type(self.target_state_token_count) is not int or self.target_state_token_count < 0:
            raise ValueError("target state token count must be a non-negative integer")
        if type(self.draft_token_ids) is not tuple or not self.draft_token_ids:
            raise TypeError("speculative draft tokens must be a non-empty immutable tuple")
        all_token_ids = (*self.verify_token_ids, self.replacement_token_id)
        if any(type(token_id) is not int or token_id < 0 for token_id in all_token_ids):
            raise ValueError("speculative token ids must be non-negative integers")
        if type(self.accepted_draft_count) is not int or not (
            0 <= self.accepted_draft_count <= len(self.draft_token_ids)
        ):
            raise ValueError("accepted draft count must fall within the proposed span")

    @property
    def verify_token_ids(self) -> tuple[int, ...]:
        """Inputs evaluated by target from ``S_t``; never a commit declaration."""

        return (self.root_token_id, *self.draft_token_ids)

    @property
    def committed_target_token_ids(self) -> tuple[int, ...]:
        """Verified inputs whose resulting target state is committed."""

        return (self.root_token_id, *self.draft_token_ids[: self.accepted_draft_count])

    @property
    def target_commit_token_count(self) -> int:
        """Boundary ``S_{t+1+a}`` selected from verification state."""

        return self.target_state_token_count + 1 + self.accepted_draft_count

    @property
    def committed_target_row_count(self) -> int:
        """Selected verification rows ``L=A+1``; even ``A=0`` commits root."""

        return self.accepted_draft_count + 1

    @property
    def emitted_token_ids(self) -> tuple[int, ...]:
        """New client-visible tokens; the prior root is not emitted twice."""

        return (*self.draft_token_ids[: self.accepted_draft_count], self.replacement_token_id)

    @property
    def next_root_token_id(self) -> int:
        return self.replacement_token_id

    @property
    def prefix_token_count(self) -> int:
        """Only target-computed state is publishable as a prefix."""

        return self.target_commit_token_count

    def prefix_checkpoint_boundary(
        self,
        *,
        block_size_tokens: int,
        available_target_boundaries: tuple[int, ...],
    ) -> int | None:
        """Select the sole complete-page checkpoint that may be published.

        ``B`` is the target-computed boundary after committing ``L=A+1``
        verification rows.  A physical prefix may end only at the greatest
        complete page ``P <= B`` and only when the caller actually retained a
        hybrid-state checkpoint at exactly ``P``.  Returning ``None`` is a
        fail-closed cache miss, never permission to combine KV at ``P`` with
        recurrent state from another boundary.
        """

        if type(block_size_tokens) is not int or block_size_tokens <= 0:
            raise ValueError("prefix block size must be a positive integer")
        if (
            type(available_target_boundaries) is not tuple
            or any(
                type(boundary) is not int or boundary < 0
                for boundary in available_target_boundaries
            )
            or tuple(sorted(set(available_target_boundaries))) != available_target_boundaries
        ):
            raise ValueError(
                "available target checkpoints must be sorted unique non-negative integers"
            )
        boundary = self.target_commit_token_count
        physical_boundary = boundary - boundary % block_size_tokens
        if physical_boundary == 0 or physical_boundary not in available_target_boundaries:
            return None
        return physical_boundary


@dataclass(frozen=True, slots=True)
class BatchPlan:
    plan_id: str
    phase: BatchPhase
    input_plan_ids: tuple[str, ...]
    segments: tuple[PlanSegment, ...]
    shape: BatchShape
    topology_id: str
    partition_id: str
    state_transactions: tuple[StateTxn, ...] = ()
    decode_extension: DecodeExtensionMeta = DecodeExtensionMeta()

    def __post_init__(self) -> None:
        if not self.plan_id or not self.topology_id or not self.partition_id:
            raise ValueError("plan, topology, and partition identity are required")
        if not self.input_plan_ids or len(set(self.input_plan_ids)) != len(self.input_plan_ids):
            raise ValueError("input plan ids must be non-empty and unique")
        if not self.segments:
            raise ValueError("a batch plan requires at least one segment")
        segment_request_ids = tuple(segment.request_id for segment in self.segments)
        if segment_request_ids != self.input_plan_ids:
            raise ValueError("batch segments must bind one-to-one and in order with input plan ids")
        if sum(segment.token_count for segment in self.segments) != self.shape.logical_m:
            raise ValueError("logical_m must equal the sum of segment token counts")
        if self.phase is not BatchPhase.MIXED and any(
            segment.phase is not self.phase for segment in self.segments
        ):
            raise ValueError("non-mixed batch cannot contain a different segment phase")
        txn_ids = tuple(txn.transaction_id for txn in self.state_transactions)
        if len(set(txn_ids)) != len(txn_ids):
            raise ValueError("state transaction ids must be unique within a batch")
        if tuple(txn.request_id for txn in self.state_transactions) != segment_request_ids:
            raise ValueError("every batch segment requires one same-order request transaction")
        if any(
            txn.prefix_publish is not None and txn.prefix_publish.input_plan_id != txn.request_id
            for txn in self.state_transactions
        ):
            raise ValueError("prefix publication must bind to its transaction request")


@dataclass(frozen=True, slots=True)
class ActiveLaunchRow:
    """One real work row from the immutable source plan."""

    logical_index: int

    def __post_init__(self) -> None:
        if self.logical_index < 0:
            raise ValueError("active launch row index must be non-negative")


@dataclass(frozen=True, slots=True)
class PaddingLaunchRow:
    """One graph-only row whose state address comes from a private scratch lease."""

    scratch_index: int

    def __post_init__(self) -> None:
        if self.scratch_index < 0:
            raise ValueError("padding scratch index must be non-negative")


LaunchRow = TypeAliasType("LaunchRow", ActiveLaunchRow | PaddingLaunchRow)


@dataclass(frozen=True, slots=True)
class GraphBucket:
    """One pre-captured performance graph and its completely qualified M domain."""

    bucket_id: str
    template_id: str
    scope_id: str
    replay_abi_id: str
    topology_id: str
    partition_id: str
    phase: BatchPhase
    capacity_items: CapacityItems
    logical_m_values: tuple[int, ...]
    captured_m: int
    row_map_schema_id: str
    scratch_schema_id: str
    graph_safety_contract_ids: tuple[str, ...]
    evidence_id: str

    def __post_init__(self) -> None:
        identities = (
            self.bucket_id,
            self.template_id,
            self.scope_id,
            self.replay_abi_id,
            self.topology_id,
            self.partition_id,
            self.row_map_schema_id,
            self.scratch_schema_id,
            self.evidence_id,
        )
        if not _exact_strings(*identities) or not all(identities):
            raise ValueError("graph bucket identities, schemas, and evidence are required")
        if type(self.phase) is not BatchPhase:
            raise TypeError("graph bucket phase must use BatchPhase")
        _validate_capacity_items(self.capacity_items, required=True, owner="graph bucket")
        if type(self.logical_m_values) is not tuple or any(
            type(value) is not int for value in self.logical_m_values
        ):
            raise TypeError("graph bucket logical M values must be exact immutable integers")
        if type(self.captured_m) is not int:
            raise TypeError("graph bucket captured M must be an exact integer")
        if type(self.graph_safety_contract_ids) is not tuple or not _exact_strings(
            *self.graph_safety_contract_ids
        ):
            raise TypeError("graph safety contract ids must be immutable canonical strings")
        if (
            not self.graph_safety_contract_ids
            or tuple(sorted(set(self.graph_safety_contract_ids))) != self.graph_safety_contract_ids
        ):
            raise ValueError("graph safety contract ids must be non-empty, unique, and sorted")
        if (
            len(self.logical_m_values) < 2
            or tuple(sorted(set(self.logical_m_values))) != self.logical_m_values
            or self.logical_m_values[0] <= 0
        ):
            raise ValueError(
                "performance graph bucket must cover multiple logical M values; "
                "values must be positive, unique, and sorted; singleton exact-M graphs "
                "are forbidden"
            )
        if self.logical_m_values[-1] != self.captured_m:
            raise ValueError("a graph bucket must admit its captured-M full replay")

    def admits(
        self,
        *,
        scope_id: str,
        replay_abi_id: str,
        phase: BatchPhase,
        logical_m: int,
        capacity_items: CapacityItems,
        topology_id: str,
        partition_id: str,
    ) -> bool:
        return (
            scope_id == self.scope_id
            and replay_abi_id == self.replay_abi_id
            and phase is self.phase
            and logical_m in self.logical_m_values
            and capacity_items == self.capacity_items
            and topology_id == self.topology_id
            and partition_id == self.partition_id
        )


@dataclass(frozen=True, slots=True)
class LaunchPlan:
    """Bind real work and isolated padding rows to one certified graph replay.

    Padding rows carry no request, token, position, cache, route, or transaction
    identity.  The executor may resolve an :class:`ActiveLaunchRow` through the
    source plan, while a :class:`PaddingLaunchRow` can resolve only through the
    named private scratch arena in an exclusive topology-wide replay lease.
    """

    launch_plan_id: str
    execution_plan_id: str
    source_plan_id: str
    logical_m: int
    bucket: GraphBucket
    rows: tuple[LaunchRow, ...]
    replay_lease: ReplayLease

    def __post_init__(self) -> None:
        if not all((self.launch_plan_id, self.execution_plan_id, self.source_plan_id)):
            raise ValueError("launch, execution, and source plan identities are required")
        if not isinstance(self.replay_lease, ReplayLease):
            raise TypeError("every performance replay requires a typed replay lease")
        if self.logical_m not in self.bucket.logical_m_values:
            raise ValueError("logical_m is outside the certified graph bucket domain")
        if len(self.rows) != self.bucket.captured_m:
            raise ValueError("launch rows must exactly fill captured_m")

        active_indices = tuple(
            row.logical_index for row in self.rows if isinstance(row, ActiveLaunchRow)
        )
        if tuple(sorted(active_indices)) != tuple(range(self.logical_m)):
            raise ValueError("launch must map every logical row exactly once")

        scratch_indices = tuple(
            row.scratch_index for row in self.rows if isinstance(row, PaddingLaunchRow)
        )
        expected_scratch = tuple(range(self.bucket.captured_m - self.logical_m))
        if tuple(sorted(scratch_indices)) != expected_scratch:
            raise ValueError("every padding row must own one unique dense scratch slot")

        lease = self.replay_lease
        expected_lease_identity = (
            self.launch_plan_id,
            self.execution_plan_id,
            self.bucket.bucket_id,
            self.bucket.template_id,
            self.bucket.topology_id,
            self.bucket.partition_id,
        )
        observed_lease_identity = (
            lease.launch_plan_id,
            lease.execution_plan_id,
            lease.bucket_id,
            lease.template_id,
            lease.topology_id,
            lease.partition_id,
        )
        if observed_lease_identity != expected_lease_identity:
            raise ValueError("replay lease identity does not match its launch and graph bucket")

    @property
    def captured_m(self) -> int:
        return self.bucket.captured_m

    @property
    def active_physical_indices(self) -> tuple[int, ...]:
        """Physical row positions whose outputs may be observed or committed."""

        return tuple(
            physical_index
            for physical_index, row in enumerate(self.rows)
            if isinstance(row, ActiveLaunchRow)
        )


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """One ordered tensor value in a semantic operator ABI.

    Integer dimensions are fixed literals. String dimensions are symbols: a
    value in ``OpSignature.shape_items`` fixes the symbol, while an input-only
    symbol is bound from the first input occurrence and must agree everywhere
    else in that call.
    """

    name: str
    dtype: ScalarDType
    shape: tuple[int | str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.name) is not str
            or not self.name
            or self.name != self.name.strip()
            or not self.name.isidentifier()
        ):
            raise ValueError("tensor ABI names must be canonical Python identifiers")
        if type(self.dtype) is not ScalarDType:
            raise TypeError("tensor ABI dtype must be a ScalarDType")
        if type(self.shape) is not tuple:
            raise TypeError("tensor ABI shape must be an immutable tuple")
        for dimension in self.shape:
            if type(dimension) is int:
                if dimension <= 0:
                    raise ValueError("literal tensor dimensions must be positive")
            elif (
                type(dimension) is not str
                or not dimension
                or dimension != dimension.strip()
                or not dimension.isidentifier()
            ):
                raise ValueError("symbolic tensor dimensions must be canonical identifiers")


@dataclass(frozen=True, slots=True)
class OpSignature:
    semantic_op: str
    phase: BatchPhase
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    accumulator_dtype: ScalarDType
    layout: str
    shape_items: tuple[tuple[str, int], ...]
    cache_form: str | None
    state_form: str | None
    topology_id: str

    def __post_init__(self) -> None:
        if not _exact_strings(self.semantic_op, self.layout, self.topology_id) or not all(
            (self.semantic_op, self.layout, self.topology_id)
        ):
            raise ValueError("op semantic, layout, and topology identity are required")
        if type(self.phase) is not BatchPhase or type(self.accumulator_dtype) is not ScalarDType:
            raise TypeError("operator phase and accumulator dtype must use contract enums")
        if any(
            value is not None and type(value) is not str
            for value in (self.cache_form, self.state_form)
        ):
            raise TypeError("operator cache and state forms must be canonical strings")
        if type(self.inputs) is not tuple or type(self.outputs) is not tuple:
            raise TypeError("operator tensor inputs and outputs must be immutable tuples")
        if type(self.shape_items) is not tuple:
            raise TypeError("operator shape items must be an immutable tuple")
        if not self.inputs or not self.outputs:
            raise ValueError("an op signature requires non-empty tensor inputs and outputs")
        if any(type(item) is not TensorSpec for item in (*self.inputs, *self.outputs)):
            raise TypeError("operator inputs and outputs must contain only TensorSpec values")
        for item in (*self.inputs, *self.outputs):
            TensorSpec.__post_init__(item)
        for kind, values in (("input", self.inputs), ("output", self.outputs)):
            names = tuple(item.name for item in values)
            if len(set(names)) != len(names):
                raise ValueError(f"operator {kind} tensor names must be unique")
        if any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not int
            for item in self.shape_items
        ):
            raise TypeError("shape items require exact immutable key/value pairs")
        keys = tuple(key for key, _ in self.shape_items)
        if tuple(sorted(keys)) != keys or len(set(keys)) != len(keys):
            raise ValueError("shape items must have unique, sorted keys")
        if any(
            not key or not key.isidentifier() or type(value) is not int or value <= 0
            for key, value in self.shape_items
        ):
            raise ValueError("shape items require canonical keys and positive integer values")

        fixed_symbols = set(keys)
        input_symbols = {
            dimension
            for tensor in self.inputs
            for dimension in tensor.shape
            if isinstance(dimension, str)
        }
        for tensor in self.outputs:
            unknown = {
                dimension
                for dimension in tensor.shape
                if isinstance(dimension, str)
                and dimension not in fixed_symbols
                and dimension not in input_symbols
            }
            if unknown:
                raise ValueError(
                    "output tensor dimensions must be fixed or bound by an input: "
                    + ", ".join(sorted(unknown))
                )

    @property
    def input_dtypes(self) -> tuple[ScalarDType, ...]:
        """Compatibility projection for metadata-only consumers."""

        return tuple(tensor.dtype for tensor in self.inputs)

    @property
    def output_dtype(self) -> ScalarDType:
        """Compatibility projection for the current single-output operator set."""

        if len(self.outputs) != 1:
            raise AttributeError("multi-output signatures do not have one output_dtype")
        return self.outputs[0].dtype


@dataclass(frozen=True, slots=True)
class GraphSafetyContract:
    """Candidate proof that padding writes are confined to private scratch."""

    contract_id: str
    row_map_schema_id: str
    isolation: PaddingIsolation
    evidence_id: str

    def __post_init__(self) -> None:
        if not _exact_strings(
            self.contract_id, self.row_map_schema_id, self.evidence_id
        ) or not all((self.contract_id, self.row_map_schema_id, self.evidence_id)):
            raise ValueError(
                "graph safety contract identity, row schema, and evidence are required"
            )
        if self.isolation is not PaddingIsolation.PRIVATE_SCRATCH:
            raise ValueError("performance graphs require private-scratch padding isolation")


@dataclass(frozen=True, slots=True)
class CandidateBinding:
    candidate_id: str
    implementation_id: str
    backend: CandidateBackend
    launch_signature: str
    workspace_bytes: int
    evidence_id: str
    graph_safety: GraphSafetyContract | None = None

    def __post_init__(self) -> None:
        identities = (
            self.candidate_id,
            self.implementation_id,
            self.launch_signature,
            self.evidence_id,
        )
        if not _exact_strings(*identities) or not all(identities):
            raise ValueError("candidate identity, backend, and launch signature are required")
        if type(self.backend) is not CandidateBackend:
            raise TypeError("candidate backend must use CandidateBackend")
        if type(self.workspace_bytes) is not int or self.workspace_bytes < 0:
            raise ValueError("candidate requires non-negative workspace and guardrail evidence")
        if self.graph_safety is not None:
            if type(self.graph_safety) is not GraphSafetyContract:
                raise TypeError("candidate graph safety must use its exact contract type")
            GraphSafetyContract.__post_init__(self.graph_safety)


@dataclass(frozen=True, slots=True)
class OpBinding:
    signature: OpSignature
    reference_implementation_id: str | None = None
    candidate: CandidateBinding | None = None

    def __post_init__(self) -> None:
        if type(self.signature) is not OpSignature:
            raise TypeError("an op binding requires an exact signature")
        OpSignature.__post_init__(self.signature)
        if (self.reference_implementation_id is None) == (self.candidate is None):
            raise ValueError(
                "an op binding must select exactly one reference or candidate implementation"
            )
        if self.reference_implementation_id is not None and (
            type(self.reference_implementation_id) is not str
            or not self.reference_implementation_id
        ):
            raise ValueError("reference implementation identity cannot be empty")
        if self.candidate is not None:
            if type(self.candidate) is not CandidateBinding:
                raise TypeError("an op binding candidate must use its exact contract type")
            CandidateBinding.__post_init__(self.candidate)


def _canonical_evidence_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _stable_evidence_digest(value: object) -> str:
    encoded = _canonical_evidence_json(value).encode()
    return "b2s256:" + hashlib.blake2s(encoded).hexdigest()


def _is_digest(value: object) -> bool:
    if type(value) is not str:
        return False
    algorithm, separator, digest = value.partition(":")
    return (
        algorithm == "b2s256"
        and separator == ":"
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )


def _signature_payload(signature: OpSignature) -> dict[str, object]:
    return {
        "semantic_op": signature.semantic_op,
        "phase": signature.phase.value,
        "inputs": tuple(_tensor_payload(tensor) for tensor in signature.inputs),
        "outputs": tuple(_tensor_payload(tensor) for tensor in signature.outputs),
        "accumulator_dtype": signature.accumulator_dtype.value,
        "layout": signature.layout,
        "shape_items": signature.shape_items,
        "cache_form": signature.cache_form,
        "state_form": signature.state_form,
        "topology_id": signature.topology_id,
    }


def _tensor_payload(tensor: TensorSpec) -> dict[str, object]:
    return {"name": tensor.name, "dtype": tensor.dtype.value, "shape": tensor.shape}


def canonical_op_signature_digest(signature: OpSignature) -> str:
    if type(signature) is not OpSignature:
        raise TypeError("operator signature identity requires an exact contract type")
    return _stable_evidence_digest(_signature_payload(signature))


def _graph_safety_payload(safety: GraphSafetyContract | None) -> dict[str, str] | None:
    if safety is None:
        return None
    return {
        "contract_id": safety.contract_id,
        "row_map_schema_id": safety.row_map_schema_id,
        "isolation": safety.isolation.value,
        "evidence_id": safety.evidence_id,
    }


def _candidate_payload(candidate: CandidateBinding, *, include_evidence: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "candidate_id": candidate.candidate_id,
        "implementation_id": candidate.implementation_id,
        "backend": candidate.backend.value,
        "launch_signature": candidate.launch_signature,
        "workspace_bytes": candidate.workspace_bytes,
        "graph_safety": _graph_safety_payload(candidate.graph_safety),
    }
    if include_evidence:
        payload["evidence_id"] = candidate.evidence_id
    return payload


def _binding_payload(binding: OpBinding, *, include_evidence: bool) -> dict[str, object]:
    candidate = binding.candidate
    if type(candidate) is not CandidateBinding:
        raise TypeError("arithmetic evidence requires an exact candidate binding")
    return {
        "signature": _signature_payload(binding.signature),
        "candidate": _candidate_payload(candidate, include_evidence=include_evidence),
    }


def _exact_strings(*values: object) -> bool:
    return all(type(value) is str for value in values)


def _exact_binding(binding: object) -> bool:
    try:
        if type(binding) is not OpBinding or binding.reference_implementation_id is not None:
            return False
        OpBinding.__post_init__(binding)
        return True
    except (AttributeError, TypeError, ValueError):
        return False


def _exact_graph_bucket(bucket: object) -> bool:
    try:
        if type(bucket) is not GraphBucket:
            return False
        GraphBucket.__post_init__(bucket)
        return True
    except (AttributeError, TypeError, ValueError):
        return False


@dataclass(frozen=True, slots=True, init=False)
class ArithmeticEvidenceSeal:
    """Trusted-process integrity record, not an arbitrary-code security boundary."""

    binding: OpBinding
    fingerprint_key: str
    source_digest: str
    binary_digest: str
    device_capability: str
    driver_version: str
    runtime_version: str
    topology_digest: str
    weight_semantics_digest: str
    required_evidence_ids: tuple[str, ...]
    evidence_id: str

    def __init__(self) -> None:
        raise TypeError("arithmetic evidence seals require a qualified guardrail issuer")


def _exact_arithmetic_seal(seal: object) -> bool:
    try:
        if type(seal) is not ArithmeticEvidenceSeal:
            return False
        environment = (seal.device_capability, seal.driver_version, seal.runtime_version)
        digests = (
            seal.fingerprint_key,
            seal.source_digest,
            seal.binary_digest,
            seal.topology_digest,
            seal.weight_semantics_digest,
            seal.evidence_id,
        )
        return (
            _exact_binding(seal.binding)
            and _exact_strings(*environment)
            and all(environment)
            and type(seal.required_evidence_ids) is tuple
            and bool(seal.required_evidence_ids)
            and seal.required_evidence_ids == tuple(sorted(set(seal.required_evidence_ids)))
            and all(_is_digest(item) for item in seal.required_evidence_ids)
            and all(_is_digest(item) for item in digests)
        )
    except (AttributeError, TypeError, ValueError):
        return False


def canonical_arithmetic_evidence_id(seal: ArithmeticEvidenceSeal) -> str:
    if not _exact_arithmetic_seal(seal):
        raise TypeError("arithmetic evidence identity requires an exact seal type")
    return _stable_evidence_digest(
        {
            "schema": "cubic.arithmetic-evidence.v2",
            "binding": _binding_payload(seal.binding, include_evidence=False),
            "fingerprint_key": seal.fingerprint_key,
            "source_digest": seal.source_digest,
            "binary_digest": seal.binary_digest,
            "device_capability": seal.device_capability,
            "driver_version": seal.driver_version,
            "runtime_version": seal.runtime_version,
            "topology_digest": seal.topology_digest,
            "weight_semantics_digest": seal.weight_semantics_digest,
            "required_evidence_ids": seal.required_evidence_ids,
        }
    )


def validate_arithmetic_evidence(seal: object, binding: object) -> bool:
    try:
        if not _exact_arithmetic_seal(seal) or not _exact_binding(binding):
            return False
        assert type(seal) is ArithmeticEvidenceSeal and type(binding) is OpBinding
        candidate = binding.candidate
        if type(candidate) is not CandidateBinding:
            return False
        return (
            _binding_payload(binding, include_evidence=False)
            == _binding_payload(seal.binding, include_evidence=False)
            and candidate.evidence_id == seal.evidence_id
            and seal.evidence_id == canonical_arithmetic_evidence_id(seal)
        )
    except (AttributeError, TypeError, ValueError):
        return False


@dataclass(frozen=True, slots=True)
class ProgramSpecialization:
    leaf_ordinal: int
    artifact_launch_digest: str

    def __post_init__(self) -> None:
        if type(self.leaf_ordinal) is not int or self.leaf_ordinal < 0:
            raise TypeError("program specialization requires a non-negative leaf reference")
        if not _is_digest(self.artifact_launch_digest):
            raise ValueError("program specialization requires an artifact/launch digest")


def _canonical_ids(values: object, *, required: bool = False) -> bool:
    return (
        type(values) is tuple
        and (not required or bool(values))
        and all(type(value) is str and bool(value) and value == value.strip() for value in values)
        and len(set(values)) == len(values)
    )


@dataclass(frozen=True, slots=True)
class ProgramOccurrence:
    occurrence_id: str
    model_node_id: str
    layer_id: str
    specialization_ordinal: int
    input_buffer_ids: tuple[str, ...]
    output_buffer_ids: tuple[str, ...]
    weight_live_binding_digests: tuple[str, ...]
    state_read_effect_ids: tuple[str, ...]
    state_write_effect_ids: tuple[str, ...]
    dependency_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not all(
            type(value) is str and bool(value) and value == value.strip()
            for value in (self.occurrence_id, self.model_node_id, self.layer_id)
        ):
            raise ValueError("program occurrence identities must be canonical strings")
        if type(self.specialization_ordinal) is not int or self.specialization_ordinal < 0:
            raise TypeError("program occurrence requires a non-negative specialization reference")
        for name, values in (
            ("input buffers", self.input_buffer_ids),
            ("output buffers", self.output_buffer_ids),
            ("state reads", self.state_read_effect_ids),
            ("state writes", self.state_write_effect_ids),
            ("dependencies", self.dependency_ids),
        ):
            if not _canonical_ids(values):
                raise TypeError(f"program occurrence {name} must be immutable canonical ids")
        if type(self.weight_live_binding_digests) is not tuple or any(
            not _is_digest(value) for value in self.weight_live_binding_digests
        ):
            raise ValueError("program occurrence weight bindings require immutable digests")


@dataclass(frozen=True, slots=True)
class CompositeProgramManifest:
    """Actual ordered program dataflow exported inside the trusted process."""

    specializations: tuple[ProgramSpecialization, ...]
    occurrences: tuple[ProgramOccurrence, ...]
    external_input_buffer_ids: tuple[str, ...]
    external_output_buffer_ids: tuple[str, ...]
    program_source_digest: str
    execution_recipe_digest: str
    config_digest: str
    topology_digest: str
    partition_digest: str

    def __post_init__(self) -> None:
        if type(self.specializations) is not tuple or not self.specializations:
            raise ValueError("composite program requires a specialization table")
        if any(type(item) is not ProgramSpecialization for item in self.specializations):
            raise TypeError("composite specialization table requires exact typed entries")
        if type(self.occurrences) is not tuple or not self.occurrences:
            raise ValueError("composite program requires an ordered occurrence manifest")
        if any(type(item) is not ProgramOccurrence for item in self.occurrences):
            raise TypeError("composite occurrence manifest requires exact typed entries")
        for name, values in (
            ("external inputs", self.external_input_buffer_ids),
            ("external outputs", self.external_output_buffer_ids),
        ):
            if not _canonical_ids(values, required=True):
                raise TypeError(f"composite program {name} require immutable canonical ids")
        if any(
            not _is_digest(value)
            for value in (
                self.program_source_digest,
                self.execution_recipe_digest,
                self.config_digest,
                self.topology_digest,
                self.partition_digest,
            )
        ):
            raise ValueError("composite program identities must use content_digest digests")


def _occurrence_payload(item: ProgramOccurrence) -> dict[str, object]:
    return {
        "occurrence_id": item.occurrence_id,
        "model_node_id": item.model_node_id,
        "layer_id": item.layer_id,
        "specialization_ordinal": item.specialization_ordinal,
        "input_buffer_ids": item.input_buffer_ids,
        "output_buffer_ids": item.output_buffer_ids,
        "weight_live_binding_digests": item.weight_live_binding_digests,
        "state_read_effect_ids": item.state_read_effect_ids,
        "state_write_effect_ids": item.state_write_effect_ids,
        "dependency_ids": item.dependency_ids,
    }


def _manifest_payload(manifest: CompositeProgramManifest) -> dict[str, object]:
    return {
        "specializations": tuple(
            {
                "leaf_ordinal": item.leaf_ordinal,
                "artifact_launch_digest": item.artifact_launch_digest,
            }
            for item in manifest.specializations
        ),
        "occurrences": tuple(_occurrence_payload(item) for item in manifest.occurrences),
        "external_input_buffer_ids": manifest.external_input_buffer_ids,
        "external_output_buffer_ids": manifest.external_output_buffer_ids,
        "program_source_digest": manifest.program_source_digest,
        "execution_recipe_digest": manifest.execution_recipe_digest,
        "config_digest": manifest.config_digest,
        "topology_digest": manifest.topology_digest,
        "partition_digest": manifest.partition_digest,
    }


def validate_program_manifest(
    manifest: object,
    binding: OpBinding,
    children: tuple[OpBinding, ...],
    arithmetic: tuple[ArithmeticEvidenceSeal, ...],
    topology_digest: str,
) -> bool:
    try:
        if (
            type(manifest) is not CompositeProgramManifest
            or not _exact_binding(binding)
            or type(children) is not tuple
            or any(not _exact_binding(child) for child in children)
            or type(arithmetic) is not tuple
            or len(arithmetic) != len(children)
            or any(type(item) is not ArithmeticEvidenceSeal for item in arithmetic)
            or not _is_digest(topology_digest)
        ):
            return False
        CompositeProgramManifest.__post_init__(manifest)
        if (
            manifest.topology_digest != topology_digest
            or len(manifest.external_input_buffer_ids) != len(binding.signature.inputs)
            or len(manifest.external_output_buffer_ids) != len(binding.signature.outputs)
        ):
            return False
        child_signatures = tuple(child.signature for child in children)
        specialization_keys: list[tuple[int, str]] = []
        covered_leaves: set[int] = set()
        for specialization in manifest.specializations:
            if type(specialization) is not ProgramSpecialization:
                return False
            ProgramSpecialization.__post_init__(specialization)
            if specialization.leaf_ordinal >= len(child_signatures):
                return False
            leaf_evidence = arithmetic[specialization.leaf_ordinal]
            if specialization.artifact_launch_digest not in leaf_evidence.required_evidence_ids:
                return False
            specialization_keys.append(
                (specialization.leaf_ordinal, specialization.artifact_launch_digest)
            )
            covered_leaves.add(specialization.leaf_ordinal)
        if specialization_keys != sorted(set(specialization_keys)) or covered_leaves != set(
            range(len(child_signatures))
        ):
            return False
        seen_occurrences: set[str] = set()
        occurrence_order: list[str] = []
        producers: dict[str, str] = {}
        consumed: set[str] = set()
        external_inputs = set(manifest.external_input_buffer_ids)
        referenced_specializations: set[int] = set()
        for occurrence in manifest.occurrences:
            if type(occurrence) is not ProgramOccurrence:
                return False
            ProgramOccurrence.__post_init__(occurrence)
            if occurrence.occurrence_id in seen_occurrences:
                return False
            if occurrence.specialization_ordinal >= len(manifest.specializations):
                return False
            specialization = manifest.specializations[occurrence.specialization_ordinal]
            signature = child_signatures[specialization.leaf_ordinal]
            if len(occurrence.input_buffer_ids) != len(signature.inputs) or len(
                occurrence.output_buffer_ids
            ) != len(signature.outputs):
                return False
            if any(not _is_digest(item) for item in occurrence.weight_live_binding_digests):
                return False
            dependency_set: set[str] = set()
            for buffer_id in occurrence.input_buffer_ids:
                consumed.add(buffer_id)
                if buffer_id in external_inputs:
                    continue
                producer = producers.get(buffer_id)
                if producer is None:
                    return False
                dependency_set.add(producer)
            dependencies = tuple(item for item in occurrence_order if item in dependency_set)
            if occurrence.dependency_ids != dependencies:
                return False
            for buffer_id in occurrence.output_buffer_ids:
                if buffer_id in external_inputs or buffer_id in producers:
                    return False
                producers[buffer_id] = occurrence.occurrence_id
            seen_occurrences.add(occurrence.occurrence_id)
            occurrence_order.append(occurrence.occurrence_id)
            referenced_specializations.add(occurrence.specialization_ordinal)
        return referenced_specializations == set(range(len(manifest.specializations))) and (
            external_inputs <= consumed
            and all(item in producers for item in manifest.external_output_buffer_ids)
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _composite_topology(arithmetic: tuple[ArithmeticEvidenceSeal, ...]) -> str | None:
    if (
        type(arithmetic) is not tuple
        or not arithmetic
        or any(type(item) is not ArithmeticEvidenceSeal for item in arithmetic)
    ):
        return None
    topology = {item.topology_digest for item in arithmetic}
    environments = {
        (item.device_capability, item.driver_version, item.runtime_version) for item in arithmetic
    }
    if len(topology) != 1 or len(environments) != 1:
        return None
    return next(iter(topology))


def composite_evidence_material(
    binding: OpBinding,
    children: tuple[OpBinding, ...],
    arithmetic: tuple[ArithmeticEvidenceSeal, ...],
    manifest: CompositeProgramManifest,
) -> str | None:
    try:
        if (
            not _exact_binding(binding)
            or type(children) is not tuple
            or type(arithmetic) is not tuple
            or len(children) < 2
            or len(children) != len(arithmetic)
            or any(not _exact_binding(child) for child in children)
        ):
            return None
        signatures = tuple(child.signature for child in children)
        keys = tuple(_canonical_evidence_json(_signature_payload(item)) for item in signatures)
        if keys != tuple(sorted(keys)) or len(set(signatures)) != len(signatures):
            return None
        if any(
            child.signature.phase is not binding.signature.phase
            or child.signature.topology_id != binding.signature.topology_id
            or not validate_arithmetic_evidence(evidence, child)
            for child, evidence in zip(children, arithmetic, strict=True)
        ):
            return None
        topology = _composite_topology(arithmetic)
        if topology is None or not validate_program_manifest(
            manifest, binding, children, arithmetic, topology
        ):
            return None
        return _stable_evidence_digest(
            {
                "schema": "cubic.composite-arithmetic.v2",
                "binding": _binding_payload(binding, include_evidence=False),
                "children": tuple(
                    _binding_payload(child, include_evidence=True) for child in children
                ),
                "arithmetic": tuple(item.evidence_id for item in arithmetic),
                "manifest": _manifest_payload(manifest),
            }
        )
    except (AttributeError, TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True, init=False)
class CompositeArithmeticEvidenceSeal:
    binding: OpBinding
    children: tuple[OpBinding, ...]
    arithmetic: tuple[ArithmeticEvidenceSeal, ...]
    manifest: CompositeProgramManifest
    evidence_id: str

    def __init__(self) -> None:
        raise TypeError("composite arithmetic seals require the guardrail factory")

    @property
    def topology_digest(self) -> str:
        return self.arithmetic[0].topology_digest


def validate_composite_arithmetic_evidence(
    seal: object,
    binding: object,
    expected_children: tuple[OpBinding, ...] | None = None,
    expected_manifest: CompositeProgramManifest | None = None,
) -> bool:
    try:
        if (
            type(seal) is not CompositeArithmeticEvidenceSeal
            or not _exact_binding(binding)
            or not _is_digest(seal.evidence_id)
        ):
            return False
        assert type(binding) is OpBinding
        candidate = binding.candidate
        material = composite_evidence_material(
            seal.binding, seal.children, seal.arithmetic, seal.manifest
        )
        if material is None:
            return False
        if expected_children is not None and (
            type(expected_children) is not tuple
            or any(not _exact_binding(child) for child in expected_children)
            or tuple(_binding_payload(child, include_evidence=True) for child in expected_children)
            != tuple(_binding_payload(child, include_evidence=True) for child in seal.children)
        ):
            return False
        if expected_manifest is not None and (
            not validate_program_manifest(
                expected_manifest,
                binding,
                seal.children if expected_children is None else expected_children,
                seal.arithmetic,
                seal.topology_digest,
            )
            or _manifest_payload(expected_manifest) != _manifest_payload(seal.manifest)
        ):
            return False
        return (
            type(candidate) is CandidateBinding
            and _binding_payload(seal.binding, include_evidence=True)
            == _binding_payload(binding, include_evidence=True)
            and candidate.evidence_id == seal.evidence_id
            and material == seal.evidence_id
        )
    except (AttributeError, TypeError, ValueError):
        return False


ArithmeticEvidenceProjection = TypeAliasType("ArithmeticEvidenceProjection", ArithmeticEvidenceSeal | CompositeArithmeticEvidenceSeal)


def _projection_signature(item: ArithmeticEvidenceProjection) -> OpSignature:
    if type(item) not in (ArithmeticEvidenceSeal, CompositeArithmeticEvidenceSeal):
        raise TypeError("operator evidence accepts only exact arithmetic seal types")
    return item.binding.signature


def canonical_phase_evidence_id(
    phase: BatchPhase, arithmetic: tuple[ArithmeticEvidenceProjection, ...]
) -> str:
    if type(phase) is not BatchPhase or type(arithmetic) is not tuple:
        raise TypeError("phase evidence requires exact immutable contract types")
    selected = tuple(
        item.evidence_id for item in arithmetic if _projection_signature(item).phase is phase
    )
    if not selected:
        raise ValueError(f"phase {phase.value!r} has no arithmetic evidence")
    return _stable_evidence_digest(
        {"schema": "cubic.phase-arithmetic.v2", "phase": phase.value, "evidence": selected}
    )


def operator_evidence_material(
    execution_plan_id: str,
    bindings: tuple[OpBinding, ...],
    graph_buckets: tuple[GraphBucket, ...],
    arithmetic: tuple[ArithmeticEvidenceProjection, ...],
) -> tuple[tuple[tuple[BatchPhase, str], ...], str] | None:
    try:
        if (
            type(execution_plan_id) is not str
            or not execution_plan_id
            or type(bindings) is not tuple
            or not bindings
            or type(graph_buckets) is not tuple
            or not graph_buckets
            or type(arithmetic) is not tuple
            or len(arithmetic) != len(bindings)
            or any(not _exact_binding(binding) for binding in bindings)
            or any(not _exact_graph_bucket(bucket) for bucket in graph_buckets)
        ):
            return None
        if any(
            not (
                validate_arithmetic_evidence(item, binding)
                if type(item) is ArithmeticEvidenceSeal
                else validate_composite_arithmetic_evidence(item, binding)
                if type(item) is CompositeArithmeticEvidenceSeal
                else False
            )
            for item, binding in zip(arithmetic, bindings, strict=True)
        ):
            return None
        phases = tuple(sorted({binding.signature.phase for binding in bindings}, key=str))
        phase_ids = tuple(
            (phase, canonical_phase_evidence_id(phase, arithmetic)) for phase in phases
        )
        phase_map = dict(phase_ids)
        if {bucket.phase for bucket in graph_buckets} != set(phases) or any(
            bucket.evidence_id != phase_map[bucket.phase] for bucket in graph_buckets
        ):
            return None
        if len({item.topology_digest for item in arithmetic}) != 1:
            return None
        evidence_id = _stable_evidence_digest(
            {
                "schema": "cubic.operator-evidence.v2",
                "execution_plan_id": execution_plan_id,
                "bindings": tuple(
                    _binding_payload(item, include_evidence=True) for item in bindings
                ),
                "buckets": tuple(_graph_bucket_payload(item) for item in graph_buckets),
                "arithmetic": tuple(item.evidence_id for item in arithmetic),
                "phases": tuple((phase.value, item) for phase, item in phase_ids),
            }
        )
        return phase_ids, evidence_id
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _graph_bucket_payload(bucket: GraphBucket) -> dict[str, object]:
    return {
        "bucket_id": bucket.bucket_id,
        "template_id": bucket.template_id,
        "scope_id": bucket.scope_id,
        "replay_abi_id": bucket.replay_abi_id,
        "topology_id": bucket.topology_id,
        "partition_id": bucket.partition_id,
        "phase": bucket.phase.value,
        "capacity_items": bucket.capacity_items,
        "logical_m_values": bucket.logical_m_values,
        "captured_m": bucket.captured_m,
        "row_map_schema_id": bucket.row_map_schema_id,
        "scratch_schema_id": bucket.scratch_schema_id,
        "graph_safety_contract_ids": bucket.graph_safety_contract_ids,
        "evidence_id": bucket.evidence_id,
    }


@dataclass(frozen=True, slots=True, init=False)
class OperatorEvidenceSeal:
    execution_plan_id: str
    arithmetic: tuple[ArithmeticEvidenceProjection, ...]
    phase_evidence_ids: tuple[tuple[BatchPhase, str], ...]
    evidence_id: str

    def __init__(self) -> None:
        raise TypeError("operator evidence seals require the guardrail factory")


def validate_operator_evidence(
    seal: object,
    execution_plan_id: object,
    bindings: object,
    graph_buckets: object,
) -> bool:
    try:
        if (
            type(seal) is not OperatorEvidenceSeal
            or type(execution_plan_id) is not str
            or type(bindings) is not tuple
            or type(graph_buckets) is not tuple
            or type(seal.execution_plan_id) is not str
            or type(seal.arithmetic) is not tuple
            or type(seal.phase_evidence_ids) is not tuple
            or any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not BatchPhase
                or not _is_digest(item[1])
                for item in seal.phase_evidence_ids
            )
            or not _is_digest(seal.evidence_id)
            or seal.execution_plan_id != execution_plan_id
            or any(not _exact_binding(binding) for binding in bindings)
            or any(not _exact_graph_bucket(bucket) for bucket in graph_buckets)
        ):
            return False
        material = operator_evidence_material(
            execution_plan_id, bindings, graph_buckets, seal.arithmetic
        )
        return material == (seal.phase_evidence_ids, seal.evidence_id)
    except (AttributeError, TypeError, ValueError):
        return False


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    execution_plan_id: str
    mode: ExecutionMode
    bindings: tuple[OpBinding, ...]
    graph_buckets: tuple[GraphBucket, ...] = ()
    operator_evidence: OperatorEvidenceSeal | None = None

    def __post_init__(self) -> None:
        if not self.execution_plan_id or not self.bindings:
            raise ValueError("execution plan identity and bindings are required")
        if len({binding.signature for binding in self.bindings}) != len(self.bindings):
            raise ValueError("an execution plan cannot bind an op signature twice")
        topology_ids = {binding.signature.topology_id for binding in self.bindings}
        if len(topology_ids) != 1:
            raise ValueError("one execution plan must bind exactly one topology")
        if self.mode is ExecutionMode.REFERENCE:
            if any(binding.candidate is not None for binding in self.bindings):
                raise ValueError("reference execution cannot contain performance candidates")
            if self.graph_buckets:
                raise ValueError("reference execution cannot contain CUDA Graph buckets")
            if self.operator_evidence is not None:
                raise ValueError("reference execution cannot contain operator evidence")
            return

        if self.mode is not ExecutionMode.PERFORMANCE:
            raise ValueError("unsupported execution mode")
        if any(binding.candidate is None for binding in self.bindings):
            raise ValueError("performance execution cannot contain a reference fallback")
        if any(not _exact_binding(binding) for binding in self.bindings):
            raise ValueError("performance execution requires exact canonical op bindings")
        if any(
            binding.candidate is None or binding.candidate.graph_safety is None
            for binding in self.bindings
        ):
            raise ValueError(
                "performance execution requires graph-safety evidence for every candidate"
            )
        if not self.graph_buckets:
            raise ValueError("performance execution requires certified CUDA Graph buckets")
        if any(not _exact_graph_bucket(bucket) for bucket in self.graph_buckets):
            raise ValueError("performance execution requires exact canonical graph buckets")
        (topology_id,) = topology_ids
        if any(bucket.topology_id != topology_id for bucket in self.graph_buckets):
            raise ValueError("performance graph bucket topology must match every op binding")
        partition_ids = {bucket.partition_id for bucket in self.graph_buckets}
        if len(partition_ids) != 1:
            raise ValueError("one performance execution plan must bind exactly one partition")

        bucket_ids = tuple(bucket.bucket_id for bucket in self.graph_buckets)
        if len(set(bucket_ids)) != len(bucket_ids):
            raise ValueError("graph bucket ids must be unique")
        template_ids = tuple(bucket.template_id for bucket in self.graph_buckets)
        if len(set(template_ids)) != len(template_ids):
            raise ValueError("each graph bucket must name one distinct captured template")

        binding_phases = {binding.signature.phase for binding in self.bindings}
        bucket_phases = {bucket.phase for bucket in self.graph_buckets}
        if bucket_phases != binding_phases:
            raise ValueError("performance graph buckets must exactly cover all binding phases")
        for phase in bucket_phases:
            phase_buckets = tuple(bucket for bucket in self.graph_buckets if bucket.phase is phase)
            domains = {
                (
                    bucket.scope_id,
                    bucket.replay_abi_id,
                    bucket.topology_id,
                    bucket.partition_id,
                    bucket.capacity_items,
                )
                for bucket in phase_buckets
            }
            for domain in domains:
                domain_buckets = (
                    bucket
                    for bucket in phase_buckets
                    if (
                        bucket.scope_id,
                        bucket.replay_abi_id,
                        bucket.topology_id,
                        bucket.partition_id,
                        bucket.capacity_items,
                    )
                    == domain
                )
                claimed_values: set[int] = set()
                for bucket in domain_buckets:
                    if claimed_values.intersection(bucket.logical_m_values):
                        raise ValueError("graph bucket logical M domains cannot overlap")
                    claimed_values.update(bucket.logical_m_values)
            phase_contracts = {
                binding.candidate.graph_safety.contract_id
                for binding in self.bindings
                if binding.signature.phase is phase
                and binding.candidate is not None
                and binding.candidate.graph_safety is not None
            }
            expected_contract_ids = tuple(sorted(phase_contracts))
            if any(
                bucket.graph_safety_contract_ids != expected_contract_ids
                for bucket in phase_buckets
            ):
                raise ValueError(
                    "every graph bucket must bind the complete phase graph-safety contract set"
                )
        if type(self.operator_evidence) is not OperatorEvidenceSeal or not (
            validate_operator_evidence(
                self.operator_evidence,
                self.execution_plan_id,
                self.bindings,
                self.graph_buckets,
            )
        ):
            raise ValueError(
                "performance execution requires complete typed arithmetic evidence coverage"
            )

    @property
    def graph_enabled(self) -> bool:
        return self.mode is ExecutionMode.PERFORMANCE

    def bucket_for(
        self,
        *,
        scope_id: str,
        replay_abi_id: str,
        phase: BatchPhase,
        logical_m: int,
        capacity_items: CapacityItems,
        topology_id: str,
        partition_id: str,
    ) -> GraphBucket:
        """Resolve a complete replay ABI to one bucket; never fall back to eager."""

        matches = tuple(
            bucket
            for bucket in self.graph_buckets
            if bucket.admits(
                scope_id=scope_id,
                replay_abi_id=replay_abi_id,
                phase=phase,
                logical_m=logical_m,
                capacity_items=capacity_items,
                topology_id=topology_id,
                partition_id=partition_id,
            )
        )
        if len(matches) != 1:
            raise ValueError(
                "logical batch has no unique certified performance graph bucket; "
                "eager/exact-M fallback is forbidden"
            )
        return matches[0]

    def validate_launch(self, batch_plan: BatchPlan, launch_plan: LaunchPlan) -> None:
        """Reject a launch assembled from a different plan, topology, or bucket."""

        if self.mode is not ExecutionMode.PERFORMANCE:
            raise ValueError("reference execution has no CUDA Graph launch plan")
        if launch_plan.execution_plan_id != self.execution_plan_id:
            raise ValueError("launch references a different execution plan")
        if launch_plan.source_plan_id != batch_plan.plan_id:
            raise ValueError("launch references a different immutable batch plan")
        if launch_plan.logical_m != batch_plan.shape.logical_m:
            raise ValueError("launch logical M differs from its batch plan")
        if launch_plan.bucket.capacity_items != batch_plan.shape.capacity_items:
            raise ValueError("launch graph capacity profile differs from its batch plan")
        if launch_plan.bucket.phase is not batch_plan.phase:
            raise ValueError("launch graph phase differs from its batch plan")
        if launch_plan.bucket.topology_id != batch_plan.topology_id:
            raise ValueError("launch graph topology differs from its batch plan")
        if launch_plan.bucket.partition_id != batch_plan.partition_id:
            raise ValueError("launch graph partition differs from its batch plan")
        expected_bucket = self.bucket_for(
            scope_id=launch_plan.bucket.scope_id,
            replay_abi_id=launch_plan.bucket.replay_abi_id,
            phase=batch_plan.phase,
            logical_m=batch_plan.shape.logical_m,
            capacity_items=batch_plan.shape.capacity_items,
            topology_id=batch_plan.topology_id,
            partition_id=batch_plan.partition_id,
        )
        if launch_plan.bucket != expected_bucket:
            raise ValueError("launch does not use the execution plan's certified graph bucket")
