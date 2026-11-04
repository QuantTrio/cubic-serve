"""Minimal eager NCCL execution for one sealed single-node partition."""

from __future__ import annotations

from typing_extensions import Self

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from types import TracebackType
from typing import cast

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from cubic_serve.contracts.execution import (
    GroupKind,
    PipelineEdge,
    RankGroup,
)
from cubic_serve.contracts.model import ScalarDType
from cubic_serve.executor.parallel.topology import SealedPartition


@dataclass(frozen=True, slots=True)
class ExpertDispatch:
    """Rows received by one EP rank plus the exact inverse routing metadata."""

    group_id: str
    owner_rank: int
    values: Tensor
    send_splits: tuple[int, ...]
    receive_splits: tuple[int, ...]
    restore_order: Tensor
    source_row_count: int


@dataclass(slots=True)
class EagerNcclPartition:
    """NCCL groups and eager operations authorized by one sealed partition."""

    sealed: SealedPartition
    rank: int
    creation_order: tuple[str, ...]
    _groups: dict[str, ProcessGroup]
    _next_sequence: dict[tuple[str, str], int] = field(default_factory=dict)
    _active_epoch: str | None = None
    _active_lane_id: str | None = None
    _epoch_failed: bool = False
    _closed: bool = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def device(self) -> torch.device:
        placement = next(
            placement
            for placement in self.sealed.topology.placements
            if placement.rank == self.rank
        )
        return torch.device("cuda", placement.device_index)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not dist.is_initialized():
            return
        for group_id in reversed(self.creation_order):
            group = self._groups.get(group_id)
            if group is not None:
                dist.destroy_process_group(group)
        dist.destroy_process_group()

    def tp_all_reduce(self, step_id: str, value: Tensor) -> Tensor:
        group, process_group = self._collective(
            step_id,
            operation="all_reduce",
            kind=GroupKind.TENSOR,
        )
        self._require_bf16(value, "TP all-reduce", allow_empty=False)
        if self.rank not in group.ranks:
            raise RuntimeError("TP all-reduce rank is outside its declared group")
        # NCCL is transport only for the eager numerical reference.  Its
        # reduction tree is deliberately unspecified and can change with the
        # input address.  Reconstruct the declared TP arithmetic independently:
        # canonical group-rank order, FP32 additions, one BF16 write boundary.
        gathered = torch.empty(
            (len(group.ranks), value.numel()),
            dtype=torch.bfloat16,
            device=value.device,
        )
        dist.all_gather_single(gathered.view(-1), value.view(-1), group=process_group)
        total = torch.zeros(value.numel(), dtype=torch.float32, device=value.device)
        for group_rank in range(len(group.ranks)):
            total.add_(gathered[group_rank].float())
        value.copy_(total.view_as(value))
        return value

    def tp_all_gather(self, step_id: str, value: Tensor, *, dim: int) -> Tensor:
        """Gather equal rank shards in canonical group-rank order.

        This eager primitive admits BF16 weights/state and the model-declared
        FP32 recurrent state. It is intentionally unavailable during graph
        capture and does not pad unequal shards.
        """

        group, process_group = self._collective(
            step_id,
            operation="all_gather",
            kind=GroupKind.TENSOR,
        )
        self._require_reference_float(value, "TP all-gather", allow_empty=True)
        if self.rank not in group.ranks:
            raise RuntimeError("TP all-gather rank is outside its declared group")
        if value.ndim == 0 or not -value.ndim <= dim < value.ndim:
            raise ValueError("TP all-gather dimension is out of range")
        normalized_dim = dim % value.ndim
        if value.numel() == 0:
            shape = list(value.shape)
            shape[normalized_dim] *= len(group.ranks)
            return value.new_empty(tuple(shape))

        leading = value.movedim(normalized_dim, 0).contiguous()
        gathered = leading.new_empty((leading.shape[0] * len(group.ranks), *leading.shape[1:]))
        dist.all_gather_single(gathered, leading, group=process_group)
        return gathered.movedim(0, normalized_dim).contiguous()

    @contextmanager
    def collective_epoch(self, epoch_id: str, *, lane_id: str | None = None) -> Iterator[None]:
        """Reuse one sealed collective recipe for exactly one eager forward."""

        self._require_live()
        if not epoch_id:
            raise ValueError("collective epoch identity is required")
        if self._active_epoch is not None or self._next_sequence:
            raise RuntimeError("a collective epoch is already active or incomplete")
        if lane_id is not None:
            self._require_lane_agreement(lane_id)
        if lane_id is not None and not any(
            step.lane_id == lane_id for step in self.sealed.partition.collectives
        ):
            raise ValueError("collective epoch lane is absent from the sealed partition")
        self._active_epoch = epoch_id
        self._active_lane_id = lane_id
        try:
            yield
            self._finish_collective_epoch(epoch_id)
        except BaseException:
            self._epoch_failed = True
            raise
        finally:
            self._active_epoch = None
            self._active_lane_id = None

    def _require_lane_agreement(self, lane_id: str) -> None:
        if not lane_id:
            raise ValueError("collective epoch lane must be non-empty")
        payload = f"{self.sealed.partition_digest}:{lane_id}".encode()
        fingerprint = int.from_bytes(
            hashlib.blake2s(payload).digest()[:8], "big"
        ) & ((1 << 63) - 1)
        local = torch.tensor([fingerprint], dtype=torch.int64, device=self.device)
        observed = torch.empty(
            self.sealed.topology.world_size,
            dtype=torch.int64,
            device=self.device,
        )
        dist.all_gather_single(observed, local)
        if not bool((observed == local[0]).all().item()):
            self._epoch_failed = True
            raise RuntimeError("collective epoch lane or sealed recipe differs across ranks")

    def pipeline_send(self, edge_id: str, value: Tensor) -> None:
        edge, group, process_group = self._pipeline_edge(edge_id)
        if self.rank != edge.source_rank:
            raise RuntimeError("pipeline send is only valid on the declared source rank")
        self._require_bf16(value, "pipeline send", allow_empty=False)
        dist.send(
            value,
            group=process_group,
            group_dst=group.ranks.index(edge.target_rank),
        )

    def pipeline_recv(self, edge_id: str, shape: tuple[int, ...]) -> Tensor:
        edge, group, process_group = self._pipeline_edge(edge_id)
        if self.rank != edge.target_rank:
            raise RuntimeError("pipeline receive is only valid on the declared target rank")
        if not shape or any(type(size) is not int or size <= 0 for size in shape):
            raise ValueError("pipeline receive shape must contain positive integer dimensions")
        value = torch.empty(shape, dtype=torch.bfloat16, device=self.device)
        dist.recv(
            value,
            group=process_group,
            group_src=group.ranks.index(edge.source_rank),
        )
        return value

    def ep_dispatch(
        self,
        step_id: str,
        values: Tensor,
        destination_ranks: Tensor,
    ) -> ExpertDispatch:
        group, process_group = self._collective(
            step_id,
            operation="all_to_all",
            kind=GroupKind.EXPERT,
        )
        self._require_bf16(values, "EP dispatch", allow_empty=True)
        if values.ndim < 1:
            raise ValueError("EP dispatch values require a row axis")
        if (
            destination_ranks.dtype is not torch.int64
            or destination_ranks.device != self.device
            or destination_ranks.shape != (values.shape[0],)
            or not destination_ranks.is_contiguous()
        ):
            raise ValueError("EP destinations must be contiguous CUDA int64 with one rank per row")

        destinations = tuple(int(rank) for rank in destination_ranks.tolist())
        if any(rank not in group.ranks for rank in destinations):
            raise ValueError("EP destination lies outside the stage-local expert group")
        restore_indices = tuple(
            row
            for destination in group.ranks
            for row, rank in enumerate(destinations)
            if rank == destination
        )
        send_splits = tuple(destinations.count(destination) for destination in group.ranks)
        restore_order = torch.tensor(restore_indices, dtype=torch.int64, device=self.device)
        send_values = values.index_select(0, restore_order).contiguous()

        send_counts = torch.tensor(send_splits, dtype=torch.int64, device=self.device)
        receive_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(receive_counts, send_counts, group=process_group)
        receive_splits = tuple(int(count) for count in receive_counts.tolist())
        received = values.new_empty((sum(receive_splits), *values.shape[1:]))
        dist.all_to_all_single(
            received,
            send_values,
            output_split_sizes=list(receive_splits),
            input_split_sizes=list(send_splits),
            group=process_group,
        )
        return ExpertDispatch(
            group_id=group.group_id,
            owner_rank=self.rank,
            values=received,
            send_splits=send_splits,
            receive_splits=receive_splits,
            restore_order=restore_order,
            source_row_count=values.shape[0],
        )

    def ep_combine(
        self,
        step_id: str,
        dispatch: ExpertDispatch,
        expert_values: Tensor,
    ) -> Tensor:
        group, process_group = self._collective(
            step_id,
            operation="all_to_all",
            kind=GroupKind.EXPERT,
        )
        self._require_bf16(expert_values, "EP combine", allow_empty=True)
        if dispatch.group_id != group.group_id or dispatch.owner_rank != self.rank:
            raise ValueError("EP combine metadata belongs to another group or rank")
        if expert_values.shape != dispatch.values.shape:
            raise ValueError("expert outputs must preserve dispatched row shape")

        returned = expert_values.new_empty(
            (sum(dispatch.send_splits), *expert_values.shape[1:])
        )
        dist.all_to_all_single(
            returned,
            expert_values.contiguous(),
            output_split_sizes=list(dispatch.send_splits),
            input_split_sizes=list(dispatch.receive_splits),
            group=process_group,
        )
        combined = expert_values.new_empty(
            (dispatch.source_row_count, *expert_values.shape[1:])
        )
        combined.index_copy_(0, dispatch.restore_order, returned)
        return combined

    def _require_live(self) -> None:
        if self._closed or not dist.is_initialized():
            raise RuntimeError("NCCL partition is closed")
        if self._epoch_failed:
            raise RuntimeError("NCCL partition cannot continue after an incomplete epoch")
        if dist.get_rank() != self.rank or torch.cuda.current_device() != self.device.index:
            raise RuntimeError("process rank or CUDA device drifted from ParallelTopology")
        compiler = getattr(torch, "compiler", None)
        if compiler is not None and compiler.is_compiling():
            raise RuntimeError("NCCL partition supports eager execution only")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("NCCL partition does not permit CUDA graph capture")

    def _require_reference_float(
        self,
        value: Tensor,
        operation: str,
        *,
        allow_empty: bool,
    ) -> None:
        self._require_live()
        if (
            value.dtype not in (torch.bfloat16, torch.float32)
            or value.device != self.device
            or value.layout is not torch.strided
            or not value.is_contiguous()
        ):
            raise ValueError(
                f"{operation} requires contiguous BF16/FP32 on the rank device"
            )
        if not allow_empty and value.numel() == 0:
            raise ValueError(f"{operation} requires a non-empty tensor")

    def _require_bf16(self, value: Tensor, operation: str, *, allow_empty: bool) -> None:
        self._require_live()
        if (
            value.dtype is not torch.bfloat16
            or value.device != self.device
            or value.layout is not torch.strided
            or not value.is_contiguous()
        ):
            raise ValueError(f"{operation} requires a contiguous BF16 tensor on the rank device")
        if not allow_empty and value.numel() == 0:
            raise ValueError(f"{operation} requires a non-empty tensor")

    def _finish_collective_epoch(self, epoch_id: str) -> None:
        if self._active_epoch != epoch_id:
            raise RuntimeError("collective epoch identity drifted")
        expected: dict[tuple[str, str], int] = {}
        for step in self.sealed.partition.collectives:
            if step.group_id in self._groups and (
                self._active_lane_id is None or step.lane_id == self._active_lane_id
            ):
                key = (step.group_id, step.lane_id)
                expected[key] = expected.get(key, 0) + 1
        if self._next_sequence != expected:
            raise RuntimeError(
                "collective epoch did not consume its complete sealed recipe: "
                f"actual={self._next_sequence}, expected={expected}"
            )
        self._next_sequence.clear()

    def _collective(
        self,
        step_id: str,
        *,
        operation: str,
        kind: GroupKind,
    ) -> tuple[RankGroup, ProcessGroup]:
        self._require_live()
        matches = tuple(
            step
            for step in self.sealed.partition.collectives
            if step.step_id == step_id
            and (self._active_lane_id is None or step.lane_id == self._active_lane_id)
        )
        if len(matches) != 1:
            raise ValueError(f"unknown collective step {step_id!r}")
        step = matches[0]
        if step.operation != operation:
            raise ValueError(f"collective step {step_id!r} is not {operation}")
        group = self._group(step.group_id, kind)
        if self.rank not in group.ranks:
            raise RuntimeError(f"rank {self.rank} is outside collective group {group.group_id!r}")
        key = (step.group_id, step.lane_id)
        expected = self._next_sequence.get(key, 0)
        if step.sequence_no != expected:
            raise RuntimeError(
                f"collective step {step_id!r} has sequence {step.sequence_no}; expected {expected}"
            )
        self._next_sequence[key] = expected + 1
        return group, self._groups[group.group_id]

    def _pipeline_edge(
        self,
        edge_id: str,
    ) -> tuple[PipelineEdge, RankGroup, ProcessGroup]:
        self._require_live()
        matches = tuple(
            edge for edge in self.sealed.partition.pipeline_edges if edge.edge_id == edge_id
        )
        if len(matches) != 1:
            raise ValueError(f"unknown pipeline edge {edge_id!r}")
        edge = matches[0]
        if edge.tensor_dtype is not ScalarDType.BF16:
            raise ValueError("the eager NCCL pipeline slice supports BF16 edges only")
        groups = tuple(
            group
            for group in self.sealed.topology.groups
            if group.kind is GroupKind.PIPELINE
            and edge.source_rank in group.ranks
            and edge.target_rank in group.ranks
        )
        if len(groups) != 1:
            raise RuntimeError("pipeline edge does not resolve to exactly one tensor lane")
        group = groups[0]
        return edge, group, self._groups[group.group_id]

    def _group(self, group_id: str, kind: GroupKind) -> RankGroup:
        matches = tuple(
            group for group in self.sealed.topology.groups if group.group_id == group_id
        )
        if len(matches) != 1:
            raise ValueError(f"unknown parallel group {group_id!r}")
        group = matches[0]
        if group.kind is not kind:
            raise ValueError(f"parallel group {group_id!r} is not {kind.value}")
        return group


def initialize_eager_nccl(
    sealed: SealedPartition,
    *,
    rank: int,
    init_method: str,
    timeout_seconds: int = 60,
) -> EagerNcclPartition:
    """Bind one process to its declared device and create every group canonically."""

    if not dist.is_available() or not dist.is_nccl_available() or not torch.cuda.is_available():
        raise RuntimeError("eager NCCL requires CUDA and torch.distributed NCCL support")
    if dist.is_initialized():
        raise RuntimeError("eager NCCL initialization requires an uninitialized process group")
    if type(rank) is not int or not 0 <= rank < sealed.topology.world_size:
        raise ValueError("rank lies outside ParallelTopology")
    if not init_method:
        raise ValueError("an explicit distributed init_method is required")
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive integer")

    placement = next(
        placement for placement in sealed.topology.placements if placement.rank == rank
    )
    if placement.device_index >= torch.cuda.device_count():
        raise RuntimeError("ParallelTopology references an unavailable CUDA device")
    torch.cuda.set_device(placement.device_index)
    timeout = timedelta(seconds=timeout_seconds)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=sealed.topology.world_size,
        rank=rank,
        timeout=timeout,
        device_id=torch.device("cuda", placement.device_index),
    )

    creation_order = tuple(
        group.group_id for group in sorted(sealed.topology.groups, key=lambda item: item.group_id)
    )
    member_groups: dict[str, ProcessGroup] = {}
    try:
        group_by_id = {group.group_id: group for group in sealed.topology.groups}
        for group_id in creation_order:
            contract = group_by_id[group_id]
            process_group = dist.new_group(
                ranks=list(contract.ranks),
                backend="nccl",
                timeout=timeout,
                use_local_synchronization=False,
                group_desc=group_id,
            )
            if rank in contract.ranks:
                member_groups[group_id] = cast(ProcessGroup, process_group)
    except BaseException:
        for process_group in reversed(tuple(member_groups.values())):
            dist.destroy_process_group(process_group)
        dist.destroy_process_group()
        raise

    return EagerNcclPartition(
        sealed=sealed,
        rank=rank,
        creation_order=creation_order,
        _groups=member_groups,
    )
