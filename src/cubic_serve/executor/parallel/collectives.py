"""Synchronous eager semantics for single-node parallel collectives.

These functions are the correctness reference for later NCCL candidates. They execute in one
process, preserve canonical rank/row order, and intentionally provide no overlap or performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from cubic_serve.contracts.execution import RankGroup
from cubic_serve.executor.runner import require_reference_execution


@dataclass(frozen=True, slots=True)
class RankTensor:
    rank: int
    value: Tensor


@dataclass(frozen=True, slots=True)
class RoutedRows:
    destination_rank: int
    values: Tensor
    source_ranks: Tensor
    source_rows: Tensor

    def __post_init__(self) -> None:
        row_count = self.values.shape[0]
        if self.values.ndim < 1:
            raise ValueError("routed values require a row axis")
        if self.source_ranks.shape != (row_count,) or self.source_rows.shape != (row_count,):
            raise ValueError("route provenance must have one entry per routed row")
        if self.source_ranks.dtype is not torch.int64 or self.source_rows.dtype is not torch.int64:
            raise TypeError("route provenance must use int64")


def _ordered(
    group: RankGroup,
    shards: tuple[RankTensor, ...],
    *,
    variable_rows: bool = False,
) -> tuple[RankTensor, ...]:
    require_reference_execution()
    by_rank = {shard.rank: shard for shard in shards}
    if len(by_rank) != len(shards) or set(by_rank) != set(group.ranks):
        raise ValueError("collective shards must contain every group rank exactly once")
    ordered = tuple(by_rank[rank] for rank in group.ranks)
    first = ordered[0].value

    def compatible_shape(value: Tensor) -> bool:
        if variable_rows:
            return value.ndim == first.ndim and value.shape[1:] == first.shape[1:]
        return value.shape == first.shape

    if any(
        not compatible_shape(shard.value)
        or shard.value.dtype != first.dtype
        or shard.value.device != first.device
        or shard.value.layout != first.layout
        for shard in ordered[1:]
    ):
        raise ValueError("collective shard structure must match across ranks")
    return ordered


def all_reduce_sum(
    group: RankGroup,
    shards: tuple[RankTensor, ...],
    *,
    accumulator_dtype: torch.dtype = torch.float32,
) -> tuple[RankTensor, ...]:
    ordered = _ordered(group, shards)
    if not ordered[0].value.is_floating_point():
        raise TypeError("reference sum requires floating tensors")
    accumulated = torch.zeros_like(ordered[0].value, dtype=accumulator_dtype)
    for shard in ordered:
        accumulated = accumulated + shard.value.to(accumulator_dtype)
    result = accumulated.to(ordered[0].value.dtype)
    return tuple(RankTensor(rank, result.clone()) for rank in group.ranks)


def all_gather(
    group: RankGroup,
    shards: tuple[RankTensor, ...],
    *,
    dim: int,
) -> tuple[RankTensor, ...]:
    ordered = _ordered(group, shards)
    if not -ordered[0].value.ndim <= dim < ordered[0].value.ndim:
        raise ValueError("all-gather dimension is out of range")
    gathered = torch.cat(tuple(shard.value for shard in ordered), dim=dim)
    return tuple(RankTensor(rank, gathered.clone()) for rank in group.ranks)


def send_forward(value: Tensor, *, source_rank: int, target_rank: int) -> RankTensor:
    require_reference_execution()
    if min(source_rank, target_rank) < 0 or source_rank == target_rank:
        raise ValueError("pipeline send requires distinct non-negative ranks")
    return RankTensor(target_rank, value.detach().clone())


def all_to_all_dispatch(
    group: RankGroup,
    shards: tuple[RankTensor, ...],
    destination_ranks: tuple[RankTensor, ...],
) -> tuple[RoutedRows, ...]:
    """Route rows in canonical ``(source rank, source row)`` order."""

    ordered = _ordered(group, shards, variable_rows=True)
    destinations_by_rank = {item.rank: item.value for item in destination_ranks}
    if set(destinations_by_rank) != set(group.ranks) or len(destinations_by_rank) != len(
        destination_ranks
    ):
        raise ValueError("destination maps must contain every group rank exactly once")

    trailing_shape = ordered[0].value.shape[1:]
    values_by_destination: dict[int, list[Tensor]] = {rank: [] for rank in group.ranks}
    sources_by_destination: dict[int, list[int]] = {rank: [] for rank in group.ranks}
    rows_by_destination: dict[int, list[int]] = {rank: [] for rank in group.ranks}
    for shard in ordered:
        if shard.value.ndim < 1:
            raise ValueError("all-to-all inputs require a row axis")
        destination = destinations_by_rank[shard.rank]
        if destination.dtype is not torch.int64 or destination.shape != (shard.value.shape[0],):
            raise ValueError("destination map must be int64 with one entry per source row")
        for source_row, target in enumerate(destination.tolist()):
            if target not in values_by_destination:
                raise ValueError(f"destination rank {target} is outside the expert group")
            values_by_destination[target].append(shard.value[source_row].detach().clone())
            sources_by_destination[target].append(shard.rank)
            rows_by_destination[target].append(source_row)

    routed: list[RoutedRows] = []
    for rank in group.ranks:
        rows = values_by_destination[rank]
        if rows:
            values = torch.stack(rows, dim=0)
        else:
            values = ordered[0].value.new_empty((0, *trailing_shape))
        routed.append(
            RoutedRows(
                destination_rank=rank,
                values=values,
                source_ranks=torch.tensor(sources_by_destination[rank], dtype=torch.int64),
                source_rows=torch.tensor(rows_by_destination[rank], dtype=torch.int64),
            )
        )
    return tuple(routed)


def all_to_all_combine(
    group: RankGroup,
    routed_outputs: tuple[RoutedRows, ...],
    *,
    source_row_counts: tuple[tuple[int, int], ...],
) -> tuple[RankTensor, ...]:
    """Invert dispatch exactly once for every source row."""

    require_reference_execution()
    by_destination = {item.destination_rank: item for item in routed_outputs}
    if set(by_destination) != set(group.ranks) or len(by_destination) != len(routed_outputs):
        raise ValueError("routed outputs must contain every destination rank exactly once")
    row_counts = dict(source_row_counts)
    if set(row_counts) != set(group.ranks) or any(count < 0 for count in row_counts.values()):
        raise ValueError("source row counts must cover every group rank")

    example = next((item.values for item in routed_outputs if item.values.shape[0]), None)
    if example is None:
        raise ValueError("cannot infer output shape from an entirely empty dispatch")
    outputs = {
        rank: example.new_empty((row_counts[rank], *example.shape[1:])) for rank in group.ranks
    }
    written = {rank: torch.zeros(row_counts[rank], dtype=torch.bool) for rank in group.ranks}
    for destination in group.ranks:
        routed = by_destination[destination]
        if routed.values.shape[0] != routed.source_ranks.shape[0]:
            raise ValueError("expert output row count differs from dispatch provenance")
        for output_row, (source_rank, source_row) in enumerate(
            zip(routed.source_ranks.tolist(), routed.source_rows.tolist(), strict=True)
        ):
            if source_rank not in outputs or not 0 <= source_row < row_counts[source_rank]:
                raise ValueError("dispatch provenance references an unknown source row")
            if bool(written[source_rank][source_row].item()):
                raise ValueError("one source row was combined more than once")
            outputs[source_rank][source_row] = routed.values[output_row]
            written[source_rank][source_row] = True
    if any(not bool(mask.all().item()) for mask in written.values()):
        raise ValueError("all-to-all combine did not restore every source row")
    return tuple(RankTensor(rank, outputs[rank]) for rank in group.ranks)
