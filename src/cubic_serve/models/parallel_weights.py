"""Small graph-family-neutral primitives for rank-local checkpoint slices."""

from __future__ import annotations

from typing_extensions import Generic, TypeVar

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor

from cubic_serve.contracts.execution import ParallelTopology


@dataclass(frozen=True, slots=True)
class IndexerTP:
    """Whole query heads, raw-key coordinates and historical blocks have owners.

    Query heads may be fewer than ranks: empty head ranges do no query work,
    while every rank still projects/compresses key coordinates and scans blocks.
    """

    query_heads: int
    head_dim: int
    compression: int
    size: int
    rank: int

    def __post_init__(self) -> None:
        if (
            any(type(n) is not int or n <= 0 for n in
                (self.query_heads, self.head_dim, self.compression, self.size))
            or type(self.rank) is not int or not 0 <= self.rank < self.size
            or self.head_dim % self.size
        ):
            raise ValueError("indexer TP requires integral key coordinates and a valid owner")

    @property
    def query_head_range(self) -> tuple[int, int]:
        return (self.query_heads * self.rank // self.size,
                self.query_heads * (self.rank + 1) // self.size)

    @property
    def local_query_heads(self) -> int:
        begin, end = self.query_head_range
        return end - begin

    @property
    def key_width(self) -> int:
        return self.head_dim // self.size

    @property
    def exchange_width(self) -> int:
        return ((self.query_heads + self.size - 1) // self.size) * self.head_dim + self.key_width

    @property
    def projection_ranges(self) -> tuple[tuple[int, int], ...]:
        begin, end = self.query_head_range
        base = self.query_heads * self.head_dim + self.rank * self.key_width
        query = ((begin * self.head_dim, end * self.head_dim),) if begin < end else ()
        return (*query, (base, base + self.key_width))

    def cache_page_groups(self, page_tokens: int) -> int:
        if type(page_tokens) is not int or page_tokens <= 0 or page_tokens % (self.compression * self.size):
            raise ValueError("indexer page must contain an integral number of groups per owner")
        return page_tokens // (self.compression * self.size)


@dataclass(frozen=True, slots=True, order=True)
class AxisSpan:
    """One non-empty half-open interval on a tensor axis."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("axis spans require 0 <= start < stop")

    @property
    def size(self) -> int:
        return self.stop - self.start


class AxisSliceBinding(Protocol):
    @property
    def tensor_name(self) -> str: ...

    @property
    def shard_axis(self) -> int: ...

    @property
    def spans(self) -> tuple[AxisSpan, ...]: ...

    @property
    def global_shape(self) -> tuple[int, ...]: ...

    @property
    def local_shape(self) -> tuple[int, ...]: ...


@dataclass(frozen=True, slots=True)
class LogicalWeightSlice:
    """Format-neutral owner coordinates; no serialized dtype or execution seal."""

    tensor_name: str
    shard_axis: int
    spans: tuple[AxisSpan, ...]
    global_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not self.tensor_name
            or not self.global_shape
            or any(n <= 0 for n in self.global_shape)
            or not 0 <= self.shard_axis < len(self.global_shape)
            or not self.spans
            or any(span.stop > self.global_shape[self.shard_axis] for span in self.spans)
        ):
            raise ValueError("logical weight owner coordinates are invalid")

    @property
    def local_shape(self) -> tuple[int, ...]:
        shape = list(self.global_shape)
        shape[self.shard_axis] = sum(span.size for span in self.spans)
        return tuple(shape)


class SafeTensorSlice(Protocol):
    def get_shape(self) -> list[int]: ...
    def __getitem__(self, index: tuple[slice, ...]) -> Tensor: ...


class SafeTensorReader(Protocol):
    def keys(self) -> list[str]: ...
    def get_slice(self, name: str) -> SafeTensorSlice: ...


BindingT = TypeVar("BindingT", bound=AxisSliceBinding)


class RankLocalTensors(Generic[BindingT]):
    """Validated immutable metadata plus resident tensors for one rank."""

    __slots__ = ("_tensors", "bindings", "device", "inventory_digest", "rank")

    def __init__(
        self,
        *,
        rank: int,
        device: torch.device,
        inventory_digest: str,
        bindings: tuple[BindingT, ...],
        tensors: dict[str, Tensor],
        dtypes: dict[str, torch.dtype],
    ) -> None:
        expected = tuple(sorted(item.tensor_name for item in bindings))
        if tuple(sorted(tensors)) != expected or tuple(sorted(dtypes)) != expected:
            raise ValueError("resident tensors differ from the rank-local binding set")
        by_name = {item.tensor_name: item for item in bindings}
        if len(by_name) != len(bindings):
            raise ValueError("rank-local tensor names must be unique")
        for name, tensor in tensors.items():
            binding = by_name[name]
            if (
                tensor.dtype is not dtypes[name]
                or tensor.device != device
                or tuple(tensor.shape) != binding.local_shape
                or tensor.layout is not torch.strided
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"resident tensor {name!r} violates its rank-local binding")
        self.rank = rank
        self.device = device
        self.inventory_digest = inventory_digest
        self.bindings = bindings
        self._tensors = dict(tensors)

    @property
    def tensor_count(self) -> int:
        return len(self._tensors)

    @property
    def resident_bytes(self) -> int:
        return sum(item.numel() * item.element_size() for item in self._tensors.values())

    def __getitem__(self, name: str) -> Tensor:
        return self._tensors[name]

    def named_tensors(self) -> tuple[tuple[str, Tensor], ...]:
        return tuple((name, self._tensors[name]) for name in sorted(self._tensors))


def materialize_axis_spans(
    reader: SafeTensorReader,
    binding: AxisSliceBinding,
    *,
    dtype: torch.dtype,
    device: torch.device,
    secondary_axis: int | None = None,
    secondary_spans: tuple[AxisSpan, ...] = (),
) -> Tensor:
    """Read declared spans on one or two independent source axes."""

    source = reader.get_slice(binding.tensor_name)
    if tuple(source.get_shape()) != binding.global_shape:
        raise ValueError("safetensors payload shape differs from its binding")
    if secondary_axis is None:
        if secondary_spans:
            raise ValueError("secondary spans require their source axis")
        secondary = (AxisSpan(0, binding.global_shape[0]),)
    else:
        if (
            secondary_axis == binding.shard_axis
            or not 0 <= secondary_axis < len(binding.global_shape)
            or not secondary_spans
            or any(span.stop > binding.global_shape[secondary_axis] for span in secondary_spans)
        ):
            raise ValueError("secondary source spans are invalid")
        secondary = secondary_spans
    target = torch.empty(binding.local_shape, dtype=dtype, device=device)
    destination = 0
    for span in binding.spans:
        secondary_destination = 0
        for other in secondary:
            index = [slice(None)] * len(binding.global_shape)
            index[binding.shard_axis] = slice(span.start, span.stop)
            if secondary_axis is not None:
                index[secondary_axis] = slice(other.start, other.stop)
            piece = source[tuple(index)]
            expected = list(binding.global_shape)
            expected[binding.shard_axis] = span.size
            if secondary_axis is not None:
                expected[secondary_axis] = other.size
            if piece.dtype is not dtype or tuple(piece.shape) != tuple(expected):
                raise ValueError("safetensors source span differs from its binding")
            destination_view = target.narrow(binding.shard_axis, destination, span.size)
            if secondary_axis is not None:
                destination_view = destination_view.narrow(
                    secondary_axis, secondary_destination, other.size
                )
            destination_view.copy_(piece)
            secondary_destination += other.size
        destination += span.size
    return target


def confined_checkpoint_path(root: Path, relative: str) -> Path:
    """Resolve an attested shard while forbidding path escape."""

    candidate = (root / relative).resolve(strict=True)
    if not candidate.is_relative_to(root):
        raise ValueError("checkpoint path escapes its attested root")
    return candidate


def equal_axis_span(dimension: int, rank: int, size: int) -> AxisSpan:
    width, remainder = divmod(dimension, size)
    if remainder or not 0 <= rank < size:
        raise ValueError("tensor dimension cannot be evenly sharded")
    return AxisSpan(rank * width, (rank + 1) * width)


def checkpoint_expert_slice(
    name: str,
    shape: tuple[int, ...],
    *,
    expert_id: int,
    expert_count: int,
    matrix_axis: int,
    topology: ParallelTopology,
    rank: int,
) -> LogicalWeightSlice | None:
    """Place a checkpoint expert on its EP owner and shard its matrix over ETP."""
    if type(expert_id) is not int or not 0 <= expert_id < expert_count:
        raise ValueError("expert ID must be in checkpoint order")
    owner, lane = topology.expert_coordinates(rank)
    owned = equal_axis_span(expert_count, owner, topology.ep_size)
    if not owned.start <= expert_id < owned.stop:
        return None
    return LogicalWeightSlice(
        name,
        matrix_axis,
        (equal_axis_span(shape[matrix_axis], lane, topology.expert_tensor_size),),
        shape,
    )


def reconstruct_axis_spans(
    gathered: Tensor,
    *,
    axis: int,
    local_size: int,
    global_size: int,
    spans_by_rank: tuple[tuple[AxisSpan, ...], ...],
) -> Tensor:
    """Restore unique semantic span order from equal-size rank-order chunks."""

    if not spans_by_rank or local_size <= 0 or global_size != local_size * len(spans_by_rank):
        raise ValueError("equal-shard reconstruction dimensions are open")
    normalized_axis = axis % gathered.ndim
    if gathered.shape[normalized_axis] != global_size:
        raise ValueError("gathered tensor has the wrong global width")
    if all(len(spans) == 1 for spans in spans_by_rank):
        cursor = 0
        for spans in spans_by_rank:
            if spans[0] != AxisSpan(cursor, cursor + local_size):
                break
            cursor += local_size
        else:
            return gathered

    pieces: list[tuple[int, int, Tensor]] = []
    for rank_index, spans in enumerate(spans_by_rank):
        chunk = gathered.narrow(normalized_axis, rank_index * local_size, local_size)
        offset = 0
        for span in spans:
            pieces.append(
                (
                    span.start,
                    span.stop,
                    chunk.narrow(normalized_axis, offset, span.size),
                )
            )
            offset += span.size
        if offset != local_size:
            raise ValueError("rank spans differ from its gathered shard")
    pieces.sort(key=lambda item: item[0])
    cursor = 0
    for start, stop, _ in pieces:
        if start != cursor:
            raise ValueError("semantic spans overlap or leave a gap")
        cursor = stop
    if cursor != global_size:
        raise ValueError("semantic spans do not close the global tensor")
    return torch.cat(tuple(item for _, _, item in pieces), dim=normalized_axis).contiguous()


__all__ = (
    "AxisSliceBinding",
    "AxisSpan",
    "IndexerTP",
    "RankLocalTensors",
    "SafeTensorReader",
    "checkpoint_expert_slice",
    "confined_checkpoint_path",
    "equal_axis_span",
    "materialize_axis_spans",
    "reconstruct_axis_spans",
)
