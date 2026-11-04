"""Decode-stage surface consumed by shared serving batch fill and sampling.

This describes static operands, not readiness: graph qualification and the
complete topology-wide replay lease remain controller responsibilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import Tensor

    from cubic_serve.contracts.execution import RankReplayResources
    from cubic_serve.models.parallel_weights import AxisSpan
    from cubic_serve.runtime.cache import CacheArena


class ServingInputBuffers(Protocol):
    """Model-independent fixed operands consumed by batch fill and sampling.

    Every state map is INT32 [source/destination, packed layers, physical rows].
    A packed slab uses owner_slot * packed_layers + layer; independently stored
    state fields use packed_layers=1 and may share one map. Zero continuation
    lengths do not identify padding; row_activity owns that distinction.
    """

    token_ids: Tensor
    position_ids: Tensor
    page_source: Tensor
    page_destination: Tensor
    token_offsets: Tensor
    page_table: Tensor
    sequence_lengths: Tensor
    segment_lengths: Tensor
    row_activity: Tensor
    norm: Tensor

    @property
    def state_slot_maps(self) -> tuple[Tensor, ...]: ...


class ServingDecodeBody(Protocol):
    rank: int
    stage: int
    tensor_rank: int
    captured_m: int
    arena: CacheArena
    embedding_mask: Tensor | None

    @property
    def embedding_span(self) -> AxisSpan | None: ...

    @property
    def buffers(self) -> ServingInputBuffers: ...

    @property
    def hidden(self) -> Tensor: ...

    def set_active_rows(self, logical_m: int) -> None: ...

    def prepare_scratch(self, resource: RankReplayResources) -> None: ...
