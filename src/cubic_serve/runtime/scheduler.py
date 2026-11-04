"""Small, plan-only continuous batching scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from cubic_serve.contracts.execution import (
    BatchPhase,
    BatchPlan,
    BatchShape,
    CapacityItems,
    DecodeExtensionMeta,
    DecodeProvider,
    PlanSegment,
    PrefixPublishPlan,
    StateAccess,
    StateTxn,
)
from cubic_serve.contracts.request import InputPlan
from cubic_serve.runtime.request import InflightSegment, RequestState

_NO_DECODE_EXTENSION = DecodeExtensionMeta()


@dataclass(frozen=True, slots=True)
class PageAllocation:
    """Exact physical token mapping returned by the cache owner."""

    cache_slots: tuple[int, ...]
    context_pages: tuple[int, ...]
    write_pages: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.cache_slots:
            raise ValueError("a page allocation must contain at least one cache slot")
        if len(set(self.cache_slots)) != len(self.cache_slots):
            raise ValueError("cache slots must be unique within one segment")
        if any(value < 0 for value in (*self.cache_slots, *self.context_pages, *self.write_pages)):
            raise ValueError("cache slots and pages must be non-negative")
        if not self.context_pages or not self.write_pages:
            raise ValueError("a page allocation must identify context and write pages")
        for name, pages in (("context", self.context_pages), ("write", self.write_pages)):
            if len(set(pages)) != len(pages):
                raise ValueError(f"{name} pages must be unique")
        if not set(self.write_pages) <= set(self.context_pages):
            raise ValueError("write pages must be part of the segment context")


@dataclass(frozen=True, slots=True)
class SegmentCompletion:
    """One scheduler projection of an already verified execution result."""

    request_id: str
    generated_token_id: int | None = None
    finished: bool = False

    def __post_init__(self) -> None:
        if type(self.request_id) is not str or not self.request_id:
            raise ValueError("completion request_id must be a non-empty string")
        if self.generated_token_id is not None:
            if type(self.generated_token_id) is not int:
                raise TypeError("completion token id must be an integer")
            if self.generated_token_id < 0:
                raise ValueError("completion token id must be non-negative")
        if type(self.finished) is not bool:
            raise TypeError("completion finished must be a bool")


class PageAllocator(Protocol):
    """Narrow scheduler-facing cache geometry and mapping interface."""

    @property
    def block_size_tokens(self) -> int: ...

    def allocate(
        self,
        *,
        request_id: str,
        token_start: int,
        token_count: int,
    ) -> PageAllocation: ...


class ContinuousScheduler:
    """Reserve decode-first rows and describe them as immutable ``BatchPlan``.

    Emitting a plan reserves each segment exactly once.  ``complete`` advances
    its state revision; ``fail`` releases the reservation for a retry.  Neither
    path executes model code or mutates cache storage.  The token budget limits
    one execution plan, never the number of requests admitted by the API.
    """

    def __init__(
        self,
        *,
        page_allocator: PageAllocator,
        token_budget: int,
        prefill_chunk_tokens: int,
        prefill_token_budget: int | None = None,
        topology_id: str,
        partition_id: str,
        owner_ranks: tuple[int, ...],
        state_ids: tuple[str, ...],
        state_commit_alignment_tokens: int,
        capacity_items: CapacityItems = (),
        decode_extension: DecodeExtensionMeta = _NO_DECODE_EXTENSION,
    ) -> None:
        if token_budget <= 0 or prefill_chunk_tokens <= 0:
            raise ValueError("token budget and prefill chunk must be positive")
        if prefill_token_budget is None:
            prefill_token_budget = token_budget
        if not 0 < prefill_token_budget <= token_budget:
            raise ValueError("prefill token budget must be positive and fit the token budget")
        if not topology_id or not partition_id:
            raise ValueError("topology and partition identity are required")
        if not owner_ranks or tuple(sorted(set(owner_ranks))) != owner_ranks:
            raise ValueError("owner ranks must be non-empty, unique, and sorted")
        if owner_ranks[0] < 0:
            raise ValueError("owner ranks must be non-negative")
        if tuple(sorted(set(state_ids))) != state_ids or any(not item for item in state_ids):
            raise ValueError("state ids must be unique, sorted, and non-empty strings")
        if state_commit_alignment_tokens <= 0:
            raise ValueError("state commit alignment must be positive")
        block_size_tokens = page_allocator.block_size_tokens
        if block_size_tokens <= 0:
            raise ValueError("allocator block size must be positive")
        if decode_extension.provider is not DecodeProvider.NONE:
            raise ValueError("speculative decode providers are unavailable in v0.1")
        capacity_items = BatchShape(1, capacity_items).capacity_items

        self._allocator = page_allocator
        self._block_size = block_size_tokens
        self._token_budget = token_budget
        self._prefill_chunk = prefill_chunk_tokens
        self._prefill_token_budget = prefill_token_budget
        self._topology_id = topology_id
        self._partition_id = partition_id
        self._owner_ranks = owner_ranks
        self._state_ids = state_ids
        self._state_alignment = state_commit_alignment_tokens
        self._capacity_items = capacity_items
        self._decode_extension = decode_extension
        self._requests: dict[str, RequestState] = {}
        self._order: list[str] = []
        self._next_index = 0
        self._plan_sequence = 0

    def admit(
        self,
        input_plan: InputPlan,
        *,
        max_new_tokens: int,
        execution_input_digest: str | None = None,
        base_revision: int = 0,
        prefill_cursor: int = 0,
    ) -> RequestState:
        if input_plan.request_id in self._requests:
            raise ValueError(f"request {input_plan.request_id!r} is already admitted")
        state = RequestState(
            input_plan=input_plan,
            execution_input_digest=(execution_input_digest or input_plan.semantic_digest),
            max_new_tokens=max_new_tokens,
            prefill_cursor=prefill_cursor,
            state_revision=base_revision,
        )
        self._requests[state.request_id] = state
        self._order.append(state.request_id)
        return state

    def request_state(self, request_id: str) -> RequestState:
        try:
            return self._requests[request_id]
        except KeyError as error:
            raise KeyError(f"unknown request {request_id!r}") from error

    def rollback_admission(self, request_id: str) -> None:
        """Remove an admitted request only before its first plan is emitted."""

        state = self.request_state(request_id)
        if (
            state.inflight is not None
            or state.generated_token_ids
            or state.decode_pending
            or state.cancelled
            or state.finished
        ):
            raise RuntimeError("scheduler admission changed and cannot be rolled back")
        index = self._order.index(request_id)
        del self._requests[request_id]
        del self._order[index]
        if index < self._next_index:
            self._next_index -= 1
        self._next_index = self._next_index % len(self._order) if self._order else 0

    @property
    def drained(self) -> bool:
        return all(
            (state.cancelled or state.finished) and state.inflight is None
            for state in self._requests.values()
        )

    def cancel(self, request_id: str) -> None:
        state = self.request_state(request_id)
        state.cancelled = True
        state.finished = False
        state.decode_pending = False

    def finish(self, request_id: str) -> None:
        state = self.request_state(request_id)
        if state.cancelled:
            return
        state.finished = True
        state.decode_pending = False

    def next_plan(self) -> BatchPlan | None:
        selected: list[tuple[RequestState, BatchPhase, int, int, PageAllocation]] = []
        remaining_budget = self._token_budget
        request_count = len(self._order)
        if request_count == 0:
            return None
        last_selected_index: int | None = None
        for phase in (BatchPhase.DECODE, BatchPhase.PREFILL):
            phase_budget = (
                remaining_budget
                if phase is BatchPhase.DECODE
                else min(remaining_budget, self._prefill_token_budget)
            )
            if phase_budget == 0:
                continue
            for offset in range(request_count):
                index = (self._next_index + offset) % request_count
                state = self._requests[self._order[index]]
                if not state.schedulable:
                    continue
                state_phase = (
                    BatchPhase.PREFILL
                    if state.prefill_cursor < state.prompt_token_count
                    else BatchPhase.DECODE
                )
                if state_phase is not phase:
                    continue

                if phase is BatchPhase.PREFILL:
                    token_start = state.prefill_cursor
                    to_block_boundary = self._block_size - token_start % self._block_size
                    # Alignment one means every token boundary is a valid state
                    # checkpoint; it must not collapse chunked prefill to M=1.
                    to_state_boundary = (
                        self._state_alignment - token_start % self._state_alignment
                        if self._state_ids and self._state_alignment > 1
                        else self._prefill_chunk
                    )
                    token_count = min(
                        self._prefill_chunk,
                        state.prompt_token_count - state.prefill_cursor,
                        phase_budget,
                        to_block_boundary,
                        to_state_boundary,
                    )
                else:
                    token_start = (
                        state.prompt_token_count + len(state.generated_token_ids) - 1
                    )
                    token_count = 1

                allocation = self._allocator.allocate(
                    request_id=state.request_id,
                    token_start=token_start,
                    token_count=token_count,
                )
                if len(allocation.cache_slots) != token_count:
                    raise ValueError("allocator must return exactly one cache slot per token")
                selected.append((state, phase, token_start, token_count, allocation))
                remaining_budget -= token_count
                phase_budget -= token_count
                last_selected_index = index
                if remaining_budget == 0 or phase_budget == 0:
                    break

        if not selected:
            return None
        assert last_selected_index is not None

        plan_id = f"batch:{self._plan_sequence}"
        phases = {phase for _, phase, _, _, _ in selected}
        batch_phase = phases.pop() if len(phases) == 1 else BatchPhase.MIXED

        segments: list[PlanSegment] = []
        transactions: list[StateTxn] = []
        for state, phase, token_start, token_count, allocation in selected:
            if phase is BatchPhase.PREFILL:
                token_ids = state.input_plan.token_ids[token_start : token_start + token_count]
                positions = state.input_plan.positions[token_start : token_start + token_count]
            else:
                if token_count != 1 or not state.generated_token_ids:
                    raise RuntimeError("decode planning requires exactly one generated input token")
                decode_index = len(state.generated_token_ids) - 1
                token_ids = (state.generated_token_ids[-1],)
                positions = (state.input_plan.decode_positions.at(decode_index),)
            segment = PlanSegment(
                request_id=state.request_id,
                input_plan_digest=state.input_plan.semantic_digest,
                execution_input_digest=state.execution_input_digest,
                phase=phase,
                produces_output=(
                    phase is BatchPhase.DECODE
                    or token_start + token_count == state.prompt_token_count
                ),
                token_start=token_start,
                token_count=token_count,
                token_ids=token_ids,
                positions=positions,
                cache_slots=allocation.cache_slots,
                cache_pages=allocation.context_pages,
            )
            target_revision = state.state_revision + 1
            prefix_publish = self._prefix_publication(state, phase, token_start, token_count)
            reads = tuple(
                access
                for owner_rank in self._owner_ranks
                for access in (
                    StateAccess(
                        resource_id=f"kv:{state.request_id}",
                        owner_rank=owner_rank,
                        pages=tuple(sorted(allocation.context_pages)),
                    ),
                    *(
                        StateAccess(
                            resource_id=f"state:{state_id}:{state.request_id}",
                            owner_rank=owner_rank,
                        )
                        for state_id in self._state_ids
                    ),
                )
            )
            writes = tuple(
                access
                for owner_rank in self._owner_ranks
                for access in (
                    StateAccess(
                        resource_id=f"kv:{state.request_id}",
                        owner_rank=owner_rank,
                        slots=tuple(sorted(set(allocation.cache_slots))),
                        pages=tuple(sorted(allocation.write_pages)),
                    ),
                    *(
                        StateAccess(
                            resource_id=f"state:{state_id}:{state.request_id}",
                            owner_rank=owner_rank,
                        )
                        for state_id in self._state_ids
                    ),
                )
            )
            txn = StateTxn(
                transaction_id=f"txn:{plan_id}:{state.request_id}",
                request_id=state.request_id,
                base_revision=state.state_revision,
                target_revision=target_revision,
                reads=reads,
                writes=writes,
                expected_owner_ranks=self._owner_ranks,
                prefix_publish=prefix_publish,
            )
            segments.append(segment)
            transactions.append(txn)

        logical_m = sum(segment.token_count for segment in segments)
        batch_plan = BatchPlan(
            plan_id=plan_id,
            phase=batch_phase,
            input_plan_ids=tuple(segment.request_id for segment in segments),
            segments=tuple(segments),
            shape=BatchShape(logical_m=logical_m, capacity_items=self._capacity_items),
            topology_id=self._topology_id,
            partition_id=self._partition_id,
            state_transactions=tuple(transactions),
            decode_extension=self._decode_extension,
        )
        for state, phase, token_start, token_count, _ in selected:
            if phase is BatchPhase.PREFILL:
                state.prefill_cursor += token_count
            else:
                state.decode_pending = False
            state.inflight = InflightSegment(
                plan_id=plan_id,
                phase=phase,
                token_start=token_start,
                token_count=token_count,
                target_revision=state.state_revision + 1,
            )
        self._next_index = (last_selected_index + 1) % request_count
        self._plan_sequence += 1
        return batch_plan

    def complete(
        self,
        request_id: str,
        *,
        generated_token_id: int | None = None,
        finished: bool = False,
    ) -> None:
        self.complete_bundle(
            (
                SegmentCompletion(
                    request_id=request_id,
                    generated_token_id=generated_token_id,
                    finished=finished,
                ),
            )
        )

    def complete_bundle(self, completions: tuple[SegmentCompletion, ...]) -> None:
        """Validate every completion before advancing any request state."""

        if not completions:
            raise ValueError("completion bundle cannot be empty")
        request_ids = tuple(completion.request_id for completion in completions)
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("completion bundle request ids must be unique")
        for completion in completions:
            self.validate_completion(
                completion.request_id,
                generated_token_id=completion.generated_token_id,
                finished=completion.finished,
            )
        for completion in completions:
            self._apply_completion(completion)

    def _apply_completion(self, completion: SegmentCompletion) -> None:
        state = self.request_state(completion.request_id)
        inflight = state.inflight
        assert inflight is not None

        state.state_revision = inflight.target_revision
        state.inflight = None
        if state.cancelled or state.finished:
            state.decode_pending = False
            return
        if completion.generated_token_id is not None:
            state.generated_token_ids += (completion.generated_token_id,)
        if completion.finished or len(state.generated_token_ids) >= state.max_new_tokens:
            state.finished = True
            state.decode_pending = False
        elif completion.generated_token_id is not None:
            state.decode_pending = True

    def validate_completion(
        self,
        request_id: str,
        *,
        generated_token_id: int | None = None,
        finished: bool = False,
    ) -> None:
        """Purely validate a completion before its cache transaction commits."""

        state = self.request_state(request_id)
        inflight = state.inflight
        if inflight is None:
            raise ValueError("request has no outstanding plan segment")
        if generated_token_id is not None and generated_token_id < 0:
            raise ValueError("generated token id must be non-negative")
        produces_output = (
            inflight.phase is BatchPhase.DECODE or state.prefill_cursor == state.prompt_token_count
        )
        if generated_token_id is not None and not produces_output:
            raise ValueError("an intermediate prefill segment cannot produce an output token")
        if (
            generated_token_id is None
            and produces_output
            and not finished
            and state.active
        ):
            raise ValueError("a completed terminal prefill/decode row requires an output token")

    def fail(self, request_id: str) -> None:
        """Release one reservation without advancing its state revision."""

        state = self.request_state(request_id)
        inflight = state.inflight
        if inflight is None:
            raise ValueError("request has no outstanding plan segment")
        if inflight.phase is BatchPhase.PREFILL:
            state.prefill_cursor = inflight.token_start
        elif state.active:
            state.decode_pending = True
        state.inflight = None

    def _prefix_publication(
        self,
        state: RequestState,
        phase: BatchPhase,
        token_start: int,
        token_count: int,
    ) -> PrefixPublishPlan | None:
        prefix_token_count = token_start + token_count
        if phase is not BatchPhase.PREFILL:
            return None
        if prefix_token_count % self._block_size:
            return None
        if not self._state_ids or prefix_token_count % self._state_alignment == 0:
            state_commit_position = prefix_token_count
        elif token_start % self._state_alignment == 0:
            state_commit_position = token_start
        else:
            raise RuntimeError("scheduler crossed a prefix without an aligned state checkpoint")
        return PrefixPublishPlan(
            input_plan_id=state.request_id,
            prefix_token_count=prefix_token_count,
            state_commit_position=state_commit_position,
            commit_alignment_tokens=self._state_alignment,
        )
