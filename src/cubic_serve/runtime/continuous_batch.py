"""Model-independent continuous-batch commands and row transactions."""

from __future__ import annotations

from cubic_serve.executor.readiness import execution_ready

import ctypes
import math
import pickle
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from multiprocessing import shared_memory

from cubic_serve.contracts.request import ReasoningControl, StructuredOutputMeta
from cubic_serve.generation.sampler import GREEDY_SAMPLING, SamplingParameters
from cubic_serve.generation.structured_sampling import ConstraintBinding, GrammarRoundCost

# Bounded transaction storage, independent of model, device and KV page size.
# Runtime also caps work by its duration, remaining output and granted pages.
RESIDENT_STEP_CAPACITY = 1024


class BatchProtocolError(ValueError):
    pass


class InputEncoderExecutionLease:
    """FIFO ownership shared by language dispatch and a colocated encoder.

    An encoder's context-manager acquisition first revokes the current decode
    window, then waits for its durable completion and lease release. Language
    dispatch uses acquire/release and cannot overtake an already waiting input.
    """

    def __init__(self, interrupt):
        self._interrupt = interrupt
        self._condition = threading.Condition()
        self._waiting = deque()
        self._owner = None

    def _acquire(self, *, preempt: bool) -> None:
        ticket = object()
        with self._condition:
            self._waiting.append(ticket)
            try:
                if preempt:
                    self._interrupt.set()
                while self._owner is not None or self._waiting[0] is not ticket:
                    self._condition.wait()
                self._waiting.popleft()
                self._owner = threading.get_ident()
            except BaseException:
                self._waiting.remove(ticket)
                self._condition.notify_all()
                raise

    def acquire(self) -> None:
        self._acquire(preempt=False)

    def release(self) -> None:
        with self._condition:
            if self._owner != threading.get_ident():
                raise RuntimeError("input encoder execution lease is not owned by this thread")
            self._owner = None
            self._condition.notify_all()

    def __enter__(self):
        self._acquire(preempt=True)
        return self

    def __exit__(self, *_):
        self.release()


class SharedPublicationInterrupt:
    """The existing process event plus revocation of mapped publication headers.

    Only the controller attaches journals, after all ranks report readiness.
    Spawned workers receive the event, never another process's mapped views.
    The stop word has no payload dependency and is naturally aligned int32.
    """

    def __init__(self, event):
        self._event = event
        self._lock = threading.Lock()
        self._journals: dict[str, shared_memory.SharedMemory] = {}

    def __getstate__(self):
        return {"event": self._event}

    def __setstate__(self, state):
        self.__init__(state["event"])

    def attach(self, names: tuple[str, ...]) -> None:
        if (type(names) is not tuple or any(type(name) is not str for name in names)
                or len(set(names)) != len(names)):
            raise BatchProtocolError("resident journal descriptors are invalid")
        with self._lock:
            try:
                for name in names:
                    if name in self._journals:
                        raise BatchProtocolError("resident journal was attached twice")
                    journal = shared_memory.SharedMemory(name=name)
                    if journal.size < 64 * 8:
                        journal.close()
                        raise BatchProtocolError("resident journal header is truncated")
                    self._journals[name] = journal
                self._write_stop(int(self._event.is_set()))
            except BaseException:
                self._close_views()
                raise

    def _write_stop(self, value: int) -> None:
        for journal in self._journals.values():
            ctypes.c_int32.from_buffer(journal.buf, 8).value = value

    def set(self) -> None:
        with self._lock:
            self._write_stop(1)
            self._event.set()

    def clear(self) -> None:
        with self._lock:
            self._event.clear()
            self._write_stop(0)

    def is_set(self) -> bool:
        return self._event.is_set()

    def _close_views(self) -> None:
        for journal in self._journals.values():
            journal.close()
        self._journals.clear()

    def close(self) -> None:
        with self._lock:
            self._close_views()


@dataclass(frozen=True, slots=True)
class GenerateCommand:
    request_id: str
    input_ids: tuple[int, ...]
    max_tokens: int
    eos_token_ids: tuple[int, ...]
    sampling: SamplingParameters = GREEDY_SAMPLING
    positions: tuple[tuple[int, int, int], ...] = ()
    decode_position_first: tuple[int, int, int] | None = None
    decode_position_step: tuple[int, int, int] | None = None
    feature_rows: tuple[bytes, ...] = ()
    feature_digest: str = ""
    structured_output: StructuredOutputMeta | None = None
    constraint_reasoning_end: tuple[int, ...] = ()
    reasoning_control: ReasoningControl | None = None


def prepared_input_fields(input_ids, prepared, *, hidden_size: int | None = None,
                          max_position: int | None = None, max_tokens: int = 1):
    """Serialize the shared immutable visual/position input at request admission."""
    if prepared is None:
        return (), None, None, (), ""
    import torch

    from cubic_serve.models.protocol import PreparedInput

    if not isinstance(prepared, PreparedInput) or prepared.input_plan.token_ids != input_ids:
        raise ValueError("prepared input differs from generation token ids")
    plan = prepared.input_plan
    positions = tuple(item.axes for item in plan.positions)
    first, step = plan.decode_positions.first.axes, plan.decode_positions.step_axes
    if any(len(axes) != 3 for axes in (*positions, first, step)):
        raise ValueError("prepared input requires three-axis positions")
    if max_position is not None and (
        any(axis >= max_position for axes in positions for axis in axes)
        or any(first[axis] + (max_tokens - 1) * step[axis] >= max_position
               for axis in range(3))
    ):
        raise ValueError("prepared input exceeds the model position axes limit")
    rows, digest = (), ""
    if prepared.feature_bundle is not None:
        prepared.feature_bundle.verify_for_input_plan(plan)
        rows = [b""] * len(input_ids)
        for binding in prepared.feature_bundle.bindings:
            placement = binding.placement
            if (placement.consumer_layers != (0,)
                    or (hidden_size is not None and binding.feature.hidden_size != hidden_size)):
                raise ValueError("visual feature must target decoder input with the declared width")
            feature = binding.feature.tensor[
                placement.feature_row_start:
                placement.feature_row_start + placement.feature_row_count
            ].detach().contiguous().cpu()
            raw = feature.view(torch.uint8).numpy()
            for offset in range(placement.token_count):
                rows[placement.token_start + offset] = raw[offset].tobytes()
        rows = tuple(rows)
        digest = prepared.feature_bundle.semantic_digest
    return positions, first, step, rows, digest


@dataclass(frozen=True, slots=True)
class BatchRow:
    """One request-owned row projected into a physical execution bucket."""

    request_id: str
    token_id: int
    position: int
    state_source_slot: int
    state_destination_slot: int
    page_source: int
    page_destination: int
    page_table: tuple[int, ...]
    reset_state: bool = False
    new_pages: tuple[int, ...] = ()
    segment_length: int = 1
    sampling: SamplingParameters = GREEDY_SAMPLING
    sample_ordinal: int = 0
    mtp_position: int = -1
    mtp_page_destination: int = -1
    mtp_page_table: tuple[int, ...] = ()
    is_decode: bool = False
    position_axes: tuple[int, int, int] | None = None
    input_feature: bytes = b""


def owned_single_token_rows(
    rows: tuple[BatchRow, ...], *, allow_prompt: bool = False,
) -> bool:
    """An unchanged one-token batch can retain exclusive owner metadata."""
    return bool(rows) and all(
        (row.is_decode or allow_prompt)
        and row.segment_length == 1
        and not row.reset_state
        and row.state_source_slot == row.state_destination_slot
        and row.page_source == row.page_destination
        and not row.input_feature
        for row in rows
    )


def owned_segment_rows(rows: tuple[BatchRow, ...], *, allow_draft_prompt: bool = False) -> tuple[int, ...]:
    """Retain complete private owner segments, without inventing padding IDs."""
    if not rows or (any(row.mtp_position >= 0 for row in rows) and not allow_draft_prompt):
        return ()
    if allow_draft_prompt and any(row.is_decode or row.mtp_position != row.position
                                 or not row.mtp_page_table for row in rows):
        return ()
    segments = batch_segments(rows)
    lengths = tuple(stop - start for start, stop in segments)
    if (not any(width > 1 for width in lengths)
            or len({rows[start].request_id for start, _ in segments}) != len(segments)):
        return ()
    if len({rows[start].state_source_slot for start, _ in segments}) != len(segments):
        return ()
    for start, stop in segments:
        segment = rows[start:stop]
        first = segment[0]
        if stop - start == 1:
            if not owned_single_token_rows(segment, allow_prompt=True):
                return ()
        elif any(
            row.is_decode or (row.mtp_position >= 0 and not allow_draft_prompt) or row.input_feature
            or row.state_source_slot != row.state_destination_slot
            or row.state_source_slot != first.state_source_slot
            or row.page_table != first.page_table
            for row in segment
        ):
            return ()
    return lengths


def resident_batch_ownership(executor, rows: tuple[BatchRow, ...], *, block_size: int):
    """Consume a completed bind's immutable CPU owner metadata before receipt."""
    window = getattr(executor, "_resident_window", None)
    if owned_single_token_rows(rows, allow_prompt=execution_ready(window, "program_qualified")):
        if hasattr(executor, "_resident_owner_lengths"):
            executor._resident_owner_lengths = executor._resident_owner_pages = ()
        return (
            tuple(row.request_id for row in rows),
            tuple(row.position % block_size for row in rows),
            tuple(len(row.page_table) for row in rows),
        )
    if not execution_ready(window, "segmented_program_qualified"):
        return None
    lengths = owned_segment_rows(rows, allow_draft_prompt=getattr(window, 'draft_maintenance', None) is not None)
    if not lengths:
        return None
    segments = batch_segments(rows)
    executor._resident_owner_lengths = lengths
    executor._resident_owner_pages = tuple(rows[stop - 1].page_destination
                                           for _, stop in segments)
    return (
        tuple(rows[start].request_id for start, _ in segments),
        tuple(rows[stop - 1].position % block_size for _, stop in segments),
        tuple(len(rows[stop - 1].page_table) for _, stop in segments),
    )


def independent_decode_partitions(
    rows: tuple[BatchRow, ...], capacities: tuple[int, ...],
) -> bool:
    """Prove that an admitted decode transaction fits existing graph lanes.

    Each recurrence segment stays whole. Cross-lane state and active-page writes
    must be disjoint from reads and other writes. Earlier prefix pages are
    immutable under the controller's append/COW contract; scanning those pages
    here would add context-length-dependent work to every decode dispatch.
    This checks partition eligibility, not request admission or ownership.
    """
    if (
        not rows or not capacities or any(size <= 0 for size in capacities)
        or len(rows) > sum(capacities)
        or any(not row.is_decode or row.input_feature for row in rows)
    ):
        return False
    segments = batch_segments(rows)
    if len({rows[start].request_id for start, _ in segments}) != len(segments):
        return False
    # State slots and KV pages have separate address spaces.
    reads: list[set[tuple[str, int]]] = [set() for _ in capacities]
    writes: list[set[tuple[str, int]]] = [set() for _ in capacities]
    lane, boundary = 0, capacities[0]
    for start, stop in segments:
        while start >= boundary:
            lane += 1
            boundary += capacities[lane]
        if stop > boundary:
            return False
        for row in rows[start:stop]:
            reads[lane].update((("state", row.state_source_slot), ("page", row.page_source)))
            writes[lane].update((
                ("state", row.state_destination_slot), ("page", row.page_destination),
            ))
    for lane in range(len(capacities)):
        for peer in range(lane):
            if (
                writes[lane] & (reads[peer] | writes[peer])
                or writes[peer] & reads[lane]
            ):
                return False
    return True


@dataclass(frozen=True, slots=True)
class BatchCommand:
    batch_id: int
    captured_m: int
    rows: tuple[BatchRow, ...]
    constraints: tuple[ConstraintBinding, ...] = ()
    released_constraints: tuple[str, ...] = ()

    @property
    def logical_m(self) -> int:
        return len(self.rows)


def valid_owner_page_program(widths, sources, program, offsets, counts, *,
                             steps, capacity, block_size, page_limit=None, table_limit=None):
    """Check ordinary COW ownership and logical coordinates before replay."""
    rows = len(widths)
    if (not rows or any(type(v) is not tuple for v in (widths, sources, program, offsets, counts))
            or len(sources) != rows or len(offsets) != rows or len(counts) != rows
            or len(program) != steps
            or any(type(k) is not int or k <= 0 for k in widths)
            or not 0 < sum(widths) <= capacity
            or any(type(p) is not int or p < 0 for p in sources)
            or len(set(sources)) != rows
            or any(type(v) is not int or not 0 <= v < block_size for v in offsets)
            or any(type(v) is not int or v < 1 for v in counts)):
        return False
    if page_limit is not None and any(page >= page_limit for page in sources):
        return False
    latest = [{count - 1: source} for count, source in zip(counts, sources, strict=True)]
    used = set(sources)
    for step, coordinates in enumerate(program):
        if type(coordinates) is not tuple or len(coordinates) != sum(widths):
            return False
        first = 0
        for owner, width in enumerate(widths):
            chosen = {}
            position = (counts[owner] - 1) * block_size + offsets[owner] + 1 + step * width
            for local in range(width):
                entry = coordinates[first + local]
                if (type(entry) is not tuple or len(entry) != 4
                        or any(type(value) is not int for value in entry)):
                    return False
                source, destination, index, reset = entry
                if (min(source, destination) < 0 or reset not in (0, 1)
                        or index != (position + local) // block_size
                        or (page_limit is not None and max(source, destination) >= page_limit)
                        or (table_limit is not None and index >= table_limit)):
                    return False
                if index in chosen:
                    if source != destination or destination != chosen[index] or reset:
                        return False
                    continue
                previous = latest[owner].get(index)
                if previous is None:
                    if (not reset or source in used
                            or index != max((*latest[owner], *chosen), default=-1) + 1):
                        return False
                    used.add(source)
                elif reset or source != previous:
                    return False
                if destination != source:
                    if destination in used:
                        return False
                    used.add(destination)
                elif width > 1:
                    return False
                chosen[index] = destination
            latest[owner].update(chosen)
            first += width
    return True


@dataclass(frozen=True, slots=True)
class DraftWindowBinding:
    """Committed owners and real cache grants for a fixed-depth transaction."""

    depth: int
    owners: tuple[int, ...]
    target_positions: tuple[int, ...]
    target_limits: tuple[int, ...]
    provider_limits: tuple[int, ...]
    remaining: tuple[int, ...]
    target_tables: tuple[tuple[int, ...], ...] = ()
    provider_tables: tuple[tuple[int, ...], ...] = ()
    decision_tokens: tuple[int, ...] = ()
    decision_rounds: tuple[int, ...] = ()

    def __post_init__(self):
        n = len(self.owners)
        if (any(type(values) is not tuple for values in (self.decision_tokens, self.decision_rounds))
                or bool(self.decision_tokens) != bool(self.decision_rounds)
                or any(type(values) is not tuple or len(values) != n
                       or any(type(v) is not int or not 0 <= v <= 0x7fffffffffffffff
                              for v in values)
                       for values in (self.decision_tokens, self.decision_rounds) if values)
                or (self.decision_tokens and not any((*self.decision_tokens,
                                                      *self.decision_rounds)))):
            raise BatchProtocolError("draft continuation has invalid decision budgets")
        if (type(self.depth) is not int or self.depth < 1 or not n
                or len(set(self.owners)) != n
                or any(type(owner) is not int or owner < 0 for owner in self.owners)
                or any(type(values) is not tuple or len(values) != n
                       or any(type(value) is not int or value < 0 for value in values)
                       for values in (self.owners, self.target_positions, self.target_limits,
                                      self.provider_limits, self.remaining))
                or any(position + self.depth >= limit
                       for position, limit in zip(self.target_positions, self.target_limits))
                or any(limit <= 0 for limit in (*self.provider_limits, *self.remaining))):
            raise BatchProtocolError("draft continuation has invalid owned limits")
        if bool(self.target_tables) != bool(self.provider_tables) or any(
            type(tables) is not tuple or len(tables) != n
            or any(type(table) is not tuple or not table or len(set(table)) != len(table)
                   or any(type(page) is not int or page < 0 for page in table)
                   for table in tables)
            for tables in (self.target_tables, self.provider_tables) if tables
        ):
            raise BatchProtocolError("draft continuation has invalid page grants")


@dataclass(frozen=True, slots=True)
class ContinueBatchCommand:
    """Advance one unchanged owned decode batch from worker-resident metadata."""

    batch_id: int
    previous_batch_id: int
    captured_m: int
    request_ids: tuple[str, ...]
    page_updates: tuple[tuple[int, int, int], ...] = ()
    max_steps: int = 1
    eos_token_ids: tuple[tuple[int, ...], ...] = ()
    max_duration_seconds: float = 0.1
    future_page_updates: tuple[tuple[int, int, int, int], ...] = ()
    prompt_remaining: tuple[int, ...] = ()
    prompt_tokens: tuple[tuple[int, ...], ...] = ()
    prompt_positions: tuple[tuple[tuple[int, int, int], ...], ...] = ()
    source_rows: tuple[int, ...] = ()
    owner_lengths: tuple[int, ...] = ()
    source_pages: tuple[int, ...] = ()
    page_program: tuple[tuple[tuple[int, int, int, int], ...], ...] = ()
    draft: DraftWindowBinding | None = None
    provider_tables: tuple[tuple[int, ...], ...] = ()

    @property
    def logical_m(self) -> int:
        if self.draft is not None:
            return len(self.request_ids) * (self.draft.depth + 1)
        return sum(self.owner_lengths) if self.owner_lengths else len(self.request_ids)


    def resident_updates(
        self,
        *,
        previous_batch_id: int,
        captured_m: int,
        request_ids: tuple[str, ...],
        offsets: tuple[int, ...],
        page_counts: tuple[int, ...],
        block_size: int,
    ) -> dict[int, tuple[int, int]] | None:
        """Validate sparse updates against the completed resident transaction."""
        updates = {physical: (index, page) for physical, index, page in self.page_updates}
        scheduled = {(0, physical): (index, page)
                     for physical, index, page in self.page_updates}
        for step, physical, index, page in self.future_page_updates:
            if not 1 <= step < self.max_steps or (step, physical) in scheduled:
                return None
            scheduled[step, physical] = (index, page)
        if (
            not request_ids
            or len(offsets) != len(request_ids)
            or len(page_counts) != len(request_ids)
            or type(self.max_steps) is not int
            or not 1 <= self.max_steps <= RESIDENT_STEP_CAPACITY
            or (self.max_steps > 1 and len(self.eos_token_ids) != len(request_ids))
            or (self.eos_token_ids and len(self.eos_token_ids) != len(request_ids))
            or any(type(token) is not int or token < 0
                   for ids in self.eos_token_ids for token in ids)
            or not isinstance(self.max_duration_seconds, int | float)
            or not math.isfinite(self.max_duration_seconds)
            or self.max_duration_seconds < 0
            or (self.max_steps > 1 and self.max_duration_seconds == 0)
            or self.previous_batch_id != previous_batch_id
            or self.captured_m != captured_m
            or self.request_ids != request_ids
            or len(updates) != len(self.page_updates)
            or not self._valid_prompt_program(len(request_ids))
            or type(self.source_pages) is not tuple or type(self.page_program) is not tuple
        ):
            return None
        if self.provider_tables and (
            self.draft is not None or not self.owner_lengths
            or type(self.provider_tables) is not tuple
            or len(self.provider_tables) != len(request_ids)
            or any(type(table) is not tuple or not table or len(set(table)) != len(table)
                   or any(type(page) is not int or page < 0 for page in table)
                   for table in self.provider_tables)
            or any(remaining <= self.max_steps * width for remaining, width in zip(
                self.prompt_remaining, self.owner_lengths, strict=True))
        ):
            return None
        if self.owner_lengths:
            if (self.source_rows or self.page_updates or self.future_page_updates
                    or not any(width > 1 for width in self.owner_lengths)
                    or any(width > 1 and remaining <= self.max_steps * width
                           for width, remaining in zip(self.owner_lengths, self.prompt_remaining,
                                                       strict=True))):
                return None
            return {} if valid_owner_page_program(
                self.owner_lengths, self.source_pages, self.page_program, offsets, page_counts,
                steps=self.max_steps, capacity=captured_m, block_size=block_size,
            ) else None
        if (
            self.source_pages or self.page_program
            or set(scheduled) != {
                (step, physical)
                for step in range(self.max_steps)
                for physical, offset in enumerate(offsets)
                if (offset + 1 + step) % block_size == 0
            }
            or any(index != page_counts[physical] + (offsets[physical] + step) // block_size
                   for (step, physical), (index, _page) in scheduled.items())
            or any(not 0 <= physical < len(offsets) for physical in updates)
            or any(
                (offset >= block_size - 1) != (physical in updates)
                for physical, offset in enumerate(offsets)
            )
            or any(index != page_counts[physical] for physical, (index, _page) in updates.items())
        ):
            return None
        return updates

    def _valid_prompt_program(self, rows: int) -> bool:
        if any(type(value) is not tuple for value in (
            self.prompt_remaining, self.prompt_tokens, self.prompt_positions,
            self.owner_lengths,
        )):
            return False
        if self.owner_lengths and (
            len(self.owner_lengths) != rows
            or any(type(width) is not int or width <= 0 for width in self.owner_lengths)
            or sum(self.owner_lengths) > self.captured_m
        ):
            return False
        if not self.prompt_remaining:
            return not self.prompt_tokens and not self.prompt_positions and not self.owner_lengths
        widths = self.owner_lengths or (1,) * rows
        return (
            len(self.prompt_remaining) == rows
            and len(self.prompt_tokens) == rows
            and len(self.prompt_positions) == rows
            and all(
                type(remaining) is int and 0 <= remaining < 2**31
                and type(tokens) is tuple and type(positions) is tuple
                and len(tokens) == min(remaining, self.max_steps * width)
                and len(positions) == len(tokens)
                and all(type(token) is int and token >= 0 for token in tokens)
                and all(type(axes) is tuple and len(axes) == 3
                        and all(type(p) is int for p in axes)
                        for axes in positions)
                for remaining, tokens, positions, width in zip(
                    self.prompt_remaining, self.prompt_tokens, self.prompt_positions, widths,
                    strict=True,
                )
            )
        )


@dataclass(frozen=True, slots=True)
class SegmentResult:
    request_id: str
    committed_row_count: int
    output_token_ids: tuple[int, ...]


def completed_resident_pages(
    offsets: tuple[int, ...],
    page_counts: tuple[int, ...],
    schedule: tuple[tuple[int, int, int, int], ...],
    steps: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Reconcile the completed prefix of an already validated page grant.

    Visit each grant once. Temporal step order determines the last installed
    page even when the immutable command stores grants in a different order.
    The caller completes this state consumption before the topology receipt.
    """
    if type(steps) is not int or steps < 0 or len(offsets) != len(page_counts):
        raise BatchProtocolError("resident page completion differs from its row domain")
    installed: dict[int, int] = {}
    added = [0] * len(offsets)
    for step, physical, _index, _page in schedule:
        if not 0 <= physical < len(offsets) or step < 0:
            raise BatchProtocolError("resident page completion has an invalid grant owner")
        if step < steps:
            installed[physical] = max(installed.get(physical, -1), step)
            added[physical] += 1
    return (
        tuple(
            steps - installed[physical] - 1 if physical in installed else offset + steps
            for physical, offset in enumerate(offsets)
        ),
        tuple(value + added[physical] for physical, value in enumerate(page_counts)),
    )


@dataclass(frozen=True, slots=True)
class BatchResult:
    rank: int
    batch_id: int
    request_ids: tuple[str, ...]
    token_ids: tuple[int, ...]
    segment_results: tuple[SegmentResult, ...] = ()
    target_ms: float = 0.0
    proposal_samples: tuple[tuple[int, float], ...] = ()
    step_ms: float = 0.0
    continuation_index: int = 0
    continuation_complete: bool = True
    # None: this producer has no GPU-period witness. Empty: a resident GPU
    # window is still publishing; its completed intervals arrive at the end.
    gpu_step_intervals_ms: tuple[float, ...] | None = None
    # Nonzero only after an ordinary carried round bound the qualified window.
    resident_draft_depth: int = 0
    # An initial/mismatched pending proposal needed seeding before this round.
    # Its one-time cost is not a steady carried-round execution price.
    draft_ingress: bool = False
    # A sealed review/repair graph followed by qualified, prefix-masked heads.
    # This is a compound cost domain; it does not grant a resident window.
    pipelined_draft_depth: int = 0
    # Present only for a completed request-owned CPU matcher transaction.
    grammar_cost: GrammarRoundCost | None = None


def resident_gpu_step_intervals(clocks: list[list[int]]) -> tuple[float, ...]:
    """Device periods between identical model-end boundaries in one window.

    Unlike CPU publication intervals, these include intervening sampling,
    communication, control and ingress without including host queue delays.
    A single model duration cannot witness a complete device period.
    """
    intervals = []
    previous_end = None
    for begin, end in clocks:
        if begin < 0 or end < begin or (previous_end is not None and end <= previous_end):
            raise BatchProtocolError("resident GPU clock order is invalid")
        if previous_end is not None:
            intervals.append((end - previous_end) / 1_000_000)
        previous_end = end
    return tuple(intervals)


def advance_carried_draft_rows(rows, pending, *, depth, block_size):
    """Advance an unchanged leased owner map from committed pending records.

    Only existing page tables are reachable. Intermediate checkpoints retain
    their physical scratch rows; neither padding nor segment continuation is
    inferred to be a new request.
    """
    advanced = []
    owners = set()
    for start, stop in batch_segments(rows):
        root, last = rows[start], rows[stop - 1]
        owner = root.state_source_slot
        carried = pending[owner]
        if (stop - start != depth + 1 or owner in owners or carried is None
                or last.state_destination_slot != owner
                or carried[0] != root.request_id or carried[3] != root.sampling
                or len(carried[6]) != depth
                or not 1 <= carried[1] - root.mtp_position <= depth + 1
                or carried[4] - root.sample_ordinal != carried[1] - root.mtp_position
                or any(not row.is_decode or row.input_feature
                       or row.state_source_slot != owner for row in rows[start:stop])):
            raise BatchProtocolError("carried continuation changed its owner transaction")
        owners.add(owner)
        delta = carried[1] - root.mtp_position
        for local, row in enumerate(rows[start:stop]):
            position, mtp_position = root.position + delta + local, carried[1] + local
            if (min(position, mtp_position) < 0
                    or position // block_size >= len(row.page_table)
                    or mtp_position // block_size >= len(row.mtp_page_table)):
                raise BatchProtocolError("carried continuation exceeded its existing page grants")
            axes = None if carried[5] is None else tuple(
                carried[5][axis] + row.position_axes[axis] - root.position_axes[axis]
                for axis in range(3))
            page = row.page_table[position // block_size]
            advanced.append(replace(
                row, token_id=carried[2] if local == 0 else carried[6][local - 1],
                position=position, mtp_position=mtp_position,
                sample_ordinal=carried[4] + local, position_axes=axes,
                page_source=page, page_destination=page,
                mtp_page_destination=row.mtp_page_table[mtp_position // block_size],
                reset_state=False, new_pages=(),
            ))
    return tuple(advanced)


def run_grammar_draft_continuation(
    command, executor, *, template, execute, previous_batch_id, request_ids,
    rank, sampler_rank, completion_statuses, completion_barrier, timeout,
    interrupt, publish, grammar_pool,
):
    """Run qualified masked rounds inside one already-owned topology lease."""
    binding = command.draft
    round_ = getattr(executor, "_pipelined_draft_round", None)
    if (binding is None or not execution_ready(round_, "grammar_continuation_qualified")
            or not (getattr(round_, "_last_review_only", False)
                    or getattr(round_, "_last_target_constraints", False))
            or binding.depth != round_.depth or template is None
            or command.previous_batch_id != previous_batch_id
            or command.captured_m != executor.captured_m
            or command.request_ids != request_ids or not template.constraints
            or tuple(dict.fromkeys(row.request_id for row in template.rows)) != request_ids
            or tuple(template.rows[start].state_source_slot
                     for start, _ in batch_segments(template.rows)) != binding.owners
            or not 2 <= command.max_steps <= RESIDENT_STEP_CAPACITY
            or grammar_pool is None
            or any((command.page_updates, command.future_page_updates, command.prompt_remaining,
                    command.prompt_tokens, command.prompt_positions, command.source_rows,
                    command.owner_lengths, command.source_pages, command.page_program))):
        raise BatchProtocolError("masked continuation differs from its completed transaction")
    world_size = len(completion_statuses) // 2
    if (world_size < 1 or len(completion_statuses) != 2 * world_size
            or not 0 <= sampler_rank < world_size or not 0 <= rank < world_size):
        raise BatchProtocolError("masked continuation storage differs from its topology")
    statuses = (completion_statuses.get_obj() if hasattr(completion_statuses, "get_obj")
                else completion_statuses)
    remaining = dict(zip(request_ids, binding.remaining, strict=True))
    decision_tokens = list(binding.decision_tokens)
    decision_rounds = list(binding.decision_rounds)
    rows = template.rows
    begun = time.monotonic()
    forward_ahead = bool(
        execution_ready(round_, "constraint_continuation_qualified")
        and getattr(round_, "_last_target_constraints", False))
    if forward_ahead:
        round_.bind_constraint_continuation(binding)
    try:
        for index in range(command.max_steps):
            started = time.monotonic_ns()
            if forward_ahead and index == 0:
                round_.start_constraint_forward()
            rows = advance_carried_draft_rows(
                rows, round_.pool.metadata, depth=binding.depth, block_size=executor._block_size)
            roots = tuple(rows[start] for start, _ in batch_segments(rows))
            if (index == 0 and tuple(row.mtp_position for row in roots) != binding.target_positions
                    or any(row.page_table != target or row.mtp_page_table != provider
                           or row.position + binding.depth >= target_limit
                           or row.mtp_position + 2 * binding.depth >= provider_limit
                           or remaining[row.request_id] < binding.depth + 1
                           for row, target, provider, target_limit, provider_limit in zip(
                               roots, binding.target_tables, binding.provider_tables,
                               binding.target_limits, binding.provider_limits, strict=True))):
                raise BatchProtocolError("masked continuation exceeded its committed extent")
            constraints = tuple(replace(
                original, revision=grammar_pool.sessions[original.request_id].revision,
                remaining=remaining[original.request_id],
            ) for original in template.constraints)
            result = execute(BatchCommand(command.batch_id, command.captured_m, rows, constraints))
            if result.pipelined_draft_depth != binding.depth or result.draft_ingress:
                raise BatchProtocolError("masked continuation lost its qualified carried round")
            executor.slot.synchronize()
            # The carried executor has consumed every graph guard and feeding-stream
            # result before returning. Each rank advances the same committed cursor.
            for owner_index, root in enumerate(roots):
                count = round_.pool.metadata[root.state_source_slot][1] - root.mtp_position
                if not 1 <= count <= binding.depth + 1:
                    raise BatchProtocolError("masked continuation returned an invalid prefix")
                remaining[root.request_id] -= count
                if decision_tokens:
                    decision_tokens[owner_index] = max(0, decision_tokens[owner_index] - count)
                    decision_rounds[owner_index] = max(0, decision_rounds[owner_index] - 1)
            predetermined_end = index + 1 == command.max_steps
            slot = (index % 2) * world_size + sampler_rank
            if rank == sampler_rank and not predetermined_end:
                statuses[slot] = int(
                    any(left < binding.depth + 1 for left in remaining.values())
                    or (bool(decision_tokens) and not any((*decision_tokens, *decision_rounds)))
                    or any(original.request_id not in grammar_pool.sessions
                           for original in template.constraints)
                    or any(token in eos for part, eos in zip(
                        result.segment_results, command.eos_token_ids or ((),) * len(roots),
                        strict=True) for token in part.output_token_ids)
                    or time.monotonic() - begun >= command.max_duration_seconds
                    or (interrupt is not None and interrupt.is_set())
                )
            completion_barrier.wait(timeout)
            complete = predetermined_end or bool(statuses[slot])
            # Same alternating-slot consumption fence as ordinary continuation.
            if complete and not predetermined_end:
                completion_barrier.wait(timeout)
            if forward_ahead and not complete:
                round_.start_constraint_forward()
            if rank == sampler_rank:
                publish(replace(
                    result, request_ids=request_ids, continuation_index=index,
                    continuation_complete=complete,
                    step_ms=(time.monotonic_ns() - started) / 1e6,
                ))
            if complete:
                return replace(template, rows=rows, constraints=constraints)
    finally:
        if forward_ahead:
            round_.close_constraint_continuation()


def run_draft_continuation(
    command, executor, *, previous_batch_id, request_ids, rank, sampler_rank,
    completion_barrier, timeout, interrupt, publish, grammar_pool=None,
):
    """Publish carried rounds while the caller retains the entire joint lease."""
    binding = command.draft
    round_ = getattr(executor, "_pipelined_draft_round", None)
    window = getattr(round_, "active_draft_window", None)
    if (binding is None or not execution_ready(window)
            or binding.depth != round_.depth
            or command.previous_batch_id != previous_batch_id
            or command.captured_m != executor.captured_m
            or command.request_ids != request_ids
            or len(request_ids) != len(binding.owners)
            or not 2 <= command.max_steps <= RESIDENT_STEP_CAPACITY
            or any((command.page_updates, command.future_page_updates, command.prompt_remaining,
                    command.prompt_tokens, command.prompt_positions, command.source_rows,
                    command.owner_lengths, command.source_pages, command.page_program))):
        raise BatchProtocolError("draft continuation differs from its completed transaction")
    window.bind(
        owners=binding.owners, target_positions=binding.target_positions,
        target_limits=binding.target_limits, provider_limits=binding.provider_limits,
        remaining=binding.remaining, eos=command.eos_token_ids,
        max_steps=command.max_steps, max_duration_seconds=command.max_duration_seconds,
        target_tables=binding.target_tables or None, provider_tables=binding.provider_tables or None,
        decision_tokens=binding.decision_tokens, decision_rounds=binding.decision_rounds,
    )
    remaining = dict(zip(request_ids, binding.remaining, strict=True))

    def commit_constraints(receipt):
        if not getattr(window, "requires_budget", False):
            return
        if grammar_pool is None:
            raise BatchProtocolError("budget continuation lost its committed constraint state")
        bindings, outputs = [], {}
        for part in receipt.segment_results:
            session = grammar_pool.sessions.get(part.request_id)
            if session is not None:
                bindings.append(replace(session.binding, revision=session.revision,
                                        remaining=remaining[part.request_id]))
                outputs[part.request_id] = part.output_token_ids
                remaining[part.request_id] -= len(part.output_token_ids)
        if bindings:
            grammar_pool.begin(tuple(bindings)).commit(outputs)

    window.replay()
    index = 0
    deadline = time.monotonic() + timeout
    if rank == sampler_rank:
        while True:
            if interrupt is not None and interrupt.is_set():
                window.host_flags[2] = 1
            if time.monotonic() > deadline:
                window.host_flags[2] = 1
                raise BatchProtocolError("draft continuation completion timed out")
            if index >= len(window.records()):
                time.sleep(0.0001)
                continue
            receipt = window.receipt(index, batch_id=command.batch_id, request_ids=request_ids)
            commit_constraints(receipt)
            if receipt.continuation_complete:
                break
            publish(receipt)
            index += 1
    window.consume()
    if rank != sampler_rank and getattr(window, "requires_budget", False):
        for index in range(len(window.records())):
            commit_constraints(window.receipt(index, batch_id=command.batch_id,
                                               request_ids=request_ids))
    if getattr(window, "requires_budget", False):
        sessions = [grammar_pool.sessions.get(request) for request in request_ids]
        window.round.budget.bind(
            [None if session is None else session.matcher.integer_snapshot()
             for session in sessions],
            stop_enabled=[session is not None and bool(session.stop_ids) for session in sessions],
            terminated=[session is not None and session.terminated for session in sessions],
        )
    # Final publication alone permits the controller to release every rank's
    # graph, page, state and pending-proposal ownership.
    completion_barrier.wait(timeout)
    if rank == sampler_rank:
        publish(replace(receipt, resident_draft_depth=binding.depth))


def run_grammar_continuation(
    command, executor, *, bindings, previous_batch_id, request_ids, offsets,
    page_counts, block_size, rank, sampler_rank, completion_statuses,
    completion_barrier, timeout, interrupt, publish, grammar_pool, prepare_masks=None,
):
    """Masked K0 steps over the existing resident input and sampling graphs.

    The completed bind owns the row map, parameters and private append pages.
    CPU matchers stay transactional. Only target forward may precede their
    preparation; sampling always waits for the mask of its exact prefix.
    """
    plans = getattr(executor._sampling, "masked_plans", ())
    if (not plans or not all(execution_ready(plan) for plan in plans)
            or getattr(executor, "_input_update_graph", None) is None
            or not bindings or grammar_pool is None or command.draft is not None
            or any((command.source_rows, command.prompt_remaining, command.owner_lengths,
                    command.future_page_updates, command.page_program, command.provider_tables))
            or command.resident_updates(
                previous_batch_id=previous_batch_id, captured_m=executor.captured_m,
                request_ids=request_ids, offsets=offsets, page_counts=page_counts,
                block_size=block_size) is None
            or any(binding.request_id not in request_ids for binding in bindings)
            or len({binding.request_id for binding in bindings}) != len(bindings)
            or any(binding.remaining <= command.max_steps for binding in bindings)
            or command.max_steps > min(block_size - offset - 1
                                       if offset < block_size - 1 else block_size
                                       for offset in offsets)):
        raise BatchProtocolError("masked K0 continuation differs from its completed owners")
    world_size = len(completion_statuses) // 2
    if (world_size < 1 or len(completion_statuses) != 2 * world_size
            or not 0 <= sampler_rank < world_size or not 0 <= rank < world_size):
        raise BatchProtocolError("masked K0 completion storage differs from its topology")
    statuses = (completion_statuses.get_obj() if hasattr(completion_statuses, "get_obj")
                else completion_statuses)
    begun = time.monotonic()
    queued = False
    for index in range(command.max_steps):
        started = time.monotonic_ns()
        updates = command.page_updates if index == 0 else ()
        if not queued:
            executor.replay_steady_continuation(len(request_ids), updates)
            executor._constraint_forward_queued_ns = time.monotonic_ns()
        queued = False
        bindings = tuple(replace(binding, revision=binding.revision + 1,
                                 remaining=binding.remaining - 1) for binding in bindings)
        grammar = grammar_pool.begin(bindings)
        owners = [request in grammar.bindings for request in request_ids]
        prepared = (None if prepare_masks is None else prepare_masks(
            executor, grammar, {request: () for request in grammar.bindings}))
        masks = [
            (grammar.mask(request, 0) if prepared is None else prepared[request][0][0])
            if owned else None for request, owned in zip(request_ids, owners, strict=True)
        ]
        upload_started = time.monotonic_ns()
        executor._sampling.bind_constraints(
            masks, owners)
        executor._constraint_mask_upload_ns = time.monotonic_ns() - upload_started
        tokens = executor.sample_steady_continuation(len(request_ids))
        local = (tokens if rank == sampler_rank else
                 tuple(int(token) for token in executor._token_ids[:len(request_ids)].cpu().tolist()))
        if len(local) != len(request_ids):
            raise BatchProtocolError("masked K0 returned a different token count")
        grammar.commit({request: (token,) for request, token in
                        zip(request_ids, local, strict=True) if request in grammar.bindings})
        executor._sampling.constraint_bound = False
        executor.slot.synchronize()
        target_ms = executor.replay_elapsed_ms()
        executor._constraint_forward_ms = target_ms
        changed = {physical for physical, _column, _page in updates}
        offsets = tuple(0 if physical in changed else offset + 1
                        for physical, offset in enumerate(offsets))
        page_counts = tuple(count + int(physical in changed)
                            for physical, count in enumerate(page_counts))
        predetermined_end = index + 1 == command.max_steps
        status = (index % 2) * world_size + sampler_rank
        if rank == sampler_rank and not predetermined_end:
            statuses[status] = int(
                any(token in eos for token, eos in zip(local, command.eos_token_ids,
                                                       strict=True))
                or any(binding.request_id not in grammar_pool.sessions for binding in bindings)
                or time.monotonic() - begun >= command.max_duration_seconds
                or (interrupt is not None and interrupt.is_set()))
        completion_barrier.wait(timeout)
        complete = predetermined_end or bool(statuses[status])
        if complete and not predetermined_end:
            completion_barrier.wait(timeout)
        if not complete:
            # The next target uses GPU token feedback. No CPU output/mask
            # buffer is a forward operand; its matcher remains prefix-ordered.
            executor.replay_steady_continuation(len(request_ids), ())
            executor._constraint_forward_queued_ns = time.monotonic_ns()
            queued = True
        if rank == sampler_rank:
            publish(BatchResult(
                rank, command.batch_id, request_ids, local, target_ms=target_ms,
                step_ms=(time.monotonic_ns() - started) / 1e6,
                continuation_index=index, continuation_complete=complete,
                grammar_cost=grammar.cost_witness(),
            ))
        if complete:
            break
    return offsets, page_counts, bindings


def run_resident_continuation(
    command: ContinueBatchCommand, executor, *, previous_batch_id: int,
    request_ids: tuple[str, ...], offsets: tuple[int, ...], page_counts: tuple[int, ...],
    block_size: int, rank: int, sampler_rank: int, completion_statuses,
    completion_barrier, timeout: float, interrupt, publish,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Run sealed resident steps under the caller's still-held topology lease.

    Each step completes and consumes its real token before advancing. Every
    rank rendezvous remains in place. Only the final receipt permits the
    controller to retire ownership or release the lease. No step crosses an
    unallocated page, changes the row map or exceeds the granted token budget.
    """
    source_rows = command.source_rows
    source_count = len(request_ids)
    if type(source_rows) is not tuple:
        raise BatchProtocolError("resident compaction map is not immutable")
    if source_rows and command.owner_lengths:
        raise BatchProtocolError("segmented owners cannot use single-token row compaction")
    if source_rows:
        if (len(offsets) != source_count or len(page_counts) != source_count
                or len(set(source_rows)) != len(source_rows)
                or len(source_rows) != len(command.request_ids)
                or not command.request_ids
                or 1 << (len(command.request_ids) - 1).bit_length() != command.captured_m
                or any(type(row) is not int or not 0 <= row < source_count for row in source_rows)
                or command.request_ids != tuple(request_ids[row] for row in source_rows)):
            raise BatchProtocolError("resident compaction differs from source request ownership")
        request_ids = tuple(request_ids[row] for row in source_rows)
        offsets = tuple(offsets[row] for row in source_rows)
        page_counts = tuple(page_counts[row] for row in source_rows)
        command = replace(command, source_rows=())
    updates = command.resident_updates(
        previous_batch_id=previous_batch_id, captured_m=executor.captured_m,
        request_ids=request_ids, offsets=offsets, page_counts=page_counts,
        block_size=block_size,
    )
    if updates is None:
        raise BatchProtocolError("steady continuation differs from its resident batch")
    if command.owner_lengths and (
        command.owner_lengths != getattr(executor, "_resident_owner_lengths", ())
        or command.logical_m != executor._bound_active_rows
        or command.source_pages
        != getattr(executor, "_resident_owner_pages", ())
    ):
        raise BatchProtocolError("resident segments differ from their bound owners")
    window = getattr(executor, "_resident_window", None)
    if source_rows:
        compaction = getattr(executor, "_resident_compaction", None)
        if (not execution_ready(compaction) or window is None
                or getattr(executor, "_features_bound", False)
                or window.supports(command) is not True):
            raise BatchProtocolError("resident compaction has no qualified graph")
        compaction.bind(source_rows, source_limit=source_count)
        compaction.replay()
        executor._sampling.all_greedy = True
        executor._sampling_bound_all_random = False
        executor._bound_active_rows = len(request_ids)
        executor._features_bound = False
    stochastic_window = getattr(executor, "_resident_stochastic_window", None)
    if stochastic_window is not None and stochastic_window.supports(command) is True:
        window = stochastic_window
    if window is not None and window.supports(command) is True:
        return window.run(
            command, request_ids=request_ids, offsets=offsets, page_counts=page_counts,
            rank=rank, sampler_rank=sampler_rank, completion_barrier=completion_barrier,
            timeout=timeout, interrupt=interrupt, publish=publish,
        )
    if command.prompt_remaining:
        raise BatchProtocolError("resident input program has no qualified graph")
    world_size = len(completion_statuses) // 2
    if (world_size < 1 or len(completion_statuses) != 2 * world_size
            or not 0 <= sampler_rank < world_size or not 0 <= rank < world_size):
        raise BatchProtocolError("resident completion storage differs from its topology")
    # This single-writer loan lasts only for the exclusively leased window.
    # Publication/consumption is ordered by the existing process barriers;
    # locking every individual status load would add serial rank handoffs.
    statuses = (completion_statuses.get_obj() if hasattr(completion_statuses, "get_obj")
                else completion_statuses)
    window_started = time.monotonic()
    for index in range(command.max_steps):
        completion_slot = (index % 2) * world_size + sampler_rank
        started = time.monotonic_ns()
        step_updates = (command.page_updates if index == 0 else tuple(
            (physical, page_index, page)
            for step, physical, page_index, page in command.future_page_updates if step == index
        ))
        changed = {physical for physical, _page_index, _page in step_updates}
        executor.replay_steady_continuation(
            len(request_ids), step_updates
        )
        tokens = executor.sample_steady_continuation(len(request_ids))
        executor.slot.synchronize()
        predetermined_end = index + 1 == command.max_steps
        if rank == sampler_rank and len(tokens) != len(request_ids):
            raise BatchProtocolError("resident continuation returned a different token count")
        if rank == sampler_rank and not predetermined_end:
            terminal = any(token in eos for token, eos in zip(
                tokens, command.eos_token_ids or ((),) * len(request_ids), strict=True
            ))
            statuses[completion_slot] = int(
                terminal
                or time.monotonic() - window_started >= command.max_duration_seconds
                or (interrupt is not None and interrupt.is_set())
            )
        completion_barrier.wait(timeout)
        complete = predetermined_end or bool(statuses[completion_slot])
        # Alternate the already-owned startup status slots: after the step
        # barrier a fast rank may begin another step before its peers read
        # this flag. The next barrier bounds that lead to one generation.
        # Before publishing a final receipt, all peers must have consumed the
        # terminal flag; a new window can then safely reuse either slot.
        # At the granted count boundary every rank already knows it must
        # finish. No shared flag is read or written at that boundary, so its
        # existing completion barrier also seals reuse by the next window.
        if complete and not predetermined_end:
            completion_barrier.wait(timeout)
        offsets = tuple(0 if physical in changed else offset + 1
                        for physical, offset in enumerate(offsets))
        page_counts = tuple(count + int(physical in changed)
                            for physical, count in enumerate(page_counts))
        if rank == sampler_rank:
            publish(BatchResult(
                rank, command.batch_id, request_ids, tokens,
                target_ms=executor.replay_elapsed_ms(),
                step_ms=(time.monotonic_ns() - started) / 1e6,
                continuation_index=index, continuation_complete=complete,
            ))
        if complete:
            break
    return offsets, page_counts


@dataclass(frozen=True, slots=True)
class CloseCommand:
    pass


WorkerCommand = BatchCommand | ContinueBatchCommand | CloseCommand

_WORKER_WIRE_TAG = "cubic.worker.v1"
_WORKER_CONTROL_WIRE_TAG = "cubic.worker.v2"
_WORKER_CLOSE = 0
_WORKER_BATCH = 1
_WORKER_CONTINUE = 2


def batch_segments(rows: tuple[BatchRow, ...]) -> tuple[tuple[int, int], ...]:
    """Validate contiguous request segments and return their half-open ranges."""

    segments: list[tuple[int, int]] = []
    start = 0
    while start < len(rows):
        length = rows[start].segment_length
        stop = start + length
        if length <= 0 or stop > len(rows):
            raise BatchProtocolError("batch has an invalid segment boundary")
        head = rows[start]
        for offset, row in enumerate(rows[start:stop]):
            if (
                row.request_id != head.request_id
                or row.position != head.position + offset
                or row.state_source_slot != head.state_source_slot
                or row.segment_length != (length if offset == 0 else 0)
                or (offset > 0 and row.reset_state)
            ):
                raise BatchProtocolError("batch request segment is not contiguous")
        segments.append((start, stop))
        start = stop
    return tuple(segments)


def serialize_worker_command(command: WorkerCommand) -> bytes:
    if type(command) is CloseCommand:
        wire = (_WORKER_WIRE_TAG, _WORKER_CLOSE)
    elif type(command) is BatchCommand:
        controls = any(
            b.reasoning_control is not None or b.stop_token_ids is not None
            for b in command.constraints
        )
        wire = (
            _WORKER_CONTROL_WIRE_TAG if controls else _WORKER_WIRE_TAG,
            _WORKER_BATCH,
            command.batch_id,
            command.captured_m,
            tuple(
                (
                    row.request_id,
                    row.token_id,
                    row.position,
                    row.state_source_slot,
                    row.state_destination_slot,
                    row.page_source,
                    row.page_destination,
                    row.page_table,
                    row.reset_state,
                    row.new_pages,
                    row.segment_length,
                    row.sampling,
                    row.sample_ordinal,
                    row.mtp_position,
                    row.mtp_page_destination,
                    row.mtp_page_table,
                    row.is_decode,
                    row.position_axes,
                    row.input_feature,
                )
                for row in command.rows
            ),
        )
        if command.constraints or command.released_constraints:
            wire += (
                tuple(
                    (
                        b.request_id,
                        b.meta.kind.value,
                        b.meta.canonical_payload,
                        b.meta.canonical_options,
                        b.revision,
                        b.remaining,
                        b.reasoning_end,
                    ) + ((
                        None if b.reasoning_control is None else (
                            b.reasoning_control.start, b.reasoning_control.end,
                            b.reasoning_control.token_budget,
                            b.reasoning_control.initially_reasoning,
                            b.reasoning_control.initial_reasoning_tokens,
                            b.reasoning_control.implicit_ends, b.reasoning_control.initial_tail,
                        ),
                        b.stop_token_ids,
                    ) if controls else ())
                    for b in command.constraints
                ),
                command.released_constraints,
            )
    elif type(command) is ContinueBatchCommand:
        wire = (
            _WORKER_WIRE_TAG,
            _WORKER_CONTINUE,
            command.batch_id,
            command.previous_batch_id,
            command.captured_m,
            command.request_ids,
            command.page_updates,
            command.max_steps,
            command.eos_token_ids,
            command.max_duration_seconds,
            command.future_page_updates,
            command.prompt_remaining,
            command.prompt_tokens,
            command.prompt_positions,
            command.source_rows,
            command.owner_lengths,
            command.source_pages,
            command.page_program,
            (None if command.draft is None else (
                command.draft.depth, command.draft.owners, command.draft.target_positions,
                command.draft.target_limits, command.draft.provider_limits, command.draft.remaining,
                command.draft.target_tables, command.draft.provider_tables,
                command.draft.decision_tokens, command.draft.decision_rounds,
            )),
            command.provider_tables,
        )
    else:
        raise BatchProtocolError("controller produced an unknown worker command")
    return pickle.dumps(wire, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_worker_command(payload: bytes) -> WorkerCommand:
    try:
        wire = pickle.loads(payload)
    except Exception as error:
        raise BatchProtocolError("worker command payload is invalid") from error
    if (type(wire) is not tuple or len(wire) < 2
            or wire[0] not in (_WORKER_WIRE_TAG, _WORKER_CONTROL_WIRE_TAG)):
        raise BatchProtocolError("worker received an unknown command")
    controls = wire[0] == _WORKER_CONTROL_WIRE_TAG
    if controls and (wire[1] != _WORKER_BATCH or len(wire) != 7):
        raise BatchProtocolError("worker control command payload is invalid")
    if wire == (_WORKER_WIRE_TAG, _WORKER_CLOSE):
        return CloseCommand()
    if len(wire) in (11, 14, 15, 18, 19, 20) and wire[1] == _WORKER_CONTINUE:
        try:
            return ContinueBatchCommand(
                wire[2], wire[3], wire[4], tuple(wire[5]), tuple(wire[6]),
                wire[7], tuple(tuple(ids) for ids in wire[8]), wire[9], tuple(wire[10]),
                *(tuple(wire[11:18]) if len(wire) >= 14 else ()),
                **({"draft": None if wire[18] is None else DraftWindowBinding(*wire[18])}
                   if len(wire) >= 19 else {}),
                **({"provider_tables": tuple(tuple(table) for table in wire[19])}
                   if len(wire) == 20 else {}),
            )
        except (TypeError, ValueError) as error:
            raise BatchProtocolError("worker continuation payload is invalid") from error
    if len(wire) not in (5, 7) or wire[1] != _WORKER_BATCH:
        raise BatchProtocolError("worker command payload is invalid")
    try:
        rows = tuple(BatchRow(*values) for values in wire[4])
        if len(wire) == 5:
            return BatchCommand(wire[2], wire[3], rows)
        from cubic_serve.contracts.request import StructuredOutputKind

        bindings = []
        # Definitions are immutable values; request cursors and controls are not
        # shared. Validate each distinct definition once within this message,
        # without retaining client schemas beyond the decoded command.
        definitions = {}
        for b in wire[5]:
            if type(b) is not tuple or len(b) != (9 if controls else 7):
                raise ValueError("worker constraint version differs from its fields")
            control = None
            if controls and b[7] is not None:
                if type(b[7]) is not tuple or len(b[7]) != 7:
                    raise ValueError("worker reasoning control has invalid fields")
                control = ReasoningControl(*b[7])
            # Preserve validation for nonstandard wire types that compare equal
            # to canonical strings (including str subclasses).
            key = (type(b[1]), b[1], type(b[2]), b[2], type(b[3]), b[3])
            meta = definitions.get(key)
            if meta is None:
                meta = StructuredOutputMeta(StructuredOutputKind(b[1]), b[2], b[3])
                definitions[key] = meta
            bindings.append(ConstraintBinding(
                b[0], meta,
                b[4], b[5], b[6], control, b[8] if controls else None,
            ))
        bindings = tuple(bindings)
        if (
            type(wire[6]) is not tuple
            or any(type(x) is not str for x in wire[6])
            or not {b.request_id for b in bindings}.issubset({row.request_id for row in rows})
        ):
            raise ValueError("constraint ownership differs from row ownership")
        return BatchCommand(wire[2], wire[3], rows, bindings, wire[6])
    except (IndexError, TypeError, ValueError) as error:
        raise BatchProtocolError("worker command payload is invalid") from error


__all__ = (
    "BatchCommand",
    "BatchProtocolError",
    "BatchResult",
    "BatchRow",
    "CloseCommand",
    "ContinueBatchCommand",
    "GenerateCommand",
    "SegmentResult",
    "WorkerCommand",
    "batch_segments",
    "deserialize_worker_command",
    "serialize_worker_command",
)
