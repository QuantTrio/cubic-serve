"""Rank-local logical-to-physical page allocation for scheduler plans.

The allocator owns only page-table geometry.  KV contents, ownership,
revisions, BF16 enforcement, and unreadable-slot checks remain exclusively in
``CacheStateStore``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock

import torch

from cubic_serve.runtime.cache.prefix import PrefixRestore
from cubic_serve.runtime.cache.transaction import CacheStateStore, RevisionConflict
from cubic_serve.runtime.scheduler import PageAllocation


@dataclass(frozen=True, slots=True)
class PageTableView:
    """Detached view whose tuple index is the logical page number.

    Released tables are retained as tombstones.  Their physical pages remain
    owned by the original request in ``CacheStateStore`` and are never reused
    by this allocator.
    """

    request_id: str
    rank_id: int
    logical_to_physical: tuple[int, ...]
    reserved_token_count: int
    prefix_token_count: int
    released: bool


@dataclass(slots=True)
class _RequestPageTable:
    physical_pages: list[int] = field(default_factory=list)
    reserved_token_count: int = 0
    prefix_token_count: int = 0
    released: bool = False
    reservations: dict[tuple[int, int], PageAllocation] = field(default_factory=dict)


class RankLocalPageAllocator:
    """A strict scheduler-facing page table backed by one rank-local store.

    New spans must be contiguous with the request's reservation frontier.
    Repeating the exact ``(token_start, token_count)`` span returns the same
    immutable allocation, which makes scheduler retry idempotent.  Context
    pages are always returned in logical page order; write pages cover only
    pages touched by the requested span.

    ``release`` retains only a logical tombstone; the store atomically retires
    the exact owner revision and returns its arena holds after the caller's
    executor barrier.
    """

    def __init__(self, store: CacheStateStore) -> None:
        if not isinstance(store, CacheStateStore):
            raise TypeError("rank-local allocator requires a CacheStateStore")
        self._store = store
        self._tables: dict[str, _RequestPageTable] = {}
        self._lock = RLock()

    @property
    def block_size_tokens(self) -> int:
        return self._store.layout.block_size_tokens

    @property
    def rank_id(self) -> int:
        return self._store.rank_id

    def allocate(
        self,
        *,
        request_id: str,
        token_start: int,
        token_count: int,
    ) -> PageAllocation:
        """Reserve one contiguous token span and return its physical mapping."""

        self._validate_request_id(request_id)
        if token_start < 0 or token_count <= 0:
            raise ValueError("token start must be non-negative and count positive")

        reservation_key = (token_start, token_count)
        with self._lock:
            table = self._tables.get(request_id)
            if table is None:
                if token_start != 0:
                    raise ValueError(
                        "a fresh request must reserve from token zero or bind a prefix"
                    )
                table = _RequestPageTable()
                self._tables[request_id] = table
            if table.released:
                raise RuntimeError(f"request {request_id!r} page table was released")

            previous = table.reservations.get(reservation_key)
            if previous is not None:
                return previous
            if token_start != table.reserved_token_count:
                raise ValueError(
                    "a new reservation must start at the contiguous token frontier; "
                    "only an exact prior span may be retried"
                )

            token_end = token_start + token_count
            required_page_count = self._page_count(token_end)
            while len(table.physical_pages) < required_page_count:
                # Append each successful allocation immediately.  If a store
                # operation fails, a retry continues from the safely owned
                # partial table instead of losing a physical page.
                physical_page = self._store.allocate_page(request_id)
                table.physical_pages.append(physical_page)

            first_write_page = token_start // self.block_size_tokens
            final_write_page = (token_end - 1) // self.block_size_tokens
            write_pages = tuple(
                table.physical_pages[logical_page]
                for logical_page in range(first_write_page, final_write_page + 1)
            )
            context_pages = tuple(table.physical_pages[:required_page_count])
            cache_slots = tuple(
                table.physical_pages[position // self.block_size_tokens] * self.block_size_tokens
                + position % self.block_size_tokens
                for position in range(token_start, token_end)
            )
            allocation = PageAllocation(
                cache_slots=cache_slots,
                context_pages=context_pages,
                write_pages=write_pages,
            )
            table.reservations[reservation_key] = allocation
            table.reserved_token_count = token_end
            return allocation

    def bind_restored_prefix(
        self,
        *,
        request_id: str,
        restore: PrefixRestore,
    ) -> PageTableView:
        """Bind a store-restored prefix as logical pages ``0..N-1``.

        ``CacheStateStore.restore_prefix`` preserves snapshot-page order in
        ``source_to_restored_pages``.  This method defines that tuple order as
        logical context order and validates every restored page as fully
        written. The reservation frontier starts at the arithmetic checkpoint,
        so an explicit replay tail may recompute and bit-check already cached KV.
        """

        self._validate_request_id(request_id)
        if restore.owner_id != request_id:
            raise ValueError("prefix restore owner must equal the scheduler request id")
        if restore.target_rank != self.rank_id:
            raise ValueError("prefix restore targets a different cache rank")
        if (
            restore.physical_block_boundary <= 0
            or restore.physical_block_boundary % self.block_size_tokens
        ):
            raise ValueError("restored physical prefix must be block aligned")
        if not (
            0
            <= restore.arithmetic_state_commit_boundary
            <= restore.physical_block_boundary
        ):
            raise ValueError("restored arithmetic boundary must lie within physical KV")
        if len(restore.replay_tokens) != (
            restore.physical_block_boundary - restore.arithmetic_state_commit_boundary
        ):
            raise ValueError("restored replay tokens do not bridge the prefix boundaries")

        expected_pages = restore.physical_block_boundary // self.block_size_tokens
        page_pairs = restore.source_to_restored_pages
        if len(page_pairs) != expected_pages:
            raise ValueError("restored page count does not match its physical boundary")
        source_pages = tuple(source for source, _ in page_pairs)
        restored_pages = tuple(target for _, target in page_pairs)
        if len(set(source_pages)) != len(source_pages):
            raise ValueError("restored prefix source pages must be unique")
        if len(set(restored_pages)) != len(restored_pages):
            raise ValueError("restored prefix target pages must be unique")

        with self._lock:
            if request_id in self._tables:
                raise ValueError(f"request {request_id!r} already has a page table")
            if self._store.current_revision(request_id) != restore.owner_revision:
                raise ValueError("prefix restore owner revision changed before binding")
            for page_id in restored_pages:
                # These public store reads jointly prove rank/owner identity
                # and that an allocator cannot turn partial KV into a prefix.
                self._store.page_revision(request_id, page_id)
                written = self._store.page_written_mask(request_id, page_id)
                if not bool(torch.all(written).item()):
                    raise ValueError("a restored prefix page must be fully written")

            self._tables[request_id] = _RequestPageTable(
                physical_pages=list(restored_pages),
                reserved_token_count=restore.arithmetic_state_commit_boundary,
                prefix_token_count=restore.physical_block_boundary,
            )
            return self.page_table(request_id)

    def release(
        self,
        request_id: str,
        *,
        expected_revision: int,
    ) -> PageTableView:
        """Retire exact owner storage and seal a non-reusable table tombstone.

        ``expected_revision`` protects the lifecycle boundary from an
        unexpected committed mutation.  The caller must coordinate its
        executor barrier before entering this lifecycle boundary.
        """

        self._validate_request_id(request_id)
        if expected_revision < 0:
            raise ValueError("expected revision must be non-negative")
        with self._lock:
            table = self._require_table(request_id)
            if table.released:
                raise RuntimeError(f"request {request_id!r} page table was released")
            try:
                self._store.retire_owner(
                    request_id,
                    expected_revision=expected_revision,
                    expected_page_ids=tuple(table.physical_pages),
                )
            except RevisionConflict as error:
                raise ValueError(
                    f"release revision changed: expected {expected_revision}"
                ) from error
            table.released = True
            table.reservations.clear()
            return self._view(request_id, table)

    def rollback_restored_binding(
        self,
        request_id: str,
        *,
        expected_pages: tuple[int, ...],
    ) -> None:
        """Undo only a fresh prefix binding that has never served a plan."""

        self._validate_request_id(request_id)
        with self._lock:
            table = self._require_table(request_id)
            if (
                tuple(table.physical_pages) != expected_pages
                or table.prefix_token_count <= 0
                or table.reservations
                or table.released
            ):
                raise RuntimeError("restored binding changed before admission rollback")
            del self._tables[request_id]

    def page_table(self, request_id: str) -> PageTableView:
        """Return a detached logical-order view, including released tombstones."""

        self._validate_request_id(request_id)
        with self._lock:
            return self._view(request_id, self._require_table(request_id))

    def _page_count(self, token_count: int) -> int:
        return (token_count + self.block_size_tokens - 1) // self.block_size_tokens

    def _require_table(self, request_id: str) -> _RequestPageTable:
        try:
            return self._tables[request_id]
        except KeyError as error:
            raise KeyError(f"unknown request page table {request_id!r}") from error

    def _view(self, request_id: str, table: _RequestPageTable) -> PageTableView:
        return PageTableView(
            request_id=request_id,
            rank_id=self.rank_id,
            logical_to_physical=tuple(table.physical_pages),
            reserved_token_count=table.reserved_token_count,
            prefix_token_count=table.prefix_token_count,
            released=table.released,
        )

    @staticmethod
    def _validate_request_id(request_id: str) -> None:
        if not request_id:
            raise ValueError("request id is required")
