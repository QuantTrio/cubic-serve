"""Request-local synchronous/asynchronous token mailbox."""

from __future__ import annotations

from typing_extensions import Self

import asyncio
import queue
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol


class TokenGenerationOverloaded(RuntimeError):
    """Request capacity is exhausted without making the executor unhealthy."""


class TokenSink(Protocol):
    def publish(self, token_id: int) -> None: ...

    def publish_many(self, token_ids: tuple[int, ...]) -> None: ...

    def finish(self) -> None: ...

    def fail(self, error: BaseException) -> None: ...


@dataclass(frozen=True, slots=True)
class _Failure:
    error: BaseException


_END = object()


class TokenStream(Iterator[int], AsyncIterator[int]):
    """Single-consumer mailbox shared by model-independent serving backends."""

    def __init__(
        self,
        on_close: Callable[[TokenStream, BaseException | None], None] | None = None,
        *,
        invalid_item_error: Callable[[], BaseException] | None = None,
    ) -> None:
        self._items: queue.SimpleQueue[object] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_event: asyncio.Event | None = None
        self._consumer_mode: str | None = None
        self._terminal = False
        self._released = False
        self._on_close = on_close
        self._invalid_item_error = invalid_item_error or (
            lambda: RuntimeError("token stream received an invalid item")
        )

    def __iter__(self) -> Self:
        return self

    def __aiter__(self) -> Self:
        return self

    def __next__(self) -> int:
        self._claim_consumer("sync")
        return self._consume(self._items.get(), asynchronous=False)

    async def __anext__(self) -> int:
        self._claim_consumer("async")
        loop = asyncio.get_running_loop()
        while True:
            with self._lock:
                if self._async_loop is None:
                    self._async_loop = loop
                    self._async_event = asyncio.Event()
                elif self._async_loop is not loop:
                    raise RuntimeError("token stream cannot move between event loops")
                assert self._async_event is not None
                self._async_event.clear()
                try:
                    item = self._items.get_nowait()
                except queue.Empty:
                    event = self._async_event
                else:
                    break
            await event.wait()
        return self._consume(item, asynchronous=True)

    def _claim_consumer(self, mode: str) -> None:
        with self._lock:
            if self._consumer_mode is None:
                self._consumer_mode = mode
            elif self._consumer_mode != mode:
                raise RuntimeError("token stream cannot mix sync and async consumers")

    def _consume(self, item: object, *, asynchronous: bool) -> int:
        if item is _END:
            self._release(None)
            if asynchronous:
                raise StopAsyncIteration
            raise StopIteration
        if isinstance(item, _Failure):
            self._release(item.error)
            raise item.error
        if type(item) is not int:
            error = self._invalid_item_error()
            self._release(error)
            raise error
        return item

    def publish(self, token_id: int) -> None:
        if type(token_id) is not int:
            raise TypeError("token stream publish requires an integer token id")
        self._put(token_id)

    def publish_many(self, token_ids: tuple[int, ...]) -> None:
        """Publish one committed decoder step with a single consumer wake-up."""

        wake = self._stage_many(token_ids)
        if wake is not None:
            self._wake(*wake)

    def _stage_many(
        self,
        token_ids: tuple[int, ...],
    ) -> tuple[asyncio.AbstractEventLoop, asyncio.Event] | None:
        if type(token_ids) is not tuple or not token_ids or any(
            type(token_id) is not int for token_id in token_ids
        ):
            raise TypeError("token stream bulk publish requires integer token ids")
        with self._lock:
            if self._terminal:
                return None
            for token_id in token_ids:
                self._items.put_nowait(token_id)
            loop, event = self._async_loop, self._async_event
        return (loop, event) if loop is not None and event is not None else None

    def _stage_one(
        self,
        token_id: int,
    ) -> tuple[asyncio.AbstractEventLoop, asyncio.Event] | None:
        if type(token_id) is not int:
            raise TypeError("token stream publish requires an integer token id")
        with self._lock:
            if self._terminal:
                return None
            self._items.put_nowait(token_id)
            loop, event = self._async_loop, self._async_event
        return (loop, event) if loop is not None and event is not None else None

    def finish(self) -> None:
        self._put(_END, terminal=True)

    def fail(self, error: BaseException) -> None:
        self._put(_Failure(error), terminal=True)

    def stop(self) -> None:
        with self._lock:
            self._terminal = True
            while True:
                try:
                    self._items.get_nowait()
                except queue.Empty:
                    break
            self._items.put_nowait(_END)
            loop, event = self._async_loop, self._async_event
        self._wake(loop, event)
        self._release(None)

    close = stop

    async def aclose(self) -> None:
        self.stop()

    def _put(self, item: object, *, terminal: bool = False) -> None:
        with self._lock:
            if self._terminal:
                return
            if terminal:
                self._terminal = True
            self._items.put_nowait(item)
            loop, event = self._async_loop, self._async_event
        self._wake(loop, event)

    @staticmethod
    def _wake(
        loop: asyncio.AbstractEventLoop | None,
        event: asyncio.Event | None,
    ) -> None:
        if loop is not None and event is not None:
            with suppress(RuntimeError):
                loop.call_soon_threadsafe(event.set)

    def _release(self, error: BaseException | None) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
            callback = self._on_close
        if callback is not None:
            callback(self, error)


class TokenWakeBatch:
    """Publish one engine step and wake each asyncio loop only once."""

    def __init__(self) -> None:
        self._events: dict[asyncio.AbstractEventLoop, list[asyncio.Event]] = {}
        self._deferred: list[tuple[TokenSink, tuple[int, ...]]] = []

    def publish(self, sink: TokenSink, token_ids: tuple[int, ...]) -> None:
        if not isinstance(sink, TokenStream):
            sink.publish_many(token_ids)
            return
        wake = sink._stage_many(token_ids)
        if wake is not None:
            loop, event = wake
            self._events.setdefault(loop, []).append(event)

    def publish_one(self, sink: TokenSink, token_id: int) -> None:
        if not isinstance(sink, TokenStream):
            sink.publish_many((token_id,))
            return
        wake = sink._stage_one(token_id)
        if wake is not None:
            loop, event = wake
            self._events.setdefault(loop, []).append(event)

    def defer(self, sink: TokenSink, token_ids: tuple[int, ...]) -> None:
        """Stage a non-terminal publication when the engine flushes this batch."""

        if not token_ids or any(type(token_id) is not int for token_id in token_ids):
            raise TypeError("token stream deferred publish requires integer token ids")
        self._deferred.append((sink, token_ids))

    def defer_one(self, sink: TokenSink, token_id: int) -> None:
        if type(token_id) is not int:
            raise TypeError("token stream deferred publish requires an integer token id")
        self._deferred.append((sink, (token_id,)))

    def flush(self) -> None:
        deferred, self._deferred = self._deferred, []
        for sink, token_ids in deferred:
            self.publish(sink, token_ids)
        events_by_loop, self._events = self._events, {}
        for loop, events in events_by_loop.items():
            with suppress(RuntimeError):
                loop.call_soon_threadsafe(self._set_events, tuple(events))

    @staticmethod
    def _set_events(events: tuple[asyncio.Event, ...]) -> None:
        for event in events:
            event.set()


__all__ = ("TokenSink", "TokenStream", "TokenWakeBatch")
