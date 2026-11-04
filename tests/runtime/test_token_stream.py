from __future__ import annotations

from typing import Any

from cubic_serve.runtime.token_stream import TokenStream, TokenWakeBatch


class _Loop:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, tuple[Any, ...]]] = []

    def call_soon_threadsafe(self, callback: Any, *args: Any) -> None:
        self.calls.append((callback, args))


class _Event:
    def __init__(self) -> None:
        self.set_count = 0

    def set(self) -> None:
        self.set_count += 1


def test_grouped_publish_wakes_one_loop_once_without_merging_request_queues() -> None:
    loop = _Loop()
    first, second = TokenStream(), TokenStream()
    first_event, second_event = _Event(), _Event()
    first._async_loop, first._async_event = loop, first_event  # type: ignore[assignment]
    second._async_loop, second._async_event = loop, second_event  # type: ignore[assignment]

    batch = TokenWakeBatch()
    batch.publish(first, (11, 12))
    batch.publish(second, (21,))
    batch.flush()

    assert len(loop.calls) == 1
    callback, args = loop.calls[0]
    callback(*args)
    assert (first_event.set_count, second_event.set_count) == (1, 1)
    assert (next(first), next(first), next(second)) == (11, 12, 21)


def test_grouped_publish_flush_is_single_use() -> None:
    loop = _Loop()
    stream = TokenStream()
    event = _Event()
    stream._async_loop, stream._async_event = loop, event  # type: ignore[assignment]
    batch = TokenWakeBatch()

    batch.publish(stream, (7,))
    batch.flush()
    batch.flush()

    assert len(loop.calls) == 1


def test_deferred_publish_does_not_touch_request_mailbox_before_flush() -> None:
    loop = _Loop()
    stream = TokenStream()
    event = _Event()
    stream._async_loop, stream._async_event = loop, event  # type: ignore[assignment]
    batch = TokenWakeBatch()

    batch.defer(stream, (7, 8))

    assert stream._items.empty()
    assert not loop.calls
    batch.flush()
    assert (next(stream), next(stream)) == (7, 8)
    assert len(loop.calls) == 1
