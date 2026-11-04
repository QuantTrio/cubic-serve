"""Shared validation and folding for public ``TokenEvent`` projections."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from cubic_serve.contracts.request import TokenEvent, TokenEventKind


class ProjectionError(ValueError):
    """A canonical event sequence cannot be projected without losing semantics."""


@dataclass(frozen=True, slots=True)
class FoldedToolCall:
    index: int
    call_id: str
    name: str
    argument_deltas: tuple[str, ...]
    input_object: dict[str, object]

    @property
    def arguments(self) -> str:
        return "".join(self.argument_deltas)


@dataclass(frozen=True, slots=True)
class FoldedBlock:
    kind: TokenEventKind
    deltas: tuple[str, ...]
    tool_index: int | None = None


@dataclass(frozen=True, slots=True)
class FoldedEvents:
    request_id: str
    deltas: tuple[TokenEvent, ...]
    blocks: tuple[FoldedBlock, ...]
    text_deltas: tuple[str, ...]
    reasoning_deltas: tuple[str, ...]
    tool_calls: tuple[FoldedToolCall, ...]
    input_tokens: int
    output_tokens: int
    finish_reason: str
    stop_sequence: str | None
    reasoning_tokens: int | None = None


def require_identity(value: str, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ProjectionError(f"{name} must be a non-empty canonical string")
    return value


def _payload(event: TokenEvent) -> dict[str, str]:
    if type(event.payload_items) is not tuple:
        raise ProjectionError("event payload_items must be a tuple")
    result: dict[str, str] = {}
    for item in event.payload_items:
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not str
        ):
            raise ProjectionError("event payload_items must contain string pairs")
        result[item[0]] = item[1]
    if tuple(result) != tuple(sorted(result)) or len(result) != len(event.payload_items):
        raise ProjectionError("event payload keys must be unique and sorted")
    return result


def _natural(value: str, path: str) -> int:
    if not value.isascii() or not value.isdecimal():
        raise ProjectionError(f"{path} must be a canonical non-negative integer")
    result = int(value)
    if str(result) != value:
        raise ProjectionError(f"{path} must be a canonical non-negative integer")
    return result


def _terminal(events: tuple[TokenEvent, ...]) -> tuple[int, int, str, str | None, int | None]:
    if len(events) < 2:
        raise ProjectionError("event sequence requires USAGE followed by FINISH")
    usage_event, finish_event = events[-2:]
    if (
        usage_event.kind is not TokenEventKind.USAGE
        or finish_event.kind is not TokenEventKind.FINISH
    ):
        raise ProjectionError("event sequence must end with exactly USAGE then FINISH")
    if usage_event.text is not None or finish_event.text is not None:
        raise ProjectionError("USAGE and FINISH events cannot carry text")

    usage = _payload(usage_event)
    if set(usage) not in (
        {"input_tokens", "output_tokens"},
        {"input_tokens", "output_tokens", "reasoning_tokens"},
    ):
        raise ProjectionError("USAGE requires token totals and optional reasoning_tokens")
    input_tokens = _natural(usage["input_tokens"], "USAGE.input_tokens")
    output_tokens = _natural(usage["output_tokens"], "USAGE.output_tokens")
    reasoning_tokens = (
        _natural(usage["reasoning_tokens"], "USAGE.reasoning_tokens")
        if "reasoning_tokens" in usage else None
    )
    if reasoning_tokens is not None and reasoning_tokens > output_tokens:
        raise ProjectionError("reasoning_tokens exceeds completion tokens")

    finish = _payload(finish_event)
    if set(finish) not in ({"reason"}, {"reason", "stop_sequence"}):
        raise ProjectionError("FINISH requires reason and optional stop_sequence")
    reason = finish["reason"]
    if reason not in ("length", "stop", "tool_calls"):
        raise ProjectionError(f"FINISH reason {reason!r} is unsupported")
    stop_sequence = finish.get("stop_sequence")
    if stop_sequence is not None and (reason != "stop" or not stop_sequence):
        raise ProjectionError("stop_sequence requires a non-empty stop FINISH reason")
    return input_tokens, output_tokens, reason, stop_sequence, reasoning_tokens


def _completed_argument_fields(source: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    source = source.lstrip()
    if not source.startswith("{"):
        return {}
    result: dict[str, object] = {}
    source = source[1:].lstrip()
    while source:
        try:
            key, end = decoder.raw_decode(source)
            source = source[end:].lstrip()
            if not isinstance(key, str) or not source.startswith(":"):
                break
            source = source[1:].lstrip()
            value, end = decoder.raw_decode(source)
            source = source[end:].lstrip()
            if source and source[0] not in ",}":
                break
            result[key] = value
            if not source.startswith(","):
                break
            source = source[1:].lstrip()
        except json.JSONDecodeError:
            break
    return result


def fold_events(events: Iterable[TokenEvent]) -> FoldedEvents:
    """Validate one complete choice and fold it once for both wire protocols."""

    sequence = tuple(events)
    if not sequence:
        raise ProjectionError("event sequence cannot be empty")
    first = sequence[0]
    if not isinstance(first, TokenEvent):
        raise ProjectionError("event sequence contains a non-TokenEvent value")
    request_id = require_identity(first.request_id, "event request_id")
    for sequence_no, event in enumerate(sequence):
        if not isinstance(event, TokenEvent):
            raise ProjectionError("event sequence contains a non-TokenEvent value")
        if event.request_id != request_id or type(event.sequence_no) is not int:
            raise ProjectionError("events must share one request_id and integer sequence numbers")
        if event.sequence_no != sequence_no:
            raise ProjectionError("event sequence numbers must be globally contiguous from zero")
        if not isinstance(event.kind, TokenEventKind):
            raise ProjectionError("event kind must be a TokenEventKind")
        if event.token_id is not None:
            raise ProjectionError("public projection cannot consume raw token_id fields")

    input_tokens, output_tokens, reason, stop_sequence, reasoning_tokens = _terminal(sequence)
    deltas = sequence[:-2]
    if any(event.kind in (TokenEventKind.USAGE, TokenEventKind.FINISH) for event in deltas):
        raise ProjectionError("USAGE and FINISH may appear only once at the terminal boundary")

    text_deltas: list[str] = []
    reasoning_deltas: list[str] = []
    tool_ids: list[str] = []
    tool_names: list[str] = []
    tool_deltas: list[list[str]] = []
    block_keys: list[tuple[TokenEventKind, int | None]] = []
    block_deltas: list[list[str]] = []

    for event in deltas:
        if event.kind is TokenEventKind.TOKEN:
            raise ProjectionError("TOKEN events require decoding before public projection")
        if event.kind not in (
            TokenEventKind.TEXT_DELTA,
            TokenEventKind.REASONING_DELTA,
            TokenEventKind.TOOL_CALL_DELTA,
        ):
            raise ProjectionError(
                f"event kind {event.kind!r} is unsupported before terminal events"
            )
        if type(event.text) is not str:
            raise ProjectionError(f"{event.kind} requires a string text delta")
        text = event.text
        payload = _payload(event)
        tool_index: int | None = None

        if event.kind is TokenEventKind.TEXT_DELTA:
            if payload:
                raise ProjectionError("TEXT_DELTA cannot carry payload_items")
            text_deltas.append(text)
        elif event.kind is TokenEventKind.REASONING_DELTA:
            if payload:
                raise ProjectionError("REASONING_DELTA cannot carry payload_items")
            reasoning_deltas.append(text)
        else:
            if "index" not in payload:
                raise ProjectionError("TOOL_CALL_DELTA requires an index")
            tool_index = _natural(payload["index"], "TOOL_CALL_DELTA.index")
            if tool_index == len(tool_ids):
                if set(payload) != {"id", "index", "name"}:
                    raise ProjectionError("a tool call's first delta requires id, index, and name")
                call_id = require_identity(payload["id"], "TOOL_CALL_DELTA.id")
                name = require_identity(payload["name"], "TOOL_CALL_DELTA.name")
                if call_id in tool_ids:
                    raise ProjectionError("tool call ids must be unique")
                tool_ids.append(call_id)
                tool_names.append(name)
                tool_deltas.append([])
            elif tool_index >= len(tool_ids) or set(payload) != {"index"}:
                raise ProjectionError("a continued tool delta requires only an existing index")
            tool_deltas[tool_index].append(text)

        # Continuous token-usage heartbeats carry no content. They must not
        # split one streamed tool invocation into multiple semantic blocks.
        if (
            not text
            and event.kind is not TokenEventKind.TOOL_CALL_DELTA
            and block_keys
            and block_keys[-1] != (event.kind, None)
        ):
            continue
        block_key = (event.kind, tool_index)
        if not block_keys or block_keys[-1] != block_key:
            block_keys.append(block_key)
            block_deltas.append([])
        block_deltas[-1].append(text)

    tool_block_indices = tuple(
        index for kind, index in block_keys if kind is TokenEventKind.TOOL_CALL_DELTA
    )
    if tool_block_indices != tuple(range(len(tool_ids))):
        raise ProjectionError("tool calls must form one contiguous block each in index order")

    tool_calls: list[FoldedToolCall] = []
    for index, (call_id, name, argument_parts) in enumerate(
        zip(tool_ids, tool_names, tool_deltas, strict=True)
    ):
        arguments = "".join(argument_parts)
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as error:
            if reason != "length":
                raise ProjectionError(
                    f"tool call {index} arguments are not complete JSON"
                ) from error
            # OpenAI exposes the exact observed argument prefix on a length
            # stop. Anthropic's object view retains only completed fields;
            # no generated closing bytes are added to the original arguments.
            parsed = _completed_argument_fields(arguments)
        if type(parsed) is not dict:
            raise ProjectionError(f"tool call {index} arguments must be a JSON object")
        tool_calls.append(
            FoldedToolCall(
                index=index,
                call_id=call_id,
                name=name,
                argument_deltas=tuple(argument_parts),
                input_object=cast(dict[str, object], parsed),
            )
        )

    if (
        (reason == "tool_calls"
        and not tool_calls)
        or (tool_calls
        and reason not in ("tool_calls", "length"))
    ):
        raise ProjectionError("tool calls and FINISH reason tool_calls must appear together")
    blocks = tuple(
        FoldedBlock(kind=key[0], tool_index=key[1], deltas=tuple(parts))
        for key, parts in zip(block_keys, block_deltas, strict=True)
    )
    return FoldedEvents(
        request_id=request_id,
        deltas=deltas,
        blocks=blocks,
        text_deltas=tuple(text_deltas),
        reasoning_deltas=tuple(reasoning_deltas),
        tool_calls=tuple(tool_calls),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        finish_reason=reason,
        stop_sequence=stop_sequence,
        reasoning_tokens=reasoning_tokens,
    )


__all__ = (
    "FoldedBlock",
    "FoldedEvents",
    "FoldedToolCall",
    "ProjectionError",
    "fold_events",
    "require_identity",
)
