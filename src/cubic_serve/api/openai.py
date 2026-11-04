"""OpenAI chat-completions projections over canonical output events."""

from __future__ import annotations

from collections.abc import Iterable

from cubic_serve.api._events import (
    FoldedEvents,
    ProjectionError,
    fold_events,
    require_identity,
)
from cubic_serve.contracts.request import TokenEvent, TokenEventKind

WireObject = dict[str, object]


def project_openai_usage(
    input_tokens: int, output_tokens: int, reasoning_tokens: int | None = None,
) -> WireObject:
    usage: WireObject = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if reasoning_tokens is not None:
        if type(reasoning_tokens) is not int or not 0 <= reasoning_tokens <= output_tokens:
            raise ProjectionError("reasoning_tokens exceeds completion tokens")
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return usage


def _usage(output: FoldedEvents) -> WireObject:
    return project_openai_usage(output.input_tokens, output.output_tokens, output.reasoning_tokens)


def _base(output: FoldedEvents, model: str, created: int, object_type: str) -> WireObject:
    return {
        "id": output.request_id,
        "object": object_type,
        "created": created,
        "model": model,
    }


def _context(events: Iterable[TokenEvent], model: str, created: int) -> tuple[FoldedEvents, str]:
    output = fold_events(events)
    canonical_model = require_identity(model, "model")
    if type(created) is not int or created < 0:
        raise ProjectionError("created must be a non-negative integer")
    return output, canonical_model


def project_openai_chat_completion(
    events: Iterable[TokenEvent], *, model: str, created: int
) -> WireObject:
    """Fold one complete event sequence into an OpenAI non-stream response."""

    output, model = _context(events, model, created)
    message: WireObject = {
        "role": "assistant",
        "content": "".join(output.text_deltas) if output.text_deltas else None,
    }
    if output.reasoning_deltas:
        message["reasoning_content"] = "".join(output.reasoning_deltas)
    if output.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in output.tool_calls
        ]
    choice: WireObject = {
        "index": 0,
        "message": message,
        "finish_reason": output.finish_reason,
    }
    if output.stop_sequence is not None:
        choice["stop_reason"] = output.stop_sequence
    response = _base(output, model, created, "chat.completion")
    response["choices"] = [choice]
    response["usage"] = _usage(output)
    return response


def project_openai_chat_completion_stream(
    events: Iterable[TokenEvent],
    *,
    model: str,
    created: int,
    include_usage: bool = False,
) -> tuple[WireObject, ...]:
    """Project the same complete events into ordered OpenAI stream chunks."""

    output, model = _context(events, model, created)
    if type(include_usage) is not bool:
        raise ProjectionError("include_usage must be a boolean")

    def chunk(choice: WireObject | None, usage: WireObject | None = None) -> WireObject:
        result = _base(output, model, created, "chat.completion.chunk")
        result["choices"] = [] if choice is None else [choice]
        if usage is not None:
            result["usage"] = usage
        return result

    chunks = [
        chunk(
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }
        )
    ]
    for event in output.deltas:
        delta: WireObject
        if event.kind is TokenEventKind.TEXT_DELTA:
            delta = {"content": event.text}
        elif event.kind is TokenEventKind.REASONING_DELTA:
            delta = {"reasoning_content": event.text}
        else:
            payload = dict(event.payload_items)
            index = int(payload["index"])
            function: WireObject = {"arguments": event.text}
            tool_delta: WireObject = {"index": index, "function": function}
            if "id" in payload:
                tool_delta.update({"id": payload["id"], "type": "function"})
                function["name"] = payload["name"]
            delta = {"tool_calls": [tool_delta]}
        chunks.append(chunk({"index": 0, "delta": delta, "finish_reason": None}))

    terminal_choice: WireObject = {
        "index": 0,
        "delta": {},
        "finish_reason": output.finish_reason,
    }
    if output.stop_sequence is not None:
        terminal_choice["stop_reason"] = output.stop_sequence
    chunks.append(chunk(terminal_choice))
    if include_usage:
        chunks.append(chunk(None, _usage(output)))
    return tuple(chunks)


__all__ = (
    "project_openai_chat_completion",
    "project_openai_chat_completion_stream",
    "project_openai_usage",
)
