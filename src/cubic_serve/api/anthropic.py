"""Anthropic messages projections over canonical output events."""

from __future__ import annotations

from collections.abc import Iterable

from cubic_serve.api._events import FoldedBlock, FoldedEvents, fold_events, require_identity
from cubic_serve.contracts.request import TokenEvent, TokenEventKind

WireObject = dict[str, object]


def _usage(output: FoldedEvents) -> WireObject:
    return {"input_tokens": output.input_tokens, "output_tokens": output.output_tokens}


def _stop(output: FoldedEvents) -> tuple[str, str | None]:
    if output.finish_reason == "length":
        return "max_tokens", None
    if output.finish_reason == "tool_calls":
        return "tool_use", None
    if output.stop_sequence is not None:
        return "stop_sequence", output.stop_sequence
    return "end_turn", None


def _content_block(block: FoldedBlock, output: FoldedEvents) -> WireObject:
    if block.kind is TokenEventKind.TEXT_DELTA:
        return {"type": "text", "text": "".join(block.deltas)}
    if block.kind is TokenEventKind.REASONING_DELTA:
        return {"type": "thinking", "thinking": "".join(block.deltas)}
    assert block.tool_index is not None
    call = output.tool_calls[block.tool_index]
    return {
        "type": "tool_use",
        "id": call.call_id,
        "name": call.name,
        "input": dict(call.input_object),
    }


def project_anthropic_message(events: Iterable[TokenEvent], *, model: str) -> WireObject:
    """Fold one complete event sequence into an Anthropic non-stream message."""

    output = fold_events(events)
    stop_reason, stop_sequence = _stop(output)
    return {
        "id": output.request_id,
        "type": "message",
        "role": "assistant",
        "model": require_identity(model, "model"),
        "content": [_content_block(block, output) for block in output.blocks],
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "usage": _usage(output),
    }


def project_anthropic_message_stream(
    events: Iterable[TokenEvent], *, model: str
) -> tuple[WireObject, ...]:
    """Project the same complete events into a valid Anthropic event lifecycle."""

    output = fold_events(events)
    model = require_identity(model, "model")
    chunks: list[WireObject] = [
        {
            "type": "message_start",
            "message": {
                "id": output.request_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": output.input_tokens, "output_tokens": 0},
            },
        }
    ]
    for index, block in enumerate(output.blocks):
        initial: WireObject
        if block.kind is TokenEventKind.TEXT_DELTA:
            initial = {"type": "text", "text": ""}
            deltas = ({"type": "text_delta", "text": delta} for delta in block.deltas)
        elif block.kind is TokenEventKind.REASONING_DELTA:
            initial = {"type": "thinking", "thinking": ""}
            deltas = ({"type": "thinking_delta", "thinking": delta} for delta in block.deltas)
        else:
            assert block.tool_index is not None
            call = output.tool_calls[block.tool_index]
            initial = {
                "type": "tool_use",
                "id": call.call_id,
                "name": call.name,
                "input": {},
            }
            deltas = ({"type": "input_json_delta", "partial_json": delta} for delta in block.deltas)
        chunks.append({"type": "content_block_start", "index": index, "content_block": initial})
        chunks.extend(
            {"type": "content_block_delta", "index": index, "delta": delta} for delta in deltas
        )
        chunks.append({"type": "content_block_stop", "index": index})

    stop_reason, stop_sequence = _stop(output)
    chunks.extend(
        (
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
                "usage": {"output_tokens": output.output_tokens},
            },
            {"type": "message_stop"},
        )
    )
    return tuple(chunks)


__all__ = (
    "project_anthropic_message",
    "project_anthropic_message_stream",
)
