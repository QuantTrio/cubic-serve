"""The sole raw engine-token to public ``TokenEvent`` pipeline."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from cubic_serve.contracts.request import (
    CanonicalChatRequest,
    ReasoningControl,
    StructuredOutputKind,
    TokenEvent,
    TokenEventKind,
    ToolDefinitions,
)


def resolve_parser_name(selection: str | bool, names: tuple[str, ...]) -> str | None:
    """Resolve a model-declared parser name; True preserves the legacy default."""
    if selection is False:
        return None
    if selection is True:
        if not names:
            raise ValueError("model does not declare this output parser")
        return names[0]
    if type(selection) is not str or selection not in names:
        raise ValueError("selected output parser is not declared by this model")
    return selection


class GenerationOutputError(ValueError):
    """Raw generation output cannot be normalized without guessing semantics."""


class IncrementalTokenDecoder(Protocol):
    """Tokenizer-owned state for exact incremental token detokenization."""

    def decode(self, token_id: int) -> str: ...

    def finish(self) -> str: ...


@dataclass(frozen=True, slots=True)
class ParsedOutputDelta:
    """Parser result without request identity or sequence ownership."""

    kind: TokenEventKind
    text: str
    payload_items: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, TokenEventKind) or self.kind not in (
            TokenEventKind.TEXT_DELTA,
            TokenEventKind.REASONING_DELTA,
            TokenEventKind.TOOL_CALL_DELTA,
        ):
            raise ValueError("parsed output kind must be text, reasoning, or tool-call delta")
        if type(self.text) is not str or type(self.payload_items) is not tuple:
            raise TypeError("parsed output text and payload tuple have invalid types")
        if any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not str
            for item in self.payload_items
        ):
            raise TypeError("parsed output payload must contain string pairs")
        keys = tuple(key for key, _ in self.payload_items)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("parsed output payload keys must be unique and sorted")
        if self.kind is not TokenEventKind.TOOL_CALL_DELTA and keys:
            raise ValueError("only tool-call deltas may carry parser metadata")
        if self.kind is TokenEventKind.TOOL_CALL_DELTA and keys not in (
            ("index",),
            ("id", "index", "name"),
        ):
            raise ValueError("tool-call delta metadata must be index or id/index/name")


class IncrementalOutputParser(Protocol):
    """Optional stateful reasoning/tool parser over decoder-owned boundaries."""

    def push(self, token_id: int, text: str) -> tuple[ParsedOutputDelta, ...]: ...

    def finish(self, text: str) -> tuple[ParsedOutputDelta, ...]: ...


def bind_request_tools(
    parser: IncrementalOutputParser | None,
    tools: ToolDefinitions,
) -> None:
    """Provide declared argument schemas to model-owned parsers that use them."""
    if parser is not None and tools.canonical_payload != "[]":
        bind = getattr(parser, "bind_tools", None)
        if bind is not None:
            bind(tools)


class ChatOutputCodec(Protocol):
    """Model-format boundary consumed by model-independent chat backends."""

    @property
    def eos_token_id(self) -> int: ...

    def admit_parser_recipe(
        self, *, reasoning_parser: str | bool, tool_call_parser: str | bool
    ) -> None: ...

    def new_decoder(self) -> IncrementalTokenDecoder: ...

    def new_parser(
        self,
        *,
        enable_thinking: bool,
        reasoning_parser: str | bool = False,
        tool_call_parser: str | bool = False,
    ) -> IncrementalOutputParser | None: ...


class JSONAnswerOutputParser:
    """Keep an initial JSON value literal, then resume native tool parsing."""

    def __init__(self, after: IncrementalOutputParser, tool_prefix: str) -> None:
        if not tool_prefix:
            raise ValueError("native tool prefix must be declared")
        self.after, self.tool_prefix = after, tool_prefix
        self.phase = "prefix"
        self.pending: list[tuple[int, str]] = []
        self.json_pending: list[tuple[int, str]] = []
        self.depth, self.quoted, self.escaped = 0, False, False
        self.root_string, self.closed = False, False

    def _json_end(self, text: str) -> int:
        end = len(text)
        for index, char in enumerate(text):
            if self.quoted:
                if self.escaped:
                    self.escaped = False
                elif char == "\\":
                    self.escaped = True
                elif char == '"':
                    self.quoted = False
                    if self.root_string and not self.depth:
                        self.phase, end = "native", index + 1
                        break
            elif char == '"':
                self.quoted = True
            elif char in "{[":
                self.depth += 1
            elif char in "}]":
                self.depth -= 1
                if self.depth == 0:
                    self.phase, end = "native", index + 1
                    break
            elif char == self.tool_prefix[0]:
                # Invalid JSON-looking text can be a native call's preamble.
                self.phase, end = "native", index
                break
        return end

    def push(self, token_id: int, text: str) -> tuple[ParsedOutputDelta, ...]:
        if self.closed:
            raise GenerationOutputError("output parser is already closed")
        if self.phase == "native":
            return self.after.push(token_id, text)
        if self.phase == "prefix":
            self.pending.append((token_id, text))
            first = "".join(value for _, value in self.pending).lstrip()
            if not first:
                return ()
            self.phase = "json" if first[0] in '{["' else "native"
            self.root_string = first[0] == '"'
            pending, self.pending = self.pending, []
            return tuple(delta for token, value in pending for delta in self.push(token, value))
        self.json_pending.append((token_id, text))
        end = self._json_end(text)
        if self.phase != "native":
            return ()
        tail = text[end:]
        complete = "".join(value for _, value in self.json_pending)
        literal = complete[: len(complete) - len(tail)] if tail else complete
        pending, self.json_pending = self.json_pending, []
        try:
            json.loads(literal)
        except json.JSONDecodeError:
            return tuple(
                delta for token, value in pending for delta in self.after.push(token, value)
            )
        return (ParsedOutputDelta(TokenEventKind.TEXT_DELTA, literal),) + (
            self.after.push(token_id, tail) if tail else ()
        )

    def _finish(self, text: str, *, truncated: bool) -> tuple[ParsedOutputDelta, ...]:
        if self.closed:
            raise GenerationOutputError("output parser finished more than once")
        self.closed = True
        finish = (
            getattr(self.after, "truncate", self.after.finish) if truncated else self.after.finish
        )
        if self.phase == "native":
            return finish(text)
        pending = self.pending if self.phase == "prefix" else self.json_pending
        combined = "".join(value for _, value in pending) + text
        first = combined.lstrip()
        try:
            _, end = json.JSONDecoder().raw_decode(first)
        except json.JSONDecodeError:
            if truncated:
                return (ParsedOutputDelta(TokenEventKind.TEXT_DELTA, combined),) if combined else ()
            # A native tool preamble can look like an incomplete JSON value.
            emitted = tuple(
                delta for token, value in pending for delta in self.after.push(token, value)
            )
            return emitted + finish(text)
        boundary = len(combined) - len(first) + end
        return (ParsedOutputDelta(TokenEventKind.TEXT_DELTA, combined[:boundary]),) + finish(
            combined[boundary:]
        )

    def finish(self, text: str) -> tuple[ParsedOutputDelta, ...]:
        return self._finish(text, truncated=False)

    def truncate(self, text: str) -> tuple[ParsedOutputDelta, ...]:
        return self._finish(text, truncated=True)


class TokenPhaseOutputParser:
    """Match declared control tokens; answer bytes have no reasoning semantics."""

    def __init__(
        self,
        after: IncrementalOutputParser | None,
        *,
        reasoning_end: tuple[int, ...] = (),
        reasoning_start: tuple[int, ...] = (),
        reasoning_implicit_end: tuple[tuple[int, ...], ...] = (),
        reasoning_prefix: tuple[int, ...] = (),
    ) -> None:
        self.after, self.end, self.start = after, reasoning_end, reasoning_start
        self.implicit = reasoning_implicit_end
        self.boundaries = (self.end, *self.implicit)
        self.phase = (
            "reasoning_start" if reasoning_start else "reasoning" if reasoning_end else "answer"
        )
        self.pending: list[tuple[int, str]] = [(token, "") for token in reasoning_prefix]
        self._prompt_pending = len(reasoning_prefix)
        if reasoning_prefix:
            self.phase = "reasoning"
        self.closed = False
        self.reasoning_token_count = 0

    @property
    def pending_reasoning_token_count(self) -> int:
        return len(self.pending) - self._prompt_pending if self.phase == "reasoning" else 0

    def push(self, token_id: int, text: str) -> tuple[ParsedOutputDelta, ...]:
        if self.closed:
            raise GenerationOutputError("output parser is already closed")
        if self.phase == "answer":
            return (
                self.after.push(token_id, text)
                if self.after is not None
                else (ParsedOutputDelta(TokenEventKind.TEXT_DELTA, text),)
                if text
                else ()
            )
        self.pending.append((token_id, text))
        ids = tuple(i for i, _ in self.pending)
        if self.phase == "reasoning_start":
            if ids == self.start[: len(ids)]:
                if ids == self.start:
                    self.pending.clear()
                    self.phase = "reasoning"
                return ()
            self.phase = "reasoning"
            pending, self.pending = self.pending, []
            return tuple(delta for i, value in pending for delta in self.push(i, value))
        emitted = []
        while self.pending and not any(
            tuple(i for i, _ in self.pending) == marker[:len(self.pending)]
            for marker in self.boundaries
        ):
            _, value = self.pending.pop(0)
            if self._prompt_pending:
                self._prompt_pending -= 1
            else:
                self.reasoning_token_count += 1
            if value:
                emitted.append(ParsedOutputDelta(TokenEventKind.REASONING_DELTA, value))
        ids = tuple(i for i, _ in self.pending)
        if self.pending and ids in self.boundaries:
            pending, self.pending = self.pending, []
            self._prompt_pending = 0
            self.phase = "answer"
            if ids != self.end:
                for token, value in pending:
                    emitted.extend(self.push(token, value))
        return tuple(emitted)

    def _finish(self, text: str, *, truncated: bool) -> tuple[ParsedOutputDelta, ...]:
        if self.closed:
            raise GenerationOutputError("output parser finished more than once")
        self.closed = True
        if self.phase == "answer" and self.after is not None:
            finish = (
                getattr(self.after, "truncate", self.after.finish)
                if truncated
                else self.after.finish
            )
            return finish(text)
        kind = (
            TokenEventKind.REASONING_DELTA
            if self.phase.startswith("reasoning")
            else TokenEventKind.TEXT_DELTA
        )
        tail = "".join(value for _, value in self.pending) + text
        if self.phase.startswith("reasoning"):
            self.reasoning_token_count += len(self.pending) - self._prompt_pending
        self._prompt_pending = 0
        self.pending.clear()
        return (ParsedOutputDelta(kind, tail),) if tail else ()

    def finish(self, text: str) -> tuple[ParsedOutputDelta, ...]:
        return self._finish(text, truncated=False)

    def truncate(self, text: str) -> tuple[ParsedOutputDelta, ...]:
        return self._finish(text, truncated=True)


def request_output_parser(
    codec: ChatOutputCodec,
    request: CanonicalChatRequest,
    *,
    enable_thinking: bool,
    reasoning_parser: str | bool,
    tool_call_parser: str | bool,
    reasoning_control: ReasoningControl | None = None,
) -> IncrementalOutputParser | None:
    from cubic_serve.generation.structured_sampling import implicit_reasoning_ends

    reasoning_prefix = reasoning_control.initial_tail if reasoning_control is not None else ()

    constrained = request.structured_output.kind is not StructuredOutputKind.IDENTITY
    if constrained:
        tools = request.structured_output.kind is StructuredOutputKind.TOOL_SCHEMA
        parser = (
            codec.new_parser(
                enable_thinking=False, reasoning_parser=False, tool_call_parser=tool_call_parser
            )
            if tools and tool_call_parser
            else None
        )
        bind_request_tools(parser, request.tools)
        if tools:
            from cubic_serve.generation.tool_output import wrap_forced_tool_parser

            payload = json.loads(request.structured_output.canonical_payload)
            parser = wrap_forced_tool_parser(request.structured_output, parser)
            prefix = getattr(codec, "tool_call_prefix", "")
            if (
                "response" in payload
                and payload.get("format", "native") == "native"
                and parser is not None and prefix and prefix[0] not in '{["'
            ):
                parser = JSONAnswerOutputParser(parser, prefix)
        end: tuple[int, ...] = ()
        start: tuple[int, ...] = ()
        if reasoning_parser and enable_thinking:
            end = tuple(getattr(codec, "reasoning_end_token_ids", ()))
            start = tuple(getattr(codec, "reasoning_start_token_ids", ()))
            if not end:
                raise GenerationOutputError("reasoning parser has no declared token boundary")
        return (
            TokenPhaseOutputParser(parser, reasoning_end=end, reasoning_start=start,
                reasoning_implicit_end=implicit_reasoning_ends(request.structured_output, codec),
                reasoning_prefix=reasoning_prefix)
            if end
            else parser
        )
    # Identity includes both absent tools and explicit tool_choice=none. A
    # configured parser is not permission to interpret ordinary answer bytes.
    if not (reasoning_parser and enable_thinking):
        return None
    end = tuple(getattr(codec, "reasoning_end_token_ids", ()))
    if end:
        return TokenPhaseOutputParser(
            None,
            reasoning_end=end,
            reasoning_start=tuple(getattr(codec, "reasoning_start_token_ids", ())),
            reasoning_implicit_end=implicit_reasoning_ends(request.structured_output, codec),
            reasoning_prefix=reasoning_prefix,
        )
    # Preserve the existing model codec contract for non-token-delimited formats.
    return codec.new_parser(
        enable_thinking=True, reasoning_parser=reasoning_parser, tool_call_parser=False
    )


def argument_types(schema: object) -> set[str]:
    if not isinstance(schema, dict):
        return {"string"}
    value = schema.get("type", ())
    types = {value} if isinstance(value, str) else set(value)
    for key in ("anyOf", "oneOf", "allOf"):
        for alternative in schema.get(key, ()):
            types.update(argument_types(alternative))
    for value in schema.get("enum", ()):
        types.add(
            "null"
            if value is None
            else {
                bool: "boolean",
                int: "integer",
                float: "number",
                str: "string",
                dict: "object",
                list: "array",
            }.get(type(value), "string")
        )
    return types or {"string"}


def typed_tool_argument(value: object, schema: object) -> object:
    if not isinstance(schema, dict):
        return value
    if isinstance(value, str):
        raw = value
        types = argument_types(schema)
        for kind in ("null", "integer", "number", "boolean", "object", "array", "string"):
            if kind not in types:
                continue
            try:
                if kind == "null" and raw.lower() == "null":
                    return None
                if kind == "integer":
                    return int(raw)
                if kind == "number":
                    number = float(raw)
                    if math.isfinite(number):
                        return int(number) if number.is_integer() else number
                if kind == "boolean" and raw.strip().lower() in ("true", "false", "1", "0"):
                    return raw.strip().lower() in ("true", "1")
                if kind in ("object", "array"):
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict if kind == "object" else list):
                        value = parsed
                        break
                if kind == "string":
                    return value
            except (ValueError, TypeError):
                continue
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        return {key: typed_tool_argument(item, properties.get(key)) for key, item in value.items()}
    if isinstance(value, list):
        return [typed_tool_argument(item, schema.get("items")) for item in value]
    return value


class GenerationOutputPipeline:
    """Convert one request's raw engine TOKEN/FINISH events exactly once."""

    def __init__(
        self,
        *,
        request_id: str,
        input_token_count: int,
        max_new_tokens: int,
        eos_token_ids: tuple[int, ...],
        decoder: IncrementalTokenDecoder,
        parser: IncrementalOutputParser | None = None,
        continuous_usage: bool = False,
        report_reasoning_tokens: bool = False,
        include_reasoning: bool = True,
    ) -> None:
        if type(request_id) is not str or not request_id or request_id != request_id.strip():
            raise ValueError("request_id must be a non-empty canonical string")
        if type(input_token_count) is not int or input_token_count < 0:
            raise ValueError("input_token_count must be a non-negative integer")
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if type(eos_token_ids) is not tuple or tuple(sorted(set(eos_token_ids))) != eos_token_ids:
            raise ValueError("EOS token ids must be a sorted unique tuple")
        if any(type(token_id) is not int or token_id < 0 for token_id in eos_token_ids):
            raise ValueError("EOS token ids must be non-negative integers")
        self._request_id = request_id
        self._input_tokens = input_token_count
        self._max_new_tokens = max_new_tokens
        self._eos_token_ids = frozenset(eos_token_ids)
        self._decoder = decoder
        self._parser = parser
        self._continuous_usage = continuous_usage
        if type(report_reasoning_tokens) is not bool or type(include_reasoning) is not bool:
            raise ValueError("reasoning output controls must be boolean")
        self._report_reasoning_tokens = report_reasoning_tokens
        self._include_reasoning = include_reasoning
        self._raw_sequence = 0
        self._output_sequence = 0
        self._output_tokens = 0
        self._last_token_id: int | None = None
        self._has_tool_delta = False
        self._finished = False
        self._failed = False

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def output_token_count(self) -> int:
        return self._output_tokens

    def consume(self, events: Iterable[TokenEvent]) -> tuple[TokenEvent, ...]:
        emitted: list[TokenEvent] = []
        for event in events:
            emitted.extend(self.push(event))
        return tuple(emitted)

    def push(self, event: TokenEvent) -> tuple[TokenEvent, ...]:
        if self._failed:
            raise GenerationOutputError("generation output pipeline is failed")
        if self._finished:
            raise GenerationOutputError("raw event arrived after terminal FINISH")
        try:
            return self._push(event)
        except Exception:
            self._failed = True
            raise

    def _push(self, event: TokenEvent) -> tuple[TokenEvent, ...]:
        if not isinstance(event, TokenEvent):
            raise GenerationOutputError("raw output contains a non-TokenEvent value")
        if event.request_id != self._request_id:
            raise GenerationOutputError("raw output mixed request identities")
        if type(event.sequence_no) is not int or event.sequence_no != self._raw_sequence:
            raise GenerationOutputError("raw event sequence must be contiguous from zero")
        if not isinstance(event.kind, TokenEventKind):
            raise GenerationOutputError("raw event kind must be a TokenEventKind")
        if event.kind is TokenEventKind.TOKEN:
            emitted = self._push_token(event)
        elif event.kind is TokenEventKind.FINISH:
            emitted = self._push_finish(event)
        else:
            raise GenerationOutputError("engine raw output may contain only TOKEN and FINISH")
        self._raw_sequence += 1
        return emitted

    def _push_token(self, event: TokenEvent) -> tuple[TokenEvent, ...]:
        if (
            type(event.token_id) is not int
            or event.token_id < 0
            or event.text is not None
            or event.payload_items
        ):
            raise GenerationOutputError("raw TOKEN has an invalid shape")
        if (
            self._last_token_id in self._eos_token_ids
            or self._output_tokens >= self._max_new_tokens
        ):
            raise GenerationOutputError("raw TOKEN arrived after an EOS or output-limit token")
        text = "" if event.token_id in self._eos_token_ids else self._decoder.decode(event.token_id)
        if type(text) is not str:
            raise GenerationOutputError("incremental decoder must return a string")
        deltas = (
            self._parser.push(event.token_id, text)
            if self._parser is not None
            else ((ParsedOutputDelta(TokenEventKind.TEXT_DELTA, text),) if text else ())
        )
        self._output_tokens += 1
        emitted = self._emit_deltas(deltas)
        if self._continuous_usage and not emitted:
            emitted = (self._emit(TokenEventKind.TEXT_DELTA, text=""),)
        self._last_token_id = event.token_id
        return emitted

    def _push_finish(self, event: TokenEvent) -> tuple[TokenEvent, ...]:
        if event.token_id is not None or event.text is not None or event.payload_items:
            raise GenerationOutputError("raw FINISH has an invalid shape")
        if self._last_token_id in self._eos_token_ids:
            reason = "stop"
        elif self._output_tokens == self._max_new_tokens:
            reason = "length"
        else:
            raise GenerationOutputError("raw FINISH is neither EOS nor max_new_tokens")

        tail = self._decoder.finish()
        if type(tail) is not str:
            raise GenerationOutputError("incremental decoder finish must return a string")
        finish_parser = (
            getattr(self._parser, "truncate", self._parser.finish)
            if self._parser is not None and reason == "length"
            else self._parser.finish
            if self._parser is not None
            else None
        )
        deltas = (
            finish_parser(tail)
            if finish_parser is not None
            else ((ParsedOutputDelta(TokenEventKind.TEXT_DELTA, tail),) if tail else ())
        )
        emitted = list(self._emit_deltas(deltas))
        if self._has_tool_delta and reason == "stop":
            reason = "tool_calls"
        usage = (
            ("input_tokens", str(self._input_tokens)),
            ("output_tokens", str(self._output_tokens)),
        )
        reasoning_tokens = self._reasoning_count()
        if reasoning_tokens is not None:
            usage += (("reasoning_tokens", str(reasoning_tokens)),)
        emitted.extend(
            (
                self._emit(
                    TokenEventKind.USAGE,
                    payload_items=usage,
                ),
                self._emit(
                    TokenEventKind.FINISH,
                    payload_items=(("reason", reason),),
                ),
            )
        )
        self._finished = True
        return tuple(emitted)

    def _emit_deltas(self, deltas: object) -> tuple[TokenEvent, ...]:
        if type(deltas) is not tuple or any(
            not isinstance(delta, ParsedOutputDelta) for delta in deltas
        ):
            raise GenerationOutputError("incremental parser must return ParsedOutputDelta tuples")
        emitted: list[TokenEvent] = []
        for delta in deltas:
            if delta.kind is TokenEventKind.REASONING_DELTA and not self._include_reasoning:
                continue
            if delta.kind is TokenEventKind.TOOL_CALL_DELTA:
                self._has_tool_delta = True
            emitted.append(
                self._emit(
                    delta.kind,
                    text=delta.text,
                    payload_items=delta.payload_items,
                )
            )
        return tuple(emitted)

    def _reasoning_count(self) -> int | None:
        if not self._report_reasoning_tokens:
            return None
        count = getattr(self._parser, "reasoning_token_count", 0)
        if type(count) is not int or not 0 <= count <= self._output_tokens:
            raise GenerationOutputError("parser reasoning token count exceeds committed output")
        return count

    def _emit(
        self,
        kind: TokenEventKind,
        *,
        text: str | None = None,
        payload_items: tuple[tuple[str, str], ...] = (),
    ) -> TokenEvent:
        event = TokenEvent(
            request_id=self._request_id,
            sequence_no=self._output_sequence,
            kind=kind,
            output_token_count=self._output_tokens if self._continuous_usage else None,
            reasoning_token_count=self._reasoning_count() if self._continuous_usage else None,
            text=text,
            payload_items=payload_items,
        )
        self._output_sequence += 1
        return event


__all__ = (
    "GenerationOutputError",
    "GenerationOutputPipeline",
    "IncrementalOutputParser",
    "IncrementalTokenDecoder",
    "ParsedOutputDelta",
)
