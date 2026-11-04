"""Common OpenAI forced-tool JSON framing, independent of a model's native syntax."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import replace
from typing import Any, cast

from cubic_serve.contracts.request import (
    CanonicalChatRequest,
    StructuredOutputKind,
    StructuredOutputMeta,
    TokenEventKind,
)
from cubic_serve.generation.output import (
    GenerationOutputError,
    IncrementalOutputParser,
    ParsedOutputDelta,
)


def prepare_forced_tool_request(
    request: CanonicalChatRequest,
    *,
    native_parser: str | bool,
) -> CanonicalChatRequest:
    if request.structured_output.kind is not StructuredOutputKind.TOOL_SCHEMA or native_parser:
        return request
    payload = json.loads(request.structured_output.canonical_payload)
    if payload["choice"] == "auto":
        raise ValueError("automatic tool choice requires an explicitly configured tool-call parser")
    response = payload.get("response")
    return replace(
        request,
        structured_output=StructuredOutputMeta.tool_schema(
            payload["tools"],
            choice=payload["choice"],
            parallel_tool_calls=payload.get("parallel_tool_calls", True),
            format="json",
            response=(
                StructuredOutputMeta(
                    StructuredOutputKind(response["kind"]), response["payload"], response["options"]
                )
                if response is not None else None
            ),
        ),
    )


def tool_sampling_constraint(request: CanonicalChatRequest) -> StructuredOutputMeta:
    """Tool prompting/parsing does not imply a strict sampling language.

    vLLM's default AUTO policy leaves non-strict automatic calls unconstrained.
    Preserve explicit output languages, forced calls, strict tools and restricted
    tool sets; keep the original request for prompt rendering and extraction.
    """
    meta = request.structured_output
    if meta.kind is not StructuredOutputKind.TOOL_SCHEMA:
        return meta
    payload = json.loads(meta.canonical_payload)
    if (payload["choice"] == "auto" and payload.get("format") != "json"
            and "response" not in payload
            and payload["tools"] == request.tools.as_list()
            and not any(t["function"].get("strict") is True for t in payload["tools"])):
        return StructuredOutputMeta.identity()
    return meta


def tool_json_schema(payload: dict[str, Any]) -> dict[str, Any]:
    functions = [tool["function"] for tool in payload["tools"]]
    if payload["choice"] == "named":
        return cast(dict[str, Any], functions[0]["parameters"])

    # Preserve each tool's local JSON Pointer root, including recursive $refs,
    # when embedding argument schemas beneath the common call envelope.
    def relocate(value: Any, root: str) -> Any:
        if isinstance(value, dict):
            return {
                k: root + v[1:]
                if k == "$ref" and isinstance(v, str) and v.startswith("#")
                else relocate(v, root)
                for k, v in value.items()
            }
        return [relocate(v, root) for v in value] if isinstance(value, list) else value

    definitions = {
        f"tool{i}": relocate(f["parameters"], f"#/$defs/tool{i}") for i, f in enumerate(functions)
    }
    schema: dict[str, Any] = {
        "type": "array",
        "minItems": 1,
        "$defs": definitions,
        "items": {
            "type": "object",
            "anyOf": [
                {
                    "properties": {
                        "name": {"type": "string", "enum": [f["name"]]},
                        "parameters": {"$ref": f"#/$defs/tool{i}"},
                    },
                    "required": ["name", "parameters"],
                    "additionalProperties": False,
                }
                for i, f in enumerate(functions)
            ],
        },
    }
    if payload.get("parallel_tool_calls") is False:
        schema["maxItems"] = 1
    return schema


def wrap_forced_tool_parser(
    meta: StructuredOutputMeta,
    parser: IncrementalOutputParser | None,
) -> IncrementalOutputParser | None:
    payload = json.loads(meta.canonical_payload)
    if payload.get("format") == "json":
        return ForcedToolOutputParser(payload, parser)
    if parser is not None and payload.get("parallel_tool_calls") is False:
        return _SingleToolOutputParser(parser)
    return parser


class _SingleToolOutputParser:
    """Apply the request's single-call output policy after native extraction.

    vLLM 003c91e tool_calls_utils.py filters later call indices for both
    streaming and complete responses. Sampling strictness is independent.
    """

    def __init__(self, parser: IncrementalOutputParser) -> None:
        self.parser = parser

    @property
    def reasoning_token_count(self) -> int:
        return getattr(self.parser, "reasoning_token_count", 0)

    @staticmethod
    def _filter(deltas: Iterable[ParsedOutputDelta]) -> tuple[ParsedOutputDelta, ...]:
        result = []
        for delta in deltas:
            if delta.kind is TokenEventKind.TOOL_CALL_DELTA:
                index = dict(delta.payload_items)["index"]
                if not index.isascii() or not index.isdecimal() or str(int(index)) != index:
                    raise GenerationOutputError("native tool index is not canonical")
                if index != "0":
                    continue
            result.append(delta)
        return tuple(result)

    def push(self, token_id: int, text: str) -> tuple[ParsedOutputDelta, ...]:
        return self._filter(self.parser.push(token_id, text))

    def finish(self, text: str) -> tuple[ParsedOutputDelta, ...]:
        return self._filter(self.parser.finish(text))

    def truncate(self, text: str) -> tuple[ParsedOutputDelta, ...]:
        return self._filter(getattr(self.parser, "truncate", self.parser.finish)(text))


class ForcedToolOutputParser:
    """Incremental JSON arguments with an optional model-owned reasoning parser.

    Required calls use the same array of {name, parameters} as vLLM. A named
    choice is the argument object alone. Only observed JSON bytes are published;
    length termination never invents closing syntax. Parsing is linear in the
    argument body and retains only the current call for independent validation.
    """

    def __init__(
        self,
        payload: dict[str, Any],
        inner: IncrementalOutputParser | None = None,
    ) -> None:
        self.inner, self.named = inner, payload["choice"] == "named"
        self.names = {tool["function"]["name"] for tool in payload["tools"]}
        self.parallel = payload.get("parallel_tool_calls", True)
        self.buffer, self.phase = "", "array"
        self.index = 0
        self.name: str | None = None
        self.key: str | None = None
        self.fields: set[str] = set()
        self.arguments: list[str] = []
        self.depth, self.quoted, self.escaped = 0, False, False
        self.header, self.closed = False, False
        if self.named:
            self.name = next(iter(self.names))

    def _delta(self, text: str, *, header: bool = False) -> ParsedOutputDelta:
        if header:
            assert self.name is not None
        metadata = (
            (
                ("id", f"call_{self.index}"),
                ("index", str(self.index)),
                ("name", cast(str, self.name)),
            )
            if header
            else (("index", str(self.index)),)
        )
        return ParsedOutputDelta(TokenEventKind.TOOL_CALL_DELTA, text, metadata)

    def _begin(self, emitted: list[ParsedOutputDelta]) -> None:
        if not self.header:
            emitted.append(self._delta("", header=True))
            self.header = True

    def _content(self, text: str) -> list[ParsedOutputDelta]:
        emitted: list[ParsedOutputDelta] = []
        if self.named:
            if text:
                self._begin(emitted)
                self.arguments.append(text)
                emitted.append(self._delta(text))
            return emitted
        self.buffer += text
        decoder = json.JSONDecoder()
        while self.buffer:
            if self.phase == "parameters":
                stop = 0
                for char in self.buffer:
                    stop += 1
                    if self.quoted:
                        if self.escaped:
                            self.escaped = False
                        elif char == "\\":
                            self.escaped = True
                        elif char == '"':
                            self.quoted = False
                    elif char == '"':
                        self.quoted = True
                    elif char in "{[":
                        self.depth += 1
                    elif char in "}]":
                        self.depth -= 1
                        if self.depth == 0:
                            break
                part, self.buffer = self.buffer[:stop], self.buffer[stop:]
                self.arguments.append(part)
                if self.header:
                    emitted.append(self._delta(part))
                if self.depth:
                    break
                if not isinstance(json.loads("".join(self.arguments)), dict):
                    raise GenerationOutputError("tool arguments must be a JSON object")
                self.phase = "field_end"
                continue
            self.buffer = self.buffer.lstrip()
            if not self.buffer:
                break
            first = self.buffer[0]
            if self.phase in ("key", "name"):
                try:
                    value, end = decoder.raw_decode(self.buffer)
                except json.JSONDecodeError:
                    break
                if not isinstance(value, str):
                    raise GenerationOutputError("tool field must be a string")
                if self.phase == "key":
                    if value not in ("name", "parameters") or value in self.fields:
                        raise GenerationOutputError("invalid or duplicate tool field")
                    self.key, self.phase = value, "colon"
                    self.fields.add(value)
                else:
                    if value not in self.names:
                        raise GenerationOutputError("tool call names an undeclared function")
                    self.name, self.phase = value, "field_end"
                    self._begin(emitted)
                    if self.arguments:
                        emitted.append(self._delta("".join(self.arguments)))
                self.buffer = self.buffer[end:]
                continue
            if self.phase == "value":
                if self.key == "name":
                    self.phase = "name"
                    continue
                if first != "{":
                    raise GenerationOutputError("tool parameters must be a JSON object")
                self.phase = "parameters"
                continue
            if self.phase == "array" and first == "[":
                self.phase = "call"
            elif self.phase == "call" and first == "{":
                if self.index and not self.parallel:
                    raise GenerationOutputError("parallel tool calls were disabled")
                self.phase = "key"
            elif self.phase == "colon" and first == ":":
                self.phase = "value"
            elif self.phase == "field_end" and first == ",":
                self.phase = "key"
            elif self.phase == "field_end" and first == "}":
                if self.fields != {"name", "parameters"} or not self.header:
                    raise GenerationOutputError("tool call is missing required fields")
                self.phase = "call_end"
                self.index += 1
                self.name, self.key, self.header = None, None, False
                self.fields, self.arguments = set(), []
            elif self.phase == "call_end" and first == ",":
                self.phase = "call"
            elif self.phase == "call_end" and first == "]":
                self.phase = "done"
            else:
                raise GenerationOutputError("invalid forced tool JSON framing")
            self.buffer = self.buffer[1:]
        return emitted

    def _convert(self, deltas: Iterable[ParsedOutputDelta]) -> tuple[ParsedOutputDelta, ...]:
        emitted: list[ParsedOutputDelta] = []
        for delta in deltas:
            if delta.kind is TokenEventKind.TEXT_DELTA:
                emitted.extend(self._content(delta.text))
            elif delta.kind is TokenEventKind.REASONING_DELTA:
                emitted.append(delta)
            else:
                raise GenerationOutputError("forced JSON parser received native tool syntax")
        return tuple(emitted)

    def push(self, token_id: int, text: str) -> tuple[ParsedOutputDelta, ...]:
        if self.closed:
            raise GenerationOutputError("tool parser is already closed")
        return (
            self._convert(self.inner.push(token_id, text))
            if self.inner is not None
            else tuple(self._content(text))
        )

    def _finish(self, text: str, *, truncated: bool) -> tuple[ParsedOutputDelta, ...]:
        if self.closed:
            raise GenerationOutputError("tool parser finished more than once")
        if self.inner is not None:
            finish = (
                getattr(self.inner, "truncate", self.inner.finish)
                if truncated
                else self.inner.finish
            )
            emitted = self._convert(finish(text))
        else:
            emitted = tuple(self._content(text))
        if not truncated:
            if self.named:
                if not isinstance(json.loads("".join(self.arguments)), dict):
                    raise GenerationOutputError("named tool requires JSON arguments")
            elif self.phase != "done" or self.buffer.strip():
                raise GenerationOutputError("forced tool JSON is incomplete")
        self.closed = True
        return emitted

    def finish(self, text: str) -> tuple[ParsedOutputDelta, ...]:
        return self._finish(text, truncated=False)

    def truncate(self, text: str) -> tuple[ParsedOutputDelta, ...]:
        return self._finish(text, truncated=True)
