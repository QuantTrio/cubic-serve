"""Canonical multimodal request plans and generated output events."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from cubic_serve._compat import StrEnum
from typing import NoReturn, cast
from urllib.parse import urlsplit

from cubic_serve.contracts.model import WeightVariantMeta


class ChatRole(StrEnum):
    """Wire-neutral roles supported by the initial chat normalization slice."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    DEVELOPER = "developer"


class StructuredOutputKind(StrEnum):
    """Backend-neutral structured-output semantics admitted from either wire API."""

    IDENTITY = "identity"
    JSON_SCHEMA = "json_schema"
    TOOL_SCHEMA = "tool_schema"
    GRAMMAR = "grammar"
    REGEX = "regex"
    CHOICE = "choice"
    STRUCTURAL_TAG = "structural_tag"


def _json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"structured output JSON has duplicate key {key!r}")
        result[key] = value
    return result


def _json_constant(value: str) -> NoReturn:
    raise ValueError(f"structured output contains non-finite JSON constant {value!r}")


def _json_value(value: object, path: str = "$") -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite JSON number")
        return value
    if type(value) in (list, tuple):
        items = cast(list[object] | tuple[object, ...], value)
        return [_json_value(item, f"{path}[{index}]") for index, item in enumerate(items)]
    if type(value) is dict:
        result: dict[str, object] = {}
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                raise ValueError(f"{path} contains a non-string JSON key")
            result[key] = _json_value(item, f"{path}.{key}")
        return result
    raise ValueError(f"{path} contains non-JSON type {type(value).__name__}")


def _canonical_json(value: object) -> str:
    normalized = _json_value(value)
    return json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _decode_canonical_json(source: str, *, label: str = "structured output payload") -> object:
    try:
        value = cast(
            object,
            json.loads(source, object_pairs_hook=_json_pairs, parse_constant=_json_constant),
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} must be valid JSON") from error
    value = _json_value(value)
    if _canonical_json(value) != source:
        raise ValueError(f"{label} must use canonical JSON encoding")
    return value


def _canonical_json_input(value: object) -> str:
    if type(value) is not str:
        return _canonical_json(value)
    try:
        decoded = cast(
            object,
            json.loads(value, object_pairs_hook=_json_pairs, parse_constant=_json_constant),
        )
    except json.JSONDecodeError as error:
        raise ValueError("structured output payload must be valid JSON") from error
    return _canonical_json(decoded)


def _strict_json_value(value: object, path: str = "$") -> object:
    """Copy one decoded JSON value while rejecting Python-only containers."""

    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite JSON number")
        return value
    if type(value) is list:
        return [
            _strict_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(cast(list[object], value))
        ]
    if type(value) is dict:
        result: dict[str, object] = {}
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                raise ValueError(f"{path} contains a non-string JSON key")
            result[key] = _strict_json_value(item, f"{path}.{key}")
        return result
    raise ValueError(f"{path} contains non-JSON type {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class ChatTemplateKwargs:
    """Deeply immutable, model-neutral chat-template keyword arguments."""

    canonical_payload: str = "{}"

    def __post_init__(self) -> None:
        if type(self.canonical_payload) is not str:
            raise TypeError("chat template kwargs payload must be a string")
        payload = _decode_canonical_json(
            self.canonical_payload, label="chat template kwargs payload"
        )
        if type(payload) is not dict:
            raise ValueError("chat template kwargs payload must be a JSON object")

    @classmethod
    def from_value(cls, value: object) -> ChatTemplateKwargs:
        payload = _strict_json_value(value)
        if type(payload) is not dict:
            raise ValueError("chat template kwargs must be a JSON object")
        return cls(_canonical_json(payload))

    def as_dict(self) -> dict[str, object]:
        """Return a fresh mutable copy for one template-render invocation."""

        return cast(
            dict[str, object],
            _decode_canonical_json(self.canonical_payload, label="chat template kwargs payload"),
        )


@dataclass(frozen=True, slots=True)
class StructuredOutputMeta:
    """One canonical payload; no compiler, matcher, or wire-only object leaks here."""

    kind: StructuredOutputKind
    canonical_payload: str
    canonical_options: str = "{}"
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.kind, StructuredOutputKind)
            or type(self.canonical_payload) is not str
        ):
            raise TypeError("structured output kind and payload have invalid types")
        payload = _decode_canonical_json(self.canonical_payload)
        options = _decode_canonical_json(self.canonical_options, label="structured output options")
        if type(options) is not dict or set(options) - {
            "disable_any_whitespace",
            "disable_additional_properties",
            "whitespace_pattern",
        }:
            raise ValueError("structured output options are invalid")
        for key, value in options.items():
            if key == "whitespace_pattern":
                if type(value) is not str or "\x00" in value:
                    raise ValueError("structured whitespace pattern must be a string without NUL")
            elif type(value) is not bool:
                raise ValueError("structured output flags must be boolean")
        if options and self.kind is not StructuredOutputKind.JSON_SCHEMA:
            raise ValueError("JSON formatting options require a JSON Schema constraint")
        if self.kind is StructuredOutputKind.IDENTITY:
            if payload is not None:
                raise ValueError("identity structured output payload must be null")
        elif self.kind is StructuredOutputKind.JSON_SCHEMA:
            if type(payload) not in (dict, bool):
                raise ValueError("structured JSON Schema must be an object or boolean")
        elif self.kind is StructuredOutputKind.TOOL_SCHEMA:
            if type(payload) is not dict:
                raise ValueError("tool payload must be an object")
            fields = cast(dict[str, object], payload)
            if (
                not {"choice", "tools"} <= set(fields)
                or set(fields) - {"choice", "tools", "parallel_tool_calls", "format", "response"}
                or fields["choice"]
                not in (
                    "auto",
                    "required",
                    "named",
                )
            ):
                raise ValueError("tool payload requires choice auto/required/named and tools")
            if type(fields["tools"]) is not list or not fields["tools"]:
                raise ValueError("tool payload requires a non-empty tool array")
            if fields.get("format", "native") not in ("native", "json"):
                raise ValueError("tool output format must be native or json")
            if fields["choice"] == "named" and len(fields["tools"]) != 1:
                raise ValueError("named tool choice requires exactly one selected tool")
            if "response" in fields:
                response = fields["response"]
                if type(response) is not dict or set(response) != {"kind", "payload", "options"}:
                    raise ValueError("tool response constraint has an invalid shape")
                response = cast(dict[str, str], response)
                nested = StructuredOutputMeta(
                    StructuredOutputKind(response["kind"]), response["payload"], response["options"]
                )
                if nested.kind in (
                    StructuredOutputKind.IDENTITY,
                    StructuredOutputKind.TOOL_SCHEMA,
                ) or (
                    fields["choice"] != "auto"
                    and nested.kind is not StructuredOutputKind.STRUCTURAL_TAG
                ):
                    raise ValueError(
                        "forced tool choice can retain only an explicit structural tag"
                    )
            if type(fields.get("parallel_tool_calls", True)) is not bool:
                raise ValueError("parallel_tool_calls must be boolean")
        elif self.kind is StructuredOutputKind.REGEX:
            if type(payload) is not str or "\x00" in payload:
                raise ValueError("structured regex must be a string without NUL")
        elif self.kind is StructuredOutputKind.CHOICE:
            if type(payload) is not list or not payload or any(type(x) is not str for x in payload):
                raise ValueError("structured choice requires a non-empty array of strings")
        elif self.kind is StructuredOutputKind.STRUCTURAL_TAG:
            if type(payload) is not dict:
                raise ValueError("structured tag must be an object")
        elif self.kind is StructuredOutputKind.GRAMMAR:
            if type(payload) is not dict:
                raise ValueError("grammar structured output payload must be an object")
            fields = cast(dict[str, object], payload)
            if set(fields) != {"dialect", "root_rule", "source"}:
                raise ValueError("grammar payload requires dialect, root_rule, and source")
            identifiers = (fields["dialect"], fields["root_rule"])
            if (
                any(
                    type(item) is not str
                    or not item
                    or any(
                        not (character.isascii() and (character.isalnum() or character in "._-"))
                        for character in item
                    )
                    for item in identifiers
                )
                or type(fields["source"]) is not str
                or not fields["source"]
                or "\r" in fields["source"]
                or "\x00" in fields["source"]
            ):
                raise ValueError("grammar payload identifiers or canonical LF source are invalid")
        raw = f"cubic.structured-output.v1\0{self.kind.value}\0{self.canonical_payload}"
        # Preserve existing identities for unchanged semantics. Formatting options
        # affect the language and therefore compiled-grammar and prefix identity.
        if options:
            raw += f"\0options\0{self.canonical_options}"
        object.__setattr__(self, "digest", f"b2s256:{hashlib.blake2s(raw.encode()).hexdigest()}")

    @classmethod
    def identity(cls) -> StructuredOutputMeta:
        return cls(StructuredOutputKind.IDENTITY, "null")

    @classmethod
    def json_schema(cls, schema: object, *, options: object = None) -> StructuredOutputMeta:
        return cls(
            StructuredOutputKind.JSON_SCHEMA,
            _canonical_json_input(schema),
            _canonical_json({} if options is None else options),
        )

    @classmethod
    def regex(cls, source: str) -> StructuredOutputMeta:
        return cls(StructuredOutputKind.REGEX, _canonical_json(source))

    @classmethod
    def choice(cls, choices: object) -> StructuredOutputMeta:
        return cls(StructuredOutputKind.CHOICE, _canonical_json(choices))

    @classmethod
    def structural_tag(cls, value: object) -> StructuredOutputMeta:
        return cls(StructuredOutputKind.STRUCTURAL_TAG, _canonical_json_input(value))

    @classmethod
    def tool_schema(
        cls,
        tools: object,
        *,
        choice: str = "auto",
        parallel_tool_calls: bool = True,
        format: str = "native",
        response: StructuredOutputMeta | None = None,
    ) -> StructuredOutputMeta:
        canonical_tools = tools if type(tools) is list else [tools]
        payload: dict[str, object] = {"choice": choice, "tools": canonical_tools}
        if type(parallel_tool_calls) is not bool:
            raise TypeError("parallel_tool_calls must be boolean")
        if not parallel_tool_calls:
            payload["parallel_tool_calls"] = False
        if format != "native":
            payload["format"] = format
        if response is not None:
            payload["response"] = {
                "kind": response.kind.value,
                "payload": response.canonical_payload,
                "options": response.canonical_options,
            }
        return cls(
            StructuredOutputKind.TOOL_SCHEMA,
            _canonical_json(payload),
        )

    @classmethod
    def grammar(
        cls,
        source: str,
        *,
        dialect: str = "xgrammar-ebnf",
        root_rule: str = "root",
    ) -> StructuredOutputMeta:
        return cls(
            StructuredOutputKind.GRAMMAR,
            _canonical_json({"dialect": dialect, "root_rule": root_rule, "source": source}),
        )


class MediaSourceKind(StrEnum):
    """Untrusted media carriers accepted before trusted media ingress."""

    URL = "url"
    BASE64 = "base64"


_BASE64_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
_BASE64_CHARACTERS = frozenset(_BASE64_ALPHABET)
_BASE64_VALUES = {character: value for value, character in enumerate(_BASE64_ALPHABET)}


def _require_content_digest_digest(value: str, *, field_name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    algorithm, separator, digest = value.partition(":")
    if algorithm != "b2s256" or separator != ":" or len(digest) != 64:
        raise ValueError(f"{field_name} must use b2s256:<64 hex chars>")
    try:
        int(digest, 16)
    except ValueError as error:
        raise ValueError(f"{field_name} contains non-hexadecimal characters") from error


def _require_canonical_mime_type(value: str) -> None:
    if not value or value != value.strip().lower() or value.count("/") != 1:
        raise ValueError("media source MIME type must be a canonical lowercase type/subtype")
    media_type, subtype = value.split("/", 1)
    token_characters = frozenset("abcdefghijklmnopqrstuvwxyz0123456789!#$&^_.+-")
    if (
        not media_type
        or not subtype
        or any(character not in token_characters for character in value if character != "/")
    ):
        raise ValueError("media source MIME type contains an invalid token character")


def _require_canonical_base64(value: str) -> None:
    """Validate canonical RFC 4648 base64 without materializing decoded bytes."""

    if type(value) is not str:
        raise TypeError("media source base64 payload must be a string")
    if not value:
        raise ValueError("media source base64 payload cannot be empty")
    if len(value) % 4:
        raise ValueError("media source base64 payload must use canonical encoding")

    padding = len(value) - len(value.rstrip("="))
    if padding > 2:
        raise ValueError("media source base64 payload must use canonical encoding")
    body = value[:-padding] if padding else value
    if not body or any(character not in _BASE64_CHARACTERS for character in body):
        raise ValueError("media source base64 payload is invalid")

    remainder = len(body) % 4
    if padding == 0:
        canonical = remainder == 0
    elif padding == 1:
        canonical = remainder == 3 and _BASE64_VALUES[body[-1]] & 0b11 == 0
    else:
        canonical = remainder == 2 and _BASE64_VALUES[body[-1]] & 0b1111 == 0
    if not canonical:
        raise ValueError("media source base64 payload must use canonical encoding")


@dataclass(frozen=True, slots=True)
class MediaSource:
    """One canonical, still-untrusted media locator or inline payload.

    This type performs no I/O. A later trusted ingress stage must resolve the
    source, enforce byte/pixel/frame limits, and replace it with a
    :class:`MediaDescriptor` derived from the actual bytes.
    """

    kind: MediaSourceKind
    uri: str | None = None
    mime_type: str | None = None
    base64_data: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MediaSourceKind):
            raise TypeError("media source kind must be a MediaSourceKind")
        if self.kind is MediaSourceKind.URL:
            if self.uri is None or self.mime_type is not None or self.base64_data is not None:
                raise ValueError("URL media source requires only uri")
            if self.uri != self.uri.strip() or any(
                ord(character) < 32 or ord(character) == 127 for character in self.uri
            ):
                raise ValueError(
                    "media source URL cannot contain surrounding or control whitespace"
                )
            parsed = urlsplit(self.uri)
            scheme = parsed.scheme.lower()
            if scheme in ("http", "https") and parsed.netloc:
                return
            if scheme == "file" and not parsed.netloc and parsed.path.startswith("/"):
                return
            raise ValueError("media source URL must be absolute HTTP(S) or file://")

        if self.uri is not None or self.mime_type is None or self.base64_data is None:
            raise ValueError("base64 media source requires only mime_type and base64_data")
        if type(self.mime_type) is not str:
            raise TypeError("media source MIME type must be a string")
        _require_canonical_mime_type(self.mime_type)
        _require_canonical_base64(self.base64_data)


@dataclass(frozen=True, slots=True)
class ChatContentPart:
    """One globally ordered chat part before media is fetched or encoded."""

    ordinal: int
    modality: str
    text: str | None = None
    media: MediaSource | None = None

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("chat content ordinal must be non-negative")
        if not self.modality or self.modality != self.modality.strip().lower():
            raise ValueError("chat content modality must be a canonical lowercase token")
        if self.modality == "text":
            if self.text is None or self.media is not None:
                raise ValueError("text chat content requires text and cannot carry media")
            return
        if self.text is not None or self.media is None:
            raise ValueError("non-text chat content requires media and cannot carry text")
        if self.media.mime_type is not None and not self.media.mime_type.startswith(
            f"{self.modality}/"
        ):
            raise ValueError("inline media MIME type differs from its chat modality")


@dataclass(frozen=True, slots=True)
class ChatToolCall:
    """One immutable historical function invocation, independent of model syntax."""

    call_id: str
    name: str
    canonical_arguments: str

    def __post_init__(self) -> None:
        if any(type(x) is not str or not x or x != x.strip() for x in (self.call_id, self.name)):
            raise ValueError("tool call identity and function name must be canonical strings")
        if type(_decode_canonical_json(self.canonical_arguments)) is not dict:
            raise ValueError("tool call arguments must be a JSON object")

    @classmethod
    def from_arguments(cls, call_id: str, name: str, arguments: object) -> ChatToolCall:
        return cls(call_id, name, _canonical_json_input(arguments))

    def template_value(self) -> dict[str, object]:
        return {
            "id": self.call_id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.loads(self.canonical_arguments),
            },
        }


@dataclass(frozen=True, slots=True)
class ToolDefinitions:
    """Prompt-visible tools survive tool_choice=none and forced-tool selection."""

    canonical_payload: str = "[]"

    def __post_init__(self) -> None:
        payload = _decode_canonical_json(self.canonical_payload, label="tool definitions")
        if type(payload) is not list:
            raise ValueError("tool definitions must be an array")
        names: list[str] = []
        for tool in payload:
            if type(tool) is not dict or set(tool) != {"type", "function"}:
                raise ValueError("tool definition requires type and function")
            function = tool["function"]
            if tool["type"] != "function" or type(function) is not dict:
                raise ValueError("tool definition must be a function")
            name = function.get("name")
            if type(name) is not str or not name or name != name.strip():
                raise ValueError("tool function name must be a canonical string")
            if type(function.get("parameters")) not in (dict, bool):
                raise ValueError("tool function parameters must be a JSON Schema")
            names.append(name)
        if len(names) != len(set(names)):
            raise ValueError("tool function names must be unique")

    @classmethod
    def from_value(cls, tools: object) -> ToolDefinitions:
        return cls(_canonical_json(tools))

    def as_list(self) -> list[dict[str, object]]:
        return cast(list[dict[str, object]], json.loads(self.canonical_payload))


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One role-bearing message whose content tuple preserves wire order."""

    role: ChatRole
    content_parts: tuple[ChatContentPart, ...]
    tool_calls: tuple[ChatToolCall, ...] = ()
    tool_call_id: str | None = None
    reasoning: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, ChatRole):
            raise TypeError("chat message role must be a ChatRole")
        if type(self.content_parts) is not tuple or not self.content_parts:
            raise ValueError("chat message requires a non-empty content part tuple")
        if any(not isinstance(part, ChatContentPart) for part in self.content_parts):
            raise TypeError("chat message content must contain only ChatContentPart values")
        if type(self.tool_calls) is not tuple or any(
            not isinstance(call, ChatToolCall) for call in self.tool_calls
        ):
            raise TypeError("historical tool calls must be immutable ChatToolCall values")
        if self.tool_calls and self.role is not ChatRole.ASSISTANT:
            raise ValueError("only assistant messages may contain tool calls")
        if len({call.call_id for call in self.tool_calls}) != len(self.tool_calls):
            raise ValueError("historical tool call ids must be unique within a message")
        if self.tool_call_id is not None and self.role is not ChatRole.TOOL:
            raise ValueError("tool_call_id belongs only to tool messages")
        if self.role is ChatRole.TOOL and self.tool_call_id is None:
            raise ValueError("tool messages require tool_call_id")
        for value in (self.tool_call_id, self.name):
            if value is not None and (
                type(value) is not str or not value or value != value.strip()
            ):
                raise ValueError("message name and tool_call_id must be canonical strings")
        if self.reasoning is not None and (
            type(self.reasoning) is not str or self.role is not ChatRole.ASSISTANT
        ):
            raise ValueError("reasoning belongs only to assistant messages")

    def template_metadata(self) -> dict[str, object]:
        result: dict[str, object] = {}
        if self.tool_calls:
            result["tool_calls"] = [call.template_value() for call in self.tool_calls]
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        if self.reasoning is not None:
            result["reasoning"] = self.reasoning
            result["reasoning_content"] = self.reasoning
        if self.name is not None:
            result["name"] = self.name
        return result


@dataclass(frozen=True, slots=True)
class SamplingMeta:
    """Optional request overrides; ``None`` delegates to the model defaults."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.temperature is not None and (
            type(self.temperature) is not float
            or not math.isfinite(self.temperature)
            or not 0.0 <= self.temperature <= 2.0
        ):
            raise ValueError("sampling temperature must be a finite float in [0, 2]")
        if self.top_p is not None and (
            type(self.top_p) is not float
            or not math.isfinite(self.top_p)
            or not 0.0 < self.top_p <= 1.0
        ):
            raise ValueError("sampling top_p must be a finite float in (0, 1]")
        if self.top_k is not None and (type(self.top_k) is not int or self.top_k <= 0):
            raise ValueError("sampling top_k must be a positive integer")
        if self.seed is not None and (
            type(self.seed) is not int or not -(2**63) <= self.seed < 2**63
        ):
            raise ValueError("sampling seed must be a signed 64-bit integer")


@dataclass(frozen=True, slots=True)
class ReasoningControl:
    """Model-declared boundaries and the open prompt's reasoning budget seed."""

    start: tuple[int, ...]
    end: tuple[int, ...]
    token_budget: int
    initially_reasoning: bool
    initial_reasoning_tokens: int = 0
    implicit_ends: tuple[tuple[int, ...], ...] = ()
    initial_tail: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.implicit_ends) is not tuple:
            raise TypeError("reasoning implicit boundaries must be immutable")
        for boundary in (self.start, self.end, *self.implicit_ends):
            if type(boundary) is not tuple or not boundary or any(
                type(token) is not int or token < 0 for token in boundary
            ):
                raise ValueError("reasoning controls require declared token boundaries")
        if (
            type(self.token_budget) is not int or self.token_budget < 0
            or type(self.initial_reasoning_tokens) is not int
            or self.initial_reasoning_tokens < 0
            or type(self.initially_reasoning) is not bool
            or (not self.initially_reasoning and self.initial_reasoning_tokens)
        ):
            raise ValueError("reasoning budget and prompt seed are invalid")
        if type(self.initial_tail) is not tuple or any(
            type(token) is not int or token < 0 for token in self.initial_tail
        ) or len(self.initial_tail) > self.initial_reasoning_tokens:
            raise ValueError("reasoning prompt tail is invalid")
        if self.initial_tail and not any(
            self.initial_tail == marker[:len(self.initial_tail)]
            and len(self.initial_tail) < len(marker)
            for marker in (self.end, *self.implicit_ends)
        ):
            raise ValueError("reasoning prompt tail must be a partial declared boundary")


@dataclass(frozen=True, slots=True)
class CanonicalChatRequest:
    """Wire-neutral chat semantics shared by OpenAI and Anthropic ingress.

    A missing output limit uses the remaining admitted context after input realization.
    Execution backends resolve it before token generation.
    """

    request_id: str
    model_id: str
    messages: tuple[ChatMessage, ...]
    max_new_tokens: int | None = None
    structured_output: StructuredOutputMeta = field(default_factory=StructuredOutputMeta.identity)
    chat_template_kwargs: ChatTemplateKwargs = field(default_factory=ChatTemplateKwargs)
    sampling: SamplingMeta = field(default_factory=SamplingMeta)
    stream_include_usage: bool = False
    stream_continuous_usage_stats: bool = False
    ignore_eos: bool = False
    tools: ToolDefinitions = field(default_factory=ToolDefinitions)
    parallel_tool_calls: bool = True
    include_reasoning: bool = True
    thinking_token_budget: int | None = None

    def __post_init__(self) -> None:
        if not self.request_id or not self.model_id:
            raise ValueError("canonical chat request and model identity are required")
        if type(self.messages) is not tuple or not self.messages:
            raise ValueError("canonical chat request requires a non-empty message tuple")
        if any(not isinstance(message, ChatMessage) for message in self.messages):
            raise TypeError("canonical chat messages must contain only ChatMessage values")
        if self.max_new_tokens is not None and (
            type(self.max_new_tokens) is not int or self.max_new_tokens <= 0
        ):
            raise ValueError("canonical max_new_tokens must be a positive integer or None")
        if not isinstance(self.structured_output, StructuredOutputMeta):
            raise TypeError("canonical structured output must be StructuredOutputMeta")
        if not isinstance(self.chat_template_kwargs, ChatTemplateKwargs):
            raise TypeError("canonical chat template kwargs must be ChatTemplateKwargs")
        if not isinstance(self.sampling, SamplingMeta):
            raise TypeError("canonical sampling metadata must be SamplingMeta")
        if type(self.stream_include_usage) is not bool:
            raise TypeError("canonical stream_include_usage must be a boolean")
        if type(self.stream_continuous_usage_stats) is not bool:
            raise TypeError("canonical continuous usage must be a boolean")
        if self.stream_continuous_usage_stats and not self.stream_include_usage:
            raise ValueError("continuous usage requires include_usage")
        if type(self.ignore_eos) is not bool:
            raise TypeError("canonical ignore_eos must be a boolean")
        if not isinstance(self.tools, ToolDefinitions):
            raise TypeError("canonical tools must be ToolDefinitions")
        if type(self.parallel_tool_calls) is not bool:
            raise TypeError("parallel_tool_calls must be boolean")
        if type(self.include_reasoning) is not bool:
            raise TypeError("include_reasoning must be boolean")
        if self.thinking_token_budget is not None and (
            type(self.thinking_token_budget) is not int or self.thinking_token_budget < 0
        ):
            raise ValueError("thinking_token_budget must be a non-negative integer or None")
        parts = tuple(part for message in self.messages for part in message.content_parts)
        if tuple(part.ordinal for part in parts) != tuple(range(len(parts))):
            raise ValueError("canonical chat content ordinals must be globally contiguous")


@dataclass(frozen=True, slots=True)
class MediaDescriptor:
    """Identity and realized bounds of media after trusted ingress normalization."""

    content_digest: str
    mime_type: str
    byte_length: int
    width: int | None = None
    height: int | None = None
    frame_count: int | None = None
    duration_ms: int | None = None

    def __post_init__(self) -> None:
        _require_content_digest_digest(self.content_digest, field_name="media digest")
        if not self.mime_type or self.mime_type != self.mime_type.lower():
            raise ValueError("media MIME type must be a canonical lowercase string")
        if self.byte_length <= 0:
            raise ValueError("media byte length must be positive")
        dimensions = (self.width, self.height, self.frame_count, self.duration_ms)
        if any(value is not None and value <= 0 for value in dimensions):
            raise ValueError("declared media dimensions and duration must be positive")
        if (self.width is None) != (self.height is None):
            raise ValueError("media width and height must be declared together")


@dataclass(frozen=True, slots=True)
class ContentPart:
    """One ordered canonical input part; modality names are extensible strings."""

    ordinal: int
    modality: str
    text: str | None = None
    media: MediaDescriptor | None = None

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("content ordinal must be non-negative")
        if not self.modality or self.modality != self.modality.strip().lower():
            raise ValueError("content modality must be a canonical lowercase token")
        if self.modality == "text":
            if self.text is None or self.media is not None:
                raise ValueError("text content requires text and cannot carry media")
        elif self.text is not None or self.media is None:
            raise ValueError("non-text content requires media and cannot carry text")


@dataclass(frozen=True, slots=True)
class TokenPosition:
    """Realized position axes for one decoder token (1D or multimodal RoPE)."""

    axes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.axes or any(axis < 0 for axis in self.axes):
            raise ValueError("token position axes must be non-empty and non-negative")


@dataclass(frozen=True, slots=True)
class DecodePositionPlan:
    """Exact affine continuation for tokens fed back during decode.

    The input processor, not the scheduler or model runtime, owns multimodal
    position semantics. ``first`` is the position of the first generated
    token when it is fed back as decode input; ``step_axes`` advances each
    realized axis for subsequent generated tokens.
    """

    first: TokenPosition
    step_axes: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.step_axes) != len(self.first.axes):
            raise ValueError("decode position step must match the realized position rank")
        if any(step < 0 for step in self.step_axes) or not any(self.step_axes):
            raise ValueError("decode position steps must be non-negative and not all zero")

    def at(self, decode_index: int) -> TokenPosition:
        if decode_index < 0:
            raise ValueError("decode position index must be non-negative")
        return TokenPosition(
            tuple(
                origin + decode_index * step
                for origin, step in zip(self.first.axes, self.step_axes, strict=True)
            )
        )


class FusionKind(StrEnum):
    """How feature rows map onto decoder placeholder tokens."""

    IDENTITY = "identity"


@dataclass(frozen=True, slots=True)
class TokenPlacement:
    """Maps one media part's decoder placeholder span to encoded feature rows."""

    content_part_ordinal: int
    token_start: int
    token_count: int
    feature_row_start: int
    feature_row_count: int
    consumer_layers: tuple[int, ...]
    consumer_site: str = "decoder.input"
    fusion: FusionKind = FusionKind.IDENTITY

    def __post_init__(self) -> None:
        if (
            min(
                self.content_part_ordinal,
                self.token_start,
                self.feature_row_start,
            )
            < 0
        ):
            raise ValueError("placement indices must be non-negative")
        if self.token_count <= 0 or self.feature_row_count <= 0:
            raise ValueError("placement token and feature counts must be positive")
        if self.fusion is FusionKind.IDENTITY and self.token_count != self.feature_row_count:
            raise ValueError("identity fusion requires one feature row per placeholder token")
        if not self.consumer_layers or tuple(sorted(set(self.consumer_layers))) != (
            self.consumer_layers
        ):
            raise ValueError("consumer layers must be non-empty, unique, and sorted")
        if self.consumer_layers[0] < 0:
            raise ValueError("consumer layers must be non-negative")
        if not self.consumer_site or self.consumer_site != self.consumer_site.strip():
            raise ValueError("consumer site identity must be non-empty and canonical")


@dataclass(frozen=True, slots=True)
class EncoderJob:
    job_id: str
    content_part_ordinal: int
    encoder_id: str
    processor_id: str
    cache_key: str

    def __post_init__(self) -> None:
        if self.content_part_ordinal < 0:
            raise ValueError("encoder job content ordinal must be non-negative")
        if not all((self.job_id, self.encoder_id, self.processor_id, self.cache_key)):
            raise ValueError("encoder job fields are required")


@dataclass(frozen=True, slots=True)
class InputPlan:
    """Fully realized request input; downstream code must not tokenize or inspect media."""

    request_id: str
    model_id: str
    processor_recipe_digest: str
    weight_variant: WeightVariantMeta
    content_parts: tuple[ContentPart, ...]
    token_ids: tuple[int, ...]
    positions: tuple[TokenPosition, ...]
    decode_positions: DecodePositionPlan
    placements: tuple[TokenPlacement, ...] = ()
    encoder_jobs: tuple[EncoderJob, ...] = ()

    def __post_init__(self) -> None:
        if not self.request_id or not self.model_id:
            raise ValueError("request and model identity are required")
        _require_content_digest_digest(
            self.processor_recipe_digest,
            field_name="processor recipe digest",
        )
        if not self.content_parts:
            raise ValueError("an input plan requires at least one content part")
        if tuple(part.ordinal for part in self.content_parts) != tuple(
            range(len(self.content_parts))
        ):
            raise ValueError("content part ordinals must be contiguous and ordered")
        if not self.token_ids or any(token_id < 0 for token_id in self.token_ids):
            raise ValueError("token ids must be non-empty and non-negative")
        if len(self.positions) != len(self.token_ids):
            raise ValueError("every token id must have one realized position")
        position_ranks = {len(position.axes) for position in self.positions}
        if len(position_ranks) != 1:
            raise ValueError("all token positions in an input plan must have the same axis count")
        if len(self.decode_positions.first.axes) not in position_ranks:
            raise ValueError("decode position rank must match realized input positions")

        part_by_ordinal = {part.ordinal: part for part in self.content_parts}
        placement_order = tuple(
            (
                placement.token_start,
                placement.content_part_ordinal,
                placement.feature_row_start,
            )
            for placement in self.placements
        )
        if placement_order != tuple(sorted(placement_order)):
            raise ValueError("token placements must use canonical token/content/feature order")
        occupied_token_indices: set[int] = set()
        occupied_feature_indices: dict[int, set[int]] = {}
        for placement in self.placements:
            part = part_by_ordinal.get(placement.content_part_ordinal)
            if part is None or part.modality == "text":
                raise ValueError("token placement must reference a non-text content part")
            if placement.token_start + placement.token_count > len(self.token_ids):
                raise ValueError("token placement exceeds the realized token sequence")
            token_indices = set(
                range(placement.token_start, placement.token_start + placement.token_count)
            )
            if occupied_token_indices & token_indices:
                raise ValueError("token placement spans cannot overlap")
            occupied_token_indices.update(token_indices)
            feature_indices = set(
                range(
                    placement.feature_row_start,
                    placement.feature_row_start + placement.feature_row_count,
                )
            )
            part_feature_indices = occupied_feature_indices.setdefault(
                placement.content_part_ordinal,
                set(),
            )
            if part_feature_indices & feature_indices:
                raise ValueError("one media part's feature placement spans cannot overlap")
            part_feature_indices.update(feature_indices)

        job_ids = tuple(job.job_id for job in self.encoder_jobs)
        if len(set(job_ids)) != len(job_ids):
            raise ValueError("encoder job ids must be unique")
        job_ordinals = tuple(job.content_part_ordinal for job in self.encoder_jobs)
        if len(set(job_ordinals)) != len(job_ordinals):
            raise ValueError("each non-text content part requires exactly one encoder job")
        for job in self.encoder_jobs:
            part = part_by_ordinal.get(job.content_part_ordinal)
            if part is None or part.modality == "text":
                raise ValueError("encoder job must reference a non-text content part")

        placed_parts = {placement.content_part_ordinal for placement in self.placements}
        encoded_parts = {job.content_part_ordinal for job in self.encoder_jobs}
        non_text_parts = {part.ordinal for part in self.content_parts if part.modality != "text"}
        if placed_parts != non_text_parts or encoded_parts != non_text_parts:
            raise ValueError("every non-text part requires one encoder job and token placements")

    @property
    def semantic_digest(self) -> str:
        """Canonical request-independent identity of the fully realized input."""

        payload = asdict(self)
        del payload["request_id"]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return f"b2s256:{hashlib.blake2s(encoded).hexdigest()}"


class TokenEventKind(StrEnum):
    TOKEN = "token"
    TEXT_DELTA = "text_delta"
    REASONING_DELTA = "reasoning_delta"
    TOOL_CALL_DELTA = "tool_call_delta"
    USAGE = "usage"
    FINISH = "finish"


@dataclass(frozen=True, slots=True)
class TokenEvent:
    """The sole canonical output stream consumed by both serving protocols."""

    request_id: str
    sequence_no: int
    kind: TokenEventKind
    token_id: int | None = None
    output_token_count: int | None = None
    text: str | None = None
    payload_items: tuple[tuple[str, str], ...] = ()
    reasoning_token_count: int | None = None

    def __post_init__(self) -> None:
        if not self.request_id or self.sequence_no < 0:
            raise ValueError("event request identity and non-negative sequence are required")
        if self.output_token_count is not None and (
            type(self.output_token_count) is not int or self.output_token_count < 0
        ):
            raise ValueError("output token count must be non-negative")
        keys = tuple(key for key, _ in self.payload_items)
        if self.reasoning_token_count is not None and (
            type(self.reasoning_token_count) is not int
            or self.reasoning_token_count < 0
            or self.output_token_count is None
            or self.reasoning_token_count > self.output_token_count
        ):
            raise ValueError("reasoning token progress must fit output token progress")
        if tuple(sorted(keys)) != keys or len(set(keys)) != len(keys):
            raise ValueError("event payload keys must be unique and sorted")
        if any(not key for key in keys):
            raise ValueError("event payload keys cannot be empty")
        if self.token_id is not None and self.token_id < 0:
            raise ValueError("event token id must be non-negative")
        if self.kind is TokenEventKind.TOKEN and self.token_id is None:
            raise ValueError("TOKEN event requires token_id")
        if (
            self.kind
            in (
                TokenEventKind.TEXT_DELTA,
                TokenEventKind.REASONING_DELTA,
                TokenEventKind.TOOL_CALL_DELTA,
            )
            and self.text is None
        ):
            raise ValueError(f"{self.kind} event requires text")
