"""Optional Qwen3.5 reasoning/tool output recipe over the common pipeline."""

from __future__ import annotations

import codecs
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, cast

from tokenizers import Tokenizer  # type: ignore[import-untyped]

from cubic_serve.contracts.request import TokenEventKind, ToolDefinitions
from cubic_serve.generation.output import (
    ParsedOutputDelta,
    argument_types,
    resolve_parser_name,
    typed_tool_argument,
)
from cubic_serve.generation.sampler import load_eos_token_ids

_RECIPE_KIND = "quanttrio.qwen3_5.output.v1"
_THINK_START = "<think>"
_THINK_END = "</think>"
_TOOL_START = "<tool_call>"
_TOOL_END = "</tool_call>"
_FUNCTION_START = "<function="
_FUNCTION_END = "</function>"
_PARAMETER_START = "<parameter="
_PARAMETER_END = "</parameter>"
_MARKERS = (
    _FUNCTION_START,
    _PARAMETER_START,
    _THINK_END,
    _THINK_START,
    _TOOL_END,
    _TOOL_START,
    _FUNCTION_END,
    _PARAMETER_END,
)
_REASONING_MARKERS = (_THINK_END, _THINK_START)
_TOOL_MARKERS = (
    _FUNCTION_START,
    _PARAMETER_START,
    _TOOL_END,
    _TOOL_START,
    _FUNCTION_END,
    _PARAMETER_END,
)
_REQUIRED_TOKEN_IDS = {
    "<|im_end|>": 248046,
    _TOOL_START: 248058,
    _TOOL_END: 248059,
    _THINK_START: 248068,
    _THINK_END: 248069,
}
_REASONING_TEMPLATE_FRAGMENTS = (
    "<think>\\n",
    "\\n</think>\\n\\n",
)
_TOOL_TEMPLATE_FRAGMENTS = (
    "<tool_call>\\n<function=",
    "<parameter=",
    "\\n</parameter>\\n",
    "</function>\\n</tool_call>",
)

_ParserState = Literal[
    "reasoning",
    "content",
    "tool_preamble",
    "function_body",
    "parameter_value",
    "tool_close",
    "tool_gap",
]


class Qwen3_5OutputError(ValueError):
    """The local tokenizer or generated Qwen output is not exactly supported."""


def _content_digest_bytes(value: bytes) -> str:
    return f"b2s256:{hashlib.blake2s(value).hexdigest()}"


def _read_file(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise Qwen3_5OutputError(f"cannot read {label} file {path}: {error}") from error


def _byte_decoder() -> dict[str, int]:
    byte_values = [
        *range(ord("!"), ord("~") + 1),
        *range(ord("¡"), ord("¬") + 1),
        *range(ord("®"), ord("ÿ") + 1),
    ]
    unicode_values = list(byte_values)
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            unicode_values.append(256 + extra)
            extra += 1
    return {
        chr(character): value for value, character in zip(byte_values, unicode_values, strict=True)
    }


_BYTE_DECODER = _byte_decoder()


class Qwen3_5TokenizerDecoder:
    """Stateful Qwen ByteLevel detokenizer with streaming UTF-8 recovery."""

    __slots__ = ("_closed", "_eos_token_id", "_tokenizer", "_utf8")

    def __init__(self, tokenizer: Tokenizer, *, eos_token_id: int) -> None:
        self._tokenizer = tokenizer
        self._eos_token_id = eos_token_id
        self._utf8: codecs.IncrementalDecoder = codecs.getincrementaldecoder("utf-8")(
            errors="replace"
        )
        self._closed = False

    def decode(self, token_id: int) -> str:
        if self._closed:
            raise Qwen3_5OutputError("token arrived after decoder finish")
        if type(token_id) is not int or token_id < 0:
            raise Qwen3_5OutputError("token id must be a non-negative integer")
        if token_id == self._eos_token_id:
            return ""
        token = self._tokenizer.id_to_token(token_id)
        if token is None:
            raise Qwen3_5OutputError(f"tokenizer has no token id {token_id}")
        try:
            encoded = bytes(_BYTE_DECODER[character] for character in token)
        except KeyError as error:
            raise Qwen3_5OutputError(
                f"token id {token_id} is not encoded by the admitted ByteLevel alphabet"
            ) from error
        return self._utf8.decode(encoded, final=False)

    def finish(self) -> str:
        if self._closed:
            raise Qwen3_5OutputError("incremental decoder finished more than once")
        self._closed = True
        pending, _ = self._utf8.getstate()
        if pending:
            # A length stop may cut ByteLevel tokens inside one codepoint.
            # llama.cpp truncates this tail and vLLM/SGLang keep it un-emitted;
            # it is not a corrupt token stream.
            self._utf8.reset()
            return ""
        return self._utf8.decode(b"", final=True)


def _partial_marker_length(value: str, markers: tuple[str, ...] = _MARKERS) -> int:
    for size in range(min(len(value), max(map(len, markers)) - 1), 0, -1):
        suffix = value[-size:]
        if any(marker.startswith(suffix) for marker in markers):
            return size
    return 0


def _first_marker(
    value: str,
    markers: tuple[str, ...] = _MARKERS,
) -> tuple[int, str] | None:
    matches = ((index, marker) for marker in markers if (index := value.find(marker)) >= 0)
    return min(matches, default=None)


def _validate_tag_name(value: str, label: str) -> None:
    if (
        not value
        or value != value.strip()
        or any(character.isspace() or character in "<>" for character in value)
    ):
        raise Qwen3_5OutputError(f"{label} must be a non-empty tag-safe name")


def _trim_wrapping_newlines(value: str) -> str:
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


class Qwen3_5OutputParser:
    """Chunk-invariant strict state machine for Qwen reasoning and tool XML."""

    __slots__ = (
        "_buffer",
        "_failed",
        "_finished",
        "_function_name",
        "_parameter_name",
        "_parameter_streamed",
        "_parameters",
        "_parse_reasoning",
        "_parse_tools",
        "_reasoning_start_allowed",
        "_state",
        "_stream_arguments",
        "_string_parameter",
        "_tool_index",
        "_tool_schemas",
    )

    def __init__(
        self,
        *,
        enable_thinking: bool,
        reasoning_parser: bool,
        tool_call_parser: bool,
    ) -> None:
        if any(
            type(setting) is not bool
            for setting in (enable_thinking, reasoning_parser, tool_call_parser)
        ):
            raise TypeError("Qwen output parser settings must be bool values")
        if not reasoning_parser and not tool_call_parser:
            raise ValueError("at least one Qwen output parser must be enabled")
        parse_reasoning = reasoning_parser and enable_thinking
        self._state: _ParserState = "reasoning" if parse_reasoning else "content"
        self._reasoning_start_allowed = parse_reasoning
        self._parse_reasoning = reasoning_parser
        self._parse_tools = tool_call_parser
        self._buffer = ""
        self._function_name: str | None = None
        self._parameter_name: str | None = None
        self._parameters: dict[str, object] = {}
        self._tool_schemas: dict[str, dict[str, Any]] = {}
        self._stream_arguments = False
        self._parameter_streamed = 0
        self._string_parameter = False
        self._tool_index = 0
        self._finished = False
        self._failed = False

    def bind_tools(self, tools: ToolDefinitions) -> None:
        if self._buffer or self._function_name or self._tool_index:
            raise Qwen3_5OutputError("tool schemas must be bound before parsing")
        self._tool_schemas = {
            cast(dict[str, Any], tool["function"])["name"]: cast(dict[str, Any], tool["function"])[
                "parameters"
            ]
            for tool in tools.as_list()
        }
        self._stream_arguments = True

    def _parameter_schema(self) -> object:
        if self._function_name is None:
            return None
        return (
            self._tool_schemas.get(self._function_name, {})
            .get("properties", {})
            .get(self._parameter_name)
        )

    def _argument_delta(self, text: str) -> ParsedOutputDelta:
        return ParsedOutputDelta(
            TokenEventKind.TOOL_CALL_DELTA, text, (("index", str(self._tool_index)),)
        )

    def push(self, token_id: int, text: str) -> tuple[ParsedOutputDelta, ...]:
        if self._failed:
            raise Qwen3_5OutputError("Qwen output parser is failed")
        if self._finished:
            raise Qwen3_5OutputError("text arrived after Qwen output parser finish")
        if type(token_id) is not int or token_id < 0 or type(text) is not str:
            raise Qwen3_5OutputError("parser input requires a non-negative token id and string")
        try:
            self._buffer += text
            return self._drain(final=False)
        except Exception:
            self._failed = True
            raise

    def finish(self, text: str) -> tuple[ParsedOutputDelta, ...]:
        if self._failed:
            raise Qwen3_5OutputError("Qwen output parser is failed")
        if self._finished:
            raise Qwen3_5OutputError("Qwen output parser finished more than once")
        if type(text) is not str:
            raise Qwen3_5OutputError("parser finish tail must be a string")
        try:
            self._buffer += text
            emitted = self._drain(final=True)
            if self._state == "reasoning":
                # EOS may terminate an unfinished reasoning span. Publish the
                # observed reasoning, as upstream parsers do, without a fake close.
                self._state = "content"
            if self._state not in ("content", "tool_gap") or self._buffer:
                raise Qwen3_5OutputError("tool call structure is not closed")
            self._finished = True
            return emitted
        except Exception:
            self._failed = True
            raise

    def truncate(self, text: str) -> tuple[ParsedOutputDelta, ...]:
        """Flush observed fields on a length stop without claiming tool completion."""
        if self._failed or self._finished:
            raise Qwen3_5OutputError("cannot truncate a closed output parser")
        if type(text) is not str:
            raise Qwen3_5OutputError("parser tail must be a string")
        try:
            self._buffer += text
            emitted = list(self._drain(final=False))
            if self._state in ("reasoning", "content") and self._buffer:
                kind = (
                    TokenEventKind.REASONING_DELTA
                    if self._state == "reasoning"
                    else TokenEventKind.TEXT_DELTA
                )
                emitted.append(ParsedOutputDelta(kind, self._buffer))
            elif self._function_name is not None and not self._stream_arguments:
                parameters = dict(self._parameters)
                if self._state == "parameter_value" and self._parameter_name is not None:
                    holdback = _partial_marker_length(self._buffer, (_PARAMETER_END,))
                    value = self._buffer[: len(self._buffer) - holdback]
                    parameters[self._parameter_name] = _trim_wrapping_newlines(value)
                index = self._tool_index
                emitted.append(
                    ParsedOutputDelta(
                        TokenEventKind.TOOL_CALL_DELTA,
                        json.dumps(
                            parameters, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                        ),
                        (
                            ("id", f"call_{index}"),
                            ("index", str(index)),
                            ("name", self._function_name),
                        ),
                    )
                )
            self._buffer = ""
            self._finished = True
            return tuple(emitted)
        except Exception:
            self._failed = True
            raise

    def _drain(self, *, final: bool) -> tuple[ParsedOutputDelta, ...]:
        emitted: list[ParsedOutputDelta] = []
        while self._buffer:
            if self._state == "reasoning":
                if not self._drain_reasoning(emitted, final=final):
                    break
            elif self._state == "content":
                if not self._drain_content(emitted, final=final):
                    break
            elif self._state == "tool_preamble":
                if not self._drain_tool_preamble(emitted, final=final):
                    break
            elif self._state == "function_body":
                if not self._drain_function_body(emitted, final=final):
                    break
            elif self._state == "parameter_value":
                if not self._drain_parameter_value(emitted):
                    break
            elif self._state == "tool_close":
                delta = self._drain_tool_close(final=final)
                if delta is None:
                    break
                emitted.append(delta)
            else:
                if not self._drain_tool_gap(final=final):
                    break
        return tuple(emitted)

    def _drain_reasoning(self, emitted: list[ParsedOutputDelta], *, final: bool) -> bool:
        markers = _MARKERS if self._parse_tools else _REASONING_MARKERS
        found = _first_marker(self._buffer, markers)
        if found is None:
            holdback = 0 if final else _partial_marker_length(self._buffer, markers)
            safe_length = len(self._buffer) - holdback
            if safe_length:
                emitted.append(
                    ParsedOutputDelta(TokenEventKind.REASONING_DELTA, self._buffer[:safe_length])
                )
                self._buffer = self._buffer[safe_length:]
                self._reasoning_start_allowed = False
            return False
        index, marker = found
        if index:
            emitted.append(ParsedOutputDelta(TokenEventKind.REASONING_DELTA, self._buffer[:index]))
            self._buffer = self._buffer[index:]
            self._reasoning_start_allowed = False
            return True
        if marker == _THINK_START and self._reasoning_start_allowed:
            self._buffer = self._buffer[len(marker) :]
            self._reasoning_start_allowed = False
            return True
        if marker == _THINK_END:
            self._buffer = self._buffer[len(marker) :]
            self._state = "content"
            self._reasoning_start_allowed = False
            return True
        raise Qwen3_5OutputError(f"invalid {marker!r} nesting inside reasoning")

    def _drain_content(self, emitted: list[ParsedOutputDelta], *, final: bool) -> bool:
        markers = (
            _MARKERS
            if self._parse_reasoning and self._parse_tools
            else _REASONING_MARKERS
            if self._parse_reasoning
            else _TOOL_MARKERS
        )
        found = _first_marker(self._buffer, markers)
        if found is None:
            holdback = 0 if final else _partial_marker_length(self._buffer, markers)
            safe_length = len(self._buffer) - holdback
            if not final:
                safe_length = len(self._buffer[:safe_length].rstrip())
            if safe_length:
                emitted.append(
                    ParsedOutputDelta(TokenEventKind.TEXT_DELTA, self._buffer[:safe_length])
                )
                self._buffer = self._buffer[safe_length:]
            return False
        index, marker = found
        if marker != _TOOL_START:
            raise Qwen3_5OutputError(f"invalid {marker!r} nesting inside content")
        visible = self._buffer[:index].rstrip()
        if visible:
            emitted.append(ParsedOutputDelta(TokenEventKind.TEXT_DELTA, visible))
        self._buffer = self._buffer[index + len(marker) :]
        self._state = "tool_preamble"
        return True

    def _drain_tool_preamble(self, emitted: list[ParsedOutputDelta], *, final: bool) -> bool:
        self._buffer = self._buffer.lstrip()
        if not self._buffer:
            return False
        if not self._buffer.startswith(_FUNCTION_START):
            if not final and _FUNCTION_START.startswith(self._buffer):
                return False
            raise Qwen3_5OutputError("tool_call must contain exactly one function block")
        name_end = self._buffer.find(">", len(_FUNCTION_START))
        if name_end < 0:
            if final:
                raise Qwen3_5OutputError("function start tag is not closed")
            return False
        name = self._buffer[len(_FUNCTION_START) : name_end]
        _validate_tag_name(name, "function name")
        # Tool declarations supply argument types, not an output allowlist.
        # vLLM 003c91e / SGLang ba433bb preserve unknown names in non-strict
        # native calls. Required/named/strict languages are sampler constraints.
        self._function_name = name
        if self._stream_arguments:
            emitted.append(
                ParsedOutputDelta(
                    TokenEventKind.TOOL_CALL_DELTA,
                    "{",
                    (
                        ("id", f"call_{self._tool_index}"),
                        ("index", str(self._tool_index)),
                        ("name", name),
                    ),
                )
            )
        self._parameters = {}
        self._buffer = self._buffer[name_end + 1 :]
        self._state = "function_body"
        return True

    def _drain_function_body(self, emitted: list[ParsedOutputDelta], *, final: bool) -> bool:
        self._buffer = self._buffer.lstrip()
        if not self._buffer:
            return False
        if self._buffer.startswith(_PARAMETER_START):
            name_end = self._buffer.find(">", len(_PARAMETER_START))
            if name_end < 0:
                if final:
                    raise Qwen3_5OutputError("parameter start tag is not closed")
                return False
            name = self._buffer[len(_PARAMETER_START) : name_end]
            _validate_tag_name(name, "parameter name")
            if name in self._parameters:
                raise Qwen3_5OutputError(f"duplicate tool parameter {name!r}")
            self._parameter_name = name
            self._parameter_streamed = 0
            self._string_parameter = argument_types(self._parameter_schema()) == {"string"}
            if self._stream_arguments and self._string_parameter:
                prefix = "," if self._parameters else ""
                emitted.append(
                    self._argument_delta(prefix + json.dumps(name, ensure_ascii=False) + ':"')
                )
            self._buffer = self._buffer[name_end + 1 :]
            self._state = "parameter_value"
            return True
        if self._buffer.startswith(_FUNCTION_END):
            self._buffer = self._buffer[len(_FUNCTION_END) :]
            self._state = "tool_close"
            return True
        if not final and any(
            marker.startswith(self._buffer) for marker in (_PARAMETER_START, _FUNCTION_END)
        ):
            return False
        raise Qwen3_5OutputError("function body contains text or invalid nesting")

    def _drain_parameter_value(self, emitted: list[ParsedOutputDelta]) -> bool:
        found = _first_marker(self._buffer, (_PARAMETER_END,))
        if found is None:
            if self._stream_arguments and self._string_parameter:
                holdback = _partial_marker_length(self._buffer, (_PARAMETER_END,))
                value = self._buffer[: len(self._buffer) - holdback]
                value = value.removeprefix("\n").removesuffix("\n")
                if len(value) > self._parameter_streamed:
                    delta = value[self._parameter_streamed :]
                    self._parameter_streamed = len(value)
                    emitted.append(
                        self._argument_delta(json.dumps(delta, ensure_ascii=False)[1:-1])
                    )
            return False
        index, marker = found
        if marker != _PARAMETER_END:
            raise Qwen3_5OutputError(f"invalid {marker!r} nesting inside parameter value")
        name = self._parameter_name
        if name is None:
            raise Qwen3_5OutputError("parameter state has no parameter name")
        raw = _trim_wrapping_newlines(self._buffer[:index])
        converted: object = typed_tool_argument(raw, self._parameter_schema())
        if self._stream_arguments:
            if self._string_parameter:
                emitted.append(
                    self._argument_delta(
                        json.dumps(raw[self._parameter_streamed :], ensure_ascii=False)[1:-1] + '"'
                    )
                )
            else:
                prefix = "," if self._parameters else ""
                emitted.append(
                    self._argument_delta(
                        prefix
                        + json.dumps(name, ensure_ascii=False)
                        + ":"
                        + json.dumps(
                            converted, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                        )
                    )
                )
        self._parameters[name] = converted
        self._parameter_name = None
        self._buffer = self._buffer[index + len(marker) :]
        self._state = "function_body"
        return True

    def _drain_tool_close(self, *, final: bool) -> ParsedOutputDelta | None:
        self._buffer = self._buffer.lstrip()
        if not self._buffer:
            return None
        if not self._buffer.startswith(_TOOL_END):
            if not final and _TOOL_END.startswith(self._buffer):
                return None
            raise Qwen3_5OutputError("function must be followed by tool_call close")
        self._buffer = self._buffer[len(_TOOL_END) :]
        name = self._function_name
        if name is None:
            raise Qwen3_5OutputError("tool state has no function name")
        arguments = json.dumps(
            self._parameters,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        index = self._tool_index
        delta = ParsedOutputDelta(
            TokenEventKind.TOOL_CALL_DELTA,
            arguments,
            (
                ("id", f"call_{index}"),
                ("index", str(index)),
                ("name", name),
            ),
        )
        if self._stream_arguments:
            delta = self._argument_delta("}")
        self._function_name = None
        self._parameters = {}
        self._tool_index += 1
        self._state = "tool_gap"
        return delta

    def _drain_tool_gap(self, *, final: bool) -> bool:
        self._buffer = self._buffer.lstrip()
        if not self._buffer:
            return False
        if self._buffer.startswith(_TOOL_START):
            self._buffer = self._buffer[len(_TOOL_START) :]
            self._state = "tool_preamble"
            return True
        if not final and _TOOL_START.startswith(self._buffer):
            return False
        self._state = "content"
        return True


class Qwen3_5OutputCodec:
    """Admit one local tokenizer/template recipe and create per-request state."""

    xgrammar_vocab_type = "byte_level"
    tool_model_format = "qwen_3_5"
    tool_reasoning = False
    tool_call_prefix = _TOOL_START
    reasoning_parser_names = ("qwen3",)
    tool_parser_names = ("qwen3_coder", "qwen3_xml")

    def _parser_selection(
        self,
        reasoning_parser: str | bool,
        tool_call_parser: str | bool,
    ) -> tuple[bool, bool]:
        return (
            resolve_parser_name(reasoning_parser, self.reasoning_parser_names) is not None,
            resolve_parser_name(tool_call_parser, self.tool_parser_names) is not None,
        )

    __slots__ = (
        "_artifact_digests",
        "_eos_token_id",
        "_eos_token_ids",
        "_parser_artifacts",
        "_template_path",
        "_tokenizer",
    )

    def __init__(
        self,
        model_root: str | Path,
        *,
        chat_template_path: str | Path | None = None,
    ) -> None:
        root = Path(model_root)
        tokenizer_path = root / "tokenizer.json"
        tokenizer_config_path = root / "tokenizer_config.json"
        template_path = (
            Path(chat_template_path)
            if chat_template_path is not None
            else root / "chat_template.jinja"
        )
        tokenizer_source = _read_file(tokenizer_path, "tokenizer")
        tokenizer_config_source = _read_file(tokenizer_config_path, "tokenizer config")
        try:
            tokenizer_document = json.loads(tokenizer_source)
            tokenizer_config = json.loads(tokenizer_config_source)
            tokenizer_text = tokenizer_source.decode("utf-8")
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise Qwen3_5OutputError("Qwen tokenizer files are malformed") from error
        if not isinstance(tokenizer_document, dict) or not isinstance(tokenizer_config, dict):
            raise Qwen3_5OutputError("Qwen tokenizer documents must be JSON objects")
        decoder_config = tokenizer_document.get("decoder")
        if not isinstance(decoder_config, dict) or decoder_config != {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        }:
            raise Qwen3_5OutputError("Qwen tokenizer must use the admitted ByteLevel decoder")
        if tokenizer_config.get("eos_token") != "<|im_end|>":
            raise Qwen3_5OutputError("Qwen tokenizer EOS token must be <|im_end|>")
        try:
            tokenizer = Tokenizer.from_str(tokenizer_text)
        except Exception as error:
            raise Qwen3_5OutputError(f"cannot load tokenizer.json: {error}") from error
        eos_token_id = _REQUIRED_TOKEN_IDS["<|im_end|>"]
        if tokenizer.token_to_id("<|im_end|>") != eos_token_id:
            raise Qwen3_5OutputError(
                f"Qwen tokenizer token '<|im_end|>' must have id {eos_token_id}"
            )
        self._tokenizer = tokenizer
        self._eos_token_id = eos_token_id
        self._eos_token_ids = load_eos_token_ids(root, tokenizer_eos=eos_token_id)
        self._template_path = template_path
        self._parser_artifacts: dict[tuple[bool, bool], tuple[tuple[str, str], ...]] = {}
        self._artifact_digests = tuple(
            sorted(
                (
                    ("tokenizer.json", _content_digest_bytes(tokenizer_source)),
                    ("tokenizer_config.json", _content_digest_bytes(tokenizer_config_source)),
                )
            )
        )

    @property
    def artifact_digests(self) -> tuple[tuple[str, str], ...]:
        return self._artifact_digests

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        return self._eos_token_ids

    @property
    def reasoning_end_token_ids(self) -> tuple[int, ...]:
        return tuple(self._tokenizer.encode(_THINK_END, add_special_tokens=False).ids)

    @property
    def reasoning_start_token_ids(self) -> tuple[int, ...]:
        return tuple(self._tokenizer.encode(_THINK_START, add_special_tokens=False).ids)


    @property
    def reasoning_implicit_end_token_ids(self) -> tuple[tuple[int, ...], ...]:
        return (tuple(self._tokenizer.encode(_TOOL_START, add_special_tokens=False).ids),)

    def admit_parser_recipe(
        self,
        *,
        reasoning_parser: str | bool,
        tool_call_parser: str | bool,
    ) -> None:
        """Validate only the explicitly selected checkpoint parser grammars."""

        reasoning_parser, tool_call_parser = self._parser_selection(
            reasoning_parser, tool_call_parser
        )
        if not reasoning_parser and not tool_call_parser:
            return
        key = (reasoning_parser, tool_call_parser)
        if key in self._parser_artifacts:
            return

        template_source = _read_file(self._template_path, "chat template")
        parser_source = _read_file(Path(__file__), "Qwen output parser source")
        try:
            template_text = template_source.decode("utf-8")
        except UnicodeDecodeError as error:
            raise Qwen3_5OutputError("Qwen chat template is not UTF-8") from error
        fragments: tuple[str, ...] = _REASONING_TEMPLATE_FRAGMENTS if reasoning_parser else ()
        if tool_call_parser:
            fragments += _TOOL_TEMPLATE_FRAGMENTS
        missing = tuple(fragment for fragment in fragments if fragment not in template_text)
        if missing:
            raise Qwen3_5OutputError(
                "chat template does not declare the selected Qwen output parser grammar"
            )
        tokens = {
            **(
                {
                    _THINK_START: _REQUIRED_TOKEN_IDS[_THINK_START],
                    _THINK_END: _REQUIRED_TOKEN_IDS[_THINK_END],
                }
                if reasoning_parser
                else {}
            ),
            **(
                {
                    _TOOL_START: _REQUIRED_TOKEN_IDS[_TOOL_START],
                    _TOOL_END: _REQUIRED_TOKEN_IDS[_TOOL_END],
                }
                if tool_call_parser
                else {}
            ),
        }
        for token, expected_id in tokens.items():
            if self._tokenizer.token_to_id(token) != expected_id:
                raise Qwen3_5OutputError(
                    f"Qwen tokenizer token {token!r} must have id {expected_id}"
                )
        self._parser_artifacts[key] = tuple(
            sorted(
                (
                    ("chat_template", _content_digest_bytes(template_source)),
                    ("parser_source", _content_digest_bytes(parser_source)),
                )
            )
        )

    def recipe_digest(
        self,
        *,
        enable_thinking: bool,
        reasoning_parser: str | bool = False,
        tool_call_parser: str | bool = False,
    ) -> str:
        reasoning_parser, tool_call_parser = self._parser_selection(
            reasoning_parser, tool_call_parser
        )
        if any(
            type(setting) is not bool
            for setting in (enable_thinking, reasoning_parser, tool_call_parser)
        ):
            raise TypeError("Qwen output recipe settings must be bool values")
        self.admit_parser_recipe(
            reasoning_parser=reasoning_parser,
            tool_call_parser=tool_call_parser,
        )
        parser_artifacts = self._parser_artifacts.get((reasoning_parser, tool_call_parser), ())
        settings: dict[str, object] = {
            "eos_token_ids": self._eos_token_ids,
            "reasoning_parser": reasoning_parser,
            "tool_call_parser": tool_call_parser,
        }
        if reasoning_parser:
            settings["enable_thinking"] = enable_thinking
        if tool_call_parser:
            settings.update(
                {
                    "parameter_values": "strings",
                    "tool_call_ids": "zero_based_call_index",
                }
            )
        recipe = {
            "artifacts": dict((*self._artifact_digests, *parser_artifacts)),
            "kind": _RECIPE_KIND,
            "runtime": {"tokenizers": version("tokenizers")},
            "settings": settings,
        }
        return _content_digest_bytes(
            json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
        )

    def new_decoder(self) -> Qwen3_5TokenizerDecoder:
        return Qwen3_5TokenizerDecoder(self._tokenizer, eos_token_id=self._eos_token_id)

    def new_parser(
        self,
        *,
        enable_thinking: bool,
        reasoning_parser: str | bool = False,
        tool_call_parser: str | bool = False,
    ) -> Qwen3_5OutputParser | None:
        if type(enable_thinking) is not bool:
            raise TypeError("enable_thinking must be bool")
        self.admit_parser_recipe(
            reasoning_parser=reasoning_parser,
            tool_call_parser=tool_call_parser,
        )
        reasoning_parser, tool_call_parser = self._parser_selection(
            reasoning_parser, tool_call_parser
        )
        if not reasoning_parser and not tool_call_parser:
            return None
        return Qwen3_5OutputParser(
            enable_thinking=enable_thinking,
            reasoning_parser=reasoning_parser,
            tool_call_parser=tool_call_parser,
        )


__all__ = (
    "Qwen3_5OutputCodec",
    "Qwen3_5OutputError",
    "Qwen3_5OutputParser",
    "Qwen3_5TokenizerDecoder",
)
