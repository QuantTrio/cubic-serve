"""Strict dependency-free normalization for public chat wire payloads.

The functions in this module consume already-decoded JSON objects. They do no
network or filesystem I/O and deliberately keep media sources untrusted; trusted
media ingress later realizes :class:`MediaSource` values into byte-derived
``MediaDescriptor`` values.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import cast

from cubic_serve.contracts.request import (
    CanonicalChatRequest,
    ChatContentPart,
    ChatMessage,
    ChatRole,
    ChatTemplateKwargs,
    ChatToolCall,
    MediaSource,
    MediaSourceKind,
    SamplingMeta,
    StructuredOutputKind,
    StructuredOutputMeta,
    ToolDefinitions,
)


class ChatNormalizationError(ValueError):
    """A public chat payload cannot be represented without changing semantics."""


MAX_MEDIA_BYTES = 32 * 1024 * 1024
MAX_INLINE_MEDIA_BASE64_CHARS = 4 * ((MAX_MEDIA_BYTES + 2) // 3)


_OPENAI_TOP_LEVEL_FIELDS = frozenset(
    {
        "grammar",
        "chat_template_kwargs",
        "ignore_eos",
        "include_reasoning",
        "max_completion_tokens",
        "max_tokens",
        "messages",
        "model",
        "parallel_tool_calls",
        "reasoning_effort",
        "response_format",
        "structured_outputs",
        "seed",
        "stream",
        "stream_options",
        "temperature",
        "thinking_token_budget",
        "tool_choice",
        "tools",
        "top_k",
        "top_p",
    }
)

_OPENAI_REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
_ANTHROPIC_TOP_LEVEL_FIELDS = frozenset(
    {
        "chat_template_kwargs",
        "grammar",
        "max_tokens",
        "messages",
        "model",
        "output_config",
        "stream",
        "system",
        "temperature",
        "thinking",
        "tool_choice",
        "tools",
        "top_k",
        "top_p",
    }
)


def _object(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ChatNormalizationError(f"{path} must be an object")
    if any(type(key) is not str for key in value):
        raise ChatNormalizationError(f"{path} keys must be strings")
    return cast(Mapping[str, object], value)


def _array(value: object, path: str, *, non_empty: bool = True) -> list[object]:
    if type(value) is not list:
        raise ChatNormalizationError(f"{path} must be an array")
    items = cast(list[object], value)
    if non_empty and not items:
        raise ChatNormalizationError(f"{path} cannot be empty")
    return items


def _fields(
    value: object,
    path: str,
    *,
    allowed: frozenset[str],
    required: frozenset[str],
) -> Mapping[str, object]:
    result = _object(value, path)
    unknown = tuple(sorted(set(result) - allowed))
    if unknown:
        raise ChatNormalizationError(f"{path} has unknown fields")
    missing = tuple(sorted(required - set(result)))
    if missing:
        raise ChatNormalizationError(f"{path} is missing required fields: {', '.join(missing)}")
    return result


def _string(value: object, path: str) -> str:
    if type(value) is not str:
        raise ChatNormalizationError(f"{path} must be a string")
    return value


def _identity(value: object, path: str) -> str:
    result = _string(value, path)
    if not result or result != result.strip():
        raise ChatNormalizationError(f"{path} must be a non-empty canonical string")
    return result


def _positive_integer(value: object, path: str) -> int:
    if type(value) is not int or value <= 0:
        raise ChatNormalizationError(f"{path} must be a positive integer")
    return value


def _optional_float(
    body: Mapping[str, object],
    field: str,
    *,
    upper: float,
    positive: bool = False,
) -> float | None:
    value = body.get(field)
    if value is None:
        return None
    if type(value) not in (int, float):
        raise ChatNormalizationError(f"request.{field} must be a number")
    try:
        result = float(cast(int | float, value))
    except OverflowError as error:
        raise ChatNormalizationError(f"request.{field} must be finite") from error
    lower_ok = result > 0.0 if positive else result >= 0.0
    if not math.isfinite(result) or not lower_ok or result > upper:
        interval = f"(0, {upper:g}]" if positive else f"[0, {upper:g}]"
        raise ChatNormalizationError(f"request.{field} must be finite and in {interval}")
    return result


def _optional_positive_integer(body: Mapping[str, object], field: str) -> int | None:
    value = body.get(field)
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise ChatNormalizationError(f"request.{field} must be a positive integer")
    return value


def _sampling(body: Mapping[str, object], *, anthropic: bool) -> SamplingMeta:
    seed = body.get("seed")
    if seed is not None and (type(seed) is not int or not -(2**63) <= seed < 2**63):
        raise ChatNormalizationError("request.seed must be a signed 64-bit integer")
    try:
        return SamplingMeta(
            temperature=_optional_float(body, "temperature", upper=1.0 if anthropic else 2.0),
            top_p=_optional_float(body, "top_p", upper=1.0, positive=True),
            top_k=_optional_positive_integer(body, "top_k"),
            seed=seed,
        )
    except ValueError as error:
        raise ChatNormalizationError("request sampling parameters are invalid") from error


def _stream(value: object, path: str) -> bool:
    if type(value) is not bool:
        raise ChatNormalizationError(f"{path} must be a boolean")
    return value


def _boolean(value: object, path: str) -> bool:
    if type(value) is not bool:
        raise ChatNormalizationError(f"{path} must be a boolean")
    return value


def _thinking_token_budget(value: object, path: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < -1:
        raise ChatNormalizationError(f"{path} must be a non-negative integer or -1")
    return None if value == -1 else value


def _chat_template_kwargs(body: Mapping[str, object]) -> ChatTemplateKwargs:
    raw_kwargs = body.get("chat_template_kwargs", {})
    if raw_kwargs is None:
        raw_kwargs = {}
    try:
        kwargs = ChatTemplateKwargs.from_value(raw_kwargs).as_dict()
    except (TypeError, ValueError) as error:
        raise ChatNormalizationError("request.chat_template_kwargs is invalid") from error

    effort_value = body.get("reasoning_effort")
    if effort_value is not None:
        effort = _string(effort_value, "request.reasoning_effort")
        if effort not in _OPENAI_REASONING_EFFORTS:
            supported = ", ".join(sorted(_OPENAI_REASONING_EFFORTS))
            raise ChatNormalizationError(f"request.reasoning_effort must be one of: {supported}")
        kwargs["reasoning_effort"] = effort
        if "enable_thinking" not in kwargs:
            kwargs["enable_thinking"] = effort != "none"

    return ChatTemplateKwargs.from_value(kwargs)


def _openai_stream_options(body: Mapping[str, object]) -> tuple[bool, bool, bool]:
    stream = False if body.get("stream") is None else _stream(body["stream"], "request.stream")
    raw_options = body.get("stream_options")
    if raw_options is None:
        return stream, False, False
    options = _fields(
        raw_options,
        "request.stream_options",
        allowed=frozenset({"include_usage", "continuous_usage_stats"}),
        required=frozenset(),
    )
    if options and not stream:
        raise ChatNormalizationError(
            "request.stream_options can only be defined when request.stream is true"
        )
    include_usage = (
        False
        if options.get("include_usage") is None
        else _boolean(options["include_usage"], "request.stream_options.include_usage")
    )
    continuous = _boolean(
        False if options.get("continuous_usage_stats") is None else options["continuous_usage_stats"],
        "request.stream_options.continuous_usage_stats",
    )
    if continuous and not include_usage:
        raise ChatNormalizationError("continuous_usage_stats requires include_usage")
    return stream, include_usage, continuous


def _schema(value: object, path: str) -> object:
    if type(value) is bool:
        return value
    return dict(_object(value, path))


def _function_tool(
    *,
    name: object,
    parameters: object,
    description: object | None,
    strict: object | None,
    path: str,
) -> dict[str, object]:
    function: dict[str, object] = {
        "name": _identity(name, f"{path}.name"),
        "parameters": _schema(parameters, f"{path}.parameters"),
    }
    if description is not None:
        function["description"] = _string(description, f"{path}.description")
    # OpenAI tools default to non-strict.  Keep that choice explicit because
    # xgrammar uses ``strict=false`` to constrain only the model's tool-call
    # framing while leaving the argument object unconstrained by the schema.
    function["strict"] = False if strict is None else _boolean(strict, f"{path}.strict")
    return {"type": "function", "function": function}


def _openai_tools(value: object) -> list[dict[str, object]]:
    tools: list[dict[str, object]] = []
    for index, raw_tool in enumerate(_array(value, "request.tools")):
        path = f"request.tools[{index}]"
        tool = _fields(
            raw_tool,
            path,
            allowed=frozenset({"function", "type"}),
            required=frozenset({"function", "type"}),
        )
        if _string(tool["type"], f"{path}.type") != "function":
            raise ChatNormalizationError(f"{path}.type must be 'function'")
        function_path = f"{path}.function"
        function = _fields(
            tool["function"],
            function_path,
            allowed=frozenset({"description", "name", "parameters", "strict"}),
            required=frozenset({"name"}),
        )
        tools.append(
            _function_tool(
                name=function["name"],
                parameters=(
                    {"type": "object", "properties": {}}
                    if function.get("parameters") is None or function.get("parameters") == {}
                    else function["parameters"]
                ),
                description=function.get("description"),
                strict=function.get("strict"),
                path=function_path,
            )
        )
    _require_unique_tool_names(tools)
    return tools


def _anthropic_tools(value: object) -> list[dict[str, object]]:
    tools: list[dict[str, object]] = []
    for index, raw_tool in enumerate(_array(value, "request.tools")):
        path = f"request.tools[{index}]"
        tool = _fields(
            raw_tool,
            path,
            allowed=frozenset({"description", "input_schema", "name", "strict"}),
            required=frozenset({"input_schema", "name"}),
        )
        tools.append(
            _function_tool(
                name=tool["name"],
                parameters=tool["input_schema"],
                description=tool.get("description"),
                strict=tool.get("strict"),
                path=path,
            )
        )
    _require_unique_tool_names(tools)
    return tools


def _require_unique_tool_names(tools: list[dict[str, object]]) -> None:
    names = tuple(cast(dict[str, object], tool["function"])["name"] for tool in tools)
    if len(set(names)) != len(names):
        raise ChatNormalizationError("request.tools function names must be unique")


def _select_tools(
    tools: list[dict[str, object]],
    *,
    choice: object | None,
    anthropic: bool,
    parallel_tool_calls: bool = True,
) -> StructuredOutputMeta:
    if not tools:
        disabled = choice == "none" or (
            anthropic and type(choice) is dict and choice == {"type": "none"}
        )
        if choice is not None and not disabled:
            raise ChatNormalizationError("request.tool_choice requires request.tools")
        return StructuredOutputMeta.identity()

    selected = tools
    if choice is None:
        mode = "auto"
    elif type(choice) is str:
        wire_choice = _string(choice, "request.tool_choice")
        if wire_choice == "none":
            return StructuredOutputMeta.identity()
        if anthropic:
            raise ChatNormalizationError("Anthropic request.tool_choice must be an object")
        if wire_choice not in ("auto", "required"):
            raise ChatNormalizationError("request.tool_choice is unsupported")
        mode = wire_choice
    else:
        tool_choice = _object(choice, "request.tool_choice")
        if not anthropic and tool_choice.get("type") == "allowed_tools":
            parsed = _fields(
                tool_choice,
                "request.tool_choice",
                allowed=frozenset({"type", "allowed_tools"}),
                required=frozenset({"type", "allowed_tools"}),
            )
            parsed = _fields(
                parsed["allowed_tools"],
                "request.tool_choice.allowed_tools",
                allowed=frozenset({"mode", "tools"}),
                required=frozenset({"mode", "tools"}),
            )
            mode = _string(parsed["mode"], "request.tool_choice.allowed_tools.mode")
            if mode not in ("auto", "required"):
                raise ChatNormalizationError("request.tool_choice.allowed_tools.mode is unsupported")
            selected = []
            for item in _array(parsed["tools"], "request.tool_choice.allowed_tools.tools"):
                reference = _fields(
                    item,
                    "request.tool_choice.allowed_tools.tools[]",
                    allowed=frozenset({"type", "function"}),
                    required=frozenset({"type", "function"}),
                )
                if reference["type"] != "function":
                    raise ChatNormalizationError("allowed tools must be function references")
                function = _fields(
                    reference["function"],
                    "request.tool_choice.allowed_tools.tools[].function",
                    allowed=frozenset({"name"}),
                    required=frozenset({"name"}),
                )
                selected.extend(_named_tool(tools, function["name"]))
            _require_unique_tool_names(selected)
            return StructuredOutputMeta.tool_schema(
                selected, choice=mode, parallel_tool_calls=parallel_tool_calls
            )
        if anthropic:
            parsed = _fields(
                tool_choice,
                "request.tool_choice",
                allowed=frozenset({"name", "type"}),
                required=frozenset({"type"}),
            )
            kind = _string(parsed["type"], "request.tool_choice.type")
            if kind == "none" and "name" not in parsed:
                return StructuredOutputMeta.identity()
            if kind in ("auto", "any") and "name" not in parsed:
                mode = "auto" if kind == "auto" else "required"
            elif kind == "tool" and "name" in parsed:
                mode = "named"
                selected = _named_tool(tools, parsed["name"])
            else:
                raise ChatNormalizationError("Anthropic request.tool_choice is unsupported")
        else:
            parsed = _fields(
                tool_choice,
                "request.tool_choice",
                allowed=frozenset({"function", "type"}),
                required=frozenset({"function", "type"}),
            )
            if _string(parsed["type"], "request.tool_choice.type") != "function":
                raise ChatNormalizationError("OpenAI named tool choice type must be 'function'")
            function = _fields(
                parsed["function"],
                "request.tool_choice.function",
                allowed=frozenset({"name"}),
                required=frozenset({"name"}),
            )
            mode = "named"
            selected = _named_tool(tools, function["name"])
    return StructuredOutputMeta.tool_schema(
        selected,
        choice=mode,
        parallel_tool_calls=parallel_tool_calls,
    )


def _tool_choice_settings(
    body: Mapping[str, object], *, anthropic: bool
) -> tuple[object | None, bool]:
    choice = body.get("tool_choice")
    if not anthropic:
        return choice, _boolean(
            True if body.get("parallel_tool_calls") is None else body["parallel_tool_calls"],
            "request.parallel_tool_calls",
        )
    if choice is None:
        return None, True
    parsed = _fields(
        choice, "request.tool_choice",
        allowed=frozenset({"type", "name", "disable_parallel_tool_use"}),
        required=frozenset({"type"}),
    )
    disable = parsed.get("disable_parallel_tool_use")
    parallel = True if disable is None else not _boolean(
        disable, "request.tool_choice.disable_parallel_tool_use"
    )
    return {key: value for key, value in parsed.items()
            if key != "disable_parallel_tool_use" and value is not None}, parallel


def _named_tool(tools: list[dict[str, object]], name: object) -> list[dict[str, object]]:
    selected_name = _identity(name, "request.tool_choice.name")
    result = [
        tool for tool in tools if cast(dict[str, object], tool["function"])["name"] == selected_name
    ]
    if not result:
        raise ChatNormalizationError("request.tool_choice names an absent tool")
    return result


def _openai_response_format(value: object) -> StructuredOutputMeta:
    path = "request.response_format"
    raw = _object(value, path)
    kind = _string(raw.get("type"), f"{path}.type")
    if kind == "text":
        _fields(raw, path, allowed=frozenset({"type"}), required=frozenset({"type"}))
        return StructuredOutputMeta.identity()
    if kind == "json_object":
        _fields(raw, path, allowed=frozenset({"type"}), required=frozenset({"type"}))
        return StructuredOutputMeta.json_schema({"type": "object"})
    if kind == "structural_tag":
        return _structural_tag(raw, path)
    if kind != "json_schema":
        raise ChatNormalizationError(f"{path}.type is unsupported")
    response = _fields(
        raw,
        path,
        allowed=frozenset({"json_schema", "type"}),
        required=frozenset({"json_schema", "type"}),
    )
    wrapper = _fields(
        response["json_schema"],
        f"{path}.json_schema",
        allowed=frozenset({"description", "name", "schema", "strict"}),
        required=frozenset({"name", "schema"}),
    )
    _identity(wrapper["name"], f"{path}.json_schema.name")
    if wrapper.get("description") is not None:
        _string(wrapper["description"], f"{path}.json_schema.description")
    if wrapper.get("strict") is not None:
        _boolean(wrapper["strict"], f"{path}.json_schema.strict")
    return StructuredOutputMeta.json_schema(
        _schema(wrapper["schema"], f"{path}.json_schema.schema")
    )


def _structural_tag(value: object, path: str) -> StructuredOutputMeta:
    raw = _object(value, path)
    if "format" in raw:
        parsed = _fields(
            raw, path, allowed=frozenset({"type", "format"}), required=frozenset({"type", "format"})
        )
        _object(parsed["format"], f"{path}.format")
    else:
        parsed = _fields(
            raw,
            path,
            allowed=frozenset({"type", "structures", "triggers"}),
            required=frozenset({"type", "structures", "triggers"}),
        )
        for index, structure in enumerate(_array(parsed["structures"], f"{path}.structures")):
            item_path = f"{path}.structures[{index}]"
            item = _fields(
                structure,
                item_path,
                allowed=frozenset({"begin", "schema", "end"}),
                required=frozenset({"begin", "schema", "end"}),
            )
            _string(item["begin"], f"{item_path}.begin")
            _string(item["end"], f"{item_path}.end")
            _schema(item["schema"], f"{item_path}.schema")
        for trigger in _array(parsed["triggers"], f"{path}.triggers"):
            _string(trigger, f"{path}.triggers")
    if parsed["type"] != "structural_tag":
        raise ChatNormalizationError(f"{path}.type must be 'structural_tag'")
    return StructuredOutputMeta.structural_tag(dict(parsed))


def _openai_structured_outputs(value: object) -> StructuredOutputMeta:
    path = "request.structured_outputs"
    selectors = frozenset({"json", "json_object", "regex", "choice", "grammar", "structural_tag"})
    option_names = frozenset(
        {"disable_any_whitespace", "disable_additional_properties", "whitespace_pattern"}
    )
    raw = _fields(value, path, allowed=selectors | option_names, required=frozenset())
    selected = [name for name in selectors if raw.get(name) is not None]
    if len(selected) != 1:
        raise ChatNormalizationError(f"{path} requires exactly one constraint")
    options: dict[str, object] = {}
    for name in option_names:
        if raw.get(name) is None:
            continue
        option = raw[name]
        if name == "whitespace_pattern":
            options[name] = _string(option, f"{path}.{name}")
        elif _boolean(option, f"{path}.{name}"):
            options[name] = True
    name = selected[0]
    source = raw[name]
    if name in ("json", "json_object"):
        if name == "json_object":
            if _boolean(source, f"{path}.json_object") is not True:
                raise ChatNormalizationError(f"{path}.json_object must be true")
            source = {"type": "object"}
        return StructuredOutputMeta.json_schema(source, options=options)
    if options:
        raise ChatNormalizationError(f"{path} JSON formatting options require json or json_object")
    if name == "regex":
        return StructuredOutputMeta.regex(_string(source, f"{path}.regex"))
    if name == "choice":
        choices = _array(source, f"{path}.choice")
        return StructuredOutputMeta.choice([_string(x, f"{path}.choice") for x in choices])
    if name == "grammar":
        return _openai_grammar(_string(source, f"{path}.grammar"))
    # Both JSON text and an object are accepted by vLLM's tag validator.
    meta = StructuredOutputMeta.structural_tag(source)
    tag = json.loads(meta.canonical_payload)
    tag.setdefault("type", "structural_tag")
    return _structural_tag(tag, f"{path}.structural_tag")


def _openai_grammar(source: str) -> StructuredOutputMeta:
    # Match the upstream syntax discriminator; actual parsing belongs to XGrammar.
    # Comments cannot introduce a spurious EBNF definition.
    ebnf = any("::=" in re.sub(r"(#|//).*$", "", line) for line in source.splitlines())
    return StructuredOutputMeta.grammar(source, dialect="xgrammar-ebnf" if ebnf else "lark")


def _anthropic_output_config(value: object) -> StructuredOutputMeta:
    config = _fields(
        value,
        "request.output_config",
        allowed=frozenset({"format", "effort"}),
        required=frozenset(),
    )
    if config.get("effort") is not None:
        effort = _string(config["effort"], "request.output_config.effort")
        if effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ChatNormalizationError("request.output_config.effort is unsupported")
    if config.get("format") is None:
        return StructuredOutputMeta.identity()
    format_ = _fields(
        config["format"],
        "request.output_config.format",
        allowed=frozenset({"schema", "type"}),
        required=frozenset({"schema", "type"}),
    )
    if _string(format_["type"], "request.output_config.format.type") != "json_schema":
        raise ChatNormalizationError("request.output_config.format.type is unsupported")
    return StructuredOutputMeta.json_schema(
        _schema(format_["schema"], "request.output_config.format.schema")
    )


def _structured_output(body: Mapping[str, object], *, anthropic: bool) -> StructuredOutputMeta:
    try:
        return _structured_output_value(body, anthropic=anthropic)
    except ChatNormalizationError:
        raise
    except (TypeError, ValueError) as error:
        raise ChatNormalizationError("request structured output is invalid") from error


def _structured_output_value(
    body: Mapping[str, object], *, anthropic: bool
) -> StructuredOutputMeta:
    raw_tools = body.get("tools")
    tools = (
        [] if raw_tools is None
        else _anthropic_tools(raw_tools) if anthropic else _openai_tools(raw_tools)
    )
    choice, parallel = _tool_choice_settings(body, anthropic=anthropic)
    tool_meta = _select_tools(
        tools,
        choice=choice,
        anthropic=anthropic,
        parallel_tool_calls=parallel,
    )
    candidates = [] if tool_meta.kind.value == "identity" else [tool_meta]
    structured = (
        _openai_structured_outputs(body["structured_outputs"])
        if not anthropic and body.get("structured_outputs") is not None
        else None
    )
    format_field = "output_config" if anthropic else "response_format"
    if body.get(format_field) is not None:
        meta = (
            _anthropic_output_config(body[format_field])
            if anthropic
            else _openai_response_format(body[format_field])
        )
        if meta.kind.value != "identity":
            if structured is not None:
                if meta.kind is not structured.kind:
                    raise ChatNormalizationError(
                        "request has conflicting structured output constraints"
                    )
                # response_format overrides the same selector and retains options.
                meta = StructuredOutputMeta(
                    meta.kind, meta.canonical_payload, structured.canonical_options
                )
            structured = meta
    if structured is not None and structured.kind.value != "identity":
        candidates.append(structured)
    if "grammar" in body:
        source = _string(body["grammar"], "request.grammar")
        candidates.append(StructuredOutputMeta.grammar(source))
    if tool_meta.kind is StructuredOutputKind.TOOL_SCHEMA and len(candidates) == 2:
        answer = candidates[1]
        payload = json.loads(tool_meta.canonical_payload)
        if payload["choice"] == "auto" or answer.kind is StructuredOutputKind.STRUCTURAL_TAG:
            return StructuredOutputMeta.tool_schema(
                payload["tools"],
                choice=payload["choice"],
                response=answer,
                parallel_tool_calls=payload.get("parallel_tool_calls", True),
            )
        # vLLM gives forced tool choice precedence over an answer format.
        return tool_meta
    if len(candidates) > 1:
        raise ChatNormalizationError("request has conflicting structured output constraints")
    return candidates[0] if candidates else StructuredOutputMeta.identity()


def _inline_source(
    *,
    modality: str,
    mime_type: object,
    data: object,
    path: str,
) -> MediaSource:
    raw_mime_type = _string(mime_type, f"{path}.media_type")
    if raw_mime_type != raw_mime_type.strip():
        raise ChatNormalizationError(f"{path}.media_type cannot contain surrounding whitespace")
    canonical_mime_type = raw_mime_type.lower()
    if not canonical_mime_type.startswith(f"{modality}/"):
        raise ChatNormalizationError(f"{path}.media_type must describe {modality} content")
    raw_data = _string(data, f"{path}.data")
    if len(raw_data) > MAX_INLINE_MEDIA_BASE64_CHARS:
        raise ChatNormalizationError(
            f"{path}.data exceeds the {MAX_INLINE_MEDIA_BASE64_CHARS}-character limit"
        )
    try:
        return MediaSource(
            kind=MediaSourceKind.BASE64,
            mime_type=canonical_mime_type,
            base64_data=raw_data,
        )
    except (TypeError, ValueError) as error:
        raise ChatNormalizationError(f"{path} has invalid inline media") from error


def _url_source(*, modality: str, uri: object, path: str) -> MediaSource:
    raw_uri = _string(uri, path)
    if raw_uri.startswith("data:"):
        header, separator, data = raw_uri.partition(",")
        if not separator or not header.endswith(";base64"):
            raise ChatNormalizationError(f"{path} data URI must use data:<mime>;base64,<payload>")
        mime_type = header[len("data:") : -len(";base64")]
        return _inline_source(
            modality=modality,
            mime_type=mime_type,
            data=data,
            path=path,
        )
    try:
        return MediaSource(kind=MediaSourceKind.URL, uri=raw_uri)
    except (TypeError, ValueError) as error:
        raise ChatNormalizationError(f"{path} has an invalid media URL") from error


def _text_part(text: object, ordinal: int, path: str) -> ChatContentPart:
    return ChatContentPart(
        ordinal=ordinal,
        modality="text",
        text=_string(text, path),
    )


def _openai_content(
    value: object,
    *,
    ordinal: int,
    path: str,
) -> tuple[tuple[ChatContentPart, ...], int]:
    if type(value) is str:
        return (_text_part(value, ordinal, path),), ordinal + 1

    items = _array(value, path)
    parts: list[ChatContentPart] = []
    for index, raw_part in enumerate(items):
        part_path = f"{path}[{index}]"
        untyped_part = _object(raw_part, part_path)
        part_type = _string(untyped_part.get("type"), f"{part_path}.type")
        if part_type == "text":
            part = _fields(
                raw_part,
                part_path,
                allowed=frozenset({"text", "type"}),
                required=frozenset({"text", "type"}),
            )
            parts.append(_text_part(part["text"], ordinal, f"{part_path}.text"))
        elif part_type in ("image_url", "video_url", "audio_url"):
            field_name = part_type
            modality = part_type.removesuffix("_url")
            part = _fields(
                raw_part,
                part_path,
                allowed=frozenset({field_name, "type"}),
                required=frozenset({field_name, "type"}),
            )
            locator_path = f"{part_path}.{field_name}"
            locator = _fields(
                part[field_name],
                locator_path,
                allowed=frozenset({"url"}),
                required=frozenset({"url"}),
            )
            parts.append(
                ChatContentPart(
                    ordinal=ordinal,
                    modality=modality,
                    media=_url_source(
                        modality=modality,
                        uri=locator["url"],
                        path=f"{locator_path}.url",
                    ),
                )
            )
        else:
            raise ChatNormalizationError(f"{part_path}.type is unsupported")
        ordinal += 1
    return tuple(parts), ordinal


def _anthropic_source(
    value: object,
    *,
    modality: str,
    path: str,
) -> MediaSource:
    untyped_source = _object(value, path)
    source_type = _string(untyped_source.get("type"), f"{path}.type")
    if source_type == "base64":
        source = _fields(
            value,
            path,
            allowed=frozenset({"data", "media_type", "type"}),
            required=frozenset({"data", "media_type", "type"}),
        )
        return _inline_source(
            modality=modality,
            mime_type=source["media_type"],
            data=source["data"],
            path=path,
        )
    if source_type == "url":
        source = _fields(
            value,
            path,
            allowed=frozenset({"type", "url"}),
            required=frozenset({"type", "url"}),
        )
        return _url_source(
            modality=modality,
            uri=source["url"],
            path=f"{path}.url",
        )
    raise ChatNormalizationError(f"{path}.type is unsupported")


def _anthropic_content(
    value: object,
    *,
    ordinal: int,
    path: str,
    system_only: bool = False,
) -> tuple[tuple[ChatContentPart, ...], int]:
    if type(value) is str:
        return (_text_part(value, ordinal, path),), ordinal + 1

    items = _array(value, path)
    parts: list[ChatContentPart] = []
    for index, raw_part in enumerate(items):
        part_path = f"{path}[{index}]"
        untyped_part = _object(raw_part, part_path)
        part_type = _string(untyped_part.get("type"), f"{part_path}.type")
        if part_type == "text":
            part = _fields(
                raw_part,
                part_path,
                allowed=frozenset({"text", "type"}),
                required=frozenset({"text", "type"}),
            )
            parts.append(_text_part(part["text"], ordinal, f"{part_path}.text"))
        elif not system_only and part_type in ("image", "video"):
            part = _fields(
                raw_part,
                part_path,
                allowed=frozenset({"source", "type"}),
                required=frozenset({"source", "type"}),
            )
            parts.append(
                ChatContentPart(
                    ordinal=ordinal,
                    modality=part_type,
                    media=_anthropic_source(
                        part["source"],
                        modality=part_type,
                        path=f"{part_path}.source",
                    ),
                )
            )
        else:
            raise ChatNormalizationError(f"{part_path}.type is unsupported")
        ordinal += 1
    return tuple(parts), ordinal


def _canonical_request(
    *,
    request_id: str,
    model_id: str,
    messages: list[ChatMessage],
    max_new_tokens: int | None,
    structured_output: StructuredOutputMeta,
    sampling: SamplingMeta,
    chat_template_kwargs: ChatTemplateKwargs | None = None,
    stream_include_usage: bool = False,
    stream_continuous_usage_stats: bool = False,
    ignore_eos: bool = False,
    tools: ToolDefinitions | None = None,
    parallel_tool_calls: bool = True,
    include_reasoning: bool = True,
    thinking_token_budget: int | None = None,
) -> CanonicalChatRequest:
    try:
        return CanonicalChatRequest(
            request_id=request_id,
            model_id=model_id,
            messages=tuple(messages),
            max_new_tokens=max_new_tokens,
            structured_output=structured_output,
            chat_template_kwargs=chat_template_kwargs or ChatTemplateKwargs(),
            sampling=sampling,
            stream_include_usage=stream_include_usage,
            stream_continuous_usage_stats=stream_continuous_usage_stats,
            ignore_eos=ignore_eos,
            tools=tools or ToolDefinitions(),
            parallel_tool_calls=parallel_tool_calls,
            include_reasoning=include_reasoning,
            thinking_token_budget=thinking_token_budget,
        )
    except (TypeError, ValueError) as error:
        raise ChatNormalizationError("request content is invalid") from error


def normalize_openai_chat_completions(
    payload: Mapping[str, object],
    *,
    request_id: str,
) -> tuple[CanonicalChatRequest, bool]:
    """Normalize one strict OpenAI ``/v1/chat/completions`` JSON body."""

    body = _fields(
        payload,
        "request",
        allowed=_OPENAI_TOP_LEVEL_FIELDS,
        required=frozenset({"messages", "model"}),
    )
    limits = {
        field: _positive_integer(body[field], f"request.{field}")
        for field in ("max_completion_tokens", "max_tokens")
        if body.get(field) is not None
    }
    max_new_tokens = limits.get("max_completion_tokens", limits.get("max_tokens"))
    model_id = _identity(body["model"], "request.model")
    canonical_request_id = _identity(request_id, "request_id")
    structured_output = _structured_output(body, anthropic=False)
    sampling = _sampling(body, anthropic=False)
    chat_template_kwargs = _chat_template_kwargs(body)
    stream, include_usage, continuous_usage = _openai_stream_options(body)
    raw_messages = _array(body["messages"], "request.messages")

    messages: list[ChatMessage] = []
    ordinal = 0
    for index, raw_message in enumerate(raw_messages):
        message_path = f"request.messages[{index}]"
        message = _fields(
            raw_message,
            message_path,
            allowed=frozenset(
                {
                    "content",
                    "role",
                    "tool_calls",
                    "tool_call_id",
                    "reasoning",
                    "reasoning_content",
                    "name",
                }
            ),
            required=frozenset({"role"}),
        )
        role_value = _string(message["role"], f"{message_path}.role")
        try:
            role = ChatRole(role_value)
        except ValueError as error:
            raise ChatNormalizationError(f"{message_path}.role is unsupported") from error
        calls: list[ChatToolCall] = []
        if message.get("tool_calls") is not None:
            for call_index, raw_call in enumerate(
                _array(message["tool_calls"], f"{message_path}.tool_calls", non_empty=False)
            ):
                call_path = f"{message_path}.tool_calls[{call_index}]"
                call = _fields(
                    raw_call,
                    call_path,
                    allowed=frozenset({"type", "id", "function"}),
                    required=frozenset({"id", "function"}),
                )
                if call.get("type", "function") != "function":
                    raise ChatNormalizationError(f"{call_path}.type must be 'function'")
                function = _fields(
                    call["function"],
                    f"{call_path}.function",
                    allowed=frozenset({"name", "arguments"}),
                    required=frozenset({"name"}),
                )
                arguments = function.get("arguments")
                try:
                    calls.append(
                        ChatToolCall.from_arguments(
                            _identity(call["id"], f"{call_path}.id"),
                            _identity(function["name"], f"{call_path}.function.name"),
                            {} if arguments is None or arguments == "" else arguments,
                        )
                    )
                except (TypeError, ValueError) as error:
                    raise ChatNormalizationError(
                        "request historical tool call is invalid"
                    ) from error
        reasoning = message.get("reasoning", message.get("reasoning_content"))
        if reasoning is not None:
            reasoning = _string(reasoning, f"{message_path}.reasoning")
        if (
            message.get("reasoning") is not None
            and message.get("reasoning_content") is not None
            and message["reasoning"] != message["reasoning_content"]
        ):
            raise ChatNormalizationError("request assistant reasoning fields conflict")
        raw_content = message.get("content")
        if raw_content is None:
            if role is not ChatRole.ASSISTANT:
                raise ChatNormalizationError(f"{message_path}.content is required")
            raw_content = ""
        content, ordinal = _openai_content(
            raw_content,
            ordinal=ordinal,
            path=f"{message_path}.content",
        )
        try:
            call_id = message.get("tool_call_id")
            name = message.get("name")
            messages.append(
                ChatMessage(
                    role=role,
                    content_parts=content,
                    tool_calls=tuple(calls),
                    tool_call_id=None
                    if call_id is None
                    else _identity(call_id, f"{message_path}.tool_call_id"),
                    reasoning=reasoning,
                    name=None if name is None else _identity(name, f"{message_path}.name"),
                )
            )
        except (TypeError, ValueError) as error:
            raise ChatNormalizationError("request message metadata is invalid") from error

    return (
        _canonical_request(
            request_id=canonical_request_id,
            model_id=model_id,
            messages=messages,
            max_new_tokens=max_new_tokens,
            structured_output=structured_output,
            sampling=sampling,
            chat_template_kwargs=chat_template_kwargs,
            stream_include_usage=include_usage,
            stream_continuous_usage_stats=continuous_usage,
            ignore_eos=_boolean(body.get("ignore_eos", False), "request.ignore_eos"),
            tools=ToolDefinitions.from_value(
                _openai_tools(body["tools"]) if body.get("tools") is not None else []
            ),
            parallel_tool_calls=_tool_choice_settings(body, anthropic=False)[1],
            include_reasoning=_boolean(
                body.get("include_reasoning", True), "request.include_reasoning"
            ),
            thinking_token_budget=_thinking_token_budget(
                body.get("thinking_token_budget"), "request.thinking_token_budget"
            ),
        ),
        stream,
    )


def _anthropic_history_message(
    role: ChatRole,
    value: object,
    *,
    ordinal: int,
    path: str,
) -> tuple[list[ChatMessage], int]:
    if type(value) is not list:
        content, ordinal = _anthropic_content(value, ordinal=ordinal, path=path)
        return [ChatMessage(role, content)], ordinal
    regular: list[object] = []
    calls: list[ChatToolCall] = []
    reasoning: list[str] = []
    messages: list[ChatMessage] = []

    def flush_user() -> None:
        nonlocal ordinal
        if regular:
            content, ordinal = _anthropic_content(regular, ordinal=ordinal, path=path)
            messages.append(ChatMessage(ChatRole.USER, content))
            regular.clear()

    for index, raw in enumerate(_array(value, path)):
        item_path = f"{path}[{index}]"
        item = _object(raw, item_path)
        kind = item.get("type")
        if kind == "tool_result" and role is ChatRole.USER:
            item = _fields(
                item,
                item_path,
                allowed=frozenset({"type", "tool_use_id", "content", "is_error"}),
                required=frozenset({"type", "tool_use_id"}),
            )
            if "is_error" in item:
                _boolean(item["is_error"], f"{item_path}.is_error")
            flush_user()
            result = item.get("content", "")
            content, ordinal = _anthropic_content(
                "" if result is None or result == [] else result, ordinal=ordinal, path=f"{item_path}.content"
            )
            messages.append(
                ChatMessage(
                    ChatRole.TOOL,
                    content,
                    tool_call_id=_identity(item["tool_use_id"], f"{item_path}.tool_use_id"),
                )
            )
        elif kind == "tool_use" and role is ChatRole.ASSISTANT:
            item = _fields(
                item,
                item_path,
                allowed=frozenset({"type", "id", "name", "input"}),
                required=frozenset({"type", "id", "name"}),
            )
            try:
                calls.append(
                    ChatToolCall.from_arguments(
                        _identity(item["id"], f"{item_path}.id"),
                        _identity(item["name"], f"{item_path}.name"),
                        dict(_object(item.get("input", {}), f"{item_path}.input")),
                    )
                )
            except (TypeError, ValueError) as error:
                raise ChatNormalizationError(f"{item_path}.input must be a JSON object") from error
        elif kind == "thinking" and role is ChatRole.ASSISTANT:
            item = _fields(
                item,
                item_path,
                allowed=frozenset({"type", "thinking", "signature"}),
                required=frozenset({"type", "thinking"}),
            )
            reasoning.append(_string(item["thinking"], f"{item_path}.thinking"))
        elif kind == "redacted_thinking" and role is ChatRole.ASSISTANT:
            # Like vLLM, accept echoed encrypted history without interpreting it.
            item = _fields(
                item,
                item_path,
                allowed=frozenset({"type", "data"}),
                required=frozenset({"type", "data"}),
            )
            _string(item["data"], f"{item_path}.data")
        else:
            regular.append(raw)
    if role is ChatRole.USER:
        flush_user()
    else:
        content, ordinal = _anthropic_content(regular or "", ordinal=ordinal, path=path)
        messages.append(
            ChatMessage(
                role,
                content,
                tool_calls=tuple(calls),
                reasoning="".join(reasoning) if reasoning else None,
            )
        )
    return messages, ordinal


def normalize_anthropic_messages(
    payload: Mapping[str, object],
    *,
    request_id: str,
) -> tuple[CanonicalChatRequest, bool]:
    """Normalize one strict Anthropic ``/v1/messages`` JSON body.

    Cubic Serve deliberately accepts ``video`` blocks with the same ``source``
    shapes as Anthropic image blocks. This is an explicit Cubic extension, not
    an inference from an unknown block type.
    """

    body = _fields(
        payload,
        "request",
        allowed=_ANTHROPIC_TOP_LEVEL_FIELDS,
        required=frozenset({"max_tokens", "messages", "model"}),
    )
    model_id = _identity(body["model"], "request.model")
    canonical_request_id = _identity(request_id, "request_id")
    max_new_tokens = _positive_integer(body["max_tokens"], "request.max_tokens")
    structured_output = _structured_output(body, anthropic=True)
    sampling = _sampling(body, anthropic=True)
    output_config = body.get("output_config")
    effort = output_config.get("effort") if isinstance(output_config, Mapping) else None
    budget = None
    if body.get("thinking") is not None:
        thinking = _object(body["thinking"], "request.thinking")
        kind = _string(thinking.get("type"), "request.thinking.type")
        if kind not in ("disabled", "adaptive", "enabled"):
            raise ChatNormalizationError("request.thinking.type is unsupported")
        allowed = {"type"} if kind == "disabled" else {"type", "display"}
        if kind == "enabled":
            allowed.add("budget_tokens")
        thinking = _fields(thinking, "request.thinking", allowed=frozenset(allowed),
                           required=frozenset({"type", "budget_tokens"}
                                              if kind == "enabled" else {"type"}))
        if thinking.get("display") is not None and thinking["display"] not in (
            "summarized", "omitted", "updates",
        ):
            raise ChatNormalizationError("request.thinking.display is unsupported")
        if kind == "disabled":
            effort = "none"
        elif kind == "enabled":
            value = thinking["budget_tokens"]
            if type(value) is not int or value < 1024:
                raise ChatNormalizationError("request.thinking.budget_tokens must be at least 1024")
            budget = value
    chat_template_kwargs = _chat_template_kwargs({
        "chat_template_kwargs": body.get("chat_template_kwargs"), "reasoning_effort": effort,
    })
    raw_messages = _array(body["messages"], "request.messages")

    messages: list[ChatMessage] = []
    ordinal = 0
    if "system" in body:
        content, ordinal = _anthropic_content(
            body["system"],
            ordinal=ordinal,
            path="request.system",
            system_only=True,
        )
        messages.append(ChatMessage(role=ChatRole.SYSTEM, content_parts=content))

    for index, raw_message in enumerate(raw_messages):
        message_path = f"request.messages[{index}]"
        message = _fields(
            raw_message,
            message_path,
            allowed=frozenset({"content", "role"}),
            required=frozenset({"content", "role"}),
        )
        role_value = _string(message["role"], f"{message_path}.role")
        if role_value not in (ChatRole.USER.value, ChatRole.ASSISTANT.value):
            raise ChatNormalizationError(f"{message_path}.role is unsupported")
        normalized, ordinal = _anthropic_history_message(
            ChatRole(role_value),
            message["content"],
            ordinal=ordinal,
            path=f"{message_path}.content",
        )
        messages.extend(normalized)

    return (
        _canonical_request(
            request_id=canonical_request_id,
            model_id=model_id,
            messages=messages,
            max_new_tokens=max_new_tokens,
            structured_output=structured_output,
            sampling=sampling,
            chat_template_kwargs=chat_template_kwargs,
            tools=ToolDefinitions.from_value(
                _anthropic_tools(body["tools"]) if body.get("tools") is not None else []
            ),
            parallel_tool_calls=_tool_choice_settings(body, anthropic=True)[1],
            thinking_token_budget=budget,
        ),
        False if "stream" not in body else _stream(body["stream"], "request.stream"),
    )


__all__ = (
    "MAX_MEDIA_BYTES",
    "MAX_INLINE_MEDIA_BASE64_CHARS",
    "ChatNormalizationError",
    "normalize_anthropic_messages",
    "normalize_openai_chat_completions",
)
