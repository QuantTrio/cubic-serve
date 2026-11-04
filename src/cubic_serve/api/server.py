"""Minimal dependency-free ASGI boundary over canonical chat events."""

from __future__ import annotations

from typing_extensions import Never, TypeVar

import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from cubic_serve._compat import StrEnum
from typing import Protocol, cast

from cubic_serve.api._events import FoldedEvents, ProjectionError, fold_events
from cubic_serve.api.anthropic import project_anthropic_message
from cubic_serve.api.openai import (
    project_openai_chat_completion,
    project_openai_usage,
)
from cubic_serve.contracts.request import (
    CanonicalChatRequest,
    TokenEvent,
    TokenEventKind,
)
from cubic_serve.inputs.normalize import (
    ChatNormalizationError,
    normalize_anthropic_messages,
    normalize_openai_chat_completions,
)

WireObject = dict[str, object]
ASGIMessage = dict[str, object]
ASGIScope = Mapping[str, object]
ASGIReceive = Callable[[], Awaitable[ASGIMessage]]
ASGISend = Callable[[ASGIMessage], Awaitable[None]]
MAX_JSON_BODY_BYTES = 2 * 1024 * 1024
_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)


@dataclass(frozen=True, slots=True)
class ChatExecution:
    """An admitted request; terminal USAGE must repeat ``input_tokens`` exactly."""

    input_tokens: int
    events: AsyncIterator[TokenEvent]

    def __post_init__(self) -> None:
        if type(self.input_tokens) is not int or self.input_tokens < 0:
            raise ValueError("execution input_tokens must be a non-negative integer")
        if not hasattr(self.events, "__anext__"):
            raise TypeError("execution events must be an async iterator")


class ChatBackend(Protocol):
    """Admission/cancellation boundary implemented by the real request executor."""

    async def start(self, request: CanonicalChatRequest) -> ChatExecution: ...

    async def cancel(self, request_id: str) -> None: ...


class BackendErrorKind(StrEnum):
    INVALID_REQUEST = "invalid_request"
    MODEL_NOT_FOUND = "model_not_found"
    OVERLOADED = "overloaded"
    UNAVAILABLE = "unavailable"


_BACKEND_STATUS = {
    BackendErrorKind.INVALID_REQUEST: 400,
    BackendErrorKind.MODEL_NOT_FOUND: 404,
    BackendErrorKind.OVERLOADED: 429,
    BackendErrorKind.UNAVAILABLE: 503,
}

_DEFAULT_PUBLIC_BACKEND_MESSAGE = {
    BackendErrorKind.INVALID_REQUEST: "invalid request",
    BackendErrorKind.MODEL_NOT_FOUND: "model not found",
    BackendErrorKind.OVERLOADED: "The model is currently at capacity. Please try again later.",
    BackendErrorKind.UNAVAILABLE: "service temporarily unavailable",
}
_REVIEWED_PUBLIC_BACKEND_MESSAGES = frozenset({
    *_DEFAULT_PUBLIC_BACKEND_MESSAGE.values(),
    "thinking_token_budget is unavailable for this model",
    "thinking_token_budget conflicts with structured_outputs",
    "request input could not be processed",
    "media input is too large",
    "image or video frame is too large",
    "request exceeds available context length",
})


class ChatBackendError(RuntimeError):
    def __init__(self, kind: BackendErrorKind, message: str, *,
                 public_message: str | None = None) -> None:
        if not isinstance(kind, BackendErrorKind):
            raise TypeError("backend error kind must be a BackendErrorKind")
        if type(message) is not str or not message or message != message.strip():
            raise ValueError("backend error message must be canonical and non-empty")
        if public_message is not None and public_message not in _REVIEWED_PUBLIC_BACKEND_MESSAGES:
            raise ValueError("backend public error message must be reviewed")
        super().__init__(message)
        self.kind = kind
        self.public_message = public_message or _DEFAULT_PUBLIC_BACKEND_MESSAGE[kind]

    @property
    def status(self) -> int:
        return _BACKEND_STATUS[self.kind]


class _Protocol(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class _ClientDisconnected(Exception):
    pass


class _ResponseSendError(RuntimeError):
    pass


class _ResponseSender:
    """Track the irreversible ASGI response-start boundary."""

    __slots__ = ("_send", "started")

    def __init__(self, send: ASGISend) -> None:
        self._send = send
        self.started = False

    async def __call__(self, message: ASGIMessage) -> None:
        if message.get("type") == "http.response.start":
            self.started = True
        try:
            await self._send(message)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise _ResponseSendError from error


class _InputError(ValueError):
    def __init__(self, status: int, message: str, *,
                 public_message: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.public_message = public_message or message


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def _error(
    protocol: _Protocol,
    status: int,
    message: str,
    kind: BackendErrorKind | None = None,
) -> WireObject:
    if status >= 500:
        message = "service temporarily unavailable" if status == 503 else "internal server error"
    if status == 401:
        openai_type = anthropic_type = "authentication_error"
    elif kind is BackendErrorKind.MODEL_NOT_FOUND or status == 404:
        openai_type = anthropic_type = "not_found_error"
    elif kind is BackendErrorKind.OVERLOADED or status == 429:
        openai_type, anthropic_type = "rate_limit_error", "overloaded_error"
    elif status >= 500:
        openai_type, anthropic_type = "server_error", "api_error"
    else:
        openai_type = anthropic_type = "invalid_request_error"
    if protocol is _Protocol.OPENAI:
        return {
            "error": {
                "message": message,
                "type": openai_type,
                "param": None,
                "code": openai_type,
            }
        }
    return {"type": "error", "error": {"type": anthropic_type, "message": message}}


async def _send_json(send: ASGISend, status: int, payload: WireObject) -> None:
    body = _json_bytes(payload)
    headers = [
        (b"content-type", b"application/json; charset=utf-8"),
        (b"content-length", str(len(body)).encode()),
    ]
    if status == 401:
        headers.append((b"www-authenticate", b"Bearer"))
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": headers,
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


async def _send_empty(send: ASGISend, status: int) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-length", b"0")],
        }
    )
    await send({"type": "http.response.body", "body": b"", "more_body": False})


def _reject_constant(value: str) -> Never:
    raise ValueError(f"non-finite JSON number {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _decode_json(body: bytes) -> Mapping[str, object]:
    try:
        value: object = json.loads(
            body.decode(errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise _InputError(400, f"request body is not strict JSON: {error}",
                          public_message="request body is not valid JSON") from error
    if type(value) is not dict:
        raise _InputError(400, "request JSON body must be an object")
    return cast(dict[str, object], value)


def _require_json(scope: ASGIScope) -> None:
    headers = scope.get("headers", [])
    if type(headers) is not list:
        raise _InputError(400, "ASGI headers must be a list",
                          public_message="invalid request headers")
    values: list[str] = []
    for item in cast(list[object], headers):
        if type(item) is not tuple or len(item) != 2:
            raise _InputError(400, "ASGI headers must contain byte pairs",
                              public_message="invalid request headers")
        name, value = item
        if type(name) is not bytes or type(value) is not bytes:
            raise _InputError(400, "ASGI headers must contain byte pairs",
                              public_message="invalid request headers")
        try:
            header_name = name.decode("ascii").lower()
            header_value = value.decode("latin-1")
        except UnicodeDecodeError as error:
            raise _InputError(400, "request header name is not ASCII") from error
        if header_name == "content-type":
            values.append(header_value)
    if len(values) != 1:
        raise _InputError(415, "Content-Type must occur once and be application/json")
    parts = tuple(part.strip().lower() for part in values[0].split(";"))
    if parts[0] != "application/json" or parts[1:] not in ((), ("charset=utf-8",)):
        raise _InputError(415, "only UTF-8 application/json is supported")


async def _read_body(receive: ASGIReceive) -> bytes:
    result = bytearray()
    more = True
    while more:
        message = await receive()
        if message.get("type") == "http.disconnect":
            raise _ClientDisconnected
        if message.get("type") != "http.request":
            raise _InputError(400, "expected an ASGI http.request message",
                              public_message="invalid request body")
        chunk, raw_more = message.get("body", b""), message.get("more_body", False)
        if type(chunk) is not bytes or type(raw_more) is not bool:
            raise _InputError(400, "invalid ASGI request body fields",
                              public_message="invalid request body")
        result.extend(chunk)
        if len(result) > MAX_JSON_BODY_BYTES:
            raise _InputError(413, "request JSON body exceeds the 2 MiB limit",
                              public_message="request body is too large")
        more = raw_more
    return bytes(result)


async def _watch_disconnect(receive: ASGIReceive) -> None:
    while True:
        message = await receive()
        if message.get("type") == "http.disconnect":
            return
        if message.get("type") != "http.request":
            raise RuntimeError("unexpected ASGI message after request body")


T = TypeVar("T")


async def _race(awaitable: Awaitable[T], disconnect: asyncio.Task[None]) -> T:
    operation = asyncio.ensure_future(awaitable)
    done, _ = await asyncio.wait((operation, disconnect), return_when=asyncio.FIRST_COMPLETED)
    if disconnect in done:
        disconnect.result()
        operation.cancel()
        await asyncio.gather(operation, return_exceptions=True)
        raise _ClientDisconnected
    return operation.result()


def _payload(event: TokenEvent) -> dict[str, str]:
    if type(event.payload_items) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or type(item[1]) is not str
        for item in event.payload_items
    ):
        raise ProjectionError("event payload must contain string pairs")
    result = dict(event.payload_items)
    if tuple(result) != tuple(sorted(result)) or len(result) != len(event.payload_items):
        raise ProjectionError("event payload keys must be unique and sorted")
    return result


class _Events:
    """Small pre-wire gate; ``fold_events`` remains the complete authority."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.items: list[TokenEvent] = []
        self.usage = False

    def push(self, value: object) -> FoldedEvents | None:
        if not isinstance(value, TokenEvent):
            raise ProjectionError("backend must yield TokenEvent values")
        if (
            value.request_id != self.request_id
            or type(value.sequence_no) is not int
            or value.sequence_no != len(self.items)
        ):
            raise ProjectionError("event identity or sequence is not canonical")
        if not isinstance(value.kind, TokenEventKind) or value.token_id is not None:
            raise ProjectionError("public events require a typed kind and no raw token_id")
        if value.kind is TokenEventKind.USAGE:
            if self.usage:
                raise ProjectionError("USAGE may appear only once")
            self.usage = True
        elif value.kind is TokenEventKind.FINISH:
            if not self.usage or not self.items or self.items[-1].kind is not TokenEventKind.USAGE:
                raise ProjectionError("FINISH must immediately follow USAGE")
            self.items.append(value)
            return fold_events(self.items)
        elif value.kind in (
            TokenEventKind.TEXT_DELTA,
            TokenEventKind.REASONING_DELTA,
            TokenEventKind.TOOL_CALL_DELTA,
        ):
            if self.usage or type(value.text) is not str:
                raise ProjectionError("delta fields or lifecycle are invalid")
            payload = _payload(value)
            if value.kind is not TokenEventKind.TOOL_CALL_DELTA:
                if payload:
                    raise ProjectionError("text and reasoning deltas cannot carry payload")
            elif set(payload) not in ({"index"}, {"id", "index", "name"}):
                raise ProjectionError("tool delta metadata is invalid")
            elif (
                not payload["index"].isascii()
                or not payload["index"].isdecimal()
                or str(int(payload["index"])) != payload["index"]
            ):
                raise ProjectionError("tool index is not canonical")
        else:
            raise ProjectionError(f"event kind {value.kind!r} is not public")
        self.items.append(value)
        return None


class _StreamProjector:
    """Project an already validated event stream without terminal replay."""

    def __init__(self, protocol: _Protocol, request: CanonicalChatRequest, created: int) -> None:
        self.protocol, self.request, self.created = protocol, request, created
        self.block: tuple[TokenEventKind, int | None] | None = None
        self.block_index = 0
        self.input_tokens = 0
        self._openai_delta_prefix = (
            b'data: {"id":'
            + _json_bytes(request.request_id)
            + b',"object":"chat.completion.chunk","created":'
            + str(created).encode("ascii")
            + b',"model":'
            + _json_bytes(request.model_id)
            + b',"choices":[{"index":0,"delta":{'
        )

    def _openai(self, choice: WireObject | None) -> WireObject:
        return {
            "id": self.request.request_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.request.model_id,
            "choices": [] if choice is None else [choice],
        }

    def start(self, input_tokens: int) -> tuple[WireObject, ...]:
        self.input_tokens = input_tokens
        if self.protocol is _Protocol.OPENAI:
            value = self._openai(
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            )
        else:
            value = {
                "type": "message_start",
                "message": {
                    "id": self.request.request_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.request.model_id,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                },
            }
        return (value,)

    def delta(self, event: TokenEvent) -> tuple[WireObject, ...]:
        values: tuple[WireObject, ...]
        if self.protocol is _Protocol.OPENAI:
            values = (self._openai(self._openai_choice(event)),)
        else:
            values = self._anthropic_delta(event)
        return values

    def delta_frame(self, event: TokenEvent) -> bytes:
        """Encode the common OpenAI text path without rebuilding its envelope."""

        if self.protocol is _Protocol.OPENAI and self.request.stream_continuous_usage_stats:
            if event.output_token_count is None:
                raise ProjectionError("continuous usage requires output progress")
            value = self._openai(self._openai_choice(event))
            value["usage"] = project_openai_usage(
                self.input_tokens, event.output_token_count, event.reasoning_token_count,
            )
            return b"data: " + _json_bytes(value) + b"\n\n"
        if self.protocol is _Protocol.OPENAI and event.kind in (
            TokenEventKind.TEXT_DELTA,
            TokenEventKind.REASONING_DELTA,
        ):
            if type(event.text) is not str:
                raise ProjectionError("delta text must be a string")
            field = (
                b'"content":'
                if event.kind is TokenEventKind.TEXT_DELTA
                else b'"reasoning_content":'
            )
            return (
                self._openai_delta_prefix
                + field
                + _json_bytes(event.text)
                + b'},"finish_reason":null}]}\n\n'
            )
        return b"".join(_frame(self.protocol, value) for value in self.delta(event))

    @staticmethod
    def _openai_choice(event: TokenEvent) -> WireObject:
        if event.kind is TokenEventKind.TEXT_DELTA:
            delta: WireObject = {"content": event.text}
        elif event.kind is TokenEventKind.REASONING_DELTA:
            delta = {"reasoning_content": event.text}
        else:
            payload = dict(event.payload_items)
            function: WireObject = {"arguments": event.text}
            tool: WireObject = {"index": int(payload["index"]), "function": function}
            if "id" in payload:
                tool.update({"id": payload["id"], "type": "function"})
                function["name"] = payload["name"]
            delta = {"tool_calls": [tool]}
        return {"index": 0, "delta": delta, "finish_reason": None}

    def _anthropic_delta(self, event: TokenEvent) -> tuple[WireObject, ...]:
        if (
            not event.text
            and event.kind is not TokenEventKind.TOOL_CALL_DELTA
            and self.block is not None
            and self.block != (event.kind, None)
        ):
            return ()
        payload = dict(event.payload_items)
        tool = int(payload["index"]) if "index" in payload else None
        block = (event.kind, tool)
        values: list[WireObject] = []
        if block != self.block:
            if self.block is not None:
                values.append({"type": "content_block_stop", "index": self.block_index})
                self.block_index += 1
            if event.kind is TokenEventKind.TEXT_DELTA:
                initial: WireObject = {"type": "text", "text": ""}
            elif event.kind is TokenEventKind.REASONING_DELTA:
                initial = {"type": "thinking", "thinking": ""}
            else:
                initial = {
                    "type": "tool_use",
                    "id": payload["id"],
                    "name": payload["name"],
                    "input": {},
                }
            values.append(
                {
                    "type": "content_block_start",
                    "index": self.block_index,
                    "content_block": initial,
                }
            )
            self.block = block
        if event.kind is TokenEventKind.TEXT_DELTA:
            delta: WireObject = {"type": "text_delta", "text": event.text}
        elif event.kind is TokenEventKind.REASONING_DELTA:
            delta = {"type": "thinking_delta", "thinking": event.text}
        else:
            delta = {"type": "input_json_delta", "partial_json": event.text}
        values.append({"type": "content_block_delta", "index": self.block_index, "delta": delta})
        return tuple(values)

    def finish(self, output: FoldedEvents) -> tuple[WireObject, ...]:
        if self.protocol is _Protocol.OPENAI:
            choice: WireObject = {
                "index": 0,
                "delta": {},
                "finish_reason": output.finish_reason,
            }
            if output.stop_sequence is not None:
                choice["stop_reason"] = output.stop_sequence
            values = [self._openai(choice)]
            if self.request.stream_include_usage:
                values.append(
                    {
                        **self._openai(None),
                        "usage": project_openai_usage(
                            output.input_tokens, output.output_tokens, output.reasoning_tokens,
                        ),
                    }
                )
            return tuple(values)

        values: list[WireObject] = []
        if self.block is not None:
            values.append(
                {"type": "content_block_stop", "index": self.block_index}
            )
            self.block = None
        if output.finish_reason == "length":
            stop_reason, stop_sequence = "max_tokens", None
        elif output.finish_reason == "tool_calls":
            stop_reason, stop_sequence = "tool_use", None
        else:
            stop_reason = "stop_sequence" if output.stop_sequence is not None else "end_turn"
            stop_sequence = output.stop_sequence
        values.extend(
            (
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": stop_reason,
                        "stop_sequence": stop_sequence,
                    },
                    "usage": {"output_tokens": output.output_tokens},
                },
                {"type": "message_stop"},
            )
        )
        return tuple(values)

    def failure(self, payload: WireObject) -> tuple[WireObject, ...]:
        if self.protocol is _Protocol.OPENAI:
            return (payload,)
        result: list[WireObject] = []
        if self.block is not None:
            result.append({"type": "content_block_stop", "index": self.block_index})
            self.block = None
        result.extend((payload, {"type": "message_stop"}))
        return tuple(result)


def _frame(protocol: _Protocol, value: WireObject) -> bytes:
    data = _json_bytes(value)
    if protocol is _Protocol.OPENAI:
        return b"data: " + data + b"\n\n"
    event = value.get("type")
    if type(event) is not str:
        raise ProjectionError("Anthropic SSE event requires a type")
    return b"event: " + event.encode("ascii") + b"\ndata: " + data + b"\n\n"


async def _stream_start(send: ASGISend) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"text/event-stream; charset=utf-8"),
                (b"cache-control", b"no-cache"),
                (b"x-accel-buffering", b"no"),
            ],
        }
    )


async def _stream_values(
    send: ASGISend, protocol: _Protocol, values: tuple[WireObject, ...]
) -> None:
    if values:
        await send(
            {
                "type": "http.response.body",
                "body": b"".join(_frame(protocol, value) for value in values),
                "more_body": True,
            }
        )


async def _stream_frame(send: ASGISend, frame: bytes) -> None:
    await send(
        {
            "type": "http.response.body",
            "body": frame,
            "more_body": True,
        }
    )


async def _stream_end(send: ASGISend, protocol: _Protocol) -> None:
    if protocol is _Protocol.OPENAI:
        await send(
            {
                "type": "http.response.body",
                "body": b"data: [DONE]\n\n",
                "more_body": True,
            }
        )
    await send({"type": "http.response.body", "body": b"", "more_body": False})


async def _stream_finish(
    send: ASGISend,
    protocol: _Protocol,
    values: tuple[WireObject, ...],
) -> None:
    body = b"".join(_frame(protocol, value) for value in values)
    if protocol is _Protocol.OPENAI:
        body += b"data: [DONE]\n\n"
    await send({"type": "http.response.body", "body": body, "more_body": False})


class _Canceller:
    def __init__(self, backend: ChatBackend, request_id: str) -> None:
        self.backend, self.request_id, self.called = backend, request_id, False

    async def __call__(self) -> None:
        if self.called:
            return
        self.called = True
        with suppress(Exception):
            await self.backend.cancel(self.request_id)


class CubicASGIApp:
    """ASGI application exposing model discovery and the two chat protocols."""

    def __init__(
        self,
        backend: ChatBackend,
        *,
        model_id: str,
        served_model_name: str | Sequence[str] | None = None,
        api_key: str | None = None,
        request_id_factory: Callable[[], str] | None = None,
        clock: Callable[[], int] | None = None,
    ) -> None:
        if type(model_id) is not str or not model_id or model_id.strip() != model_id:
            raise ValueError("ASGI model_id must be canonical and non-empty")
        if api_key is not None and (
            type(api_key) is not str or not api_key
            or any(not 33 <= ord(char) <= 126 for char in api_key)
        ):
            raise ValueError("api_key must be non-empty printable ASCII without spaces")
        self._api_key_digest = (
            None if api_key is None else hashlib.sha256(api_key.encode("ascii")).digest()
        )
        names = ((model_id,) if served_model_name is None else
                 (served_model_name,) if isinstance(served_model_name, str) else
                 tuple(served_model_name))
        if not names or any(type(name) is not str or not name or name.strip() != name
                            for name in names):
            raise ValueError("served model names must be canonical and non-empty")
        self.backend = backend
        self.model_id = model_id
        self.served_model_names = tuple(dict.fromkeys(names))
        self.request_id_factory = request_id_factory or (lambda: f"req_{uuid.uuid4().hex[:16]}")
        self.clock = clock or (lambda: int(time.time()))
        self.model_created = self._created()

    def _ready(self) -> bool:
        value = getattr(self.backend, "ready", True)
        return type(value) is bool and value

    def _authorized(self, scope: ASGIScope, path: str) -> bool:
        expected = self._api_key_digest
        if expected is None:
            return True
        headers = scope.get("headers", [])
        if type(headers) is not list:
            return False
        authorization: list[bytes] = []
        anthropic_keys: list[bytes] = []
        for item in headers:
            if (type(item) is not tuple or len(item) != 2
                    or any(type(part) is not bytes for part in item)):
                return False
            name, value = item
            if name.lower() == b"authorization":
                authorization.append(value)
            elif name.lower() == b"x-api-key":
                anthropic_keys.append(value)
        if len(authorization) > 1 or len(anthropic_keys) > 1:
            return False
        if authorization:
            scheme, _, credential = authorization[0].partition(b" ")
            if scheme.lower() != b"bearer":
                return False
        elif path == "/v1/messages" and anthropic_keys:
            credential = anthropic_keys[0]
        else:
            return False
        return hmac.compare_digest(hashlib.sha256(credential).digest(), expected)

    async def __call__(self, scope: ASGIScope, receive: ASGIReceive, send: ASGISend) -> None:
        if scope.get("type") != "http":
            raise RuntimeError("CubicASGIApp supports only ASGI HTTP scopes")
        response = _ResponseSender(send)
        path = scope.get("path")
        if type(path) is not str:
            path = ""
        if (self._api_key_digest is not None
                and (path == "/v1" or path.startswith("/v1/"))
                and scope.get("method") != "OPTIONS"
                and not self._authorized(scope, path)):
            protocol = _Protocol.ANTHROPIC if path == "/v1/messages" else _Protocol.OPENAI
            await _send_json(response, 401, _error(protocol, 401, "Unauthorized"))
            return
        if path in ("/health", "/v1/health"):
            if scope.get("method") != "GET":
                await _send_json(
                    response,
                    405,
                    _error(_Protocol.OPENAI, 405, "health endpoint requires GET"),
                )
                return
            await _send_empty(response, 200 if self._ready() else 503)
            return
        if path == "/v1/models":
            if scope.get("method") != "GET":
                await _send_json(
                    response,
                    405,
                    _error(_Protocol.OPENAI, 405, "models endpoint requires GET"),
                )
                return
            if not self._ready():
                await _send_json(
                    response,
                    503,
                    _error(
                        _Protocol.OPENAI,
                        503,
                        "served model topology is unavailable",
                        BackendErrorKind.UNAVAILABLE,
                    ),
                )
                return
            await _send_json(
                response,
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": name,
                            "object": "model",
                            "created": self.model_created,
                            "owned_by": "cubic-serve",
                        } for name in self.served_model_names
                    ],
                },
            )
            return
        protocol = {
            "/v1/chat/completions": _Protocol.OPENAI,
            "/v1/messages": _Protocol.ANTHROPIC,
        }.get(path)
        if protocol is None:
            await _send_json(response, 404, _error(_Protocol.OPENAI, 404, "route not found"))
            return
        if scope.get("method") != "POST":
            await _send_json(
                response, 405, _error(protocol, 405, "chat endpoints require POST")
            )
            return
        if not self._ready():
            await _send_json(
                response,
                503,
                _error(
                    protocol,
                    503,
                    "served model topology is unavailable",
                    BackendErrorKind.UNAVAILABLE,
                ),
            )
            return
        request_id = self.request_id_factory()
        request_started = time.monotonic()
        _LOGGER.info("[POST] path=%s request_id=%s", path, request_id)
        try:
            _require_json(scope)
            payload = _decode_json(await _read_body(receive))
            if protocol is _Protocol.OPENAI:
                request, stream = normalize_openai_chat_completions(payload, request_id=request_id)
            else:
                request, stream = normalize_anthropic_messages(payload, request_id=request_id)
        except _ClientDisconnected:
            return
        except _InputError as error:
            _LOGGER.exception(
                "request failed request_id=%s method=POST path=%s "
                "status=%d duration_ms=%.3f error=%s",
                request_id,
                path,
                error.status,
                (time.monotonic() - request_started) * 1000,
                error,
            )
            await _send_json(
                response, error.status, _error(protocol, error.status, error.public_message)
            )
            return
        except ChatNormalizationError as error:
            _LOGGER.exception(
                "request failed request_id=%s method=POST path=%s "
                "status=400 duration_ms=%.3f error=%s",
                request_id,
                path,
                (time.monotonic() - request_started) * 1000,
                error,
            )
            await _send_json(response, 400, _error(protocol, 400, str(error)))
            return

        disconnect = asyncio.create_task(_watch_disconnect(receive))
        cancel = _Canceller(self.backend, request.request_id)
        try:
            if request.model_id not in self.served_model_names:
                raise ChatBackendError(BackendErrorKind.MODEL_NOT_FOUND, "requested model is not served")
            backend_request = (request if request.model_id == self.model_id else
                               replace(request, model_id=self.model_id))
            if request.model_id != self.served_model_names[0]:
                request = replace(request, model_id=self.served_model_names[0])
            execution = await _race(self.backend.start(backend_request), disconnect)
            if not isinstance(execution, ChatExecution):
                raise TypeError("backend.start must return ChatExecution")
            if stream:
                stream_error = await _race(
                    self._stream(request, execution, protocol, cancel, response),
                    disconnect,
                )
                if stream_error is not None:
                    _LOGGER.error(
                        "request failed request_id=%s method=POST path=%s "
                        "transport_status=200 duration_ms=%.3f error=%s",
                        request.request_id,
                        path,
                        (time.monotonic() - request_started) * 1000,
                        stream_error,
                        exc_info=(
                            type(stream_error),
                            stream_error,
                            stream_error.__traceback__,
                        ),
                    )
                    return
            else:
                await _race(
                    self._complete(request, execution, protocol, cancel, response),
                    disconnect,
                )
        except _ClientDisconnected:
            await cancel()
        except _ResponseSendError:
            await cancel()
        except ChatBackendError as error:
            _LOGGER.exception(
                "request failed request_id=%s method=POST path=%s "
                "status=%d duration_ms=%.3f error=%s",
                request.request_id,
                path,
                error.status,
                (time.monotonic() - request_started) * 1000,
                error,
            )
            if response.started:
                await cancel()
                raise
            await _send_json(
                response,
                error.status,
                _error(protocol, error.status, error.public_message, error.kind),
            )
        except ProjectionError:
            _LOGGER.exception(
                "request failed request_id=%s method=POST path=%s status=500 duration_ms=%.3f",
                request.request_id,
                path,
                (time.monotonic() - request_started) * 1000,
            )
            await cancel()
            if response.started:
                raise
            await _send_json(
                response,
                500,
                _error(protocol, 500, "backend produced an invalid canonical event stream"),
            )
        except asyncio.CancelledError:
            await cancel()
            raise
        except Exception:
            _LOGGER.exception(
                "request failed request_id=%s method=POST path=%s status=500 duration_ms=%.3f",
                request.request_id,
                path,
                (time.monotonic() - request_started) * 1000,
            )
            await cancel()
            if response.started:
                raise
            await _send_json(
                response, 500, _error(protocol, 500, "internal server error")
            )
        finally:
            disconnect.cancel()
            await asyncio.gather(disconnect, return_exceptions=True)

    async def _complete(
        self,
        request: CanonicalChatRequest,
        execution: ChatExecution,
        protocol: _Protocol,
        cancel: _Canceller,
        send: ASGISend,
    ) -> Exception | None:
        sequence = _Events(request.request_id)
        try:
            folded = await self._consume(sequence, execution.events)
            if folded.input_tokens != execution.input_tokens:
                raise ProjectionError("admission and terminal input token counts differ")
            if protocol is _Protocol.OPENAI:
                response = project_openai_chat_completion(
                    sequence.items, model=request.model_id, created=self._created(),
                )
            else:
                response = project_anthropic_message(sequence.items, model=request.model_id)
            await _send_json(send, 200, response)
        except ProjectionError:
            _LOGGER.exception(
                "non-streaming response projection failed request_id=%s",
                request.request_id,
            )
            await cancel()
            await _send_json(
                send,
                500,
                _error(protocol, 500, "backend produced an invalid canonical event stream"),
            )
        except ChatBackendError:
            await cancel()
            raise

    async def _stream(
        self,
        request: CanonicalChatRequest,
        execution: ChatExecution,
        protocol: _Protocol,
        cancel: _Canceller,
        send: ASGISend,
    ) -> None:
        sequence = _Events(request.request_id)
        projector = _StreamProjector(protocol, request, self._created())
        try:
            first = await execution.events.__anext__()
            folded = sequence.push(first)
        except StopAsyncIteration as error:
            await cancel()
            raise ProjectionError("event stream ended before USAGE and FINISH") from error
        except (_ClientDisconnected, asyncio.CancelledError):
            raise
        except Exception:
            await cancel()
            raise

        try:
            await _stream_start(send)
            await _stream_values(send, protocol, projector.start(execution.input_tokens))
            if folded is None and first.kind in (
                TokenEventKind.TEXT_DELTA,
                TokenEventKind.REASONING_DELTA,
                TokenEventKind.TOOL_CALL_DELTA,
            ):
                await _stream_frame(send, projector.delta_frame(first))
            while True:
                event = await execution.events.__anext__()
                folded = sequence.push(event)
                if folded is None:
                    if event.kind in (
                        TokenEventKind.TEXT_DELTA,
                        TokenEventKind.REASONING_DELTA,
                        TokenEventKind.TOOL_CALL_DELTA,
                    ):
                        await _stream_frame(send, projector.delta_frame(event))
                    continue
                if folded.input_tokens != execution.input_tokens:
                    raise ProjectionError("admission and terminal input token counts differ")
                await _stream_finish(send, protocol, projector.finish(folded))
                return None
        except StopAsyncIteration:
            error = ProjectionError("event stream ended before USAGE and FINISH")
            await self._stream_failure(
                protocol,
                projector,
                cancel,
                send,
                error,
            )
            return error
        except _ResponseSendError:
            await cancel()
            raise
        except _ClientDisconnected:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._stream_failure(protocol, projector, cancel, send, error)
            return error

    @staticmethod
    async def _stream_failure(
        protocol: _Protocol,
        projector: _StreamProjector,
        cancel: _Canceller,
        send: ASGISend,
        error: Exception,
    ) -> None:
        _LOGGER.error(
            "stream response failed request_id=%s",
            projector.request.request_id,
            exc_info=(type(error), error, error.__traceback__),
        )
        await cancel()
        if isinstance(error, ChatBackendError):
            status, message, kind = error.status, error.public_message, error.kind
        elif isinstance(error, ProjectionError):
            status, message, kind = (
                500,
                "backend produced an invalid canonical event stream",
                None,
            )
        else:
            status, message, kind = 500, "internal server error", None
        await _stream_values(
            send, protocol, projector.failure(_error(protocol, status, message, kind))
        )
        await _stream_end(send, protocol)

    @staticmethod
    async def _consume(
        sequence: _Events,
        events: AsyncIterator[TokenEvent],
    ) -> FoldedEvents:
        while True:
            try:
                event = await events.__anext__()
            except StopAsyncIteration as error:
                raise ProjectionError("event stream ended before USAGE and FINISH") from error
            folded = sequence.push(event)
            if folded is not None:
                return folded

    def _created(self) -> int:
        value = self.clock()
        if type(value) is not int or value < 0:
            raise ValueError("clock must return a non-negative integer")
        return value


__all__ = (
    "MAX_JSON_BODY_BYTES",
    "BackendErrorKind",
    "ChatBackend",
    "ChatBackendError",
    "ChatExecution",
    "CubicASGIApp",
)
