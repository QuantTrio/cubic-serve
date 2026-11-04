from __future__ import annotations

import asyncio
import json
import unittest
from collections.abc import AsyncIterator, Callable
from unittest.mock import patch

from cubic_serve.api import server
from cubic_serve.api.server import (
    BackendErrorKind,
    ChatBackendError,
    ChatExecution,
    CubicASGIApp,
)
from cubic_serve.contracts.request import CanonicalChatRequest, TokenEvent, TokenEventKind

EventFactory = Callable[[CanonicalChatRequest], AsyncIterator[TokenEvent]]


def _event(
    request_id: str,
    sequence_no: int,
    kind: TokenEventKind,
    *,
    text: str | None = None,
    payload: tuple[tuple[str, str], ...] = (),
) -> TokenEvent:
    return TokenEvent(
        request_id=request_id,
        sequence_no=sequence_no,
        kind=kind,
        text=text,
        payload_items=payload,
    )


async def _text_events(
    request: CanonicalChatRequest,
    *deltas: str,
    input_tokens: int = 3,
) -> AsyncIterator[TokenEvent]:
    for index, delta in enumerate(deltas):
        yield _event(request.request_id, index, TokenEventKind.TEXT_DELTA, text=delta)
    yield _event(
        request.request_id,
        len(deltas),
        TokenEventKind.USAGE,
        payload=(("input_tokens", str(input_tokens)), ("output_tokens", str(len(deltas)))),
    )
    yield _event(
        request.request_id,
        len(deltas) + 1,
        TokenEventKind.FINISH,
        payload=(("reason", "stop"),),
    )


class _Backend:
    def __init__(
        self,
        factory: EventFactory,
        *,
        input_tokens: int = 3,
        start_error: ChatBackendError | None = None,
    ) -> None:
        self.factory = factory
        self.input_tokens = input_tokens
        self.start_error = start_error
        self.requests: list[CanonicalChatRequest] = []
        self.cancelled: list[str] = []
        self.ready = True

    async def start(self, request: CanonicalChatRequest) -> ChatExecution:
        self.requests.append(request)
        if self.start_error is not None:
            raise self.start_error
        return ChatExecution(self.input_tokens, self.factory(request))

    async def cancel(self, request_id: str) -> None:
        self.cancelled.append(request_id)


def _openai_payload(*, stream: bool, include_usage: bool = False) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": "fixture",
        "messages": [{"role": "user", "content": "hello"}],
        "max_completion_tokens": 8,
        "stream": stream,
    }
    if include_usage:
        payload["stream_options"] = {"include_usage": True}
    return payload


def _anthropic_payload(*, stream: bool) -> dict[str, object]:
    return {
        "model": "fixture",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        "max_tokens": 8,
        "stream": stream,
    }


async def _invoke(
    app: CubicASGIApp,
    path: str,
    payload: object,
    *,
    method: str = "POST",
    content_type: bytes = b"application/json",
    raw_body: bytes | None = None,
    headers: list[tuple[bytes, bytes]] | None = None,
    on_send: Callable[[dict[str, object], asyncio.Queue[dict[str, object]]], None] | None = None,
) -> list[dict[str, object]]:
    body = raw_body if raw_body is not None else json.dumps(payload).encode()
    incoming: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    midpoint = max(1, len(body) // 2)
    await incoming.put({"type": "http.request", "body": body[:midpoint], "more_body": True})
    await incoming.put({"type": "http.request", "body": body[midpoint:], "more_body": False})
    sent: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return await incoming.get()

    async def send(message: dict[str, object]) -> None:
        sent.append(message)
        if on_send is not None:
            on_send(message, incoming)

    scope: dict[str, object] = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": [(b"content-type", content_type), *(headers or [])],
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=2)
    return sent


def _body(messages: list[dict[str, object]]) -> bytes:
    chunks: list[bytes] = []
    for message in messages:
        if message.get("type") != "http.response.body":
            continue
        chunk = message.get("body", b"")
        assert isinstance(chunk, bytes)
        chunks.append(chunk)
    return b"".join(chunks)


def _json_response(messages: list[dict[str, object]]) -> dict[str, object]:
    value = json.loads(_body(messages))
    assert isinstance(value, dict)
    return value


def _sse_json(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    frames = _body(messages).decode().split("\n\n")
    result: list[dict[str, object]] = []
    for frame in frames:
        if not frame or frame == "data: [DONE]":
            continue
        data_line = next(line for line in frame.splitlines() if line.startswith("data: "))
        value = json.loads(data_line[6:])
        assert isinstance(value, dict)
        result.append(value)
    return result


class ServerTests(unittest.TestCase):
    def test_api_key_rejects_invalid_configuration(self) -> None:
        backend = _Backend(lambda request: _text_events(request, "hello"))
        for key in ("", "has space", "newline\n", "\x00", "\x7f", "秘密", 123):
            with self.subTest(key=key), self.assertRaises(ValueError):
                CubicASGIApp(backend, model_id="fixture", api_key=key)

    def test_api_key_rejects_before_body_read_or_backend_admission(self) -> None:
        key = "test-secret-key"
        backend = _Backend(lambda request: _text_events(request, "hello"))
        backend.ready = False
        app = CubicASGIApp(backend, model_id="fixture", api_key=key)
        bad_headers = (
            [],
            [(b"authorization", b"Bearer wrong")],
            [(b"authorization", b"Basic test-secret-key")],
            [(b"authorization", b"test-secret-key")],
            [(b"authorization", b"Bearer ")],
            [(b"authorization", b"Bearer  test-secret-key")],
            [(b"authorization", b"Bearer test-secret-key ")],
            [(b"authorization", b"Bearer test-secret-key\xff")],
            [(b"authorization", b"Bearer test-secret-key"),
             (b"Authorization", b"Bearer wrong")],
            [(b"authorization", b"Bearer wrong"), (b"x-api-key", key.encode())],
            [(b"x-api-key", key.encode()), (b"x-api-key", key.encode())],
        )

        async def no_body_read():
            self.fail("unauthenticated requests must not read the body")

        for path in ("/v1/models", "/v1/chat/completions", "/v1/messages", "/v1/health"):
            for headers in bad_headers:
                with self.subTest(path=path, headers=headers):
                    sent = []

                    async def send(message, output=sent):
                        output.append(message)

                    asyncio.run(app({"type": "http", "method": "POST", "path": path,
                                     "headers": headers}, no_body_read, send))
                    self.assertEqual(sent[0]["status"], 401)
                    self.assertIn((b"www-authenticate", b"Bearer"), sent[0]["headers"])
                    response = _json_response(sent)
                    self.assertEqual(response["error"]["type"], "authentication_error")
                    self.assertEqual(response["error"]["message"], "Unauthorized")
                    self.assertNotIn(key.encode(), _body(sent))
        self.assertEqual(backend.requests, [])
        self.assertEqual(backend.cancelled, [])

    def test_api_key_accepts_bearer_for_models_and_both_chat_protocols(self) -> None:
        headers = [(b"Authorization", b"bEaReR test-secret-key")]
        for path, payload in (("/v1/chat/completions", _openai_payload),
                              ("/v1/messages", _anthropic_payload)):
            for stream in (False, True):
                with self.subTest(path=path, stream=stream):
                    backend = _Backend(lambda request: _text_events(request, "hello"))
                    app = CubicASGIApp(backend, model_id="fixture", api_key="test-secret-key")
                    with patch.object(server.hmac, "compare_digest",
                                      wraps=server.hmac.compare_digest) as compare:
                        result = asyncio.run(_invoke(app, path, payload(stream=stream),
                                                     headers=headers))
                    self.assertEqual(result[0]["status"], 200)
                    self.assertIn(b"hello", _body(result))
                    self.assertEqual(len(backend.requests), 1)
                    self.assertEqual(compare.call_count, 1)
        result = asyncio.run(_invoke(app, "/v1/models", {}, method="GET", headers=headers))
        self.assertEqual(result[0]["status"], 200)

    def test_api_key_supports_anthropic_header_only_for_messages(self) -> None:
        for stream in (False, True):
            backend = _Backend(lambda request: _text_events(request, "hello"))
            app = CubicASGIApp(backend, model_id="fixture", api_key="test-secret-key")
            headers = [(b"X-Api-Key", b"test-secret-key")]
            result = asyncio.run(_invoke(app, "/v1/messages", _anthropic_payload(stream=stream),
                                         headers=headers))
            self.assertEqual(result[0]["status"], 200)
            self.assertIn(b"hello", _body(result))
            result = asyncio.run(_invoke(app, "/v1/models", {}, method="GET", headers=headers))
            self.assertEqual(result[0]["status"], 401)

    def test_api_key_leaves_root_health_and_options_unauthenticated(self) -> None:
        backend = _Backend(lambda request: _text_events(request, "hello"))
        app = CubicASGIApp(backend, model_id="fixture", api_key="test-secret-key")
        result = asyncio.run(_invoke(app, "/health", {}, method="GET"))
        self.assertEqual(result[0]["status"], 200)
        result = asyncio.run(_invoke(app, "/v1/models", {}, method="OPTIONS"))
        self.assertEqual(result[0]["status"], 405)
        result = asyncio.run(_invoke(app, "/v1/health", {}, method="GET",
                                     headers=[(b"authorization", b"Bearer test-secret-key")]))
        self.assertEqual(result[0]["status"], 200)

    def test_default_request_id_is_compact(self) -> None:
        backend = _Backend(lambda request: _text_events(request, "unused"))
        app = CubicASGIApp(backend, model_id="fixture")

        self.assertRegex(app.request_id_factory(), r"^req_[0-9a-f]{16}$")

    def test_health_reflects_backend_readiness(self) -> None:
        backend = _Backend(lambda request: _text_events(request, "unused"))
        app = CubicASGIApp(backend, model_id="fixture")

        for path in ("/health", "/v1/health"):
            healthy = asyncio.run(_invoke(app, path, {}, method="GET"))
            self.assertEqual(healthy[0]["status"], 200)
            self.assertEqual(_body(healthy), b"")
            backend.ready = False
            unhealthy = asyncio.run(_invoke(app, path, {}, method="GET"))
            self.assertEqual(unhealthy[0]["status"], 503)
            self.assertEqual(_body(unhealthy), b"")
            backend.ready = True

    def test_unavailable_backend_is_not_advertised_or_admitted(self) -> None:
        backend = _Backend(lambda request: _text_events(request, "unused"))
        backend.ready = False
        app = CubicASGIApp(backend, model_id="fixture")

        models = asyncio.run(_invoke(app, "/v1/models", {}, method="GET"))
        chat = asyncio.run(
            _invoke(app, "/v1/chat/completions", _openai_payload(stream=True))
        )

        self.assertEqual((models[0]["status"], chat[0]["status"]), (503, 503))
        self.assertIn(b"service temporarily unavailable", _body(models))
        self.assertEqual(backend.requests, [])

    def test_models_lists_the_frozen_served_model_without_backend_admission(self) -> None:
        backend = _Backend(lambda request: _text_events(request, "unused"))
        app = CubicASGIApp(backend, model_id="fixture", clock=lambda: 7)

        response = asyncio.run(_invoke(app, "/v1/models", {}, method="GET"))

        self.assertEqual(response[0]["status"], 200)
        self.assertEqual(
            _json_response(response),
            {
                "object": "list",
                "data": [
                    {
                        "id": "fixture",
                        "object": "model",
                        "created": 7,
                        "owned_by": "cubic-serve",
                    }
                ],
            },
        )
        self.assertEqual(backend.requests, [])
        self.assertEqual(backend.cancelled, [])

        wrong_method = asyncio.run(_invoke(app, "/v1/models", {}))
        self.assertEqual(wrong_method[0]["status"], 405)

    def test_public_model_aliases_are_listed_without_backend_admission(self) -> None:
        backend = _Backend(lambda request: _text_events(request, "hello"))
        app = CubicASGIApp(backend, model_id="internal:checkpoint",
                           served_model_name=["public", "compat", "public"])
        response = asyncio.run(_invoke(app, "/v1/models", {}, method="GET"))
        self.assertEqual(response[0]["status"], 200)
        self.assertEqual([item["id"] for item in _json_response(response)["data"]],
                         ["public", "compat"])
        self.assertEqual(backend.requests, [])

    def test_public_model_aliases_preserve_backend_identity_in_both_protocols(self) -> None:
        for path, make_payload in (("/v1/chat/completions", _openai_payload),
                                   ("/v1/messages", _anthropic_payload)):
            for stream in (False, True):
                for alias in ("public", "compat"):
                    with self.subTest(path=path, stream=stream, alias=alias):
                        backend = _Backend(lambda request: _text_events(request, "hello"))
                        app = CubicASGIApp(backend, model_id="internal:checkpoint",
                                           served_model_name=["public", "compat"])
                        payload = make_payload(stream=stream)
                        payload["model"] = alias
                        response = asyncio.run(_invoke(app, path, payload))
                        self.assertEqual(response[0]["status"], 200)
                        self.assertEqual(len(backend.requests), 1)
                        request = backend.requests[0]
                        self.assertEqual(request.model_id, "internal:checkpoint")
                        self.assertEqual(request.max_new_tokens, 8)
                        self.assertEqual(request.messages[0].content_parts[0].text, "hello")
                        frames = _sse_json(response) if stream else [_json_response(response)]
                        models = [frame.get("model", frame.get("message", {}).get("model"))
                                  for frame in frames]
                        self.assertIn("public", models)
                        self.assertTrue(all(name in (None, "public") for name in models))

    def test_unknown_model_alias_is_rejected_before_inference_or_stream_start(self) -> None:
        for path, make_payload in (("/v1/chat/completions", _openai_payload),
                                   ("/v1/messages", _anthropic_payload)):
            for stream in (False, True):
                for unknown in ("missing", "internal:checkpoint"):
                    with self.subTest(path=path, stream=stream, unknown=unknown):
                        backend = _Backend(lambda request: _text_events(request, "unused"))
                        app = CubicASGIApp(backend, model_id="internal:checkpoint",
                                           served_model_name="public")
                        payload = make_payload(stream=stream)
                        payload["model"] = unknown
                        response = asyncio.run(_invoke(app, path, payload))
                        self.assertEqual(response[0]["status"], 404)
                        self.assertEqual(_json_response(response)["error"]["message"],
                                         "model not found")
                        self.assertNotIn(b"internal:checkpoint", _body(response))
                        self.assertEqual(backend.requests, [])
                        self.assertEqual(backend.cancelled, [])

    def test_public_model_name_validation(self) -> None:
        backend = _Backend(lambda request: _text_events(request, "unused"))
        for names in ([], (), "", " ", " padded", ["public", ""], [1]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                CubicASGIApp(backend, model_id="fixture", served_model_name=names)
        app = CubicASGIApp(backend, model_id="fixture", served_model_name="public")
        self.assertEqual(app.served_model_names, ("public",))

    def test_tool_completion_matches_stream_and_complete_for_every_choice(self) -> None:
        async def tool_events(request: CanonicalChatRequest) -> AsyncIterator[TokenEvent]:
            yield _event(request.request_id, 0, TokenEventKind.TOOL_CALL_DELTA, text="{}",
                         payload=(("id", "call-0"), ("index", "0"), ("name", "f")))
            yield _event(request.request_id, 1, TokenEventKind.USAGE,
                         payload=(("input_tokens", "3"), ("output_tokens", "1")))
            yield _event(request.request_id, 2, TokenEventKind.FINISH,
                         payload=(("reason", "tool_calls"),))

        for choice in ("auto", "required", {"type": "function", "function": {"name": "f"}}):
            for stream in (False, True):
                with self.subTest(choice=choice, stream=stream):
                    backend = _Backend(tool_events)
                    app = CubicASGIApp(backend, model_id="fixture")
                    payload = _openai_payload(stream=stream)
                    payload.update(tools=[{"type": "function", "function": {"name": "f"}}],
                                   tool_choice=choice)
                    messages = asyncio.run(_invoke(app, "/v1/chat/completions", payload))
                    self.assertEqual(messages[0]["status"], 200)
                    response = _sse_json(messages)[-1] if stream else json.loads(_body(messages))
                    self.assertEqual(response["choices"][0]["finish_reason"], "tool_calls")
                    if stream:
                        calls = [tool for item in _sse_json(messages)
                                 for result in item["choices"]
                                 for tool in result["delta"].get("tool_calls", [])]
                        self.assertEqual(calls[0], {"index": 0, "id": "call-0",
                            "type": "function", "function": {"name": "f", "arguments": "{}"}})
                    else:
                        call = response["choices"][0]["message"]["tool_calls"][0]
                        self.assertEqual(call["function"], {"name": "f", "arguments": "{}"})
                    self.assertFalse(backend.cancelled)

    def test_tool_length_limit_preserves_partial_arguments_for_every_choice(self) -> None:
        async def tool_events(request: CanonicalChatRequest) -> AsyncIterator[TokenEvent]:
            yield _event(request.request_id, 0, TokenEventKind.TOOL_CALL_DELTA, text='{"x":',
                         payload=(("id", "call-0"), ("index", "0"), ("name", "f")))
            yield _event(request.request_id, 1, TokenEventKind.USAGE,
                         payload=(("input_tokens", "3"), ("output_tokens", "8")))
            yield _event(request.request_id, 2, TokenEventKind.FINISH,
                         payload=(("reason", "length"),))

        for choice in ("auto", "required", {"type": "function", "function": {"name": "f"}}):
            for stream in (False, True):
                with self.subTest(choice=choice, stream=stream):
                    backend = _Backend(tool_events)
                    app = CubicASGIApp(backend, model_id="fixture")
                    payload = _openai_payload(stream=stream, include_usage=stream)
                    payload.update(tools=[{"type": "function", "function": {"name": "f"}}],
                                   tool_choice=choice)
                    messages = asyncio.run(_invoke(app, "/v1/chat/completions", payload))
                    self.assertEqual(messages[0]["status"], 200)
                    if stream:
                        chunks = _sse_json(messages)
                        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "length")
                        arguments = "".join(tool["function"]["arguments"] for item in chunks
                            for result in item["choices"]
                            for tool in result["delta"].get("tool_calls", []))
                        usage = chunks[-1]["usage"]
                    else:
                        document = json.loads(_body(messages))
                        result = document["choices"][0]
                        self.assertEqual(result["finish_reason"], "length")
                        arguments = result["message"]["tool_calls"][0]["function"]["arguments"]
                        usage = document["usage"]
                    self.assertEqual(arguments, '{"x":')
                    self.assertEqual(usage["completion_tokens"], 8)
                    self.assertFalse(backend.cancelled)

    def test_openai_stream_sends_each_event_before_requesting_the_next(self) -> None:
        first_sent = asyncio.Event()

        async def gated(request: CanonicalChatRequest) -> AsyncIterator[TokenEvent]:
            yield _event(request.request_id, 0, TokenEventKind.TEXT_DELTA, text="first")
            await first_sent.wait()
            yield _event(request.request_id, 1, TokenEventKind.TEXT_DELTA, text="second")
            yield _event(
                request.request_id,
                2,
                TokenEventKind.USAGE,
                payload=(("input_tokens", "3"), ("output_tokens", "2")),
            )
            yield _event(
                request.request_id,
                3,
                TokenEventKind.FINISH,
                payload=(("reason", "stop"),),
            )

        def observe(
            message: dict[str, object],
            _incoming: asyncio.Queue[dict[str, object]],
        ) -> None:
            body = message.get("body", b"")
            if isinstance(body, bytes) and b'"content":"first"' in body:
                first_sent.set()

        backend = _Backend(gated)
        app = CubicASGIApp(
            backend,
            model_id="fixture",
            request_id_factory=lambda: "request-1",
            clock=lambda: 7,
        )
        messages = asyncio.run(
            _invoke(
                app,
                "/v1/chat/completions",
                _openai_payload(stream=True, include_usage=True),
                on_send=observe,
            )
        )

        chunks = _sse_json(messages)
        self.assertTrue(first_sent.is_set())
        self.assertEqual(messages[0]["status"], 200)
        self.assertEqual(
            chunks[0]["choices"][0]["delta"], {"role": "assistant", "content": ""}
        )
        self.assertEqual(chunks[1]["choices"][0]["delta"], {"content": "first"})
        self.assertEqual(chunks[2]["choices"][0]["delta"], {"content": "second"})
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(chunks[-1]["usage"]["prompt_tokens"], 3)
        self.assertTrue(_body(messages).endswith(b"data: [DONE]\n\n"))
        self.assertEqual(backend.cancelled, [])

    def test_openai_stream_omits_usage_tail_unless_requested(self) -> None:
        backend = _Backend(lambda request: _text_events(request, "answer"))
        app = CubicASGIApp(
            backend,
            model_id="fixture",
            request_id_factory=lambda: "request-1",
            clock=lambda: 7,
        )

        with patch.object(server._LOGGER, "info") as info:
            messages = asyncio.run(
                _invoke(app, "/v1/chat/completions", _openai_payload(stream=True))
            )

        chunks = _sse_json(messages)
        self.assertEqual(info.call_count, 1)
        self.assertEqual(
            info.call_args.args,
            (
                "[POST] path=%s request_id=%s",
                "/v1/chat/completions",
                "request-1",
            ),
        )
        self.assertEqual(
            chunks[0]["choices"][0]["delta"], {"role": "assistant", "content": ""}
        )
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        self.assertFalse(any(chunk.get("choices") == [] for chunk in chunks))
        self.assertFalse(any("usage" in chunk for chunk in chunks))
        self.assertTrue(_body(messages).endswith(b"data: [DONE]\n\n"))

    def test_anthropic_stream_is_incremental_and_has_balanced_lifecycle(self) -> None:
        first_sent = asyncio.Event()

        async def gated(request: CanonicalChatRequest) -> AsyncIterator[TokenEvent]:
            yield _event(request.request_id, 0, TokenEventKind.REASONING_DELTA, text="why")
            await first_sent.wait()
            yield _event(request.request_id, 1, TokenEventKind.TEXT_DELTA, text="answer")
            yield _event(
                request.request_id,
                2,
                TokenEventKind.USAGE,
                payload=(("input_tokens", "3"), ("output_tokens", "2")),
            )
            yield _event(
                request.request_id,
                3,
                TokenEventKind.FINISH,
                payload=(("reason", "stop"),),
            )

        def observe(
            message: dict[str, object],
            _incoming: asyncio.Queue[dict[str, object]],
        ) -> None:
            body = message.get("body", b"")
            if isinstance(body, bytes) and b'"thinking":"why"' in body:
                first_sent.set()

        backend = _Backend(gated)
        app = CubicASGIApp(
            backend, model_id="fixture", request_id_factory=lambda: "request-1"
        )
        messages = asyncio.run(
            _invoke(app, "/v1/messages", _anthropic_payload(stream=True), on_send=observe)
        )

        chunks = _sse_json(messages)
        types = [chunk["type"] for chunk in chunks]
        self.assertTrue(first_sent.is_set())
        self.assertEqual(types[0], "message_start")
        self.assertEqual(types.count("content_block_start"), 2)
        self.assertEqual(types.count("content_block_stop"), 2)
        self.assertEqual(types[-2:], ["message_delta", "message_stop"])
        self.assertNotIn(b"[DONE]", _body(messages))

    def test_non_stream_protocols_fold_the_same_canonical_event_source(self) -> None:
        openai_backend = _Backend(lambda request: _text_events(request, "same"))
        anthropic_backend = _Backend(lambda request: _text_events(request, "same"))
        openai = asyncio.run(
            _invoke(
                CubicASGIApp(
                    openai_backend,
                    model_id="fixture",
                    request_id_factory=lambda: "request-1",
                    clock=lambda: 9,
                ),
                "/v1/chat/completions",
                _openai_payload(stream=False),
            )
        )
        anthropic = asyncio.run(
            _invoke(
                CubicASGIApp(
                    anthropic_backend,
                    model_id="fixture",
                    request_id_factory=lambda: "request-1",
                ),
                "/v1/messages",
                _anthropic_payload(stream=False),
            )
        )

        openai_response = _json_response(openai)
        anthropic_response = _json_response(anthropic)
        self.assertEqual(openai_response["choices"][0]["message"]["content"], "same")
        self.assertEqual(anthropic_response["content"], [{"type": "text", "text": "same"}])
        self.assertEqual(openai_backend.requests, anthropic_backend.requests)

    def test_disconnect_aborts_the_admitted_request_once(self) -> None:
        never = asyncio.Event()

        async def blocked(request: CanonicalChatRequest) -> AsyncIterator[TokenEvent]:
            yield _event(request.request_id, 0, TokenEventKind.TEXT_DELTA, text="visible")
            await never.wait()

        disconnected = False

        def disconnect(
            message: dict[str, object],
            incoming: asyncio.Queue[dict[str, object]],
        ) -> None:
            nonlocal disconnected
            body = message.get("body", b"")
            if not disconnected and isinstance(body, bytes) and b'"content":"visible"' in body:
                disconnected = True
                incoming.put_nowait({"type": "http.disconnect"})

        backend = _Backend(blocked)
        app = CubicASGIApp(
            backend,
            model_id="fixture",
            request_id_factory=lambda: "request-disconnect",
        )
        messages = asyncio.run(
            _invoke(app, "/v1/chat/completions", _openai_payload(stream=True), on_send=disconnect)
        )

        self.assertEqual(backend.cancelled, ["request-disconnect"])
        self.assertNotIn(b"[DONE]", _body(messages))
        self.assertFalse(any(message.get("more_body") is False for message in messages[1:]))

    def test_strict_json_and_normalization_fail_before_backend_admission(self) -> None:
        backend = _Backend(lambda request: _text_events(request, "unused"))
        app = CubicASGIApp(
            backend, model_id="fixture", request_id_factory=lambda: "request-1"
        )
        duplicate = b'{"model":"fixture","model":"other","messages":[],"max_completion_tokens":1}'
        duplicate_response = asyncio.run(
            _invoke(app, "/v1/chat/completions", {}, raw_body=duplicate)
        )
        unknown = dict(_anthropic_payload(stream=False), unknown=True)
        unknown_response = asyncio.run(_invoke(app, "/v1/messages", unknown))
        media_type_response = asyncio.run(
            _invoke(
                app,
                "/v1/chat/completions",
                _openai_payload(stream=False),
                content_type=b"text/plain",
            )
        )

        self.assertEqual(duplicate_response[0]["status"], 400)
        duplicate_error = _json_response(duplicate_response)["error"]
        self.assertIsInstance(duplicate_error, dict)
        self.assertEqual(duplicate_error["type"], "invalid_request_error")
        self.assertEqual(unknown_response[0]["status"], 400)
        self.assertEqual(_json_response(unknown_response)["type"], "error")
        self.assertEqual(media_type_response[0]["status"], 415)
        self.assertEqual(backend.requests, [])

    def test_backend_admission_errors_have_protocol_specific_json_envelopes(self) -> None:
        error = ChatBackendError(BackendErrorKind.OVERLOADED, "capacity exhausted")
        openai = asyncio.run(
            _invoke(
                CubicASGIApp(
                    _Backend(lambda request: _text_events(request), start_error=error),
                    model_id="fixture",
                ),
                "/v1/chat/completions",
                _openai_payload(stream=False),
            )
        )
        anthropic = asyncio.run(
            _invoke(
                CubicASGIApp(
                    _Backend(lambda request: _text_events(request), start_error=error),
                    model_id="fixture",
                ),
                "/v1/messages",
                _anthropic_payload(stream=False),
            )
        )

        self.assertEqual(openai[0]["status"], 429)
        self.assertEqual(_json_response(openai)["error"]["type"], "rate_limit_error")
        self.assertEqual(anthropic[0]["status"], 429)
        self.assertEqual(_json_response(anthropic)["error"]["type"], "overloaded_error")

    def test_private_backend_diagnostics_stay_out_of_json_and_sse(self) -> None:
        private = "worker /private/model/index.json failed; increase --gpu-memory-gib"
        with self.assertRaisesRegex(ValueError, "must be reviewed"):
            ChatBackendError(
                BackendErrorKind.INVALID_REQUEST, private,
                public_message=private,
            )
        for kind, expected_status in (
            (BackendErrorKind.INVALID_REQUEST, 400),
            (BackendErrorKind.UNAVAILABLE, 503),
        ):
            error = ChatBackendError(kind, private)
            for path, payload in (
                ("/v1/chat/completions", _openai_payload(stream=False)),
                ("/v1/messages", _anthropic_payload(stream=False)),
            ):
                response = asyncio.run(_invoke(
                    CubicASGIApp(
                        _Backend(lambda request: _text_events(request), start_error=error),
                        model_id="fixture",
                    ),
                    path,
                    payload,
                ))
                self.assertEqual(response[0]["status"], expected_status)
                self.assertNotIn(b"/private/", _body(response))
                self.assertNotIn(b"--gpu-memory-gib", _body(response))

        async def interrupted(request: CanonicalChatRequest) -> AsyncIterator[TokenEvent]:
            yield _event(request.request_id, 0, TokenEventKind.TEXT_DELTA, text="partial")
            raise ChatBackendError(BackendErrorKind.INVALID_REQUEST, private)

        for path, payload in (
            ("/v1/chat/completions", _openai_payload(stream=True)),
            ("/v1/messages", _anthropic_payload(stream=True)),
        ):
            response = asyncio.run(_invoke(
                CubicASGIApp(_Backend(interrupted), model_id="fixture"),
                path,
                payload,
            ))
            self.assertEqual(response[0]["status"], 200)
            self.assertNotIn(b"/private/", _body(response))
            self.assertNotIn(b"--gpu-memory-gib", _body(response))
            errors = [chunk["error"] for chunk in _sse_json(response) if "error" in chunk]
            self.assertEqual(errors[-1]["message"], "invalid request")

    def test_request_validation_does_not_echo_arbitrary_field_or_json_key(self) -> None:
        app = CubicASGIApp(
            _Backend(lambda request: _text_events(request)), model_id="fixture"
        )
        payload = dict(_openai_payload(stream=False))
        payload["/private/model/secret --gpu-memory-gib"] = True
        unknown = asyncio.run(_invoke(app, "/v1/chat/completions", payload))
        duplicate = asyncio.run(_invoke(
            app,
            "/v1/chat/completions",
            {},
            raw_body=b'{"model":"fixture","private --gpu-memory-gib":1,'
                     b'"private --gpu-memory-gib":2}',
        ))
        for response in (unknown, duplicate):
            self.assertEqual(response[0]["status"], 400)
            self.assertNotIn(b"/private/", _body(response))
            self.assertNotIn(b"--gpu-memory-gib", _body(response))

    def test_openai_error_before_first_stream_chunk_remains_plain_json(self) -> None:
        async def rejected(request: CanonicalChatRequest) -> AsyncIterator[TokenEvent]:
            if request.request_id:
                raise ChatBackendError(BackendErrorKind.OVERLOADED, "capacity exhausted")
            yield _event(request.request_id, 0, TokenEventKind.TEXT_DELTA, text="unreachable")

        backend = _Backend(rejected)
        response = asyncio.run(
            _invoke(
                CubicASGIApp(
                    backend,
                    model_id="fixture",
                    request_id_factory=lambda: "request-rejected",
                ),
                "/v1/chat/completions",
                _openai_payload(stream=True, include_usage=True),
            )
        )

        self.assertEqual(response[0]["status"], 429)
        self.assertIn(
            (b"content-type", b"application/json; charset=utf-8"),
            response[0]["headers"],
        )
        self.assertEqual(_json_response(response)["error"]["type"], "rate_limit_error")
        self.assertNotIn(b"data: ", _body(response))
        self.assertEqual(backend.cancelled, ["request-rejected"])

        async def invalid(request: CanonicalChatRequest) -> AsyncIterator[TokenEvent]:
            yield _event(request.request_id, 1, TokenEventKind.TEXT_DELTA, text="bad")

        invalid_backend = _Backend(invalid)
        invalid_response = asyncio.run(
            _invoke(
                CubicASGIApp(
                    invalid_backend,
                    model_id="fixture",
                    request_id_factory=lambda: "request-invalid",
                ),
                "/v1/chat/completions",
                _openai_payload(stream=True),
            )
        )
        self.assertEqual(invalid_response[0]["status"], 500)
        self.assertEqual(_json_response(invalid_response)["error"]["type"], "server_error")
        self.assertNotIn(b"data: ", _body(invalid_response))
        self.assertEqual(invalid_backend.cancelled, ["request-invalid"])

    def test_openai_error_after_first_stream_chunk_stays_sse_then_done(self) -> None:
        async def interrupted(request: CanonicalChatRequest) -> AsyncIterator[TokenEvent]:
            yield _event(request.request_id, 0, TokenEventKind.TEXT_DELTA, text="partial")
            raise ChatBackendError(BackendErrorKind.UNAVAILABLE, "worker unavailable")

        backend = _Backend(interrupted)
        response = asyncio.run(
            _invoke(
                CubicASGIApp(
                    backend,
                    model_id="fixture",
                    request_id_factory=lambda: "request-interrupted",
                ),
                "/v1/chat/completions",
                _openai_payload(stream=True, include_usage=True),
            )
        )

        chunks = _sse_json(response)
        self.assertEqual(response[0]["status"], 200)
        self.assertIn(
            (b"content-type", b"text/event-stream; charset=utf-8"),
            response[0]["headers"],
        )
        self.assertEqual(
            chunks[0]["choices"][0]["delta"], {"role": "assistant", "content": ""}
        )
        self.assertEqual(chunks[1]["choices"][0]["delta"], {"content": "partial"})
        self.assertEqual(chunks[-1]["error"]["type"], "server_error")
        self.assertTrue(_body(response).endswith(b"data: [DONE]\n\n"))
        self.assertEqual(backend.cancelled, ["request-interrupted"])

    def test_stream_transport_failure_after_start_does_not_restart_response(self) -> None:
        failed = False

        def fail_first_body(
            message: dict[str, object],
            _incoming: asyncio.Queue[dict[str, object]],
        ) -> None:
            nonlocal failed
            if not failed and message.get("type") == "http.response.body":
                failed = True
                raise OSError("transport closed")

        backend = _Backend(lambda request: _text_events(request, "answer"))
        response = asyncio.run(
            _invoke(
                CubicASGIApp(
                    backend,
                    model_id="fixture",
                    request_id_factory=lambda: "request-transport",
                ),
                "/v1/chat/completions",
                _openai_payload(stream=True),
                on_send=fail_first_body,
            )
        )

        starts = [message for message in response if message["type"] == "http.response.start"]
        self.assertTrue(failed)
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0]["status"], 200)
        self.assertIn(
            (b"content-type", b"text/event-stream; charset=utf-8"),
            starts[0]["headers"],
        )
        self.assertEqual(backend.cancelled, ["request-transport"])

    def test_json_transport_failure_after_start_does_not_restart_response(self) -> None:
        failed = False

        def fail_body(
            message: dict[str, object],
            _incoming: asyncio.Queue[dict[str, object]],
        ) -> None:
            nonlocal failed
            if not failed and message.get("type") == "http.response.body":
                failed = True
                raise OSError("transport closed")

        backend = _Backend(lambda request: _text_events(request, "answer"))
        response = asyncio.run(
            _invoke(
                CubicASGIApp(
                    backend,
                    model_id="fixture",
                    request_id_factory=lambda: "request-transport",
                ),
                "/v1/chat/completions",
                _openai_payload(stream=False),
                on_send=fail_body,
            )
        )

        starts = [message for message in response if message["type"] == "http.response.start"]
        self.assertTrue(failed)
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0]["status"], 200)
        self.assertIn(
            (b"content-type", b"application/json; charset=utf-8"),
            starts[0]["headers"],
        )
        self.assertEqual(backend.cancelled, ["request-transport"])

    def test_invalid_canonical_non_stream_is_500_and_cancelled(self) -> None:
        async def invalid(request: CanonicalChatRequest) -> AsyncIterator[TokenEvent]:
            yield _event(request.request_id, 1, TokenEventKind.TEXT_DELTA, text="bad")

        backend = _Backend(invalid)
        response = asyncio.run(
            _invoke(
                CubicASGIApp(
                    backend,
                    model_id="fixture",
                    request_id_factory=lambda: "request-invalid",
                ),
                "/v1/messages",
                _anthropic_payload(stream=False),
            )
        )

        self.assertEqual(response[0]["status"], 500)
        self.assertEqual(_json_response(response)["error"]["type"], "api_error")
        self.assertEqual(backend.cancelled, ["request-invalid"])

    def test_invalid_canonical_stream_emits_typed_sse_error_without_success_terminal(self) -> None:
        async def invalid(request: CanonicalChatRequest) -> AsyncIterator[TokenEvent]:
            yield _event(request.request_id, 0, TokenEventKind.TEXT_DELTA, text="partial")
            yield _event(request.request_id, 2, TokenEventKind.TEXT_DELTA, text="bad")

        backend = _Backend(invalid)
        response = asyncio.run(
            _invoke(
                CubicASGIApp(
                    backend,
                    model_id="fixture",
                    request_id_factory=lambda: "request-invalid",
                ),
                "/v1/messages",
                _anthropic_payload(stream=True),
            )
        )

        chunks = _sse_json(response)
        self.assertEqual(chunks[-2]["type"], "error")
        self.assertEqual(chunks[-2]["error"]["type"], "api_error")
        self.assertEqual(chunks[-1]["type"], "message_stop")
        self.assertEqual(backend.cancelled, ["request-invalid"])


if __name__ == "__main__":
    unittest.main()

class ContinuousUsageProjectionTests(unittest.TestCase):
    def test_reasoning_progress_is_included_in_continuous_usage(self):
        from cubic_serve.inputs import normalize_openai_chat_completions

        payload = _openai_payload(stream=True, include_usage=True)
        payload["stream_options"]["continuous_usage_stats"] = True
        request, _ = normalize_openai_chat_completions(payload, request_id="request-1")
        projector = server._StreamProjector(server._Protocol.OPENAI, request, 0)
        projector.start(7)
        event = TokenEvent(
            "request-1", 0, TokenEventKind.REASONING_DELTA, text="why",
            output_token_count=3, reasoning_token_count=2,
        )
        value = json.loads(projector.delta_frame(event)[6:])
        self.assertEqual(value["usage"], {
            "prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10,
            "completion_tokens_details": {"reasoning_tokens": 2},
        })

    def test_progress_is_emitted_even_for_empty_text(self):
        from cubic_serve.inputs import normalize_openai_chat_completions
        payload = _openai_payload(stream=True, include_usage=True)
        payload["stream_options"]["continuous_usage_stats"] = True
        normalized = normalize_openai_chat_completions(payload, request_id="request-1")
        request = normalized[0]
        projector = server._StreamProjector(server._Protocol.OPENAI, request, 0)
        projector.start(7)
        event = TokenEvent("request-1", 0, TokenEventKind.TEXT_DELTA, text="", output_token_count=1)
        value = json.loads(projector.delta_frame(event)[6:])
        self.assertEqual(value["choices"][0]["delta"], {"content": ""})
        self.assertEqual(value["usage"], {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8})
