from __future__ import annotations

import asyncio
import json
import unittest
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import replace
from unittest.mock import patch

from cubic_serve.api.server import (
    BackendErrorKind,
    ChatBackendError,
    ChatExecution,
    CubicASGIApp,
)
from cubic_serve.contracts.model import WeightVariantMeta
from cubic_serve.contracts.request import (
    CanonicalChatRequest,
    ChatContentPart,
    ChatMessage,
    ChatRole,
    ChatTemplateKwargs,
    ContentPart,
    DecodePositionPlan,
    InputPlan,
    SamplingMeta,
    StructuredOutputMeta,
    TokenEvent,
    TokenEventKind,
    TokenPosition,
)
from cubic_serve.inputs.chat import ChatProcessorError
from cubic_serve.inputs.multimodal import EncoderAdmissionError
from cubic_serve.runtime.performance_chat import (
    PerformanceChatBackend,
    PerformanceGenerationMetadata,
)

MODEL_ID = "qwen3_5:fixture"
WEIGHT_VARIANT = WeightVariantMeta(MODEL_ID, "fixture-revision")


class _Processor:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.thinking_modes: list[bool] = []

    def build_chat_input_plan(
        self,
        request: CanonicalChatRequest,
        *,
        weight_variant: WeightVariantMeta,
        enable_thinking: bool,
    ) -> InputPlan:
        self.calls.append(request.request_id)
        self.thinking_modes.append(enable_thinking)
        return InputPlan(
            request_id=request.request_id,
            model_id=request.model_id,
            processor_recipe_digest=f"b2s256:{'a' * 64}",
            weight_variant=weight_variant,
            content_parts=(ContentPart(0, "text", text="fixture"),),
            token_ids=(11, 12, 13),
            positions=tuple(TokenPosition((index,) * 3) for index in range(3)),
            decode_positions=DecodePositionPlan(TokenPosition((3, 3, 3)), (1, 1, 1)),
        )


class _Decoder:
    def __init__(self) -> None:
        self.closed = False

    def decode(self, token_id: int) -> str:
        if token_id == _Codec.eos_token_id:
            return ""
        return chr(ord("a") + token_id - 1)

    def finish(self) -> str:
        self.closed = True
        return ""


class _Codec:
    eos_token_id = 9

    def __init__(self) -> None:
        self.admissions: list[tuple[bool, bool]] = []
        self.parser_modes: list[tuple[bool, bool, bool]] = []

    def admit_parser_recipe(
        self,
        *,
        reasoning_parser: bool,
        tool_call_parser: bool,
    ) -> None:
        self.admissions.append((reasoning_parser, tool_call_parser))

    def new_decoder(self) -> _Decoder:
        return _Decoder()

    def new_parser(
        self,
        *,
        enable_thinking: bool,
        reasoning_parser: bool,
        tool_call_parser: bool,
    ) -> None:
        self.parser_modes.append(
            (enable_thinking, reasoning_parser, tool_call_parser)
        )
        return None


class _SyncExecutor:
    def __init__(
        self,
        tokens: tuple[int, ...] = (1, 2),
        *,
        max_sequence_tokens: int = 4096,
    ) -> None:
        self.tokens = tokens
        self.max_sequence_tokens = max_sequence_tokens
        self.calls: list[tuple[tuple[int, ...], int, PerformanceGenerationMetadata]] = []
        self.cancelled: list[str] = []

    @property
    def ready(self) -> bool:
        return True

    def generate(
        self,
        input_ids: tuple[int, ...],
        max_tokens: int,
        metadata: PerformanceGenerationMetadata,
    ) -> tuple[int, ...] | AsyncIterator[int]:
        self.calls.append((input_ids, max_tokens, metadata))
        return self.tokens

    def cancel(self, request_id: str) -> Awaitable[None] | None:
        self.cancelled.append(request_id)
        return None


class _GatedSyncExecutor(_SyncExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.second_started = asyncio.Event()
        self.release_second = asyncio.Event()
        self.finished = asyncio.Event()

    def generate(
        self,
        input_ids: tuple[int, ...],
        max_tokens: int,
        metadata: PerformanceGenerationMetadata,
    ) -> AsyncIterator[int]:
        self.calls.append((input_ids, max_tokens, metadata))

        async def tokens() -> AsyncIterator[int]:
            yield 1
            self.second_started.set()
            await asyncio.wait_for(self.release_second.wait(), 2)
            self.finished.set()
            yield 2

        return tokens()


class _AsyncExecutor(_SyncExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False
        self.overrun = False

    def generate(
        self,
        input_ids: tuple[int, ...],
        max_tokens: int,
        metadata: PerformanceGenerationMetadata,
    ) -> AsyncIterator[int]:
        self.calls.append((input_ids, max_tokens, metadata))

        async def tokens() -> AsyncIterator[int]:
            try:
                yield 3
                yield _Codec.eos_token_id
                self.overrun = True
                yield 4
            finally:
                self.closed = True

        return tokens()

    async def cancel(self, request_id: str) -> None:
        await asyncio.sleep(0)
        self.cancelled.append(request_id)


def _request(
    request_id: str,
    *,
    maximum: int = 2,
    structured_output: StructuredOutputMeta | None = None,
    chat_template_kwargs: dict[str, object] | None = None,
    sampling: SamplingMeta | None = None,
    ignore_eos: bool = False,
) -> CanonicalChatRequest:
    return CanonicalChatRequest(
        request_id=request_id,
        model_id=MODEL_ID,
        messages=(
            ChatMessage(
                ChatRole.USER,
                (ChatContentPart(0, "text", text="hello"),),
            ),
        ),
        max_new_tokens=maximum,
        structured_output=structured_output or StructuredOutputMeta.identity(),
        chat_template_kwargs=ChatTemplateKwargs.from_value(chat_template_kwargs or {}),
        sampling=sampling or SamplingMeta(),
        ignore_eos=ignore_eos,
    )


def _backend(
    executor: object,
    processor: _Processor | None = None,
    codec: _Codec | None = None,
    *,
    enable_thinking: bool = False,
    reasoning_parser: bool = False,
    tool_call_parser: bool = False,
) -> PerformanceChatBackend:
    return PerformanceChatBackend(
        executor,  # type: ignore[arg-type]
        model_id=MODEL_ID,
        weight_variant=WEIGHT_VARIANT,
        processor=processor or _Processor(),  # type: ignore[arg-type]
        output_codec=codec or _Codec(),  # type: ignore[arg-type]
        enable_thinking=enable_thinking,
        reasoning_parser=reasoning_parser,
        tool_call_parser=tool_call_parser,
    )


async def _collect(
    backend: PerformanceChatBackend, request_id: str
) -> tuple[ChatExecution, tuple[TokenEvent, ...]]:
    execution = await backend.start(_request(request_id))
    return execution, tuple([event async for event in execution.events])


async def _invoke(
    app: CubicASGIApp,
    path: str,
    payload: dict[str, object],
    *,
    on_send: Callable[[dict[str, object]], None] | None = None,
) -> list[dict[str, object]]:
    incoming: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    await incoming.put(
        {
            "type": "http.request",
            "body": json.dumps(payload).encode(),
            "more_body": False,
        }
    )
    sent: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return await incoming.get()

    async def send(message: dict[str, object]) -> None:
        sent.append(message)
        if on_send is not None:
            on_send(message)

    await app(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
        send,
    )
    return sent


def _body(messages: list[dict[str, object]]) -> bytes:
    return b"".join(
        value
        for message in messages
        if message.get("type") == "http.response.body"
        and isinstance((value := message.get("body", b"")), bytes)
    )


class PerformanceChatBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_optional_token_limits_through_public_chat_path(self) -> None:
        for stream in (False, True):
            for fields, expected in (
                ({}, 3),
                ({"max_tokens": None, "max_completion_tokens": None}, 3),
                ({"max_tokens": 2, "max_completion_tokens": None}, 2),
                ({"max_tokens": None, "max_completion_tokens": 1}, 1),
                ({"max_tokens": 2, "max_completion_tokens": 1}, 1),
            ):
                with self.subTest(stream=stream, fields=fields):
                    executor = _SyncExecutor((1, 2, 3, 4), max_sequence_tokens=6)
                    app = CubicASGIApp(_backend(executor), model_id=MODEL_ID)
                    response = await _invoke(app, "/v1/chat/completions", {
                        "model": MODEL_ID,
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": stream,
                        **fields,
                    })
                    self.assertEqual(response[0]["status"], 200)
                    self.assertEqual(executor.calls[0][1], expected)
                    if stream:
                        self.assertIn(b'"finish_reason":"length"', _body(response))
                    else:
                        document = json.loads(_body(response))
                        self.assertEqual(document["usage"]["completion_tokens"], expected)
                        self.assertEqual(document["choices"][0]["finish_reason"], "length")

    async def test_omitted_token_limit_preserves_eos(self) -> None:
        executor = _SyncExecutor((1, _Codec.eos_token_id, 2), max_sequence_tokens=6)
        backend = _backend(executor)
        execution = await backend.start(replace(_request("default-eos"), max_new_tokens=None))
        events = [event async for event in execution.events]
        self.assertEqual(executor.calls[0][1], 3)
        self.assertEqual(dict(events[-1].payload_items)["reason"], "stop")

    async def test_omitted_limit_rejects_exhausted_prompt_before_headers(self) -> None:
        for capacity in (2, 3):
            with self.subTest(capacity=capacity):
                executor = _SyncExecutor(max_sequence_tokens=capacity)
                backend = _backend(executor)
                app = CubicASGIApp(backend, model_id=MODEL_ID,
                                   request_id_factory=lambda: "full-context")
                response = await _invoke(app, "/v1/chat/completions", {
                    "model": MODEL_ID,
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                })
                self.assertEqual(response[0]["status"], 400)
                self.assertIn(b"request exceeds available context length", _body(response))
                self.assertEqual(executor.calls, [])
                self.assertNotIn("full-context", backend._known_request_ids)

    async def test_non_strict_auto_tools_keep_native_parsing_without_a_grammar(self):
        from cubic_serve.contracts.request import StructuredOutputKind
        from cubic_serve.inputs.normalize import normalize_openai_chat_completions

        class Processor(_Processor):
            def build_chat_input_plan(self, request, **kwargs):
                self.request = request
                return super().build_chat_input_plan(request, **kwargs)

        processor, executor, codec = Processor(), _SyncExecutor((1, 9, 2, 3)), _Codec()
        request, _ = normalize_openai_chat_completions({
            "model": MODEL_ID, "messages": [{"role": "user", "content": "question"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "tool_choice": "auto", "ignore_eos": True, "max_tokens": 4,
        }, request_id="auto-native")
        backend = _backend(executor, processor, codec, tool_call_parser=True)
        with patch("cubic_serve.generation.xgrammar.frontend_constraint_factory",
                   side_effect=AssertionError("non-strict auto tools must not compile a grammar")):
            execution = await backend.start(request)
            tuple([event async for event in execution.events])
        self.assertIs(processor.request, request)
        self.assertIs(processor.request.structured_output.kind, StructuredOutputKind.TOOL_SCHEMA)
        self.assertIs(executor.calls[0][2].structured_output.kind, StructuredOutputKind.IDENTITY)
        self.assertTrue(executor.calls[0][2].ignore_eos)
        self.assertEqual(codec.parser_modes, [(False, False, True)])

    async def test_budget_only_request_carries_prompt_seed_without_json_compilation(self):
        executor = _SyncExecutor((8, 2, 9))
        executor.supports_structured_output = True
        codec = _Codec()
        codec.reasoning_start_token_ids = (12,)
        codec.reasoning_end_token_ids = (8,)
        backend = _backend(executor, codec=codec, reasoning_parser=True)
        request = replace(_request("budget", maximum=4), thinking_token_budget=0)
        with patch("cubic_serve.generation.xgrammar.frontend_constraint_factory",
                   side_effect=AssertionError("budget-only frontend must not compile JSON")):
            execution = await backend.start(request)
            events = tuple([event async for event in execution.events])
        control = executor.calls[0][2].reasoning_control
        self.assertTrue(control.initially_reasoning)
        self.assertEqual(control.initial_reasoning_tokens, 1)
        self.assertEqual(dict(events[-2].payload_items)["reasoning_tokens"], "0")
        self.assertEqual(dict(events[-2].payload_items)["output_tokens"], "3")

    async def test_budget_admission_requires_parser_and_rejects_initial_grammar_conflict(self):
        from cubic_serve.generation.xgrammar import XGrammarConstraintFactory

        executor = _SyncExecutor()
        executor.supports_structured_output = True
        processor = _Processor()
        backend = _backend(executor, processor=processor)
        with self.assertRaises(ChatBackendError) as context:
            await backend.start(replace(_request("no-parser"), thinking_token_budget=0))
        self.assertIs(context.exception.kind, BackendErrorKind.INVALID_REQUEST)
        self.assertEqual(processor.calls, [])
        codec = _Codec()
        codec.reasoning_start_token_ids, codec.reasoning_end_token_ids = (12,), (8,)
        backend = PerformanceChatBackend(
            executor, model_id=MODEL_ID, weight_variant=WEIGHT_VARIANT,
            processor=processor, output_codec=codec, reasoning_parser=True,
            structured_output_in_reasoning=True,
        )
        backend._constraint_factory = XGrammarConstraintFactory(
            tuple(chr(ord("a") + i - 1) for i in range(16)),
            vocab_type="raw", vocab_size=16, stop_token_ids=(9,),
        )
        request = replace(_request("conflict", structured_output=StructuredOutputMeta.regex("b")),
                          thinking_token_budget=0)
        with self.assertRaises(ChatBackendError) as context:
            await backend.start(request)
        self.assertIs(context.exception.kind, BackendErrorKind.INVALID_REQUEST)
        self.assertNotIn(request.request_id, backend._known_request_ids)
        self.assertEqual(executor.calls, [])

    async def test_public_stream_stops_on_secondary_eos_without_following_content(self):
        codec = _Codec()
        codec.eos_token_ids = (8, 9)
        app = CubicASGIApp(
            _backend(_SyncExecutor((1, 8, 2, 9)), codec=codec), model_id=MODEL_ID,
        )
        messages = await _invoke(app, "/v1/chat/completions", {
            "model": MODEL_ID, "messages": [{"role": "user", "content": "hello"}],
            "max_completion_tokens": 4, "stream": True,
            "stream_options": {"include_usage": True},
        })
        payloads = [line[6:] for line in _body(messages).decode().splitlines()
                    if line.startswith("data: ")]
        self.assertEqual(payloads[-1], "[DONE]")
        chunks = [json.loads(value) for value in payloads[:-1]]
        choices = [choice for chunk in chunks for choice in chunk.get("choices", [])]
        self.assertEqual("".join(choice.get("delta", {}).get("content", "")
                                 for choice in choices), "a")
        self.assertEqual([choice["finish_reason"] for choice in choices
                          if choice.get("finish_reason")], ["stop"])

    async def test_secondary_eos_stops_before_following_tokens_and_respects_ignore_eos(self):
        codec = _Codec()
        codec.eos_token_ids = (8, 9)
        for ignore in (False, True):
            with self.subTest(ignore_eos=ignore):
                executor = _SyncExecutor((1, 8, 2, 9))
                backend = _backend(executor, codec=codec)
                execution = await backend.start(_request(
                    f"secondary-eos-{ignore}", maximum=4, ignore_eos=ignore,
                ))
                events = tuple([event async for event in execution.events])
                content = "".join(
                    e.text or "" for e in events if e.kind is TokenEventKind.TEXT_DELTA
                )
                self.assertEqual(content, "ahb" if ignore else "a")
                self.assertEqual(
                    dict(events[-1].payload_items)["reason"], "length" if ignore else "stop"
                )
                self.assertEqual(
                    dict(events[-2].payload_items)["output_tokens"], "4" if ignore else "2"
                )

    async def test_request_thinking_override_drives_explicit_reasoning_parser(self) -> None:
        for ordinal, (service_default, request_value) in enumerate(
            ((False, True), (True, False))
        ):
            with self.subTest(
                service_default=service_default,
                request_value=request_value,
            ):
                processor = _Processor()
                codec = _Codec()
                backend = _backend(
                    _SyncExecutor(),
                    processor,
                    codec,
                    enable_thinking=service_default,
                    reasoning_parser=True,
                )
                execution = await backend.start(
                    _request(
                        f"thinking-{ordinal}",
                        chat_template_kwargs={"enable_thinking": request_value},
                    )
                )
                tuple([event async for event in execution.events])

                self.assertEqual(processor.thinking_modes, [request_value])
                self.assertEqual(codec.admissions, [(True, False)])
                self.assertEqual(
                    codec.parser_modes, [(True, True, False)] if request_value else []
                )

    async def test_default_output_path_does_not_construct_any_parser(self) -> None:
        codec = _Codec()
        execution = await _backend(
            _SyncExecutor(),
            codec=codec,
            enable_thinking=True,
        ).start(_request("no-parser"))
        tuple([event async for event in execution.events])

        self.assertEqual(codec.admissions, [])
        self.assertEqual(codec.parser_modes, [])

    async def test_tool_request_without_parser_is_rejected_before_processing(self) -> None:
        processor = _Processor()
        executor = _SyncExecutor((1,))
        request = _request(
            "tool-without-parser",
            structured_output=StructuredOutputMeta.tool_schema(
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object"},
                    },
                }
            ),
        )

        with self.assertRaises(ChatBackendError) as raised:
            await _backend(executor, processor).start(request)

        self.assertEqual(raised.exception.kind, BackendErrorKind.INVALID_REQUEST)
        self.assertIn("explicitly configured tool-call parser", str(raised.exception))
        self.assertEqual((processor.calls, executor.calls), ([], []))

    async def test_encoder_budget_rejection_is_an_invalid_request(self) -> None:
        processor = _Processor()
        executor = _SyncExecutor()
        app = CubicASGIApp(
            _backend(executor, processor),
            model_id=MODEL_ID,
            request_id_factory=lambda: "oversize-image",
        )
        with patch.object(processor, "build_chat_input_plan",
                          side_effect=EncoderAdmissionError(
                              "required_input_rows=7688 admitted_input_rows=4096; "
                              "increase --multimodal-memory-gib",
                              public_message="image or video frame is too large")):
            with self.assertLogs("cubic_serve.api.server", level="ERROR") as logs:
                response = await _invoke(
                    app,
                    "/v1/chat/completions",
                    {
                        "model": MODEL_ID,
                        "messages": [{"role": "user", "content": "oversize image"}],
                        "max_tokens": 3,
                        "stream": True,
                    },
                )
        self.assertEqual(response[0]["status"], 400)
        self.assertIn(b"image or video frame is too large", _body(response))
        self.assertNotIn(b"--multimodal-memory-gib", _body(response))
        self.assertNotIn(b"required_input_rows", _body(response))
        self.assertIn("increase --multimodal-memory-gib", "\n".join(logs.output))
        self.assertNotIn(b"data:", _body(response))
        self.assertEqual(executor.calls, [])

    async def test_processor_diagnostic_paths_stay_in_server_log(self) -> None:
        processor = _Processor()
        executor = _SyncExecutor()
        app = CubicASGIApp(
            _backend(executor, processor),
            model_id=MODEL_ID,
            request_id_factory=lambda: "processor-failed",
        )
        diagnostic = "cannot read /private/model/tokenizer.json; --chat-template is invalid"
        with patch.object(processor, "build_chat_input_plan",
                          side_effect=ChatProcessorError(diagnostic)):
            with self.assertLogs("cubic_serve.api.server", level="ERROR") as logs:
                response = await _invoke(
                    app,
                    "/v1/chat/completions",
                    {
                        "model": MODEL_ID,
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 3,
                        "stream": True,
                    },
                )
        self.assertEqual(response[0]["status"], 400)
        self.assertIn(b"request input could not be processed", _body(response))
        self.assertNotIn(b"/private/", _body(response))
        self.assertNotIn(b"--chat-template", _body(response))
        self.assertIn(diagnostic, "\n".join(logs.output))
        self.assertEqual(executor.calls, [])

    async def test_context_capacity_rejects_before_stream_http_200(self) -> None:
        executor = _SyncExecutor((1,), max_sequence_tokens=4)
        processor = _Processor()
        app = CubicASGIApp(
            _backend(executor, processor),
            model_id=MODEL_ID,
            request_id_factory=lambda: "over-context",
            clock=lambda: 7,
        )

        response = await _invoke(
            app,
            "/v1/chat/completions",
            {
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 3,
                "stream": True,
            },
        )

        self.assertEqual(response[0]["status"], 400)
        self.assertNotIn(b"data:", _body(response))
        self.assertIn(b"request exceeds available context length", _body(response))
        self.assertNotIn(b"qualified performance context capacity", _body(response))
        self.assertEqual(processor.calls, ["over-context"])
        self.assertEqual(executor.calls, [])

    async def test_cache_overload_rejects_before_headers_for_both_protocols(self) -> None:
        from cubic_serve.runtime.performance import Qwen3_5TP2PP2Overloaded

        for asynchronous in (False, True):
            for path in ("/v1/chat/completions", "/v1/messages"):
                for stream in (False, True):
                    executor = _SyncExecutor()
                    backend = _backend(executor)
                    private = "KV/state admission capacity exhausted /private/model"

                    def refuse(*args):
                        raise Qwen3_5TP2PP2Overloaded(private)

                    async def refuse_async(*args):
                        refuse(*args)

                    executor.generate = refuse_async if asynchronous else refuse
                    app = CubicASGIApp(backend, model_id=MODEL_ID,
                                       request_id_factory=lambda: "full-pool")
                    response = await _invoke(app, path, {
                        "model": MODEL_ID,
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 2, "stream": stream,
                    })
                    self.assertEqual(response[0]["status"], 429)
                    document = json.loads(_body(response))
                    self.assertEqual(document["error"]["message"],
                                     "The model is currently at capacity. Please try again later.")
                    self.assertNotIn(private.encode(), _body(response))
                    self.assertNotIn(b"data:", _body(response))
                    self.assertFalse(backend._active)
                    self.assertNotIn("full-pool", backend._known_request_ids)

    async def test_admission_opens_source_without_waiting_for_a_token(self) -> None:
        from cubic_serve.runtime.token_stream import TokenStream

        executor = _SyncExecutor()
        stream = TokenStream()
        executor.generate = lambda *args: stream
        backend = _backend(executor)
        execution = await asyncio.wait_for(backend.start(_request("no-token-yet")), 1)
        self.assertEqual(execution.input_tokens, 3)
        await backend.cancel("no-token-yet")
        self.assertFalse(backend._active)
        self.assertTrue(stream._released)
        self.assertEqual(tuple([event async for event in execution.events]), ())

    async def test_capacity_exhaustion_during_generation_is_nonfatal_public_error(self) -> None:
        from cubic_serve.runtime.performance import Qwen3_5TP2PP2Overloaded

        executor = _SyncExecutor()

        async def tokens():
            yield 1
            raise Qwen3_5TP2PP2Overloaded("private cache capacity detail")

        executor.generate = lambda *args: tokens()
        backend = _backend(executor)
        execution = await backend.start(_request("growth-full"))
        with self.assertRaises(ChatBackendError) as raised:
            tuple([event async for event in execution.events])
        self.assertEqual(raised.exception.kind, BackendErrorKind.OVERLOADED)
        self.assertEqual(raised.exception.public_message,
                         "The model is currently at capacity. Please try again later.")
        self.assertFalse(backend._active)
        self.assertTrue(backend.ready)

    async def test_sync_greedy_tokens_use_the_canonical_output_pipeline(self) -> None:
        executor = _SyncExecutor()
        execution, events = await _collect(_backend(executor), "sync")

        self.assertEqual(execution.input_tokens, 3)
        self.assertEqual(
            [event.kind for event in events],
            [
                TokenEventKind.TEXT_DELTA,
                TokenEventKind.TEXT_DELTA,
                TokenEventKind.USAGE,
                TokenEventKind.FINISH,
            ],
        )
        self.assertEqual([event.sequence_no for event in events], list(range(4)))
        self.assertEqual("".join(event.text or "" for event in events), "ab")
        self.assertEqual(
            dict(events[-2].payload_items),
            {"input_tokens": "3", "output_tokens": "2"},
        )
        self.assertEqual(dict(events[-1].payload_items), {"reason": "length"})
        input_ids, maximum, metadata = executor.calls[0]
        self.assertEqual((input_ids, maximum), ((11, 12, 13), 2))
        self.assertEqual(metadata.request_id, "sync")
        self.assertEqual(metadata.sampling, SamplingMeta())
        self.assertEqual(metadata.structured_output, StructuredOutputMeta.identity())
        self.assertEqual(executor.cancelled, [])

    async def test_sampling_metadata_reaches_executor_unchanged(self) -> None:
        executor = _SyncExecutor((1,))
        sampling = SamplingMeta(temperature=0.7, top_p=0.85, top_k=24, seed=91)
        execution = await _backend(executor).start(
            _request("sampled", maximum=1, sampling=sampling)
        )
        tuple([event async for event in execution.events])

        self.assertEqual(executor.calls[0][2].sampling, sampling)

    async def test_long_decode_does_not_emit_per_request_progress(self) -> None:
        token_count = 512
        execution = await _backend(_SyncExecutor((1,) * token_count)).start(
            _request("long-log", maximum=token_count)
        )

        with self.assertNoLogs("cubic_serve.runtime.performance_chat", level="INFO"):
            tuple([event async for event in execution.events])

    async def test_public_sse_emits_first_content_before_generation_finishes(self) -> None:
        executor = _GatedSyncExecutor()
        app = CubicASGIApp(
            _backend(executor),
            model_id=MODEL_ID,
            request_id_factory=lambda: "incremental-sse",
            clock=lambda: 7,
        )
        observed_before_finish = False

        def observe(message: dict[str, object]) -> None:
            nonlocal observed_before_finish
            body = message.get("body", b"")
            if isinstance(body, bytes) and b'"content":"a"' in body:
                observed_before_finish = not executor.finished.is_set()
                executor.release_second.set()

        response = await _invoke(
            app,
            "/v1/chat/completions",
            {
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 2,
                "stream": True,
            },
            on_send=observe,
        )

        self.assertTrue(executor.second_started.is_set())
        self.assertTrue(observed_before_finish)
        self.assertTrue(executor.finished.is_set())
        self.assertIn(b'"content":"a"', _body(response))
        self.assertIn(b'"content":"b"', _body(response))
        self.assertTrue(_body(response).endswith(b"data: [DONE]\n\n"))

    async def test_async_generator_stops_at_eos_and_is_closed(self) -> None:
        executor = _AsyncExecutor()
        _, events = await _collect(_backend(executor), "async")

        self.assertEqual(
            [event.kind for event in events],
            [TokenEventKind.TEXT_DELTA, TokenEventKind.USAGE, TokenEventKind.FINISH],
        )
        self.assertEqual(events[0].text, "c")
        self.assertEqual(dict(events[-2].payload_items)["output_tokens"], "2")
        self.assertEqual(dict(events[-1].payload_items), {"reason": "stop"})
        self.assertTrue(executor.closed)
        self.assertFalse(executor.overrun)
        self.assertEqual(executor.cancelled, [])

    async def test_ignore_eos_runs_until_requested_output_length(self) -> None:
        executor = _AsyncExecutor()
        execution = await _backend(executor).start(
            _request("ignore-eos", maximum=3, ignore_eos=True)
        )
        events = tuple([event async for event in execution.events])

        self.assertEqual("".join(event.text or "" for event in events), "cd")
        self.assertEqual(dict(events[-2].payload_items)["output_tokens"], "3")
        self.assertEqual(dict(events[-1].payload_items), {"reason": "length"})
        self.assertTrue(executor.calls[0][2].ignore_eos)
        self.assertTrue(executor.overrun)
        self.assertTrue(executor.closed)

    async def test_token_iteration_never_enters_asyncio_worker_pool(self) -> None:
        execution = await _backend(_AsyncExecutor()).start(_request("no-worker-pool"))

        with patch(
            "asyncio.to_thread",
            side_effect=AssertionError("token output must not use asyncio.to_thread"),
        ):
            events = tuple([event async for event in execution.events])

        self.assertEqual(events[0].text, "c")
        self.assertEqual(events[-1].kind, TokenEventKind.FINISH)

    async def test_constraint_is_rejected_before_processing_or_generation(self) -> None:
        processor = _Processor()
        executor = _SyncExecutor((1,))
        backend = _backend(executor, processor)
        constrained = _request(
            "same-id",
            maximum=1,
            structured_output=StructuredOutputMeta.grammar('root ::= "x"'),
        )

        with self.assertRaises(ChatBackendError) as raised:
            await backend.start(constrained)
        self.assertEqual(raised.exception.kind, BackendErrorKind.INVALID_REQUEST)
        self.assertIn("unconstrained generation", str(raised.exception))
        self.assertEqual((processor.calls, executor.calls), ([], []))

        execution = await backend.start(_request("same-id", maximum=1))
        tuple([event async for event in execution.events])
        self.assertEqual(processor.calls, ["same-id"])
        self.assertEqual(len(executor.calls), 1)

    async def test_openai_non_stream_and_anthropic_stream_share_events(self) -> None:
        executor = _SyncExecutor()
        request_ids = iter(("openai", "anthropic"))
        app = CubicASGIApp(
            _backend(executor),
            model_id=MODEL_ID,
            request_id_factory=lambda: next(request_ids),
            clock=lambda: 7,
        )
        openai = await _invoke(
            app,
            "/v1/chat/completions",
            {
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "hello"}],
                "max_completion_tokens": 2,
                "stream": False,
            },
        )
        anthropic = await _invoke(
            app,
            "/v1/messages",
            {
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 2,
                "stream": True,
            },
        )

        self.assertEqual((openai[0]["status"], anthropic[0]["status"]), (200, 200))
        document = json.loads(_body(openai))
        self.assertEqual(document["choices"][0]["message"]["content"], "ab")
        self.assertEqual(document["choices"][0]["finish_reason"], "length")
        stream = _body(anthropic)
        self.assertIn(b"event: message_start", stream)
        self.assertIn(b'"text":"a"', stream)
        self.assertIn(b'"text":"b"', stream)
        self.assertIn(b"event: message_stop", stream)
        self.assertEqual(len(executor.calls), 2)


if __name__ == "__main__":
    unittest.main()
