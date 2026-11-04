from __future__ import annotations

import asyncio
import importlib.util
import json
import threading
import unittest
from pathlib import Path

from cubic_serve.api.server import BackendErrorKind, ChatBackendError, CubicASGIApp
from cubic_serve.contracts.request import (
    CanonicalChatRequest,
    ChatContentPart,
    ChatMessage,
    ChatRole,
    ChatTemplateKwargs,
    ContentPart,
    DecodePositionPlan,
    InputPlan,
    StructuredOutputMeta,
    TokenEventKind,
    TokenPosition,
)
from cubic_serve.generation import ParsedOutputDelta
from cubic_serve.generation.constraints import (
    BackendTokenMask,
    ConstraintMatcher,
    ConstraintSpec,
)
from cubic_serve.models import FullBF16WeightPolicy, load_model_adapter
from cubic_serve.runtime.cache import KVLayout
from cubic_serve.runtime.chat import ReferenceChatBackend
from cubic_serve.runtime.engine import EngineStep, ReferenceEngine

REPOSITORY = Path(__file__).resolve().parents[3]
TOY_WEIGHT_POLICY = FullBF16WeightPolicy(
    expected_names=(
        "embedding",
        "k_weight",
        "lm_weight",
        "out_weight",
        "q_weight",
        "v_weight",
    )
)


class _FixtureProcessor:
    def __init__(self, barrier: threading.Barrier | None = None) -> None:
        self._barrier = barrier
        self.thinking_modes: list[bool] = []

    def build_chat_input_plan(
        self,
        request: CanonicalChatRequest,
        *,
        weight_variant: object,
        enable_thinking: bool,
    ) -> InputPlan:
        self.thinking_modes.append(enable_thinking)
        if self._barrier is not None:
            self._barrier.wait(timeout=5)
        text = "".join(
            part.text or ""
            for message in request.messages
            for part in message.content_parts
        )
        token_ids = (1, 2, 3)
        return InputPlan(
            request_id=request.request_id,
            model_id=request.model_id,
            processor_recipe_digest=f"b2s256:{'a' * 64}",
            weight_variant=weight_variant,  # type: ignore[arg-type]
            content_parts=(ContentPart(0, "text", text=text),),
            token_ids=token_ids,
            positions=tuple(TokenPosition((index,)) for index in range(len(token_ids))),
            decode_positions=DecodePositionPlan(TokenPosition((len(token_ids),)), (1,)),
        )


class _FixtureDecoder:
    def decode(self, token_id: int) -> str:
        return chr(ord("a") + token_id % 26)

    def finish(self) -> str:
        return ""


class _FixtureParser:
    def push(self, token_id: int, text: str) -> tuple[ParsedOutputDelta, ...]:
        del token_id
        return (ParsedOutputDelta(TokenEventKind.TEXT_DELTA, text),)

    def finish(self, text: str) -> tuple[ParsedOutputDelta, ...]:
        return (ParsedOutputDelta(TokenEventKind.TEXT_DELTA, text),) if text else ()


class _FixtureCodec:
    eos_token_id = 100

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

    def new_decoder(self) -> _FixtureDecoder:
        return _FixtureDecoder()

    def new_parser(
        self,
        *,
        enable_thinking: bool,
        reasoning_parser: bool,
        tool_call_parser: bool,
    ) -> _FixtureParser:
        self.parser_modes.append(
            (enable_thinking, reasoning_parser, tool_call_parser)
        )
        return _FixtureParser()


class _ConstraintMatcher:
    def __init__(
        self,
        digest: str,
        accepted: list[int],
        *,
        terminate_after: int,
    ) -> None:
        self.constraint_digest = digest
        self.accepted = accepted
        self.terminate_after = terminate_after

    def fill_token_mask(self) -> BackendTokenMask:
        return BackendTokenMask(None, False)

    def apply_token_mask(self, logits: object, payload: object) -> None:
        raise AssertionError("no-op matcher masks are never applied")

    def accept_token(self, token_id: int) -> bool:
        self.accepted.append(token_id)
        return True

    def is_terminated(self) -> bool:
        return len(self.accepted) >= self.terminate_after

    def fork(self) -> ConstraintMatcher:
        return _ConstraintMatcher(
            self.constraint_digest,
            list(self.accepted),
            terminate_after=self.terminate_after,
        )


class _ConstraintFactory:
    backend_id = "fixture"

    def __init__(self, *, terminate_after: int) -> None:
        self.terminate_after = terminate_after
        self.sessions: list[list[int]] = []

    def require_ready(self, spec: ConstraintSpec) -> None:
        return None

    def create_matcher(self, spec: ConstraintSpec) -> ConstraintMatcher:
        accepted: list[int] = []
        self.sessions.append(accepted)
        return _ConstraintMatcher(
            spec.digest,
            accepted,
            terminate_after=self.terminate_after,
        )


class _RecordingEngine(ReferenceEngine):
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.steps: list[EngineStep] = []
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    def step(self) -> EngineStep | None:
        step = super().step()
        if step is not None:
            self.steps.append(step)
        return step


async def _invoke_raw(
    app: CubicASGIApp,
    path: str,
    payload: dict[str, object],
) -> tuple[int, bytes]:
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
    body = b"".join(
        message.get("body", b"")
        for message in sent
        if message.get("type") == "http.response.body"
        and isinstance(message.get("body", b""), bytes)
    )
    status = sent[0].get("status")
    assert isinstance(status, int)
    return status, body


async def _invoke_json(
    app: CubicASGIApp, path: str, payload: dict[str, object]
) -> tuple[int, dict[str, object]]:
    status, body = await _invoke_raw(app, path, payload)
    document = json.loads(body)
    assert isinstance(document, dict)
    return status, document


class ReferenceChatBackendTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.loaded = load_model_adapter(REPOSITORY / "examples" / "toy_adapter.py")

    def engine(self, *, eos_token_ids: tuple[int, ...] = (100,)) -> _RecordingEngine:
        return _RecordingEngine(
            self.loaded,
            kv_layout=KVLayout((0,), 4, 1, 8),
            weight_policy=TOY_WEIGHT_POLICY,
            token_budget=8,
            prefill_chunk_tokens=4,
            eos_token_ids=eos_token_ids,
        )

    def request(
        self,
        request_id: str,
        *,
        maximum: int | None = 2,
        structured_output: StructuredOutputMeta | None = None,
        chat_template_kwargs: dict[str, object] | None = None,
    ) -> CanonicalChatRequest:
        return CanonicalChatRequest(
            request_id=request_id,
            model_id=self.loaded.descriptor.model_spec.model_id,
            messages=(
                ChatMessage(
                    ChatRole.USER,
                    (ChatContentPart(0, "text", text=f"hello {request_id}"),),
                ),
            ),
            max_new_tokens=maximum,
            structured_output=structured_output or StructuredOutputMeta.identity(),
            chat_template_kwargs=ChatTemplateKwargs.from_value(
                chat_template_kwargs or {}
            ),
        )

    async def test_omitted_limit_uses_remaining_reference_context(self) -> None:
        backend = ReferenceChatBackend(
            self.engine(), processor=_FixtureProcessor(), output_codec=_FixtureCodec(),
            max_sequence_tokens=6,
        )
        execution = await backend.start(self.request("default-limit", maximum=None))
        events = [event async for event in execution.events]
        self.assertEqual(dict(events[-2].payload_items)["output_tokens"], "3")
        self.assertEqual(dict(events[-1].payload_items)["reason"], "length")

    async def test_omitted_limit_rejects_exhausted_reference_context(self) -> None:
        backend = ReferenceChatBackend(
            self.engine(), processor=_FixtureProcessor(), output_codec=_FixtureCodec(),
            max_sequence_tokens=3,
        )
        with self.assertRaises(ChatBackendError) as caught:
            await backend.start(self.request("full-context", maximum=None))
        self.assertEqual(caught.exception.kind, BackendErrorKind.INVALID_REQUEST)
        self.assertNotIn("full-context", backend._known_request_ids)

    async def test_request_thinking_override_drives_explicit_reasoning_parser(self) -> None:
        for ordinal, (service_default, request_value) in enumerate(
            ((False, True), (True, False))
        ):
            with self.subTest(
                service_default=service_default,
                request_value=request_value,
            ):
                processor = _FixtureProcessor()
                codec = _FixtureCodec()
                backend = ReferenceChatBackend(
                    self.engine(),
                    processor=processor,  # type: ignore[arg-type]
                    output_codec=codec,  # type: ignore[arg-type]
                    enable_thinking=service_default,
                    reasoning_parser=True,
                )
                execution = await backend.start(
                    self.request(
                        f"thinking-{ordinal}",
                        chat_template_kwargs={"enable_thinking": request_value},
                    )
                )
                tuple([event async for event in execution.events])

                self.assertEqual(processor.thinking_modes, [request_value])
                self.assertEqual(codec.admissions, [(True, False)])
                self.assertEqual(codec.parser_modes, [(True, True, False)] if request_value else [])

    async def test_default_output_path_does_not_construct_any_parser(self) -> None:
        codec = _FixtureCodec()
        backend = ReferenceChatBackend(
            self.engine(),
            processor=_FixtureProcessor(),  # type: ignore[arg-type]
            output_codec=codec,  # type: ignore[arg-type]
            enable_thinking=True,
        )

        execution = await backend.start(self.request("no-parser"))
        tuple([event async for event in execution.events])

        self.assertEqual(codec.admissions, [])
        self.assertEqual(codec.parser_modes, [])

    async def test_tool_request_without_parser_is_rejected_before_processing(self) -> None:
        processor = _FixtureProcessor()
        backend = ReferenceChatBackend(
            self.engine(),
            processor=processor,  # type: ignore[arg-type]
            output_codec=_FixtureCodec(),  # type: ignore[arg-type]
        )
        request = self.request(
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
            await backend.start(request)

        self.assertEqual(raised.exception.kind, BackendErrorKind.INVALID_REQUEST)
        self.assertIn("explicitly configured tool-call parser", str(raised.exception))
        self.assertEqual(processor.thinking_modes, [])

    async def test_one_driver_batches_and_streams_two_requests_online(self) -> None:
        engine = self.engine()
        backend = ReferenceChatBackend(
            engine,
            processor=_FixtureProcessor(threading.Barrier(2)),  # type: ignore[arg-type]
            output_codec=_FixtureCodec(),  # type: ignore[arg-type]
            max_buffered_events=2,
        )
        first, second = await asyncio.gather(
            backend.start(self.request("first")),
            backend.start(self.request("second")),
        )

        async def collect(execution: object) -> tuple[object, ...]:
            events = execution.events  # type: ignore[attr-defined]
            return tuple([event async for event in events])

        first_events, second_events = await asyncio.gather(collect(first), collect(second))

        self.assertEqual(
            {segment.request_id for segment in engine.steps[0].batch_plan.segments},
            {"first", "second"},
        )
        for request_id, events in (("first", first_events), ("second", second_events)):
            self.assertEqual([event.request_id for event in events], [request_id] * len(events))
            self.assertEqual(
                [event.sequence_no for event in events],
                list(range(len(events))),
            )
            self.assertEqual(events[-2].kind, TokenEventKind.USAGE)
            self.assertEqual(events[-1].kind, TokenEventKind.FINISH)
            self.assertEqual(dict(events[-2].payload_items)["input_tokens"], "3")

    async def test_cancel_closes_publication_but_keeps_committed_owner_state(self) -> None:
        engine = self.engine()
        backend = ReferenceChatBackend(
            engine,
            processor=_FixtureProcessor(),  # type: ignore[arg-type]
            output_codec=_FixtureCodec(),  # type: ignore[arg-type]
            max_buffered_events=1,
        )
        execution = await backend.start(self.request("cancelled", maximum=4))
        first = await execution.events.__anext__()
        self.assertEqual(first.kind, TokenEventKind.TEXT_DELTA)

        await backend.cancel("cancelled")
        with self.assertRaises(StopAsyncIteration):
            await execution.events.__anext__()

        state = engine.scheduler.request_state("cancelled")
        self.assertTrue(state.cancelled)
        self.assertFalse(state.finished)
        self.assertGreater(engine.store.current_revision("cancelled"), 0)
        self.assertFalse(engine.allocator.page_table("cancelled").released)

    async def test_unknown_model_fails_before_engine_admission(self) -> None:
        engine = self.engine()
        backend = ReferenceChatBackend(
            engine,
            processor=_FixtureProcessor(),  # type: ignore[arg-type]
            output_codec=_FixtureCodec(),  # type: ignore[arg-type]
        )
        request = self.request("wrong")
        request = CanonicalChatRequest(
            request.request_id,
            "missing/model",
            request.messages,
            request.max_new_tokens,
        )

        with self.assertRaises(ChatBackendError) as raised:
            await backend.start(request)
        self.assertEqual(raised.exception.kind, BackendErrorKind.MODEL_NOT_FOUND)
        with self.assertRaises(KeyError):
            engine.scheduler.request_state("wrong")

    async def test_asgi_both_protocols_close_the_public_request_chain(self) -> None:
        engine = self.engine()
        backend = ReferenceChatBackend(
            engine,
            processor=_FixtureProcessor(),  # type: ignore[arg-type]
            output_codec=_FixtureCodec(),  # type: ignore[arg-type]
        )
        request_ids = iter(("wire-openai", "wire-anthropic"))
        model = self.loaded.descriptor.model_spec.model_id
        app = CubicASGIApp(
            backend,
            model_id=model,
            request_id_factory=lambda: next(request_ids),
            clock=lambda: 7,
        )

        openai_status, openai = await _invoke_json(
            app,
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "hello"}],
                "max_completion_tokens": 2,
            },
        )
        anthropic_status, anthropic = await _invoke_json(
            app,
            "/v1/messages",
            {
                "model": model,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 2,
            },
        )

        self.assertEqual((openai_status, anthropic_status), (200, 200))
        self.assertEqual(openai["choices"][0]["finish_reason"], "length")  # type: ignore[index]
        self.assertEqual(openai["usage"]["prompt_tokens"], 3)  # type: ignore[index]
        self.assertEqual(anthropic["stop_reason"], "max_tokens")
        self.assertEqual(anthropic["usage"]["input_tokens"], 3)  # type: ignore[index]
        self.assertEqual(
            openai["choices"][0]["message"]["content"],  # type: ignore[index]
            anthropic["content"][0]["text"],  # type: ignore[index]
        )

    async def test_public_json_schema_owns_one_matcher_through_engine_sampling(self) -> None:
        engine = self.engine()
        factory = _ConstraintFactory(terminate_after=2)
        backend = ReferenceChatBackend(
            engine,
            processor=_FixtureProcessor(),  # type: ignore[arg-type]
            output_codec=_FixtureCodec(),  # type: ignore[arg-type]
            constraint_factory=factory,
        )
        app = CubicASGIApp(
            backend,
            model_id=self.loaded.descriptor.model_spec.model_id,
            request_id_factory=lambda: "structured",
            clock=lambda: 7,
        )
        status, response = await _invoke_json(
            app,
            "/v1/chat/completions",
            {
                "model": self.loaded.descriptor.model_spec.model_id,
                "messages": [{"role": "user", "content": "hello"}],
                "max_completion_tokens": 2,
                "reasoning_effort": "none",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "schema": {"type": "string"},
                        "strict": True,
                    },
                },
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(response["choices"][0]["finish_reason"], "length")  # type: ignore[index]
        self.assertEqual(len(factory.sessions), 1)
        generated = engine.scheduler.request_state("structured").generated_token_ids
        self.assertEqual(factory.sessions[0], list(generated))

    @unittest.skipUnless(importlib.util.find_spec("xgrammar"), "requires xgrammar extra")
    async def test_native_tools_stream_through_reference_sampling_with_usage_heartbeats(
        self,
    ) -> None:
        from cubic_serve.generation.xgrammar import XGrammarConstraintFactory
        from cubic_serve.models.qwen3_5.output import Qwen3_5OutputParser

        vocab = (
            "<tool_call>\n<function=weather>\n",
            "<parameter=city>",
            "Tokyo",
            "</parameter>",
            "\n<parameter=count>",
            "42",
            "\n</function>\n</tool_call>",
        )
        vocab += tuple(f"<unused-{i}>" for i in range(24)) + ("<eos>",)

        class Decoder:
            def decode(self, token_id):
                return vocab[token_id]

            def finish(self):
                return ""

        class Codec(_FixtureCodec):
            eos_token_id = 31

            def new_decoder(self):
                return Decoder()

            def new_parser(self, **kwargs):
                return Qwen3_5OutputParser(
                    enable_thinking=False, reasoning_parser=False, tool_call_parser=True
                )

        backend = ReferenceChatBackend(
            self.engine(eos_token_ids=(31,)),
            processor=_FixtureProcessor(),
            output_codec=Codec(),
            tool_call_parser="qwen3_coder",
            constraint_factory=XGrammarConstraintFactory(
                vocab,
                vocab_type="raw",
                vocab_size=32,
                stop_token_ids=(31,),
                max_threads=2,
                tool_model_format="qwen_3_5",
                tool_reasoning=False,
            ),
        )
        app = CubicASGIApp(
            backend,
            model_id=self.loaded.descriptor.model_spec.model_id,
            request_id_factory=lambda: "native-usage",
            clock=lambda: 7,
        )
        status, wire = await _invoke_raw(
            app,
            "/v1/chat/completions",
            {
                "model": self.loaded.descriptor.model_spec.model_id,
                "messages": [{"role": "user", "content": "call weather"}],
                "max_tokens": 32,
                "reasoning_effort": "none",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "strict": True,
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {"type": "string", "enum": ["Tokyo"]},
                                    "count": {"type": "integer", "enum": [42]},
                                },
                                "required": ["city", "count"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
                "tool_choice": "required",
                "stream": True,
                "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            },
        )
        self.assertEqual(status, 200)
        data = [line[6:] for line in wire.splitlines() if line.startswith(b"data: ")]
        self.assertEqual(data[-1], b"[DONE]")
        events = [json.loads(item) for item in data[:-1]]
        self.assertFalse(any("error" in item for item in events), events)
        text = "".join(
            tool.get("function", {}).get("arguments", "")
            for item in events
            for choice in item.get("choices", [])
            for tool in choice.get("delta", {}).get("tool_calls", [])
        )
        self.assertEqual(json.loads(text), {"city": "Tokyo", "count": 42})

    @unittest.skipUnless(importlib.util.find_spec("xgrammar"), "requires xgrammar extra")
    async def test_auto_tool_and_answer_union_reaches_reference_public_path(self) -> None:
        from cubic_serve.generation.xgrammar import XGrammarConstraintFactory
        vocab = tuple(chr(ord("a") + index) for index in range(32))
        codec = _FixtureCodec()
        codec.eos_token_id = 31
        backend = ReferenceChatBackend(
            self.engine(eos_token_ids=(31,)), processor=_FixtureProcessor(), output_codec=codec,
            tool_call_parser=True, constraint_factory=XGrammarConstraintFactory(
                vocab, vocab_type="raw", vocab_size=32, stop_token_ids=(31,), max_threads=2,
                tool_model_format="qwen_3_5", tool_reasoning=False),
        )
        app = CubicASGIApp(backend, model_id=self.loaded.descriptor.model_spec.model_id,
                           request_id_factory=lambda: "combined", clock=lambda: 7)
        status, response = await _invoke_json(app, "/v1/chat/completions", {
            "model": self.loaded.descriptor.model_spec.model_id,
            "messages": [{"role": "user", "content": "answer"}], "max_tokens": 3,
            "reasoning_effort": "none", "tools": [{"type": "function", "function": {
                "name": "weather", "parameters": {"type": "object"}}}],
            "tool_choice": "auto", "structured_outputs": {"choice": ["de"]},
        })
        self.assertEqual(status, 200, response)
        self.assertEqual(response["choices"][0]["message"]["content"], "de")

    @unittest.skipUnless(importlib.util.find_spec("xgrammar"), "requires xgrammar extra")
    async def test_forced_tools_use_reference_model_and_json_without_native_parser(self) -> None:
        from cubic_serve.generation.xgrammar import XGrammarConstraintFactory

        vocab = tuple('{}":,[]fxnameprts') + tuple(f"<unused-{i}>" for i in range(14))
        self.assertEqual(len(vocab), 31)
        vocab += ("<eos>",)

        class Decoder:
            def decode(self, token_id):
                return vocab[token_id]

            def finish(self):
                return ""

        class Codec(_FixtureCodec):
            eos_token_id = 31

            def new_decoder(self):
                return Decoder()

        tool = {
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string", "enum": ["a"]}},
                    "required": ["x"],
                    "additionalProperties": False,
                },
            },
        }
        choices = ("required", {"type": "function", "function": {"name": "f"}})
        for index, (choice, stream) in enumerate(
            (choice, stream) for choice in choices for stream in (False, True)
        ):
            backend = ReferenceChatBackend(
                self.engine(eos_token_ids=(31,)),
                processor=_FixtureProcessor(),
                output_codec=Codec(),
                constraint_factory=XGrammarConstraintFactory(
                    vocab, vocab_type="raw", vocab_size=32, stop_token_ids=(31,), max_threads=2
                ),
            )
            app = CubicASGIApp(
                backend,
                model_id=self.loaded.descriptor.model_spec.model_id,
                request_id_factory=lambda: f"forced-{index}",
                clock=lambda: 7,
            )
            status, wire = await _invoke_raw(
                app,
                "/v1/chat/completions",
                {
                    "model": self.loaded.descriptor.model_spec.model_id,
                    "messages": [{"role": "user", "content": "call f"}],
                    "max_tokens": 64,
                    "reasoning_effort": "none",
                    "tools": [tool],
                    "tool_choice": choice,
                    "parallel_tool_calls": False,
                    "stream": stream,
                },
            )
            self.assertEqual(status, 200, wire)
            if stream:
                data = [line[6:] for line in wire.splitlines() if line.startswith(b"data: ")]
                self.assertEqual(data[-1], b"[DONE]")
                chunks = [json.loads(item) for item in data[:-1]]
                self.assertFalse(any("error" in item for item in chunks), chunks)
                results = [result for item in chunks for result in item["choices"]]
                self.assertEqual(results[-1]["finish_reason"], "tool_calls")
                calls = [call for result in results for call in result["delta"].get("tool_calls", [])]
                self.assertEqual({call["index"] for call in calls}, {0})
                self.assertEqual([call["function"]["name"] for call in calls
                                  if "name" in call["function"]], ["f"])
                arguments = "".join(call["function"].get("arguments", "") for call in calls)
            else:
                result = json.loads(wire)["choices"][0]
                self.assertEqual(result["finish_reason"], "tool_calls")
                calls = result["message"]["tool_calls"]
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["function"]["name"], "f")
                arguments = calls[0]["function"]["arguments"]
            self.assertEqual(json.loads(arguments), {"x": "a"})

    @unittest.skipUnless(importlib.util.find_spec("xgrammar"), "requires xgrammar extra")
    async def test_public_selectors_drive_real_xgrammar_and_reference_model_sampling(self) -> None:
        from cubic_serve.generation.xgrammar import XGrammarConstraintFactory

        # This is the actual independent reference model, not forced token output.
        # Its unconstrained scores are masked by the real library before sampling.
        vocab = tuple(chr(ord("a") + index) for index in range(32))
        factory = XGrammarConstraintFactory(
            vocab, vocab_type="raw", vocab_size=32, stop_token_ids=(31,), max_threads=2
        )
        cases = (
            {"regex": "de"},
            {"choice": ["de"]},
            {"grammar": 'start: "d" "e"'},
            {
                "structural_tag": {
                    "type": "structural_tag",
                    "format": {"type": "const_string", "value": "de"},
                }
            },
        )
        for index, constraint in enumerate(cases):
            with self.subTest(constraint=constraint):
                engine = self.engine(eos_token_ids=(31,))
                codec = _FixtureCodec()
                codec.eos_token_id = 31
                backend = ReferenceChatBackend(
                    engine,
                    processor=_FixtureProcessor(),
                    output_codec=codec,
                    constraint_factory=factory,
                )
                request_id = f"real-grammar-{index}"
                identities = iter((request_id, request_id + "-stream"))
                app = CubicASGIApp(
                    backend,
                    model_id=self.loaded.descriptor.model_spec.model_id,
                    request_id_factory=lambda values=identities: next(values),
                    clock=lambda: 7,
                )
                status, response = await _invoke_json(
                    app,
                    "/v1/chat/completions",
                    {
                        "model": self.loaded.descriptor.model_spec.model_id,
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 3,
                        "reasoning_effort": "none",
                        "structured_outputs": constraint,
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(response["choices"][0]["message"]["content"], "de")
                self.assertEqual(
                    engine.scheduler.request_state(request_id).generated_token_ids, (3, 4, 31)
                )
                status, wire = await _invoke_raw(
                    app,
                    "/v1/chat/completions",
                    {
                        "model": self.loaded.descriptor.model_spec.model_id,
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 3,
                        "reasoning_effort": "none",
                        "stream": True,
                        "structured_outputs": constraint,
                    },
                )
                self.assertEqual(status, 200)
                data = [line[6:] for line in wire.splitlines() if line.startswith(b"data: ")]
                self.assertEqual(data[-1], b"[DONE]")
                chunks = [json.loads(line) for line in data[:-1]]
                self.assertEqual(
                    "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks),
                    "de",
                )
                self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    @unittest.skipUnless(importlib.util.find_spec("xgrammar"), "requires xgrammar extra")
    async def test_invalid_grammar_returns_private_400_and_does_not_poison_reference(self) -> None:
        from cubic_serve.generation.xgrammar import XGrammarConstraintFactory

        engine = self.engine(eos_token_ids=(31,))
        codec = _FixtureCodec()
        codec.eos_token_id = 31
        backend = ReferenceChatBackend(
            engine,
            processor=_FixtureProcessor(),
            output_codec=codec,
            constraint_factory=XGrammarConstraintFactory(
                tuple(chr(ord("a") + index) for index in range(32)),
                vocab_type="raw",
                vocab_size=32,
                stop_token_ids=(31,),
                max_threads=2,
            ),
        )
        identities = iter(("bad-grammar", "good-grammar"))
        app = CubicASGIApp(
            backend,
            model_id=self.loaded.descriptor.model_spec.model_id,
            request_id_factory=lambda: next(identities),
            clock=lambda: 7,
        )
        payload = {
            "model": self.loaded.descriptor.model_spec.model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
            "reasoning_effort": "none",
            "structured_outputs": {"regex": "["},
        }
        status, response = await _invoke_json(app, "/v1/chat/completions", payload)
        self.assertEqual(status, 400)
        self.assertEqual(response["error"]["message"], "invalid request")
        payload["structured_outputs"] = {"choice": ["d"]}
        status, response = await _invoke_json(app, "/v1/chat/completions", payload)
        self.assertEqual(status, 200)
        self.assertEqual(response["choices"][0]["message"]["content"], "d")

    async def test_missing_constraint_rejects_and_length_stop_keeps_its_matcher(self) -> None:
        payload = {
            "model": self.loaded.descriptor.model_spec.model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_completion_tokens": 1,
            "reasoning_effort": "none",
            "grammar": 'root ::= "x"',
        }
        missing = ReferenceChatBackend(
            self.engine(),
            processor=_FixtureProcessor(),  # type: ignore[arg-type]
            output_codec=_FixtureCodec(),  # type: ignore[arg-type]
        )
        status, _ = await _invoke_json(
            CubicASGIApp(
                missing,
                model_id=self.loaded.descriptor.model_spec.model_id,
                request_id_factory=lambda: "missing",
            ),
            "/v1/chat/completions",
            payload,
        )
        self.assertEqual(status, 503)
        with self.assertRaises(KeyError):
            missing._engine.scheduler.request_state("missing")

        incomplete = ReferenceChatBackend(
            self.engine(),
            processor=_FixtureProcessor(),  # type: ignore[arg-type]
            output_codec=_FixtureCodec(),  # type: ignore[arg-type]
            constraint_factory=_ConstraintFactory(terminate_after=2),
        )
        status, partial = await _invoke_json(
            CubicASGIApp(
                incomplete,
                model_id=self.loaded.descriptor.model_spec.model_id,
                request_id_factory=lambda: "incomplete",
            ),
            "/v1/chat/completions",
            payload,
        )
        self.assertEqual(status, 200)
        self.assertEqual(partial["choices"][0]["finish_reason"], "length")


if __name__ == "__main__":
    unittest.main()
