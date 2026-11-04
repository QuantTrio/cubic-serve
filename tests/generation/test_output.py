from __future__ import annotations

import codecs
import unittest
from typing import Any

from cubic_serve.api import (
    project_anthropic_message,
    project_openai_chat_completion,
)
from cubic_serve.contracts.request import TokenEvent, TokenEventKind
from cubic_serve.generation import (
    GenerationOutputError,
    GenerationOutputPipeline,
    ParsedOutputDelta,
)
from cubic_serve.generation.output import TokenPhaseOutputParser


def _raw_events(
    token_ids: tuple[int, ...], *, request_id: str = "request-1"
) -> tuple[TokenEvent, ...]:
    events = tuple(
        TokenEvent(
            request_id=request_id,
            sequence_no=index,
            kind=TokenEventKind.TOKEN,
            token_id=token_id,
        )
        for index, token_id in enumerate(token_ids)
    )
    return (
        *events,
        TokenEvent(
            request_id=request_id,
            sequence_no=len(token_ids),
            kind=TokenEventKind.FINISH,
        ),
    )


class ToyEngine:
    def __init__(self, token_ids: tuple[int, ...]) -> None:
        self._token_ids = token_ids

    def run(self) -> tuple[TokenEvent, ...]:
        return _raw_events(self._token_ids)


class Utf8ByteDecoder:
    """Toy decoder that deliberately receives one UTF-8 byte per token."""

    def __init__(self, *, eos_token_id: int) -> None:
        self._eos = eos_token_id
        self._decoder = codecs.getincrementaldecoder("utf-8")("strict")

    def decode(self, token_id: int) -> str:
        if token_id == self._eos:
            return ""
        return self._decoder.decode(bytes((token_id,)), final=False)

    def finish(self) -> str:
        return self._decoder.decode(b"", final=True)


class MappingDecoder:
    def __init__(self, values: dict[int, str], *, tail: str = "") -> None:
        self._values = values
        self._tail = tail

    def decode(self, token_id: int) -> str:
        return self._values.get(token_id, "")

    def finish(self) -> str:
        return self._tail


class ScriptedParser:
    def push(self, token_id: int, text: str) -> tuple[ParsedOutputDelta, ...]:
        if token_id == 10:
            return (ParsedOutputDelta(TokenEventKind.REASONING_DELTA, text),)
        if token_id == 11:
            return (ParsedOutputDelta(TokenEventKind.TEXT_DELTA, text),)
        if token_id == 12:
            return (
                ParsedOutputDelta(
                    TokenEventKind.TOOL_CALL_DELTA,
                    text,
                    (("id", "call-0"), ("index", "0"), ("name", "lookup")),
                ),
            )
        if token_id == 13:
            return (
                ParsedOutputDelta(
                    TokenEventKind.TOOL_CALL_DELTA,
                    text,
                    (("index", "0"),),
                ),
            )
        return ()

    def finish(self, text: str) -> tuple[ParsedOutputDelta, ...]:
        return (ParsedOutputDelta(TokenEventKind.TEXT_DELTA, text),) if text else ()


class GenerationOutputPipelineTests(unittest.TestCase):
    def test_prompt_boundary_prefix_never_counts_as_generated_reasoning(self) -> None:
        parser = TokenPhaseOutputParser(None, reasoning_end=(7, 8), reasoning_prefix=(7,))
        self.assertEqual(parser.push(8, "</end>"), ())
        self.assertEqual(parser.reasoning_token_count, 0)
        self.assertEqual(parser.push(2, "answer")[0].kind, TokenEventKind.TEXT_DELTA)
        parser = TokenPhaseOutputParser(None, reasoning_end=(7, 8), reasoning_prefix=(7,))
        events = parser.push(2, "thought")
        self.assertEqual(parser.reasoning_token_count, 1)
        self.assertEqual(events[0].text, "thought")

    def test_reasoning_usage_counts_tokens_not_decoded_fragments(self) -> None:
        # Three UTF-8 byte tokens emit one character; control markers emit none.
        token_ids = (7, 8, 0xE4, 0xBD, 0xA0, 3, 5, ord("!"))
        for visible in (True, False):
            for chunk in (1, 3, len(token_ids) + 1):
                pipeline = GenerationOutputPipeline(
                    request_id="request-1", input_token_count=4,
                    max_new_tokens=len(token_ids), eos_token_ids=(),
                    decoder=Utf8ByteDecoder(eos_token_id=0),
                    parser=TokenPhaseOutputParser(
                        None, reasoning_start=(7, 8), reasoning_end=(3, 5)
                    ),
                    continuous_usage=True, report_reasoning_tokens=True,
                    include_reasoning=visible,
                )
                raw = _raw_events(token_ids)
                events = tuple(
                    event
                    for start in range(0, len(raw), chunk)
                    for event in pipeline.consume(raw[start:start + chunk])
                )
                response = project_openai_chat_completion(events, model="fixture", created=0)
                message = response["choices"][0]["message"]
                self.assertEqual(message["content"], "!")
                self.assertEqual(message.get("reasoning_content"), "你" if visible else None)
                self.assertEqual(response["usage"]["completion_tokens"], len(token_ids))
                self.assertEqual(
                    response["usage"]["completion_tokens_details"], {"reasoning_tokens": 3}
                )
                counts = [event.reasoning_token_count for event in events]
                self.assertEqual(counts, sorted(counts))
                self.assertEqual(counts[-1], 3)

    def test_unclosed_reasoning_counts_buffered_marker_prefix_at_length(self) -> None:
        pipeline = GenerationOutputPipeline(
            request_id="request-1", input_token_count=1, max_new_tokens=3,
            eos_token_ids=(), decoder=MappingDecoder({1: "why", 3: "<", 4: "end"}),
            parser=TokenPhaseOutputParser(None, reasoning_end=(3, 4, 5)),
            report_reasoning_tokens=True,
        )
        response = project_openai_chat_completion(
            pipeline.consume(_raw_events((1, 3, 4))), model="fixture", created=0
        )
        self.assertEqual(response["choices"][0]["message"]["reasoning_content"], "why<end")
        self.assertEqual(
            response["usage"]["completion_tokens_details"], {"reasoning_tokens": 3}
        )

    def test_implicit_tool_boundary_is_excluded_from_reasoning_usage(self) -> None:
        parser = TokenPhaseOutputParser(
            None, reasoning_end=(9,), reasoning_implicit_end=((4, 5),)
        )
        deltas = tuple(
            delta for token, text in ((1, "why"), (2, ""), (4, "<tool"), (5, ">"))
            for delta in parser.push(token, text)
        )
        self.assertEqual(parser.reasoning_token_count, 2)
        self.assertEqual(
            "".join(
                delta.text or "" for delta in deltas if delta.kind is TokenEventKind.TEXT_DELTA
            ),
            "<tool>",
        )

    def test_configured_reasoning_parser_reports_zero_when_thinking_disabled(self) -> None:
        for report in (True, False):
            pipeline = GenerationOutputPipeline(
                request_id="request-1", input_token_count=1, max_new_tokens=1,
                eos_token_ids=(), decoder=MappingDecoder({1: "answer"}),
                report_reasoning_tokens=report,
            )
            response = project_openai_chat_completion(
                pipeline.consume(_raw_events((1,))), model="fixture", created=0
            )
            if report:
                self.assertEqual(
                    response["usage"]["completion_tokens_details"], {"reasoning_tokens": 0}
                )
            else:
                self.assertNotIn("completion_tokens_details", response["usage"])

    def test_secondary_eos_is_consumed_without_decoding_or_leaking_its_marker(self):
        for eos in (8, 9):
            with self.subTest(eos=eos):
                pipeline = GenerationOutputPipeline(
                    request_id="request-1", input_token_count=3, max_new_tokens=4,
                    eos_token_ids=(8, 9),
                    decoder=MappingDecoder({1: "answer", 8: "<|endoftext|>", 9: "<|im_end|>"}),
                )
                events = pipeline.consume(_raw_events((1, eos)))
                document = project_openai_chat_completion(events, model="fixture", created=0)
                self.assertEqual(document["choices"][0]["message"]["content"], "answer")
                self.assertEqual(document["choices"][0]["finish_reason"], "stop")
                self.assertEqual(document["usage"]["completion_tokens"], 2)

    def test_toy_engine_utf8_tokens_reach_both_public_projections(self) -> None:
        eos = 0
        engine = ToyEngine((0xE4, 0xBD, 0xA0, ord("!"), eos))
        pipeline = GenerationOutputPipeline(
            request_id="request-1",
            input_token_count=9,
            max_new_tokens=8,
            eos_token_ids=(eos,),
            decoder=Utf8ByteDecoder(eos_token_id=eos),
        )

        emitted_per_raw = tuple(pipeline.push(event) for event in engine.run())
        self.assertEqual(emitted_per_raw[0:2], ((), ()))
        self.assertEqual(emitted_per_raw[2][0].text, "你")
        canonical = tuple(event for group in emitted_per_raw for event in group)
        self.assertEqual(
            tuple(event.sequence_no for event in canonical), tuple(range(len(canonical)))
        )
        self.assertEqual(
            tuple(event.kind for event in canonical),
            (
                TokenEventKind.TEXT_DELTA,
                TokenEventKind.TEXT_DELTA,
                TokenEventKind.USAGE,
                TokenEventKind.FINISH,
            ),
        )

        openai: Any = project_openai_chat_completion(canonical, model="fixture", created=0)
        anthropic: Any = project_anthropic_message(canonical, model="fixture")
        self.assertEqual(openai["choices"][0]["message"]["content"], "你!")
        self.assertEqual(openai["choices"][0]["finish_reason"], "stop")
        self.assertEqual(openai["usage"]["prompt_tokens"], 9)
        self.assertEqual(openai["usage"]["completion_tokens"], 5)
        self.assertEqual(anthropic["content"], [{"type": "text", "text": "你!"}])
        self.assertEqual(anthropic["stop_reason"], "end_turn")

    def test_max_new_tokens_maps_to_length_and_flushes_decoder_tail(self) -> None:
        pipeline = GenerationOutputPipeline(
            request_id="request-1",
            input_token_count=4,
            max_new_tokens=1,
            eos_token_ids=(),
            decoder=MappingDecoder({1: ""}, tail="held"),
        )
        canonical = pipeline.consume(_raw_events((1,)))

        self.assertEqual([event.text for event in canonical[:-2]], ["held"])
        self.assertEqual(
            dict(canonical[-2].payload_items),
            {"input_tokens": "4", "output_tokens": "1"},
        )
        self.assertEqual(canonical[-1].payload_items, (("reason", "length"),))
        self.assertTrue(pipeline.finished)
        self.assertEqual(pipeline.output_token_count, 1)

    def test_optional_parser_emits_reasoning_and_complete_tool_call(self) -> None:
        eos = 99
        pipeline = GenerationOutputPipeline(
            request_id="request-1",
            input_token_count=2,
            max_new_tokens=8,
            eos_token_ids=(eos,),
            decoder=MappingDecoder(
                {10: "why", 11: "answer", 12: '{"key":', 13: '"value"}', eos: ""}
            ),
            parser=ScriptedParser(),
        )
        canonical = pipeline.consume(_raw_events((10, 11, 12, 13, eos)))

        self.assertEqual(canonical[-1].payload_items, (("reason", "tool_calls"),))
        openai: Any = project_openai_chat_completion(canonical, model="fixture", created=0)
        anthropic: Any = project_anthropic_message(canonical, model="fixture")
        message = openai["choices"][0]["message"]
        self.assertEqual(message["reasoning_content"], "why")
        self.assertEqual(message["content"], "answer")
        self.assertEqual(message["tool_calls"][0]["function"]["arguments"], '{"key":"value"}')
        self.assertEqual(
            [block["type"] for block in anthropic["content"]],
            ["thinking", "text", "tool_use"],
        )
        self.assertEqual(anthropic["content"][-1]["input"], {"key": "value"})

    def test_mixed_unknown_and_duplicate_raw_terminal_fail_closed(self) -> None:
        def pipeline(*, maximum: int = 2, eos: tuple[int, ...] = ()) -> GenerationOutputPipeline:
            return GenerationOutputPipeline(
                request_id="request-1",
                input_token_count=1,
                max_new_tokens=maximum,
                eos_token_ids=eos,
                decoder=MappingDecoder({1: "a", 2: "b", 9: ""}),
            )

        invalid_first_events = (
            TokenEvent("other", 0, TokenEventKind.TOKEN, token_id=1),
            TokenEvent("request-1", 1, TokenEventKind.TOKEN, token_id=1),
            TokenEvent("request-1", 0, TokenEventKind.TEXT_DELTA, text="raw text"),
            TokenEvent(
                "request-1",
                0,
                TokenEventKind.TOKEN,
                token_id=1,
                payload_items=(("unknown", "x"),),
            ),
            TokenEvent("request-1", 0, TokenEventKind.FINISH),
        )
        for event in invalid_first_events:
            with self.subTest(event=event), self.assertRaises(GenerationOutputError):
                pipeline().push(event)

        after_eos = pipeline(eos=(9,))
        after_eos.push(TokenEvent("request-1", 0, TokenEventKind.TOKEN, token_id=9))
        with self.assertRaises(GenerationOutputError):
            after_eos.push(TokenEvent("request-1", 1, TokenEventKind.TOKEN, token_id=1))

        completed = pipeline(maximum=1)
        completed.consume(_raw_events((1,)))
        with self.assertRaisesRegex(GenerationOutputError, "after terminal"):
            completed.push(TokenEvent("request-1", 2, TokenEventKind.FINISH))

    def test_invalid_decoder_or_parser_output_poison_the_pipeline(self) -> None:
        class BadDecoder:
            def decode(self, token_id: int) -> str:
                return 1  # type: ignore[return-value]

            def finish(self) -> str:
                return ""

        class BadParser:
            def push(self, token_id: int, text: str) -> tuple[ParsedOutputDelta, ...]:
                return []  # type: ignore[return-value]

            def finish(self, text: str) -> tuple[ParsedOutputDelta, ...]:
                return ()

        for decoder, parser in (
            (BadDecoder(), None),
            (MappingDecoder({1: "a"}), BadParser()),
        ):
            pipeline = GenerationOutputPipeline(
                request_id="request-1",
                input_token_count=1,
                max_new_tokens=1,
                eos_token_ids=(),
                decoder=decoder,
                parser=parser,
            )
            token = TokenEvent("request-1", 0, TokenEventKind.TOKEN, token_id=1)
            with self.assertRaises(GenerationOutputError):
                pipeline.push(token)
            with self.assertRaisesRegex(GenerationOutputError, "failed"):
                pipeline.push(token)


if __name__ == "__main__":
    unittest.main()

class ContinuousUsageTests(unittest.TestCase):
    def test_hidden_and_buffered_tokens_have_monotonic_progress(self):
        class Decoder:
            def decode(self, token_id):
                return "字" if token_id == 2 else ""
            def finish(self):
                return "tail"
        pipeline = GenerationOutputPipeline(request_id="request-1", input_token_count=7,
            max_new_tokens=3, eos_token_ids=(), decoder=Decoder(), continuous_usage=True)
        events = pipeline.consume(_raw_events((1, 2, 3)))
        self.assertEqual([e.output_token_count for e in events], [1, 2, 3, 3, 3, 3])
        self.assertEqual([e.sequence_no for e in events], list(range(6)))
        self.assertEqual([e.text for e in events[:4]], ["", "字", "", "tail"])
