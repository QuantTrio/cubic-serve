from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from typing import Any, ClassVar

from cubic_serve.api import project_anthropic_message, project_openai_chat_completion
from cubic_serve.contracts.request import TokenEvent, TokenEventKind
from cubic_serve.generation import GenerationOutputPipeline, ParsedOutputDelta

_MODEL_ROOT = Path("/pubdata/llm_models/Qwen/Qwen3.8-27B")
_OUTPUT_FILES = (
    _MODEL_ROOT / "tokenizer.json",
    _MODEL_ROOT / "tokenizer_config.json",
    _MODEL_ROOT / "chat_template.jinja",
)
try:
    import tokenizers as _tokenizers  # noqa: F401
except ModuleNotFoundError:
    _HAS_TOKENIZERS = False
else:
    _HAS_TOKENIZERS = True


def _semantics(
    deltas: tuple[ParsedOutputDelta, ...],
) -> tuple[str, str, tuple[tuple[str, tuple[tuple[str, str], ...]], ...]]:
    reasoning = "".join(
        delta.text for delta in deltas if delta.kind is TokenEventKind.REASONING_DELTA
    )
    text = "".join(delta.text for delta in deltas if delta.kind is TokenEventKind.TEXT_DELTA)
    tools = tuple(
        (delta.text, delta.payload_items)
        for delta in deltas
        if delta.kind is TokenEventKind.TOOL_CALL_DELTA
    )
    return reasoning, text, tools


@unittest.skipUnless(
    _HAS_TOKENIZERS and all(path.is_file() for path in _OUTPUT_FILES),
    "the local Qwen3.8 tokenizer/template and tokenizers package are required",
)
class Qwen3_5OutputTests(unittest.TestCase):
    codec: ClassVar[Any]
    tokenizer: ClassVar[Any]

    @classmethod
    def setUpClass(cls) -> None:
        from tokenizers import Tokenizer

        from cubic_serve.models.qwen3_5.output import Qwen3_5OutputCodec

        cls.codec = Qwen3_5OutputCodec(_MODEL_ROOT)
        cls.tokenizer = Tokenizer.from_file(str(_MODEL_ROOT / "tokenizer.json"))

    def parse_chunks(
        self,
        chunks: tuple[str, ...],
        *,
        enable_thinking: bool = True,
        reasoning_parser: bool = True,
        tool_call_parser: bool = True,
    ) -> tuple[ParsedOutputDelta, ...]:
        parser = self.codec.new_parser(
            enable_thinking=enable_thinking,
            reasoning_parser=reasoning_parser,
            tool_call_parser=tool_call_parser,
        )
        if parser is None:
            text = "".join(chunks)
            return (ParsedOutputDelta(TokenEventKind.TEXT_DELTA, text),) if text else ()
        emitted: list[ParsedOutputDelta] = []
        for index, chunk in enumerate(chunks):
            emitted.extend(parser.push(index, chunk))
        emitted.extend(parser.finish(""))
        return tuple(emitted)

    def test_real_tokenizer_incremental_utf8_golden(self) -> None:
        ids = tuple(self.tokenizer.encode("🫠<tool_call>", add_special_tokens=False).ids)
        self.assertEqual(ids, (9008, 104, 254, 248058))
        decoder = self.codec.new_decoder()
        deltas = tuple(decoder.decode(token_id) for token_id in ids)
        self.assertEqual(deltas, ("", "", "🫠", "<tool_call>"))
        self.assertEqual(decoder.decode(self.codec.eos_token_id), "")
        self.assertEqual(decoder.finish(), "")

    def test_length_stop_discards_only_an_incomplete_utf8_tail(self) -> None:
        decoder = self.codec.new_decoder()
        self.assertEqual(decoder.decode(9008), "")
        self.assertEqual(decoder.decode(104), "")
        self.assertEqual(decoder.finish(), "")

    def test_invalid_generated_byte_is_replaced_without_killing_stream(self) -> None:
        decoder = self.codec.new_decoder()
        self.assertEqual(decoder.decode(222), "�")
        self.assertEqual(decoder.decode(32), "A")
        self.assertEqual(decoder.finish(), "")

    def test_valid_reasoning_and_tool_is_invariant_to_exhaustive_and_random_cuts(
        self,
    ) -> None:
        generated = (
            "推理🫠\n</think>\n\n"
            "<tool_call>\n<function=lookup>\n"
            "<parameter=unit>\nc\n</parameter>\n"
            "<parameter=city>\n北京\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        expected = (
            "推理🫠\n",
            "",
            (
                (
                    '{"city":"北京","unit":"c"}',
                    (("id", "call_0"), ("index", "0"), ("name", "lookup")),
                ),
            ),
        )
        self.assertEqual(_semantics(self.parse_chunks((generated,))), expected)
        for first in range(len(generated) + 1):
            for second in range(first, len(generated) + 1):
                with self.subTest(first=first, second=second):
                    chunks = (
                        generated[:first],
                        generated[first:second],
                        generated[second:],
                    )
                    self.assertEqual(_semantics(self.parse_chunks(chunks)), expected)
        generator = random.Random(0)
        for iteration in range(128):
            cuts = sorted(generator.sample(range(1, len(generated)), generator.randrange(1, 12)))
            chunks = tuple(
                generated[start:end]
                for start, end in zip((0, *cuts), (*cuts, len(generated)), strict=True)
            )
            with self.subTest(random_partition=iteration):
                self.assertEqual(_semantics(self.parse_chunks(chunks)), expected)

    def test_reasoning_and_tool_parsers_have_four_independent_modes(self) -> None:
        tool = (
            "<tool_call>\n<function=lookup>\n"
            "<parameter=city>\n北京\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        generated = f"推理\n</think>\n\n{tool}"
        parsed_tool = (
            (
                '{"city":"北京"}',
                (("id", "call_0"), ("index", "0"), ("name", "lookup")),
            ),
        )
        cases = (
            (False, False, ("", generated, ())),
            (True, False, ("推理\n", f"\n\n{tool}", ())),
            (False, True, ("", "推理\n</think>", parsed_tool)),
            (True, True, ("推理\n", "", parsed_tool)),
        )
        for reasoning_parser, tool_call_parser, expected in cases:
            with self.subTest(
                reasoning_parser=reasoning_parser,
                tool_call_parser=tool_call_parser,
            ):
                self.assertEqual(
                    _semantics(
                        self.parse_chunks(
                            tuple(generated),
                            reasoning_parser=reasoning_parser,
                            tool_call_parser=tool_call_parser,
                        )
                    ),
                    expected,
                )

    def test_disabled_template_thinking_stays_content_with_reasoning_parser(self) -> None:
        generated = "plain answer"
        self.assertEqual(
            _semantics(
                self.parse_chunks(
                    tuple(generated),
                    enable_thinking=False,
                    reasoning_parser=True,
                    tool_call_parser=False,
                )
            ),
            ("", generated, ()),
        )

    def test_reasoning_only_rejects_a_second_think_close_in_content(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid '</think>' nesting inside content"):
            self.parse_chunks(
                tuple("why</think>answer</think>"),
                reasoning_parser=True,
                tool_call_parser=False,
            )

    def test_native_single_call_policy_keeps_usage_for_all_generated_tokens(self) -> None:
        import json

        from cubic_serve.generation.output import request_output_parser
        from cubic_serve.inputs.normalize import normalize_openai_chat_completions

        text = ("<tool_call><function=listed><parameter=x>17</parameter>"
                "</function></tool_call>\n"
                "<tool_call><function=unlisted><parameter=x>19</parameter>"
                "</function></tool_call>")
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        for parallel in (False, True):
            with self.subTest(parallel=parallel):
                request, _ = normalize_openai_chat_completions({
                    "model": "test", "messages": [{"role": "user", "content": "tools"}],
                    "max_tokens": len(ids), "parallel_tool_calls": parallel,
                    "tools": [{"type": "function", "function": {
                        "name": "listed", "parameters": {"type": "object", "properties": {
                            "x": {"type": "integer"},
                        }},
                    }}],
                }, request_id="single")
                parser = request_output_parser(
                    self.codec, request, enable_thinking=False, reasoning_parser="qwen3",
                    tool_call_parser="qwen3_coder",
                )
                pipeline = GenerationOutputPipeline(
                    request_id="single", input_token_count=7, max_new_tokens=len(ids),
                    eos_token_ids=(), decoder=self.codec.new_decoder(), parser=parser,
                    report_reasoning_tokens=True,
                )
                events = []
                for i, token in enumerate(ids):
                    events.extend(pipeline.push(TokenEvent(
                        "single", i, TokenEventKind.TOKEN, token_id=token,
                    )))
                events.extend(pipeline.push(TokenEvent("single", len(ids), TokenEventKind.FINISH)))
                response = project_openai_chat_completion(events, model="test", created=0)
                calls = response["choices"][0]["message"]["tool_calls"]
                self.assertEqual(len(calls), 2 if parallel else 1)
                self.assertEqual(calls[0]["function"]["name"], "listed")
                self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"x": 17})
                self.assertEqual(response["usage"]["completion_tokens"], len(ids))
                self.assertEqual(response["usage"]["completion_tokens_details"]["reasoning_tokens"], 0)

    def test_native_unknown_tool_name_is_output_not_a_stream_error(self) -> None:
        import json

        from cubic_serve.contracts.request import ToolDefinitions
        from cubic_serve.models.qwen3_5.output import Qwen3_5OutputParser

        text = ("<tool_call><function=unlisted><parameter=x>17</parameter>"
                "</function></tool_call>")
        tools = ToolDefinitions.from_value([{
            "type": "function", "function": {"name": "listed", "parameters": {
                "type": "object", "properties": {"x": {"type": "integer"}},
            }},
        }])
        for width in (1, 3, len(text)):
            with self.subTest(width=width):
                parser = Qwen3_5OutputParser(
                    enable_thinking=False, reasoning_parser=False, tool_call_parser=True,
                )
                parser.bind_tools(tools)
                deltas = []
                for index, start in enumerate(range(0, len(text), width)):
                    deltas.extend(parser.push(index, text[start:start + width]))
                deltas.extend(parser.finish(""))
                call = [d for d in deltas if d.kind is TokenEventKind.TOOL_CALL_DELTA]
                self.assertEqual(dict(call[0].payload_items)["name"], "unlisted")
                self.assertEqual(json.loads("".join(d.text for d in call)), {"x": "17"})

    def test_tools_stream_incrementally_with_declared_argument_types(self) -> None:
        import json

        from cubic_serve.contracts.request import ToolDefinitions
        from cubic_serve.models.qwen3_5.output import Qwen3_5OutputParser

        tools = ToolDefinitions.from_value(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "s": {"type": "string"},
                                "i": {"type": "integer"},
                                "b": {"type": "boolean"},
                                "n": {"type": ["null", "string"]},
                                "a": {"type": "array", "items": {"type": "integer"}},
                            },
                        },
                    },
                }
            ]
        )
        text = (
            '<tool_call><function=f><parameter=s>\n001"東京\n</parameter>'
            "<parameter=i>42</parameter><parameter=b>true</parameter>"
            '<parameter=n>null</parameter><parameter=a>["1", "2"]</parameter>'
            "</function></tool_call>"
        )
        for chunks in ((text,), tuple(text)):
            parser = Qwen3_5OutputParser(
                enable_thinking=False, reasoning_parser=False, tool_call_parser=True
            )
            parser.bind_tools(tools)
            emitted = []
            for chunk in chunks:
                emitted.extend(parser.push(1, chunk))
            emitted.extend(parser.finish(""))
            self.assertEqual(
                json.loads("".join(d.text for d in emitted)),
                {"s": '001"東京', "i": 42, "b": True, "n": None, "a": [1, 2]},
            )
            self.assertEqual(dict(emitted[0].payload_items)["name"], "f")
            self.assertTrue(all(dict(d.payload_items).get("index") == "0" for d in emitted))
        parser = Qwen3_5OutputParser(
            enable_thinking=False, reasoning_parser=False, tool_call_parser=True
        )
        parser.bind_tools(tools)
        first = parser.push(1, "<tool_call><function=f><parameter=s>abc")
        self.assertTrue("".join(d.text for d in first).endswith('"abc'))
        # A length stop preserves an incomplete argument string.
        self.assertEqual(parser.truncate(""), ())

    def test_auto_answer_json_preserves_quoted_control_text_then_allows_a_tool(self) -> None:
        from cubic_serve.generation.output import JSONAnswerOutputParser
        from cubic_serve.models.qwen3_5.output import Qwen3_5OutputParser

        answer = '{"value":"<tool_call><think>\\"</think>"}'
        tool = "<tool_call><function=f></function></tool_call>"
        for text in (answer, answer + tool, "call " + tool, '"not json ' + tool):
            for chunks in ((text,), tuple(text)):
                parser = JSONAnswerOutputParser(
                    Qwen3_5OutputParser(
                        enable_thinking=False, reasoning_parser=False, tool_call_parser=True
                    ),
                    "<tool_call>",
                )
                events = []
                for chunk in chunks:
                    events.extend(parser.push(0, chunk))
                events.extend(parser.finish(""))
                reasoning, content, calls = _semantics(tuple(events))
                self.assertEqual(reasoning, "")
                self.assertEqual(
                    content,
                    answer
                    if text.startswith("{")
                    else '"not json'
                    if text.startswith('"')
                    else "call",
                )
                self.assertEqual(len(calls), int(text.endswith(tool)))

    def test_structured_answers_keep_literal_control_text_in_both_reasoning_modes(self) -> None:
        import json

        from cubic_serve.contracts.request import (
            CanonicalChatRequest,
            ChatContentPart,
            ChatMessage,
            ChatRole,
            StructuredOutputMeta,
        )
        from cubic_serve.generation.output import request_output_parser

        answer = json.dumps({"value": "<think></think><tool_call>"})
        request = CanonicalChatRequest(
            "literal",
            "qwen",
            (ChatMessage(ChatRole.USER, (ChatContentPart(0, "text", text="answer"),)),),
            256,
            structured_output=StructuredOutputMeta.json_schema({"type": "object"}),
        )
        for thinking in (False, True):
            parser = request_output_parser(
                self.codec,
                request,
                enable_thinking=thinking,
                reasoning_parser="qwen3",
                tool_call_parser="qwen3_coder",
            )
            text = ("<think>reason</think>" if thinking else "") + answer
            ids = self.tokenizer.encode(text, add_special_tokens=False).ids + [
                self.codec.eos_token_id
            ]
            pipeline = GenerationOutputPipeline(
                request_id="literal",
                input_token_count=1,
                max_new_tokens=len(ids),
                eos_token_ids=self.codec.eos_token_ids,
                decoder=self.codec.new_decoder(),
                parser=parser,
            )
            events = []
            for index, token in enumerate(ids):
                events.extend(
                    pipeline.push(
                        TokenEvent("literal", index, TokenEventKind.TOKEN, token_id=token)
                    )
                )
            events.extend(pipeline.push(TokenEvent("literal", len(ids), TokenEventKind.FINISH)))
            self.assertEqual(
                "".join(e.text or "" for e in events if e.kind is TokenEventKind.TEXT_DELTA), answer
            )
            self.assertEqual(
                "".join(e.text or "" for e in events if e.kind is TokenEventKind.REASONING_DELTA),
                "reason" if thinking else "",
            )
            self.assertFalse(any(e.kind is TokenEventKind.TOOL_CALL_DELTA for e in events))

    def test_content_can_follow_a_completed_tool_call(self) -> None:
        reason, text, calls = _semantics(
            self.parse_chunks(("</think><tool_call><function=f></function></tool_call>tail",))
        )
        self.assertEqual((reason, text), ("", "tail"))
        self.assertEqual(len(calls), 1)

    def test_eos_can_end_reasoning_without_fabricating_a_close(self) -> None:
        reasoning, text, tools = _semantics(self.parse_chunks(("unfinished reasoning",)))
        self.assertEqual(reasoning, "unfinished reasoning")
        self.assertEqual((text, tools), ("", ()))

    def test_tool_parameter_control_text_is_literal_across_chunk_boundaries(self) -> None:
        import json
        from cubic_serve.contracts.request import ToolDefinitions
        from cubic_serve.models.qwen3_5.output import Qwen3_5OutputParser

        value = '<think>literal</think><tool_call></tool_call><function=f>'
        generated = '<tool_call><function=f><parameter=x>\n' + value + '\n</parameter></function></tool_call>'
        for cuts in ((), tuple(range(len(generated)))):
            parser = Qwen3_5OutputParser(enable_thinking=False, reasoning_parser=False, tool_call_parser=True)
            parser.bind_tools(ToolDefinitions.from_value([{'type':'function','function':{'name':'f','parameters':{'type':'object','properties':{'x':{'type':'string'}}}}}]))
            events = []
            start = 0
            for stop in (*cuts, len(generated)):
                events.extend(parser.push(1, generated[start:stop]))
                start = stop
            events.extend(parser.finish(''))
            arguments = ''.join(e.text for e in events if e.kind is TokenEventKind.TOOL_CALL_DELTA)
            self.assertEqual(json.loads(arguments), {'x': value})

    def test_duplicate_and_unclosed_tool_xml_fail_closed(self) -> None:
        invalid = (
            (
                "duplicate",
                "</think><tool_call><function=f>"
                "<parameter=x>1</parameter><parameter=x>2</parameter>"
                "</function></tool_call>",
            ),
            (
                "unclosed",
                "</think><tool_call><function=f><parameter=x>1</parameter>",
            ),
        )
        for label, generated in invalid:
            partitions = (
                (generated,),
                tuple(generated),
                (generated[: len(generated) // 2], generated[len(generated) // 2 :]),
            )
            for chunks in partitions:
                with (
                    self.subTest(label=label, chunk_count=len(chunks)),
                    self.assertRaises(ValueError),
                ):
                    self.parse_chunks(chunks)

    def test_recipe_binds_only_the_selected_output_parsers(self) -> None:
        artifacts = dict(self.codec.artifact_digests)
        self.assertEqual(set(artifacts), {"tokenizer.json", "tokenizer_config.json"})
        from cubic_serve.models.qwen3_5 import output

        decoder_thinking = self.codec.recipe_digest(enable_thinking=True)
        decoder_no_thinking = self.codec.recipe_digest(enable_thinking=False)
        reasoning_thinking = self.codec.recipe_digest(
            enable_thinking=True,
            reasoning_parser=True,
        )
        reasoning_no_thinking = self.codec.recipe_digest(
            enable_thinking=False,
            reasoning_parser=True,
        )
        tool_thinking = self.codec.recipe_digest(
            enable_thinking=True,
            tool_call_parser=True,
        )
        tool_no_thinking = self.codec.recipe_digest(
            enable_thinking=False,
            tool_call_parser=True,
        )
        self.assertRegex(decoder_thinking, r"^b2s256:[0-9a-f]{64}$")
        self.assertEqual(decoder_thinking, decoder_no_thinking)
        self.assertNotEqual(reasoning_thinking, reasoning_no_thinking)
        self.assertEqual(tool_thinking, tool_no_thinking)

        source = (_MODEL_ROOT / "chat_template.jinja").read_text()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jinja") as template:
            template.write(source + "\n{# identity change #}\n")
            template.flush()
            alternate = output.Qwen3_5OutputCodec(
                _MODEL_ROOT,
                chat_template_path=template.name,
            )
            alternate_decoder = alternate.recipe_digest(enable_thinking=True)
            alternate_reasoning = alternate.recipe_digest(
                enable_thinking=True,
                reasoning_parser=True,
            )
        self.assertEqual(decoder_thinking, alternate_decoder)
        self.assertNotEqual(reasoning_thinking, alternate_reasoning)

    def test_recipe_binds_complete_eos_set(self) -> None:
        from unittest.mock import patch

        from cubic_serve.models.qwen3_5 import output

        primary = self.codec.eos_token_id
        with patch.object(output, "load_eos_token_ids", return_value=(primary,)):
            single = output.Qwen3_5OutputCodec(_MODEL_ROOT)
        with patch.object(output, "load_eos_token_ids", return_value=(primary - 1, primary)):
            multiple = output.Qwen3_5OutputCodec(_MODEL_ROOT)
        self.assertNotEqual(
            single.recipe_digest(enable_thinking=True),
            multiple.recipe_digest(enable_thinking=True),
        )

    def test_decoder_only_admission_does_not_require_parser_template_grammar(self) -> None:
        from cubic_serve.models.qwen3_5.output import (
            Qwen3_5OutputCodec,
            Qwen3_5OutputError,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jinja") as template:
            template.write("{{ messages }}")
            template.flush()
            codec = Qwen3_5OutputCodec(
                _MODEL_ROOT,
                chat_template_path=template.name,
            )
            self.assertIsNone(
                codec.new_parser(
                    enable_thinking=True,
                    reasoning_parser=False,
                    tool_call_parser=False,
                )
            )
            decoder = codec.new_decoder()
            self.assertEqual(decoder.decode(codec.eos_token_id), "")
            self.assertEqual(decoder.finish(), "")
            for reasoning_parser, tool_call_parser in ((True, False), (False, True)):
                with (
                    self.subTest(
                        reasoning_parser=reasoning_parser,
                        tool_call_parser=tool_call_parser,
                    ),
                    self.assertRaises(Qwen3_5OutputError),
                ):
                    codec.new_parser(
                        enable_thinking=True,
                        reasoning_parser=reasoning_parser,
                        tool_call_parser=tool_call_parser,
                    )

    def test_no_parser_pipeline_publishes_reasoning_and_tool_markup_as_content(self) -> None:
        generated = "<think>private</think><tool_call><function=lookup></function></tool_call>"
        token_ids = (
            *self.tokenizer.encode(generated, add_special_tokens=False).ids,
            self.codec.eos_token_id,
        )
        raw = (
            *(
                TokenEvent("raw-output", index, TokenEventKind.TOKEN, token_id=token_id)
                for index, token_id in enumerate(token_ids)
            ),
            TokenEvent("raw-output", len(token_ids), TokenEventKind.FINISH),
        )
        pipeline = GenerationOutputPipeline(
            request_id="raw-output",
            input_token_count=1,
            max_new_tokens=len(token_ids) + 1,
            eos_token_ids=(self.codec.eos_token_id,),
            decoder=self.codec.new_decoder(),
            parser=None,
        )

        canonical = pipeline.consume(raw)

        self.assertEqual(
            "".join(event.text or "" for event in canonical),
            generated,
        )
        self.assertFalse(
            any(
                event.kind in (TokenEventKind.REASONING_DELTA, TokenEventKind.TOOL_CALL_DELTA)
                for event in canonical
            )
        )
        self.assertEqual(dict(canonical[-1].payload_items), {"reason": "stop"})

    def test_real_tokens_flow_through_pipeline_and_both_public_projections(self) -> None:
        generated = (
            "考虑\n</think>\n\n<tool_call>\n<function=lookup>\n"
            "<parameter=city>\n北京\n</parameter>\n</function>\n</tool_call>"
        )
        token_ids = (
            *self.tokenizer.encode(generated, add_special_tokens=False).ids,
            self.codec.eos_token_id,
        )
        raw = (
            *(
                TokenEvent("real-output", index, TokenEventKind.TOKEN, token_id=token_id)
                for index, token_id in enumerate(token_ids)
            ),
            TokenEvent("real-output", len(token_ids), TokenEventKind.FINISH),
        )
        pipeline = GenerationOutputPipeline(
            request_id="real-output",
            input_token_count=13,
            max_new_tokens=len(token_ids) + 1,
            eos_token_ids=(self.codec.eos_token_id,),
            decoder=self.codec.new_decoder(),
            parser=self.codec.new_parser(
                enable_thinking=True,
                reasoning_parser=True,
                tool_call_parser=True,
            ),
        )
        canonical = pipeline.consume(raw)
        self.assertEqual(canonical[-1].payload_items, (("reason", "tool_calls"),))
        openai: Any = project_openai_chat_completion(canonical, model="qwen", created=0)
        anthropic: Any = project_anthropic_message(canonical, model="qwen")
        self.assertEqual(openai["choices"][0]["message"]["reasoning_content"], "考虑\n")
        self.assertEqual(
            openai["choices"][0]["message"]["tool_calls"][0]["function"],
            {"name": "lookup", "arguments": '{"city":"北京"}'},
        )
        self.assertEqual(anthropic["content"][0]["type"], "thinking")
        self.assertEqual(anthropic["content"][1]["type"], "tool_use")
        self.assertEqual(anthropic["content"][1]["input"], {"city": "北京"})


if __name__ == "__main__":
    unittest.main()
