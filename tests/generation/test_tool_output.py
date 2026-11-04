from __future__ import annotations

import json
import random
import unittest

from cubic_serve.contracts.request import StructuredOutputMeta
from cubic_serve.generation.output import GenerationOutputError
from cubic_serve.generation.tool_output import ForcedToolOutputParser, tool_json_schema


class ForcedToolParserTests(unittest.TestCase):
    def test_auto_tool_sampling_does_not_change_prompt_or_relax_explicit_constraints(self):
        from cubic_serve.contracts.request import StructuredOutputKind
        from cubic_serve.generation.tool_output import tool_sampling_constraint
        from cubic_serve.inputs.normalize import normalize_openai_chat_completions

        tools = [{"type": "function", "function": {"name": name}} for name in ("f", "g")]
        base = {"model": "m", "messages": [{"role": "user", "content": "question"}],
                "tools": tools, "max_tokens": 16}
        request, _ = normalize_openai_chat_completions(base, request_id="auto")
        original = request.structured_output
        self.assertIs(tool_sampling_constraint(request).kind, StructuredOutputKind.IDENTITY)
        self.assertIs(request.structured_output, original)
        self.assertEqual([t["function"]["name"] for t in request.tools.as_list()], ["f", "g"])
        tag = {"type": "structural_tag", "format": {"type": "const_string", "value": "a"}}
        for overrides in (
            {"tool_choice": "required"},
            {"tool_choice": {"type": "function", "function": {"name": "f"}}},
            {"tools": [{"type": "function", "function": {"name": "f", "strict": True}}]},
            {"structured_outputs": {"structural_tag": tag}},
            {"response_format": {"type": "json_object"}},
            {"tool_choice": {"type": "allowed_tools", "allowed_tools": {
                "mode": "auto", "tools": [{"type": "function", "function": {"name": "f"}}]}}},
        ):
            request, _ = normalize_openai_chat_completions(base | overrides, request_id="protected")
            self.assertIs(tool_sampling_constraint(request), request.structured_output)

    def test_explicit_json_tag_without_native_parser_keeps_forced_tool_projection(self):
        from types import SimpleNamespace

        from cubic_serve.contracts.request import TokenEventKind
        from cubic_serve.generation.output import request_output_parser
        from cubic_serve.generation.tool_output import prepare_forced_tool_request
        from cubic_serve.inputs.normalize import normalize_openai_chat_completions

        for choice, body in (
            ("required", '[{"name":"f","parameters":{"x":"ok"}}]'),
            ({"type": "function", "function": {"name": "f"}}, '{"x":"ok"}'),
        ):
            tag = {"type": "structural_tag", "format": {"type": "const_string", "value": body}}
            request, _ = normalize_openai_chat_completions({
                "model": "m", "messages": [{"role": "user", "content": "question"}],
                "max_tokens": 32, "tools": [{"type": "function", "function": {"name": "f"}}],
                "tool_choice": choice, "structured_outputs": {"structural_tag": tag},
            }, request_id="forced")
            request = prepare_forced_tool_request(request, native_parser=False)
            parser = request_output_parser(
                SimpleNamespace(tool_call_prefix="<tool_call>"), request,
                enable_thinking=False, reasoning_parser=False, tool_call_parser=False,
            )
            events = parser.push(2, body) + parser.finish("")
            self.assertTrue(all(event.kind is TokenEventKind.TOOL_CALL_DELTA for event in events))
            self.assertEqual(json.loads("".join(event.text or "" for event in events)), {"x": "ok"})
            self.assertEqual(dict(events[0].payload_items)["name"], "f")

    def test_explicit_native_tag_is_never_replaced_by_forced_json(self):
        from cubic_serve.generation.tool_output import prepare_forced_tool_request
        from cubic_serve.inputs.normalize import normalize_openai_chat_completions

        tag = {"type": "structural_tag", "format": {"type": "const_string", "value": "a"}}
        for choice in ("auto", "required", {"type": "function", "function": {"name": "f"}}):
            request, _ = normalize_openai_chat_completions({
                "model": "m", "messages": [{"role": "user", "content": "question"}],
                "max_tokens": 32, "tools": [{"type": "function", "function": {"name": "f"}}],
                "tool_choice": choice, "structured_outputs": {"structural_tag": tag},
            }, request_id="native")
            self.assertIs(prepare_forced_tool_request(request, native_parser="native"), request)
            if choice == "auto":
                with self.assertRaisesRegex(ValueError, "requires an explicitly configured"):
                    prepare_forced_tool_request(request, native_parser=False)
            else:
                forced = prepare_forced_tool_request(request, native_parser=False)
                payload = json.loads(forced.structured_output.canonical_payload)
                self.assertEqual(payload["format"], "json")
                self.assertEqual(json.loads(payload["response"]["payload"]), tag)

    def payload(self, choice="required", parallel=True):
        return json.loads(
            StructuredOutputMeta.tool_schema(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "f",
                            "parameters": {
                                "type": "object",
                                "properties": {"x": {"type": "string"}},
                            },
                        },
                    }
                ],
                choice=choice,
                parallel_tool_calls=parallel,
                format="json",
            ).canonical_payload
        )

    def parse(self, text, cuts, *, choice="required", parallel=True, truncated=False):
        parser = ForcedToolOutputParser(self.payload(choice, parallel))
        events = []
        start = 0
        for stop in (*cuts, len(text)):
            events.extend(parser.push(1, text[start:stop]))
            start = stop
        events.extend(parser.truncate("") if truncated else parser.finish(""))
        calls = {}
        for event in events:
            meta = dict(event.payload_items)
            call = calls.setdefault(int(meta["index"]), {"name": "", "arguments": ""})
            call["name"] += meta.get("name", "")
            call["arguments"] += event.text
        return calls

    def test_token_phase_uses_control_ids_and_retains_partial_boundaries(self):
        from cubic_serve.contracts.request import TokenEventKind
        from cubic_serve.generation.output import TokenPhaseOutputParser

        parser = TokenPhaseOutputParser(None, reasoning_start=(7, 8), reasoning_end=(9, 10))
        events = []
        for token, text in (
            (7, "<"),
            (8, "begin>"),
            (1, "literal </think>"),
            (9, "</"),
            (10, "end>"),
            (2, '"<think>"'),
        ):
            events.extend(parser.push(token, text))
        events.extend(parser.finish(""))
        self.assertEqual(
            "".join(e.text for e in events if e.kind is TokenEventKind.REASONING_DELTA),
            "literal </think>",
        )
        self.assertEqual(
            "".join(e.text for e in events if e.kind is TokenEventKind.TEXT_DELTA), '"<think>"'
        )
        parser = TokenPhaseOutputParser(None, reasoning_end=(9, 10))
        self.assertEqual(parser.push(9, "</"), ())
        self.assertEqual(parser.truncate("")[0].text, "</")

    def test_implicit_reasoning_boundary_is_forwarded_as_answer_tokens(self):
        from cubic_serve.generation.output import TokenPhaseOutputParser
        from cubic_serve.contracts.request import TokenEventKind

        parser = TokenPhaseOutputParser(None, reasoning_end=(9,), reasoning_implicit_end=((7, 8),))
        events = parser.push(1, "reason <tool_call>") + parser.push(7, "<tool_")
        events += parser.push(8, "call>") + parser.push(1, "body") + parser.finish("")
        self.assertEqual(''.join(e.text for e in events if e.kind is TokenEventKind.REASONING_DELTA), "reason <tool_call>")
        self.assertEqual(''.join(e.text for e in events if e.kind is TokenEventKind.TEXT_DELTA), "<tool_call>body")

    def test_identity_answer_does_not_parse_unrequested_tool_or_reasoning_markup(self):
        from types import SimpleNamespace
        from cubic_serve.generation.output import request_output_parser
        from cubic_serve.inputs.normalize import normalize_openai_chat_completions

        def unexpected(**kwargs):
            raise AssertionError("ordinary text must not construct a tool parser")

        codec = SimpleNamespace(
            reasoning_start_token_ids=(7,), reasoning_end_token_ids=(9,), new_parser=unexpected
        )
        payload = {"model": "m", "messages": [{"role": "user", "content": "code"}], "max_tokens": 32}
        for tools in ({}, {"tools": [{"type": "function", "function": {"name": "f"}}], "tool_choice": "none"}):
            request, _ = normalize_openai_chat_completions(payload | tools, request_id="literal")
            self.assertIsNone(request_output_parser(codec, request, enable_thinking=False,
                reasoning_parser="qwen3", tool_call_parser="qwen3_coder"))
            parser = request_output_parser(codec, request, enable_thinking=True,
                reasoning_parser="qwen3", tool_call_parser="qwen3_coder")
            events = parser.push(1, "reason") + parser.push(9, "</think>")
            literal = '<think>literal</think><tool_call><function=f>example</function></tool_call>'
            events += parser.push(1, literal) + parser.finish("")
            self.assertEqual([e.text for e in events], ["reason", literal])

    def test_required_and_named_are_chunk_invariant_and_keep_nested_escaped_values(self):
        arguments = {"x": 'a"\\東京\n', "items": [1, {"b": True}], "n": None}
        texts = [
            json.dumps([{"name": "f", "parameters": arguments}], ensure_ascii=False),
            json.dumps([{"parameters": arguments, "name": "f"}], ensure_ascii=False),
        ]
        for text in texts:
            for cuts in (
                (),
                tuple(range(len(text))),
                tuple(sorted(random.Random(7).sample(range(len(text)), 11))),
            ):
                calls = self.parse(text, cuts)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["name"], "f")
                self.assertEqual(json.loads(calls[0]["arguments"]), arguments)
        text = json.dumps(arguments)
        calls = self.parse(text, tuple(range(len(text))), choice="named")
        self.assertEqual(json.loads(calls[0]["arguments"]), arguments)

    def test_parameters_stream_before_call_closes_and_length_preserves_partial_json(self):
        text = '[{"name":"f","parameters":{"x":"unfinished'
        calls = self.parse(text, tuple(range(len(text))), truncated=True)
        self.assertEqual(calls[0], {"name": "f", "arguments": '{"x":"unfinished'})
        with self.assertRaises(GenerationOutputError):
            self.parse(text, (), truncated=False)

    def test_parallel_calls_and_exact_field_ownership(self):
        text = '[{"name":"f","parameters":{}},{"name":"f","parameters":{"x":"b"}}]'
        calls = self.parse(text, tuple(range(len(text))))
        self.assertEqual(len(calls), 2)
        with self.assertRaises(GenerationOutputError):
            self.parse(text, (), parallel=False)
        for text in (
            "[]",
            '[{"name":"foreign","parameters":{}}]',
            '[{"name":"f","name":"f","parameters":{}}]',
            '[{"name":"f","parameters":[]}]',
            '[{"name":"f"}]',
        ):
            with self.subTest(text=text), self.assertRaises(GenerationOutputError):
                self.parse(text, ())

    def test_required_schema_relocates_each_local_pointer_without_mutating_request(self):
        payload = self.payload()
        params = payload["tools"][0]["function"]["parameters"]
        params["$defs"] = {"value": {"type": "integer"}}
        params["properties"] = {"x": {"$ref": "#/$defs/value"}}
        before = json.dumps(payload)
        schema = tool_json_schema(payload)
        self.assertEqual(
            schema["$defs"]["tool0"]["properties"]["x"]["$ref"], "#/$defs/tool0/$defs/value"
        )
        self.assertEqual(json.dumps(payload), before)


if __name__ == "__main__":
    unittest.main()
