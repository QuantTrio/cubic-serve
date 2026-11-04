from __future__ import annotations

import json
import unittest
from typing import Any, cast
from unittest.mock import patch

import cubic_serve.inputs.normalize as normalization
from cubic_serve.contracts.request import (
    CanonicalChatRequest,
    ChatContentPart,
    ChatMessage,
    ChatRole,
    ChatTemplateKwargs,
    MediaSourceKind,
    SamplingMeta,
    StructuredOutputKind,
)
from cubic_serve.inputs import (
    ChatNormalizationError,
    normalize_anthropic_messages,
    normalize_openai_chat_completions,
)

_INLINE_PNG = "iVBORw=="


def _openai_payload() -> dict[str, Any]:
    return {
        "model": "fixture-model",
        "max_completion_tokens": 17,
        "stream": True,
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "rules"}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_INLINE_PNG}"},
                    },
                    {"type": "text", "text": ""},
                    {
                        "type": "video_url",
                        "video_url": {"url": "https://media.example/clip.mp4"},
                    },
                    {"type": "text", "text": "after"},
                ],
            },
            {"role": "assistant", "content": "answer"},
        ],
    }


def _anthropic_payload() -> dict[str, Any]:
    return {
        "model": "fixture-model",
        "max_tokens": 17,
        "stream": True,
        "system": [{"type": "text", "text": "rules"}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": _INLINE_PNG,
                        },
                    },
                    {"type": "text", "text": ""},
                    {
                        "type": "video",
                        "source": {
                            "type": "url",
                            "url": "https://media.example/clip.mp4",
                        },
                    },
                    {"type": "text", "text": "after"},
                ],
            },
            {"role": "assistant", "content": "answer"},
        ],
    }


class ChatNormalizationTests(unittest.TestCase):
    def test_vllm_nullable_tool_parameters_and_stream_flags(self) -> None:
        payload = _openai_payload()
        payload.update(
            tools=[{"type": "function", "function": {"name": "f", "parameters": None}}],
            stream=True, stream_options={"include_usage": None, "continuous_usage_stats": None},
        )
        request, stream = normalize_openai_chat_completions(payload, request_id="nullable")
        self.assertTrue(stream)
        self.assertFalse(request.stream_include_usage)
        self.assertEqual(request.tools.as_list()[0]["function"]["parameters"], {
            "type": "object", "properties": {},
        })
        payload.pop("stream_options")
        payload["stream"] = None
        _, stream = normalize_openai_chat_completions(payload, request_id="nullable")
        self.assertFalse(stream)

    def test_nested_structural_tag_defaults_its_type_like_xgrammar(self) -> None:
        payload = _openai_payload()
        payload["structured_outputs"] = {
            "structural_tag": json.dumps({"format": {"type": "const_string", "value": "a"}}),
        }
        request, _ = normalize_openai_chat_completions(payload, request_id="tag")
        self.assertEqual(json.loads(request.structured_output.canonical_payload), {
            "type": "structural_tag", "format": {"type": "const_string", "value": "a"},
        })

    def test_openai_thinking_budget_validates_zero_unlimited_and_invalid_values(self) -> None:
        for value, expected in ((None, None), (-1, None), (0, 0), (17, 17)):
            payload = _openai_payload()
            payload["thinking_token_budget"] = value
            request, _ = normalize_openai_chat_completions(payload, request_id="budget")
            self.assertEqual(request.thinking_token_budget, expected)
        for value in (True, 1.0, -2, "10"):
            payload["thinking_token_budget"] = value
            with self.subTest(value=value), self.assertRaises(ChatNormalizationError):
                normalize_openai_chat_completions(payload, request_id="invalid")

    def test_anthropic_thinking_maps_to_common_budget_effort_and_template_controls(self) -> None:
        for thinking, budget, effort in (
            ({"type": "enabled", "budget_tokens": 1024, "display": "omitted"}, 1024, "high"),
            ({"type": "disabled"}, None, "none"),
            ({"type": "adaptive", "display": "summarized"}, None, "high"),
        ):
            payload = _anthropic_payload()
            payload.update(thinking=thinking, output_config={"effort": "high"})
            request, _ = normalize_anthropic_messages(payload, request_id="thinking")
            self.assertEqual(request.thinking_token_budget, budget)
            self.assertEqual(request.chat_template_kwargs.as_dict()["reasoning_effort"], effort)
        for thinking in (
            {"type": "enabled", "budget_tokens": 1023},
            {"type": "enabled", "budget_tokens": True},
            {"type": "disabled", "display": "omitted"},
            {"type": "adaptive", "budget_tokens": 1024},
        ):
            payload["thinking"] = thinking
            with self.subTest(thinking=thinking), self.assertRaises(ChatNormalizationError):
                normalize_anthropic_messages(payload, request_id="invalid")

    def test_anthropic_parallel_policy_matches_openai_metadata(self) -> None:
        for kind in ("auto", "any", "tool"):
            for disable in (True, False, None):
                payload = _anthropic_payload()
                payload["tools"] = [{"name": "f", "input_schema": {"type": "object"}}]
                payload["tool_choice"] = {
                    "type": kind, "name": "f" if kind == "tool" else None,
                    "disable_parallel_tool_use": disable,
                }
                request, _ = normalize_anthropic_messages(payload, request_id="parallel")
                self.assertIs(request.parallel_tool_calls, disable is not True)
                grammar = json.loads(request.structured_output.canonical_payload)
                self.assertIs(grammar.get("parallel_tool_calls", True), disable is not True)
        payload["tool_choice"]["disable_parallel_tool_use"] = "false"
        with self.assertRaises(ChatNormalizationError):
            normalize_anthropic_messages(payload, request_id="invalid")

    def test_anthropic_output_effort_can_accompany_schema_or_stand_alone(self) -> None:
        for format_ in (None, {"type": "json_schema", "schema": {"type": "integer"}}):
            payload = _anthropic_payload()
            payload["output_config"] = {"effort": "xhigh", "format": format_}
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            request, _ = normalize_anthropic_messages(payload, request_id="effort")
            self.assertEqual(request.chat_template_kwargs.as_dict(), {
                "enable_thinking": False, "reasoning_effort": "xhigh",
            })
            self.assertIs(
                request.structured_output.kind,
                (StructuredOutputKind.IDENTITY
                 if format_ is None else StructuredOutputKind.JSON_SCHEMA),
            )
        payload["output_config"]["effort"] = "unsupported"
        with self.assertRaises(ChatNormalizationError):
            normalize_anthropic_messages(payload, request_id="invalid")

    def test_response_schema_description_and_nullable_strict_do_not_change_grammar(self) -> None:
        payload = _openai_payload()
        payload["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "answer", "description": "Response contract", "strict": None,
            "schema": {"type": "integer"},
        }}
        request, _ = normalize_openai_chat_completions(payload, request_id="request-1")
        self.assertEqual(
            json.loads(request.structured_output.canonical_payload), {"type": "integer"}
        )

    def test_nullable_optional_structured_fields_and_reasoning_visibility(self) -> None:
        payload = _openai_payload()
        payload.update(
            tools=None, response_format=None, parallel_tool_calls=None, include_reasoning=False
        )
        request, _ = normalize_openai_chat_completions(payload, request_id="request-1")
        self.assertFalse(request.include_reasoning)
        self.assertTrue(request.parallel_tool_calls)
        self.assertIs(request.structured_output.kind, StructuredOutputKind.IDENTITY)
        for invalid in (None, 0, "false"):
            payload["include_reasoning"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ChatNormalizationError):
                normalize_openai_chat_completions(payload, request_id="request-1")

    def test_openai_ignore_eos_is_strict_and_defaults_off(self) -> None:
        payload = _openai_payload()
        payload["ignore_eos"] = True
        request, _ = normalize_openai_chat_completions(payload, request_id="request-1")
        self.assertTrue(request.ignore_eos)

        default_request, _ = normalize_openai_chat_completions(
            _openai_payload(), request_id="request-1"
        )
        self.assertFalse(default_request.ignore_eos)

        payload["ignore_eos"] = 1
        with self.assertRaisesRegex(ChatNormalizationError, "ignore_eos.*boolean"):
            normalize_openai_chat_completions(payload, request_id="request-1")

    def test_openai_template_kwargs_are_strict_immutable_json(self) -> None:
        payload = _openai_payload()
        raw_kwargs: dict[str, object] = {
            "nested": {
                "array": [None, True, 7, 1.25, "value"],
            },
            "z": "last",
        }
        payload["chat_template_kwargs"] = raw_kwargs

        request, _ = normalize_openai_chat_completions(payload, request_id="request-1")
        self.assertIsInstance(request.chat_template_kwargs, ChatTemplateKwargs)
        self.assertEqual(
            request.chat_template_kwargs.canonical_payload,
            '{"nested":{"array":[null,true,7,1.25,"value"]},"z":"last"}',
        )

        cast(dict[str, object], raw_kwargs["nested"])["array"] = ["mutated"]
        first_copy = request.chat_template_kwargs.as_dict()
        cast(dict[str, object], first_copy["nested"])["array"] = ["mutated again"]
        self.assertEqual(
            request.chat_template_kwargs.as_dict(),
            {"nested": {"array": [None, True, 7, 1.25, "value"]}, "z": "last"},
        )

    def test_openai_template_kwargs_reject_python_only_or_nonfinite_values(self) -> None:
        invalid_values: tuple[object, ...] = (
            [],
            {"tuple": (1, 2)},
            {"nonfinite": float("nan")},
            {1: "non-string key"},
        )
        for value in invalid_values:
            payload = _openai_payload()
            payload["chat_template_kwargs"] = value
            with self.subTest(value=value), self.assertRaises(ChatNormalizationError):
                normalize_openai_chat_completions(payload, request_id="request-1")

        with self.assertRaisesRegex(ValueError, "chat template kwargs payload"):
            ChatTemplateKwargs('{"noncanonical": true}')

    def test_top_level_reasoning_effort_has_upstream_template_precedence(self) -> None:
        explicit_enable = _openai_payload()
        explicit_enable["chat_template_kwargs"] = {
            "enable_thinking": False,
            "reasoning_effort": "low",
        }
        explicit_enable["reasoning_effort"] = "xhigh"
        request, _ = normalize_openai_chat_completions(explicit_enable, request_id="request-1")
        self.assertEqual(
            request.chat_template_kwargs.as_dict(),
            {"enable_thinking": False, "reasoning_effort": "xhigh"},
        )

        for effort, enabled in (("none", False), ("medium", True)):
            payload = _openai_payload()
            payload["reasoning_effort"] = effort
            request, _ = normalize_openai_chat_completions(payload, request_id="request-1")
            self.assertEqual(
                request.chat_template_kwargs.as_dict(),
                {"enable_thinking": enabled, "reasoning_effort": effort},
            )

        invalid = _openai_payload()
        invalid["reasoning_effort"] = "ultra"
        with self.assertRaisesRegex(ChatNormalizationError, "reasoning_effort"):
            normalize_openai_chat_completions(invalid, request_id="request-1")

    def test_openai_stream_options_normalize_only_include_usage(self) -> None:
        payload = _openai_payload()
        payload["stream_options"] = {"include_usage": True}
        request, stream = normalize_openai_chat_completions(payload, request_id="request-1")
        self.assertTrue(stream)
        self.assertTrue(request.stream_include_usage)

        default_request, _ = normalize_openai_chat_completions(
            _openai_payload(), request_id="request-1"
        )
        self.assertFalse(default_request.stream_include_usage)

        non_streaming = _openai_payload()
        non_streaming["stream"] = False
        non_streaming["stream_options"] = {"include_usage": False}
        unsupported = _openai_payload()
        unsupported["stream_options"] = {"continuous_usage_stats": True}
        malformed = _openai_payload()
        malformed["stream_options"] = {"include_usage": 1}
        for value in (non_streaming, unsupported, malformed):
            with self.subTest(payload=value), self.assertRaises(ChatNormalizationError):
                normalize_openai_chat_completions(value, request_id="request-1")

    def test_both_protocols_share_tool_json_and_grammar_metadata(self) -> None:
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }
        openai = _openai_payload()
        openai["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": schema,
                    "strict": True,
                },
            }
        ]
        openai["tool_choice"] = "required"
        anthropic = _anthropic_payload()
        anthropic["tools"] = [
            {
                "name": "weather",
                "description": "Get weather",
                "input_schema": schema,
                "strict": True,
            }
        ]
        anthropic["tool_choice"] = {"type": "any"}

        openai_request, _ = normalize_openai_chat_completions(openai, request_id="request-1")
        anthropic_request, _ = normalize_anthropic_messages(anthropic, request_id="request-1")
        self.assertEqual(openai_request, anthropic_request)
        self.assertEqual(
            openai_request.structured_output.kind,
            StructuredOutputKind.TOOL_SCHEMA,
        )

        openai.pop("tools")
        openai.pop("tool_choice")
        openai["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "weather", "strict": True, "schema": schema},
        }
        anthropic.pop("tools")
        anthropic.pop("tool_choice")
        anthropic["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        openai_request, _ = normalize_openai_chat_completions(openai, request_id="request-1")
        anthropic_request, _ = normalize_anthropic_messages(anthropic, request_id="request-1")
        self.assertEqual(openai_request, anthropic_request)
        self.assertEqual(
            openai_request.structured_output.kind,
            StructuredOutputKind.JSON_SCHEMA,
        )

        grammar = 'root ::= "yes" | "no"\n'
        openai.pop("response_format")
        anthropic.pop("output_config")
        openai["grammar"] = anthropic["grammar"] = grammar
        openai_request, _ = normalize_openai_chat_completions(openai, request_id="request-1")
        anthropic_request, _ = normalize_anthropic_messages(anthropic, request_id="request-1")
        self.assertEqual(openai_request, anthropic_request)
        self.assertEqual(openai_request.structured_output.kind, StructuredOutputKind.GRAMMAR)

    def test_vllm_structured_output_selectors_are_canonical(self) -> None:
        cases = (
            ({"json": '{"type":"integer"}'}, StructuredOutputKind.JSON_SCHEMA),
            ({"json_object": True}, StructuredOutputKind.JSON_SCHEMA),
            ({"regex": "a[bc]"}, StructuredOutputKind.REGEX),
            ({"choice": ["ab", "ac", ""]}, StructuredOutputKind.CHOICE),
            ({"grammar": 'root ::= "a"'}, StructuredOutputKind.GRAMMAR),
            ({"grammar": 'start: "a"'}, StructuredOutputKind.GRAMMAR),
            (
                {
                    "structural_tag": {
                        "type": "structural_tag",
                        "format": {"type": "const_string", "value": "a"},
                    }
                },
                StructuredOutputKind.STRUCTURAL_TAG,
            ),
        )
        for constraint, kind in cases:
            with self.subTest(constraint=constraint):
                payload = _openai_payload()
                payload["structured_outputs"] = constraint
                request, _ = normalize_openai_chat_completions(payload, request_id="structured")
                self.assertIs(request.structured_output.kind, kind)

    def test_structured_format_override_preserves_options_and_prefix_identity(self) -> None:
        payload = _openai_payload()
        payload["structured_outputs"] = {"json": {"type": "number"}, "disable_any_whitespace": True}
        payload["response_format"] = {"type": "json_object"}
        request, _ = normalize_openai_chat_completions(payload, request_id="structured")
        self.assertEqual(request.structured_output.canonical_payload, '{"type":"object"}')
        self.assertEqual(
            request.structured_output.canonical_options, '{"disable_any_whitespace":true}'
        )
        payload["structured_outputs"] = {"json": {"type": "number"}}
        normal, _ = normalize_openai_chat_completions(payload, request_id="structured")
        self.assertNotEqual(request.structured_output.digest, normal.structured_output.digest)

    def test_structured_conflicts_and_malformed_selectors_fail_before_admission(self) -> None:
        for constraint in (
            {},
            {"regex": "a", "choice": ["a"]},
            {"choice": []},
            {"choice": [1]},
            {"json_object": False},
            {"regex": "\x00"},
            {"json": {}, "disable_any_whitespace": 1},
            {"structural_tag": {"type": "structural_tag"}},
            {"grammar": ""},
            {"json": "not JSON"},
        ):
            with self.subTest(constraint=constraint), self.assertRaises(ChatNormalizationError):
                payload = _openai_payload()
                payload["structured_outputs"] = constraint
                normalize_openai_chat_completions(payload, request_id="bad")

    def test_structured_conflicts_and_unknown_tool_shapes_fail_before_admission(self) -> None:
        valid_tool = {
            "type": "function",
            "function": {
                "name": "weather",
                "parameters": {"type": "object"},
                "strict": True,
            },
        }
        cases: list[dict[str, Any]] = []
        conflict = _openai_payload()
        conflict.update(
            {
                "tools": [valid_tool],
                "grammar": 'root ::= "x"',
                "structured_outputs": {"regex": "x"},
            }
        )
        cases.append(conflict)
        unknown = _openai_payload()
        unknown["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "parameters": {},
                    "strict": True,
                    "implementation": "python",
                },
            }
        ]
        cases.append(unknown)
        absent = _openai_payload()
        absent.update(
            {
                "tools": [valid_tool],
                "tool_choice": {"type": "function", "function": {"name": "missing"}},
            }
        )
        cases.append(absent)
        for payload in cases:
            with self.subTest(payload=payload), self.assertRaises(ChatNormalizationError):
                normalize_openai_chat_completions(payload, request_id="request-1")

        nonfinite = _openai_payload()
        nonfinite["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "nonfinite",
                "schema": {"minimum": float("nan")},
                "strict": True,
            },
        }
        with self.assertRaisesRegex(ChatNormalizationError, "structured output is invalid"):
            normalize_openai_chat_completions(nonfinite, request_id="request-1")

    def test_non_strict_and_default_public_structured_requests_are_canonical(self) -> None:
        for strict in (None, False):
            openai = _openai_payload()
            function: dict[str, object] = {
                "name": "weather",
                "parameters": {"type": "object"},
            }
            if strict is not None:
                function["strict"] = strict
            openai["tools"] = [{"type": "function", "function": function}]
            openai_request, _ = normalize_openai_chat_completions(openai, request_id="request-1")
            self.assertEqual(
                openai_request.structured_output.kind,
                StructuredOutputKind.TOOL_SCHEMA,
            )
            self.assertIn('"strict":false', openai_request.structured_output.canonical_payload)

            anthropic = _anthropic_payload()
            tool: dict[str, object] = {
                "name": "weather",
                "input_schema": {"type": "object"},
            }
            if strict is not None:
                tool["strict"] = strict
            anthropic["tools"] = [tool]
            anthropic_request, _ = normalize_anthropic_messages(anthropic, request_id="request-1")
            self.assertEqual(openai_request, anthropic_request)

            openai = _openai_payload()
            wrapper: dict[str, object] = {
                "name": "answer",
                "schema": {"type": "object"},
            }
            if strict is not None:
                wrapper["strict"] = strict
            openai["response_format"] = {
                "type": "json_schema",
                "json_schema": wrapper,
            }
            json_request, _ = normalize_openai_chat_completions(openai, request_id="request-1")
            self.assertEqual(
                json_request.structured_output.kind,
                StructuredOutputKind.JSON_SCHEMA,
            )

    def test_tool_choice_none_preserves_definitions_without_a_constraint(self) -> None:
        tool = {
            "type": "function",
            "function": {
                "name": "weather",
                "parameters": {"type": "object"},
            },
        }
        openai = _openai_payload()
        openai.update({"tools": [tool], "tool_choice": "none"})
        anthropic = _anthropic_payload()
        anthropic.update(
            {
                "tools": [{"name": "weather", "input_schema": {"type": "object"}}],
                "tool_choice": {"type": "none"},
            }
        )
        openai_request, _ = normalize_openai_chat_completions(openai, request_id="request-1")
        anthropic_request, _ = normalize_anthropic_messages(anthropic, request_id="request-1")
        self.assertEqual(openai_request, anthropic_request)
        self.assertEqual(
            openai_request.structured_output.kind,
            StructuredOutputKind.IDENTITY,
        )

    def test_tool_history_and_reasoning_are_immutable_model_neutral_metadata(self) -> None:
        payload = _openai_payload()
        payload["messages"] = [
            {"role": "user", "content": "Find the weather"},
            {
                "role": "assistant",
                "content": None,
                "reasoning": "Look it up",
                "tool_calls": [
                    {
                        "id": "call-a",
                        "type": "function",
                        "function": {"name": "weather", "arguments": '{"city":"北京","days":2}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-a", "content": "sunny"},
        ]
        request, _ = normalize_openai_chat_completions(payload, request_id="history")
        assistant = request.messages[1]
        self.assertEqual(assistant.reasoning, "Look it up")
        self.assertEqual(
            assistant.template_metadata()["tool_calls"][0]["function"]["arguments"],
            {"city": "北京", "days": 2},
        )
        self.assertIs(request.messages[2].role, ChatRole.TOOL)
        self.assertEqual(request.messages[2].tool_call_id, "call-a")
        payload["messages"][1]["tool_calls"][0]["function"]["arguments"] = "{}"
        self.assertIn("days", assistant.tool_calls[0].canonical_arguments)

    def test_anthropic_tool_and_reasoning_history_share_the_openai_canonical_contract(self) -> None:
        anthropic = {
            "model": "fixture",
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "choose", "signature": "opaque"},
                        {
                            "type": "tool_use",
                            "id": "call-a",
                            "name": "weather",
                            "input": {"city": "Tokyo"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call-a", "content": "sunny"},
                        {"type": "text", "text": "thanks"},
                    ],
                },
            ],
        }
        openai = {
            "model": "fixture",
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning": "choose",
                    "tool_calls": [
                        {
                            "id": "call-a",
                            "type": "function",
                            "function": {"name": "weather", "arguments": '{"city":"Tokyo"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-a", "content": "sunny"},
                {"role": "user", "content": "thanks"},
            ],
        }
        left, _ = normalize_anthropic_messages(anthropic, request_id="history")
        right, _ = normalize_openai_chat_completions(openai, request_id="history")
        self.assertEqual(left.messages, right.messages)
        anthropic["messages"][-1]["content"].insert(0, {"type": "text", "text": "before"})
        request, _ = normalize_anthropic_messages(anthropic, request_id="ordered")
        self.assertEqual(
            [m.role for m in request.messages[-3:]], [ChatRole.USER, ChatRole.TOOL, ChatRole.USER]
        )

    def test_allowed_tool_subset_and_auto_answer_format_are_independent_of_prompt_definitions(
        self,
    ) -> None:
        payload = _openai_payload()
        payload["tools"] = [
            {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
            for name in ("weather", "lookup")
        ]
        payload["tool_choice"] = {
            "type": "allowed_tools",
            "allowed_tools": {
                "mode": "auto",
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
            },
        }
        payload["structured_outputs"] = {"choice": ["yes", "no"]}
        request, _ = normalize_openai_chat_completions(payload, request_id="combined")
        self.assertEqual(len(request.tools.as_list()), 2)
        grammar = json.loads(request.structured_output.canonical_payload)
        self.assertEqual(grammar["tools"][0]["function"]["name"], "lookup")
        self.assertEqual(grammar["response"]["kind"], "choice")
        payload["tool_choice"] = "required"
        request, _ = normalize_openai_chat_completions(payload, request_id="forced")
        self.assertNotIn("response", json.loads(request.structured_output.canonical_payload))

    def test_parallel_policy_and_named_choice_preserve_full_tool_definitions(self) -> None:
        payload = _openai_payload()
        payload["tools"] = [
            {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
            for name in ("weather", "lookup")
        ]
        payload["tool_choice"] = {"type": "function", "function": {"name": "lookup"}}
        payload["parallel_tool_calls"] = False
        request, _ = normalize_openai_chat_completions(payload, request_id="named")
        self.assertEqual(len(request.tools.as_list()), 2)
        self.assertFalse(request.parallel_tool_calls)
        self.assertIn('"parallel_tool_calls":false', request.structured_output.canonical_payload)
        self.assertNotIn('"name":"weather"', request.structured_output.canonical_payload)

    def test_explicit_structural_tag_survives_every_tool_choice(self) -> None:
        tag = {"type": "structural_tag", "format": {"type": "const_string", "value": "a"}}
        for choice in ("auto", "required", {"type": "function", "function": {"name": "lookup"}}):
            payload = _openai_payload()
            payload.update(
                tools=[{"type": "function", "function": {
                    "name": "lookup", "parameters": {"type": "object"},
                }}], tool_choice=choice, structured_outputs={"structural_tag": tag},
            )
            request, _ = normalize_openai_chat_completions(payload, request_id="combined")
            meta = json.loads(request.structured_output.canonical_payload)
            self.assertEqual(meta["choice"], choice if isinstance(choice, str) else "named")
            self.assertEqual(json.loads(meta["response"]["payload"]), tag)

    def test_invalid_tool_history_and_conflicting_reasoning_fail_before_admission(self) -> None:
        cases = (
            {"role": "tool", "content": "result"},
            {"role": "user", "content": "x", "reasoning": "hidden"},
            {"role": "assistant", "content": "x", "reasoning": "a", "reasoning_content": "b"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "a", "function": {"name": "f", "arguments": "not JSON"}}],
            },
        )
        for message in cases:
            with self.subTest(message=message), self.assertRaises(ChatNormalizationError):
                payload = _openai_payload()
                payload["messages"] = [message]
                normalize_openai_chat_completions(payload, request_id="bad-history")

    def test_openai_and_anthropic_converge_without_reordering_parts(self) -> None:
        openai, openai_stream = normalize_openai_chat_completions(
            _openai_payload(), request_id="request-1"
        )
        anthropic, anthropic_stream = normalize_anthropic_messages(
            _anthropic_payload(), request_id="request-1"
        )

        self.assertEqual(openai, anthropic)
        self.assertTrue(openai_stream)
        self.assertEqual(openai_stream, anthropic_stream)
        self.assertEqual(
            tuple(message.role for message in openai.messages),
            (ChatRole.SYSTEM, ChatRole.USER, ChatRole.ASSISTANT),
        )
        self.assertEqual(
            tuple(len(message.content_parts) for message in openai.messages),
            (1, 5, 1),
        )
        parts = tuple(part for message in openai.messages for part in message.content_parts)
        self.assertEqual(tuple(part.ordinal for part in parts), tuple(range(7)))
        self.assertEqual(
            tuple(part.modality for part in parts),
            ("text", "text", "image", "text", "video", "text", "text"),
        )
        self.assertEqual(parts[3].text, "")
        inline_image = parts[2].media
        remote_video = parts[4].media
        assert inline_image is not None and remote_video is not None
        self.assertEqual(inline_image.kind, MediaSourceKind.BASE64)
        self.assertEqual(inline_image.mime_type, "image/png")
        self.assertIsNone(inline_image.uri)
        self.assertEqual(remote_video.kind, MediaSourceKind.URL)

    def test_openai_max_tokens_alias_has_one_unambiguous_canonical_field(self) -> None:
        current_payload = _openai_payload()
        legacy_payload = _openai_payload()
        legacy_payload["max_tokens"] = legacy_payload.pop("max_completion_tokens")

        current = normalize_openai_chat_completions(current_payload, request_id="request-1")
        legacy = normalize_openai_chat_completions(legacy_payload, request_id="request-1")
        self.assertEqual(current, legacy)

        for fields, expected in (
            ({}, None),
            ({"max_tokens": None}, None),
            ({"max_completion_tokens": None}, None),
            ({"max_tokens": None, "max_completion_tokens": None}, None),
            ({"max_tokens": 7, "max_completion_tokens": None}, 7),
            ({"max_tokens": None, "max_completion_tokens": 11}, 11),
            ({"max_tokens": 7, "max_completion_tokens": 11}, 11),
        ):
            with self.subTest(fields=fields):
                payload = _openai_payload()
                payload.pop("max_completion_tokens")
                payload.update(fields)
                request, _ = normalize_openai_chat_completions(payload, request_id="request-1")
                self.assertEqual(request.max_new_tokens, expected)

    def test_openai_token_limits_validate_each_non_null_field(self) -> None:
        for field in ("max_tokens", "max_completion_tokens"):
            for value in (False, 0, -1, 1.5, "17"):
                with self.subTest(field=field, value=value):
                    payload = _openai_payload()
                    payload.update({"max_tokens": 7, "max_completion_tokens": 11, field: value})
                    with self.assertRaises(ChatNormalizationError):
                        normalize_openai_chat_completions(payload, request_id="request-1")

    def test_sampling_is_canonical_and_absence_preserves_model_defaults(self) -> None:
        openai = _openai_payload()
        openai.update(temperature=0.75, top_p=0.9, top_k=32, seed=-17)
        anthropic = _anthropic_payload()
        anthropic.update(temperature=0.5, top_p=0.8, top_k=16)

        openai_request, _ = normalize_openai_chat_completions(openai, request_id="request-1")
        anthropic_request, _ = normalize_anthropic_messages(anthropic, request_id="request-2")
        default_request, _ = normalize_openai_chat_completions(
            _openai_payload(), request_id="request-3"
        )

        self.assertEqual(openai_request.sampling, SamplingMeta(0.75, 0.9, 32, -17))
        self.assertEqual(anthropic_request.sampling, SamplingMeta(0.5, 0.8, 16))
        self.assertEqual(default_request.sampling, SamplingMeta())

    def test_sampling_rejects_noncanonical_values(self) -> None:
        cases = (
            ("temperature", -0.1),
            ("temperature", 2.1),
            ("temperature", float("nan")),
            ("temperature", 10**1000),
            ("temperature", True),
            ("top_p", 0.0),
            ("top_p", 1.1),
            ("top_p", float("inf")),
            ("top_k", 0),
            ("top_k", 1.0),
            ("top_k", True),
            ("seed", -(2**63) - 1),
            ("seed", 2**63),
            ("seed", 1.0),
            ("seed", True),
        )
        for field, value in cases:
            payload = _openai_payload()
            payload[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ChatNormalizationError):
                normalize_openai_chat_completions(payload, request_id="request-1")

        anthropic = _anthropic_payload()
        anthropic["temperature"] = 1.1
        with self.assertRaises(ChatNormalizationError):
            normalize_anthropic_messages(anthropic, request_id="request-1")

    def test_unsupported_fields_and_unknown_nested_fields_fail_closed(self) -> None:
        openai_cases: list[tuple[str, dict[str, Any]]] = []
        for field, value in (
            ("stop", ["done"]),
            ("tools", []),
        ):
            payload = _openai_payload()
            payload[field] = value
            openai_cases.append((field, payload))
        nested = _openai_payload()
        nested["messages"][1]["content"][1]["image_url"]["detail"] = "auto"
        openai_cases.append(("image detail", nested))

        for label, payload in openai_cases:
            with (
                self.subTest(protocol="openai", field=label),
                self.assertRaises(ChatNormalizationError),
            ):
                normalize_openai_chat_completions(payload, request_id="request-1")

        for field, value in (("seed", 7), ("tools", [])):
            payload = _anthropic_payload()
            payload[field] = value
            with (
                self.subTest(protocol="anthropic", field=field),
                self.assertRaises(ChatNormalizationError),
            ):
                normalize_anthropic_messages(payload, request_id="request-1")

    def test_unsupported_roles_and_content_types_fail_closed(self) -> None:
        openai_role = _openai_payload()
        openai_role["messages"][1]["role"] = "tool"

        openai_audio = _openai_payload()
        openai_audio["messages"][1]["content"] = [{"type": "input_audio", "input_audio": {}}]

        anthropic_role = _anthropic_payload()
        anthropic_role["messages"][0]["role"] = "system"

        anthropic_tool = _anthropic_payload()
        anthropic_tool["messages"][0]["content"] = [{"type": "tool_use", "id": "tool-1"}]

        cases = (
            (normalize_openai_chat_completions, openai_role),
            (normalize_openai_chat_completions, openai_audio),
            (normalize_anthropic_messages, anthropic_role),
            (normalize_anthropic_messages, anthropic_tool),
        )
        for normalizer, payload in cases:
            with (
                self.subTest(normalizer=normalizer.__name__),
                self.assertRaises(ChatNormalizationError),
            ):
                normalizer(payload, request_id="request-1")

    def test_malformed_values_and_non_boolean_stream_fail_closed(self) -> None:
        invalid_stream = _openai_payload()
        invalid_stream["stream"] = "true"

        boolean_limit = _anthropic_payload()
        boolean_limit["max_tokens"] = True

        empty_content = _openai_payload()
        empty_content["messages"][1]["content"] = []

        invalid_base64 = _anthropic_payload()
        invalid_base64["messages"][0]["content"][1]["source"]["data"] = "AB=="

        wrong_mime = _anthropic_payload()
        wrong_mime["messages"][0]["content"][1]["source"]["media_type"] = "video/mp4"

        relative_url = _openai_payload()
        relative_url["messages"][1]["content"][3]["video_url"]["url"] = "./clip.mp4"

        cases = (
            (normalize_openai_chat_completions, invalid_stream),
            (normalize_anthropic_messages, boolean_limit),
            (normalize_openai_chat_completions, empty_content),
            (normalize_anthropic_messages, invalid_base64),
            (normalize_anthropic_messages, wrong_mime),
            (normalize_openai_chat_completions, relative_url),
        )
        for normalizer, payload in cases:
            with (
                self.subTest(normalizer=normalizer.__name__),
                self.assertRaises(ChatNormalizationError),
            ):
                normalizer(payload, request_id="request-1")

    def test_inline_size_limit_precedes_canonical_contract_admission(self) -> None:
        payload = _anthropic_payload()
        payload["messages"][0]["content"][1]["source"]["data"] = "AAAA"

        with (
            patch.object(normalization, "MAX_INLINE_MEDIA_BASE64_CHARS", 3),
            self.assertRaisesRegex(ChatNormalizationError, "3-character limit"),
        ):
            normalize_anthropic_messages(payload, request_id="request-1")

    def test_canonical_request_rejects_non_global_ordinals(self) -> None:
        first = ChatMessage(
            role=ChatRole.USER,
            content_parts=(ChatContentPart(ordinal=0, modality="text", text="a"),),
        )
        second = ChatMessage(
            role=ChatRole.ASSISTANT,
            content_parts=(ChatContentPart(ordinal=0, modality="text", text="b"),),
        )
        with self.assertRaisesRegex(ValueError, "globally contiguous"):
            CanonicalChatRequest(
                request_id="request-1",
                model_id="fixture-model",
                messages=(first, second),
                max_new_tokens=1,
            )


if __name__ == "__main__":
    unittest.main()


class ContinuousUsageNormalizationTests(unittest.TestCase):
    def test_continuous_usage_requires_streaming_and_usage(self):
        payload = _openai_payload()
        payload["stream_options"] = {"include_usage": True, "continuous_usage_stats": True}
        normalized = normalize_openai_chat_completions(payload, request_id="request-1")
        self.assertTrue(normalized[0].stream_continuous_usage_stats)
        payload["stream_options"]["include_usage"] = False
        with self.assertRaises(ChatNormalizationError):
            normalize_openai_chat_completions(payload, request_id="request-1")
        payload["stream_options"]["include_usage"] = True
        payload["stream"] = False
        with self.assertRaises(ChatNormalizationError):
            normalize_openai_chat_completions(payload, request_id="request-1")
