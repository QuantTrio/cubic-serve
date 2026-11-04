from __future__ import annotations

import re
import unittest
from pathlib import Path
from typing import Any, ClassVar

from cubic_serve.contracts.model import WeightVariantMeta
from cubic_serve.inputs import (
    normalize_anthropic_messages,
    normalize_openai_chat_completions,
)

_MODEL_ROOT = Path("/pubdata/llm_models/Qwen/Qwen3.8-27B")
_PROCESSOR_FILES = (
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


@unittest.skipUnless(
    _HAS_TOKENIZERS and all(path.is_file() for path in _PROCESSOR_FILES),
    "the local Qwen3.8 processor files and tokenizers package are required",
)
class Qwen3_5TextProcessorTests(unittest.TestCase):
    processor: ClassVar[Any]
    weight_variant: ClassVar[WeightVariantMeta]

    @classmethod
    def setUpClass(cls) -> None:
        from cubic_serve.models.qwen3_5 import Qwen3_5TextProcessor

        cls.processor = Qwen3_5TextProcessor(_MODEL_ROOT)
        cls.weight_variant = WeightVariantMeta(
            base_model_id="qwen3_5:qwen3.8-27b:text",
            base_revision="b2s256:" + "1" * 64,
        )

    def test_exposes_model_padded_vocabulary_for_constraint_compilation(self) -> None:
        vocabulary = self.processor.encoded_vocabulary(vocab_size=248320)
        self.assertEqual(len(vocabulary), 248320)
        self.assertEqual(vocabulary[248046], "<|im_end|>")
        with self.assertRaises(ValueError):
            self.processor.encoded_vocabulary(vocab_size=248046)

    def build_hello(self, request_id: str = "hello") -> Any:
        return self.processor.build_input_plan(
            request_id=request_id,
            model_id="qwen3_5:qwen3.8-27b:text",
            weight_variant=self.weight_variant,
            messages=({"role": "user", "content": "Hello"},),
            enable_thinking=False,
        )

    def test_hello_matches_checkpoint_golden_and_realizes_text_mrope(self) -> None:
        plan = self.build_hello()

        self.assertEqual(
            self.processor.render_messages(
                ({"role": "user", "content": "Hello"},),
                enable_thinking=False,
            ),
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )
        self.assertEqual(
            plan.token_ids,
            (
                248045,
                846,
                198,
                9419,
                248046,
                198,
                248045,
                74455,
                198,
                248068,
                271,
                248069,
                271,
            ),
        )
        self.assertEqual(
            tuple(position.axes for position in plan.positions),
            tuple((index, index, index) for index in range(13)),
        )
        self.assertEqual(plan.decode_positions.first.axes, (13, 13, 13))
        self.assertEqual(plan.decode_positions.step_axes, (1, 1, 1))
        self.assertEqual(plan.placements, ())
        self.assertEqual(plan.encoder_jobs, ())

    def test_recipe_identity_is_canonical_deterministic_and_setting_bound(self) -> None:
        first = self.build_hello("first")
        second = self.build_hello("second")
        thinking = self.processor.build_input_plan(
            request_id="thinking",
            model_id=first.model_id,
            weight_variant=self.weight_variant,
            messages=({"role": "user", "content": "Hello"},),
            enable_thinking=True,
        )

        self.assertRegex(
            first.processor_recipe_digest,
            re.compile(r"^b2s256:[0-9a-f]{64}$"),
        )
        self.assertEqual(first.processor_recipe_digest, second.processor_recipe_digest)
        self.assertEqual(first.semantic_digest, second.semantic_digest)
        self.assertNotEqual(
            first.processor_recipe_digest,
            thinking.processor_recipe_digest,
        )

    def test_structured_media_tools_and_unknown_roles_fail_closed(self) -> None:
        cases: tuple[dict[str, object], ...] = (
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "file:///x"}}],
            },
            {"role": "unknown", "content": ""},
            {"role": "tool", "content": "result"},
        )
        for message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(
                    ValueError,
                    "structured or media|fields differ|unsupported role|invalid tool or reasoning",
                ),
            ):
                self.processor.build_input_plan(
                    request_id="unsupported",
                    model_id="qwen3_5:qwen3.8-27b:text",
                    weight_variant=self.weight_variant,
                    messages=(message,),
                )

    def test_both_public_protocols_reach_the_same_real_input_plan(self) -> None:
        openai, _ = normalize_openai_chat_completions(
            {
                "model": "qwen3_5:qwen3.8-27b:text",
                "max_completion_tokens": 1,
                "messages": [{"role": "user", "content": "Hello"}],
            },
            request_id="openai",
        )
        anthropic, _ = normalize_anthropic_messages(
            {
                "model": "qwen3_5:qwen3.8-27b:text",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "Hello"}],
            },
            request_id="anthropic",
        )

        openai_plan = self.processor.build_chat_input_plan(
            openai,
            weight_variant=self.weight_variant,
            enable_thinking=False,
        )
        anthropic_plan = self.processor.build_chat_input_plan(
            anthropic,
            weight_variant=self.weight_variant,
            enable_thinking=False,
        )

        self.assertEqual(openai_plan.semantic_digest, anthropic_plan.semantic_digest)
        self.assertEqual(openai_plan.token_ids, self.build_hello().token_ids)

    def test_real_template_renders_tool_round_trip_and_disabled_tool_definitions(self) -> None:
        request, _ = normalize_openai_chat_completions(
            {
                "model": self.weight_variant.base_model_id,
                "max_tokens": 32,
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "weather", "parameters": {"type": "object"}},
                    }
                ],
                "tool_choice": "none",
                "messages": [
                    {"role": "user", "content": "Weather in Beijing?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "reasoning": "Check weather",
                        "tool_calls": [
                            {
                                "id": "call-a",
                                "type": "function",
                                "function": {
                                    "name": "weather",
                                    "arguments": '{"city":"Beijing","days":2}',
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call-a", "content": "Sunny"},
                ],
            },
            request_id="round-trip",
        )
        plan = self.processor.build_chat_input_plan(request, weight_variant=self.weight_variant)
        prompt = self.processor._tokenizer.decode(list(plan.token_ids), skip_special_tokens=False)
        self.assertIn('"name": "weather"', prompt)
        self.assertIn("<function=weather>", prompt)
        self.assertIn("<parameter=city>", prompt)
        self.assertIn("Beijing", prompt)
        self.assertIn("Sunny", prompt)
        self.assertIn("Check weather", prompt)

    def test_public_template_kwargs_override_default_and_reach_checkpoint_jinja(self) -> None:
        request, _ = normalize_openai_chat_completions(
            {
                "model": "qwen3_5:qwen3.8-27b:text",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "Hello"}],
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "reasoning_effort": "low",
                    "vendor_options": {"mode": "fixture", "levels": [1, 2]},
                },
            },
            request_id="template-kwargs",
        )

        plan = self.processor.build_chat_input_plan(
            request,
            weight_variant=self.weight_variant,
            enable_thinking=False,
        )
        rendered = self.processor.render_messages(
            ({"role": "user", "content": "Hello"},),
            enable_thinking=False,
            chat_template_kwargs=request.chat_template_kwargs.as_dict(),
        )

        self.assertIn("Reasoning effort is set to low.", rendered)
        self.assertTrue(rendered.endswith("<|im_start|>assistant\n<think>\n"))
        self.assertNotEqual(plan.token_ids, self.build_hello().token_ids)
        without_vendor = self.processor.build_input_plan(
            request_id="without-vendor",
            model_id=request.model_id,
            weight_variant=self.weight_variant,
            messages=({"role": "user", "content": "Hello"},),
            chat_template_kwargs={
                "enable_thinking": True,
                "reasoning_effort": "low",
            },
        )
        self.assertEqual(plan.token_ids, without_vendor.token_ids)
        self.assertNotEqual(plan.processor_recipe_digest, without_vendor.processor_recipe_digest)

    def test_checkpoint_template_owns_reasoning_effort_validation(self) -> None:
        request, _ = normalize_openai_chat_completions(
            {
                "model": "qwen3_5:qwen3.8-27b:text",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "Hello"}],
                "reasoning_effort": "high",
            },
            request_id="invalid-effort",
        )

        with self.assertRaisesRegex(ValueError, "Unexpected reasoning effort high"):
            self.processor.build_chat_input_plan(
                request,
                weight_variant=self.weight_variant,
                enable_thinking=False,
            )

    def test_checkpoint_template_enforces_system_message_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "System message must be at the beginning"):
            self.processor.render_messages(
                (
                    {"role": "user", "content": "Hello"},
                    {"role": "system", "content": "too late"},
                )
            )

    def test_tools_enter_checkpoint_prompt_and_all_constraints_bind_recipe(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                    "strict": True,
                },
            }
        ]
        tool_request, _ = normalize_openai_chat_completions(
            {
                "model": "qwen3_5:qwen3.8-27b:text",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": tools,
                "tool_choice": "required",
            },
            request_id="tool",
        )
        json_request, _ = normalize_openai_chat_completions(
            {
                "model": "qwen3_5:qwen3.8-27b:text",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "Hello"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "integer",
                        "schema": {"type": "integer"},
                        "strict": True,
                    },
                },
            },
            request_id="json",
        )
        tool_plan = self.processor.build_chat_input_plan(
            tool_request,
            weight_variant=self.weight_variant,
            enable_thinking=False,
        )
        json_plan = self.processor.build_chat_input_plan(
            json_request,
            weight_variant=self.weight_variant,
            enable_thinking=False,
        )
        plain = self.build_hello()
        rendered = self.processor.render_messages(
            ({"role": "user", "content": "Hello"},),
            enable_thinking=False,
            structured_output=tool_request.structured_output,
        )

        self.assertIn("<tools>", rendered)
        self.assertIn('"name": "weather"', rendered)
        self.assertNotEqual(tool_plan.token_ids, plain.token_ids)
        self.assertEqual(json_plan.token_ids, plain.token_ids)
        self.assertNotEqual(json_plan.processor_recipe_digest, plain.processor_recipe_digest)
        self.assertNotEqual(tool_plan.processor_recipe_digest, json_plan.processor_recipe_digest)


if __name__ == "__main__":
    unittest.main()
