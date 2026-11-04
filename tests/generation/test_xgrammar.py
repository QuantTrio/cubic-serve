from __future__ import annotations

import importlib.util
import unittest
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast
from unittest.mock import patch

import torch

from cubic_serve.contracts.request import StructuredOutputMeta
from cubic_serve.generation import GreedySampler
from cubic_serve.generation.constraints import (
    ConstraintLifecycleError,
    ConstraintReadinessError,
    ConstraintSpec,
    LogitConstraint,
    UnavailableConstraintFactory,
    constraint_spec_from_meta,
    create_logit_constraint,
    replay_logit_constraint,
)
from cubic_serve.generation.xgrammar import (
    XGrammarConstraintFactory,
    load_xgrammar_constraint_factory,
)

_HAS_XGRAMMAR = importlib.util.find_spec("xgrammar") is not None
_VOCAB = (
    "<s>",
    "</s>",
    "a",
    "b",
    "c",
    "1",
    "{",
    "}",
    '"',
    ":",
    " ",
    "<tool_call>\n<function=weather>\n",
    "<parameter=city>",
    "Beijing",
    "</parameter>",
    "\n</function>\n</tool_call>",
    "<parameter=extra>",
    "value",
)
_TOOLS = [
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
_QWEN_TOKENIZER = Path("/pubdata/llm_models/Qwen/Qwen3.8-27B/tokenizer.json")


@unittest.skipUnless(_HAS_XGRAMMAR, "xgrammar optional extra is not installed")
def test_native_mask_cohorts_preserve_heterogeneous_prefixes_and_padding_words():
    for vocabulary in (8, 33):
        vocab = ('<s>', '</s>', 'a', 'b', 'c', *('x' + str(i) for i in range(vocabulary - 5)))
        factory = XGrammarConstraintFactory(vocab, vocab_type='raw', vocab_size=vocabulary,
                                           stop_token_ids=(1,), max_threads=1)
        for patterns in (('(ab|ac){4}', '(ab|ac){4}', 'b', 'b'), ('.*',) * 4):
            matchers = tuple(factory.create_matcher(ConstraintSpec(StructuredOutputMeta.regex(p)))
                             for p in patterns)
            assert matchers[1].accept_token(2)
            assert matchers[3].accept_token(3)
            before, positions = [], tuple(m._position for m in matchers)
            for matcher in matchers:
                mask = matcher.fill_token_mask()
                before.append((mask.needs_apply, mask.payload.clone()))
            after = matchers[0].fill_mask_cohort(matchers, 2)
            assert tuple(m._position for m in matchers) == positions
            for matcher, actual, (needed, expected) in zip(matchers, after, before, strict=True):
                assert actual.needs_apply is needed
                assert actual.payload is matcher._bitmask
                if needed:
                    assert torch.equal(actual.payload, expected)


def _factory(
    *,
    tool_model_format: str | None = None,
    tool_reasoning: bool = True,
) -> XGrammarConstraintFactory:
    return XGrammarConstraintFactory(
        _VOCAB,
        vocab_type="raw",
        vocab_size=len(_VOCAB),
        stop_token_ids=(1,),
        tool_model_format=tool_model_format,
        tool_reasoning=tool_reasoning,
    )


def _step(constraint: LogitConstraint, token_id: int) -> torch.Tensor:
    mask = constraint.mask()
    logits = torch.arange(len(_VOCAB), dtype=torch.float32)
    constraint.apply(logits, mask)
    constraint.accept(token_id, mask)
    return logits


@unittest.skipUnless(_HAS_XGRAMMAR, "xgrammar optional extra is not installed")
class XGrammarRealBackendTests(unittest.TestCase):
    def test_private_checkpoint_restores_native_history_eos_and_rejects_foreign_owner(self):
        matcher = _factory().create_matcher(ConstraintSpec.regex('a[bc]'))
        self.assertTrue(matcher.accept_token(2))
        checkpoint = matcher.checkpoint()
        branch = matcher.fork()
        with self.assertRaises(ValueError):
            branch.restore_checkpoint(checkpoint)
        for tail in (3, 4):
            self.assertTrue(matcher.accept_token(tail))
            self.assertTrue(matcher.accept_token(1))
            self.assertTrue(matcher.is_terminated())
            matcher.restore_checkpoint(checkpoint)
            self.assertFalse(matcher.is_terminated())
            mask = matcher.fill_token_mask().payload
            self.assertEqual({i for i in range(len(_VOCAB))
                              if int(mask[0, i//32]) & (1 << (i%32))}, {3, 4})

    def test_backend_version_and_execution_identity_are_exact(self) -> None:
        self.assertEqual(version("xgrammar"), "0.2.8")
        self.assertEqual(_factory().backend_id, "xgrammar/0.2.8:torch_native")

    def test_grammar_mask_fork_and_snapshot_replay_are_request_private(self) -> None:
        factory = _factory()
        spec = ConstraintSpec.grammar('root ::= "a" ("b" | "c")')
        parent = create_logit_constraint(spec=spec, request_id="parent", factory=factory)
        first_logits = _step(parent, 2)
        self.assertEqual(torch.isfinite(first_logits).nonzero().flatten().tolist(), [2])

        snapshot = parent.snapshot()
        branch = parent.fork("branch")
        restored = replay_logit_constraint(
            snapshot=snapshot,
            request_id="restored",
            factory=factory,
        )
        parent_logits = _step(parent, 3)
        branch_logits = _step(branch, 4)
        restored_logits = _step(restored, 4)
        self.assertEqual(torch.isfinite(parent_logits).nonzero().flatten().tolist(), [3, 4])
        self.assertTrue(torch.equal(branch_logits, restored_logits))

        for constraint in (parent, branch, restored):
            _step(constraint, 1)
            self.assertTrue(constraint.finish().matcher_terminated)
        self.assertEqual(parent.accepted_token_ids, (2, 3, 1))
        self.assertEqual(branch.accepted_token_ids, (2, 4, 1))

    def test_json_schema_executes_real_incremental_matcher(self) -> None:
        constraint = create_logit_constraint(
            spec=ConstraintSpec.json_schema({"type": "integer"}),
            request_id="json",
            factory=_factory(),
        )
        logits = _step(constraint, 5)
        self.assertEqual(torch.isfinite(logits).nonzero().flatten().tolist(), [5])
        _step(constraint, 1)
        self.assertTrue(constraint.finish().matcher_terminated)

    def test_regex_choice_and_lark_use_real_masks_and_private_state(self) -> None:
        specs = (
            ConstraintSpec.regex("a[bc]"),
            ConstraintSpec.choice(["ab", "ac"]),
            ConstraintSpec.grammar('start: "a" ("b" | "c")', dialect="lark"),
        )
        factory = _factory()
        for spec in specs:
            with self.subTest(kind=spec.kind):
                left = create_logit_constraint(spec=spec, request_id="left", factory=factory)
                right = create_logit_constraint(spec=spec, request_id="right", factory=factory)
                first = _step(left, 2)
                self.assertEqual(torch.isfinite(first).nonzero().flatten().tolist(), [2])
                _step(left, 3)
                _step(left, 1)
                _step(right, 2)
                second = _step(right, 4)
                self.assertEqual(torch.isfinite(second).nonzero().flatten().tolist(), [3, 4])
                _step(right, 1)
                self.assertTrue(left.finish().matcher_terminated)
                self.assertTrue(right.finish().matcher_terminated)

    def test_modern_and_legacy_structural_tags_execute_incrementally(self) -> None:
        modern = {
            "type": "structural_tag",
            "format": {
                "type": "tag",
                "begin": "{",
                "content": {"type": "const_string", "value": "a"},
                "end": "}",
            },
        }
        legacy = {
            "type": "structural_tag",
            "structures": [{"begin": "{", "schema": {"type": "integer"}, "end": "}"}],
            "triggers": ["{"],
        }
        for value, middle in ((modern, 2), (legacy, 5)):
            with self.subTest(value=value):
                constraint = create_logit_constraint(
                    spec=ConstraintSpec.structural_tag(value), request_id="tag", factory=_factory()
                )
                for token_id in (6, middle, 7, 1):
                    _step(constraint, token_id)
                self.assertTrue(constraint.finish().matcher_terminated)

    def test_json_filters_control_tokens_but_structural_languages_keep_them(self) -> None:
        factory = XGrammarConstraintFactory(
            ("<s>", "</s>", "a", '"', "<control>"),
            vocab_type="raw",
            vocab_size=5,
            stop_token_ids=(1,),
            special_token_ids=(0, 1, 4),
        )
        matcher = factory.create_matcher(ConstraintSpec.json_schema({"type": "string"}))
        self.assertTrue(matcher.accept_token(3))
        mask = matcher.fill_token_mask().payload
        self.assertTrue(int(mask[0, 0]) & (1 << 2))
        self.assertFalse(int(mask[0, 0]) & (1 << 4))
        tag = StructuredOutputMeta.structural_tag(
            {"type": "structural_tag", "format": {"type": "const_string", "value": "<control>"}}
        )
        tagged = factory.create_matcher(ConstraintSpec(tag))
        self.assertTrue(tagged.accept_token(4))
        self.assertTrue(tagged.accept_token(1))
        self.assertTrue(tagged.is_terminated())

    def test_choice_literals_preserve_control_characters_and_unicode(self) -> None:
        from cubic_serve.generation.xgrammar import _ebnf_literal

        self.assertEqual(
            _ebnf_literal('a"\\\n\r\t\x00\b\x7fé'), '"a\\"\\\\\\n\\r\\t\\u0000\\u0008\\u007fé"'
        )

    def test_fork_uses_native_state_without_replaying_the_prefix(self) -> None:
        from cubic_serve.generation.xgrammar import _XGrammarMatcher

        matcher = cast(_XGrammarMatcher, _factory().create_matcher(ConstraintSpec.regex("a[bc]")))
        self.assertTrue(matcher.accept_token(2))
        with patch.object(
            matcher._runtime, "matcher", side_effect=AssertionError("fork must not replay history")
        ):
            branch = matcher.fork()
        self.assertTrue(matcher.accept_token(3))
        self.assertTrue(branch.accept_token(4))
        self.assertFalse(matcher.is_terminated())
        self.assertTrue(matcher.accept_token(1))
        self.assertFalse(branch.is_terminated())
        self.assertTrue(branch.accept_token(1))
        self.assertTrue(matcher.is_terminated())
        self.assertTrue(branch.is_terminated())

    def test_json_options_bind_identity_and_unsupported_pattern_fails_closed(self) -> None:
        normal = ConstraintSpec.json_schema({"type": "integer"})
        compact = ConstraintSpec.json_schema(
            {"type": "integer"}, options={"disable_any_whitespace": True}
        )
        self.assertNotEqual(normal.digest, compact.digest)
        constraint = create_logit_constraint(spec=compact, request_id="compact", factory=_factory())
        _step(constraint, 5)
        _step(constraint, 1)
        self.assertTrue(constraint.finish().matcher_terminated)
        unsupported = ConstraintSpec.json_schema(
            {"type": "integer"}, options={"whitespace_pattern": " "}
        )
        with self.assertRaisesRegex(ConstraintReadinessError, "whitespace_pattern"):
            create_logit_constraint(spec=unsupported, request_id="unsupported", factory=_factory())

    def test_auto_tools_combine_answer_formats_without_unconstrained_plain_text(self) -> None:
        answers = (
            (StructuredOutputMeta.json_schema({"type": "integer", "enum": [1]}), 5),
            (StructuredOutputMeta.regex("a"), 2),
            (StructuredOutputMeta.choice(["a"]), 2),
            (StructuredOutputMeta.grammar('root ::= "a"'), 2),
            (StructuredOutputMeta.grammar('start: "a"', dialect="lark"), 2),
        )
        factory = _factory(tool_model_format="qwen_3_5", tool_reasoning=False)
        for answer, token in answers:
            meta = StructuredOutputMeta.tool_schema(_TOOLS, choice="auto", response=answer)
            for tokens in ((token, 1), (11, 12, 13, 14, 15, 1)):
                with self.subTest(kind=answer.kind, tokens=tokens):
                    constraint = create_logit_constraint(
                        spec=ConstraintSpec(meta), request_id="combined", factory=factory)
                    for value in tokens:
                        _step(constraint, value)
                    self.assertTrue(constraint.finish().matcher_terminated)

    def test_explicit_structural_tag_remains_authoritative_with_auto_tools(self) -> None:
        factory = _factory(tool_model_format="qwen_3_5", tool_reasoning=False)
        native = factory._runtime.structural_tag(
            model_format="qwen_3_5", tools=_TOOLS, tool_choice="required", reasoning=False,
        ).model_dump()["format"]
        answer = StructuredOutputMeta.structural_tag({
            "type": "structural_tag",
            "format": {"type": "sequence", "elements": [
                {"type": "const_string", "value": "a"}, native,
            ]},
        })
        for choice in ("auto", "required", "named"):
            meta = StructuredOutputMeta.tool_schema(_TOOLS, choice=choice, response=answer)
            matcher = factory.create_matcher(ConstraintSpec(meta))
            self.assertFalse(matcher.accept_token(11))
            self.assertTrue(matcher.accept_token(2))
            self.assertFalse(matcher.accept_token(1))
            for token in (11, 12, 13, 14, 15, 1):
                self.assertTrue(matcher.accept_token(token))
            self.assertTrue(matcher.is_terminated())

    def test_explicit_tag_needs_no_builtin_tool_grammar_and_preserves_legacy(self) -> None:
        factory = _factory()
        answers = (
            (StructuredOutputMeta.structural_tag({
                "type": "structural_tag", "format": {"type": "const_string", "value": "a"},
            }), (2, 1)),
            (StructuredOutputMeta.structural_tag({
                "type": "structural_tag",
                "structures": [{"begin": "{", "schema": {"type": "integer"}, "end": "}"}],
                "triggers": ["{"],
            }), (6, 5, 7, 1)),
        )
        for answer, tokens in answers:
            with self.subTest(kind=answer.canonical_payload):
                meta = StructuredOutputMeta.tool_schema(_TOOLS, choice="auto", response=answer)
                direct = factory.create_matcher(ConstraintSpec(answer))
                combined = factory.create_matcher(ConstraintSpec(meta))
                for token in tokens:
                    self.assertTrue(torch.equal(
                        direct.fill_token_mask().payload, combined.fill_token_mask().payload,
                    ))
                    self.assertTrue(direct.accept_token(token))
                    self.assertTrue(combined.accept_token(token))
                self.assertTrue(direct.is_terminated())
                self.assertTrue(combined.is_terminated())

    def test_structural_format_compositions_preserve_the_declared_language(self) -> None:
        a = {"type": "const_string", "value": "a"}
        b = {"type": "const_string", "value": "b"}
        tag = {"type": "tag", "begin": "{", "content": a, "end": "}"}
        token_tag = {
            "type": "tag", "begin": {"type": "token", "token": 6},
            "content": a, "end": {"type": "token", "token": 7},
        }
        # Expected accepted/rejected token paths are independent of the compiler.
        cases = (
            (a, (2,), (3,)),
            ({"type": "sequence", "elements": [a, b]}, (2, 3), (3,)),
            ({"type": "or", "elements": [a, b]}, (3,), (4,)),
            ({"type": "optional", "content": a}, (), (3,)),
            ({"type": "plus", "content": a}, (2, 2), (3,)),
            ({"type": "star", "content": a}, (), (3,)),
            ({"type": "repeat", "min": 2, "max": 3, "content": a}, (2, 2), (2, 1)),
            ({"type": "regex", "pattern": "[ab]c"}, (3, 4), (4,)),
            ({"type": "grammar", "grammar": 'root ::= "ab"'}, (2, 3), (3,)),
            ({"type": "json_schema", "json_schema": {"type": "integer", "enum": [1]}},
             (5,), (2,)),
            ({"type": "qwen_xml_parameter", "json_schema": {
                "type": "object", "properties": {"city": {"type": "string"}},
                "required": ["city"], "additionalProperties": False,
            }}, (12, 13, 14), (16,)),
            ({"type": "json_schema", "style": "qwen_xml", "json_schema": {
                "type": "object", "properties": {"city": {"type": "string"}},
                "required": ["city"], "additionalProperties": False,
            }}, (12, 13, 14), (16,)),
            ({"type": "token", "token": 2}, (2,), (3,)),
            ({"type": "exclude_token", "exclude_tokens": [2]}, (3,), (2,)),
            ({"type": "any_tokens", "exclude_tokens": [4], "max_tokens": 2},
             (2, 3), (4,)),
            ({"type": "any_text", "excludes": ["c"]}, (2, 3), (4,)),
            (tag, (6, 2, 7), (6, 3)),
            (token_tag, (6, 2, 7), (6, 3)),
            ({"type": "tags_with_separator", "tags": [tag], "separator": " ",
              "at_least_one": True}, (6, 2, 7, 10, 6, 2, 7), (1,)),
            ({"type": "triggered_tags", "tags": [tag], "triggers": ["{"],
              "at_least_one": True, "stop_after_first": True},
             (6, 2, 7), (6, 3)),
            ({"type": "triggered_tags", "tags": [tag], "triggers": ["{"],
              "stop_after_first": True}, (3, 6, 2, 7), (6, 3)),
            ({"type": "token_triggered_tags", "tags": [token_tag], "trigger_tokens": [6],
              "at_least_one": True, "stop_after_first": True}, (6, 2, 7), (6, 3)),
            ({"type": "token_triggered_tags", "tags": [token_tag], "trigger_tokens": [6],
              "stop_after_first": True}, (3, 6, 2, 7), (6, 3)),
            ({"type": "dispatch", "rules": [["{", a]], "loop": False}, (6, 2), (6, 3)),
            ({"type": "token_dispatch", "rules": [[6, a]], "loop": False}, (6, 2), (6, 3)),
        )
        factory = _factory()
        for format_, accepted, rejected in cases:
            with self.subTest(format=format_["type"]):
                spec = ConstraintSpec.structural_tag({"type": "structural_tag", "format": format_})
                valid = factory.create_matcher(spec)
                for token in (*accepted, 1):
                    mask = valid.fill_token_mask().payload
                    self.assertTrue(
                        int(mask[0, token // 32]) & (1 << (token % 32)),
                        (format_["type"], token, accepted),
                    )
                    self.assertTrue(valid.accept_token(token))
                self.assertTrue(valid.is_terminated())
                invalid = factory.create_matcher(spec)
                self.assertFalse(all(invalid.accept_token(token) for token in rejected))

    def test_native_tool_string_control_text_is_accepted_and_parsed_literally(self) -> None:
        import json
        import string
        from cubic_serve.contracts.request import TokenEventKind, ToolDefinitions
        from cubic_serve.models.qwen3_5.output import Qwen3_5OutputParser
        from cubic_serve.generation.output import TokenPhaseOutputParser
        from cubic_serve.generation.structured_sampling import (
            ConstraintBinding, GrammarPool, DelayedConstraintMatcher,
        )

        value = '<think></think><tool_call></tool_call>'
        vocab = ['<unused>', '<eos>', *string.printable]
        trigger = tuple(vocab.index(c) for c in '<tool_call>')
        end = (vocab.index('~'),)
        tools = [{'type':'function','function':{'name':'f','strict':True,'parameters':{'type':'object','properties':{'s':{'type':'string','enum':[value]}},'required':['s'],'additionalProperties':False}}}]
        factory = XGrammarConstraintFactory(vocab, vocab_type='raw', vocab_size=len(vocab),
            stop_token_ids=(1,), tool_model_format='qwen_3_5', tool_reasoning=False,
            reasoning_implicit_end_token_ids=(trigger,))
        meta = StructuredOutputMeta.tool_schema(tools, choice='required', parallel_tool_calls=False)
        matcher = DelayedConstraintMatcher(factory.create_matcher(ConstraintSpec(meta)), end, (1,), (trigger,))
        pool = GrammarPool(factory, vocab_size=len(vocab), stop_token_ids=(1,))
        round_ = pool.begin((ConstraintBinding('r', meta, 0, 1024, end),))
        parser = Qwen3_5OutputParser(enable_thinking=False, reasoning_parser=False, tool_call_parser=True)
        parser.bind_tools(ToolDefinitions.from_value(tools))
        parser = TokenPhaseOutputParser(parser, reasoning_end=end, reasoning_implicit_end=(trigger,))
        text = 'reason' + '<tool_call>\n<function=f>\n<parameter=s>\n' + value + '\n</parameter>\n</function>\n</tool_call>'
        events = []
        tokens = []
        for depth, character in enumerate(text):
            token = vocab.index(character)
            reference_mask = matcher.fill_token_mask()
            worker_mask = round_.mask('r', depth)
            if reference_mask.needs_apply:
                self.assertTrue(torch.equal(reference_mask.payload[0], worker_mask))
            else:
                self.assertIsNone(worker_mask)
            self.assertTrue(matcher.accept_token(token))
            round_.propose('r', token, depth)
            tokens.append(token)
            events.extend(parser.push(token, character))
        round_.mask('r', len(tokens))
        round_.propose('r', 1, len(tokens))
        self.assertTrue(matcher.accept_token(1))
        round_.commit({'r': (*tokens, 1)})
        self.assertNotIn('r', pool.sessions)
        events.extend(parser.finish(''))
        self.assertEqual(json.loads(''.join(e.text for e in events if e.kind is TokenEventKind.TOOL_CALL_DELTA)), {'s':value})

    def test_qwen_3_5_tool_schema_compiles_and_accepts_xml_call(self) -> None:
        factory = _factory(
            tool_model_format="qwen_3_5",
            tool_reasoning=False,
        )
        constraint = create_logit_constraint(
            spec=ConstraintSpec.tool_schema(_TOOLS, choice="required"),
            request_id="tool",
            factory=factory,
        )
        for token_id in (11, 12, 13, 14, 15, 1):
            _step(constraint, token_id)
        snapshot = constraint.finish()
        self.assertTrue(snapshot.matcher_terminated)
        self.assertEqual(snapshot.accepted_token_ids, (11, 12, 13, 14, 15, 1))

    def test_non_strict_tool_keeps_framing_without_enforcing_argument_schema(self) -> None:
        function = dict(cast(dict[str, object], _TOOLS[0]["function"]))
        function["strict"] = False
        tools = [{"type": "function", "function": function}]
        constraint = create_logit_constraint(
            spec=ConstraintSpec.tool_schema(tools, choice="required"),
            request_id="non-strict-tool",
            factory=_factory(tool_model_format="qwen_3_5", tool_reasoning=False),
        )
        for token_id in (11, 16, 17, 14, 15, 1):
            _step(constraint, token_id)
        self.assertTrue(constraint.finish().matcher_terminated)

    def test_mask_application_is_pinned_to_torch_native_without_compile(self) -> None:
        import xgrammar

        factory = _factory()
        constraint = create_logit_constraint(
            spec=ConstraintSpec.grammar('root ::= "a"'),
            request_id="native",
            factory=factory,
        )
        original = xgrammar.apply_token_bitmask_inplace
        backends: list[str | None] = []

        def observe(*args: object, **kwargs: object) -> None:
            backend = kwargs.get("backend")
            backends.append(backend if isinstance(backend, str) else None)
            original(*args, **kwargs)  # type: ignore[arg-type]

        with (
            patch.object(xgrammar, "apply_token_bitmask_inplace", observe),
            patch.object(torch, "compile", side_effect=AssertionError("compile is forbidden")),
        ):
            _step(constraint, 2)
        self.assertEqual(backends, ["torch_native"])

    def test_unsupported_semantics_fail_readiness_without_fallback(self) -> None:
        with self.assertRaisesRegex(ConstraintReadinessError, "explicit xgrammar model format"):
            create_logit_constraint(
                spec=ConstraintSpec.tool_schema(_TOOLS),
                request_id="missing-tool-format",
                factory=_factory(),
            )
        with self.assertRaisesRegex(ConstraintReadinessError, "does not support grammar dialect"):
            create_logit_constraint(
                spec=ConstraintSpec.grammar('root ::= "a"', dialect="gbnf"),
                request_id="wrong-dialect",
                factory=_factory(),
            )
        with self.assertRaises(ConstraintReadinessError):
            create_logit_constraint(
                spec=ConstraintSpec.tool_schema(_TOOLS),
                request_id="unknown-format",
                factory=_factory(tool_model_format="invented"),
            )

    def test_wrong_logit_vocabulary_fails_constraint_session(self) -> None:
        constraint = create_logit_constraint(
            spec=ConstraintSpec.grammar('root ::= "a"'),
            request_id="wrong-logits",
            factory=_factory(),
        )
        mask = constraint.mask()
        with self.assertRaisesRegex(ConstraintLifecycleError, "backend failed to apply mask"):
            constraint.apply(torch.zeros(len(_VOCAB) - 1), mask)
        self.assertTrue(constraint.failed)


class XGrammarOptionalLoadingTests(unittest.TestCase):
    def test_missing_package_returns_explicit_unavailable_factory(self) -> None:
        with patch(
            "cubic_serve.generation.xgrammar.version",
            side_effect=PackageNotFoundError("xgrammar"),
        ):
            factory = load_xgrammar_constraint_factory(
                _VOCAB,
                vocab_type="raw",
                vocab_size=len(_VOCAB),
                stop_token_ids=(1,),
            )
        self.assertIsInstance(factory, UnavailableConstraintFactory)
        with self.assertRaisesRegex(ConstraintReadinessError, "not downgraded to identity"):
            create_logit_constraint(
                spec=ConstraintSpec.json_schema({"type": "integer"}),
                request_id="missing",
                factory=factory,
            )


@unittest.skipUnless(
    _HAS_XGRAMMAR and _QWEN_TOKENIZER.is_file(),
    "real Qwen3.8 tokenizer and xgrammar are required",
)
class QwenTokenizerXGrammarTests(unittest.TestCase):
    def test_public_tool_meta_drives_real_qwen_matcher_and_sampler(self) -> None:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(_QWEN_TOKENIZER))
        vocab = tokenizer.get_vocab(with_added_tokens=True)
        encoded_vocab = [""] * (max(vocab.values()) + 1)
        for token, token_id in vocab.items():
            encoded_vocab[token_id] = token
        factory = XGrammarConstraintFactory(
            encoded_vocab,
            vocab_type="byte_level",
            vocab_size=248320,
            stop_token_ids=(248046,),
            tool_model_format="qwen_3_5",
            tool_reasoning=False,
            max_threads=2,
        )
        meta = StructuredOutputMeta.tool_schema(_TOOLS, choice="required")
        constraint = create_logit_constraint(
            spec=constraint_spec_from_meta(meta),
            request_id="real-qwen-tool",
            factory=factory,
        )
        output = (
            "<tool_call>\n<function=weather>\n<parameter=city>\nBeijing\n"
            "</parameter>\n</function>\n</tool_call>"
        )
        token_ids = [*tokenizer.encode(output, add_special_tokens=False).ids, 248046]
        sampler = GreedySampler()
        for expected in token_ids:
            logits = torch.full((248320,), -10.0)
            logits[expected] = 10.0
            self.assertEqual(sampler.sample(logits, constraint), expected)
        self.assertTrue(constraint.finish().matcher_terminated)


if __name__ == "__main__":
    unittest.main()
