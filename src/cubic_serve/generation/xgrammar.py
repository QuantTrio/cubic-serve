"""Optional xgrammar 0.2.8 adapter for request-local token constraints.

The compiler and compiled grammars are shared.  Every matcher and bitmask is
owned by one request.  Mask application always uses xgrammar's eager
``torch_native`` backend; this module has no implicit compilation fallback.
"""

from __future__ import annotations

import importlib
import json
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType
from typing import Any, Literal, Protocol, cast

import torch

from cubic_serve.contracts.request import StructuredOutputKind, StructuredOutputMeta
from cubic_serve.generation.constraints import (
    BackendTokenMask,
    ConstraintCompilationError,
    ConstraintFactory,
    ConstraintKind,
    ConstraintMatcher,
    ConstraintReadinessError,
    ConstraintSpec,
    UnavailableConstraintFactory,
)

XGrammarVocabType = Literal["raw", "byte_fallback", "byte_level"]
ToolChoice = Literal["auto", "required"]

_SUPPORTED_VERSION = "0.2.8"
_VOCAB_TYPES = {
    "raw": "RAW",
    "byte_fallback": "BYTE_FALLBACK",
    "byte_level": "BYTE_LEVEL",
}


class _NativeMatcher(Protocol):
    def fill_next_token_bitmask(self, bitmask: torch.Tensor, index: int = 0) -> bool: ...

    def accept_token(self, token_id: int) -> bool: ...

    def is_terminated(self) -> bool: ...

    def fork(self) -> _NativeMatcher: ...

    def rollback(self, num_tokens: int) -> None: ...


class _Compiler(Protocol):
    def compile_json_schema(
        self,
        schema: str,
        *,
        any_whitespace: bool,
        strict_mode: bool,
        separators: tuple[str, str] | None = None,
    ) -> object: ...

    def compile_grammar(self, source: str, *, root_rule_name: str) -> object: ...

    def compile_regex(self, source: str) -> object: ...

    def compile_lark(self, source: str) -> object: ...

    def compile_structural_tag(self, structural_tag: object) -> object: ...


class _XGrammarRuntime:
    """Typed, deliberately tiny view of the optional dynamic module."""

    def __init__(self, module: ModuleType) -> None:
        self._module = module
        self._mask_batchers: dict[int, Any] = {}

    def _attribute(self, name: str) -> object:
        try:
            return vars(self._module)[name]
        except KeyError as error:
            raise ConstraintReadinessError(f"xgrammar has no required API {name}") from error

    def tokenizer_info(
        self,
        encoded_vocab: list[str | bytes],
        *,
        vocab_type: XGrammarVocabType,
        vocab_size: int,
        stop_token_ids: list[int],
        add_prefix_space: bool,
    ) -> object:
        vocab_enum = self._attribute("VocabType")
        enum_value = getattr(vocab_enum, _VOCAB_TYPES[vocab_type])
        constructor = cast(Callable[..., object], self._attribute("TokenizerInfo"))
        return constructor(
            encoded_vocab,
            enum_value,
            vocab_size=vocab_size,
            stop_token_ids=stop_token_ids,
            add_prefix_space=add_prefix_space,
        )

    def compiler(
        self,
        tokenizer_info: object,
        *,
        max_threads: int,
        cache_limit_bytes: int,
    ) -> _Compiler:
        constructor = cast(Callable[..., object], self._attribute("GrammarCompiler"))
        return cast(
            _Compiler,
            constructor(
                tokenizer_info,
                max_threads=max_threads,
                cache_enabled=True,
                cache_limit_bytes=cache_limit_bytes,
            ),
        )

    def matcher(self, compiled: object) -> _NativeMatcher:
        constructor = cast(Callable[..., object], self._attribute("GrammarMatcher"))
        return cast(_NativeMatcher, constructor(compiled))

    def allocate_mask(self, vocab_size: int) -> torch.Tensor:
        allocate = cast(
            Callable[[int, int], torch.Tensor],
            self._attribute("allocate_token_bitmask"),
        )
        return allocate(1, vocab_size)

    def batch_masks(self, sources, workers):
        """Fill independent private prefixes; preserve the per-matcher CPU ABI."""
        if (type(workers) is not int or workers < 2 or not sources
                or len({id(source) for source in sources}) != len(sources)
                or any(source._runtime is not self
                       or source._vocab_size != sources[0]._vocab_size for source in sources)):
            raise ValueError("grammar mask cohort differs from its backend ownership")
        batcher = self._mask_batchers.get(workers)
        if batcher is None:
            constructor = cast(Callable[..., Any], self._attribute("BatchGrammarMatcher"))
            batcher = self._mask_batchers[workers] = constructor(max_threads=workers)
        vocabulary = sources[0]._vocab_size
        packed = torch.empty((len(sources), (vocabulary + 31) // 32),
                             dtype=torch.int32, device="cpu")
        batcher.batch_fill_next_token_bitmask([source._matcher for source in sources], packed)
        # Padding bits have no token identity, including a partial final word.
        restricted = packed[:, :-1].ne(-1).any(dim=1)
        last = (1 << (vocabulary % 32)) - 1 if vocabulary % 32 else -1
        restricted.logical_or_(packed[:, -1].bitwise_and(last).ne(last))
        results = []
        for source, row, needed in zip(sources, packed, restricted.tolist(), strict=True):
            source._bitmask.copy_(row.unsqueeze(0))
            results.append(BackendTokenMask(source._bitmask, needed))
        return tuple(results)

    def apply_mask(self, logits: torch.Tensor, bitmask: torch.Tensor, vocab_size: int) -> None:
        apply = cast(Callable[..., object], self._attribute("apply_token_bitmask_inplace"))
        apply(
            logits.unsqueeze(0),
            bitmask,
            vocab_size=vocab_size,
            backend="torch_native",
        )

    def structural_tag(
        self,
        *,
        model_format: str,
        tools: list[object],
        tool_choice: ToolChoice,
        reasoning: bool,
        parallel_tool_calls: bool = True,
    ) -> object:
        create = cast(Callable[..., object], self._attribute("get_model_structural_tag"))
        return create(
            model_format,
            tools=tools,
            tool_choice=tool_choice,
            reasoning=reasoning,
            any_order=False,
            exclude_special_tokens=True,
            parallel_tool_calls=parallel_tool_calls,
        )


class _XGrammarMatcher:
    """One stateful xgrammar matcher and mask, never shared across requests."""

    def __init__(
        self,
        runtime: _XGrammarRuntime,
        compiled: object,
        *,
        vocab_size: int,
        constraint_digest: str,
        native_matcher: _NativeMatcher | None = None,
        position: int = 0,
    ) -> None:
        self._runtime = runtime
        self._compiled = compiled
        self._vocab_size = vocab_size
        self._constraint_digest = constraint_digest
        self._matcher = runtime.matcher(compiled) if native_matcher is None else native_matcher
        self._bitmask = runtime.allocate_mask(vocab_size)
        self._position = position

    @property
    def constraint_digest(self) -> str:
        return self._constraint_digest

    @property
    def batch_mask_source(self):
        return self

    @property
    def batch_mask_key(self):
        return id(self._runtime), self._vocab_size

    def fill_mask_cohort(self, sources, workers):
        return self._runtime.batch_masks(sources, workers)

    def fill_token_mask(self) -> BackendTokenMask:
        needs_apply = self._matcher.fill_next_token_bitmask(self._bitmask)
        if type(needs_apply) is not bool:
            raise RuntimeError("xgrammar returned a non-boolean mask state")
        return BackendTokenMask(self._bitmask, needs_apply)

    def apply_token_mask(self, logits: torch.Tensor, payload: object) -> None:
        if payload is not self._bitmask:
            raise ValueError("xgrammar mask belongs to another request matcher")
        if logits.numel() != self._vocab_size:
            raise ValueError(
                f"xgrammar logits have {logits.numel()} values, expected {self._vocab_size}"
            )
        bitmask = self._bitmask.to(device=logits.device)
        self._runtime.apply_mask(logits, bitmask, self._vocab_size)

    def accept_token(self, token_id: int) -> bool:
        accepted = self._matcher.accept_token(token_id)
        if accepted is True:
            self._position += 1
        return accepted

    def is_terminated(self) -> bool:
        terminated = self._matcher.is_terminated()
        if type(terminated) is not bool:
            raise RuntimeError("xgrammar returned a non-boolean termination state")
        return terminated

    def fork(self) -> ConstraintMatcher:
        return _XGrammarMatcher(
            self._runtime,
            self._compiled,
            vocab_size=self._vocab_size,
            constraint_digest=self._constraint_digest,
            native_matcher=self._matcher.fork(),
            position=self._position,
        )

    def checkpoint(self) -> object:
        return self, self._position

    def restore_checkpoint(self, checkpoint: object) -> None:
        if (type(checkpoint) is not tuple or len(checkpoint) != 2
                or checkpoint[0] is not self or type(checkpoint[1]) is not int
                or not 0 <= checkpoint[1] <= self._position):
            raise ValueError("xgrammar checkpoint differs from its private prefix")
        count = self._position - checkpoint[1]
        if count:
            self._matcher.rollback(count)
        self._position = checkpoint[1]


class XGrammarConstraintFactory:
    """Shared xgrammar compiler producing request-private matcher state."""

    def __init__(
        self,
        encoded_vocab: Sequence[str | bytes],
        *,
        vocab_type: XGrammarVocabType,
        vocab_size: int,
        stop_token_ids: Sequence[int],
        add_prefix_space: bool = False,
        tool_model_format: str | None = None,
        tool_reasoning: bool = True,
        any_whitespace: bool = True,
        max_threads: int = 8,
        cache_limit_bytes: int = 64 * 1024 * 1024,
        special_token_ids: Sequence[int] = (),
        reasoning_implicit_end_token_ids: Sequence[Sequence[int]] = (),
    ) -> None:
        vocab = _validate_tokenizer(
            encoded_vocab,
            vocab_type=vocab_type,
            vocab_size=vocab_size,
            stop_token_ids=stop_token_ids,
        )
        if type(add_prefix_space) is not bool or type(any_whitespace) is not bool:
            raise TypeError("xgrammar tokenizer and whitespace flags must be bool")
        if tool_model_format is not None and (
            type(tool_model_format) is not str
            or not tool_model_format
            or tool_model_format != tool_model_format.strip()
        ):
            raise ValueError("xgrammar tool model format must be a non-empty canonical string")
        if type(tool_reasoning) is not bool:
            raise ValueError("xgrammar tool reasoning flag must be bool")
        if type(max_threads) is not int or max_threads < 1:
            raise ValueError("xgrammar max_threads must be a positive integer")
        if type(cache_limit_bytes) is not int or cache_limit_bytes < -1:
            raise ValueError("xgrammar cache limit must be -1 or non-negative")

        self.reasoning_implicit_end_token_ids = tuple(tuple(t) for t in reasoning_implicit_end_token_ids)
        if any(not trigger or any(type(t) is not int or not 0 <= t < vocab_size for t in trigger)
               for trigger in self.reasoning_implicit_end_token_ids):
            raise ValueError("implicit reasoning boundary is outside the vocabulary")
        specials = tuple(special_token_ids)
        if any(type(t) is not int or not 0 <= t < vocab_size for t in specials):
            raise ValueError("xgrammar special token ids must be in the vocabulary")
        runtime = _load_runtime()
        try:
            tokenizer_info = runtime.tokenizer_info(
                vocab,
                vocab_type=vocab_type,
                vocab_size=vocab_size,
                stop_token_ids=list(stop_token_ids),
                add_prefix_space=add_prefix_space,
            )
            compiler = runtime.compiler(
                tokenizer_info,
                max_threads=max_threads,
                cache_limit_bytes=cache_limit_bytes,
            )
        except Exception as error:
            raise ConstraintReadinessError(
                "xgrammar tokenizer/compiler initialization failed"
            ) from error

        self._runtime = runtime
        self._full_vocab_compiler = compiler
        if specials:
            # Like vLLM's two-compiler boundary: ordinary JSON/regex/EBNF
            # excludes control tokens; structural tags and Lark may name them.
            filtered = list(vocab)
            for token in specials:
                filtered[token] = ""
            compiler = runtime.compiler(
                runtime.tokenizer_info(
                    filtered,
                    vocab_type=vocab_type,
                    vocab_size=vocab_size,
                    stop_token_ids=list(stop_token_ids),
                    add_prefix_space=add_prefix_space,
                ),
                max_threads=max_threads,
                cache_limit_bytes=cache_limit_bytes,
            )
        self._compiler = compiler
        self._vocab_size = vocab_size
        self._tool_model_format = tool_model_format
        self._tool_reasoning = tool_reasoning
        self._any_whitespace = any_whitespace
        self._cache: OrderedDict[str, object] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def backend_id(self) -> str:
        return f"xgrammar/{_SUPPORTED_VERSION}:torch_native"

    def require_ready(self, spec: ConstraintSpec) -> None:
        self._compiled(spec)

    def create_matcher(self, spec: ConstraintSpec) -> ConstraintMatcher:
        return _XGrammarMatcher(
            self._runtime,
            self._compiled(spec),
            vocab_size=self._vocab_size,
            constraint_digest=spec.digest,
        )

    def _compiled(self, spec: ConstraintSpec) -> object:
        if not isinstance(spec, ConstraintSpec):
            raise TypeError("xgrammar requires a ConstraintSpec")
        if spec.kind is ConstraintKind.IDENTITY:
            raise ConstraintReadinessError("identity constraints must not enter xgrammar")
        with self._lock:
            cached = self._cache.get(spec.digest)
            if cached is not None:
                self._cache.move_to_end(spec.digest)
                return cached
            try:
                compiled = self._compile(spec)
            except ConstraintReadinessError:
                raise
            except (ValueError, RuntimeError) as error:
                raise ConstraintCompilationError(
                    f"xgrammar cannot compile {spec.kind.value} constraint"
                ) from error
            except Exception as error:
                raise ConstraintReadinessError(
                    f"xgrammar cannot compile {spec.kind.value} constraint"
                ) from error
            self._cache[spec.digest] = compiled
            while len(self._cache) > 128:
                self._cache.popitem(last=False)
            return compiled

    def _answer_format(self, response: dict[str, str]) -> dict[str, object]:
        spec = ConstraintSpec(
            StructuredOutputMeta(
                StructuredOutputKind(response["kind"]), response["payload"], response["options"]
            )
        )
        payload = json.loads(spec.canonical_payload)
        if spec.kind is ConstraintKind.JSON_SCHEMA:
            options = json.loads(spec.meta.canonical_options)
            if "whitespace_pattern" in options:
                raise ConstraintCompilationError("xgrammar does not support whitespace_pattern")
            grammar_type = cast(Any, self._runtime._attribute("Grammar"))
            grammar = grammar_type.from_json_schema(
                spec.canonical_payload,
                any_whitespace=not options.get("disable_any_whitespace", False),
                separators=(",", ":") if options.get("disable_any_whitespace") else None,
                strict_mode=True,
            )
            return {"type": "grammar", "grammar": str(grammar)}
        if spec.kind is ConstraintKind.REGEX:
            return {"type": "regex", "pattern": payload}
        if spec.kind is ConstraintKind.CHOICE:
            return {
                "type": "or",
                "elements": [{"type": "const_string", "value": v} for v in payload],
            }
        if spec.kind is ConstraintKind.GRAMMAR:
            grammar_type = cast(Any, self._runtime._attribute("Grammar"))
            grammar = (
                grammar_type.from_lark(payload["source"])
                if payload["dialect"] == "lark"
                else grammar_type.from_ebnf(payload["source"], root_rule_name=payload["root_rule"])
            )
            return {"type": "grammar", "grammar": str(grammar)}
        if spec.kind is ConstraintKind.STRUCTURAL_TAG:
            if "structures" not in payload:
                return cast(dict[str, object], payload["format"])
            return {
                "type": "triggered_tags",
                "triggers": payload["triggers"],
                "tags": [
                    {
                        "type": "tag",
                        "begin": tag["begin"],
                        "end": tag["end"],
                        "content": {"type": "json_schema", "json_schema": tag["schema"]},
                    }
                    for tag in payload["structures"]
                ],
            }
        raise ConstraintCompilationError("unsupported answer constraint for automatic tools")

    def _compile(self, spec: ConstraintSpec) -> object:
        if spec.kind is ConstraintKind.JSON_SCHEMA:
            options = json.loads(spec.meta.canonical_options)
            if "whitespace_pattern" in options:
                # XGrammar 0.2.8 has no arbitrary-whitespace-pattern API. Do not
                # silently ignore a requested language restriction.
                raise ConstraintCompilationError("xgrammar does not support whitespace_pattern")
            any_whitespace = self._any_whitespace and not options.get(
                "disable_any_whitespace", False
            )
            return self._compiler.compile_json_schema(
                spec.canonical_payload,
                any_whitespace=any_whitespace,
                strict_mode=True,
                separators=None if any_whitespace else (",", ":"),
            )
        payload = json.loads(spec.canonical_payload)
        if spec.kind is ConstraintKind.REGEX:
            return self._compiler.compile_regex(payload)
        if spec.kind is ConstraintKind.CHOICE:
            return self._compiler.compile_grammar(
                "root ::= " + " | ".join(_ebnf_literal(value) for value in payload),
                root_rule_name="root",
            )
        if spec.kind is ConstraintKind.STRUCTURAL_TAG:
            if "structures" in payload:
                # Convert the legacy public format to the current compiler API.
                item_type = cast(
                    Callable[..., object], self._runtime._attribute("StructuralTagItem")
                )
                tag_type = self._runtime._attribute("StructuralTag")
                method_name = "from_legacy_structural_tag"
                convert = cast(Callable[..., object], getattr(tag_type, method_name))
                items = [
                    item_type(
                        begin=item["begin"], schema=json.dumps(item["schema"]), end=item["end"]
                    )
                    for item in payload["structures"]
                ]
                return self._full_vocab_compiler.compile_structural_tag(
                    convert(items, payload["triggers"])
                )
            return self._full_vocab_compiler.compile_structural_tag(spec.canonical_payload)
        if spec.kind is ConstraintKind.GRAMMAR:
            if payload["dialect"] == "lark":
                return self._full_vocab_compiler.compile_lark(payload["source"])
            if payload["dialect"] != "xgrammar-ebnf":
                raise ConstraintReadinessError(
                    f"xgrammar does not support grammar dialect {payload['dialect']!r}"
                )
            return self._compiler.compile_grammar(
                payload["source"], root_rule_name=payload["root_rule"]
            )
        if spec.kind is ConstraintKind.TOOL_SCHEMA:
            response = payload.get("response")
            if response and response["kind"] == StructuredOutputKind.STRUCTURAL_TAG.value:
                # An explicit structural tag defines the complete response,
                # including any native tool framing. Adding a generated tool
                # alternative would bypass that language's required structure.
                return self._compile(ConstraintSpec(StructuredOutputMeta(
                    StructuredOutputKind.STRUCTURAL_TAG,
                    response["payload"],
                    response["options"],
                )))
            if payload.get("format") == "json":
                from cubic_serve.generation.tool_output import tool_json_schema

                return self._compiler.compile_json_schema(
                    json.dumps(tool_json_schema(payload)),
                    any_whitespace=True,
                    strict_mode=True,
                )
            if self._tool_model_format is None:
                raise ConstraintReadinessError(
                    "tool schema requires an explicit xgrammar model format"
                )
            structured = cast(dict[str, object], payload)
            structural_tag = self._runtime.structural_tag(
                model_format=self._tool_model_format,
                tools=cast(list[object], structured["tools"]),
                tool_choice=cast(
                    ToolChoice,
                    "required"
                    if (structured["choice"] == "named" or "response" in structured)
                    else structured["choice"],
                ),
                reasoning=self._tool_reasoning,
                parallel_tool_calls=cast(bool, structured.get("parallel_tool_calls", True)),
            )
            if "response" in structured:
                tag = cast(Any, structural_tag).model_dump()
                combined = {
                    "type": "structural_tag",
                    "format": {
                        "type": "or",
                        "elements": [
                            tag["format"],
                            self._answer_format(cast(dict[str, str], structured["response"])),
                        ],
                    },
                }
                return self._full_vocab_compiler.compile_structural_tag(json.dumps(combined))
            return self._full_vocab_compiler.compile_structural_tag(structural_tag)
        raise ConstraintReadinessError(f"xgrammar does not support {spec.kind.value}")


def _ebnf_literal(value: str) -> str:
    escapes = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\r": "\\r", "\t": "\\t"}
    return (
        '"'
        + "".join(
            escapes.get(
                character,
                f"\\u{ord(character):04x}"
                if ord(character) < 32 or ord(character) == 127
                else character,
            )
            for character in value
        )
        + '"'
    )


def _validate_tokenizer(
    encoded_vocab: Sequence[str | bytes],
    *,
    vocab_type: XGrammarVocabType,
    vocab_size: int,
    stop_token_ids: Sequence[int],
) -> list[str | bytes]:
    if isinstance(encoded_vocab, str | bytes):
        raise TypeError("xgrammar encoded vocabulary must be a sequence of tokens")
    vocab = list(encoded_vocab)
    if not vocab or any(type(token) not in (str, bytes) for token in vocab):
        raise ValueError("xgrammar encoded vocabulary contains invalid tokens")
    if vocab_type not in _VOCAB_TYPES:
        raise ValueError(f"unsupported xgrammar vocabulary type {vocab_type!r}")
    if type(vocab_size) is not int or vocab_size < len(vocab):
        raise ValueError("xgrammar vocab_size must cover the encoded vocabulary")
    stops = tuple(stop_token_ids)
    if (
        not stops
        or len(set(stops)) != len(stops)
        or any(
            type(token_id) is not int or token_id < 0 or token_id >= vocab_size
            for token_id in stops
        )
    ):
        raise ValueError("xgrammar stop token ids must be unique in-range integers")
    return vocab


def _load_runtime() -> _XGrammarRuntime:
    try:
        installed = version("xgrammar")
        module = importlib.import_module("xgrammar")
    except (PackageNotFoundError, ModuleNotFoundError, ImportError) as error:
        raise ConstraintReadinessError(
            "python package xgrammar is not installed or importable"
        ) from error
    if installed != _SUPPORTED_VERSION:
        raise ConstraintReadinessError(
            f"xgrammar {installed} is incompatible; expected {_SUPPORTED_VERSION}"
        )
    return _XGrammarRuntime(module)


def load_xgrammar_constraint_factory(
    encoded_vocab: Sequence[str | bytes],
    *,
    vocab_type: XGrammarVocabType,
    vocab_size: int,
    stop_token_ids: Sequence[int],
    add_prefix_space: bool = False,
    tool_model_format: str | None = None,
    tool_reasoning: bool = True,
    any_whitespace: bool = True,
    max_threads: int = 8,
    cache_limit_bytes: int = 64 * 1024 * 1024,
    special_token_ids: Sequence[int] = (),
    reasoning_implicit_end_token_ids: Sequence[Sequence[int]] = (),
) -> ConstraintFactory:
    """Load the optional backend or return an explicit unavailable factory."""

    try:
        return XGrammarConstraintFactory(
            encoded_vocab,
            vocab_type=vocab_type,
            vocab_size=vocab_size,
            stop_token_ids=stop_token_ids,
            add_prefix_space=add_prefix_space,
            tool_model_format=tool_model_format,
            tool_reasoning=tool_reasoning,
            any_whitespace=any_whitespace,
            max_threads=max_threads,
            cache_limit_bytes=cache_limit_bytes,
            special_token_ids=special_token_ids,
            reasoning_implicit_end_token_ids=reasoning_implicit_end_token_ids,
        )
    except ConstraintReadinessError as error:
        return UnavailableConstraintFactory("xgrammar", str(error))


def frontend_constraint_factory(
    processor: object, codec: object, vocab_size: int
) -> ConstraintFactory:
    """Consume the selected model's existing frontend declaration, including HF adapters."""
    vocabulary = cast(Callable[..., Sequence[str | bytes]], cast(Any, processor).encoded_vocabulary)
    return load_xgrammar_constraint_factory(
        vocabulary(vocab_size=vocab_size),
        vocab_type=cast(XGrammarVocabType, getattr(codec, "xgrammar_vocab_type", None)),
        vocab_size=vocab_size,
        stop_token_ids=cast(Any, codec).eos_token_ids,
        tool_model_format=getattr(codec, "tool_model_format", None),
        tool_reasoning=getattr(codec, "tool_reasoning", True),
        special_token_ids=getattr(processor, "constraint_special_token_ids", ()),
        reasoning_implicit_end_token_ids=getattr(codec, "reasoning_implicit_end_token_ids", ()),
    )


__all__ = (
    "ToolChoice",
    "XGrammarConstraintFactory",
    "XGrammarVocabType",
    "load_xgrammar_constraint_factory",
)
