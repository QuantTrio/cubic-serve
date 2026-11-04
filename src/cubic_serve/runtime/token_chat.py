"""Shared public-chat adapter over an explicitly selected token generator.

The adapter deliberately owns no scheduler or model fallback.  A generator
produces sampled token ids, while the injected processor and output codec remain
the only chat-template, detokenization, reasoning, tool-parser, and wire
event authorities.  This mirrors the minimal generation loop in llama.cpp
``458681e1`` and the token-only PP return boundary in vLLM ``89af4f1`` and
SGLang ``ba433bb``.
"""

from __future__ import annotations

from typing_extensions import TypeAliasType

import asyncio
import inspect
import logging
import threading
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from typing import Protocol, cast

from cubic_serve.api.server import BackendErrorKind, ChatBackendError, ChatExecution
from cubic_serve.contracts.model import WeightVariantMeta
from cubic_serve.contracts.request import (
    CanonicalChatRequest,
    ReasoningControl,
    SamplingMeta,
    StructuredOutputKind,
    StructuredOutputMeta,
    TokenEvent,
    TokenEventKind,
)
from cubic_serve.generation import ChatOutputCodec, GenerationOutputPipeline
from cubic_serve.generation.constraints import (
    ConstraintCompilationError,
    ConstraintFactory,
    ConstraintReadinessError,
    ConstraintSpec,
)
from cubic_serve.inputs.chat import (
    ChatInputProcessor,
    ChatProcessorError,
    resolve_chat_template_kwargs,
)
from cubic_serve.inputs.multimodal import MultimodalInputPolicy, MultimodalPreprocessingGate
from cubic_serve.models import PreparedInput
from cubic_serve.runtime.token_stream import TokenGenerationOverloaded

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TokenGenerationMetadata:
    """Request identity and sampling semantics bound to one generator call."""

    request_id: str
    structured_output: StructuredOutputMeta
    sampling: SamplingMeta
    prepared_input: PreparedInput | None = None
    ignore_eos: bool = False
    constraint_reasoning_end: tuple[int, ...] = ()
    reasoning_control: ReasoningControl | None = None

    def __post_init__(self) -> None:
        if (
            type(self.request_id) is not str
            or not self.request_id
            or self.request_id.strip() != self.request_id
        ):
            raise ValueError("token generation requires a canonical request id")
        if not isinstance(self.structured_output, StructuredOutputMeta):
            raise TypeError("token generation requires structured-output metadata")
        if not isinstance(self.sampling, SamplingMeta):
            raise TypeError("token generation requires sampling metadata")
        if self.prepared_input is not None and not isinstance(self.prepared_input, PreparedInput):
            raise TypeError("token prepared input has an invalid type")
        if type(self.ignore_eos) is not bool:
            raise TypeError("token ignore_eos must be a boolean")
        if type(self.constraint_reasoning_end) is not tuple or any(
            type(t) is not int or t < 0 for t in self.constraint_reasoning_end
        ):
            raise TypeError("constraint reasoning boundary must contain token ids")
        if self.reasoning_control is not None and not isinstance(
            self.reasoning_control, ReasoningControl
        ):
            raise TypeError("invalid token reasoning control")


TokenSource = TypeAliasType("TokenSource", Sequence[int] | Iterator[int] | AsyncIterator[int])
TokenGenerateResult = TypeAliasType("TokenGenerateResult", TokenSource | Awaitable[TokenSource])
TokenCancelResult = TypeAliasType("TokenCancelResult", Awaitable[None] | None)


class TokenChatExecutor(Protocol):
    """Qualified boundary whose generate/cancel admission calls never block on tokens."""

    @property
    def max_sequence_tokens(self) -> int: ...

    @property
    def ready(self) -> bool: ...

    def generate(
        self,
        input_ids: tuple[int, ...],
        max_tokens: int,
        metadata: TokenGenerationMetadata,
    ) -> TokenGenerateResult: ...

    def cancel(self, request_id: str) -> TokenCancelResult: ...


@dataclass(slots=True)
class _ActiveGeneration:
    input_ids: tuple[int, ...]
    max_tokens: int
    metadata: TokenGenerationMetadata
    pipeline: GenerationOutputPipeline
    stopped: asyncio.Event = field(default_factory=asyncio.Event)
    source: TokenSource | None = None
    events_started: bool = False


class TokenChatBackend:
    """Adapt an explicitly selected token executor to the canonical chat event ABI.

    Numerical and execution-mode qualification belong to the executor. This
    transport neither grants graph qualification nor selects a fallback.
    """

    def __init__(
        self,
        executor: TokenChatExecutor,
        *,
        model_id: str,
        weight_variant: WeightVariantMeta,
        processor: ChatInputProcessor,
        output_codec: ChatOutputCodec,
        enable_thinking: bool = True,
        reasoning_parser: str | bool = False,
        tool_call_parser: str | bool = False,
        multimodal_policy: MultimodalInputPolicy | None = None,
        structured_output_in_reasoning: bool = False,
        structured_output_disable_whitespace: bool = False,
    ) -> None:
        if not callable(getattr(executor, "generate", None)) or not callable(
            getattr(executor, "cancel", None)
        ):
            raise TypeError("performance chat executor requires generate and cancel methods")
        max_sequence_tokens = getattr(executor, "max_sequence_tokens", None)
        if type(max_sequence_tokens) is not int or max_sequence_tokens <= 0:
            raise ValueError(
                "performance chat executor requires a positive max_sequence_tokens capacity"
            )
        if type(getattr(executor, "ready", None)) is not bool:
            raise ValueError("performance chat executor requires a boolean ready state")
        if type(model_id) is not str or not model_id or model_id.strip() != model_id:
            raise ValueError("performance chat requires a canonical model id")
        if not isinstance(weight_variant, WeightVariantMeta):
            raise TypeError("performance chat requires WeightVariantMeta")
        if weight_variant.base_model_id != model_id:
            raise ValueError("performance chat model and weight identities differ")
        if any(type(setting) is not bool for setting in (
            enable_thinking, structured_output_in_reasoning, structured_output_disable_whitespace,
        )):
            raise TypeError("performance chat boolean settings must be bool values")
        if any(type(value) not in (str, bool) for value in (reasoning_parser, tool_call_parser)):
            raise TypeError("output parser selections require names or booleans")
        eos_token_id = output_codec.eos_token_id
        if type(eos_token_id) is not int or eos_token_id < 0:
            raise ValueError("output codec requires one non-negative EOS token id")
        eos_token_ids = getattr(output_codec, "eos_token_ids", (eos_token_id,))
        if (
            type(eos_token_ids) is not tuple
            or any(type(token) is not int or token < 0 for token in eos_token_ids)
            or tuple(sorted(set(eos_token_ids))) != eos_token_ids
            or eos_token_id not in eos_token_ids
        ):
            raise ValueError("output codec requires canonical EOS identities including its primary")
        if reasoning_parser or tool_call_parser:
            output_codec.admit_parser_recipe(
                reasoning_parser=reasoning_parser,
                tool_call_parser=tool_call_parser,
            )

        self._executor = executor
        self._model_id = model_id
        self._weight_variant = weight_variant
        self._processor = processor
        self._output_codec = output_codec
        self._enable_thinking = enable_thinking
        self._reasoning_parser = reasoning_parser
        self._tool_call_parser = tool_call_parser
        self._structured_output_in_reasoning = structured_output_in_reasoning
        self._structured_output_disable_whitespace = structured_output_disable_whitespace
        self._max_sequence_tokens = max_sequence_tokens
        self._eos_token_ids = eos_token_ids
        self._active: dict[str, _ActiveGeneration] = {}
        self._known_request_ids: set[str] = set()
        self._constraint_factory: ConstraintFactory | None = None
        self._constraint_lock = threading.Lock()
        self._media_preprocessing = MultimodalPreprocessingGate(
            (multimodal_policy or MultimodalInputPolicy()).preprocess_workers
        )

    @property
    def ready(self) -> bool:
        return self._executor.ready

    async def start(self, request: CanonicalChatRequest) -> ChatExecution:
        if not isinstance(request, CanonicalChatRequest):
            raise TypeError("performance chat requires a CanonicalChatRequest")
        if self._structured_output_disable_whitespace:
            from cubic_serve.generation.structured_sampling import compact_json_constraint

            request = replace(
                request, structured_output=compact_json_constraint(request.structured_output)
            )
        if request.model_id != self._model_id:
            raise ChatBackendError(
                BackendErrorKind.MODEL_NOT_FOUND,
                f"model {request.model_id!r} is not served by this executor",
            )
        if request.request_id in self._known_request_ids:
            raise ChatBackendError(
                BackendErrorKind.INVALID_REQUEST,
                "request id was already admitted",
            )
        try:
            _, enable_thinking = resolve_chat_template_kwargs(
                request,
                default_enable_thinking=self._enable_thinking,
            )
        except ChatProcessorError as error:
            raise ChatBackendError(
                BackendErrorKind.INVALID_REQUEST,
                str(error),
                public_message=error.public_message,
            ) from error
        if request.structured_output.kind is StructuredOutputKind.TOOL_SCHEMA:
            from cubic_serve.generation.tool_output import prepare_forced_tool_request

            try:
                request = prepare_forced_tool_request(request, native_parser=self._tool_call_parser)
            except ValueError as error:
                raise ChatBackendError(BackendErrorKind.INVALID_REQUEST, str(error)) from error

        from cubic_serve.generation.tool_output import tool_sampling_constraint

        sampling_constraint = tool_sampling_constraint(request)
        if (
            (sampling_constraint.kind is not StructuredOutputKind.IDENTITY
             or request.thinking_token_budget is not None)
            and getattr(self._executor, "supports_structured_output", False) is not True
        ):
            raise ChatBackendError(
                BackendErrorKind.INVALID_REQUEST,
                "performance chat currently supports only unconstrained generation",
            )
        if request.thinking_token_budget is not None and not self._reasoning_parser:
            raise ChatBackendError(
                BackendErrorKind.INVALID_REQUEST, "thinking budget requires a reasoning parser",
                public_message="thinking_token_budget is unavailable for this model",
            )
        if sampling_constraint.kind is not StructuredOutputKind.IDENTITY:
            self._known_request_ids.add(request.request_id)
            try:
                await asyncio.to_thread(
                    self._admit_constraint, ConstraintSpec(sampling_constraint)
                )
            except ConstraintCompilationError as error:
                self._known_request_ids.discard(request.request_id)
                raise ChatBackendError(BackendErrorKind.INVALID_REQUEST, str(error)) from error
            except ConstraintReadinessError as error:
                self._known_request_ids.discard(request.request_id)
                raise ChatBackendError(BackendErrorKind.UNAVAILABLE, str(error)) from error
            except BaseException:
                self._known_request_ids.discard(request.request_id)
                raise

        self._known_request_ids.add(request.request_id)
        try:
            builder = getattr(self._processor, "build_chat_prepared_input", None)
            if builder is None:
                builder = self._processor.build_chat_input_plan
            process = (
                self._media_preprocessing.run
                if any(
                    part.modality != "text"
                    for message in request.messages
                    for part in message.content_parts
                )
                else asyncio.to_thread
            )
            realized = await process(
                builder,
                request,
                weight_variant=self._weight_variant,
                enable_thinking=enable_thinking,
            )
            prepared_input = (
                realized if isinstance(realized, PreparedInput) else PreparedInput(realized)
            )
            input_plan = prepared_input.input_plan
        except asyncio.CancelledError:
            self._known_request_ids.discard(request.request_id)
            raise
        except ChatProcessorError as error:
            self._known_request_ids.discard(request.request_id)
            raise ChatBackendError(
                BackendErrorKind.INVALID_REQUEST,
                str(error),
                public_message=error.public_message,
            ) from error
        except Exception as error:
            self._known_request_ids.discard(request.request_id)
            raise ChatBackendError(
                BackendErrorKind.UNAVAILABLE,
                "performance chat input processing failed",
            ) from error

        max_new_tokens = request.max_new_tokens
        if max_new_tokens is None:
            max_new_tokens = self._max_sequence_tokens - len(input_plan.token_ids)
        required_sequence_tokens = len(input_plan.token_ids) + max_new_tokens - 1
        if max_new_tokens <= 0 or required_sequence_tokens > self._max_sequence_tokens:
            self._known_request_ids.discard(request.request_id)
            raise ChatBackendError(
                BackendErrorKind.INVALID_REQUEST,
                "prompt plus completion exceeds qualified performance context capacity: "
                f"{required_sequence_tokens} > {self._max_sequence_tokens}",
                public_message="request exceeds available context length",
            )

        from cubic_serve.generation.structured_sampling import reasoning_control_from_prompt

        try:
            control = reasoning_control_from_prompt(
                request, self._output_codec, input_plan.token_ids
            )
            if (control is not None
                    and sampling_constraint.kind is not StructuredOutputKind.IDENTITY):
                await asyncio.to_thread(
                    self._admit_reasoning_control, request, control, enable_thinking
                )
        except ConstraintCompilationError as error:
            self._known_request_ids.discard(request.request_id)
            raise ChatBackendError(
                BackendErrorKind.INVALID_REQUEST, str(error),
                public_message="thinking_token_budget conflicts with structured_outputs",
            ) from error
        except ValueError as error:
            self._known_request_ids.discard(request.request_id)
            raise ChatBackendError(
                BackendErrorKind.INVALID_REQUEST, str(error),
                public_message="thinking_token_budget is unavailable for this model",
            ) from error
        except BaseException:
            self._known_request_ids.discard(request.request_id)
            raise
        metadata = TokenGenerationMetadata(
            request_id=request.request_id,
            structured_output=sampling_constraint,
            sampling=request.sampling,
            prepared_input=prepared_input,
            ignore_eos=request.ignore_eos,
            constraint_reasoning_end=(
                tuple(getattr(self._output_codec, "reasoning_end_token_ids", ()))
                if enable_thinking and self._reasoning_parser
                and not self._structured_output_in_reasoning
                else ()
            ),
            reasoning_control=control,
        )
        # Grammar completion is a separate stop condition. A terminated
        # matcher must never turn into unconstrained generation after EOS.
        eos_token_ids = (
            ()
            if request.ignore_eos
            and sampling_constraint.kind is StructuredOutputKind.IDENTITY
            else self._eos_token_ids
        )
        from cubic_serve.generation.output import request_output_parser

        parser = request_output_parser(
            self._output_codec,
            request,
            enable_thinking=enable_thinking,
            reasoning_parser=self._reasoning_parser,
            tool_call_parser=self._tool_call_parser,
            reasoning_control=control,
        )

        active = _ActiveGeneration(
            input_ids=input_plan.token_ids,
            max_tokens=max_new_tokens,
            metadata=metadata,
            pipeline=GenerationOutputPipeline(
                request_id=request.request_id,
                input_token_count=len(input_plan.token_ids),
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_token_ids,
                decoder=self._output_codec.new_decoder(),
                parser=parser,
                continuous_usage=request.stream_continuous_usage_stats,
                report_reasoning_tokens=bool(self._reasoning_parser),
                include_reasoning=request.include_reasoning,
            ),
        )
        self._active[request.request_id] = active
        try:
            active.source = await self._open_source(active)
            if active.stopped.is_set():
                raise asyncio.CancelledError
        except BaseException as error:
            await self._abort_quietly(active)
            await self._close_source(active.source)
            self._known_request_ids.discard(request.request_id)
            if isinstance(error, TokenGenerationOverloaded):
                raise ChatBackendError(BackendErrorKind.OVERLOADED, str(error)) from error
            if isinstance(error, Exception) and not isinstance(error, ChatBackendError):
                _LOGGER.exception("token admission failed request_id=%s", request.request_id)
                raise ChatBackendError(
                    BackendErrorKind.UNAVAILABLE, "token admission failed",
                ) from error
            raise
        return ChatExecution(len(input_plan.token_ids), self._events(active))

    async def cancel(self, request_id: str) -> None:
        active = self._active.get(request_id)
        if active is not None:
            await self._abort(active)

    def _admit_constraint(self, spec: ConstraintSpec) -> None:
        from cubic_serve.generation.xgrammar import frontend_constraint_factory

        with self._constraint_lock:
            if self._constraint_factory is None:
                self._constraint_factory = frontend_constraint_factory(
                    self._processor, self._output_codec, getattr(self._executor, "vocab_size", 0)
                )
        self._constraint_factory.require_ready(spec)

    def _admit_reasoning_control(self, request, control, enable_thinking) -> None:
        from cubic_serve.generation.constraints import create_logit_constraint
        from cubic_serve.generation.structured_sampling import (
            DelayedConstraintFactory,
            implicit_reasoning_ends,
        )

        factory = self._constraint_factory
        if (enable_thinking and self._reasoning_parser and not self._structured_output_in_reasoning
                and factory is not None):
            factory = DelayedConstraintFactory(
                factory, tuple(getattr(self._output_codec, "reasoning_end_token_ids", ())),
                self._eos_token_ids,
                implicit_reasoning_ends(request.structured_output, self._output_codec),
            )
        create_logit_constraint(
            spec=ConstraintSpec(request.structured_output), request_id=request.request_id,
            factory=factory, reasoning_control=control,
        )

    async def _events(self, active: _ActiveGeneration) -> AsyncIterator[TokenEvent]:
        if active.stopped.is_set():
            return
        active.events_started = True
        raw_sequence = 0
        try:
            source = active.source
            if source is None:
                raise RuntimeError("token source was not admitted")
            async for token_id in self._iterate(source):
                if active.stopped.is_set():
                    return
                if type(token_id) is not int or token_id < 0:
                    raise TypeError("performance generator emitted an invalid token id")
                raw = TokenEvent(
                    request_id=active.metadata.request_id,
                    sequence_no=raw_sequence,
                    kind=TokenEventKind.TOKEN,
                    token_id=token_id,
                )
                raw_sequence += 1
                for event in active.pipeline.push(raw):
                    yield event
                if (
                    token_id in (() if active.metadata.ignore_eos else self._eos_token_ids)
                    or active.pipeline.output_token_count >= active.max_tokens
                ):
                    break

            if active.stopped.is_set():
                return
            finish = TokenEvent(
                request_id=active.metadata.request_id,
                sequence_no=raw_sequence,
                kind=TokenEventKind.FINISH,
            )
            for event in active.pipeline.push(finish):
                yield event
            if not active.pipeline.finished:
                raise RuntimeError("performance output pipeline did not finish")
        except (asyncio.CancelledError, GeneratorExit):
            await self._abort(active)
            raise
        except ChatBackendError:
            await self._abort_quietly(active)
            raise
        except TokenGenerationOverloaded as error:
            await self._abort_quietly(active)
            raise ChatBackendError(BackendErrorKind.OVERLOADED, str(error)) from error
        except Exception as error:
            _LOGGER.exception("token stream failed request_id=%s", active.metadata.request_id)
            await self._abort_quietly(active)
            raise ChatBackendError(
                BackendErrorKind.UNAVAILABLE,
                "token generation failed",
            ) from error
        finally:
            await self._close_source(active.source)
            if self._active.get(active.metadata.request_id) is active:
                del self._active[active.metadata.request_id]

    async def _open_source(self, active: _ActiveGeneration) -> TokenSource:
        result: object = self._executor.generate(
            active.input_ids,
            active.max_tokens,
            active.metadata,
        )
        if inspect.isawaitable(result):
            result = await cast(Awaitable[object], result)
        if isinstance(result, str | bytes | bytearray):
            raise TypeError("performance generate result cannot be text or bytes")
        if hasattr(result, "__aiter__"):
            iterator = cast(AsyncIterable[int], result).__aiter__()
            if not hasattr(iterator, "__anext__"):
                raise TypeError("performance async token source has no __anext__")
            return iterator
        if isinstance(result, Sequence):
            return result
        if isinstance(result, Iterator):
            raise TypeError(
                "performance sync token iterators must also expose an async iterator view"
            )
        raise TypeError(
            "performance generate must return a token sequence, iterator, or async iterator"
        )

    @staticmethod
    async def _iterate(source: TokenSource) -> AsyncIterator[int]:
        if hasattr(source, "__anext__"):
            async for token_id in cast(AsyncIterator[int], source):
                yield token_id
            return
        if isinstance(source, Sequence):
            for token_id in source:
                yield token_id
            return
        raise TypeError("performance token source is neither async nor an in-memory sequence")

    async def _abort(self, active: _ActiveGeneration) -> None:
        request_id = active.metadata.request_id
        if self._active.get(request_id) is not active:
            return
        del self._active[request_id]
        active.stopped.set()
        try:
            result: object = self._executor.cancel(request_id)
            if inspect.isawaitable(result):
                await cast(Awaitable[object], result)
        finally:
            if not active.events_started:
                source, active.source = active.source, None
                await self._close_source(source)

    async def _abort_quietly(self, active: _ActiveGeneration) -> None:
        with suppress(Exception):
            await self._abort(active)

    @staticmethod
    async def _close_source(source: TokenSource | None) -> None:
        if source is None:
            return
        close = getattr(source, "aclose", None)
        if callable(close):
            with suppress(Exception):
                result = close()
                if inspect.isawaitable(result):
                    await cast(Awaitable[object], result)
            return
        close = getattr(source, "close", None)
        if callable(close):
            with suppress(Exception):
                close()


__all__ = (
    "TokenCancelResult",
    "TokenChatBackend",
    "TokenChatExecutor",
    "TokenGenerateResult",
    "TokenGenerationMetadata",
    "TokenSource",
)
