"""Common chat backends over reference and qualified token executors."""

from __future__ import annotations

from typing_extensions import TypeAliasType

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass, field, replace

from cubic_serve.api.server import (
    BackendErrorKind,
    ChatBackendError,
    ChatExecution,
)
from cubic_serve.contracts.request import (
    CanonicalChatRequest,
    StructuredOutputKind,
    TokenEvent,
    TokenEventKind,
)
from cubic_serve.generation import ChatOutputCodec, GenerationOutputPipeline
from cubic_serve.generation.constraints import (
    ConstraintCompilationError,
    ConstraintFactory,
    ConstraintReadinessError,
    constraint_spec_from_meta,
    create_logit_constraint,
)
from cubic_serve.inputs.chat import (
    ChatInputProcessor,
    ChatProcessorError,
    resolve_chat_template_kwargs,
)
from cubic_serve.models import PreparedInput
from cubic_serve.runtime.engine import ReferenceEngine

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _StreamEnd:
    pass


@dataclass(frozen=True, slots=True)
class _StreamFailure:
    error: ChatBackendError


_StreamItem = TypeAliasType("_StreamItem", TokenEvent | _StreamEnd | _StreamFailure)
_END = _StreamEnd()


@dataclass(slots=True)
class _RequestChannel:
    pipeline: GenerationOutputPipeline
    queue: asyncio.Queue[_StreamItem]
    cancelled: bool = False
    publication_stopped: asyncio.Event = field(default_factory=asyncio.Event)

    def stop_publication(self) -> None:
        """Make cancellation visible before waiting for engine-side quiescence."""

        self.cancelled = True
        self.publication_stopped.set()
        while not self.queue.empty():
            self.queue.get_nowait()
        self.queue.put_nowait(_END)


class ReferenceChatBackend:
    """One online driver for all text requests admitted to a reference engine.

    This is intentionally a composition, not another scheduler.  The engine's
    continuous scheduler owns batching; this driver advances one ``EngineStep``
    at a time, demultiplexes raw events by request id, and runs the sole public
    output pipeline.  Cancellation closes publication first and then marks the
    scheduler request cancelled.  It does not claim to reclaim model state.
    """

    def __init__(
        self,
        engine: ReferenceEngine,
        *,
        processor: ChatInputProcessor,
        output_codec: ChatOutputCodec,
        enable_thinking: bool = True,
        reasoning_parser: str | bool = False,
        tool_call_parser: str | bool = False,
        use_prefix_cache: bool = True,
        max_sequence_tokens: int | None = None,
        max_buffered_events: int = 64,
        constraint_factory: ConstraintFactory | None = None,
        structured_output_in_reasoning: bool = False,
        structured_output_disable_whitespace: bool = False,
    ) -> None:
        if not isinstance(engine, ReferenceEngine):
            raise TypeError("reference chat backend requires a ReferenceEngine")
        boolean_settings = (enable_thinking, use_prefix_cache, structured_output_in_reasoning,
                            structured_output_disable_whitespace)
        if any(type(setting) is not bool for setting in boolean_settings):
            raise TypeError("reference chat boolean settings must be bool values")
        if any(type(value) not in (str, bool) for value in (reasoning_parser, tool_call_parser)):
            raise TypeError("output parser selections require names or booleans")
        if type(max_buffered_events) is not int or max_buffered_events <= 0:
            raise ValueError("max_buffered_events must be a positive integer")
        if max_sequence_tokens is not None and (
            type(max_sequence_tokens) is not int or max_sequence_tokens <= 0
        ):
            raise ValueError("max_sequence_tokens must be a positive integer or None")
        if engine.eos_token_ids != getattr(
            output_codec, "eos_token_ids", (output_codec.eos_token_id,)
        ):
            raise ValueError("reference engine and output codec EOS identities differ")
        if reasoning_parser or tool_call_parser:
            output_codec.admit_parser_recipe(
                reasoning_parser=reasoning_parser,
                tool_call_parser=tool_call_parser,
            )

        self._engine = engine
        self._processor = processor
        self._output_codec = output_codec
        self._enable_thinking = enable_thinking
        self._reasoning_parser = reasoning_parser
        self._tool_call_parser = tool_call_parser
        self._use_prefix_cache = use_prefix_cache
        self._max_sequence_tokens = max_sequence_tokens
        self._max_buffered_events = max_buffered_events
        self._constraint_factory = constraint_factory
        self._structured_output_in_reasoning = structured_output_in_reasoning
        self._structured_output_disable_whitespace = structured_output_disable_whitespace
        self._model_id = engine.loaded_adapter.descriptor.model_spec.model_id
        self._weight_variant = engine.loaded_adapter.descriptor.weight_variant
        self._engine_calls = asyncio.Lock()
        self._channels: dict[str, _RequestChannel] = {}
        self._known_request_ids: set[str] = set()
        self._driver: asyncio.Task[None] | None = None
        self._fatal = False

    async def start(self, request: CanonicalChatRequest) -> ChatExecution:
        if not isinstance(request, CanonicalChatRequest):
            raise TypeError("reference chat backend requires a CanonicalChatRequest")
        if self._structured_output_disable_whitespace:
            from cubic_serve.generation.structured_sampling import compact_json_constraint

            request = replace(
                request, structured_output=compact_json_constraint(request.structured_output)
            )
        if request.thinking_token_budget is not None and not self._reasoning_parser:
            raise ChatBackendError(
                BackendErrorKind.INVALID_REQUEST, "thinking budget requires a reasoning parser",
                public_message="thinking_token_budget is unavailable for this model",
            )
        if self._fatal:
            raise ChatBackendError(
                BackendErrorKind.UNAVAILABLE,
                "reference chat driver is unavailable",
            )
        if request.model_id != self._model_id:
            raise ChatBackendError(
                BackendErrorKind.MODEL_NOT_FOUND,
                f"model {request.model_id!r} is not served by this engine",
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

        self._known_request_ids.add(request.request_id)
        try:
            builder = getattr(self._processor, "build_chat_prepared_input", None)
            if builder is None:
                builder = self._processor.build_chat_input_plan
            realized = await asyncio.to_thread(
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
                "chat input processing failed",
            ) from error

        max_new_tokens = request.max_new_tokens
        if max_new_tokens is None and self._max_sequence_tokens is None:
            self._known_request_ids.discard(request.request_id)
            raise ChatBackendError(
                BackendErrorKind.UNAVAILABLE,
                "reference backend has no declared context limit",
            )
        if max_new_tokens is None:
            max_new_tokens = self._max_sequence_tokens - len(input_plan.token_ids)
        if max_new_tokens <= 0 or (
            self._max_sequence_tokens is not None
            and len(input_plan.token_ids) + max_new_tokens - 1 > self._max_sequence_tokens
        ):
            self._known_request_ids.discard(request.request_id)
            raise ChatBackendError(
                BackendErrorKind.INVALID_REQUEST,
                "prompt plus completion exceeds reference context capacity",
                public_message="request exceeds available context length",
            )

        from cubic_serve.generation.structured_sampling import reasoning_control_from_prompt

        try:
            control = reasoning_control_from_prompt(
                request, self._output_codec, input_plan.token_ids
            )
        except ValueError as error:
            self._known_request_ids.discard(request.request_id)
            raise ChatBackendError(
                BackendErrorKind.INVALID_REQUEST, str(error),
                public_message="thinking_token_budget is unavailable for this model",
            ) from error
        from cubic_serve.generation.tool_output import tool_sampling_constraint

        sampling_constraint = tool_sampling_constraint(request)
        constraint = None
        if (sampling_constraint.kind is not StructuredOutputKind.IDENTITY
                or control is not None):
            try:
                factory = self._constraint_factory
                end = tuple(getattr(self._output_codec, "reasoning_end_token_ids", ()))
                if (factory is not None and end and enable_thinking and self._reasoning_parser
                        and not self._structured_output_in_reasoning):
                    from cubic_serve.generation.structured_sampling import (
                        DelayedConstraintFactory,
                        implicit_reasoning_ends,
                    )

                    factory = DelayedConstraintFactory(factory, end, self._engine.eos_token_ids,
                        implicit_reasoning_ends(sampling_constraint, self._output_codec))
                constraint = await asyncio.to_thread(
                    create_logit_constraint,
                    spec=constraint_spec_from_meta(sampling_constraint),
                    request_id=request.request_id,
                    factory=factory,
                    reasoning_control=control,
                )
            except asyncio.CancelledError:
                self._known_request_ids.discard(request.request_id)
                raise
            except ConstraintCompilationError as error:
                self._known_request_ids.discard(request.request_id)
                raise ChatBackendError(BackendErrorKind.INVALID_REQUEST, str(error)) from error
            except ConstraintReadinessError as error:
                self._known_request_ids.discard(request.request_id)
                raise ChatBackendError(BackendErrorKind.UNAVAILABLE, str(error)) from error

        from cubic_serve.generation.output import request_output_parser

        parser = request_output_parser(
            self._output_codec,
            request,
            enable_thinking=enable_thinking,
            reasoning_parser=self._reasoning_parser,
            tool_call_parser=self._tool_call_parser,
            reasoning_control=control,
        )

        pipeline = GenerationOutputPipeline(
            request_id=request.request_id,
            input_token_count=len(input_plan.token_ids),
            max_new_tokens=max_new_tokens,
            eos_token_ids=(
                ()
                if request.ignore_eos
                and sampling_constraint.kind is StructuredOutputKind.IDENTITY
                else self._engine.eos_token_ids
            ),
            decoder=self._output_codec.new_decoder(),
            parser=parser,
            continuous_usage=request.stream_continuous_usage_stats,
            report_reasoning_tokens=bool(self._reasoning_parser),
            include_reasoning=request.include_reasoning,
        )
        channel = _RequestChannel(
            pipeline=pipeline,
            queue=asyncio.Queue(maxsize=self._max_buffered_events),
        )

        async with self._engine_calls:
            if self._fatal:
                self._known_request_ids.discard(request.request_id)
                raise ChatBackendError(
                    BackendErrorKind.UNAVAILABLE,
                    "reference chat driver is unavailable",
                )
            admission = asyncio.create_task(
                asyncio.to_thread(
                    self._engine.admit,
                    prepared_input,
                    max_new_tokens=max_new_tokens,
                    use_prefix_cache=self._use_prefix_cache,
                    constraint=constraint,
                    ignore_eos=request.ignore_eos,
                )
            )
            try:
                await asyncio.shield(admission)
            except asyncio.CancelledError:
                admitted = await self._settle_admission(admission)
                if admitted:
                    self._engine.cancel(request.request_id)
                else:
                    self._known_request_ids.discard(request.request_id)
                raise
            except Exception as error:
                self._known_request_ids.discard(request.request_id)
                raise ChatBackendError(
                    BackendErrorKind.UNAVAILABLE,
                    "reference engine rejected request admission",
                ) from error
            self._channels[request.request_id] = channel

        if self._driver is None or self._driver.done():
            self._driver = asyncio.create_task(self._drive(), name="cubic-reference-chat-driver")
        return ChatExecution(
            input_tokens=len(input_plan.token_ids),
            events=self._events(channel),
        )

    async def cancel(self, request_id: str) -> None:
        channel = self._channels.pop(request_id, None)
        if channel is None:
            return
        channel.stop_publication()
        async with self._engine_calls:
            await asyncio.to_thread(self._engine.cancel, request_id)

    async def _drive(self) -> None:
        current = asyncio.current_task()
        try:
            while self._channels:
                async with self._engine_calls:
                    if not self._channels:
                        return
                    step = await asyncio.to_thread(self._engine.step)
                if step is None:
                    raise RuntimeError("reference engine stalled with admitted chat requests")

                for raw_event in step.events:
                    channel = self._channels.get(raw_event.request_id)
                    if channel is None or channel.cancelled:
                        continue
                    public_events = channel.pipeline.push(raw_event)
                    for public_event in public_events:
                        if not await self._put(channel, public_event):
                            break
                    if raw_event.kind is TokenEventKind.FINISH and not channel.cancelled:
                        if not channel.pipeline.finished:
                            raise RuntimeError("terminal raw event did not finish output pipeline")
                        if not await self._put(channel, _END):
                            continue
                        self._channels.pop(raw_event.request_id, None)
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception:
            _LOGGER.exception("reference chat driver failed requests=%s", tuple(self._channels))
            self._fatal = True
            active = tuple(self._channels.items())
            self._channels.clear()
            for _, channel in active:
                while not channel.queue.empty():
                    channel.queue.get_nowait()
                channel.queue.put_nowait(
                    _StreamFailure(
                        ChatBackendError(
                            BackendErrorKind.UNAVAILABLE,
                            "reference chat driver failed",
                        )
                    )
                )
            async with self._engine_calls:
                for request_id, _ in active:
                    with suppress(Exception):
                        await asyncio.to_thread(self._engine.cancel, request_id)
        finally:
            if self._driver is current:
                self._driver = None

    @staticmethod
    async def _settle_admission(admission: asyncio.Task[object]) -> bool:
        """Wait for an uncancellable worker to finish before releasing its gate."""

        while not admission.done():
            try:
                await asyncio.shield(admission)
            except asyncio.CancelledError:
                continue
            except Exception:
                return False
        try:
            admission.result()
        except Exception:
            return False
        return True

    @staticmethod
    async def _put(channel: _RequestChannel, item: _StreamItem) -> bool:
        """Bound buffering without letting a cancelled consumer block the driver."""

        if channel.cancelled:
            return False
        put = asyncio.create_task(channel.queue.put(item))
        stopped = asyncio.create_task(channel.publication_stopped.wait())
        done, _ = await asyncio.wait((put, stopped), return_when=asyncio.FIRST_COMPLETED)
        if stopped in done:
            put.cancel()
            await asyncio.gather(put, return_exceptions=True)
            return False
        stopped.cancel()
        await asyncio.gather(stopped, return_exceptions=True)
        return not channel.cancelled

    @staticmethod
    async def _events(channel: _RequestChannel) -> AsyncIterator[TokenEvent]:
        while True:
            item = await channel.queue.get()
            if channel.cancelled or isinstance(item, _StreamEnd):
                return
            if isinstance(item, _StreamFailure):
                raise item.error
            yield item


__all__ = ("ReferenceChatBackend",)
