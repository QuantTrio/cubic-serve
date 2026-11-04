"""Quantized checkpoint loading and explicitly selected native eager execution."""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch import Tensor

from cubic_serve.contracts.execution import ParallelTopology
from cubic_serve.executor.runner import configure_reference_math
from cubic_serve.models.builtin import (
    builtin_model_definition,
    builtin_text_config,
    parse_builtin_model_config_json,
)
from cubic_serve.models.parallel_weights import AxisSliceBinding, confined_checkpoint_path
from cubic_serve.models.quantized_checkpoint import (
    EncodedWeight,
    QuantizedCheckpoint,
    QuantizedMatrix,
    pin_host_tensor,
    use_pinned_host_storage,
)
from cubic_serve.models.qwen3_5.model import (
    Qwen3_5Config,
    Qwen3_5ModelOperators,
    Qwen3_5ReferenceModel,
)
from cubic_serve.models.qwen3_5_moe.config import Qwen3_5MoeConfig
from cubic_serve.models.qwen4_exp.model import Qwen4ExpConfig
from cubic_serve.runtime.token_chat import TokenGenerationMetadata
from cubic_serve.runtime.token_stream import TokenStream

_LOGGER = logging.getLogger(__name__)


class _PendingRankWeights(Mapping):
    """One private producer; on-demand binding overlaps subsequent reads.

    Pending values are construction operands, not admitted execution weights.
    Joining the producer proves the entire original checkpoint transaction and
    removes this helper before any model execution or graph capture.
    """

    def __init__(self, requests, device):
        from collections import deque
        from threading import Condition

        self.device = device
        self._condition = Condition()
        # Create the stream on the constructor thread, before concurrent work
        # can enter capture. The reader never allocates a stream during capture.
        if device.type == "cuda":
            from cubic_serve.ops.cuda.startup import startup_cuda_lock
            with startup_cuda_lock(device):
                self._stream = torch.cuda.Stream(device=device)
        else:
            self._stream = None
        self._remaining = {item[0].logical.name: item for item in requests}
        self._known = frozenset(self._remaining)
        # Encoded pageable operands can make progress during another startup
        # owner's CUDA capture. Actual constructor demands always take priority.
        ordered = sorted(requests, key=lambda item: bool(
            device.type == "cuda" and (item[0].encoding is None
                                      or item[0].logical.storage == "pinned_host")))
        self._ordered = deque(item[0].logical.name for item in ordered)
        self._demand = deque()
        self._values = {}
        self._events = {}
        self._error = None
        self._cancelled = False
        self._done = False
        self.finished_at = None
        self.wait_seconds = 0.

    def order(self):
        while self._remaining:
            with self._condition:
                if self._cancelled:
                    raise RuntimeError("private weight preparation cancelled")
                queue = self._demand if self._demand else self._ordered
                name = queue.popleft()
                item = self._remaining.pop(name, None)
            if item is not None:
                yield item

    def publish(self, name, value):
        event = None
        if isinstance(value, Tensor) and value.is_cuda:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
        with self._condition:
            if self._cancelled:
                raise RuntimeError("private weight preparation cancelled")
            self._values[name] = value
            if event is not None:
                self._events[name] = event
            self._condition.notify_all()

    def start(self, load):
        from contextlib import nullcontext
        from threading import Thread
        import time

        def run():
            try:
                context = (torch.cuda.device(self.device) if self.device.type == "cuda"
                           else nullcontext())
                with context:
                    stream = self._stream
                    stream_context = (torch.cuda.stream(stream) if stream is not None
                                      else nullcontext())
                    with stream_context:
                        load(self.order(), self.publish)
                    if stream is not None:
                        from cubic_serve.ops.cuda.startup import startup_cuda_lock
                        with startup_cuda_lock(self.device):
                            stream.synchronize()
            except BaseException as error:
                with self._condition:
                    self._error = error
            finally:
                with self._condition:
                    self.finished_at = time.perf_counter()
                    self._done = True
                    self._condition.notify_all()

        self._thread = Thread(target=run, name="checkpoint-reader")
        self._thread.start()

    def __getitem__(self, name):
        import time

        started = time.perf_counter()
        with self._condition:
            if name not in self._known:
                raise KeyError(name)
            if name in self._remaining:
                self._demand.append(name)
            self._condition.wait_for(lambda: name in self._values or self._done)
            self.wait_seconds += time.perf_counter() - started
            if self._error is not None:
                raise self._error
            value = self._values[name]
            event = self._events.pop(name, None)
        if event is not None:
            torch.cuda.current_stream(self.device).wait_event(event)
        return value

    def finish(self, *, cancel=False):
        with self._condition:
            self._cancelled = cancel
            self._condition.notify_all()
        self._thread.join()
        if self._error is not None and not cancel:
            raise self._error
        return self._values

    def __len__(self):
        return len(self._known)

    def __iter__(self):
        return iter(self.finish())


class QuantizedWeights:
    """Original host checkpoint payloads and independently owned device representations.

    Encoded weights are prepared at binding, never during a graph replay. Plain
    execution weights remain resident on the target device and follow its dtype.
    """

    def __init__(
        self,
        plan: QuantizedCheckpoint,
        *,
        device: torch.device | str,
        compute_dtype: torch.dtype,
        include_mtp: bool = True,
        owner_bindings: tuple[AxisSliceBinding, ...] | None = None,
        rank: int = 0,
        progress: Callable[[str, int, int], None] | None = None,
        _overlap_loading: bool = False,
    ) -> None:
        self.device = torch.device(device)
        # Each format retains its source owner/receipt alongside a registered device tensor.
        self.prepared_weights: dict[
            tuple[object, ...], tuple[object, Tensor | QuantizedMatrix | None]
        ] = {}
        self.materialized_weights: dict[
            tuple[object, ...], tuple[QuantizedMatrix, QuantizedMatrix]
        ] = {}
        self.rank = rank
        self.bindings = owner_bindings or ()
        self.inventory_digest = plan.inventory.inventory_digest
        self.semantic_plan_digest = plan.semantic_digest
        self.model_identity = plan.inventory.model_root.name
        self._weights: dict[str, Tensor | QuantizedMatrix]
        bindings = [
            w
            for w in plan.weights
            if w.logical.partition in ("language", "lm_head")
            or (include_mtp and w.logical.partition == "mtp")
        ]
        if owner_bindings is not None:
            by_name = {weight.logical.name: weight for weight in bindings}
            if len({b.tensor_name for b in owner_bindings}) != len(owner_bindings):
                raise ValueError("quantized owner bindings contain duplicate logical weights")
            for binding in owner_bindings:
                weight = by_name.get(binding.tensor_name)
                if weight is None or binding.global_shape != weight.logical.shape:
                    raise ValueError("quantized ownership differs from admitted logical weights")
                if not 0 <= binding.shard_axis < len(binding.global_shape):
                    raise ValueError("quantized ownership axis is out of bounds")
                shape = list(binding.global_shape)
                shape[binding.shard_axis] = sum(span.size for span in binding.spans)
                if tuple(shape) != binding.local_shape:
                    raise ValueError("quantized local shape differs from owner spans")
            requests = tuple((by_name[b.tensor_name], b.shard_axis, b.spans)
                             for b in owner_bindings)
            options = dict(device=self.device, compute_dtype=compute_dtype,
                           packed_device="cpu",
                           progress=(lambda done, total: progress("base", done, total))
                           if progress is not None else None)
            if _overlap_loading:
                pending = _PendingRankWeights(requests, self.device)
                self._weights = pending
                self.tensor_count = len(requests)
                pending.start(lambda order, publish: plan.materialize_many(
                    requests, **options, _request_order=order, _pending_weight=publish))
                return
            self._weights = plan.materialize_many(requests, **options)
            self.tensor_count = len(self._weights)
            return
        by_shard: dict[str, list[tuple[EncodedWeight, str, str]]] = defaultdict(list)
        payloads: dict[str, dict[str, Tensor]] = {w.logical.name: {} for w in bindings}
        for weight in bindings:
            for suffix, name, shard in weight.payload:
                by_shard[shard].append((weight, suffix, name))
        for shard, members in sorted(by_shard.items()):
            path = confined_checkpoint_path(plan.inventory.model_root, shard)
            plan.validate_shard(shard)
            before = path.stat()
            with plan._open_reader(path) as reader:
                for weight, suffix, name in members:
                    if weight.stacked_source is None:
                        tensor = reader.get_tensor(name)
                    else:
                        from cubic_serve.models.parallel_weights import AxisSpan
                        from cubic_serve.models.quantized_checkpoint import (
                            stacked_payload_selection,
                        )

                        tensor = reader.get_slice(name)[
                            stacked_payload_selection(
                                weight, suffix, 0, AxisSpan(0, weight.logical.shape[0])
                            )
                        ].squeeze(0)
                    dtype = (
                        compute_dtype
                        if weight.encoding is None and weight.logical.dtype == "BF16"
                        else tensor.dtype
                    )
                    host = weight.logical.storage == "pinned_host" and use_pinned_host_storage(
                        self.device
                    )
                    payloads[weight.logical.name][suffix] = (
                        pin_host_tensor(tensor.to(dtype=dtype).contiguous()) if host else
                        tensor.to("cpu" if weight.encoding is not None else self.device,
                                  dtype=dtype, copy=weight.encoding is not None)
                    )
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise ValueError("checkpoint shard changed while loading quantized payloads")
            _LOGGER.info("Loaded packed checkpoint shard %s", shard)
        self._weights = {
            w.logical.name: payloads[w.logical.name]["weight"]
            if w.encoding is None
            else QuantizedMatrix(
                payloads[w.logical.name],
                w.encoding,
                (w.logical.shape[0], w.logical.shape[1]),
                w.logical.output_partitions,
            )
            for w in bindings
        }
        self.tensor_count = len(self._weights)

    def __getitem__(self, name: str) -> Tensor | QuantizedMatrix:
        return self._weights[name]

    def finish_loading(self, *, cancel: bool = False) -> None:
        if isinstance(self._weights, _PendingRankWeights):
            pending = self._weights
            values = pending.finish(cancel=cancel)
            self.loading_finished_at = pending.finished_at
            self.loading_wait_seconds = pending.wait_seconds
            self._weights = values

    @property
    def resident_bytes(self) -> int:
        storages = {
            (t.device, t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
            for _, t in self.named_tensors()
            if t.is_cuda
        }
        return sum(storages.values())

    @property
    def storage_bytes(self) -> dict[str, int]:
        """Startup residency of actual backing stores, without reading tensor values.

        Views and cached aliases count once. Pinned host payloads remain host
        bytes even when CUDA kernels access them through mapped pointers.
        """
        totals = dict(device=0, pinned_host=0, pageable_host=0)
        seen = set()
        for _, tensor in self.named_tensors():
            storage = tensor.untyped_storage()
            identity = tensor.device, storage.data_ptr()
            if identity in seen:
                continue
            seen.add(identity)
            kind = ("device" if tensor.is_cuda else
                    "pinned_host" if tensor.is_pinned() else "pageable_host")
            totals[kind] += storage.nbytes()
        return totals

    def named_tensors(self) -> tuple[tuple[str, Tensor], ...]:
        result = []
        for name, weight in sorted(self._weights.items()):
            if isinstance(weight, Tensor):
                result.append((name, weight))
            else:
                result.extend(
                    (name.removesuffix(".weight") + "." + suffix, value)
                    for suffix, value in sorted(weight.payload.items())
                )
        for index, (_, value) in enumerate(self.prepared_weights.values()):
            if isinstance(value, Tensor):
                result.append((f"prepared.{index}", value))
            elif value is not None:
                result.extend(
                    (f"prepared.{index}.{name}", tensor)
                    for name, tensor in value.payload.items()
                )
        result.extend(
            (f"materialized.{index}.{name}", tensor)
            for index, (_, matrix) in enumerate(self.materialized_weights.values())
            for name, tensor in matrix.payload.items()
        )
        return tuple(result)


def load_quantized_rank_weights(
    config: Qwen3_5Config | Qwen3_5MoeConfig | Qwen4ExpConfig,
    plan: QuantizedCheckpoint,
    topology: ParallelTopology,
    rank: int,
    *,
    include_mtp: bool = False,
    owner_bindings: tuple[AxisSliceBinding, ...] | None = None,
    progress: Callable[[str, int, int], None] | None = None,
    _overlap_loading: bool = False,
) -> QuantizedWeights:
    """Consume one controller-admitted inventory and model-owned rank geometry.

    This is a loading boundary, not model graph/performance admission. Expert
    placement belongs to the MoE adapter; serialization stays format-owned.
    """
    placement = next((p for p in topology.placements if p.rank == rank), None)
    if placement is None:
        raise ValueError("quantized rank is outside the declared topology")
    if owner_bindings is not None:
        owners = owner_bindings
    else:
        owners = builtin_model_definition(config).owner_slices(
            config, topology, rank, include_mtp=include_mtp)
    text = builtin_text_config(config)
    compute = {"float16": torch.float16, "bfloat16": torch.bfloat16}[text.dtype]
    return QuantizedWeights(
        plan,
        device=f"cuda:{placement.device_index}",
        compute_dtype=compute,
        include_mtp=include_mtp,
        owner_bindings=owners,
        rank=rank,
        progress=progress,
        _overlap_loading=_overlap_loading,
    )


def load_quantized_model(
    model_root: str | Path,
    *,
    device: torch.device | str = "cuda:0",
    dtype: str = "bf16",
    recurrent_state_dtype: torch.dtype = torch.bfloat16,
    include_mtp: bool = True,
    operators: Qwen3_5ModelOperators,
) -> tuple[Qwen3_5ReferenceModel, QuantizedCheckpoint]:
    """Admit the complete checkpoint; execute its text/MTP graph in eager mode."""
    configure_reference_math()
    root = Path(model_root)
    config = parse_builtin_model_config_json(
        (root / "config.json").read_bytes(), compute_dtype=dtype
    )
    model_factory = builtin_model_definition(config).text_model
    if model_factory is None:
        raise ValueError("quantized text execution requires a declared native text constructor")
    from cubic_serve.models.builtin import admit_model_checkpoint

    plan = admit_model_checkpoint(root, model_config=config)
    if plan.declared_total_size != plan.inventory.total_tensor_bytes:
        _LOGGER.warning(
            "Checkpoint index size summary=%s; authoritative physical tensor bytes=%s",
            plan.declared_total_size,
            plan.inventory.total_tensor_bytes,
        )
    compute = {"float16": torch.float16, "bfloat16": torch.bfloat16}[
        builtin_text_config(config).dtype
    ]
    weights = QuantizedWeights(plan, device=device, compute_dtype=compute, include_mtp=include_mtp)
    model = model_factory(
        config=config, weights=weights, operators=operators,
        recurrent_state_dtype=recurrent_state_dtype,
    )
    return model, plan


class QuantizedEagerExecutor:
    """One native eager worker with request-local state and the shared token mailbox.

    This is a qualification/debugging executor, not a batching implementation.
    It makes no concurrency-throughput or CUDA Graph readiness claim.
    """

    def __init__(
        self,
        model: Qwen3_5ReferenceModel,
        *,
        model_root: str | Path,
        max_sequence_tokens: int | None = None,
        prefill_chunk_tokens: int = 16,
    ) -> None:
        from cubic_serve.generation.sampler import load_sampling_defaults
        from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators

        if not isinstance(model.default_operators, PackedLinearOperators):
            raise ValueError(
                "public quantized eager execution requires native packed linear operators"
            )
        if type(prefill_chunk_tokens) is not int or prefill_chunk_tokens <= 0:
            raise ValueError("prefill chunk size must be positive")
        if max_sequence_tokens is not None and (
            type(max_sequence_tokens) is not int or max_sequence_tokens <= 0
        ):
            raise ValueError("maximum sequence length must be positive")
        self._operator_config = model.default_operators.config
        self.model = model
        self.max_sequence_tokens = min(
            max_sequence_tokens or model.config.graph.text.max_position_embeddings,
            model.config.graph.text.max_position_embeddings,
        )
        self.prefill_chunk_tokens = prefill_chunk_tokens
        self._sampling_defaults = load_sampling_defaults(Path(model_root))
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="quantized-eager")
        self._cancelled: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self.ready = True

    def generate(
        self,
        input_ids: tuple[int, ...],
        max_tokens: int,
        metadata: TokenGenerationMetadata,
    ) -> TokenStream:
        if not self.ready or not input_ids or max_tokens <= 0:
            raise ValueError("quantized eager generation is not admitted")
        if len(input_ids) + max_tokens > self.max_sequence_tokens:
            raise ValueError("generation exceeds the declared context limit")
        if (
            metadata.prepared_input is not None
            and metadata.prepared_input.feature_bundle is not None
        ):
            raise ValueError("quantized eager multimodal execution is not qualified")
        cancelled = threading.Event()
        stream = TokenStream(on_close=lambda *_: cancelled.set())
        with self._lock:
            if not self.ready or metadata.request_id in self._cancelled:
                raise ValueError("closed executor or duplicate eager request identity")
            self._cancelled[metadata.request_id] = cancelled
            self._worker.submit(self._generate, input_ids, max_tokens, metadata, cancelled, stream)
        return stream

    def _generate(
        self,
        input_ids: tuple[int, ...],
        max_tokens: int,
        metadata: TokenGenerationMetadata,
        cancelled: threading.Event,
        stream: TokenStream,
    ) -> None:
        from cubic_serve.executor.runner import ReferenceExecutionContext
        from cubic_serve.generation.sampler import resolve_sampling
        from cubic_serve.generation.tp_sampling import candidate_probabilities
        from cubic_serve.ops.strategies import operator_strategy_scope

        try:
            if cancelled.is_set():
                return
            parameters = resolve_sampling(self._sampling_defaults, metadata.sampling)
            device = self.model.weights.device
            generator = torch.Generator(device=device).manual_seed(parameters.seed)
            cache = None
            with (
                torch.inference_mode(),
                ReferenceExecutionContext(),
                operator_strategy_scope(self._operator_config, device=str(device), model=""),
            ):
                for start in range(0, len(input_ids), self.prefill_chunk_tokens):
                    if cancelled.is_set():
                        return
                    ids = torch.tensor(
                        [input_ids[start : start + self.prefill_chunk_tokens]],
                        dtype=torch.int64,
                        device=device,
                    )
                    result = self.model.forward(ids, cache=cache)
                    cache = result.cache
                for step in range(max_tokens):
                    if cancelled.is_set():
                        return
                    logits = result.logits[0, -1].float()
                    if not bool(logits.isfinite().all()):
                        raise ValueError("quantized eager produced nonfinite logits")
                    if parameters.temperature == 0:
                        token = int(logits.argmax())
                    else:
                        order = torch.argsort(logits, descending=True, stable=True)[
                            : parameters.top_k
                        ]
                        probabilities = candidate_probabilities(
                            logits[order],
                            temperature=parameters.temperature,
                            top_k=parameters.top_k,
                            top_p=parameters.top_p,
                        )
                        token = int(order[torch.multinomial(probabilities, 1, generator=generator)])
                    stream.publish(token)
                    if step + 1 < max_tokens:
                        ids = torch.tensor([[token]], dtype=torch.int64, device=device)
                        result = self.model.forward(ids, cache=cache)
                        cache = result.cache
        except Exception as error:
            stream.fail(error)
        finally:
            stream.finish()
            with self._lock:
                self._cancelled.pop(metadata.request_id, None)

    def cancel(self, request_id: str) -> None:
        with self._lock:
            event = self._cancelled.get(request_id)
            if event is not None:
                event.set()

    def close(self) -> None:
        with self._lock:
            self.ready = False
            for event in self._cancelled.values():
                event.set()
        # Queued cancelled jobs run their finally block and close the mailbox.
        self._worker.shutdown(wait=True)
