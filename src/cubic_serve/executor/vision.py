"""Startup-bound visual graphs over the shared model math and format operators."""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import time
import multiprocessing
import pickle
import signal
import tempfile
import threading
import traceback
from contextlib import ExitStack, nullcontext, suppress
from datetime import timedelta
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch

from cubic_serve.executor.runner import configure_reference_math
from cubic_serve.inputs.multimodal import (
    EncoderAdmissionError, EncoderMemoryBudget, MultimodalInputPolicy,
    bounded_encoder_groups,
)
from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.models.qwen3_5.multimodal import (
    _bilinear_position_inputs,
    _position_ids,
    _validate_inputs,
    load_qwen_vision_reference,
)
from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators, PackedProjectionDispatch
from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope
from cubic_serve.runtime.cache import DeviceMemoryBudget

_LOGGER = logging.getLogger(__name__)


def _vision_weight_progress(device, completed, total):
    from cubic_serve.runtime.logging import format_weight_progress
    _LOGGER.info("weights/vision device=%s %s", device,
                 format_weight_progress(completed, total))


def load_qwen_vision_graph(model_root, *, device, operator_config=None, memory_bytes=3 << 29,
                           resources=None, device_memory_budget=None, checkpoint=None,
                           tp_size=1, tp_rank=0, tp_group=None, model_config=None):
    configure_reference_math()
    root = Path(model_root)
    native = PackedLinearOperators(operator_config or OperatorConfig())
    with torch.cuda.device(device):
        def reduce(value):
            torch.distributed.all_reduce(value, group=tp_group)
            return value
        subject = load_qwen_vision_reference(
            root, device=device, operators=native, checkpoint=checkpoint,
            packed_device="cpu",
            tp_size=tp_size, tp_rank=tp_rank, reduce=reduce if tp_size > 1 else None,
            progress=partial(_vision_weight_progress, device), model_config=model_config)
        pixel_limit = int(json.loads((root / "preprocessor_config.json").read_text())[
            "size"]["longest_edge"])
        return QwenVisionGraph(subject, native, memory_bytes=memory_bytes,
                               pixel_limit=pixel_limit, resources=resources,
                               device_memory_budget=device_memory_budget, tp_group=tp_group)


class _BoundRankSums:
    """Existing ordered collective ABI with lease-private stable operands."""

    def __init__(self, group, *, calibrate_transport=True):
        self.group = group
        self.calibrate_transport = calibrate_transport
        self.entries = {}

    def __call__(self, value):
        from cubic_serve.ops.cuda.collective import FixedRankBF16SumPrimitive
        key = tuple(value.shape), value.dtype
        if key not in self.entries:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("vision rank sum was not bound before capture")
            operand = torch.empty_like(value)
            self.entries[key] = operand, FixedRankBF16SumPrimitive(
                operand, self.group, calibrate_transport=self.calibrate_transport)
        operand, primitive = self.entries[key]
        operand.copy_(value)
        primitive.launch(operand)
        return operand


class _BoundOperators:
    """Reuse the existing sealed occurrence ABI; capture never binds a matrix."""

    def __init__(self, native):
        self.native = native
        self.linear_dispatch = PackedProjectionDispatch(native)
        self.embedding_dispatch = PackedProjectionDispatch(native, operation="embedding")
        self.entries = {}
        self.input_buffers = {}
        self.output_buffers = {}
        self.sealed = False

    def _call(self, inputs, weight, *, embedding=False, dtype=None):
        key = embedding, id(weight), tuple(inputs.shape)
        dispatch = self.embedding_dispatch if embedding else self.linear_dispatch
        if not self.sealed:
            shape = (*inputs.shape, weight.shape[1]) if embedding else (
                inputs.shape[0], weight.shape[0]
            )
            input_key = embedding, tuple(inputs.shape), inputs.dtype, inputs.device
            output_key = embedding, shape, dtype or inputs.dtype, inputs.device
            x = self.input_buffers.get(input_key)
            if x is None:
                x = torch.empty_like(inputs)
                self.input_buffers[input_key] = x
            y = self.output_buffers.get(output_key)
            if y is None:
                y = torch.empty(shape, device=inputs.device, dtype=dtype or inputs.dtype)
                self.output_buffers[output_key] = y
            dispatch.prepare(x, weight, y)
            self.entries[key] = x, y
            return (self.native.embedding(inputs, weight, dtype=dtype) if embedding
                    else self.native.linear(inputs, weight))
        x, y = self.entries[key]
        x.copy_(inputs)
        dispatch.launch(x, weight, y)
        return y

    def linear(self, inputs, weight):
        return self._call(inputs, weight)

    def embedding(self, ids, weight, *, dtype):
        return self._call(ids, weight, embedding=True, dtype=dtype)

    def seal(self):
        if any(not key[0] for key in self.entries):
            self.linear_dispatch.seal()
        if any(key[0] for key in self.entries):
            self.embedding_dispatch.seal()
        self.sealed = True


class _VisionBucket:
    def __init__(self, subject, native, capacity, stream, pool, tp_group=None):
        self.capacity = capacity
        self.subject = subject
        self.operators = _BoundOperators(native)
        v = subject.vision_config
        device = subject.weights.device
        width = v.input_channels * v.temporal_patch_size * v.patch_size**2
        self.pixels = torch.zeros((capacity, width), device=device, dtype=torch.float32)
        self.positions = torch.zeros((capacity, 2), device=device, dtype=torch.int64)
        self.indices = torch.zeros((4, capacity), device=device, dtype=torch.int64)
        self.fractions = torch.zeros((4, capacity), device=device, dtype=torch.float32)
        self.mask = torch.eye(capacity, device=device, dtype=torch.bool)
        stream.wait_stream(torch.cuda.current_stream(device))

        def forward():
            return subject.encode(
                self.pixels, self.positions, self.indices, self.fractions, None, self.mask
            )

        subject.operators = self.operators
        self.rank_sums = _BoundRankSums(tp_group) if subject.parallel.size > 1 else None
        if self.rank_sums is not None:
            subject.reduce = self.rank_sums
        with operator_strategy_scope(native.config, device="", model=""), torch.cuda.stream(stream):
            forward()
            self.operators.seal()
            expected = forward().clone()
            stream.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream, pool=pool):
                self.output = forward()
            self.graph.replay()
            stream.synchronize()
            if not torch.equal(self.output, expected):
                raise RuntimeError("visual graph capture changed its fixed-input arithmetic")

    def fill(self, pixels, grid):
        v = self.subject.vision_config
        logical = pixels.shape[0]
        if logical > self.capacity:
            raise ValueError("visual input exceeds its captured bucket")
        positions = _position_ids(grid, v.spatial_merge_size)
        indices, fractions = _bilinear_position_inputs(
            grid, num_grid_per_side=math.isqrt(v.position_count),
            spatial_merge_size=v.spatial_merge_size,
        )
        frames = torch.repeat_interleave(
            torch.arange(sum(int(t) for t, _, _ in grid.tolist())),
            torch.tensor([int(h*w) for t, h, w in grid.tolist() for _ in range(int(t))]),
        )
        mask = frames[:, None] == frames[None, :]
        self.pixels.zero_()
        self.pixels[:logical].copy_(pixels)
        self.positions.zero_()
        self.positions[:logical].copy_(positions)
        self.indices.zero_()
        self.indices[:, :logical].copy_(indices)
        self.fractions.zero_()
        self.fractions[:, :logical].copy_(fractions)
        self.mask.zero_()
        self.mask.diagonal().fill_(True)
        self.mask[:logical, :logical].copy_(mask)


class QwenVisionGraph:
    """One visual encoder lease, with sealed buckets and immutable result copies."""

    def __init__(self, subject, native, *, memory_bytes: int, pixel_limit: int,
                 resources=None, device_memory_budget: DeviceMemoryBudget | None = None,
                 tp_group=None):
        if subject.weights.device.type != "cuda" or type(memory_bytes) is not int or memory_bytes <= 0:
            raise ValueError("visual graphs require CUDA and a positive bounded memory share")
        self.config, self.weights = subject.config, subject.weights
        self.input_device = torch.device("cpu")
        self.vision_config = subject.vision_config
        self.subject, self.native = subject, native
        self.resources = resources or EncoderMemoryBudget(
            MultimodalInputPolicy(encoder_memory_bytes=memory_bytes))
        device = self.weights.device
        v = self.vision_config
        minimum = v.spatial_merge_size**2
        if minimum & (minimum - 1):
            raise ValueError("visual merge extent must admit power-of-two graph buckets")
        with torch.inference_mode(), torch.cuda.device(device):
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info(device)
            available = (device_memory_budget.remaining_bytes(free)
                         if device_memory_budget is not None else free)
            self.resources.initialize(available)
            budget = self.resources.total_bytes
            # Bound the whole bucket portfolio, including the sealed native
            # projection operands retained for every capacity. The geometric
            # sum of capacities is less than twice the largest capacity.
            # Packed bindings retain occurrence-local storage, while their
            # input/output buffers are pooled by shape. Plain torch linear
            # temporaries are reused by the graph pool across layers; price
            # eight BF16 buffer equivalents of the widest input and output.
            # The measured graph reservation below remains the admission gate.
            matrices = (
                weight for name, weight in self.weights.named_tensors()
                if len(weight.shape) == 2 and ".pos_embed." not in name
            )
            packed, plain = [], []
            for weight in matrices:
                (packed if isinstance(weight, QuantizedMatrix) else plain).append(weight.shape)
            linear_bytes = 3 * sum(sum(shape) for shape in packed)
            if plain:
                linear_bytes += 16 * (max(shape[0] for shape in plain)
                                      + max(shape[1] for shape in plain))
            linear_bytes += 2 * (4 * width_for(v) + 16 * v.hidden_size)
            # Query-row tiled attention bounds score/probability workspaces
            # linearly in input length; the frame-isolation mask is quadratic.
            linear_bytes += 12 * (v.head_count // subject.parallel.size) * min(
                512, max(minimum, pixel_limit // v.patch_size**2))
            quadratic_bytes = 2
            # Ordered rank sums retain their gather/operand workspaces per bucket.
            if subject.parallel.size > 1:
                linear_bytes += 4 * (subject.parallel.size + 1) * max(v.hidden_size, v.output_hidden_size)
            declared = max(minimum, pixel_limit // v.patch_size**2)
            # Input upload plus one device result copy. Request outputs and the
            # shared feature cache are host resident, including queued requests.
            self.staging_bytes_per_row = (
                width_for(v) * 4 + 4 * v.output_hidden_size // v.spatial_merge_size**2)
            maximum = self.resources.capture_capacity(
                linear_bytes=linear_bytes, quadratic_bytes=quadratic_bytes,
                staging_bytes_per_row=self.staging_bytes_per_row,
                minimum=minimum, declared=declared)
            if subject.parallel.size > 1:
                common = torch.tensor(maximum, dtype=torch.int64, device=device)
                torch.distributed.all_reduce(common, op=torch.distributed.ReduceOp.MIN,
                                             group=tp_group)
                maximum = int(common.item())
            self.stream = torch.cuda.Stream(device=device)
            pool = torch.cuda.graph_pool_handle()
            self.buckets = {}
            resident_before = torch.cuda.memory_allocated(device)
            _LOGGER.info("vision CUDA Graph preparation started device=%s max_patch_tokens=%d "
                         "compute_mode=%s", device, maximum, native.config.compute_mode)
            capacity = minimum
            while capacity <= maximum:
                started = time.perf_counter()
                self.buckets[capacity] = _VisionBucket(subject, native, capacity, self.stream, pool,
                                                       tp_group)
                measured_resident = torch.cuda.memory_allocated(device) - resident_before
                fits = self.resources.graph_capture_fits(
                    resident_bytes=measured_resident,
                    staging_bytes=capacity * self.staging_bytes_per_row)
                if subject.parallel.size > 1:
                    common.fill_(int(fits))
                    torch.distributed.all_reduce(common, op=torch.distributed.ReduceOp.MIN,
                                                 group=tp_group)
                    fits = bool(common.item())
                if not fits:
                    # Retire the last capture on every peer. Its native bindings
                    # are also retained by subject until explicitly unbound.
                    self.stream.synchronize()
                    subject.operators = native
                    if subject.parallel.size > 1:
                        subject.reduce = None
                    rejected = self.buckets.pop(capacity)
                    rejected.graph.reset()
                    del rejected
                    gc.collect()
                    torch.cuda.empty_cache()
                    if not self.buckets:
                        raise RuntimeError("encoder memory budget cannot admit its minimum graph")
                    released = self.resources.graph_capture_fits(
                        resident_bytes=torch.cuda.memory_allocated(device) - resident_before,
                        staging_bytes=max(self.buckets) * self.staging_bytes_per_row)
                    if subject.parallel.size > 1:
                        common.fill_(int(released))
                        torch.distributed.all_reduce(common, op=torch.distributed.ReduceOp.MIN,
                                                     group=tp_group)
                        released = bool(common.item())
                    if not released:
                        raise RuntimeError("encoder capture did not release its excess residency")
                    _LOGGER.info("vision graph portfolio ready device=%s max_patch_tokens=%d",
                                 device, max(self.buckets))
                    break
                maximum = self.resources.capture_capacity(
                    linear_bytes=linear_bytes, quadratic_bytes=quadratic_bytes,
                    staging_bytes_per_row=self.staging_bytes_per_row,
                    minimum=minimum, declared=declared,
                    measured_capacity=capacity, measured_resident_bytes=measured_resident)
                if subject.parallel.size > 1:
                    # All peers finish the same bucket before agreeing whether
                    # the next one fits; never diverge inside collective capture.
                    common.fill_(maximum)
                    torch.distributed.all_reduce(common, op=torch.distributed.ReduceOp.MIN,
                                                 group=tp_group)
                    maximum = int(common.item())
                _LOGGER.info("vision CUDA Graph captured device=%s patch_tokens=%d duration_s=%.3f "
                             "resident_bytes=%d planned_max_patch_tokens=%d",
                             device, capacity, time.perf_counter() - started,
                             measured_resident, maximum)
                capacity *= 2
            if not self.buckets:
                raise RuntimeError("visual graph budget cannot admit the minimum image")
            self.resident_graph_bytes = torch.cuda.memory_allocated(device) - resident_before
            maximum = max(self.buckets)
            torch.cuda.empty_cache()
            _LOGGER.info("vision graph memory device=%s resident_bytes=%d staging_bytes=%d "
                         "budget_bytes=%d max_patch_tokens=%d", device,
                         self.resident_graph_bytes, maximum * self.staging_bytes_per_row,
                         budget, maximum)
            self.resources.reserve(resident_bytes=self.resident_graph_bytes,
                                   staging_bytes=maximum * self.staging_bytes_per_row)
            if device_memory_budget is not None:
                free, _ = torch.cuda.mem_get_info(device)
                device_memory_budget.require_headroom(
                    free, additional_bytes=self.resources.staging_bytes)
        native.seal_dynamic_calibration()
        subject.operators = native
        if subject.parallel.size > 1:
            # Oversize independent frames use eager execution after readiness.
            # Their row counts are request-dependent, so a new TP shape must
            # never launch cold transport calibration in the shared GPU process.
            subject.reduce = _BoundRankSums(tp_group, calibrate_transport=False)
        self.max_patch_tokens = max(self.buckets)
        self.max_input_rows = self.max_patch_tokens
        declared_rows = pixel_limit // v.patch_size**2
        low, high = self.max_input_rows, max(self.max_input_rows, declared_rows)
        while low < high:
            middle = (low + high + 1) // 2
            if (self.resources.resident_bytes +
                    max(self.resources.staging_bytes,
                        self._eager_transient_bytes(middle)) <= budget):
                low = middle
            else:
                high = middle - 1
        self.max_dynamic_input_rows = low
        self.peak_input_bytes = (
            self._eager_transient_bytes(low) if low > self.max_input_rows
            else self.resources.staging_bytes)
        _LOGGER.info("visual graph ready device=%s patch_tokens=%d memory_budget_bytes=%d "
                     "resident_graph_bytes=%d max_dynamic_patch_tokens=%d "
                     "peak_input_bytes=%d "
                     "compute_mode=%s buckets=%s", device, self.max_patch_tokens, budget,
                     self.resident_graph_bytes, self.max_dynamic_input_rows,
                     self.peak_input_bytes,
                     native.config.compute_mode, tuple(self.buckets))

    def _eager_transient_bytes(self, rows: int) -> int:
        v = self.vision_config
        local_heads = v.head_count // self.subject.parallel.size
        # One independent frame: upload/result staging, frame mask, 512-row
        # attention score/probability work, and live layer activations.
        return (rows * self.staging_bytes_per_row + rows**2 +
                512 * rows * local_heads * 8 +
                rows * max(v.hidden_size, v.intermediate_size) * 16)

    @torch.inference_mode()
    def forward(self, pixel_values, grid_thw):
        """Encode independent temporal groups into request-owned BF16 host rows.

        Qwen visual positions and attention are independent per temporal group;
        temporal pairing, frame order and decoder timestamps stay model owned.
        Bounded grouping reuses only startup-captured graphs. One spatial group
        must fit; a long video does not require a whole-video square graph.
        """
        grid = grid_thw.detach().cpu()
        v = self.vision_config
        _validate_inputs(pixel_values, grid.to(pixel_values.device), v, pixel_values.device)
        frames = [(1, int(h), int(w)) for t, h, w in grid.tolist() for _ in range(int(t))]
        lengths = [h*w for _, h, w in frames]
        groups = bounded_encoder_groups(lengths, self.max_patch_tokens,
                                        allow_oversize=True)
        maximum = max(sum(lengths[start:stop]) for start, stop in groups)
        if maximum > self.max_patch_tokens:
            if maximum > self.max_dynamic_input_rows:
                raise _encoder_capacity_error(
                    "image or video frame", maximum, self.max_dynamic_input_rows)
            v = self.vision_config
            lease = self.resources.lease_dynamic(self._eager_transient_bytes(maximum))
        else:
            lease = self.resources.lease(maximum * self.staging_bytes_per_row)
        with lease, \
                torch.cuda.device(self.weights.device):
            self.stream.wait_stream(torch.cuda.current_stream(self.weights.device))
            output = torch.empty((sum(lengths) // v.spatial_merge_size**2,
                                  v.output_hidden_size), dtype=torch.bfloat16)
            patch_offset = feature_offset = 0
            with torch.cuda.stream(self.stream):
                for start, stop in groups:
                    logical = sum(lengths[start:stop])
                    pixels = pixel_values[patch_offset:patch_offset+logical].to(self.weights.device)
                    if logical > self.max_patch_tokens:
                        if stop != start + 1:
                            raise RuntimeError("oversize encoder item must own its segment")
                        frame_grid = torch.tensor(frames[start:stop], dtype=torch.int64,
                                                  device=self.weights.device)
                        mask = torch.ones((logical, logical), dtype=torch.bool,
                                          device=self.weights.device)
                        positions = _position_ids(frame_grid, v.spatial_merge_size)
                        indices, fractions = _bilinear_position_inputs(
                            frame_grid, num_grid_per_side=math.isqrt(v.position_count),
                            spatial_merge_size=v.spatial_merge_size,
                        )
                        _LOGGER.info("vision eager capacity miss device=%s patch_tokens=%d "
                                     "max_graph_patch_tokens=%d", self.weights.device,
                                     logical, self.max_patch_tokens)
                        with operator_strategy_scope(self.native.config, device="", model=""):
                            features = self.subject.encode(
                                pixels, positions, indices, fractions, None, mask)
                    else:
                        capacity = max(1 << (logical - 1).bit_length(), min(self.buckets))
                        bucket = self.buckets[capacity]
                        bucket.fill(pixels, torch.tensor(frames[start:stop], dtype=torch.int64))
                        bucket.graph.replay()
                        features = bucket.output
                    feature_rows = logical // v.spatial_merge_size**2
                    output[feature_offset:feature_offset+feature_rows].copy_(
                        features[:feature_rows], non_blocking=False)
                    # Drop the upload before allocating the next segment's input.
                    del pixels, features
                    patch_offset += logical
                    feature_offset += feature_rows
                self.stream.synchronize()
                if maximum > self.max_patch_tokens and isinstance(self.subject.reduce, _BoundRankSums):
                    # Eager shapes come from client media geometry.  Their TP
                    # workspaces are safe to release after stream completion;
                    # retaining one binding per shape would grow without bound.
                    self.subject.reduce.entries.clear()
                return output


def width_for(vision):
    return vision.input_channels * vision.temporal_patch_size * vision.patch_size**2


def _encoder_capacity_error(item_label: str, required_rows: int,
                            admitted_rows: int) -> EncoderAdmissionError:
    return EncoderAdmissionError(
        f"{item_label} is too large for the configured multimodal GPU memory budget: "
        f"required_input_rows={required_rows} admitted_input_rows={admitted_rows}; "
        "increase --multimodal-memory-gib",
        public_message=f"{item_label} is too large",
    )


def _pack_encoder_metadata(value):
    """Transport model metadata without changing its shape, dtype or values."""
    if value is None:
        return None
    host = value.detach().to("cpu").contiguous()
    return tuple(host.shape), host.dtype, host.reshape(-1).view(torch.uint8).numpy().tobytes()


def _unpack_encoder_metadata(value):
    if value is None:
        return None
    shape, dtype, payload = value
    return (torch.frombuffer(bytearray(payload), dtype=dtype).reshape(shape) if payload
            else torch.empty(shape, dtype=dtype, device="cpu"))


def serve_input_encoder(encoder, connection, rank):
    """Use the encoder's private stream, storage, communicator and host lease."""
    torch.cuda.set_device(encoder.weights.device)
    binding = connection.recv()
    if binding is None:
        return
    pixels, output = binding
    while True:
        command = connection.recv()
        if command is None:
            return
        if len(command) == 2:
            logical, metadata = command
            item_pixels, item_output = pixels, output
        else:
            logical, metadata, item_pixels, item_output = command
        with torch.cuda.stream(encoder.stream):
            features = encoder.forward(item_pixels[:logical], _unpack_encoder_metadata(metadata))
        if rank == 0:
            item_output[:features.shape[0]].copy_(features)
        connection.send(("complete", features.shape[0]))


def prepare_owned_input_encoder(admission, *, topology, rank, device):
    """Bind the input factory to first-stage workers without changing LM groups."""
    factory = getattr(admission, "input_encoder_factory", None)
    if factory is None:
        return None
    recipe = getattr(admission, "input_encoder_program_recipe", None)
    if recipe is not None:
        source = getattr(admission, "model_definition_source", None)
        if (source is None
                or source.execution_recipe(admission.input_encoder_model_config) != recipe):
            raise ValueError("input encoder recipe differs from controller admission")
    ranks = [p.rank for p in topology.placements if p.pipeline_stage == 0]
    group = torch.distributed.new_group(ranks)
    if rank not in ranks:
        return None
    local_rank = torch.distributed.get_group_rank(group, rank)
    encoder = factory(device=device, tp_rank=local_rank, tp_group=group)
    admission.input_encoder_reserves[device.index] = encoder.peak_input_bytes
    admission.input_encoder_connection.send((
        "ready", encoder.max_input_rows, encoder.resources.staging_bytes,
        encoder.resident_graph_bytes, encoder.max_dynamic_input_rows,
        encoder.peak_input_bytes))
    return encoder, local_rank


def start_owned_input_encoder(owned, connection, report_failure):
    if owned is None:
        return None
    encoder, rank = owned

    def encode():
        try:
            serve_input_encoder(encoder, connection, rank)
        except BaseException:
            failure = traceback.format_exc()
            with suppress(OSError, EOFError):
                connection.send(("failure", failure))
                report_failure(failure)
            # Stop the device owner before another graph or communicator can
            # consume resources left poisoned by the input encoder.
            os._exit(1)

    thread = threading.Thread(target=encode, name="input-encoder", daemon=True)
    thread.start()
    return thread


def finish_owned_input_encoder(thread):
    if thread is not None:
        thread.join(timeout=5)
        if thread.is_alive():
            # Teardown must never destroy communicators under a live encoder.
            os._exit(1)


def _encoder_rank(metadata_path, rank, device_index, connection, init_method, world_size):
    """Persistent encoder ranks; shared host buffers carry input/result once."""
    from cubic_serve.runtime.logging import configure_logging
    configure_logging()
    started = time.perf_counter()
    _LOGGER.info("vision worker bootstrap entered rank=%d", rank)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if not any(name in os.environ for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")):
        torch.set_num_threads(4)
    configure_reference_math()
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    definition_source = None
    encoder = None
    try:
        torch.distributed.init_process_group(
            "nccl", init_method=init_method, rank=rank, world_size=world_size,
            device_id=device, timeout=timedelta(seconds=120))
        with open(metadata_path, "rb") as file:
            metadata = pickle.load(file)
        definition_source = getattr(metadata, "model_definition_source", None)
        if definition_source is None:
            factory = metadata
        else:
            if (definition_source.execution_recipe(metadata.config)
                    != metadata.model_program_recipe):
                raise ValueError("input encoder recipe differs from controller admission")
            factory = metadata.factory
        _LOGGER.info("vision worker initialized rank=%d metadata_bytes=%d duration_s=%.3f",
                     rank, Path(metadata_path).stat().st_size, time.perf_counter() - started)
        encoder = factory(device=device, tp_rank=rank, tp_group=torch.distributed.group.WORLD)
        connection.send(("ready", encoder.max_input_rows, encoder.resources.staging_bytes,
                         encoder.resident_graph_bytes, encoder.max_dynamic_input_rows,
                         encoder.peak_input_bytes))
        serve_input_encoder(encoder, connection, rank)
    except BaseException:
        with suppress(OSError, EOFError):
            connection.send(("failure", traceback.format_exc()))
        raise
    finally:
        with ExitStack() as cleanup:
            if definition_source is not None:
                cleanup.callback(definition_source.close)
            cleanup.callback(connection.close)
            if torch.distributed.is_initialized():
                cleanup.callback(torch.distributed.destroy_process_group)
            if definition_source is not None and encoder is not None:
                close = getattr(encoder, "close", None)
                if close is not None:
                    cleanup.callback(close)


class InputEncoderGroup:
    """A fixed encoder TP topology with one exclusive, bounded input lease.

    The factory binds model math and native format operators; the segmenter
    declares independent inputs. Device ownership, shared transport, completion,
    failure poisoning and shutdown are common to input encoders.
    """

    def __init__(self, factory, *, devices, config, semantic_digest, segmenter, module,
                 input_width, output_width, merge_extent, timeout=1200, colocated=False,
                 admission_item_label="input media item", model_definition_source=None):
        if model_definition_source is not None:
            model_definition_source.verify_source()
            self.model_program_recipe = model_definition_source.execution_recipe(config)
            self.model_definition_namespace = model_definition_source.module_name
            semantic_digest = model_definition_source.program_identity(
                semantic_digest, self.model_program_recipe,
            )
        self.config = config
        self.input_device = torch.device("cpu")
        self.weights = SimpleNamespace(device=self.input_device,
                                       semantic_plan_digest=semantic_digest)
        self.devices = tuple(devices)
        if not self.devices or len(set(self.devices)) != len(self.devices):
            raise ValueError("encoder TP requires distinct physical devices")
        self._segmenter = segmenter
        self._timeout = timeout
        self._lock = threading.Lock()
        self.execution_lease = None
        self._closed = False
        self._processes = []
        self._connections = []
        self.colocated = colocated
        self.factory = factory
        self._input_width, self._output_width = input_width, output_width
        self._merge_extent, self._module = merge_extent, module
        self._admission_item_label = admission_item_label
        self._directory = tempfile.TemporaryDirectory(prefix="cubic-input-encoder-")
        if colocated:
            self.staging_by_device = {}
            self.input_reserves_by_device = {}
            return
        metadata = Path(self._directory.name) / "encoder.pkl"
        with metadata.open("wb") as file:
            value = (factory if model_definition_source is None else
                     model_definition_source.bind_metadata(SimpleNamespace(
                         factory=factory, config=config,
                         model_program_recipe=self.model_program_recipe,
                     )))
            pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
        context = multiprocessing.get_context("spawn")
        try:
            for rank, index in enumerate(self.devices):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_encoder_rank, args=(str(metadata), rank, index, child,
                    (Path(self._directory.name) / "nccl-store").as_uri(), len(self.devices)),
                    name=f"input-encoder-tp-{rank}")
                self._connections.append(parent)
                self._processes.append(process)
                try:
                    process.start()
                finally:
                    child.close()
            self._bind()
        except BaseException:
            self._close_locked()
            raise

    def attach_main_workers(self, processes, connections):
        if not self.colocated or self._connections or self._closed:
            raise RuntimeError("encoder cannot attach to these main workers")
        self._processes = list(processes)
        self._connections = list(connections)
        self._bind()

    def _bind(self):
        ready = [self._receive(rank, "ready") for rank in range(len(self.devices))]
        capacities = {item[1] for item in ready}
        if len(capacities) != 1:
            raise RuntimeError("encoder TP ranks admitted different graph portfolios")
        self.max_input_rows = capacities.pop()
        self.max_patch_tokens = self.max_input_rows
        self.max_dynamic_input_rows = min(
            item[4] if len(item) > 4 else item[1] for item in ready)
        self.staging_by_device = {d: r[2] for d, r in zip(self.devices, ready, strict=True)}
        self.input_reserves_by_device = {
            d: (r[5] if len(r) > 5 else r[2])
            for d, r in zip(self.devices, ready, strict=True)
        }
        self._pixels = torch.empty(self.max_input_rows, self._input_width, device="cpu").share_memory_()
        self._output = torch.empty(self.max_input_rows // self._merge_extent, self._output_width,
                                   dtype=torch.bfloat16, device="cpu").share_memory_()
        for connection in self._connections:
            connection.send((self._pixels, self._output))
        _LOGGER.info("input encoder ready module=%s parallelism=tp tp_size=%d "
                     "devices=%s input_rows=%d colocated=%s", self._module, len(self.devices),
                     self.devices, self.max_input_rows, self.colocated)

    def _receive(self, rank, expected):
        deadline = time.monotonic() + self._timeout
        connection = self._connections[rank]
        while not connection.poll(0.1):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"input encoder rank {rank} did not finish {expected}")
            if any(not process.is_alive() for process in self._processes):
                raise RuntimeError("an input encoder rank exited before topology completion")
        message = connection.recv()
        if message[0] == "failure":
            raise RuntimeError(f"input encoder rank {rank} failed:\n{message[1]}")
        if message[0] != expected:
            raise RuntimeError("input encoder returned an invalid transaction result")
        return message

    @torch.inference_mode()
    def forward(self, pixels, grid):
        with self._lock:
            if self._closed:
                raise RuntimeError("input encoder topology is closed or poisoned")
            segments = tuple(self._segmenter(pixels, grid, self.max_input_rows))
            for chunk, _ in segments:
                if chunk.shape[0] > self.max_dynamic_input_rows:
                    raise _encoder_capacity_error(
                        self._admission_item_label, chunk.shape[0],
                        self.max_dynamic_input_rows)
            eager_rows = max((chunk.shape[0] for chunk, _ in segments), default=0)
            exclusive = self.colocated and eager_rows > self.max_input_rows
            if exclusive and self.execution_lease is None:
                raise RuntimeError("colocated dynamic encoder has no topology execution lease")
            try:
                execution_lease = self.execution_lease if exclusive else nullcontext()
                with execution_lease:
                    outputs = []
                    for chunk, chunk_grid in segments:
                        logical = chunk.shape[0]
                        if logical > self.max_input_rows:
                            item_pixels = torch.empty_like(chunk).share_memory_()
                            item_output = torch.empty(
                                (logical // self._merge_extent, self._output_width),
                                dtype=torch.bfloat16, device="cpu").share_memory_()
                            item_pixels.copy_(chunk)
                            command = (logical, _pack_encoder_metadata(chunk_grid),
                                       item_pixels, item_output)
                        else:
                            item_pixels, item_output = self._pixels, self._output
                            item_pixels[:logical].copy_(chunk)
                            command = logical, _pack_encoder_metadata(chunk_grid)
                        for connection in self._connections:
                            connection.send(command)
                        completed = [self._receive(rank, "complete")
                                     for rank in range(len(self.devices))]
                        counts = {item[1] for item in completed}
                        if len(counts) != 1:
                            raise RuntimeError("input encoder ranks disagree on feature ownership")
                        outputs.append(item_output[:counts.pop()].clone())
                    return torch.cat(outputs)
            except BaseException:
                self._close_locked()
                raise

    def _close_locked(self):
        self._closed = True
        for connection in self._connections:
            with suppress(OSError, EOFError):
                connection.send(None)
        deadline = time.monotonic() + 5
        for process in (() if self.colocated else self._processes):
            if process.pid is not None:
                process.join(max(0, deadline - time.monotonic()))
        for process in (() if self.colocated else self._processes):
            if process.is_alive():
                process.terminate()
                process.join(5)
        for connection in self._connections:
            connection.close()
        self._directory.cleanup()

    def close(self):
        with self._lock:
            if not self._closed:
                self._close_locked()


def _qwen_segments(pixels, grid, capacity, *, vision):
    _validate_inputs(pixels, grid, vision, torch.device("cpu"))
    frames = [(1, int(h), int(w)) for t, h, w in grid.tolist() for _ in range(int(t))]
    lengths = [h*w for _, h, w in frames]
    offset = 0
    for start, stop in bounded_encoder_groups(lengths, capacity, allow_oversize=True):
        count = sum(lengths[start:stop])
        yield pixels[offset:offset+count], torch.tensor(frames[start:stop], dtype=torch.int64)
        offset += count


def _load_qwen_encoder_rank(*, device, tp_rank, tp_group, model_root, checkpoint,
                            operator_config, memory_bytes, tp_size, device_memory_budgets,
                            model_config=None):
    budget = device_memory_budgets.get(device.index)
    if budget is not None:
        budget.apply_allocator_limit()
    return load_qwen_vision_graph(
        model_root, device=device, checkpoint=checkpoint, operator_config=operator_config,
        memory_bytes=memory_bytes, device_memory_budget=budget,
        tp_size=tp_size, tp_rank=tp_rank, tp_group=tp_group, model_config=model_config)


def load_qwen_vision_tp(model_root, *, devices, operator_config=None, memory_bytes=3 << 29,
                        device_memory_budgets=None, checkpoint=None, colocated=False,
                        model_config=None, model_definition_source=None):
    """The model's common visual graph follows the serving tensor group."""
    from cubic_serve.models.builtin import parse_model_config_json, resolve_model_definition
    from cubic_serve.models.quantized_checkpoint import CheckpointWeights
    from cubic_serve.models.qwen3_5.multimodal import vision_weight_requests
    root = Path(model_root)
    definition = (None if model_definition_source is None
                  else model_definition_source.definition)
    config = (parse_model_config_json((root / "config.json").read_bytes(), definition=definition)
              if model_config is None else model_config)
    started = time.perf_counter()
    _LOGGER.info("vision checkpoint metadata started model=%s tp_size=%d", root, len(devices))
    logical = resolve_model_definition(config, definition=definition).logical_weights(config)
    if checkpoint is None:
        checkpoint = CheckpointWeights.admit(root, logical, verify_payloads=False,
                                             require_quantization=False)
    else:
        checkpoint.validate_for(root, logical)
    _LOGGER.info("vision checkpoint metadata complete duration_s=%.3f tp_size=%d",
                 time.perf_counter() - started, len(devices))
    vision = getattr(config, "graph", config).vision
    # Reject unsupported geometry before spawning ranks or allocating weights.
    for rank in range(len(devices)):
        vision_weight_requests(checkpoint, vision, size=len(devices), rank=rank)
    factory = partial(_load_qwen_encoder_rank, model_root=root,
                      checkpoint=checkpoint.payload_view(partitions=("vision",)),
                      operator_config=operator_config, memory_bytes=memory_bytes,
                      tp_size=len(devices), device_memory_budgets=device_memory_budgets or {},
                      model_config=config)
    _LOGGER.info("vision workers initialization started tp_size=%d devices=%s",
                 len(devices), tuple(devices))
    group = InputEncoderGroup(
        factory, devices=devices, config=config, semantic_digest=checkpoint.semantic_digest,
        module="vision_tower",
        segmenter=partial(_qwen_segments, vision=vision), input_width=width_for(vision),
        output_width=vision.output_hidden_size, merge_extent=vision.spatial_merge_size**2,
        colocated=colocated, admission_item_label="image or video frame",
        model_definition_source=model_definition_source)
    group.vision_config = vision
    return group
